"""Evaluate sealed cluster checkpoints locally; never modify the cluster bundle.

Run as: python tools/local_longrun_evaluation.py ACTION --bundle DIR --output DIR --ae DIR.
The bundle is the archived cluster workspace together with the source, config, NCHLT data
and models/mt5-base it was run with; its code is imported from the bundle, not from this
checkout. Actions: download-scorer (fetch the pinned mGPT revision; needs network), smoke
(validation-only compatibility and timing check, writes local-smoke.json), final-test
(encode the test split and evaluate each selected checkpoint, writing
runs/<name>/final-test.json) and score (mGPT perplexity and review sheets once every test
evaluation exists). All outputs go to --output, which must lie outside the bundle.
Interrupted runs exit with code 75; rerun the same command to resume.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace


def main():
    """Verify the bundle against its sealed selection, then run one evaluation action."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['smoke', 'final-test', 'score', 'download-scorer'])
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ae', type=Path, required=True)
    a = p.parse_args()
    a.bundle = a.bundle.resolve()
    a.output = a.output.resolve()
    if a.output == a.bundle or a.bundle in a.output.parents:
        p.error('Output must be separate from the immutable bundle')
    # Import the archived source so the evaluated code is the code that trained the models.
    sys.path.insert(0, str(a.bundle))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tools import afrikaans_longrun as protocol
    from tools import longrun_training as engine
    from tools.longrun_common import read_json, write_json, digest, identity, read_rows, normalize
    if Path(protocol.__file__).resolve().parent.parent != a.bundle:
        raise RuntimeError('Evaluation must import the archived cluster source')
    if a.action == 'download-scorer':
        from huggingface_hub import snapshot_download
        snapshot_download('ai-forever/mGPT', revision=protocol.SCORER_REVISION)
        return
    import torch
    # Hugging Face loads are offline from here on; only download-scorer uses the hub.
    os.environ['HF_HUB_OFFLINE'] = '1'
    engine.install_signals()
    a.output.mkdir(parents=True, exist_ok=True)
    config = read_json(a.bundle / 'configs/afrikaans_longrun.json')
    plan = read_json(a.bundle / 'plan.json')
    selection = read_json(a.bundle / 'selection.json')
    suite = read_json(a.bundle / 'suite-status.json')
    specs = suite['completed']
    if suite['status'] != 'completed' or identity(plan) != selection['plan_sha256']:
        raise ValueError('Suite incomplete or sealed plan changed')
    if identity(config) != plan['config_sha256']:
        raise ValueError('Configuration changed')
    args = SimpleNamespace(workspace=a.output, config=a.bundle / 'configs/afrikaans_longrun.json',
                           data=a.bundle / 'data/nchlt_afrikaans/processed/stage_5_7',
                           ae=a.ae.resolve(), model_cache=a.bundle / 'models/mt5-base')
    original = read_json(a.bundle / 'inputs.json')
    local = protocol.input_contract(args, config)
    # Only software versions/hardware may differ. Weights, code, data and settings must match.
    for key in original:
        if key != 'packages' and local[key] != original[key]:
            raise ValueError(f'Cluster input mismatch: {key}')
    for spec in specs:
        name = spec['name']
        if digest(a.bundle / 'runs' / name / 'best.pt') != selection['checkpoints'][name]:
            raise ValueError(f'Selected checkpoint changed: {name}')
    engine.configure_torch(config)
    hw = engine.hardware()
    contract = dict(selection=selection, inputs=local, hardware=hw, driver_sha256=digest(Path(__file__)))
    contract_path = a.output / 'local-contract.json'
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError('Local environment changed; use a fresh output folder')
    write_json(contract_path, contract)
    write_json(a.output / 'invocations' / f'{time.time_ns()}.json', protocol.provenance(args, hw))
    # Copy the validation-stage files so summarize() can build full tables in the output
    # folder; existing copies are left alone and the bundle is only read.
    for spec in specs:
        dest = a.output / 'runs' / spec['name']
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('status.json', 'config.json', 'metrics.jsonl', 'selected-validation.json',
                     'validation-selected-samples.json'):
            source = a.bundle / 'runs' / spec['name'] / name
            if source.exists() and not (dest / name).exists():
                shutil.copyfile(source, dest / name)
    if a.action == 'score':
        if any(not (a.output / 'runs' / s['name'] / 'final-test.json').exists() for s in specs):
            raise ValueError('Finish all test evaluations before scoring')
        code = protocol.score_generations(args, config, specs, engine, lambda: float('inf'))
        protocol.summarize(a.output)
        raise SystemExit(code)
    cache = engine.load_cache(a.bundle / 'cache', original)
    ae, tok = engine.load_ae(args.ae, args.model_cache, config)
    if a.action == 'smoke':
        # Validation only: do not touch the test set during compatibility/timing checks.
        reports = []
        for model_name in config['models']:
            spec = next(s for s in specs if s['model'] == model_name)
            saved = engine.load_checkpoint(a.bundle / 'runs' / spec['name'] / 'best.pt')
            model = engine.make_model(config, model_name)
            model.load_state_dict(saved['ema'], strict=True)
            model.eval().requires_grad_(False)
            del saved
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            smoke_config = dict(config, validation_examples=32, train_probe_examples=32)
            metrics = engine.validation(model, cache, smoke_config)
            sample_start = time.monotonic()
            # Same seed offset as the cluster's selected-validation samples.
            rows = engine.generate(model, ae, tok, config, 3, config['sample_seed'] + 10000)
            seconds_per_sample = (time.monotonic() - sample_start) / 3
            reports.append(dict(model=model_name, seconds=time.monotonic()-start,
                                seconds_per_sample=seconds_per_sample,
                                estimated_test_generation_hours=seconds_per_sample*config['final_samples']*
                                sum(s['model'] == model_name for s in specs)/3600,
                                peak_bytes=torch.cuda.max_memory_allocated(), metrics=metrics, samples=rows))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(a.output / 'local-smoke.json', dict(passed=True, contract=identity(contract), reports=reports))
        print(json.dumps(reports, indent=2))
        return
    smoke = read_json(a.output / 'local-smoke.json')
    if not smoke['passed'] or smoke['contract'] != identity(contract):
        raise ValueError('Pass local smoke first')
    # Test latents are encoded once and saved, so a resumed job reuses them.
    test_path = a.output / 'test-latents.pt'
    if not test_path.exists():
        latents = []
        rows = read_rows(args.data / 'test.jsonl')
        with torch.no_grad():
            for i in range(0, len(rows), config['cache_batch']):
                if engine.STOP:
                    raise InterruptedError()
                batch = tok([r['text'] for r in rows[i:i+config['cache_batch']]], return_tensors='pt',
                            padding='max_length', truncation=True, max_length=config['max_length']).to('cuda')
                latents.append(ae.get_diffusion_latent(ae.get_encoder()(**batch), batch.attention_mask).float().cpu())
        engine.atomic_save(test_path, torch.cat(latents))
    # validation() reads cache['valid'], so the test latents replace it.
    cache['valid'] = engine.load_checkpoint(test_path)
    train_texts = {normalize(r['text']) for r in read_rows(args.data / 'train.jsonl')}
    for spec in specs:
        folder = a.output / 'runs' / spec['name']
        if (folder / 'final-test.json').exists():
            continue
        if engine.STOP:
            raise InterruptedError()
        saved = engine.load_checkpoint(a.bundle / 'runs' / spec['name'] / 'best.pt')
        if saved['contract'] != dict(inputs=original, config=config, spec=spec):
            raise ValueError('Checkpoint contract mismatch')
        step = saved['step']
        model = engine.make_model(config, spec['model'])
        model.load_state_dict(saved['ema'], strict=True)
        model.eval().requires_grad_(False)
        del saved
        metric_path = folder / 'test-denoising.json'
        if not metric_path.exists():
            metrics = engine.validation(model, cache, config, full=True)
            write_json(metric_path, {k.replace('valid', 'test'): v for k, v in metrics.items()})
        sample_path = folder / 'test-selected-samples.json'
        rows = read_json(sample_path) if sample_path.exists() else []
        # Same test seed offset as afrikaans_longrun.py's final-test action. Samples are made in
        # chunks of 10 and saved after each, so an interruption loses at most one chunk; the
        # seed check guards that a resumed list continues the same sequence.
        start_seed = config['sample_seed'] + 100000
        if [r['seed'] for r in rows] != list(range(start_seed, start_seed + len(rows))):
            raise ValueError('Saved sample seeds changed')
        while len(rows) < config['final_samples']:
            rows.extend(engine.generate(model, ae, tok, config, min(10, config['final_samples']-len(rows)), start_seed+len(rows)))
            write_json(sample_path, rows)
            print(f"{spec['name']}: {len(rows)}/{config['final_samples']} samples", flush=True)
        write_json(folder / 'final-test.json', dict(run=spec['name'], model=spec['model'],
            training_seed=spec['seed'], selected_step=step, split='test', denoising=read_json(metric_path),
            generation=protocol.generation_metrics(rows, train_texts), sample_file=sample_path.name,
            selection='sealed cluster minimum validation EMA velocity MSE',
            environment='local; see local-contract.json; not bitwise equivalent to cluster',
            test_reuse='Held out from diffusion tuning; historical AE test evaluation exists'))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    protocol.summarize(a.output)


if __name__ == '__main__':
    try:
        main()
    except InterruptedError:
        print('Interrupted safely; rerun the same command to resume.', flush=True)
        raise SystemExit(75)
