"""Follow-up inference checks F1-F3 (PROTOCOL.md in this folder). Inference only.

Runs on the cluster from the screen bundle (copy to job/followup.py there):
  gates -> F3 reconstruction (B0, A, D on extra_monitor) -> F1 self-conditioning audit
  (D runs, then B0 runs if uploaded) -> F2 fitted prior on D -> mGPT scoring.
Every output is skipped when it already exists, so a resubmission resumes. Exit 75 = clean stop.
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BUNDLE_ROOT', HERE.parent))
for p in (ROOT, ROOT / 'vendor', HERE):
    sys.path.insert(0, str(p))
from diffusion_stage import log, read_json, write_json, digest, scorer, score_file  # noqa: E402

D_AE_SHA = '14d9cae39517747074f8369e21596b87c00a3ed3c40b1fd085999d071b68e9f2'
B0_AE_SHA = '123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'
RUNS = [f'{m}-seed{s}' for m in ('small', 'published') for s in (42, 43, 44)]
AUDIT_SEED, AUDIT_SAMPLES = 70000, 100
PRIOR_SEEDS, PRIOR_SAMPLES = (40000, 41000, 42000), 1000
MONITOR_ROWS = 681
NORM = 8.0
BEAM = dict(min_length=5, num_beams=4, do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.2)


class Stop(Exception):
    pass


def check(deadline):
    from tools import longrun_training as engine
    if engine.STOP or time.monotonic() > deadline:
        raise Stop


def gates(a):
    """Hash every fixed input; stop at once if a pinned hash differs."""
    screen = ROOT / 'outputs/screen'
    d_ae, a_ae = screen / 'arms/D/best', screen / 'arms/A/best'
    assert digest(d_ae / 'model.pt') == D_AE_SHA, 'Autoencoder D differs from the screen winner'
    assert digest(a.old_ae / 'model.pt') == B0_AE_SHA, 'Autoencoder B0 differs from job 56531'
    selection = read_json(screen / 'diffusion/selection.json')['checkpoints']
    new = {}
    for run in RUNS:
        h = digest(screen / f'diffusion/runs/{run}/best.pt')
        assert h == selection[run], f'{run} best.pt differs from the sealed selection'
        new[run] = h
    old = {run: digest(a.b0_runs / run / 'best.pt') for run in RUNS if (a.b0_runs / run / 'best.pt').exists()}
    monitor = ROOT / 'data/extra_afrikaans/processed/extra_monitor.jsonl'
    rows = [json.loads(s) for s in monitor.read_text(encoding='utf8').splitlines() if s.strip()]
    assert len(rows) == MONITOR_ROWS and all({'id', 'document_id', 'text'} <= set(r) for r in rows)
    code = {p: digest(ROOT / p) for p in ('tools/longrun_training.py', 'diffusion/text_denoising_diffusion.py',
                                         'job/recon_eval.py', 'job/diffusion_stage.py')}
    contract = dict(script_sha256=digest(__file__), protocol_sha256=digest(a.protocol) if a.protocol.exists() else None,
                    autoencoders=dict(D=D_AE_SHA, B0=B0_AE_SHA, A=digest(a_ae / 'model.pt')),
                    new_denoisers=new, b0_denoisers=old, b0_missing=[r for r in RUNS if r not in old],
                    extra_monitor_sha256=digest(monitor), code=code,
                    seeds=dict(audit=AUDIT_SEED, prior=PRIOR_SEEDS, recon_bootstrap=20261107, audit_bootstrap=20261106))
    write_json(a.output / 'contract.json', contract)
    log('gates_passed', b0_runs=len(old))
    return contract


def f3_reconstruction(a, deadline):
    from recon_eval import evaluate
    screen = ROOT / 'outputs/screen'
    panel = ROOT / 'data/extra_afrikaans/processed/extra_monitor.jsonl'
    for name, ae_dir in (('B0', a.old_ae), ('A', screen / 'arms/A/best'), ('D', screen / 'arms/D/best')):
        dest = a.output / 'recon681' / f'{name}.jsonl'
        if dest.with_suffix('.summary.json').exists():
            continue
        check(deadline)
        summary = evaluate(ae_dir, a.model_cache, panel, dest)
        log('recon_done', ae=name, exact=summary['exact'], chrf=summary['chrf'], seconds=summary['seconds'])


def load_system(path, ae_dir, a, cache):
    """EMA weights and the configuration sealed inside the checkpoint; the AE is cached per family."""
    from tools import longrun_training as engine
    saved = engine.load_checkpoint(path, 'cuda')
    config, spec = saved['contract']['config'], saved['contract']['spec']
    engine.configure_torch(config)
    model = engine.make_model(config, spec['model'])
    model.load_state_dict(saved['ema'], strict=True)
    model.eval().requires_grad_(False)
    del saved
    if str(ae_dir) not in cache:
        cache.clear()
        cache[str(ae_dir)] = engine.load_ae(ae_dir, a.model_cache, config)
    ae, tok = cache[str(ae_dir)]
    return model, ae, tok, config


def f1_audit(a, deadline):
    import torch
    from tools import longrun_training as engine
    out = a.output / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    repro_path = out / 'reproduction.json'
    repro = read_json(repro_path) if repro_path.exists() else {}
    screen = ROOT / 'outputs/screen/diffusion/runs'
    systems = [(f'D-{r}', screen / r, ROOT / 'outputs/screen/arms/D/best') for r in RUNS]
    systems += [(f'B0-{r}', a.b0_runs / r, a.old_ae) for r in RUNS if (a.b0_runs / r / 'best.pt').exists()]
    cache = {}
    for name, folder, ae_dir in systems:
        files = {mode: out / f'{name}-{mode}.json' for mode in ('off', 'on')}
        if all(f.exists() for f in files.values()) and name in repro:
            continue
        check(deadline)
        model, ae, tok, config = load_system(folder / 'best.pt', ae_dir, a, cache)
        if name not in repro:
            stored = read_json(folder / 'validation-selected-samples.json')[:2]
            model.normalize_sampling_self_condition = False
            again = engine.generate(model, ae, tok, dict(config, normalize_sampling_self_condition=False), 2, stored[0]['seed'])
            repro[name] = [dict(seed=s['seed'], match=s['text'] == g['text']) for s, g in zip(stored, again)]
            write_json(repro_path, repro)
        for mode, path in files.items():
            if path.exists():
                continue
            check(deadline)
            flag = mode == 'on'
            model.normalize_sampling_self_condition = flag
            began = time.monotonic()
            rows = engine.generate(model, ae, tok, dict(config, normalize_sampling_self_condition=flag), AUDIT_SAMPLES, AUDIT_SEED)
            write_json(path, dict(system=name, normalize_sampling_self_condition=flag, seed_start=AUDIT_SEED, rows=rows))
            log('audit_done', system=name, mode=mode, seconds=round(time.monotonic() - began))
        del model
        torch.cuda.empty_cache()
    cache.clear()
    torch.cuda.empty_cache()


def ledoit_wolf(x, chunk=4096):
    """Ledoit-Wolf (2004) shrinkage towards (trace(S)/p)I, closed-form intensity, float64.
    Same code as latent_controls_2026-10-30/run_controls.py."""
    import numpy as np
    n, p = x.shape
    mu = x.mean(0)
    s = np.zeros((p, p))
    beta_ = 0.
    for i in range(0, n, chunk):
        c = x[i:i + chunk] - mu
        s += c.T @ c
        c2 = c * c
        beta_ += float((c2.T @ c2).sum())
    s /= n
    m = np.trace(s) / p
    delta_ = float((s * s).sum())
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * m * np.trace(s) + p * m * m) / p
    beta = min(beta, delta)
    shrinkage = 0. if beta == 0 else beta / delta
    cov = (1 - shrinkage) * s
    cov[np.diag_indices(p)] += shrinkage * m
    return mu, cov, shrinkage, m


def f2_prior(a, deadline):
    import numpy as np
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from tools import longrun_training as engine
    out = a.output / 'prior'
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in PRIOR_SEEDS if not (out / f'C2-D-{s}.json').exists()]
    if not todo:
        return
    check(deadline)
    cache = ROOT / 'outputs/screen/diffusion/cache'
    manifest = read_json(cache / 'manifest.json')
    parts = []
    for shard in manifest['splits']['train']['shards']:
        assert digest(cache / shard['file']) == shard['sha256'], shard['file']
        parts.append(engine.load_checkpoint(cache / shard['file'])['latents'].float())
    z = torch.cat(parts)
    slots, dim = z.shape[1], z.shape[2]
    assert list(z.shape) == [47767, 64, 64] and torch.isfinite(z).all()
    x = z.reshape(len(z), slots * dim).double().numpy()
    del z, parts
    mu, cov, shrinkage, target = ledoit_wolf(x)
    chol = np.linalg.cholesky(cov)

    def norms(v):
        n = np.linalg.norm(v, axis=-1)
        return dict(mean=float(n.mean()), sd=float(n.std(ddof=1)), min=float(n.min()), max=float(n.max()))
    record_path = out / 'c2-record.json'
    record = read_json(record_path) if record_path.exists() else dict(
        estimator='Ledoit-Wolf 2004, target (trace(S)/p)I, closed-form intensity, float64, NumPy',
        shrinkage_intensity=shrinkage, target_scale=target, n_training=len(x), dimensions=slots * dim,
        training_slot_norms=norms(x.reshape(-1, slots, dim)), banks={})
    del x
    ae, tok = engine.load_ae(ROOT / 'outputs/screen/arms/D/best', a.model_cache, dict(latent_dim=64, latent_length=64, max_length=128))
    for bank in todo:
        pre = np.stack([mu + chol @ np.random.Generator(np.random.PCG64(bank + i)).standard_normal(slots * dim) for i in range(PRIOR_SAMPLES)])
        post = pre.reshape(PRIOR_SAMPLES, slots, dim)
        post = post / np.linalg.norm(post, axis=-1, keepdims=True) * NORM
        record['banks'][str(bank)] = dict(slot_norms_before=norms(pre.reshape(-1, slots, dim)), slot_norms_after=norms(post),
                                          mean_distance_after=float(np.linalg.norm(post.reshape(PRIOR_SAMPLES, -1).mean(0) - mu)))
        latents = torch.from_numpy(post).float()
        torch.save(latents, out / f'C2-D-{bank}-latents.pt')
        rows = []
        with torch.inference_mode():
            for i in range(PRIOR_SAMPLES):
                check(deadline)
                lat = latents[i:i + 1].cuda()
                ids = ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(lat)), max_length=128, **BEAM)[0]
                eos = (ids[1:] == tok.eos_token_id).nonzero(as_tuple=True)[0]
                n = int(eos[0]) + 2 if len(eos) else len(ids)
                raw = tok.decode(ids[:n], skip_special_tokens=False)
                rows.append(dict(seed=bank + i, text=tok.decode(ids[:n], skip_special_tokens=True).strip(), decoder_tokens=n,
                                 eos=bool(len(eos)), length_cap=n >= 128, sentinel='<extra_id_' in raw))
        write_json(out / f'C2-D-{bank}.json', rows)
        write_json(record_path, record)
        log('prior_done', bank=bank)
    del ae
    torch.cuda.empty_cache()


def score_all(a, deadline):
    files = sorted((a.output / 'audit').glob('*-o[nf]*.json')) + sorted((a.output / 'prior').glob('C2-D-?????.json'))
    files = [f for f in files if not f.name.endswith('-mgpt.json')]
    todo = [f for f in files if not f.with_name(f.stem + '-mgpt.json').exists()]
    if not todo:
        return
    model, tok, score_text = scorer(a)
    for f in todo:
        if not score_file(f, f.with_name(f.stem + '-mgpt.json'), model, tok, score_text, deadline):
            raise Stop
        log('scored', file=f.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--old-ae', type=Path, required=True)
    p.add_argument('--mgpt', type=Path, required=True)
    p.add_argument('--b0-runs', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    deadline = time.monotonic() + a.seconds - 600
    from tools import longrun_training as engine
    engine.install_signals()
    a.output.mkdir(parents=True, exist_ok=True)
    gates(a)
    try:
        f3_reconstruction(a, deadline)
        f1_audit(a, deadline)
        f2_prior(a, deadline)
        score_all(a, deadline)
    except (Stop, InterruptedError):
        log('stopped_cleanly')
        return 75
    write_json(a.output / 'followup_complete.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
