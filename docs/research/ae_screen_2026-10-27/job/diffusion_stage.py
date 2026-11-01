"""Jobs 2 and 3: diffusion on the winning autoencoder, then evaluation and scoring (PROTOCOL.md section 7).

Training reuses tools/afrikaans_longrun.py unchanged, with the benchmark's config except the
autoencoder hash and latent length. Phases:
  train  : prepare -> smoke -> benchmark -> plan -> run          (job 2)
  finish : run (resume if needed) -> evaluate -> final-test -> mGPT scoring -> random-latent control (job 3)
Every action is skipped when its output exists, so a resubmission resumes. Exit 75 = clean stop.
"""
import argparse, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BUNDLE_ROOT', HERE.parent))
sys.path.insert(0, str(ROOT))
MGPT_SHA = 'c45c522411e64dbed6da1ed0ce0839bf03931d3a17b80182d06d4dab8d8b2fbb'
CONTROL_SAMPLES, CONTROL_SEED = 1000, 30000


def log(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Longrun:
    def __init__(self, a, winner, deadline):
        self.a, self.winner, self.deadline = a, winner, deadline
        self.workspace = a.output / 'diffusion'

    def remaining(self):
        return self.deadline - time.monotonic()

    def __call__(self, action):
        seconds = self.remaining() - 600  # leave ten minutes for export
        if seconds <= 60:
            return 75
        log('longrun', action=action, seconds=int(seconds))
        cmd = [sys.executable, '-u', str(ROOT / 'tools/afrikaans_longrun.py'), action, '--config', str(self.a.output / 'winner/longrun_config.json'),
               '--workspace', str(self.workspace), '--data', str(ROOT / 'data/nchlt_afrikaans/processed/stage_5_7'),
               '--ae', self.winner['ae_dir'], '--model-cache', str(self.a.model_cache), '--seconds', str(int(seconds))]
        return subprocess.run(cmd, cwd=ROOT).returncode


def train_phase(run):
    w = run.workspace
    steps = [('prepare', lambda: (w / 'cache/manifest.json').exists() and read_json(w / 'cache/manifest.json').get('complete')),
             ('smoke', lambda: (w / 'smoke.json').exists()),
             ('benchmark', lambda: (w / 'benchmark.json').exists()),
             ('plan', lambda: (w / 'plan.json').exists()),
             ('run', lambda: (w / 'suite-status.json').exists() and read_json(w / 'suite-status.json')['status'] in ('completed', 'under_budget'))]
    for action, done in steps:
        if done():
            continue
        code = run(action)
        if code == 75:
            return 75
        if code not in (0, 2):  # run returns 2 for under_budget, which is a valid end state
            raise RuntimeError(f'{action} failed with exit {code}')
    return 0


def scorer(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert digest(a.mgpt / 'pytorch_model.bin') == MGPT_SHA, 'mGPT weights differ from the pinned revision'
    tok = AutoTokenizer.from_pretrained(a.mgpt, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.mgpt, local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    from tools.compare_afrikaans_generators import score_text
    # Anchors: the first three human validation references scored for the existing benchmark.
    for text, nll, tokens in json.loads((ROOT / 'panels/mgpt_anchors.json').read_text(encoding='utf8')):
        value = score_text(model, tok, text, 'cuda')
        assert value['tokens'] == tokens and abs(value['nll'] - nll) / max(1, abs(nll)) < 1e-4, 'mGPT anchor mismatch'
    return model, tok, score_text


def score_file(path, destination, model, tok, score_text, deadline):
    if destination.exists():
        return True
    rows = read_json(path)
    rows = rows['rows'] if isinstance(rows, dict) else rows
    scored, nll, tokens = [], 0.0, 0
    for row in rows:
        if time.monotonic() > deadline:
            return False
        value = score_text(model, tok, row['text'], 'cuda')
        scored.append(dict(row, mgpt=value))
        if value:
            nll += value['nll']
            tokens += value['tokens']
    write_json(destination, dict(scorer='ai-forever/mGPT', revision='40897bd7c8b47a76802c411108ca6220438b8b40', count=len(rows),
                                 empty=sum(r['mgpt'] is None for r in scored),
                                 token_weighted_perplexity=math.exp(nll / tokens) if tokens else None, rows=scored))
    return True


def random_latent_control(a, ae_dir, latents, destination, deadline):
    """E1: decode norm-8 Gaussian latents with the autoencoder's decoder and the benchmark beam settings."""
    if destination.exists():
        return True
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from tools.longrun_training import load_ae
    ae, tok = load_ae(ae_dir, a.model_cache, dict(latent_dim=64, latent_length=latents, max_length=128))
    rows = []
    with torch.inference_mode():
        for i in range(CONTROL_SAMPLES):
            if time.monotonic() > deadline:
                return False
            g = torch.Generator(device='cpu').manual_seed(CONTROL_SEED + i)
            z = torch.randn(1, latents, 64, generator=g)
            z = (torch.nn.functional.normalize(z, dim=-1) * 8.0).cuda()
            ids = ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z)), max_length=128, min_length=5,
                              num_beams=4, do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.2)[0].tolist()
            rows.append(dict(seed=CONTROL_SEED + i, text=tok.decode(ids, skip_special_tokens=True).strip()))
    write_json(destination, rows)
    del ae
    torch.cuda.empty_cache()
    return True


def finish_phase(run, a, winner):
    code = train_phase(run)
    if code:
        return code
    w = run.workspace
    if not (w / 'selection.json').exists():
        code = run('evaluate')
        if code:
            return code
    specs = read_json(w / 'suite-status.json')['completed']
    if not all((w / 'runs' / s['name'] / 'final-test.json').exists() for s in specs):
        code = run('final-test')
        if code:
            return code
    deadline = run.deadline - 600
    controls = a.output / 'controls'
    controls.mkdir(exist_ok=True)
    old_latents, new_latents = 32, read_json(a.output / 'winner/longrun_config.json')['latent_length']
    for name, ae_dir, latents in (('B0', a.old_ae, old_latents), ('winner', Path(winner['ae_dir']), new_latents)):
        if not random_latent_control(a, ae_dir, latents, controls / f'random-latents-{name}.json', deadline):
            return 75
    model, tok, score_text = scorer(a)
    files = [(w / 'runs' / s['name'] / f'{split}-selected-samples.json', w / 'runs' / s['name'] / f'{split}-mgpt.json')
             for s in specs for split in ('validation', 'test')]
    files += [(controls / f'random-latents-{n}.json', controls / f'random-latents-{n}-mgpt.json') for n in ('B0', 'winner')]
    for source, destination in files:
        if not score_file(source, destination, model, tok, score_text, deadline):
            return 75
    subprocess.run([sys.executable, str(ROOT / 'tools/afrikaans_longrun.py'), 'summarize', '--workspace', str(w)], cwd=ROOT)
    write_json(a.output / 'diffusion_complete.json', dict(runs=[s['name'] for s in specs], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('finish_complete', runs=len(specs))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('train', 'finish'))
    p.add_argument('--output', type=Path, required=True, help='the screen output directory from job 1')
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--old-ae', type=Path, required=True)
    p.add_argument('--mgpt', type=Path, required=True)
    p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    # As in screen.py: outlive the training subprocess on TERM; in-process scoring stops at the deadline.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    deadline = time.monotonic() + a.seconds
    winner_path = a.output / 'winner/winner.json'
    if not winner_path.exists():
        log('skip', reason='screen produced no winner')
        return 0
    winner = read_json(winner_path)
    if not winner['proceed_to_diffusion']:
        log('skip', reason='winner did not improve selection CE over B0; diffusion not run (PROTOCOL.md 5.3)')
        return 0
    run = Longrun(a, winner, deadline)
    return train_phase(run) if a.phase == 'train' else finish_phase(run, a, winner)


if __name__ == '__main__':
    raise SystemExit(main())
