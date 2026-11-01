"""Job 1: autoencoder screen, winner selection and reconstruction comparison (PROTOCOL.md sections 4-6).

Resumable: completed stages are skipped on rerun. Exit 75 means a clean stop that should be
resubmitted with the same output directory.
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BUNDLE_ROOT', HERE.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
from recon_eval import evaluate, read_jsonl

ARMS = [  # order is fixed in the protocol
    dict(name='A', latents=32, extra=False),
    dict(name='B', latents=64, extra=False),
    dict(name='C', latents=32, extra=True),
    dict(name='D', latents=64, extra=True),
]
OLD_AE_SHA = '123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'
PARSIMONY = 0.02          # a 64-slot arm must beat its 32-slot partner's selection CE by more than 2% (relative)
BOOTSTRAP, SEED = 10000, 20261028


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


def log(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)), flush=True)


def cached_eval(name, ae_dir, panel, out, model_cache, decode):
    path = out / 'recon' / f'{name}-{panel}.jsonl'
    summary = path.with_suffix('.summary.json')
    if summary.exists():
        return json.loads(summary.read_text(encoding='utf8'))
    log('recon_eval', ae=name, panel=panel, decode=decode)
    return evaluate(ae_dir, model_cache, ROOT / 'panels' / f'{panel}.jsonl', path, decode)


def bootstrap(out, first, second, panel, metrics=('exact', 'chrf', 'nll')):
    """Paired document bootstrap of second minus first on the same passages."""
    import numpy as np
    a = {r['id']: r for r in read_jsonl(out / 'recon' / f'{first}-{panel}.jsonl')}
    b = {r['id']: r for r in read_jsonl(out / 'recon' / f'{second}-{panel}.jsonl')}
    assert a.keys() == b.keys()
    docs = sorted({r['document_id'] for r in a.values()})
    table = []
    for d in docs:
        ids = [i for i in a if a[i]['document_id'] == d]
        row = [len(ids), sum(a[i]['tokens'] for i in ids)]
        for m in metrics:
            row += [sum(a[i][m] for i in ids), sum(b[i][m] for i in ids)]
        table.append(row)
    t = np.array(table, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = t[rng.integers(0, len(t), (BOOTSTRAP, len(t)))].sum(1)
    point = t.sum(0)
    result = {}
    for k, m in enumerate(metrics):
        den = 1 if m == 'nll' else 0     # CE is per token, the others per passage
        diff = lambda s: (s[..., 3 + 2 * k] - s[..., 2 + 2 * k]) / s[..., den]
        name = 'ce' if m == 'nll' else m
        lo, hi = np.quantile(diff(samples), [.025, .975])
        result[name] = dict(difference=float(diff(point)), low=float(lo), high=float(hi))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--old-ae', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--arm-hours', type=float, default=6.0)
    p.add_argument('--preflight-updates', type=int, default=30)
    a = p.parse_args()
    # The scheduler's TERM reaches the whole process group; the trainer saves and exits 75, and
    # this process must outlive it to report that. Children reinstall their own handlers.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    import torch
    assert torch.cuda.is_available(), 'No usable GPU'
    x = torch.randn(256, 256, device='cuda')
    assert torch.isfinite(x @ x.T).all(), 'CUDA arithmetic failed'
    log('gpu', name=torch.cuda.get_device_name(), vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
    assert digest(a.old_ae / 'model.pt') == OLD_AE_SHA, 'Old autoencoder hash mismatch'

    # Baseline B0 (job 56531) in this environment, on every panel.
    for panel, decode in (('selection', False), ('V', True), ('reserve', True)):
        cached_eval('B0', a.old_ae, panel, out, a.model_cache, decode)

    env = dict(os.environ, BUNDLE_ROOT=str(ROOT))
    for arm in ARMS:
        folder = out / 'arms' / arm['name']
        status_path = folder / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['complete']:
            continue
        common = [sys.executable, '-u', str(HERE / 'train_ae.py'), '--arm', arm['name'], '--latents', str(arm['latents']),
                  '--model-cache', str(a.model_cache)] + (['--extra'] if arm['extra'] else [])
        # Effective batch is fixed at 128; only the micro-batch split may change if memory requires.
        batch = None
        for micro, accumulation in ((32, 4), (16, 8)):
            preflight = folder / f'preflight-{micro}x{accumulation}'
            split = ['--micro-batch', str(micro), '--accumulation', str(accumulation)]
            if not (preflight / 'preflight.json').exists():
                log('preflight', arm=arm['name'], micro_batch=micro, accumulation=accumulation)
                code = subprocess.run(common + split + ['--output', str(preflight), '--preflight', str(a.preflight_updates)], env=env).returncode
                if code == 75:
                    return 75
                if code != 0:
                    log('preflight_failed', arm=arm['name'], micro_batch=micro, exit=code)
                    continue
            batch = split
            log('train', arm=arm['name'], preflight=json.loads((preflight / 'preflight.json').read_text()))
            break
        if batch is None:
            raise RuntimeError(f"Arm {arm['name']} failed preflight at both micro-batch sizes")
        result = subprocess.run(common + batch + ['--output', str(folder), '--max-seconds', str(a.arm_hours * 3600)], env=env)
        if result.returncode == 75:
            return 75
        if result.returncode != 0:
            raise RuntimeError(f"Arm {arm['name']} failed with exit {result.returncode}")

    # Every arm on the selection panel (CE, FP32) and the V panel (decoded).
    arms = {}
    for arm in ARMS:
        ae_dir = out / 'arms' / arm['name'] / 'best'
        status = json.loads((out / 'arms' / arm['name'] / 'status.json').read_text())
        arms[arm['name']] = dict(arm, status=status, selection=cached_eval(arm['name'], ae_dir, 'selection', out, a.model_cache, False),
                                 V=cached_eval(arm['name'], ae_dir, 'V', out, a.model_cache, True))
    b0 = dict(selection=cached_eval('B0', a.old_ae, 'selection', out, a.model_cache, False),
              V=cached_eval('B0', a.old_ae, 'V', out, a.model_cache, True),
              reserve=cached_eval('B0', a.old_ae, 'reserve', out, a.model_cache, True))

    # Selection rule (PROTOCOL.md section 5.2), applied once.
    selection_path = out / 'selection.json'
    if not selection_path.exists():
        eligible = {k: v for k, v in arms.items() if v['status']['complete'] and v['V']['cap'] + v['V']['no_eos'] <= 2}
        ranked = sorted(eligible, key=lambda k: eligible[k]['selection']['ce'])
        if not ranked:
            write_json(selection_path, dict(winner=None, reason='no eligible arm'))
        else:
            winner = ranked[0]
            partner = {'B': 'A', 'D': 'C'}.get(winner)
            if partner in eligible and eligible[winner]['selection']['ce'] > eligible[partner]['selection']['ce'] * (1 - PARSIMONY):
                winner = partner
            proceed = eligible[winner]['selection']['ce'] < b0['selection']['ce']
            write_json(selection_path, dict(winner=winner, ranked=ranked, parsimony=PARSIMONY,
                       selection_ce={k: v['selection']['ce'] for k, v in eligible.items()}, b0_selection_ce=b0['selection']['ce'],
                       proceed_to_diffusion=proceed, winner_sha256=digest(out / 'arms' / winner / 'best/model.pt')))
    selection = json.loads(selection_path.read_text())
    log('selection', **selection)

    report = dict(b0=b0, arms={k: dict(selection=v['selection'], V=v['V'], status=v['status']) for k, v in arms.items()}, selection=selection)
    # Factorial and baseline contrasts on the V panel (descriptive, paired by passage).
    contrasts = {'A_minus_B0': ('B0', 'A'), 'recipe_plus_slots_B_minus_B0': ('B0', 'B'), 'slots_B_minus_A': ('A', 'B'),
                 'slots_D_minus_C': ('C', 'D'), 'data_C_minus_A': ('A', 'C'), 'data_D_minus_B': ('B', 'D')}
    report['V_contrasts'] = {name: bootstrap(out, first, second, 'V') for name, (first, second) in contrasts.items()}
    if selection.get('winner'):
        w = selection['winner']
        winner_dir = out / 'arms' / w / 'best'
        report['winner_reserve'] = cached_eval(f'winner-{w}', winner_dir, 'reserve', out, a.model_cache, True)
        report['reserve_contrast_winner_minus_B0'] = bootstrap(out, 'B0', f'winner-{w}', 'reserve')
        report['V_contrast_winner_minus_B0'] = bootstrap(out, 'B0', w, 'V')
        config = json.loads((ROOT / 'configs/afrikaans_longrun.json').read_text())
        config.update(name=f'afrikaans-ae-screen-winner-{w}', expected_autoencoder_sha256=selection['winner_sha256'],
                      latent_length=next(arm['latents'] for arm in ARMS if arm['name'] == w))
        (out / 'winner').mkdir(exist_ok=True)
        write_json(out / 'winner/longrun_config.json', config)
        write_json(out / 'winner/winner.json', dict(arm=w, ae_dir=str(winner_dir), proceed_to_diffusion=selection['proceed_to_diffusion']))
    write_json(out / 'screen_report.json', report)
    log('screen_complete', winner=selection.get('winner'), proceed=selection.get('proceed_to_diffusion'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
