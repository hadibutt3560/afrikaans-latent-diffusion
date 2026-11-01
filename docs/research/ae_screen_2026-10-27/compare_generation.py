"""Compare the new diffusion suite with the existing five-seed benchmark (PROTOCOL.md section 7).

Usage: python compare_generation.py <retrieved results root containing outputs/screen/diffusion>
Reads the old benchmark from outputs/longrun-local-evaluation/runs/. No model is run.
"""
import json, math, sys, unicodedata
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'outputs/longrun-local-evaluation/runs'
TRAIN = ROOT / 'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'
REFERENCE = ROOT / 'outputs/longrun-local-evaluation/validation-reference-mgpt.json'
BANDS = [(1, 10), (11, 20), (21, 35), (36, 10 ** 9)]   # the 27 October audit bands
SIZES = ('small', 'published')
MIN_BAND_ROWS = 20   # NOTE_2026-11-01_empty_bands.md: standardised PPL needs at least this many scored rows per band


def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def band(words):
    return next(i for i, (a, b) in enumerate(BANDS) if a <= words <= b)


def band_weights():
    rows = json.loads(REFERENCE.read_text(encoding='utf8'))['rows']
    counts = np.zeros(len(BANDS))
    for r in rows:
        counts[band(len(r['text'].split()))] += 1
    return counts / counts.sum()


def run_metrics(rows, weights, train):
    scored = [r for r in rows if r.get('mgpt')]
    raw = math.exp(sum(r['mgpt']['nll'] for r in scored) / sum(r['mgpt']['tokens'] for r in scored))
    log_band = []
    for k in range(len(BANDS)):
        rs = [r for r in scored if band(len(r['text'].split())) == k]
        log_band.append(sum(r['mgpt']['nll'] for r in rs) / sum(r['mgpt']['tokens'] for r in rs) if rs else float('nan'))
    counts = [sum(band(len(r['text'].split())) == k for r in scored) for k in range(len(BANDS))]
    standardised = math.exp(float(np.dot(weights, log_band))) if min(counts) >= MIN_BAND_ROWS else None
    bigrams = [tuple(w[i:i + 2]) for r in rows for w in [r['text'].split()] for i in range(len(w) - 1)]
    return dict(raw_ppl=raw, standardised_ppl=standardised, distinct2=len(set(bigrams)) / max(1, len(bigrams)),
                copies=sum(norm(r['text']) in train for r in rows), mean_words=float(np.mean([len(r['text'].split()) for r in rows])),
                empty=sum(not r.get('mgpt') for r in rows), band_counts=counts)


def load(folder, size):
    runs = sorted(folder.glob(f'{size}-seed*/test-mgpt.json'))
    return {p.parent.name: json.loads(p.read_text(encoding='utf8'))['rows'] for p in runs}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    from scipy.stats import t
    half = t.ppf(.975, df) * math.sqrt(va + vb)
    d = b.mean() - a.mean()
    return dict(difference=float(d), low=float(d - half), high=float(d + half), df=float(df))


def main():
    new_root = Path(sys.argv[1]) / 'outputs/screen'
    weights = band_weights()
    train = {norm(json.loads(line)['text']) for line in TRAIN.read_text(encoding='utf8').splitlines() if line.strip()}
    report = dict(weights=weights.tolist(), sizes={})
    controls = {}
    for name in ('B0', 'winner'):
        path = new_root / 'controls' / f'random-latents-{name}-mgpt.json'
        if path.exists():
            controls[name] = run_metrics(json.loads(path.read_text(encoding='utf8'))['rows'], weights, train)
    report['random_latent_control'] = controls
    for size in SIZES:
        old = {k: run_metrics(v, weights, train) for k, v in load(OLD, size).items()}
        new = {k: run_metrics(v, weights, train) for k, v in load(new_root / 'diffusion/runs', size).items()}
        if len(new) < 2:
            report['sizes'][size] = dict(status='fewer than two new runs', new_runs=list(new))
            continue
        mean = lambda d, m: None if any(x[m] is None for x in d.values()) else float(np.mean([x[m] for x in d.values()]))
        supported = all(x['standardised_ppl'] is not None for x in (*old.values(), *new.values(), *controls.values()))
        test = {m: welch([x[m] for x in old.values()], [x[m] for x in new.values()]) for m in ('standardised_ppl', 'raw_ppl')
                if m == 'raw_ppl' or supported}
        if not supported:
            test['standardised_ppl'] = dict(status='insufficient_support', min_band_rows=MIN_BAND_ROWS)
        improved = None if not supported else (test['standardised_ppl']['high'] < 0
                    and mean(new, 'distinct2') >= 0.95 * mean(old, 'distinct2')
                    and sum(x['copies'] for x in new.values()) / len(new) <= 2 * sum(x['copies'] for x in old.values()) / len(old)
                    and ('winner' not in controls or controls['winner']['standardised_ppl'] > mean(new, 'standardised_ppl')))
        report['sizes'][size] = dict(old=old, new=new, welch_new_minus_old=test, improved_by_protocol=improved,
                                     means={side: {m: mean(d, m) for m in ('raw_ppl', 'standardised_ppl', 'distinct2', 'mean_words')}
                                            for side, d in (('old', old), ('new', new))})
    out = new_root / 'generation_comparison.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({s: report['sizes'][s].get('means', report['sizes'][s]) | {'improved': report['sizes'][s].get('improved_by_protocol')}
                      for s in SIZES}, indent=2))


if __name__ == '__main__':
    main()
