"""Tokenisation fertility from saved outputs (README.md in this folder). No model is run.

Fertility is mT5 tokens (no special tokens) per whitespace word, pooled over a bank
(total tokens / total words). Paths default to the research repository layout; pass
--research to point elsewhere.
"""
import argparse, json, math, statistics
from pathlib import Path

from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--research', default='C:/Users/SCIENTIST/Documents/GitHub/research')
R = Path(ap.parse_args().research)
TOK = AutoTokenizer.from_pretrained(str(R / 'outputs/longrun-local-inputs/models/mt5-base'))


def jl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def counts(text):
    return len(TOK(text, add_special_tokens=False)['input_ids']), len(text.split())


def pooled(texts):
    t = w = 0
    for x in texts:
        a, b = counts(x)
        t += a; w += b
    return t / w


def bank(rows):
    return pooled([r['text'] for r in rows if r['text'].strip()])


out = {'definition': 'mT5 tokens without special tokens per whitespace word, pooled over the bank'}
ev = R / 'outputs/longrun-local-evaluation'
d_runs = R / 'outputs/cluster-results/ae-diffusion-77176/screen/diffusion/runs'
banks = {'human test': bank(json.loads((ev / 'test-reference-mgpt.json').read_text(encoding='utf8'))['rows']),
         'AR test': bank(jl(R / 'outputs/ar-test-20261105/ar-test-scored.jsonl'))}
for label, folder, seeds in (('original', ev / 'runs', range(42, 47)), ('retrained', d_runs, range(42, 45))):
    for fam, name in (('small', 'small'), ('published', 'large')):
        vals = [bank(json.loads((folder / f'{fam}-seed{s}' / 'test-mgpt.json').read_text(encoding='utf8'))['rows']) for s in seeds]
        banks[f'{label} {name} diffusion test'] = dict(mean=statistics.mean(vals), min=min(vals), max=max(vals), runs=len(vals))
out['generation_banks'] = banks

# Reconstruction against source fertility: 681 extra-data passages (original, A, D)
rec = R / 'outputs/cluster-results/followup-81249/outputs/followup/recon681'
src = jl(rec / 'B0.jsonl')
fert = {r['id']: (lambda c: c[0] / c[1])(counts(r['source'])) for r in src}
ordered = sorted(fert.values())
cut = [ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]]
out['recon681_source_fertility'] = dict(pooled=pooled([r['source'] for r in src]), tercile_cuts=cut)
res = {}
for arm in ('B0', 'A', 'D'):
    rows = jl(rec / f'{arm}.jsonl')
    terc = {0: [], 1: [], 2: []}
    for r in rows:
        f = fert[r['id']]
        terc[0 if f < cut[0] else 1 if f < cut[1] else 2].append(r)
    rho_chrf = spearmanr([fert[r['id']] for r in rows], [r['chrf'] for r in rows])
    rho_len = spearmanr([fert[r['id']] for r in rows], [r['tokens'] for r in rows])
    res[arm] = dict(exact_by_tercile=[sum(r['exact'] for r in v) / len(v) for v in terc.values()],
                    n_by_tercile=[len(v) for v in terc.values()],
                    chrf_by_tercile=[statistics.mean(r['chrf'] for r in v) for v in terc.values()],
                    spearman_fertility_chrf=[rho_chrf.statistic, rho_chrf.pvalue],
                    spearman_fertility_tokens=[rho_len.statistic, rho_len.pvalue])
out['recon681_by_source_fertility'] = res
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results/fertility.json').write_text(json.dumps(out, indent=1), encoding='utf8')
print(json.dumps(out, indent=1))
