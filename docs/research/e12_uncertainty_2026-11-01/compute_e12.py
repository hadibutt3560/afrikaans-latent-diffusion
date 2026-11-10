"""E12 uncertainty figures for the manuscript (1 November 2026).

- Large-minus-small test perplexity: Welch 95% interval and exact permutation
  test over all 252 divisions of the ten run means into two groups of five.
- Distinct-1 means and SDs per denoiser size.
- Full-decoder retraining (F12, update 26,520) minus original: paired document
  bootstrap of the mean sentence-chrF change on the 81 development passages,
  10,000 replicates, seed 20261101; conditional on the selected checkpoint.

Inputs: docs/research/stage4_final_evaluation_2026-10-23/test_seed_table.csv and
outputs/cluster-results/job-74177-review/job-74177/development-{B0,F12-026520}.json
(the latter are local artifacts; outputs/ is not tracked by git).
Writes e12.json next to this script.
"""
import csv, itertools, json, math, random
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
rows = list(csv.DictReader(open(ROOT / 'docs/research/stage4_final_evaluation_2026-10-23/test_seed_table.csv', encoding='utf8')))
s = [float(r['ppl']) for r in rows if r['family'] == 'small']
l = [float(r['ppl']) for r in rows if r['family'] == 'published']
d = np.mean(l) - np.mean(s)
vs, vl = np.var(s, ddof=1) / 5, np.var(l, ddof=1) / 5
se = math.sqrt(vs + vl)
df = (vs + vl) ** 2 / (vs ** 2 / 4 + vl ** 2 / 4)
t = stats.t.ppf(0.975, df)
pooled = s + l
count = total = 0
for c in itertools.combinations(range(10), 5):
    a = [pooled[i] for i in c]
    b = [pooled[i] for i in range(10) if i not in c]
    total += 1
    count += abs(np.mean(a) - np.mean(b)) >= abs(d) - 1e-12
d1 = {fam: [float(r['distinct_1']) for r in rows if r['family'] == fam] for fam in ('small', 'published')}

job = ROOT / 'outputs/cluster-results/job-74177-review/job-74177'
base = json.load(open(job / 'development-B0.json', encoding='utf8'))['rows']
f12 = json.load(open(job / 'development-F12-026520.json', encoding='utf8'))['rows']
assert [r['id'] for r in base] == [r['id'] for r in f12]
docs = {}
for b, f in zip(base, f12):
    docs.setdefault(b['document_id'], []).append(f['chrf'] - b['chrf'])
keys = list(docs)
rng = random.Random(20261101)
reps = []
for _ in range(10000):
    sample = [x for k in (rng.choice(keys) for _ in keys) for x in docs[k]]
    reps.append(sum(sample) / len(sample))

out = dict(
    ppl=dict(small=s, large=l, diff=d, welch_low=d - t * se, welch_high=d + t * se, df=df,
             perm_p=count / total, perm_count=f'{count}/{total}'),
    distinct1={k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in d1.items()},
    chrf=dict(diff=float(np.mean([f['chrf'] - b['chrf'] for b, f in zip(base, f12)])),
              low=float(np.percentile(reps, 2.5)), high=float(np.percentile(reps, 97.5)),
              docs=len(keys), reps=10000, seed=20261101),
)
(Path(__file__).parent / 'e12.json').write_text(json.dumps(out, indent=1), encoding='utf8')
print(json.dumps(out, indent=1))
