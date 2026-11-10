"""CPU analysis of the follow-up outputs, as fixed in PROTOCOL.md sections 3-5.

Usage: python analyze.py <followup output dir> <77176 screen dir> <research repo root>
Writes analysis.json next to this script.
"""
import json, math, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OUT, SCREEN, REPO = (Path(p) for p in sys.argv[1:4])
BANDS = [(1, 10), (11, 20), (21, 35), (36, 10**9)]
WEIGHTS = [0.140, 0.223, 0.349, 0.288]
MARGIN = 0.10


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def rows(p):
    d = read(p)
    return d['rows'] if isinstance(d, dict) else d


def jsonl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def band(words):
    return next(i for i, (a, b) in enumerate(BANDS) if a <= words <= b)


def norm(t):
    return ' '.join(unicodedata.normalize('NFC', t).casefold().split())


def bank_stats(rs):
    scored = [r for r in rs if r.get('mgpt')]
    nll = sum(r['mgpt']['nll'] for r in scored)
    tok = sum(r['mgpt']['tokens'] for r in scored)
    by = defaultdict(lambda: [0.0, 0])
    for r in scored:
        b = by[band(len(r['text'].split()))]
        b[0] += r['mgpt']['nll']
        b[1] += r['mgpt']['tokens']
    counts = Counter(band(len(r['text'].split())) for r in scored)
    supported = all(counts[k] >= 20 for k in range(4))
    std = math.exp(sum(w * by[k][0] / by[k][1] for k, w in enumerate(WEIGHTS))) if supported else None
    bigrams = [tuple(r['text'].split()[i:i + 2]) for r in rs for i in range(len(r['text'].split()) - 1)]
    return dict(n=len(rs), ppl=math.exp(nll / tok), std_ppl=std, band_counts=[counts[k] for k in range(4)],
                mean_words=sum(len(r['text'].split()) for r in rs) / len(rs),
                distinct2=len(set(bigrams)) / max(1, len(bigrams)), sentinel=sum('<extra_id' in r['text'] or r.get('sentinel', False) for r in rs),
                eos=sum(r.get('eos', True) for r in rs), cap=sum(r.get('length_cap', False) for r in rs))


def boot(values, seed, reps=2000):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [means[int(.025 * reps)], means[int(.975 * reps) - 1]]


# F1: self-conditioning audit
audit = {}
for off in sorted((OUT / 'audit').glob('*-off-mgpt.json')):
    name = off.name[:-len('-off-mgpt.json')]
    a, b = rows(off), rows(off.with_name(f'{name}-on-mgpt.json'))
    assert [r['seed'] for r in a] == [r['seed'] for r in b]
    diffs = [(y['mgpt']['nll'] / y['mgpt']['tokens']) - (x['mgpt']['nll'] / x['mgpt']['tokens'])
             for x, y in zip(a, b) if x.get('mgpt') and y.get('mgpt')]
    audit[name] = dict(off=bank_stats(a), on=bank_stats(b), identical_text=sum(x['text'] == y['text'] for x, y in zip(a, b)),
                       paired_mean=sum(diffs) / len(diffs), ci=boot(diffs, 20261106), n=len(diffs))
decision = {}
for fam in ('D', 'B0'):
    runs = {k: v for k, v in audit.items() if k.startswith(fam + '-')}
    if not runs:
        continue
    pooled = sum(v['paired_mean'] for v in runs.values()) / len(runs)
    inside = all(-MARGIN <= v['ci'][0] and v['ci'][1] <= MARGIN for v in runs.values())
    sign = max(sum(v['paired_mean'] > 0 for v in runs.values()), sum(v['paired_mean'] < 0 for v in runs.values()))
    decision[fam] = dict(pooled_mean=pooled, all_ci_within_margin=inside, same_sign=sign,
                         verdict='no material effect' if inside else ('material effect' if abs(pooled) > MARGIN and sign >= 5 else 'inconclusive'))

# F2: fitted prior on D, against D diffusion test banks and D isotropic control
train_texts = {norm(r['text']) for r in jsonl(REPO / 'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl')}
extra_texts = {norm(r['text']) for r in jsonl(REPO / 'data/extra_afrikaans/processed/extra_train.jsonl')}
def grams(t, n=4):
    w = norm(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
train4 = set().union(*(grams(t) for t in train_texts))
extra4 = set().union(*(grams(t) for t in extra_texts))
def copying(rs):
    g = [x for r in rs for x in grams(r['text'])]
    return dict(exact_nchlt=sum(norm(r['text']) in train_texts for r in rs), exact_extra=sum(norm(r['text']) in extra_texts for r in rs),
                share4_nchlt=sum(x in train4 for x in g) / max(1, len(g)), share4_extra=sum(x in extra4 for x in g) / max(1, len(g)))
prior = {}
for f in sorted((OUT / 'prior').glob('C2-D-*-mgpt.json')):
    rs = rows(f)
    prior[f.name[:-10]] = dict(**bank_stats(rs), **copying(rs))
for run in sorted((SCREEN / 'diffusion/runs').iterdir()):
    rs = rows(run / 'test-mgpt.json')
    prior[f'D-diffusion-{run.name}'] = dict(**bank_stats(rs), **copying(rs))
iso = rows(SCREEN / 'controls/random-latents-winner-mgpt.json')
prior['D-isotropic'] = dict(**bank_stats(iso), **copying(iso))
pub = [v['std_ppl'] for k, v in prior.items() if k.startswith('D-diffusion-published')]
c2 = [v['std_ppl'] for k, v in prior.items() if k.startswith('C2-D')]
prior_rule = dict(published_range=[min(pub), max(pub)], published_mean=sum(pub) / len(pub), c2_mean=sum(c2) / len(c2),
                  verdict='within published range' if min(pub) <= sum(c2) / len(c2) <= max(pub) else
                  ('clearly worse (>1.20x)' if sum(c2) / len(c2) > 1.2 * sum(pub) / len(pub) else 'neither'))

# F3: reconstruction on extra_monitor
recon = {k: jsonl(OUT / f'recon681/{k}.jsonl') for k in ('B0', 'A', 'D')}
src = {r['id']: r['source'] for r in jsonl(REPO / 'data/extra_afrikaans/processed/extra_monitor.jsonl')}
def summary(rs):
    e, p, m = (sum(r[k] for r in rs) for k in ('number_expected', 'number_predicted', 'number_matched'))
    return dict(n=len(rs), exact=sum(r['exact'] for r in rs), chrf=sum(r['chrf'] for r in rs) / len(rs),
                ce=sum(r['nll'] for r in rs) / sum(r['tokens'] for r in rs), numeral_recall=m / e if e else None,
                numeral_precision=m / p if p else None, cap=sum(r['at_cap'] for r in rs))
f3 = {k: dict(all=summary(v), **{s: summary([r for r in v if src[r['id']] == s]) for s in ('gov-za', 'vukuzenzele')}) for k, v in recon.items()}
docs = sorted({r['document_id'] for r in recon['B0']})
rng = random.Random(20261107)
by_doc = {k: defaultdict(list) for k in recon}
for k, v in recon.items():
    for r in v:
        by_doc[k][r['document_id']].append(r)
reps = defaultdict(list)
for _ in range(2000):
    pick = [docs[rng.randrange(len(docs))] for _ in docs]
    for k in ('A', 'D'):
        a = [r for d in pick for r in by_doc[k][d]]
        b = [r for d in pick for r in by_doc['B0'][d]]
        reps[f'{k}-B0 exact'].append(sum(r['exact'] for r in a) / len(a) - sum(r['exact'] for r in b) / len(b))
        reps[f'{k}-B0 chrf'].append(sum(r['chrf'] for r in a) / len(a) - sum(r['chrf'] for r in b) / len(b))
for k in ('A', 'D'):
    f3[f'{k}-B0'] = dict(exact_diff=f3[k]['all']['exact'] / 681 - f3['B0']['all']['exact'] / 681,
                         exact_ci=[sorted(reps[f'{k}-B0 exact'])[i] for i in (50, 1949)],
                         chrf_diff=f3[k]['all']['chrf'] - f3['B0']['all']['chrf'],
                         chrf_ci=[sorted(reps[f'{k}-B0 chrf'])[i] for i in (50, 1949)], documents=len(docs))

result = dict(f1_audit=audit, f1_decision=decision, f2_prior=prior, f2_rule=prior_rule, f3_reconstruction=f3,
              reproduction=read(OUT / 'audit/reproduction.json'))
Path(__file__).with_name('analysis.json').write_text(json.dumps(result, indent=1), encoding='utf8')
print(json.dumps(dict(f1_decision=decision, f2_rule=prior_rule, f3={k: f3[k] for k in ('A-B0', 'D-B0')},
                      f3_all={k: f3[k]['all'] for k in ('B0', 'A', 'D')}), indent=1))
