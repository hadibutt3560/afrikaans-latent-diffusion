"""Write summary.json and RESULTS.md for the Swahili check (PROTOCOL.md section 8) from the run's existing outputs.

Usage: python summarise.py <outputs/swahili-check-20261101>
CPU only; no model is loaded. Reads s1-reconstruction.jsonl, s2-random.jsonl, s3-lid.jsonl, s4.json, gates.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

OTHER_SWAHILI = {'swa_Latn', 'swc_Latn'}   # Swahili codes other than swh_Latn (section 6)
BINS = {0: '<=64', 1: '65-128', 2: '129-192'}
RATIO = 1.20                               # section 5 wording rule
BOOT, BOOT_SEED = 2000, 20261104           # section 4


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def s1_readouts(rows):
    tokens = sum(r['tokens'] for r in rows)
    src, pred, match = (sum(r[k] for r in rows) for k in ('num_src', 'num_pred', 'num_match'))
    return dict(n=len(rows), ce=sum(r['ce'] * r['tokens'] for r in rows) / tokens,
                accuracy=sum(r['acc'] * r['tokens'] for r in rows) / tokens,
                exact=sum(r['exact'] for r in rows), exact_rate=float(np.mean([r['exact'] for r in rows])),
                chrf=float(np.mean([r['chrf'] for r in rows])),
                numeral_recall=match / src if src else None, numeral_precision=match / pred if pred else None,
                cap=sum(r['cap'] for r in rows))


def file_bootstrap(correct, shuffled):
    """Source-file bootstrap for exact rate, mean chrF (correct arm) and correct-minus-shuffled CE."""
    by_file = defaultdict(list)
    shuffled_by_id = {r['id']: r for r in shuffled}
    for r in correct:
        by_file[r['file']].append((r, shuffled_by_id[r['id']]))
    files = sorted(by_file)
    rng = np.random.default_rng(BOOT_SEED)
    stats = defaultdict(list)
    for _ in range(BOOT):
        pairs = [p for f in rng.choice(files, len(files)) for p in by_file[f]]
        c = [p[0] for p in pairs]; s = [p[1] for p in pairs]
        stats['exact_rate'].append(np.mean([r['exact'] for r in c]))
        stats['chrf'].append(np.mean([r['chrf'] for r in c]))
        ce = lambda rs: sum(r['ce'] * r['tokens'] for r in rs) / sum(r['tokens'] for r in rs)
        stats['ce_correct_minus_shuffled'].append(ce(c) - ce(s))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}, len(files)


def main():
    out = Path(sys.argv[1])
    s1 = jsonl(out / 's1-reconstruction.jsonl')
    arms = {a: [r for r in s1 if r['arm'] == a] for a in ('correct', 'shuffled')}
    recon = {a: s1_readouts(rs) | dict(bins={BINS[b]: s1_readouts([r for r in rs if r['bin'] == b])
                                             for b in sorted({r['bin'] for r in rs})}) for a, rs in arms.items()}
    intervals, n_files = file_bootstrap(arms['correct'], arms['shuffled'])
    recon['difference_ce_correct_minus_shuffled'] = recon['correct']['ce'] - recon['shuffled']['ce']
    recon['bootstrap_95'] = intervals | dict(replicates=BOOT, seed=BOOT_SEED, source_files=n_files)

    s4 = json.loads((out / 's4.json').read_text(encoding='utf8'))
    s2 = jsonl(out / 's2-random.jsonl')
    diffusion = [k for k in s4 if k.endswith('-valid') and k != 'real-valid']
    diffusion_ppl = float(np.mean([s4[k]['ppl'] for k in diffusion]))
    random_ppl = s4['random-latents']['ppl']
    random = dict(n=len(s2), ppl=random_ppl, diffusion_mean_ppl=diffusion_ppl, ratio=random_ppl / diffusion_ppl,
                  mean_words=s4['random-latents']['mean_words'], distinct2=s4['random-latents']['distinct2'],
                  empty=sum(r['score'].get('empty', False) for r in s2), sentinel=sum(r['sentinel'] for r in s2),
                  wording=('random latents decoded by the same decoder reach similar raw perplexity'
                           if random_ppl <= RATIO * diffusion_ppl else 'random latents score clearly worse than diffusion samples'))

    lid = defaultdict(lambda: defaultdict(list))
    for r in jsonl(out / 's3-lid.jsonl'):
        lid[r['bank']]['pooled'].append(r['label']); lid[r['bank']][r['band']].append(r['label'])
    share = lambda labels, test: sum(map(test, labels)) / len(labels)
    language = {bank: {group: dict(n=len(l), swh_Latn=share(l, lambda x: x == 'swh_Latn'),
                                   other_swahili=share(l, lambda x: x in OTHER_SWAHILI), eng_Latn=share(l, lambda x: x == 'eng_Latn'))
                       for group, l in groups.items()} for bank, groups in lid.items()}

    summary = dict(protocol='docs/research/swahili_check_2026-11-01/PROTOCOL.md', gates=json.loads((out / 'gates.json').read_text(encoding='utf8')),
                   s1_reconstruction=recon, s2_random_latents=random, s3_language=language, s4_length_and_copying=s4)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    c, sh, ci = recon['correct'], recon['shuffled'], recon['bootstrap_95']
    lines = ['# Swahili check: results', '',
             'Produced by `summarise.py` from the outputs of `run_check.py`, under PROTOCOL.md. All results are descriptive and on validation data.', '',
             '## S1 reconstruction (200 passages)', '',
             '| Arm | CE | Token accuracy | Exact | chrF | Numeral recall | Numeral precision | At cap |', '|---|---|---|---|---|---|---|---|']
    for name, a in (('Correct', c), ('Shuffled', sh)):
        fmt = lambda x: '–' if x is None else f'{x:.2f}'
        lines.append(f"| {name} | {a['ce']:.3f} | {a['accuracy']:.3f} | {a['exact']}/{a['n']} | {a['chrf']:.1f} | {fmt(a['numeral_recall'])} | {fmt(a['numeral_precision'])} | {a['cap']} |")
    lines += ['', f"Source-file bootstrap ({BOOT} replicates, {ci['source_files']} files): exact rate [{ci['exact_rate'][0]:.3f}, {ci['exact_rate'][1]:.3f}], "
              f"chrF [{ci['chrf'][0]:.1f}, {ci['chrf'][1]:.1f}], CE correct − shuffled {recon['difference_ce_correct_minus_shuffled']:.3f} "
              f"[{ci['ce_correct_minus_shuffled'][0]:.3f}, {ci['ce_correct_minus_shuffled'][1]:.3f}].", '',
              '## S2 random latents (1,000)', '',
              f"Perplexity {random['ppl']:.1f} against a diffusion validation mean of {random['diffusion_mean_ppl']:.2f} (ratio {random['ratio']:.2f}). "
              f"Fixed wording: **{random['wording']}**. Mean words {random['mean_words']:.1f}, distinct-2 {random['distinct2']:.3f}, "
              f"empty {random['empty']}, containing `<extra_id_` {random['sentinel']}.", '',
              '## S3 language identification (GlotLID top label, pooled)', '', '| Bank | n | swh_Latn | Other Swahili | eng_Latn |', '|---|---|---|---|---|']
    for bank, groups in language.items():
        p = groups['pooled']
        lines.append(f"| {bank} | {p['n']} | {p['swh_Latn']:.3f} | {p['other_swahili']:.3f} | {p['eng_Latn']:.3f} |")
    lines += ['', '## S4 length and copying', '', '| Bank | n | Mean words | mGPT PPL | distinct-2 | Exact copies | 4-gram overlap | High-overlap rows |', '|---|---|---|---|---|---|---|---|']
    for bank, v in s4.items():
        if isinstance(v, dict) and 'ppl' in v:
            lines.append(f"| {bank} | {v['n']} | {v['mean_words']:.1f} | {v['ppl']:.2f} | {v['distinct2']:.3f} | {v.get('exact_copies', '–')} | "
                         f"{v.get('fourgram_overlap', float('nan')):.3f} | {v.get('high_overlap_rows', '–')} |")
    lines += ['', 'The supplied splits have documented cross-split overlap, so the reference overlap is itself inflated.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps(dict(exact=c['exact'], chrf=round(c['chrf'], 1), chrf_ci=ci['chrf'], shuffled_chrf=round(sh['chrf'], 1),
                          numeral_recall=c['numeral_recall'], random=random['wording'], ratio=round(random['ratio'], 2))))


if __name__ == '__main__':
    main()
