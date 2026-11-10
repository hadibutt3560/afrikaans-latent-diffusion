"""Verify and summarize a matched self-conditioning experiment, without generation."""
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_inference_ablation import weighted, interval


def main():
    root = Path(sys.argv[1])
    groups, metadata, stats = {}, {}, {}
    for arm in ('control', 'aligned'):
        meta = json.loads((root/arm/'metadata.json').read_text())
        summary = json.loads((root/f'quality-{arm}'/'summary.json').read_text())
        assert meta['status'] == summary['status'] == 'completed'
        assert meta['normalize_sampling_self_condition'] == (arm == 'aligned')
        assert meta['sampler'] == 'ddim' and meta['inference_steps'] == 50
        assert meta['decoding']['max_length'] == 128
        path = root/arm/'trajectories.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['trace_sha256']
        rows = [json.loads(line) for line in (root/f'quality-{arm}'/'scored_trajectories.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(rows) == len(meta['seeds'])*50
        assert {(r['seed'],r['step']) for r in rows} == {(s,t) for s in meta['seeds'] for t in range(1,51)}
        final = sorted((r for r in rows if r['step']==50), key=lambda r:r['seed'])
        assert all(r['mgpt_score'] is not None for r in final)
        groups[arm], metadata[arm] = final, meta
        stats[arm] = dict(token_weighted_perplexity=weighted(final),
            median_perplexity=statistics.median(r['mgpt_perplexity'] for r in final),
            capped_without_eos=sum(r['reached_length_cap'] and not r['ended_with_eos'] for r in final))
    for key in ('seeds','diffusion_sha256','autoencoder_sha256','decoding','precision','gpu'):
        assert metadata['control'][key] == metadata['aligned'][key], key
    control, aligned = groups['control'], groups['aligned']
    n = len(control)
    wins = sum(a['mgpt_perplexity'] < c['mgpt_perplexity'] for a,c in zip(aligned,control))
    ratio = weighted(aligned)/weighted(control)
    bounds = interval(aligned,control)
    report = dict(stats=stats, seeds=metadata['control']['seeds'], lower_pairs=wins,
                  perplexity_ratio=ratio, paired_bootstrap_95_interval=bounds,
                  human_review='pending', scope='Fixed checkpoint, inference-seed confirmation; not independent training replicates or untouched test evaluation')
    (root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# Self-conditioning comparison', '',
           '| Setting | Token-weighted perplexity | Median | Capped without EOS |',
           '|---|---:|---:|---:|']
    for arm,r in stats.items():
        lines.append(f"| {arm} | {r['token_weighted_perplexity']:.2f} | {r['median_perplexity']:.2f} | {r['capped_without_eos']}/{n} |")
    lines += ['', f'Aligned lower in {wins}/{n} paired passages. Perplexity ratio {ratio:.3f}; exploratory paired bootstrap 95% interval {bounds[0]:.3f} to {bounds[1]:.3f}.', '',
              'Lower perplexity is not proof of better Afrikaans. Human grammar and coherence review is pending. No checkpoint changed. No untouched test evaluation.', '',
              'Review blind_review.csv without consulting review_key.csv. Rate grammar and coherence separately from 1 (very poor) to 5 (very good); leave uncertain cases blank and explain in notes.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    review = [(arm,r) for arm,rows in groups.items() for r in rows]
    random.Random(20260929).shuffle(review)
    with (root/'blind_review.csv').open('w',encoding='utf-8-sig',newline='') as f, (root/'review_key.csv').open('w',encoding='utf-8-sig',newline='') as k:
        writer, key = csv.writer(f), csv.writer(k)
        writer.writerow(['id','text','grammar_1_to_5','coherence_1_to_5','notes'])
        key.writerow(['id','arm','seed'])
        for i,(arm,row) in enumerate(review,1):
            writer.writerow([i,row['text'],'','',''])
            key.writerow([i,arm,row['seed']])
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
