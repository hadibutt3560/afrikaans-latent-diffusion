"""Summarize existing scored trajectories without running models."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    results = {}
    for budget in (50, 100, 150, 250):
        directory = args.root / f'quality-{budget}'
        summary = json.loads((directory / 'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError(f'Incomplete scoring: {budget}')
        rows = [json.loads(line) for line in (directory / 'scored_trajectories.jsonl').read_text(encoding='utf8').splitlines()]
        seeds = sorted({r['seed'] for r in rows})
        first = [next(r for r in rows if r['seed'] == s and r['step'] == 1) for s in seeds]
        final = [next(r for r in rows if r['seed'] == s and r['step'] == budget) for s in seeds]
        scores = [r['mgpt_score'] for r in final]
        results[budget] = dict(
            seeds=len(seeds),
            final_token_weighted_perplexity=math.exp(sum(s['nll'] for s in scores) / sum(s['tokens'] for s in scores)),
            final_median_perplexity=statistics.median(r['mgpt_perplexity'] for r in final),
            first_median_perplexity=statistics.median(r['mgpt_perplexity'] for r in first),
            final_better_than_first=sum(b['mgpt_perplexity'] < a['mgpt_perplexity'] for a, b in zip(first, final)),
            final_perplexity_by_seed={r['seed']: r['mgpt_perplexity'] for r in final},
            final_samples=[dict(seed=r['seed'], text=r['text']) for r in final])
    lines = ['# Inference-budget comparison', '',
             'Same frozen 50,000-step diffusion checkpoint, 20 matched seeds per budget, mGPT scoring of every intermediate prediction. No retraining.', '',
             '| Steps | Final token-weighted perplexity | Final median perplexity | First median perplexity | Final better than first |',
             '|---|---:|---:|---:|---:|']
    for budget, r in results.items():
        lines.append(f"| {budget} | {r['final_token_weighted_perplexity']:.2f} | {r['final_median_perplexity']:.2f} | {r['first_median_perplexity']:.2f} | {r['final_better_than_first']}/{r['seeds']} |")
    lines += ['', 'Lower perplexity is greater predictability under mGPT, not proof of good grammar, meaning, or factual accuracy. Intermediate predictions are not independent shorter runs. These 20 seeds are exploratory and are not independent training replicates. No untouched test evaluation was performed. Do not choose a stopping point solely by minimum perplexity.', '', '## Examples (seed 42, fixed in advance)']
    for budget, r in results.items():
        lines += ['', f'### {budget} steps', r['final_samples'][0]['text']]
    (args.root / 'budget_comparison.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf8')
    (args.root / 'BUDGET_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines[:11]))


if __name__ == '__main__':
    main()
