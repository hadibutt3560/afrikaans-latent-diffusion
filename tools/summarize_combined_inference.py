"""Summarize completed combined inference study from saved scores only."""
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_inference_ablation import interval, weighted

root = Path(sys.argv[1])
groups, results = {}, {}
for steps in (50, 250):
    for arm in ('ddim64', 'ddim128', 'ddpm128'):
        name = f'{arm}-{steps}'
        folder = root / f'quality-{name}'
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['status'] == 'completed'
        rows = [json.loads(line) for line in (folder / 'scored_trajectories.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(rows) == 20 * steps
        assert {(r['seed'], r['step']) for r in rows} == {(s,t) for s in range(82,102) for t in range(1,steps+1)}
        final = sorted((r for r in rows if r['step'] == steps), key=lambda r:r['seed'])
        groups[name] = final
        results[name] = dict(perplexity=weighted(final), median=statistics.median(r['mgpt_perplexity'] for r in final),
            unfinished=sum(r['reached_length_cap'] and not r['ended_with_eos'] for r in final),
            mean_words=statistics.mean(r['word_count'] for r in final),
            last_ten_change_fraction=sum(bool(r['changed_from_previous']) for r in rows if r['step'] > steps-10)/200,
            final_better_than_first=sum(s['final_perplexity'] < s['first_perplexity'] for s in summary['stability'].values()))
comparisons = {}
for steps in (50,250):
    candidate=groups[f'ddim128-{steps}']
    for control_name in (f'ddim64-{steps}', f'ddpm128-{steps}'):
        control=groups[control_name]
        comparisons[f'ddim128-{steps} vs {control_name}'] = dict(
            ratio=weighted(candidate)/weighted(control), interval=interval(candidate,control),
            lower_pairs=sum(a['mgpt_perplexity'] < b['mgpt_perplexity'] for a,b in zip(candidate,control)),
            identical_pairs=sum(a['text'] == b['text'] for a,b in zip(candidate,control)))
lines=['# Combined inference results', '', 'All six conditions and all 18,000 intermediate scores completed. Same saved checkpoint; fresh seeds 82-101. No training.', '',
    '| Setting | Final token-weighted PPL | Median PPL | Capped without EOS | Last ten transitions changed |',
    '|---|---:|---:|---:|---:|']
for name,r in results.items():
    lines.append(f"| {name} | {r['perplexity']:.2f} | {r['median']:.2f} | {r['unfinished']}/20 | {r['last_ten_change_fraction']:.1%} |")
lines += ['', '## Paired comparisons', '', 'Ratio below one favours DDIM-128 on perplexity only. Intervals are exploratory paired-seed 95% bootstrap intervals, not adjusted for multiple comparisons.']
for name,r in comparisons.items():
    lines.append(f"- {name}: ratio {r['ratio']:.3f}, interval {r['interval'][0]:.3f} to {r['interval'][1]:.3f}; lower in {r['lower_pairs']}/20, identical text in {r['identical_pairs']}/20.")
lines += ['', '## Fixed examples (seed 82)', '']
for name,rows in groups.items():
    lines += [f'### {name}', rows[0]['text'], '']
lines += ['Lower perplexity and greater stability do not establish grammar or meaning. Formal fluent-speaker review remains pending. No untouched test evaluation or independent training replicate was performed. EOS can be forced at a generation limit, so absence of cap-without-EOS does not prove natural completion.']
(root/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(root/'comparison.json').write_text(json.dumps(dict(results=results,comparisons=comparisons),indent=2),encoding='utf8')
print('\n'.join(lines))
