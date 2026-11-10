"""Score final outputs of the fixed eight-arm inference experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.compare_afrikaans_generators import score_text, diversity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    groups, hashes = {}, {}
    for steps in (50, 250):
        for arm in ('control', 'ddim', 'projection', 'length128'):
            name = f'{arm}-{steps}'
            folder = a.root / name
            m = json.loads((folder / 'metadata.json').read_text())
            assert m['status'] == 'completed' and m['inference_steps'] == steps
            assert m['seeds'] == list(range(62, 82))
            path = folder / 'trajectories.jsonl'
            rows = [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
            assert len(rows) == steps * 20
            assert len({(x['seed'], x['step']) for x in rows}) == len(rows)
            final = sorted((x for x in rows if x['step'] == steps), key=lambda x: x['seed'])
            assert len(final) == 20
            groups[name] = final
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.snapshot), local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
    a.output.mkdir(parents=True)
    cache, results = {}, {}
    for name, rows in groups.items():
        for row in rows:
            text = row['text']
            if text not in cache:
                cache[text] = score_text(model, tokenizer, text, 'cuda')
            row['mgpt_score'] = cache[text]
        scores = [r['mgpt_score'] for r in rows if r['mgpt_score'] is not None]
        results[name] = dict(diversity([r['text'] for r in rows]),
            token_weighted_perplexity=math.exp(sum(s['nll'] for s in scores)/sum(s['tokens'] for s in scores)),
            median_perplexity=statistics.median(s['perplexity'] for s in scores),
            capped_without_eos=sum(r['reached_length_cap'] and not r['ended_with_eos'] for r in rows),
            mean_raw_latent_norm=statistics.mean(r['raw_latent_mean_norm'] for r in rows))
        (a.output / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
        print(name, results[name], flush=True)
    for name, rows in groups.items():
        control = groups['control-' + name.rsplit('-', 1)[1]]
        results[name]['paired_lower_perplexity_than_control'] = sum(
            r['mgpt_score']['perplexity'] < c['mgpt_score']['perplexity']
            for r,c in zip(rows,control) if r['mgpt_score'] and c['mgpt_score'])
    report = dict(scorer='ai-forever/mGPT', revision=a.snapshot.name,
        precision='float32', context=512, stride=256, source_hashes=hashes, results=results,
        scope='Final outputs only, 20 fresh matched seeds, single training checkpoint; not untouched test evaluation')
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    lines = ['# Cluster inference experiment: final-output quality', '',
        '| Setting | Perplexity (token-weighted) | Median | Distinct-2 | Lower PPL than control | Capped without EOS |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['token_weighted_perplexity']:.2f} | {r['median_perplexity']:.2f} | {100*r['distinct_2']:.1f}% | {r['paired_lower_perplexity_than_control']}/20 | {r['capped_without_eos']}/20 |")
    lines += ['', 'Only final outputs are scored here. Lower perplexity does not establish grammatical or semantic quality. Length, generic phrasing, and repetition can affect scores. Compare each change against its same-budget control, not previous runs on different hardware. Small sample, no confidence intervals, no final hypothesis claim.', '', '## Fixed example: seed 62']
    for name,rows in groups.items():
        lines += ['', f'### {name}', rows[0]['text']]
    (a.output / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
