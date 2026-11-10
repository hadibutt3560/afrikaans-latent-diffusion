"""Summarize full trajectories and score them with a frozen mGPT reference model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inference_trace_utils import annotate


def prepare_scorer():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info('ai-forever/mGPT')
    files = {f.rfilename for f in info.siblings}
    weights = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    if weights not in files:
        raise RuntimeError('Expected standalone official mGPT weight file not found')
    path = snapshot_download('ai-forever/mGPT', revision=info.sha,
        allow_patterns=[weights, 'config.json', 'generation_config.json', 'tokenizer*',
                        'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json'])
    print(json.dumps({'scorer_snapshot': path, 'revision': info.sha}), flush=True)


def summarize_seed(rows):
    rows = annotate(sorted(rows, key=lambda r: r['step']))
    run, longest = 0, 0
    for row in rows[1:]:
        run = 0 if row['changed_from_previous'] else run + 1
        longest = max(longest, run)
    count = len(rows) - 1
    return dict(observed_points=len(rows), transitions=count,
        changed=sum(bool(x['changed_from_previous']) for x in rows[1:]),
        revisits=sum(x['revisits_earlier_text'] for x in rows),
        longest_unchanged_transition_run=longest, final_unchanged_transition_run=run,
        mean_word_edits=statistics.mean(x['word_edits_from_previous'] for x in rows[1:]) if count else None,
        capped_outputs=sum(x['reached_length_cap'] for x in rows))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-scorer', action='store_true')
    p.add_argument('--trace-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--reference-count', type=int, default=32)
    p.add_argument('--scorer-snapshot', type=Path)
    p.add_argument('--stability-only', action='store_true')
    p.add_argument('--validation', type=Path, default=ROOT / 'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl')
    a = p.parse_args()
    if a.prepare_scorer:
        prepare_scorer()
        return
    if not a.trace_dir or not a.output_dir or a.reference_count < 1:
        p.error('Require trace/output directories and a positive reference count')
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    metadata = json.loads((a.trace_dir / 'metadata.json').read_text())
    if metadata['status'] != 'completed' or metadata['observe_every'] != 1:
        raise ValueError('Require completed every-step traces')
    rows = [json.loads(x) for x in (a.trace_dir / 'trajectories.jsonl').read_text(encoding='utf8').splitlines()]
    seeds = sorted({r['seed'] for r in rows})
    for seed in seeds:
        if sorted(r['step'] for r in rows if r['seed'] == seed) != list(range(1, metadata['inference_steps'] + 1)):
            raise ValueError('Missing or duplicate steps')
    a.output_dir.mkdir(parents=True)
    report = dict(status='scoring', source_trace=str(a.trace_dir),
        trace_sha256=hashlib.sha256((a.trace_dir / 'trajectories.jsonl').read_bytes()).hexdigest(),
        stability={str(seed): summarize_seed([r for r in rows if r['seed'] == seed]) for seed in seeds},
        scorer='ai-forever/mGPT', precision='float32', context=512, stride=256,
        reference_split='validation', reference_count=a.reference_count,
        limitations=['Two trajectories are a diagnostic, not a general performance estimate.',
                    'mGPT perplexity is a predictability proxy, not a semantic correctness score.',
                    'Intermediate predicted clean text at step k is not a completed k-step sampling run.',
                    'Natural-versus-scrambled text is only a basic scorer sanity check.',
                    'No test texts, model updates or checkpoint selection based on these scores.'])
    write_json(a.output_dir / 'summary.json', report)
    if a.stability_only:
        report.update(status='stability_completed_quality_not_scored')
        write_json(a.output_dir / 'summary.json', report)
        lines = ['# Every-step stability pilot', '',
                 'Two passages, 250 DDPM inference steps each, existing 50,000-step checkpoint. No retraining.', '',
                 '| Seed | Changes / adjacent comparisons | Exact returns to earlier wording | Final unchanged transitions | Mean word edits |',
                 '|---|---|---|---|---|']
        for seed, r in report['stability'].items():
            lines.append(f"| {seed} | {r['changed']}/{r['transitions']} | {r['revisits']} | {r['final_unchanged_transition_run']} | {r['mean_word_edits']:.2f} |")
        lines += ['', 'These are passage-level changes, not counts of individual sentences. Exact revisits are finite observed returns, not proof of an infinite loop.',
                  'The sampler still terminates at step 250. Frequent changes alone do not establish poor quality: DDPM sampling is stochastic.',
                  'No external quality scores are included in this stability-only report.', '',
                  'See ../READ_RESULTS.md for all decoded texts, and ../changes_by_step.csv for counts at each step.']
        (a.output_dir / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf8')
        print(json.dumps(report, indent=2), flush=True)
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub.constants import HF_HUB_CACHE
    from tools.compare_afrikaans_generators import score_text
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the scoring pilot')
    snapshot = a.scorer_snapshot
    if snapshot is None:
        candidates = list((Path(HF_HUB_CACHE) / 'models--ai-forever--mGPT' / 'snapshots').glob('*'))
        if len(candidates) != 1:
            raise ValueError('Specify --scorer-snapshot: require one unambiguous cached mGPT revision')
        snapshot = candidates[0]
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
    report['resolved_revision'] = snapshot.name
    report['scorer_snapshot'] = str(snapshot)
    cache = {}
    start = time.monotonic()

    def score(text):
        if text not in cache:
            cache[text] = score_text(model, tokenizer, text, 'cuda')
        return cache[text]

    scored = []
    for index, row in enumerate(rows, 1):
        result = score(row['text'])
        scored.append(dict(row, mgpt_perplexity=None if result is None else result['perplexity'],
                           mgpt_score=result))
        with (a.output_dir / 'scored_trajectories.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(scored[-1], ensure_ascii=False, allow_nan=False) + '\n')
        if index % 50 == 0:
            print(f'Scored {index}/{len(rows)} intermediate outputs', flush=True)
    real = [json.loads(x) for x in a.validation.read_text(encoding='utf8').splitlines() if x.strip()]
    random.Random(42).shuffle(real)
    if len(real) < a.reference_count:
        raise ValueError('Not enough reference examples')
    controls = []
    for index, item in enumerate(real[:a.reference_count]):
        words = item['text'].split()
        random.Random(10000 + index).shuffle(words)
        corrupted = ' '.join(words)
        controls.append(dict(id=item.get('id'), text=item['text'], shuffled_text=corrupted,
                             real_score=score(item['text']), shuffled_score=score(corrupted)))
    write_json(a.output_dir / 'validation_controls.json', controls)
    report['calibration'] = dict(real_mean_perplexity=statistics.mean(x['real_score']['perplexity'] for x in controls),
        real_median_perplexity=statistics.median(x['real_score']['perplexity'] for x in controls),
        shuffled_mean_perplexity=statistics.mean(x['shuffled_score']['perplexity'] for x in controls),
        shuffled_median_perplexity=statistics.median(x['shuffled_score']['perplexity'] for x in controls),
        pairs_where_real_is_lower=sum(x['real_score']['perplexity'] < x['shuffled_score']['perplexity'] for x in controls),
        validation_sha256=hashlib.sha256(a.validation.read_bytes()).hexdigest())
    for seed in seeds:
        group = sorted((x for x in scored if x['seed'] == seed), key=lambda r: r['step'])
        valid = [x for x in group if x['mgpt_perplexity'] is not None]
        best = min(valid, key=lambda x: x['mgpt_perplexity'])
        report['stability'][str(seed)].update(first_perplexity=group[0]['mgpt_perplexity'],
            final_perplexity=group[-1]['mgpt_perplexity'], lowest_perplexity_step=best['step'],
            lowest_perplexity=best['mgpt_perplexity'], empty_outputs=len(group)-len(valid))
    with (a.output_dir / 'quality_by_step.csv').open('w', newline='', encoding='utf8') as f:
        fields = ['seed','step','changed_from_previous','word_edits_from_previous','revisits_earlier_text',
                  'decoder_tokens','reached_length_cap','mgpt_perplexity']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(scored)
    report.update(status='completed', scorer_seconds=time.monotonic()-start)
    write_json(a.output_dir / 'summary.json', report)
    lines = ['# Every-step inference analysis', '',
             'Existing step-50,000 diffusion checkpoint; 250 DDPM steps; two seeds. No retraining.', '',
             '| Seed | Changed transitions | Returns to earlier text | Final unchanged transitions | First / final mGPT perplexity |',
             '|---|---|---|---|---|']
    for seed, r in report['stability'].items():
        lines.append(f"| {seed} | {r['changed']}/{r['transitions']} | {r['revisits']} | {r['final_unchanged_transition_run']} | {r['first_perplexity']:.2f} / {r['final_perplexity']:.2f} |")
    c = report['calibration']
    lines += ['', f"Scorer sanity check: real validation mean perplexity {c['real_mean_perplexity']:.2f}; scrambled-word control {c['shuffled_mean_perplexity']:.2f}. Real text scored lower in {c['pairs_where_real_is_lower']}/{a.reference_count} pairs.",
              f"Real validation median perplexity: {c['real_median_perplexity']:.2f}. Means can be dominated by outliers; inspect the saved control texts before interpreting absolute scores.",
              '', '## Interpretation limits', *['- '+x for x in report['limitations']], '',
              'Do not select a stopping step solely by the minimum perplexity: generic or repetitive text can score well. Meaning still requires inspection.', '',
              '## Selected intermediate outputs']
    for seed in seeds:
        lines += ['', f'### Seed {seed}']
        for row in scored:
            if row['seed'] == seed and row['step'] in (1, 50, 100, 150, 200, 225, 250):
                lines += ['', f"**Step {row['step']} | mGPT perplexity {row['mgpt_perplexity']:.2f}**", row['text']]
    (a.output_dir / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
