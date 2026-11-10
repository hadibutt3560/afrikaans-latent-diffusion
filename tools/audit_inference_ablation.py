"""Read-only integrity, stability and paired uncertainty audit of saved experiments."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.inference_trace_utils import annotate, word_tokens


def weighted(rows):
    scores = [r['mgpt_score'] for r in rows]
    return math.exp(sum(s['nll'] for s in scores) / sum(s['tokens'] for s in scores))


def interval(candidate, control):
    rng = random.Random(20260926)
    ratios = []
    for _ in range(10000):
        indices = rng.choices(range(len(control)), k=len(control))
        ratios.append(weighted([candidate[i] for i in indices]) / weighted([control[i] for i in indices]))
    ratios.sort()
    return [ratios[249], ratios[9749]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train', type=Path, default=Path('data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    normalize = lambda text: ' '.join(text.casefold().split())
    train = {normalize(json.loads(line)['text']) for line in a.train.read_text(encoding='utf8').splitlines() if line.strip()}
    score_report = json.loads((a.scores / 'summary.json').read_text())
    groups, audit, metadata = {}, {}, {}
    review = []
    common_hashes = None
    for budget in (50, 250):
        for arm in ('control', 'ddim', 'projection', 'length128'):
            name = f'{arm}-{budget}'
            folder = a.traces / name
            m = json.loads((folder / 'metadata.json').read_text())
            metadata[name] = m
            pair = (m['diffusion_sha256'], m['autoencoder_sha256'])
            if common_hashes is None:
                common_hashes = pair
            assert pair == common_hashes
            assert m['status'] == 'completed' and m['inference_steps'] == budget
            assert m['sampler'] == ('ddim' if arm == 'ddim' else 'ddpm')
            assert m['project_decoder_latents'] == (arm == 'projection')
            assert m['decoding']['max_length'] == (128 if arm == 'length128' else 64)
            assert m['seeds'] == list(range(62, 82)) and m['observe_every'] == 1
            source = folder / 'trajectories.jsonl'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == score_report['source_hashes'][name]
            rows = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines()]
            assert {(r['seed'], r['step']) for r in rows} == {(s,t) for s in range(62,82) for t in range(1,budget+1)}
            assert len(rows) == 20 * budget
            assert all(math.isfinite(r['raw_latent_mean_norm']) for r in rows)
            changed = revisits = tail_changed = 0
            for seed in range(62, 82):
                trajectory = sorted((r for r in rows if r['seed'] == seed), key=lambda r:r['step'])
                verified = annotate(trajectory)
                for original, recalculated in zip(trajectory, verified):
                    for key in ('changed_from_previous', 'word_edits_from_previous', 'revisits_earlier_text'):
                        assert original[key] == recalculated[key], (name, seed, key)
                changed += sum(bool(r['changed_from_previous']) for r in trajectory[1:])
                revisits += sum(r['revisits_earlier_text'] for r in trajectory)
                tail_changed += sum(bool(r['changed_from_previous']) for r in trajectory[-10:])
            final = sorted((r for r in rows if r['step'] == budget), key=lambda r:r['seed'])
            scored = json.loads((a.scores / f'{name}.json').read_text(encoding='utf8'))
            assert [(r['seed'],r['text']) for r in final] == [(r['seed'],r['text']) for r in scored]
            assert all(r['mgpt_score'] and math.isfinite(r['mgpt_score']['nll']) for r in scored)
            groups[name] = scored
            audit[name] = dict(changed_fraction=changed/(20*(budget-1)), revisits=revisits,
                last_ten_changed_fraction=tail_changed/200,
                exact_train_matches=sum(normalize(r['text']) in train for r in final),
                capped_without_eos=sum(r['reached_length_cap'] and not r['ended_with_eos'] for r in final),
                empty=sum(not r['text'].strip() for r in final),
                unique=len({r['text'] for r in final}),
                mean_words=statistics.mean(len(word_tokens(r['text'])) for r in final),
                mean_raw_norm=statistics.mean(r['raw_latent_mean_norm'] for r in final))
            for r in final:
                review.append(dict(arm=name, seed=r['seed'], text=r['text']))
    for name, candidate in groups.items():
        control = groups['control-' + name.rsplit('-',1)[1]]
        a_stats = audit[name]
        a_stats['perplexity_ratio'] = weighted(candidate)/weighted(control)
        a_stats['paired_bootstrap_95_interval'] = interval(candidate, control)
        a_stats['identical_to_control'] = sum(x['text'] == y['text'] for x,y in zip(candidate,control))
        a_stats['lower_perplexity_pairs'] = sum(x['mgpt_score']['perplexity'] < y['mgpt_score']['perplexity'] for x,y in zip(candidate,control))
    a.output.mkdir(parents=True)
    (a.output/'audit.json').write_text(json.dumps(dict(checkpoint_hashes=common_hashes, arms=audit),indent=2),encoding='utf8')
    random.Random(20260926).shuffle(review)
    with (a.output/'blind_review.csv').open('w',newline='',encoding='utf-8-sig') as f, (a.output/'review_key.csv').open('w',newline='',encoding='utf-8-sig') as key:
        writer=csv.writer(f); keys=csv.writer(key)
        writer.writerow(['id','text','grammar_1_to_5','coherence_1_to_5','notes'])
        keys.writerow(['id','arm','seed'])
        for i,r in enumerate(review,1):
            writer.writerow([i,r['text'],'','','']); keys.writerow([i,r['arm'],r['seed']])
    lines=['# Inference experiment audit', '', 'All 24,000 rows passed completeness, source-score hash, metadata, finite latent norm, and recomputed text-change checks. All arms use the same recorded checkpoint hashes.', '',
        '| Arm | PPL ratio to control | Paired bootstrap 95% interval | Same text as control | Last 10 steps changed | Exact training matches |',
        '|---|---:|---|---:|---:|---:|']
    for name,r in audit.items():
        lo,hi=r['paired_bootstrap_95_interval']
        lines.append(f"| {name} | {r['perplexity_ratio']:.3f} | {lo:.3f}–{hi:.3f} | {r['identical_to_control']}/20 | {r['last_ten_changed_fraction']:.1%} | {r['exact_train_matches']}/20 |")
    lines += ['', 'Ratios below 1 favour the candidate on token-weighted mGPT perplexity only. Intervals resample matched seeds 10,000 times; they are exploratory, not multiplicity-adjusted, and do not represent training-seed uncertainty. The same final model generates all passages. Absence of exact normalized training matches does not rule out partial copying or memorization.', '',
        'Human review remains pending: blind_review.csv contains all 160 shuffled passages with blank grammar/coherence ratings. Keep review_key.csv hidden from the reviewer. No automatic ratings are passed off as native-speaker judgments. End-of-sequence is not proof of grammatical completion.', '',
        'All-step text stability was checked, but this audit uses existing final-output mGPT scores, not new perplexity scoring of all intermediate cluster texts. No model or result was overwritten.']
    (a.output/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    main()
