"""Score both generators and real validation text with one frozen external model.

Reports exploratory validation diagnostics, not a final untouched-test conclusion.

Inputs: a baseline directory holding protocol.json and samples.txt (the
autoregressive samples, one per line), a diffusion samples file from the run
named in the protocol, and the validation split, whose hash must match the
protocol. Outputs: <group>-scored.json per group and comparison.json in a new
output directory. Example: python tools/compare_afrikaans_generators.py
--baseline-dir DIR --diffusion-samples FILE --output-dir NEW_DIR
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def diversity(texts):
    """Return sample counts, mean token count and distinct-n/repetition-n for n = 1-4."""
    # Tokens are word-character runs or single punctuation marks, so
    # 'mean_word_tokens' counts punctuation too.
    words = [re.findall(r'\w+|[^\w\s]', t, flags=re.UNICODE) for t in texts]
    result = {'samples': len(texts), 'empty_samples': sum(not t.strip() for t in texts),
              'unique_sentences': len(set(texts)),
              'mean_word_tokens': sum(map(len, words)) / len(words)}
    # Pooled over the whole group: distinct-n = unique n-grams / all n-grams,
    # and repetition-n = 1 - distinct-n. n-grams do not cross sample boundaries.
    for n in (1, 2, 3, 4):
        counts = Counter(tuple(w[i:i+n]) for w in words for i in range(len(w)-n+1))
        total = sum(counts.values())
        result[f'distinct_{n}'] = len(counts) / total if total else None
        result[f'repetition_{n}'] = 1 - len(counts) / total if total else None
    return result


def score_text(model, tokenizer, text, device, context=512):
    """Score text under the causal language model.

    Returns the summed negative log-likelihood (nats), the number of scored
    tokens and the perplexity, or None if the text produces no tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return None
    # A start token is prepended so the first real token is also predicted;
    # the start token itself is never scored. EOS is the fallback if there is no BOS.
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None:
        raise ValueError('External scorer must define BOS or EOS')
    ids = torch.tensor([[bos] + tokens], device=device)
    previous_end, nll, count = 1, 0.0, 0
    stride = context // 2
    # Each token is scored once; long texts are not silently truncated.
    # Windows of `context` tokens advance by half a window. Positions already
    # scored by the previous window (and always the window's first position)
    # get label -100 so the loss ignores them; tokens after the first window
    # are therefore scored with at least context - stride tokens of left context.
    for begin in range(0, ids.size(1), stride):
        end = min(begin + context, ids.size(1))
        batch = ids[:, begin:end]
        labels = batch.clone()
        labels[:, :max(1, previous_end-begin)] = -100
        # The model shifts labels by one internally, so count from position 1.
        n = int((labels[:, 1:] != -100).sum())
        if n:
            with torch.no_grad():
                loss = float(model(input_ids=batch, labels=labels).loss)
            if not math.isfinite(loss):
                raise RuntimeError('External scorer returned non-finite loss')
            # loss is the mean over the n scored tokens; convert back to a sum.
            nll += loss * n
            count += n
        previous_end = end
        if end == ids.size(1):
            break
    return {'nll': nll, 'tokens': count, 'perplexity': math.exp(nll / count)}


def main():
    """Check inputs against the baseline protocol, score all three groups and write the reports."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--diffusion-samples', type=Path, required=True)
    p.add_argument('--source-run-local', type=Path, help='Downloaded copy of the protocol source run; args hash must still match')
    p.add_argument('--validation', type=Path, default=Path('data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--scorer', default='ai-forever/mGPT')
    p.add_argument('--revision', default='main')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--allow-download', action='store_true')
    a = p.parse_args()
    if a.count <= 0:
        p.error('count must be positive')
    # Refuse to reuse an output directory, so earlier results are never overwritten.
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    # Provenance checks: the diffusion run's args.json, the location of the
    # samples file and the validation split must all match the baseline protocol.
    protocol = json.loads((a.baseline_dir / 'protocol.json').read_text())
    source_run = a.source_run_local or Path(protocol['source_run'])
    source_args = source_run / 'args.json'
    if hashlib.sha256(source_args.read_bytes()).hexdigest() != protocol['source_args_sha256']:
        raise ValueError('Diffusion args changed since baseline setup')
    if a.diffusion_samples.resolve().parent != source_run.resolve():
        raise ValueError('Samples must come from the matched diffusion run')
    if hashlib.sha256(a.validation.read_bytes()).hexdigest() != protocol['dataset_hashes']['valid']:
        raise ValueError('Validation split differs from baseline protocol')
    # Generated samples are taken in file order. Blank lines are kept: they
    # count as empty samples and are left out of the perplexity averages.
    texts = {}
    for name, path in [('autoregressive', a.baseline_dir / 'samples.txt'), ('diffusion', a.diffusion_samples)]:
        lines = path.read_text(encoding='utf8').splitlines()
        if len(lines) < a.count:
            raise ValueError(f'{name} has only {len(lines)} samples; require {a.count}')
        texts[name] = lines[:a.count]
    real = [json.loads(x)['text'] for x in a.validation.read_text(encoding='utf8').splitlines() if x.strip()]
    # A fixed seed (42) draws the same validation subset on every run.
    random.Random(42).shuffle(real)
    if len(real) < a.count:
        raise ValueError('Insufficient real validation texts')
    texts['real_validation'] = real[:a.count]
    a.output_dir.mkdir(parents=True)
    # Offline by default: the scorer must already be cached unless --allow-download is given.
    tokenizer = AutoTokenizer.from_pretrained(a.scorer, revision=a.revision, local_files_only=not a.allow_download)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(a.scorer, revision=a.revision,
        local_files_only=not a.allow_download, torch_dtype=torch.float32).to(device).eval()
    report = {'scorer': a.scorer, 'requested_revision': a.revision,
              'protocol_source_run': protocol['source_run'], 'local_source_run': str(source_run),
              'source_args_sha256': hashlib.sha256(source_args.read_bytes()).hexdigest(),
              # Commit hash of the loaded weights, if transformers recorded one.
              'resolved_revision': getattr(model.config, '_commit_hash', None),
              # Recorded, not passed in: these mirror score_text's defaults
              # (context=512, stride=context // 2).
              'precision': 'float32', 'context': 512, 'stride': 256,
              'reference_split': 'validation only', 'count_per_group': a.count,
              'source_hashes': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                               [a.diffusion_samples, a.baseline_dir / 'samples.txt', a.validation]},
              'limitations': ['Single seed; exploratory comparison, not final hypothesis test.',
                             'Different sampling algorithms; length distributions must be inspected.',
                             'Real validation text is untruncated; generated text has a length cap.',
                             'Wall-time allowance is provisional; hardware and pretraining differ.',
                             'Diversity uses shared Unicode regex tokenization, not legacy spaCy metrics.'],
              'groups': {}}
    for name, group in texts.items():
        rows = [score_text(model, tokenizer, text, device) for text in group]
        usable = [x for x in rows if x is not None]
        if not usable:
            raise ValueError(f'All {name} texts empty')
        stats = diversity(group)
        # Three aggregates: mean and median of per-sample perplexities, and a
        # token-weighted perplexity from the pooled NLL, which long samples
        # influence more.
        stats['mean_sentence_perplexity'] = sum(x['perplexity'] for x in usable) / len(usable)
        stats['median_sentence_perplexity'] = statistics.median(x['perplexity'] for x in usable)
        stats['token_weighted_perplexity'] = math.exp(sum(x['nll'] for x in usable) / sum(x['tokens'] for x in usable))
        stats['scored_samples'] = len(usable)
        report['groups'][name] = stats
        (a.output_dir / (name + '-scored.json')).write_text(json.dumps(
            [{'text': t, 'score': s} for t, s in zip(group, rows)], ensure_ascii=False, indent=2), encoding='utf8')
        print(name, stats, flush=True)
    ar, diffusion = report['groups']['autoregressive'], report['groups']['diffusion']
    # Diffusion mean perplexity divided by the autoregressive one, then checked
    # against a 1.2x bound. The check is None if either generator produced an
    # empty sample (empty samples are excluded from the perplexity means).
    report['exploratory_mean_perplexity_ratio'] = diffusion['mean_sentence_perplexity'] / ar['mean_sentence_perplexity']
    report['within_1_2_on_this_sample_only'] = (report['exploratory_mean_perplexity_ratio'] <= 1.2
        if not ar['empty_samples'] and not diffusion['empty_samples'] else None)
    # Always False: this script is an exploratory validation comparison (see the
    # limitations above), not the final hypothesis test.
    report['hypothesis_proven'] = False
    (a.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
