"""Time-budgeted, randomly initialized causal Transformer baseline.

No diffusion or autoencoder weights are loaded. Test data are never evaluated here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def pack_texts(tokenizer, texts, length):
    # Use the same truncated mT5 sequence (including EOS), plus a causal BOS.
    encoded = tokenizer(texts, truncation=True, max_length=length,
                        padding='max_length', return_tensors='pt')
    bos = torch.full((len(texts), 1), tokenizer.pad_token_id, dtype=torch.long)
    ids = torch.cat([bos, encoded['input_ids']], dim=1)
    mask = torch.cat([torch.ones_like(bos), encoded['attention_mask']], dim=1)
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, 0] = -100
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def read_texts(path):
    return [json.loads(line)['text'] for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def make_model(tokenizer, length, width=256, layers=4):
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=length + 1, n_ctx=length + 1,
        n_embd=width, n_layer=layers, n_head=width // 64,
        bos_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, resid_pdrop=0.1, embd_pdrop=0.1,
        attn_pdrop=0.1, use_cache=False))


@torch.no_grad()
def validate(model, packed, device, amp):
    model.eval()
    total, count = 0.0, 0
    for i in range(len(packed['input_ids'])):
        batch = {k: v[i:i+1].to(device) for k, v in packed.items()}
        n = int((batch['labels'][:, 1:] != -100).sum())
        with amp():
            loss = model(**batch).loss
        total += float(loss) * n
        count += n
    model.train()
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--match-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path)
    p.add_argument('--budget-seconds', type=float, default=16266)
    p.add_argument('--max-steps', type=int, default=1000000000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--validation-examples', type=int, default=128)
    p.add_argument('--sample-count', type=int, default=100)
    p.add_argument('--sample-max-tokens', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if min(args.budget_seconds, args.max_steps, args.save_every,
           args.validation_examples, args.sample_count, args.sample_max_tokens) <= 0:
        p.error('Budgets, counts and intervals must be positive')
    source = json.loads((args.match_run / 'args.json').read_text())
    if source.get('dataset_name') not in ('nchlt_afrikaans', 'nchlt-afrikaans'):
        raise ValueError('Expected an NCHLT Afrikaans diffusion run')
    if not source.get('using_latent_model'):
        raise ValueError('Source args do not confirm compressed latent diffusion')
    data = args.dataset_dir or Path(source.get('dataset_dir') or 'data/nchlt_afrikaans/processed/stage_5_7')
    length = int(source['max_seq_len'])
    effective = int(source.get('global_train_batch_size') or
                    source['train_batch_size'] * source['gradient_accumulation_steps'])
    if effective <= 0:
        raise ValueError('Invalid source batch size')
    tokenizer = AutoTokenizer.from_pretrained(source['enc_dec_model'], local_files_only=True)
    if args.sample_max_tokens > length:
        raise ValueError('Sample cap cannot exceed training sequence length')
    protocol = {
        'source_run': str(args.match_run.resolve()), 'source_args_sha256': digest(args.match_run / 'args.json'),
        'dataset_hashes': {s: digest(data / (s + '.jsonl')) for s in ('train', 'valid', 'test')},
        'tokenizer': source['enc_dec_model'],
        'tokenizer_vocab_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        'max_seq_len': length, 'causal_bos_extra_position': True,
        'microbatch': 1, 'gradient_accumulation': effective, 'effective_batch': effective,
        'initialization': 'random, tied input/output embeddings; no pretrained model weights',
        'width': 64 if args.smoke else 256, 'layers': 1 if args.smoke else 4,
        'budget_seconds': args.budget_seconds, 'seed': args.seed, 'smoke': args.smoke,
        'validation_examples': args.validation_examples,
        'sample_count': args.sample_count, 'sample_max_tokens': args.sample_max_tokens,
        'sampling': 'unconditional top-p=0.95, temperature=1; not identical to diffusion beam decoding',
        'budget_scope': 'training, periodic validation and checkpointing after tokenization; excludes final export/sampling',
        'comparison_limits': 'Diffusion job 57231 total elapsed 16266s is a provisional wall-time allowance, not matched FLOPs. Verify same GPU and actual diffusion training time. Autoencoder and pretraining costs are not included. Parameter counts differ.',
    }
    if args.check_only:
        print(json.dumps(protocol, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not train on the login node')
    out = args.output_dir
    if args.resume:
        previous = json.loads((out / 'protocol.json').read_text())
        for key in protocol:
            if previous[key] != protocol[key]:
                raise ValueError('Resume protocol changed: ' + key)
    elif out.exists():
        raise FileExistsError(f'{out} exists; use a new directory or --resume')
    else:
        out.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    amp = lambda: torch.autocast('cuda', dtype=precision, enabled=precision != torch.float32)
    train_texts = read_texts(data / 'train.jsonl')
    if args.smoke:
        train_texts = train_texts[:64]
    valid_texts = read_texts(data / 'valid.jsonl')
    random.Random(args.seed).shuffle(valid_texts)
    valid_texts = valid_texts[:args.validation_examples]
    train = pack_texts(tokenizer, train_texts, length)
    valid = pack_texts(tokenizer, valid_texts, length)
    model = make_model(tokenizer, length, protocol['width'], protocol['layers']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    step, elapsed, best = 0, 0.0, float('inf')
    protocol.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    parameters=sum(x.numel() for x in model.parameters()), precision=str(precision))
    if args.resume:
        ckpt = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, elapsed, best = ckpt['step'], ckpt['elapsed'], ckpt['best']
        torch.set_rng_state(ckpt['rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ckpt['cuda_rng']])
    write_json(out / 'protocol.json', protocol)
    print(json.dumps(protocol, indent=2), flush=True)
    print('Tokenization complete; timed training starts.', flush=True)
    start = time.monotonic()
    base_elapsed = elapsed
    model.train()

    def save():
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       step=step, elapsed=base_elapsed + time.monotonic() - start,
                       best=best, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(payload, out / 'last.tmp')
        os.replace(out / 'last.tmp', out / 'last.pt')

    try:
        while step < args.max_steps and base_elapsed + time.monotonic() - start < args.budget_seconds:
            optimizer.zero_grad(set_to_none=True)
            lr = 0.0002 * min((step + 1) / 100, 1.0)
            for group in optimizer.param_groups:
                group['lr'] = lr
            loss_sum = 0.0
            for _ in range(effective):
                index = torch.randint(len(train_texts), (1,))
                batch = {k: v[index].to(device) for k, v in train.items()}
                with amp():
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss; last completed checkpoint preserved')
                (loss / effective).backward()
                loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = dict(step=step, train_loss=loss_sum / effective, grad_norm=float(norm),
                          elapsed=base_elapsed + time.monotonic() - start, learning_rate=lr)
            if step % args.save_every == 0 or step == args.max_steps:
                ce = validate(model, valid, device, amp)
                if not math.isfinite(ce):
                    raise RuntimeError('Non-finite validation loss')
                record['validation_cross_entropy'] = ce
                if ce < best:
                    best = ce
                    torch.save(model.state_dict(), out / 'best.tmp')
                    os.replace(out / 'best.tmp', out / 'best.pt')
                    write_json(out / 'best_selection.json', dict(step=step, validation_cross_entropy=ce))
                save()
            with (out / 'training_metrics.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        save()
        print('Interrupted; completed updates saved. Resume with identical protocol.', flush=True)
        return
    save()
    timed_phase_seconds = base_elapsed + time.monotonic() - start
    # Final candidate competes on validation only; no test-based selection.
    ce = validate(model, valid, device, amp)
    if not math.isfinite(ce):
        raise RuntimeError('Non-finite final validation loss')
    if ce < best:
        best = ce
        torch.save(model.state_dict(), out / 'best.tmp')
        os.replace(out / 'best.tmp', out / 'best.pt')
        write_json(out / 'best_selection.json', dict(step=step, validation_cross_entropy=ce))
    save()
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
    model.save_pretrained(out / 'best_model', safe_serialization=True)
    tokenizer.save_pretrained(out / 'best_model')
    model.eval()
    torch.manual_seed(args.seed + 10000)
    samples = []
    with torch.no_grad():
        for _ in range(args.sample_count):
            ids = torch.tensor([[tokenizer.pad_token_id]], device=device)
            with amp():
                tokens = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                        top_k=0, top_p=0.95, temperature=1.0,
                                        max_new_tokens=args.sample_max_tokens, use_cache=True)
            samples.append(tokenizer.decode(tokens[0, 1:], skip_special_tokens=True).replace('\n', ' '))
    (out / 'samples.txt').write_text('\n'.join(samples) + '\n', encoding='utf8')
    write_json(out / 'summary.json', dict(status='completed', step=step,
               elapsed_budget_seconds=timed_phase_seconds,
               total_loop_and_final_evaluation_seconds=base_elapsed + time.monotonic() - start,
               best_validation_cross_entropy=best, test_evaluated=False,
               stop_reason='step_limit' if step >= args.max_steps else 'time_budget',
               quality_claim='smoke test only' if args.smoke else 'await shared external evaluation'))
    print('Baseline complete; no test evaluation or comparison claim made.', flush=True)


if __name__ == '__main__':
    main()
