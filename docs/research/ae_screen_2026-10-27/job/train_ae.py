"""Train one language autoencoder arm with the published full-mT5 recipe (PROTOCOL.md section 4).

All of mT5 plus a fresh Perceiver compression/reconstruction module is trained with teacher-forced
cross-entropy. Selection and early stopping use token-weighted CE on the fixed selection panel.
Writes <output>/best/{model.pt,args.json} in the format tools/longrun_training.load_ae expects,
plus latest.pt for resume, metrics.jsonl and status.json.
"""
import argparse, hashlib, json, math, os, random, signal, sys, time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BUNDLE_ROOT', HERE.parent))
sys.path.insert(0, str(ROOT))
import torch

STOP = False


def handle_stop(*_):
    global STOP
    STOP = True


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(path, state):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def model_args(a):
    """Fields read by latent_models.latent_utils.get_latent_model and tools.longrun_training.load_ae."""
    return dict(enc_dec_model=str(a.model_cache), num_encoder_latents=a.latents, num_decoder_latents=a.latents,
                dim_ae=64, num_layers=3, l2_normalize_latents=True, max_seq_len=128, lm_mode='ft',
                recipe='full mT5 fine-tune (Lovelace et al. 2023, Table 13), 3 Perceiver layers as in job 56531')


def build_model(a):
    from latent_models.latent_utils import get_latent_model
    args = SimpleNamespace(**model_args(a))
    # lm_mode='ft' already makes everything trainable; the mask is set explicitly so smoke mode can narrow it.
    from latent_models.perceiver_ae import PerceiverAutoEncoder
    ae, tokenizer, _ = get_latent_model(args)
    # The Perceiver is absent from the mT5 checkpoint. Transformers 5.x leaves such modules
    # uninitialised (values up to 1e33 were observed), while 4.38 keeps PyTorch defaults. Rebuild it
    # explicitly with a seeded PyTorch-default initialisation so both versions start identically.
    torch.manual_seed(a.seed)
    fresh = PerceiverAutoEncoder(dim_lm=ae.config.d_model, num_encoder_latents=a.latents, num_decoder_latents=a.latents,
                                 dim_ae=64, depth=3, transformer_decoder=True, l2_normalize_latents=True, max_seq_len=192)
    ae.perceiver_ae.load_state_dict(fresh.state_dict(), strict=True)
    del fresh
    bad = [n for n, p in ae.named_parameters() if not torch.isfinite(p).all() or float(p.detach().abs().max()) > 1e3]
    if bad:
        raise ValueError(f'Uninitialised or extreme weights after loading: {bad[:5]}')
    for p in ae.parameters():
        p.requires_grad_(not a.smoke)
    if a.smoke:
        # Local 12 GB test only: train the Perceiver and last decoder block so the loop fits in memory.
        for name, p in ae.named_parameters():
            if 'perceiver' in name or name.startswith('decoder.block.11') or name.startswith('decoder.final_layer_norm'):
                p.requires_grad_(True)
    return ae.cuda(), tokenizer


def tokenize(tokenizer, rows):
    ids = tokenizer([r['text'] for r in rows], truncation=True, max_length=128)['input_ids']
    return [dict(id=r['id'], ids=x) for r, x in zip(rows, ids)]


def collate(items, pad_id):
    width = max(len(x['ids']) for x in items)
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, x in enumerate(items):
        input_ids[i, :len(x['ids'])] = torch.tensor(x['ids'])
        mask[i, :len(x['ids'])] = 1
    labels = input_ids.masked_fill(mask == 0, -100)
    return input_ids.cuda(), mask.cuda(), labels.cuda()


def forward_loss(ae, input_ids, mask, labels):
    encoder_outputs = ae.get_encoder()(input_ids=input_ids, attention_mask=mask)
    ae.encoder_output_to_decoder_input(encoder_outputs, mask)
    return ae(labels=labels, encoder_outputs=encoder_outputs)


@torch.no_grad()
def evaluate_ce(ae, items, pad_id, batch=8):
    was_training = ae.training
    ae.eval()
    nll, tokens, correct = 0.0, 0, 0
    for i in range(0, len(items), batch):
        input_ids, mask, labels = collate(items[i:i + batch], pad_id)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = forward_loss(ae, input_ids, mask, labels)
        # Sum in FP32 over the full vocabulary without materialising masked copies (250k-entry logits).
        logits = out.logits
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation loss')
        valid = labels.ne(-100)
        nll += float(loss)
        tokens += int(valid.sum())
        correct += int((logits.argmax(-1).eq(labels) & valid).sum())
        del out, logits, loss
    ae.train(was_training)
    return dict(ce=nll / tokens, accuracy=correct / tokens, tokens=tokens)


def lr_at(update, a):
    if update < a.warmup:
        return a.lr * (update + 1) / a.warmup
    return a.lr * max(0.0, (a.max_updates - update) / max(1, a.max_updates - a.warmup))


def epoch_order(n, seed, epoch):
    order = list(range(n))
    random.Random(seed * 1000003 + epoch).shuffle(order)
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    p.add_argument('--latents', type=int, choices=(32, 64), required=True)
    p.add_argument('--extra', action='store_true', help='add data/extra_afrikaans extra_train.jsonl to training')
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/nchlt_afrikaans/processed/stage_5_7')
    p.add_argument('--extra-data', type=Path, default=ROOT / 'data/extra_afrikaans/processed')
    p.add_argument('--panels', type=Path, default=ROOT / 'panels')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-updates', type=int, default=10000)
    p.add_argument('--micro-batch', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--min-updates', type=int, default=2000)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--relative-delta', type=float, default=0.005)
    p.add_argument('--max-seconds', type=float, default=6 * 3600)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--preflight', type=int, default=0, help='run this many updates, write preflight.json, save nothing else')
    p.add_argument('--smoke', action='store_true', help='local test: tiny trainable subset')
    a = p.parse_args()
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    ae, tokenizer = build_model(a)
    pad_id = tokenizer.pad_token_id
    rows = read_jsonl(a.data / 'train.jsonl')
    sources = {'nchlt_train': len(rows)}
    if a.extra:
        extra = read_jsonl(a.extra_data / 'extra_train.jsonl')
        rows += extra
        sources['extra_train'] = len(extra)
    train = tokenize(tokenizer, rows)
    selection = tokenize(tokenizer, read_jsonl(a.panels / 'selection.jsonl'))
    probe_rows = random.Random(a.seed).sample(read_jsonl(a.data / 'train.jsonl'), 1024)
    train_probe = tokenize(tokenizer, probe_rows)
    monitor = tokenize(tokenizer, read_jsonl(a.extra_data / 'extra_monitor.jsonl')) if (a.extra_data / 'extra_monitor.jsonl').exists() else []

    trainable = [p_ for p_ in ae.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)
    contract = dict(arm=a.arm, model=model_args(a), sources=sources,
                    hyper=dict(max_updates=a.max_updates, micro_batch=a.micro_batch, accumulation=a.accumulation, lr=a.lr,
                               warmup=a.warmup, weight_decay=a.weight_decay, eval_every=a.eval_every, min_updates=a.min_updates,
                               patience=a.patience, relative_delta=a.relative_delta, seed=a.seed, precision='bf16 autocast, fp32 weights',
                               dropout='mT5 config default (0.1) active in training; off in evaluation', smoke=a.smoke),
                    trainable_parameters=sum(x.numel() for x in trainable), total_parameters=sum(x.numel() for x in ae.parameters()),
                    data_sha256={'train.jsonl': digest(a.data / 'train.jsonl'), 'selection.jsonl': digest(a.panels / 'selection.jsonl'),
                                 **({'extra_train.jsonl': digest(a.extra_data / 'extra_train.jsonl')} if a.extra else {})})

    state = dict(update=0, epoch=0, position=0, best_ce=math.inf, best_update=None, stale=0, history=[], seconds=0.0)
    latest = a.output / 'latest.pt'
    if latest.exists() and not a.preflight:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Resume contract differs; use a new output directory')
        ae.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
        torch.cuda.empty_cache()
        print(json.dumps(dict(event='resumed', update=state['update'])), flush=True)
    else:
        write_json(a.output / 'contract.json', contract)

    def save_latest():
        atomic_save(latest, dict(model=ae.state_dict(), optimizer=optimizer.state_dict(), state=state, contract=contract,
                                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

    def save_best():
        best = a.output / 'best'
        best.mkdir(exist_ok=True)
        atomic_save(best / 'model.pt', dict(model={k: v.detach().cpu() for k, v in ae.state_dict().items()}, update=state['update']))
        write_json(best / 'args.json', model_args(a))

    def log(row):
        with (a.output / 'metrics.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(row) + '\n')

    ae.train()
    order = epoch_order(len(train), a.seed, state['epoch'])
    step_times, reason = [], None
    torch.cuda.reset_peak_memory_stats()
    while True:
        if STOP:
            reason = 'signal'
            break
        if state['update'] >= a.max_updates:
            reason = 'max_updates'
            break
        elapsed = state['seconds'] + (time.monotonic() - started)
        if not a.preflight and elapsed >= a.max_seconds:
            reason = 'time_cap'
            break
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total, examples = 0.0, 0
        batch_tokens = []
        for _ in range(a.accumulation):
            if state['position'] + a.micro_batch > len(order):
                state['epoch'] += 1
                state['position'] = 0
                order = epoch_order(len(train), a.seed, state['epoch'])
            items = [train[i] for i in order[state['position']:state['position'] + a.micro_batch]]
            state['position'] += a.micro_batch
            input_ids, mask, labels = collate(items, pad_id)
            batch_tokens.append((input_ids, mask, labels))
        token_total = sum(int(l.ne(-100).sum()) for _, _, l in batch_tokens)
        for input_ids, mask, labels in batch_tokens:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = forward_loss(ae, input_ids, mask, labels)
            # Token-weighted across the accumulated batch, as in the earlier autoencoder engines.
            n = int(labels.ne(-100).sum())
            loss = out.loss.float() * n / token_total
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at update {state['update']}")
            loss.backward()
            total += float(loss.detach())
            examples += labels.size(0)
        for group in optimizer.param_groups:
            group['lr'] = lr_at(state['update'], a)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at update {state['update']}")
        optimizer.step()
        state['update'] += 1
        torch.cuda.synchronize()
        step_times.append(time.monotonic() - tick)
        if state['update'] % 50 == 0 or a.preflight:
            log(dict(update=state['update'], train_ce=total, grad_norm=grad_norm, lr=lr_at(state['update'] - 1, a),
                     epoch=state['epoch'] + state['position'] / len(train), step_seconds=step_times[-1]))
        if a.preflight and state['update'] >= a.preflight:
            reason = 'preflight'
            break
        if state['update'] % a.eval_every == 0:
            sel = evaluate_ce(ae, selection, pad_id)
            probe = evaluate_ce(ae, train_probe, pad_id)
            mon = evaluate_ce(ae, monitor, pad_id) if monitor else None
            improved = sel['ce'] < state['best_ce'] * (1 - a.relative_delta)
            if sel['ce'] < state['best_ce']:
                state['best_ce'], state['best_update'] = sel['ce'], state['update']
                save_best()
            state['stale'] = 0 if improved else state['stale'] + 1
            row = dict(update=state['update'], selection=sel, train_probe=probe, extra_monitor=mon, best_ce=state['best_ce'],
                       best_update=state['best_update'], stale=state['stale'], peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            state['history'].append(row)
            log(dict(event='eval', **row))
            print(json.dumps(dict(arm=a.arm, update=state['update'], selection_ce=round(sel['ce'], 4),
                                  train_probe_ce=round(probe['ce'], 4), best_update=state['best_update'])), flush=True)
            state['seconds'] += time.monotonic() - started
            started = time.monotonic()
            save_latest()
            if state['update'] >= a.min_updates and state['stale'] >= a.patience:
                reason = 'early_stopped'
                break

    if a.preflight:
        if reason == 'signal':
            return 75
        times = step_times[2:] or step_times
        write_json(a.output / 'preflight.json', dict(arm=a.arm, updates=len(step_times), median_step_seconds=sorted(times)[len(times) // 2],
                   peak_gib=torch.cuda.max_memory_allocated() / 2**30, examples_per_update=a.micro_batch * a.accumulation,
                   projected_hours_max_updates=sorted(times)[len(times) // 2] * a.max_updates / 3600,
                   final_train_ce=total))
        return 0
    state['seconds'] += time.monotonic() - started
    save_latest()
    status = dict(arm=a.arm, reason=reason, updates=state['update'], best_update=state['best_update'], best_selection_ce=state['best_ce'],
                  hours=state['seconds'] / 3600, best_sha256=digest(a.output / 'best/model.pt') if (a.output / 'best/model.pt').exists() else None,
                  complete=reason in ('early_stopped', 'max_updates', 'time_cap'))
    write_json(a.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['complete'] else 75


if __name__ == '__main__':
    raise SystemExit(main())
