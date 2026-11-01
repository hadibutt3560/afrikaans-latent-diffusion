"""Overnight local follow-ups to the 1 November review (PROTOCOL.md in this folder).

Usage:
  python run_followups.py --root PATH_TO_DATA_REPO            # everything, resumable
  python run_followups.py --root PATH_TO_DATA_REPO --smoke    # a few rows per task, separate output folder
  python run_followups.py --root ... --only numerals,overlap   # chosen tasks

--root is the checkout that holds outputs/ and data/ (the original working folder).
Loading, decoding and scoring are imported unchanged from latent_controls_2026-10-30/run_controls.py.
Each task writes its own files and a .done marker; rerunning skips finished tasks and resumes decoding.
"""
import argparse, csv, importlib.util, json, math, random, re, sys, time, traceback, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NUM = re.compile(r'\d+(?:[.,]\d+)*')
WEIGHTS = np.array([0.140, 0.223, 0.349, 0.288])
MIN_BAND = 20
SHUFFLE_SEEDS = range(20261111, 20261116)
RANDOM_BANKS = {'C1-31000': ('C1', 31000), 'C1-32000': ('C1', 32000), 'C2-41000': ('C2', 41000), 'C2-42000': ('C2', 42000)}
EVAL_BANKS = ['H', 'R', 'C1', 'C2', 'C3', 'AR-hist'] + [f'{s}-seed{k}' for s in ('small', 'published') for k in range(42, 47)]

RC = OUT = LOG = None
SMOKE = False


def log(event, **fields):
    line = json.dumps(dict(time=time.strftime('%H:%M:%S'), event=event, **fields), ensure_ascii=False)
    print(line, flush=True)
    with LOG.open('a', encoding='utf8') as f: f.write(line + '\n')


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def norm_words(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split()).split()


def grams(words, n):
    return [hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)]


def longest_run(flags, n):
    """Longest shared word span implied by consecutive shared n-grams (0 if none)."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best + n - 1 if best else 0


def reserve_documents(valid):
    docs = sorted({r['document_id'] for r in valid}); random.Random(RC.SEED_VALID_DOCS).shuffle(docs)
    return set(docs[64:128])


def numeral_counts(ref, pred):
    a, b = Counter(NUM.findall(ref)), Counter(NUM.findall(pred))
    return dict(expected=sum(a.values()), produced=sum(b.values()), matched=sum((a & b).values()),
                invented=sum((b - a).values()), dropped=sum((a - b).values()))


def numeral_summary(pairs):
    rows = [dict(id=i, **numeral_counts(ref, pred)) for i, ref, pred in pairs]
    t = {k: sum(r[k] for r in rows) for k in ('expected', 'produced', 'matched', 'invented', 'dropped')}
    return dict(passages=len(rows), passages_with_numerals=sum(r['expected'] > 0 for r in rows), **t,
                recall=t['matched'] / t['expected'] if t['expected'] else None,
                precision=t['matched'] / t['produced'] if t['produced'] else None,
                passages_with_invented=[r for r in rows if r['invented']])


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def standardised(rows):
    """Raw and length-standardised perplexity from rows carrying mgpt nll/tokens."""
    scored = [r for r in rows if r.get('mgpt')]
    counts, logs = [], []
    for k in range(len(RC.BANDS)):
        rs = [r for r in scored if RC.band(len(r['text'].split())) == k]
        counts.append(len(rs))
        logs.append(sum(r['mgpt']['nll'] for r in rs) / sum(r['mgpt']['tokens'] for r in rs) if rs else float('nan'))
    raw = math.exp(sum(r['mgpt']['nll'] for r in scored) / sum(r['mgpt']['tokens'] for r in scored))
    std = math.exp(float(np.dot(WEIGHTS, logs))) if min(counts) >= MIN_BAND else None
    bigrams = [tuple(w[i:i + 2]) for r in rows for w in [r['text'].split()] for i in range(len(w) - 1)]
    return dict(n=len(rows), raw_ppl=raw, standardised_ppl=std, band_counts=counts,
                distinct2=len(set(bigrams)) / max(1, len(bigrams)), sentinel=sum(bool(r.get('sentinel')) for r in rows),
                empty=sum(not r['text'] for r in rows))


# ---------------------------------------------------------------- CPU tasks

def task_numerals():
    audit = RC.rows_of(RC.STAGE1 / 'examples.jsonl')
    controls = RC.ROOT / 'outputs/latent-controls-20261030/scored-R.jsonl'
    out = dict(audit_81=numeral_summary([(r['id'], r['model_view'], r['correct']['text']) for r in audit]))
    if controls.exists():
        out['controls_R_811'] = numeral_summary([(r['id'], r['model_view'], r['text']) for r in RC.rows_of(controls)])
    save('numerals.json', out)
    return {k: dict(recall=v['recall'], precision=v['precision'], invented=v['invented']) for k, v in out.items()}


def train_index(n):
    index, docs = defaultdict(set), {}
    for r in RC.rows_of(RC.TRAIN):
        for g in set(grams(norm_words(r['text']), n)): index[g].add(r['document_id'])
    return index


def task_overlap():
    index = train_index(12)
    valid = RC.rows_of(RC.VALID); reserve = reserve_documents(valid)
    rows = [r for r in valid if r['document_id'] not in reserve]
    affected, spans, shared_total, boiler = [], [], 0, 0
    for r in rows:
        g = grams(norm_words(r['text']), 12)
        flags = [x in index for x in g]
        if any(flags):
            affected.append(r['document_id']); spans.append(longest_run(flags, 12))
            for x, f in zip(g, flags):
                if f: shared_total += 1; boiler += len(index[x]) >= 5
    out = dict(validation_passages=len(rows), validation_documents=len({r['document_id'] for r in rows}),
               passages_sharing_12gram=len(affected), documents_affected=len(set(affected)),
               longest_span_words=dict(zip(('p50', 'p90', 'max'), np.percentile(spans, [50, 90, 100]).tolist())) if spans else None,
               shared_12grams=shared_total, share_in_5plus_training_documents=boiler / shared_total if shared_total else None,
               note='Test split and reserve documents not read.')
    save('overlap.json', out)
    return {k: out[k] for k in ('passages_sharing_12gram', 'documents_affected', 'share_in_5plus_training_documents')}


def task_lengths():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RC.BUNDLE / 'models/mt5-base', local_files_only=True)
    valid = RC.rows_of(RC.VALID); reserve = reserve_documents(valid)
    sets = dict(train=RC.rows_of(RC.TRAIN), validation_non_reserve=[r for r in valid if r['document_id'] not in reserve],
                extra_monitor=RC.rows_of(RC.ROOT / 'data/extra_afrikaans/processed/extra_monitor.jsonl'))
    out = {}
    for name, rows in sets.items():
        n = np.array([len(x) for x in tok([r['text'] for r in rows], truncation=False)['input_ids']])
        out[name] = dict(passages=len(n), percentiles=dict(zip(('p50', 'p90', 'p99', 'max'), np.percentile(n, [50, 90, 99, 100]).tolist())),
                         over_128=int((n > 128).sum()), share_over_128=float((n > 128).mean()))
    save('lengths.json', out)
    return {k: v['share_over_128'] for k, v in out.items()}


def task_figure1():
    data = RC.read(RC.ROOT / 'docs/research/stage4_final_evaluation_2026-10-23/length_bands.json')
    with (OUT / 'figure1.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.writer(f); w.writerow(['bank', 'band', 'n', 'ppl'])
        for bank, bands in data.items():
            for b, v in bands.items(): w.writerow([bank, b, v['n'], round(v['ppl'], 4)])
    return dict(banks=len(data))


def task_partial_copies():
    index = train_index(8)
    pkg = RC.ROOT / 'outputs/evaluation-package-20261101'
    out = {}
    for bank in EVAL_BANKS:
        path = pkg / f'bank-{bank}.jsonl'
        if not path.exists(): continue
        total = shared = near = 0; spans = []
        for r in RC.rows_of(path):
            g = grams(norm_words(r['text']), 8); flags = [x in index for x in g]
            total += len(g); shared += sum(flags); spans.append(longest_run(flags, 8))
            near += len(g) >= 4 and sum(flags) >= 0.9 * len(g)
        out[bank] = dict(rows=len(spans), eightgram_share_in_training=shared / total if total else None,
                         near_copy_rows=near, longest_span_p90=float(np.percentile(spans, 90)), longest_span_max=int(max(spans)))
    save('partial_copies.json', out)
    return {k: round(v['eightgram_share_in_training'] or 0, 4) for k, v in out.items()}


def task_scorers():
    pkg = RC.ROOT / 'outputs/evaluation-package-20261101'
    human = None; out = {}
    for bank in EVAL_BANKS:
        b, g = pkg / f'bank-{bank}.jsonl', pkg / f'goldfish-{bank}.jsonl'
        if not (b.exists() and g.exists()): continue
        rows = [dict(r, goldfish=s['goldfish']) for r, s in zip(RC.rows_of(b), RC.rows_of(g)) if r.get('mgpt') and s.get('goldfish')]
        m = np.array([r['mgpt']['nll'] / r['mgpt']['tokens'] for r in rows]); q = np.array([r['goldfish']['nll'] / r['goldfish']['tokens'] for r in rows])
        if bank == 'H': human = float(np.median(q))
        bands = {}
        for k, (lo, hi) in enumerate(RC.BANDS):
            rs = [r for r in rows if RC.band(len(r['text'].split())) == k]
            if rs: bands[f'{lo}-{hi if hi < 10**8 else "+"}'] = dict(n=len(rs),
                mgpt=math.exp(sum(r['mgpt']['nll'] for r in rs) / sum(r['mgpt']['tokens'] for r in rs)),
                goldfish=math.exp(sum(r['goldfish']['nll'] for r in rs) / sum(r['goldfish']['tokens'] for r in rs)))
        out[bank] = dict(rows=len(rows), spearman_loss_per_token=spearman(m, q), bands=bands, _goldfish_losses=q)
    for bank, v in out.items():
        q = v.pop('_goldfish_losses')
        v['share_better_than_median_human_goldfish'] = float((q < human).mean()) if human is not None else None
    save('scorers.json', out)
    return {k: round(v['spearman_loss_per_token'], 3) for k, v in out.items()}


# ---------------------------------------------------------------- GPU tasks

class Models:
    """B0 decoder and the decode/metric functions, set up once (same settings as run_controls.py)."""
    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from transformers.modeling_outputs import BaseModelOutput
        from sacrebleu.metrics import CHRF
        sys.path.insert(0, str(RC.BUNDLE))
        from tools.longrun_training import load_ae
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available(), 'CUDA is required for the GPU tasks'
        assert RC.digest(RC.AE / 'model.pt') == RC.AE_SHA
        self.torch, self.BMO = torch, BaseModelOutput
        self.tok = AutoTokenizer.from_pretrained(RC.BUNDLE / 'models/mt5-base', local_files_only=True)
        self.ae, _ = load_ae(RC.AE, RC.BUNDLE / 'models/mt5-base', dict(latent_dim=RC.DIM, latent_length=RC.SLOTS, max_length=RC.MAX_TOKENS))
        self.chrf = CHRF(char_order=6, word_order=0, beta=2)

    def decode(self, z):
        ids = self.ae.generate(encoder_outputs=self.BMO(last_hidden_state=self.ae.get_decoder_input(z)), **RC.BEAM)[0]
        eos = (ids[1:] == self.tok.eos_token_id).nonzero(as_tuple=True)[0]
        n = int(eos[0]) + 2 if len(eos) else len(ids)
        raw = self.tok.decode(ids[:n], skip_special_tokens=False)
        return dict(text=self.tok.decode(ids[:n], skip_special_tokens=True).strip(), sentinel='<extra_id_' in raw,
                    length_cap=n >= RC.MAX_TOKENS, generated_ids=ids.tolist())

    def encode_cache_path(self, text):
        batch = self.tok([text], return_tensors='pt', padding='max_length', truncation=True, max_length=RC.MAX_TOKENS).to('cuda')
        return self.ae.get_diffusion_latent(self.ae.get_encoder()(**batch), batch.attention_mask).float()

    def encode_audit_path(self, ids):
        x = self.torch.tensor([ids], device='cuda'); mask = self.torch.ones_like(x)
        return self.ae.get_diffusion_latent(self.ae.get_encoder()(input_ids=x, attention_mask=mask), mask)

    def metrics(self, ref, pred):
        return dict(exact=int(ref == pred), chrf=self.chrf.sentence_score(pred, [ref]).score, **numeral_counts(ref, pred))


def done_rows(path):
    return sum(1 for line in path.open(encoding='utf8') if line.strip()) if path.exists() else 0


def append(path, row):
    with path.open('a', encoding='utf8') as f: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def task_extra_monitor(M):
    rows = RC.rows_of(RC.ROOT / 'data/extra_afrikaans/processed/extra_monitor.jsonl')[:5 if SMOKE else None]
    path = OUT / 'extra_monitor.jsonl'; start = done_rows(path)
    with M.torch.inference_mode():
        for i, r in enumerate(rows[start:], start):
            ids = M.tok(r['text'], truncation=True, max_length=RC.MAX_TOKENS)['input_ids']
            view = M.tok.decode(ids, skip_special_tokens=True)
            out = M.decode(M.encode_cache_path(r['text']))
            append(path, dict(id=r['id'], document_id=r['document_id'], source=r.get('source'), model_view=view, output=out['text'],
                              length_cap=out['length_cap'], **M.metrics(view, out['text'])))
            if (i + 1) % 100 == 0: log('extra_monitor', done=i + 1, of=len(rows))
    res = RC.rows_of(path)
    nums = {k: sum(r[k] for r in res) for k in ('expected', 'produced', 'matched')}
    summary = dict(passages=len(res), exact=sum(r['exact'] for r in res), exact_rate=float(np.mean([r['exact'] for r in res])),
                   chrf=float(np.mean([r['chrf'] for r in res])), numeral_recall=nums['matched'] / nums['expected'] if nums['expected'] else None,
                   numeral_precision=nums['matched'] / nums['produced'] if nums['produced'] else None,
                   length_cap=sum(r['length_cap'] for r in res),
                   by_source={s: dict(n=len(rs), exact=sum(r['exact'] for r in rs), chrf=float(np.mean([r['chrf'] for r in rs])))
                              for s in sorted({r['source'] for r in res}) for rs in [[r for r in res if r['source'] == s]]})
    save('extra_monitor_summary.json', summary)
    return {k: summary[k] for k in ('passages', 'exact', 'chrf', 'numeral_recall')}


def task_shuffle_seeds(M):
    selected = RC.read(RC.STAGE1 / 'manifest.json')['selected']
    stored = {r['id']: r for r in RC.rows_of(RC.STAGE1 / 'examples.jsonl')}
    items = [dict(id=r['id'], ids=r['ids'], document_id=stored[r['id']]['document_id'], bin=stored[r['id']]['bin'],
                  view=stored[r['id']]['model_view'], stored=stored[r['id']]['correct']['generated_ids']) for r in selected]
    if SMOKE: items = items[:6]
    with M.torch.inference_mode():
        latents = [M.encode_audit_path(it['ids']) for it in items]
        mismatched = [it['id'] for it, z in zip(items, latents)
                      if M.ae.generate(encoder_outputs=M.BMO(last_hidden_state=M.ae.get_decoder_input(z)), **RC.BEAM)[0].tolist() != it['stored']]
        if mismatched and not SMOKE: raise RuntimeError(f'Gate: correct arm does not reproduce stored outputs for {len(mismatched)} passages')
        results = {}
        for seed in SHUFFLE_SEEDS:
            rng = random.Random(seed); rows = []
            for i, it in enumerate(items):
                pool = [j for j, o in enumerate(items) if o['bin'] == it['bin'] and o['document_id'] != it['document_id']] \
                       or [j for j, o in enumerate(items) if o['document_id'] != it['document_id']]
                j = rng.choice(pool)
                out = M.decode(latents[j])
                rows.append(dict(id=it['id'], donor=items[j]['id'], **M.metrics(it['view'], out['text'])))
            results[seed] = dict(exact=sum(r['exact'] for r in rows), chrf=float(np.mean([r['chrf'] for r in rows])), rows=rows)
            log('shuffle_seed', seed=seed, exact=results[seed]['exact'], chrf=results[seed]['chrf'])
    chrfs = [v['chrf'] for v in results.values()]
    save('shuffle_seeds.json', dict(passages=len(items), correct_arm_reproduced=not mismatched, seeds=results,
                                    chrf_mean=float(np.mean(chrfs)), chrf_range=[min(chrfs), max(chrfs)]))
    return dict(chrf_mean=float(np.mean(chrfs)), chrf_range=[min(chrfs), max(chrfs)], exact=[v['exact'] for v in results.values()])


def random_latents(M, kind, base, n):
    torch, F = M.torch, M.torch.nn.functional
    if kind == 'C1':
        return [F.normalize(torch.randn(1, RC.SLOTS, RC.DIM, generator=torch.Generator(device='cpu').manual_seed(base + i)), dim=-1)[0] * RC.NORM
                for i in range(n)]
    if not hasattr(M, 'c2_fit'):
        x = RC.train_latents(torch).reshape(-1, RC.SLOTS * RC.DIM).double().numpy()
        mu, cov, _, _ = RC.ledoit_wolf(x); M.c2_fit = (mu, np.linalg.cholesky(cov))
    mu, chol = M.c2_fit
    pre = np.stack([mu + chol @ np.random.Generator(np.random.PCG64(base + i)).standard_normal(RC.SLOTS * RC.DIM) for i in range(n)])
    return list(torch.from_numpy(RC.project(pre.reshape(n, RC.SLOTS, RC.DIM))).float())


def task_random_banks(M):
    n = 4 if SMOKE else RC.N_CONTROL
    with M.torch.inference_mode():
        for name, (kind, base) in RANDOM_BANKS.items():
            path = OUT / f'random-{name}.jsonl'; start = done_rows(path)
            if start >= n: continue
            began = time.monotonic()
            for i, z in enumerate(random_latents(M, kind, base, n)[start:], start):
                out = M.decode(z.unsqueeze(0).cuda())
                append(path, dict(bank=name, seed=base + i, text=out['text'], sentinel=out['sentinel'], length_cap=out['length_cap']))
                if (i + 1) % 100 == 0: log('random_bank', bank=name, done=i + 1, seconds_per_row=(time.monotonic() - began) / (i + 1 - start))
    return 'decoded'


def score_random_banks(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, str(RC.BUNDLE))
    from tools.compare_afrikaans_generators import score_text
    assert RC.digest(args.mgpt / 'pytorch_model.bin') == RC.MGPT_SHA, 'mGPT weights differ from the pinned revision'
    mtok = AutoTokenizer.from_pretrained(args.mgpt, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.mgpt, local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    out = {}
    with torch.inference_mode():
        for name in RANDOM_BANKS:
            src, dst = OUT / f'random-{name}.jsonl', OUT / f'random-{name}-scored.jsonl'
            rows = RC.rows_of(src); start = done_rows(dst)
            for r in rows[start:]:
                append(dst, dict(r, mgpt=score_text(model, mtok, r['text'], 'cuda') if r['text'] else None))
            out[name] = standardised(RC.rows_of(dst))
    for name in ('C1', 'C2'):
        orig = RC.ROOT / f'outputs/latent-controls-20261030/scored-{name}.jsonl'
        if orig.exists(): out[f'{name}-original'] = standardised(RC.rows_of(orig))
    for kind in ('C1', 'C2'):
        std = [v['standardised_ppl'] for k, v in out.items() if k.startswith(kind) and v['standardised_ppl']]
        raw = [v['raw_ppl'] for k, v in out.items() if k.startswith(kind)]
        out[f'{kind}_spread'] = dict(banks=len(raw), raw_range=[min(raw), max(raw)] if raw else None, standardised_range=[min(std), max(std)] if std else None)
    save('random_banks.json', out)
    del model; torch.cuda.empty_cache()
    return {k: v for k, v in out.items() if k.endswith('_spread')}


# ---------------------------------------------------------------- runner

def main():
    global RC, OUT, LOG, SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True, help='checkout holding outputs/ and data/')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--mgpt', type=Path)
    args = ap.parse_args()
    SMOKE = args.smoke
    spec = importlib.util.spec_from_file_location('run_controls', args.root / 'docs/research/latent_controls_2026-10-30/run_controls.py')
    RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)
    args.mgpt = args.mgpt or RC.MGPT_DEFAULT
    OUT = args.root / ('outputs/local-followups-smoke' if SMOKE else 'outputs/local-followups'); OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / 'log.jsonl'
    RC.check_inputs()
    log('inputs_checked', root=str(args.root), smoke=SMOKE)

    cpu = dict(numerals=task_numerals, overlap=task_overlap, lengths=task_lengths, figure1=task_figure1,
               partial_copies=task_partial_copies, scorers=task_scorers)
    gpu = dict(extra_monitor=task_extra_monitor, shuffle_seeds=task_shuffle_seeds, random_banks=task_random_banks)
    chosen = set(filter(None, args.only.split(','))) or set(cpu) | set(gpu) | {'score_random_banks'}
    status_path = OUT / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else {}

    def run(name, fn, *a):
        if (OUT / f'{name}.done').exists() and not SMOKE:
            log('skip', task=name); return
        began = time.monotonic(); log('start', task=name)
        try:
            result = fn(*a)
            (OUT / f'{name}.done').write_text('ok', encoding='utf8')
            status[name] = dict(ok=True, minutes=round((time.monotonic() - began) / 60, 1), result=result)
            log('done', task=name, minutes=status[name]['minutes'], result=result)
        except Exception as e:
            status[name] = dict(ok=False, error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-2000:])
            log('failed', task=name, error=status[name]['error'])
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding='utf8')

    for name, fn in cpu.items():
        if name in chosen: run(name, fn)
    if chosen & set(gpu):
        try:
            M = Models()
        except Exception as e:
            status['gpu_setup'] = dict(ok=False, error=f'{type(e).__name__}: {e}'); log('failed', task='gpu_setup', error=str(e)); M = None
        if M:
            for name, fn in gpu.items():
                if name in chosen: run(name, fn, M)
            del M.ae; M.torch.cuda.empty_cache()
    if 'score_random_banks' in chosen and (OUT / 'random_banks.done').exists():
        run('score_random_banks', score_random_banks, args)

    lines = ['# Local follow-ups: summary', '', f'Output folder: `{OUT}`', '', '| Task | Status | Minutes | Key result |', '|---|---|---|---|']
    for name, s in status.items():
        lines.append(f"| {name} | {'ok' if s.get('ok') else 'FAILED'} | {s.get('minutes', '')} | {json.dumps(s.get('result', s.get('error')), ensure_ascii=False)[:300]} |")
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    log('all_finished', failed=[k for k, s in status.items() if not s.get('ok')])


if __name__ == '__main__':
    main()
