"""Swahili checks under PROTOCOL.md (approved 1 November 2026).

Stages, each resumable from its own output file:
  gates  G1 hashes, G2 scorer reproduction, G3 autoencoder reproduction, G4 generation (informational)
  s1     reconstruction panel, correct and shuffled latents
  s2     1,000 random latents decoded and scored with mGPT
  s3     GlotLID on every bank
  s4     length bands and copying against the training split
Run: E:/afrikaans-eval-20261031/env/Scripts/python.exe run_check.py [--quick]
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # required by use_deterministic_algorithms
import argparse, hashlib, json, random, re, sys, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROJ = REPO / 'handoff_2026-10-20/github-research-diffusion/project'
ENGINE = PROJ / 'swahili_72h'
ROOT = REPO / 'swahili_72h'                      # assets/ and data/
RES = PROJ / 'results/swahili72'                 # stored samples and scores
CKPT = Path('E:/swahili-check-20261101/handoff/research-handoff/project/results/swahili72')
GLOTLID = Path('E:/afrikaans-eval-20261031/models/glotlid/model.bin')
OUT = REPO / 'outputs/swahili-check-20261101'
RUNS = ['small-42', 'small-43', 'published-42', 'published-43']
GEN = dict(min_length=5, num_beams=4, do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.2)
NUM = re.compile(r'\d+(?:[.,]\d+)*')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path, encoding='utf8')]


def wjson(name, value):
    (OUT / name).write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding='utf8')


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', s).casefold().split())


def band(n):
    return '1-10' if n <= 10 else '11-20' if n <= 20 else '21-35' if n <= 35 else '36+'


def tok_bin(n):
    return 0 if n <= 64 else 1 if n <= 128 else 2


def model_ids(tok, text):
    ids = tok.encode(text, add_special_tokens=True)          # same rule as tokenize_data.py
    return (ids[:191] + [tok.eos_token_id]) if len(ids) > 192 else ids


# ---------------------------------------------------------------- engine
def load_engine():
    sys.path.insert(0, 'E:/swahili-check-20261101/extra')  # einops only, kept out of the pinned env
    sys.path.insert(0, str(ENGINE))
    import engine as E
    c = json.loads((ENGINE / 'config.json').read_text(encoding='utf8'))
    return E, c


def gates(E, c, tok, quick):
    if (OUT / 'gates.json').exists():
        return json.loads((OUT / 'gates.json').read_text())
    import torch
    files = {'ae': CKPT / 'ae/best.pt', **{r: CKPT / f'runs/{r}/best.pt' for r in RUNS},
             'engine': ENGINE / 'engine.py', 'config': ENGINE / 'config.json',
             'valid': ROOT / 'data/valid.jsonl', 'train': ROOT / 'data/train.jsonl', 'glotlid': GLOTLID,
             'protocol': Path(__file__).parent / 'PROTOCOL.md', 'runner': Path(__file__)}
    contract = dict(hashes={k: sha(v) for k, v in files.items()},
                    versions=dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
                                  sacrebleu=__import__('sacrebleu').__version__, numpy=np.__version__),
                    gpu=torch.cuda.get_device_name())
    wjson('contract.json', contract)
    E.setup(c)
    g = {}
    scorer, stok = E.scorer(ROOT)
    for name in ('real-valid', 'small-42-valid'):
        stored = json.loads((RES / f'scores/{name}.json').read_text(encoding='utf8'))[:3]
        for r in stored:
            new = E.score(scorer, stok, r['text'])['perplexity']
            g.setdefault('G2', []).append(dict(bank=name, stored=r['score']['perplexity'], new=new,
                                               rel=abs(new - r['score']['perplexity']) / r['score']['perplexity']))
    del scorer; torch.cuda.empty_cache()
    g['G2_pass'] = all(x['rel'] <= 1e-4 for x in g['G2'])
    ae = E.ae_model(ROOT, c, CKPT / 'ae/best.pt')
    valid = jl(ROOT / 'data/valid.jsonl')
    values = np.zeros((len(valid), 192), dtype=np.int32)
    for i, r in enumerate(valid):
        ids = model_ids(tok, r['text']); values[i, :len(ids)] = ids
    v = E.validate(ae, values, c, 'ae', count=64 if quick else 1024)
    gate = json.loads((RES / 'ae-gate.json').read_text())['validation']
    g['G3'] = dict(new=v, stored=dict(loss=gate['loss'], token_accuracy=gate['token_accuracy']))
    g['G3_pass'] = (abs(v['loss'] - gate['loss']) / gate['loss'] <= .02 and
                    abs(v['token_accuracy'] - gate['token_accuracy']) / gate['token_accuracy'] <= .02)
    try:
        dm = E.inference_checkpoint(E.diffusion(c, 'small'), CKPT / 'runs/small-42/best.pt')
        stored = json.loads((RES / 'runs/small-42/valid-samples.json').read_text(encoding='utf8'))[:2]
        new = E.generate(dm, ae, tok, c, 2, stored[0]['seed'])
        g['G4'] = [dict(seed=s['seed'], match=s['text'] == n['text'], stored=s['text'][:120], new=n['text'][:120])
                   for s, n in zip(stored, new)]
        del dm
    except Exception as ex:  # informational only
        g['G4'] = dict(error=repr(ex))
    del ae; torch.cuda.empty_cache()
    g['passed'] = g['G2_pass'] and (g['G3_pass'] or quick)
    wjson('gates.json', g)
    return g


# ---------------------------------------------------------------- S1
def panel(valid):
    order = sorted(range(len(valid)), key=lambda i: valid[i]['id'])
    perm = np.random.default_rng(20261102).permutation(len(order))
    per_file, chosen = Counter(), []
    for j in perm:
        i = order[j]; f = valid[i]['source_file']
        if per_file[f] < 6:
            per_file[f] += 1; chosen.append(i)
        if len(chosen) == 200:
            break
    return chosen


@__import__('torch').no_grad()
def s1(E, c, tok, quick):
    import torch
    from sacrebleu.metrics import CHRF
    from transformers.modeling_outputs import BaseModelOutput
    path = OUT / 's1-reconstruction.jsonl'
    if (OUT / 's1-done.json').exists():
        return
    valid = jl(ROOT / 'data/valid.jsonl')
    ids = panel(valid)[: 6 if quick else 200]
    rows = [dict(i=i, id=valid[i]['id'], file=valid[i]['source_file'], ids=model_ids(tok, valid[i]['text'])) for i in ids]
    for r in rows:
        r['bin'] = tok_bin(len(r['ids'])); r['source'] = tok.decode(r['ids'], skip_special_tokens=True)
    rng = random.Random(20261103); donors = {}
    for r in rows:   # donor: another file, same token-length bin (nearest bin if none)
        pool = [o for o in rows if o['file'] != r['file'] and o['bin'] == r['bin']]
        if not pool:
            pool = sorted([o for o in rows if o['file'] != r['file']], key=lambda o: abs(o['bin'] - r['bin']))[:5]
            r['donor_fallback'] = True
        donors[r['id']] = rng.choice(pool)['id']
    wjson('panel.json', [dict(i=r['i'], id=r['id'], file=r['file'], bin=r['bin'], donor=donors[r['id']]) for r in rows])
    ae = E.ae_model(ROOT, c, CKPT / 'ae/best.pt').eval()
    chrf = CHRF(char_order=6, word_order=0)
    latents, norms = {}, []
    for r in rows:
        x = torch.tensor([r['ids']], device='cuda'); m = x.ne(0)
        with E.amp(c):
            z = ae.get_diffusion_latent(ae.get_encoder()(input_ids=x, attention_mask=m), m)
        latents[r['id']] = z.float(); norms += z.float().norm(dim=-1).flatten().tolist()
    wjson('latent-norm.json', dict(mean_slot_norm=float(np.mean(norms)), sd=float(np.std(norms))))
    by = {r['id']: r for r in rows}
    with open(path, 'w', encoding='utf8') as f:
        for r in rows:
            for arm, zid in (('correct', r['id']), ('shuffled', donors[r['id']])):
                z = latents[zid]
                x = torch.tensor([r['ids']], device='cuda'); labels = x.masked_fill(x.eq(0), -100)
                with E.amp(c):
                    enc = BaseModelOutput(last_hidden_state=ae.get_decoder_input(z.to(next(ae.parameters()).dtype) if False else z))
                    out = ae(labels=labels, encoder_outputs=enc)
                    pred = ae.generate(encoder_outputs=enc, max_length=c['max_length'], **GEN)[0]
                n = int(labels.ne(-100).sum())
                acc = int(((out.logits.argmax(-1) == labels) & labels.ne(-100)).sum()) / n
                text = tok.decode(pred, skip_special_tokens=True).strip()
                src = r['source'].strip()
                ns, np_ = NUM.findall(src), NUM.findall(text)
                match = sum((Counter(ns) & Counter(np_)).values())
                f.write(json.dumps(dict(id=r['id'], file=r['file'], bin=r['bin'], arm=arm, ce=float(out.loss), tokens=n,
                                        acc=acc, exact=text == src, chrf=chrf.sentence_score(text, [src]).score,
                                        num_src=len(ns), num_pred=len(np_), num_match=match,
                                        cap=len(pred) >= c['max_length'], source=src, output=text), ensure_ascii=False) + '\n')
            log('s1', r['id'][:8])
    del ae; torch.cuda.empty_cache()
    wjson('s1-done.json', dict(rows=len(rows)))


# ---------------------------------------------------------------- S2
@__import__('torch').no_grad()
def s2(E, c, tok, quick):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    path = OUT / 's2-random.jsonl'
    done = {json.loads(l)['i'] for l in open(path, encoding='utf8')} if path.exists() else set()
    n = 10 if quick else 1000
    if len(done) >= n:
        return
    target = json.loads((OUT / 'latent-norm.json').read_text())['mean_slot_norm']
    ae = E.ae_model(ROOT, c, CKPT / 'ae/best.pt').eval()
    texts = {}
    with open(path, 'a', encoding='utf8') as f:
        for i in range(n):
            if i in done:
                continue
            g = torch.Generator('cpu').manual_seed(40000 + i)
            z = torch.randn(1, 32, 64, generator=g)
            z = (z / z.norm(dim=-1, keepdim=True) * target).cuda()
            with E.amp(c):
                ids = ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z)),
                                  max_length=c['max_length'], **GEN)[0]
            raw = tok.decode(ids, skip_special_tokens=False)
            texts[i] = dict(i=i, seed=40000 + i, text=tok.decode(ids, skip_special_tokens=True).strip(),
                            sentinel='<extra_id_' in raw, tokens=len(ids))
            if i % 50 == 0:
                log('s2 decode', i)
    del ae; torch.cuda.empty_cache()
    scorer, stok = E.scorer(ROOT)
    with open(path, 'a', encoding='utf8') as f:
        for i, r in sorted(texts.items()):
            r['score'] = E.score(scorer, stok, r['text'])
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    del scorer; torch.cuda.empty_cache()


# ---------------------------------------------------------------- banks for S3/S4
def banks():
    b = {'real-valid': [r['text'] for r in json.loads((RES / 'scores/real-valid.json').read_text(encoding='utf8'))]}
    for r in RUNS:
        b[f'{r}-valid'] = [x['text'] for x in json.loads((RES / f'scores/{r}-valid.json').read_text(encoding='utf8'))]
    p = OUT / 's2-random.jsonl'
    if p.exists():
        b['random-latents'] = [r['text'] for r in sorted(jl(p), key=lambda r: r['i'])]
    p = OUT / 's1-reconstruction.jsonl'
    if p.exists():
        s = jl(p)
        b['panel-sources'] = [r['source'] for r in s if r['arm'] == 'correct']
        b['panel-reconstructions'] = [r['output'] for r in s if r['arm'] == 'correct']
    return b


def s3():
    import fasttext
    model = fasttext.load_model(str(GLOTLID))
    rows = []
    for name, texts in banks().items():
        for t in texts:
            if not t.strip():
                rows.append(dict(bank=name, label=None, band=band(0))); continue
            lab, p = model.predict(t.replace('\n', ' '), k=1)
            rows.append(dict(bank=name, label=lab[0].replace('__label__', ''), p=float(p[0]), band=band(len(t.split()))))
    with open(OUT / 's3-lid.jsonl', 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def s4():
    def grams(text):
        w = norm(text).split()
        return [hash(' '.join(w[i:i + 4])) for i in range(len(w) - 3)]
    train = jl(ROOT / 'data/train.jsonl')
    exact = {norm(r['text']) for r in train}
    G = set()
    for r in train:
        G.update(grams(r['text']))
    scores = {'real-valid': json.loads((RES / 'scores/real-valid.json').read_text(encoding='utf8'))}
    for r in RUNS:
        scores[f'{r}-valid'] = json.loads((RES / f'scores/{r}-valid.json').read_text(encoding='utf8'))
    p = OUT / 's2-random.jsonl'
    if p.exists():
        scores['random-latents'] = sorted(jl(p), key=lambda r: r['i'])
    out = {}
    for name, rows in scores.items():
        texts = [r['text'] for r in rows]
        g_all = [grams(t) for t in texts]
        tot = sum(len(g) for g in g_all); hit = sum(sum(x in G for x in g) for g in g_all)
        high = sum(1 for g in g_all if len(g) >= 4 and sum(x in G for x in g) / len(g) >= .9)
        bands = defaultdict(lambda: [0, 0.0, 0])
        for r in rows:
            s = r['score']
            if s.get('empty'):
                continue
            b = bands[band(len(r['text'].split()))]; b[0] += 1; b[1] += s['nll']; b[2] += s['tokens']
        nll = sum(r['score']['nll'] for r in rows if not r['score'].get('empty'))
        ntok = sum(r['score']['tokens'] for r in rows if not r['score'].get('empty'))
        bg = Counter()
        for t in texts:
            w = t.split(); bg.update(zip(w, w[1:]))
        out[name] = dict(n=len(rows), mean_words=float(np.mean([len(t.split()) for t in texts])),
                         ppl=float(np.exp(nll / ntok)), distinct2=len(bg) / max(1, sum(bg.values())),
                         exact_copies=sum(norm(t) in exact for t in texts), fourgram_overlap=hit / max(1, tot),
                         high_overlap_rows=high,
                         bands={k: dict(n=v[0], ppl=float(np.exp(v[1] / v[2]))) for k, v in sorted(bands.items())})
    wjson('s4.json', out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    global OUT
    if a.quick:
        OUT = OUT.parent / (OUT.name + '-quick')
    OUT.mkdir(parents=True, exist_ok=True)
    E, c = load_engine()
    from transformers import T5Tokenizer
    tok = T5Tokenizer.from_pretrained(str(ROOT / 'assets/mt5-base'), local_files_only=True, legacy=True)
    g = gates(E, c, tok, a.quick); log('gates', {k: v for k, v in g.items() if k.endswith('pass') or k == 'passed'})
    if not g['passed']:
        log('gate failed; stopping'); return 3
    s1(E, c, tok, a.quick); log('s1 done')
    s2(E, c, tok, a.quick); log('s2 done')
    s3(); log('s3 done')
    s4(); log('s4 done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
