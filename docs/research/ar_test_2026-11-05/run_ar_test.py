"""AR baseline on the test split (PROTOCOL.md in this folder). Inference only; resumable.

Generates 1,000 AR samples with the 22 October audit's cap-127 settings, scores them with the
frozen mGPT scorer, then compares every test bank against AR under the 1.20 rule.
"""
import importlib.util, json, math, os, random, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/ar-test-20261105'
AR = ROOT / 'outputs/cluster-results/ar-baseline-57759'
EVAL = ROOT / 'outputs/longrun-local-evaluation'
NEW = ROOT / 'outputs/cluster-results/ae-diffusion-77176/screen/diffusion/runs'
TRAIN = ROOT / 'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'
SCORER = ROOT / 'outputs/longrun-local-inputs/tools/compare_afrikaans_generators.py'
REV = '40897bd7c8b47a76802c411108ca6220438b8b40'
AR_SHA = 'ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076'
SEED0, N, CAP = 120000, 1000, 127
BANDS, WEIGHTS = [(1, 10), (11, 20), (21, 35), (36, 10**9)], [0.140, 0.223, 0.349, 0.288]
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(ROOT / 'docs/research/ar_validation_audit_2026-10-22'))


def log(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=False), flush=True)


def jsonl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def generate(torch, audit):
    path = OUT / 'ar-test-samples.jsonl'
    done = jsonl(path) if path.exists() else []
    assert [r['seed'] for r in done] == list(range(SEED0, SEED0 + len(done)))
    if len(done) == N:
        return
    tok, model, _ = audit.load_ar()
    start = time.monotonic()
    with path.open('a', encoding='utf8') as stream, torch.inference_mode():
        for seed in range(SEED0 + len(done), SEED0 + N):
            torch.manual_seed(seed)
            random.seed(seed)
            ids = torch.tensor([[0]], device='cuda')
            tokens = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, top_k=0, top_p=.95, temperature=1.,
                                    num_beams=1, max_new_tokens=CAP, min_new_tokens=0, repetition_penalty=1., no_repeat_ngram_size=0,
                                    use_cache=True, pad_token_id=0, bos_token_id=0, eos_token_id=1)[0].tolist()
            raw = tok.decode(tokens[1:], skip_special_tokens=False)
            stream.write(json.dumps(dict(seed=seed, text=tok.decode(tokens[1:], skip_special_tokens=True).strip(), generated_tokens=len(tokens) - 1,
                                         eos=1 in tokens[1:], length_cap=len(tokens) - 1 >= CAP, sentinel='<extra_id_' in raw), ensure_ascii=False) + '\n')
            stream.flush()
            if (seed - SEED0 + 1) % 100 == 0:
                log('generated', done=seed - SEED0 + 1, seconds=round(time.monotonic() - start))


def score(torch, audit):
    path = OUT / 'ar-test-scored.jsonl'
    rows = jsonl(OUT / 'ar-test-samples.jsonl')
    done = jsonl(path) if path.exists() else []
    if len(done) == N:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    module = audit.load_module('frozen_scorer', SCORER)
    snap = snapshot_download('ai-forever/mGPT', revision=REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snap, local_files_only=True, dtype='float32').cuda().eval().requires_grad_(False)
    anchors = []
    for name in ('small-seed42', 'published-seed42'):
        for r in read(EVAL / 'runs' / name / 'test-mgpt.json')['rows'][:2]:
            new = module.score_text(model, tok, r['text'], 'cuda')
            anchors.append(dict(run=name, seed=r['seed'], old=r['mgpt'], new=new,
                                rel=abs(new['nll'] - r['mgpt']['nll']) / max(1., abs(r['mgpt']['nll'])), tokens_equal=new['tokens'] == r['mgpt']['tokens']))
    ok = all(a['rel'] <= 1e-4 and a['tokens_equal'] for a in anchors)
    (OUT / 'anchors.json').write_text(json.dumps(dict(status='passed' if ok else 'failed', rows=anchors), indent=1), encoding='utf8')
    assert ok, 'Scorer anchors failed; stop'
    assert [r['text'] for r in done] == [r['text'] for r in rows[:len(done)]]
    with path.open('a', encoding='utf8') as stream:
        for r in rows[len(done):]:
            stream.write(json.dumps(dict(r, mgpt=module.score_text(model, tok, r['text'], 'cuda') if r['text'] else None), ensure_ascii=False) + '\n')
    log('scored', n=N)


def band(w):
    return next(i for i, (a, b) in enumerate(BANDS) if a <= w <= b)


def stats(rows, train_texts, train4):
    s = [r for r in rows if r.get('mgpt')]
    nll, tok = sum(r['mgpt']['nll'] for r in s), sum(r['mgpt']['tokens'] for r in s)
    by = {k: [0., 0] for k in range(4)}
    cnt = [0] * 4
    for r in s:
        b = band(len(r['text'].split()))
        by[b][0] += r['mgpt']['nll']; by[b][1] += r['mgpt']['tokens']; cnt[b] += 1
    std = math.exp(sum(w * by[k][0] / by[k][1] for k, w in enumerate(WEIGHTS))) if min(cnt) >= 20 else None
    words = [r['text'].split() for r in rows]
    bi = [tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1)]
    g4 = [tuple(x.casefold() for x in w[i:i + 4]) for w in words for i in range(len(w) - 3)]
    return dict(n=len(rows), empty=len(rows) - len(s), ppl=math.exp(nll / tok), std_ppl=std, band_counts=cnt,
                mean_words=sum(map(len, words)) / len(rows), distinct2=len(set(bi)) / max(1, len(bi)),
                exact_copies=sum(' '.join(r['text'].split()).casefold() in train_texts for r in rows),
                share4_train=sum(x in train4 for x in g4) / max(1, len(g4)))


def boot_logvar(rows, key, reps=2000, seed=20261105):
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        v = stats_fast(pick)[key]
        if v:
            vals.append(math.log(v))
    return statistics.variance(vals)


def stats_fast(rows):
    s = [r for r in rows if r.get('mgpt')]
    nll, tok = sum(r['mgpt']['nll'] for r in s), sum(r['mgpt']['tokens'] for r in s)
    by = {k: [0., 0] for k in range(4)}
    cnt = [0] * 4
    for r in s:
        b = band(len(r['text'].split()))
        by[b][0] += r['mgpt']['nll']; by[b][1] += r['mgpt']['tokens']; cnt[b] += 1
    std = math.exp(sum(w * by[k][0] / by[k][1] for k, w in enumerate(WEIGHTS))) if min(cnt) >= 20 else None
    return dict(ppl=math.exp(nll / tok), std_ppl=std)


def analyse():
    from scipy import stats as st
    train_texts = {' '.join(r['text'].split()).casefold() for r in jsonl(TRAIN)}
    train4 = set()
    for t in train_texts:
        w = t.split()
        train4.update(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    ar = jsonl(OUT / 'ar-test-scored.jsonl')
    banks = {'AR': stats(ar, train_texts, train4), 'human': stats(read(EVAL / 'test-reference-mgpt.json')['rows'], train_texts, train4)}
    systems = {}
    for label, folder, fams in (('B0', EVAL / 'runs', ('small', 'published')), ('D', NEW, ('small', 'published'))):
        for fam in fams:
            runs = sorted(p for p in folder.iterdir() if p.name.startswith(fam + '-'))
            systems[f'{label}-{fam}'] = {p.name: stats(read(p / 'test-mgpt.json')['rows'], train_texts, train4) for p in runs}
    ratios = {}
    for key in ('ppl', 'std_ppl'):
        var_ar = boot_logvar(ar, key)
        for name, runs in systems.items():
            logs = [math.log(v[key]) for v in runs.values()]
            n = len(logs)
            point = statistics.mean(v[key] for v in runs.values()) / banks['AR'][key]
            se = math.sqrt(statistics.variance(logs) / n + var_ar)
            t = st.t.ppf(.975, n - 1)
            lo, hi = point * math.exp(-t * se), point * math.exp(t * se)
            verdict = ('clearly passes' if hi <= 1.2 else 'passes') if point <= 1.2 else ('clearly fails' if lo > 1.2 else 'inconclusive')
            ratios[f'{name} {key}'] = dict(ratio=point, ci=[lo, hi], runs=n, verdict=verdict)
    summary = {name: dict(ppl=statistics.mean(v['ppl'] for v in r.values()), std_ppl=statistics.mean(v['std_ppl'] for v in r.values()),
                          mean_words=statistics.mean(v['mean_words'] for v in r.values()), copies=sum(v['exact_copies'] for v in r.values()))
               for name, r in systems.items()}
    out = dict(banks=banks, systems=systems, system_means=summary, ratios=ratios)
    (OUT / 'analysis.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    print(json.dumps(dict(AR=banks['AR'], human=banks['human'], system_means=summary, ratios=ratios), indent=1))


def main():
    import torch, transformers, hashlib
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    audit = importlib.import_module('run_audit')
    assert audit.digest(AR / 'best_model/model.safetensors') == AR_SHA, 'AR weights differ from the 22 October audit'
    contract = dict(protocol_sha256=audit.digest(HERE / 'PROTOCOL.md'), script_sha256=audit.digest(__file__), ar_sha256=AR_SHA,
                    scorer_sha256=audit.digest(SCORER), seeds=[SEED0, SEED0 + N - 1], cap=CAP, torch=torch.__version__,
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(), test_text_read='human test references (already scored) only')
    if (OUT / 'contract.json').exists():
        assert read(OUT / 'contract.json') == contract, 'Contract changed; use a new directory'
    else:
        (OUT / 'contract.json').write_text(json.dumps(contract, indent=1), encoding='utf8')
    generate(torch, audit)
    score(torch, audit)
    analyse()
    log('complete')


if __name__ == '__main__':
    main()
