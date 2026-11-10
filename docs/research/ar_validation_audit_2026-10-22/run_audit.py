"""Read-only model audit and validation inference; never trains or reads test data."""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/ar-validation-audit-20261022'
AR = ROOT / 'outputs/cluster-results/ar-baseline-57759'
DATA = ROOT / 'data/nchlt_afrikaans/processed/stage_5_7'
BUNDLE = ROOT / 'outputs/longrun-local-inputs'
EVAL = ROOT / 'outputs/longrun-local-evaluation'
REV = '40897bd7c8b47a76802c411108ca6220438b8b40'
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def log(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), ensure_ascii=False), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup():
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this bounded audit')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    contract = dict(protocol_sha256=digest(HERE / 'PROTOCOL.md'),
                    model_sha256=digest(AR / 'best_model/model.safetensors'),
                    config_sha256=digest(AR / 'best_model/config.json'),
                    train_sha256=digest(DATA / 'train.jsonl'), valid_sha256=digest(DATA / 'valid.jsonl'),
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(), precision='float32 unless explicitly BF16 reproduction',
                    scorer_revision=REV, test_accessed=False)
    if (OUT / 'contract.json').exists():
        if read(OUT / 'contract.json') != contract:
            raise RuntimeError('Audit contract changed; use a new experiment directory')
    else:
        save('contract.json', contract)
    return torch


def load_ar():
    from transformers import AutoTokenizer, GPT2LMHeadModel
    tok = AutoTokenizer.from_pretrained(AR / 'best_model', local_files_only=True)
    model, info = GPT2LMHeadModel.from_pretrained(AR / 'best_model', local_files_only=True,
        dtype='float32', attn_implementation='eager', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Non-strict model load: {info}')
    model = model.cuda().eval().requires_grad_(False)
    return tok, model, info


def validation_rows():
    return [json.loads(line) for line in (DATA / 'valid.jsonl').read_text(encoding='utf8').splitlines() if line]


def pack(tok, texts, torch):
    enc = tok(texts, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
    bos = torch.full((len(texts), 1), tok.pad_token_id, dtype=torch.long)
    ids = torch.cat([bos, enc['input_ids']], dim=1)
    mask = torch.cat([torch.ones_like(bos), enc['attention_mask']], dim=1)
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, 0] = -100
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def evaluate(model, tok, rows, torch, bf16=False, batch_size=4):
    start = time.monotonic()
    results = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            if time.monotonic() - start > 3600:
                raise TimeoutError('One-hour stage limit')
            subset = rows[offset:offset+batch_size]
            batch = {k: v.cuda() for k, v in pack(tok, [r['text'] for r in subset], torch).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                batch['labels'][:, 1:], reduction='none', ignore_index=-100)
            counts = (batch['labels'][:, 1:] != -100).sum(1)
            totals = loss.sum(1)
            if not torch.isfinite(totals).all():
                raise FloatingPointError('Non-finite validation NLL')
            for row, nll, n in zip(subset, totals.tolist(), counts.tolist()):
                results.append(dict(id=row['id'], document_id=row['document_id'], nll=nll, tokens=n))
            del logits, loss, batch
            if offset % 256 == 0:
                log('validation_progress', done=len(results), total=len(rows), bf16=bf16,
                    seconds=round(time.monotonic()-start, 2))
    total_nll, total_n = sum(r['nll'] for r in results), sum(r['tokens'] for r in results)
    ce = total_nll / total_n
    return dict(examples=len(rows), tokens=total_n, cross_entropy=ce, causal_perplexity=math.exp(ce),
                seconds=time.monotonic()-start, precision='bf16_autocast' if bf16 else 'float32', rows=results)


def preflight(torch):
    from transformers import AutoTokenizer
    tok, model, info = load_ar()
    protocol = read(AR / 'protocol.json')
    assert digest(DATA / 'train.jsonl') == protocol['dataset_hashes']['train']
    assert digest(DATA / 'valid.jsonl') == protocol['dataset_hashes']['valid']
    vocab = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()
    assert vocab == protocol['tokenizer_vocab_sha256'], 'Vocabulary differs'
    assert (tok.pad_token_id, tok.eos_token_id, model.config.bos_token_id) == (0, 1, 0)
    assert sum(p.numel() for p in model.parameters()) == 67218176
    raw = torch.load(AR / 'best.pt', map_location='cpu', weights_only=True)
    state = model.state_dict()
    assert set(raw) == set(state), 'Export key mismatch'
    equal = all(torch.equal(raw[k], state[k].cpu()) for k in raw)
    assert equal, 'Export tensor differs from selected best.pt'
    del raw, state
    other = AutoTokenizer.from_pretrained(BUNDLE / 'models/mt5-base', local_files_only=True)
    rows = validation_rows()
    texts = [r['text'] for r in rows]
    a = tok(texts, truncation=True, max_length=128)['input_ids']
    b = other(texts, truncation=True, max_length=128)['input_ids']
    assert a == b, 'AR and diffusion tokenizer encodings differ'
    probe = pack(tok, texts[:3], torch)
    assert (probe['labels'][:, 0] == -100).all()
    assert (probe['labels'][probe['attention_mask'] == 0] == -100).all()
    assert (probe['attention_mask'][:, 0] == 1).all()
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    selected = [rows[i] for i in indices[:128]]
    bf = evaluate(model, tok, selected, torch, bf16=True, batch_size=1)
    fp = evaluate(model, tok, selected, torch, bf16=False, batch_size=1)
    historical = read(AR / 'best_selection.json')['validation_cross_entropy']
    difference = bf['cross_entropy'] - historical
    report = dict(status='passed' if abs(difference) <= .02 else 'failed_reproduction',
        loading_info={k: sorted(v) if isinstance(v, set) else v for k,v in info.items()}, parameters=sum(p.numel() for p in model.parameters()),
        weights_equal_best_pt=equal, best_pt_sha256=digest(AR/'best.pt'),
        tokenizer_vocab_sha256=vocab, validation_tokenizer_matches=len(rows),
        bos_pad_eos=[0,0,1], selection_indices=indices[:128], historical_ce=historical,
        bf16_ce_difference=difference, bf16=bf, fp32=fp,
        original_training_script_sha256=digest(ROOT/'tools/train_afrikaans_baseline.py'),
        handoff_training_script_sha256=digest(ROOT/'handoff_2026-10-20/github-research-diffusion/project/tools/train_afrikaans_baseline.py'))
    save('preflight.json', report)
    log('preflight_complete', status=report['status'], bf16_ce=bf['cross_entropy'], fp32_ce=fp['cross_entropy'], difference=difference)
    assert report['status'] == 'passed', 'Historical CE mismatch; stop'


def full_validation(torch):
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    tok, model, _ = load_ar()
    report = evaluate(model, tok, validation_rows(), torch)
    report['selection_changed'] = False
    save('full-validation.json', report)
    log('full_validation_complete', **{k:v for k,v in report.items() if k!='rows'})


def generate(torch):
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    tok, model, _ = load_ar()
    start = time.monotonic()
    for cap in (127, 64):
        path = OUT / f'ar-cap{cap}-samples.jsonl'
        existing = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if path.exists() else []
        assert [r['seed'] for r in existing] == list(range(30000,30000+len(existing)))
        with path.open('a', encoding='utf8') as stream, torch.inference_mode():
            for seed in range(30000 + len(existing), 30256):
                if time.monotonic()-start > 3600:
                    raise TimeoutError('One-hour generation limit; partial output retained')
                torch.manual_seed(seed)
                random.seed(seed)
                ids = torch.tensor([[0]], device='cuda')
                tokens = model.generate(ids, attention_mask=torch.ones_like(ids),
                    do_sample=True, top_k=0, top_p=.95, temperature=1., num_beams=1,
                    max_new_tokens=cap, min_new_tokens=0, repetition_penalty=1., no_repeat_ngram_size=0,
                    use_cache=True, pad_token_id=0, bos_token_id=0, eos_token_id=1)[0].tolist()
                raw = tok.decode(tokens[1:], skip_special_tokens=False)
                row = dict(seed=seed, token_ids=tokens, text=tok.decode(tokens[1:], skip_special_tokens=True).strip(),
                    decoder_tokens=len(tokens), generated_tokens=len(tokens)-1,
                    eos=1 in tokens[1:], length_cap=len(tokens)-1>=cap, sentinel='<extra_id_' in raw,
                    max_new_tokens=cap)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                if (seed-30000+1) % 16 == 0:
                    log('generation_progress', cap=cap, done=seed-30000+1, seconds=round(time.monotonic()-start,2))
    save('generation-complete.json', dict(count_per_arm=256, caps=[127,64], seconds=time.monotonic()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), precision='float32', training_performed=False))


def scoring(torch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    assert read(OUT/'preflight.json')['status']=='passed'
    script = BUNDLE/'tools/compare_afrikaans_generators.py'
    expected = read(EVAL/'local-contract.json')['inputs']['code']['tools/compare_afrikaans_generators.py']
    assert digest(script)==expected
    scoring_module = load_module('frozen_scorer',script)
    snap = snapshot_download('ai-forever/mGPT', revision=REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snap, local_files_only=True, dtype='float32').cuda().eval().requires_grad_(False)
    anchors=[]
    for name in ('small-seed42','published-seed42'):
        for row in read(EVAL/'runs'/name/'validation-mgpt.json')['rows'][:2]:
            new = scoring_module.score_text(model,tok,row['text'],'cuda')
            old = row['mgpt']
            delta=abs(new['nll']-old['nll'])/max(abs(old['nll']),1.)
            assert new['tokens']==old['tokens']
            anchors.append(dict(run=name,seed=row['seed'],old=old,new=new,relative_nll_difference=delta))
    anchor_report=dict(status='passed' if max(r['relative_nll_difference'] for r in anchors)<=1e-4 else 'failed',
                       rows=anchors,attention_implementation=model.config._attn_implementation,
                       scorer_source_sha256=digest(script),snapshot=snap)
    if not (OUT/'scorer-anchors.json').exists():
        save('scorer-anchors.json',anchor_report)
    assert anchor_report['status']=='passed', 'Scorer compatibility failed'
    groups={f'ar-cap{cap}':[json.loads(s) for s in (OUT/f'ar-cap{cap}-samples.jsonl').read_text(encoding='utf8').splitlines()] for cap in (127,64)}
    assert all(len(v)==256 for v in groups.values())
    groups['ar-historical100']=[dict(index=i,text=t) for i,t in enumerate((AR/'samples.txt').read_text(encoding='utf8').splitlines())]
    assert len(groups['ar-historical100'])==100
    start=time.monotonic()
    for name, rows in groups.items():
        path=OUT/f'{name}-scored.jsonl'
        existing=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if path.exists() else []
        assert [r['text'] for r in existing]==[r['text'] for r in rows[:len(existing)]]
        with path.open('a',encoding='utf8') as stream:
            for i,row in enumerate(rows[len(existing):],start=len(existing)):
                if time.monotonic()-start>3600:
                    raise TimeoutError('One-hour scoring limit')
                value=scoring_module.score_text(model,tok,row['text'],'cuda')
                stream.write(json.dumps(dict(**row,mgpt=value),ensure_ascii=False)+'\n')
                stream.flush()
                if (i+1)%32==0:
                    log('scoring_progress',group=name,done=i+1,total=len(rows),seconds=round(time.monotonic()-start,2))
    save('scoring-complete.json',dict(seconds=time.monotonic()-start,groups={k:len(v) for k,v in groups.items()},
         peak_cuda_bytes=torch.cuda.max_memory_allocated(),revision=REV,context=512,stride=256,precision='float32'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['preflight','validation','generate','score'])
    a=p.parse_args()
    torch=setup()
    log('stage_start',action=a.action,script_sha256=digest(__file__))
    {'preflight':preflight,'validation':full_validation,'generate':generate,'score':scoring}[a.action](torch)


if __name__=='__main__':
    main()
