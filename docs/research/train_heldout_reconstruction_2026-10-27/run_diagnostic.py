"""Train-versus-held-out reconstruction diagnostic (PROTOCOL.md, approved 27 October 2026).

Inference only. `--dry-run` builds the panels and prints their sizes without loading the
autoencoder or writing anything; the full run writes to a new outputs directory.
"""
import os, sys, json, hashlib, random, time, re, argparse, datetime
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'outputs/longrun-local-inputs'
AE=ROOT/'outputs/cluster-results/afrikaans-56531/best'
TRAIN=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'
VALID=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl'
STAGE1=ROOT/'outputs/reconstruction-audit-20261022'
AE_SHA='123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'
TRAIN_SHA='2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab'
VALID_SHA='bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e'
# SHA-256 of the approved PROTOCOL.md with line endings normalised to LF (commit ad8ac72).
PROTOCOL_SHA='4ba58f57e401966a156dffd49249d4c9867cc68a6388c07231204449e9333fab'
SEED_VALID_DOCS=91427   # same call and seed as the 22 October audit
SEED_TRAIN=91430        # training document order and within-document order
SEED_DONOR_T=91432
SEED_DONOR_V=91433
SEED_VALID_WITHIN=91434
MAX_TOKENS=128
VALID_CAP=4
BEAM=dict(max_length=128,min_length=5,num_beams=4,do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2,length_penalty=1.0)
GREEDY=dict(max_length=128,min_length=5,num_beams=1,do_sample=False,no_repeat_ngram_size=0,repetition_penalty=1.0,length_penalty=1.0)
GPU_SECONDS_CAP=7200
MEMORY_CAP=10*1024**3
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf8')

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def lf_digest(p): return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def rows_of(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]
def bin_of(n): return 'short' if n<=32 else 'medium' if n<=64 else 'long'

def annotate(tok,r):
    if 'ids' in r: return r
    full=tok(r['text'],truncation=False)['input_ids']
    r['full_tokens']=len(full)
    r['bin']=bin_of(len(full))
    r['ids']=tok(r['text'],truncation=True,max_length=MAX_TOKENS)['input_ids']
    r['model_view']=tok.decode(r['ids'],skip_special_tokens=True)
    return r

def derange(panel,seed):
    """Same-bin, cross-document donor permutation, as in the 22 October audit."""
    rng=random.Random(seed); donors={}
    for label in ('short','medium','long'):
        ix=[i for i,r in enumerate(panel) if r['bin']==label]
        if not ix: continue
        if len(ix)<2: raise ValueError(f'Bin {label} has one passage; no donor possible')
        for _ in range(100000):
            perm=ix.copy(); rng.shuffle(perm)
            if all(panel[i]['document_id']!=panel[j]['document_id'] for i,j in zip(ix,perm)): break
        else: raise ValueError(f'No cross-document derangement in bin {label}')
        donors.update(zip(ix,perm))
    return donors

def build_panels(tok):
    stage1=read(STAGE1/'manifest.json')
    valid=rows_of(VALID)
    groups=defaultdict(list)
    for r in valid: groups[r['document_id']].append(r)
    docs=sorted(groups); random.Random(SEED_VALID_DOCS).shuffle(docs)
    dev_docs={r['document_id'] for r in stage1['selected']}
    reserve_docs={r['document_id'] for r in stage1['reserved']}
    assert set(docs[:64])==dev_docs, 'Dev documents are not positions 0-63'
    assert set(docs[64:128])==reserve_docs, 'Reserve documents are not positions 64-127'
    held_docs=docs[128:]
    assert len(held_docs)==123 and not (set(held_docs)&(dev_docs|reserve_docs))
    rng=random.Random(SEED_VALID_WITHIN); panel_v=[]
    for d in held_docs:
        candidates=[annotate(tok,r) for r in groups[d]]
        candidates=[r for r in candidates if r['full_tokens']<=MAX_TOKENS]
        rng.shuffle(candidates)
        picked=[]; bins=set()
        for r in candidates:
            if r['bin'] not in bins and len(picked)<VALID_CAP: picked.append(r); bins.add(r['bin'])
        for r in candidates:
            if len(picked)<VALID_CAP and r not in picked: picked.append(r)
        panel_v.extend(picked)
    quota=Counter(r['bin'] for r in panel_v)
    train=rows_of(TRAIN)
    tgroups=defaultdict(list)
    for r in train: tgroups[r['document_id']].append(r)
    tdocs=sorted(tgroups); rng=random.Random(SEED_TRAIN); rng.shuffle(tdocs)
    remaining=dict(quota); panel_t=[]
    for d in tdocs:
        if not any(remaining.values()): break
        candidates=tgroups[d].copy(); rng.shuffle(candidates)
        for r in candidates:
            annotate(tok,r)
            if r['full_tokens']<=MAX_TOKENS and remaining.get(r['bin'],0)>0:
                panel_t.append(r); remaining[r['bin']]-=1; break
    if any(remaining.values()): raise ValueError(f'Training quotas unfilled: {remaining}')
    # Re-tokenise the anchor from its text so the comparison with the stored ids is not circular.
    anchor=[annotate(tok,{k:r[k] for k in ('id','document_id','text')}) for r in stage1['selected']]
    for a,s in zip(anchor,stage1['selected']): assert a['ids']==s['ids'], 'Anchor tokenisation differs from 22 October'
    assert not ({r['document_id'] for r in panel_t}&{r['document_id'] for r in valid})
    return dict(anchor=anchor,T=panel_t,V=panel_v,quota=dict(quota))

def strip(r): return {k:r[k] for k in ('id','document_id','bin','full_tokens','ids','model_view','text')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools')); from as_run import as_run_sha256, as_run_matches  # protocols are checked as they were at tag as-run
    assert as_run_sha256(HERE/'PROTOCOL.md')==PROTOCOL_SHA, 'PROTOCOL.md at tag as-run differs from the approved version'
    assert digest(TRAIN)==TRAIN_SHA and digest(VALID)==VALID_SHA
    original=read(BUNDLE/'inputs.json')
    for p,h in original['code'].items(): assert digest(BUNDLE/p)==h,p
    sys.path.insert(0,str(BUNDLE))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BUNDLE/'models/mt5-base',local_files_only=True)
    panels=build_panels(tok)
    donors={'T':derange(panels['T'],SEED_DONOR_T),'V':derange(panels['V'],SEED_DONOR_V)}
    sizes={k:dict(n=len(panels[k]),documents=len({r['document_id'] for r in panels[k]}),bins=dict(Counter(r['bin'] for r in panels[k]))) for k in ('anchor','T','V')}
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,sizes=sizes),indent=2)); return

    import torch, transformers, sacrebleu
    from transformers.modeling_outputs import BaseModelOutput
    from sacrebleu.metrics import CHRF
    from tools.longrun_training import load_ae
    torch.set_num_threads(4)
    torch.manual_seed(SEED_TRAIN)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    assert digest(AE/'model.pt')==AE_SHA
    OUT=ROOT/f"outputs/train-heldout-recon-{datetime.date.today():%Y%m%d}"
    OUT.mkdir(exist_ok=False)
    def save(name,obj):
        with (OUT/name).open('x',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    def fail(gate,detail):
        save('failure.json',dict(gate=gate,detail=detail)); raise SystemExit(f'{gate} failed: {detail}')
    save('manifest.json',dict(anchor=[strip(r) for r in panels['anchor']],T=[strip(r) for r in panels['T']],V=[strip(r) for r in panels['V']],
        donor_indices={k:{str(i):j for i,j in v.items()} for k,v in donors.items()},quota=panels['quota'],sizes=sizes))
    save('contract.json',dict(protocol_sha256_lf=PROTOCOL_SHA,protocol_sha256_raw=digest(HERE/'PROTOCOL.md'),script_sha256=digest(__file__),
        manifest_sha256=digest(OUT/'manifest.json'),ae_sha256=AE_SHA,train_sha256=TRAIN_SHA,valid_sha256=VALID_SHA,source_code=original['code'],
        packages=dict(torch=torch.__version__,transformers=transformers.__version__,sacrebleu=sacrebleu.__version__),gpu=torch.cuda.get_device_name(),
        precision='FP32',beam=BEAM,greedy=GREEDY,seeds=dict(valid_docs=SEED_VALID_DOCS,train=SEED_TRAIN,donor_T=SEED_DONOR_T,donor_V=SEED_DONOR_V,valid_within=SEED_VALID_WITHIN,bootstrap=91431)))
    print(json.dumps(dict(event='manifest_frozen',sizes=sizes)),flush=True)

    ae,loaded_tok=load_ae(AE,BUNDLE/'models/mt5-base',dict(latent_dim=64,latent_length=32,max_length=MAX_TOKENS))
    assert loaded_tok.get_vocab()==tok.get_vocab()
    chrf=CHRF(char_order=6,word_order=0,beta=2)
    def metrics(ref,pred):
        a,b=Counter(re.findall(r'\d+(?:[.,]\d+)*',ref)),Counter(re.findall(r'\d+(?:[.,]\d+)*',pred))
        return dict(exact=int(ref==pred),similarity=SequenceMatcher(None,ref,pred,autojunk=False).ratio(),chrf=chrf.sentence_score(pred,[ref]).score,
            number_expected=sum(a.values()),number_predicted=sum(b.values()),number_matched=sum((a&b).values()))
    def encode(r):
        x=torch.tensor([r['ids']],device='cuda'); mask=torch.ones_like(x)
        z=ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
        if not torch.isfinite(z).all(): fail('G2','non-finite latent '+r['id'])
        return z
    def decode(r,enc,settings):
        ids=ae.generate(encoder_outputs=enc,**settings)[0].tolist()
        pred=tok.decode(ids,skip_special_tokens=True).strip()
        return dict(text=pred,generated_ids=ids,eos=tok.eos_token_id in ids,at_cap=len(ids)>=MAX_TOKENS,
            model_metrics=metrics(r['model_view'],pred),source_metrics=metrics(r['text'],pred))
    def evaluate(r,z,greedy):
        labels=torch.tensor([r['ids']],device='cuda')
        enc=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z))
        answer=ae(encoder_outputs=enc,labels=labels)
        if not torch.isfinite(answer.loss): fail('G2','non-finite loss '+r['id'])
        ce=float(answer.loss); correct=int((answer.logits.argmax(-1)==labels).sum()); del answer
        out=dict(ce=ce,nll=ce*len(r['ids']),tokens=len(r['ids']),correct_tokens=correct,beam=decode(r,enc,BEAM))
        if greedy: out['greedy']=decode(r,enc,GREEDY)
        return out

    start=time.monotonic()
    with torch.inference_mode():
        # G1: the anchor must reproduce the stored 22 October correct-arm outputs exactly.
        stored=[json.loads(s) for s in (STAGE1/'examples.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        assert [s['id'] for s in stored]==[r['id'] for r in panels['anchor']]
        mismatched=[]
        for r,s in zip(panels['anchor'],stored):
            enc=BaseModelOutput(last_hidden_state=ae.get_decoder_input(encode(r)))
            ids=ae.generate(encoder_outputs=enc,**BEAM)[0].tolist()
            if ids!=s['correct']['generated_ids']: mismatched.append(r['id'])
            with (OUT/'examples-anchor.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(dict(id=r['id'],generated_ids=ids,reproduced=ids==s['correct']['generated_ids']))+'\n')
        if mismatched: fail('G1',dict(mismatched=len(mismatched),ids=mismatched))
        print(json.dumps(dict(event='anchor_reproduced',n=len(stored),seconds=time.monotonic()-start)),flush=True)

        total=len(panels['T'])+len(panels['V']); done=0; panel_start=time.monotonic()
        for name in ('T','V'):
            panel=panels[name]; latents=[encode(r).cpu() for r in panel]
            for i,r in enumerate(panel):
                j=donors[name][i]
                row=dict(panel=name,id=r['id'],document_id=r['document_id'],bin=r['bin'],source=r['text'],model_view=r['model_view'],full_tokens=r['full_tokens'],
                    donor_id=panel[j]['id'],correct=evaluate(r,latents[i].cuda(),True),shuffled=evaluate(r,latents[j].cuda(),False))
                with (OUT/f'examples-{name}.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                done+=1
                if torch.cuda.max_memory_allocated()>MEMORY_CAP: fail('G2','peak allocation above 10 GiB')
                if done==32:
                    projected=(time.monotonic()-panel_start)/done*total+(panel_start-start)
                    print(json.dumps(dict(event='projection',done=done,projected_seconds=projected)),flush=True)
                    if projected>GPU_SECONDS_CAP: fail('G2',f'projected runtime {projected:.0f} s exceeds {GPU_SECONDS_CAP} s')
                if done%32==0: print(json.dumps(dict(done=done,total=total,seconds=time.monotonic()-start)),flush=True)
    save('runtime.json',dict(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,strict_load=True,chrf_signature=str(chrf.get_signature())))
    print('INFERENCE_COMPLETE',flush=True)

if __name__=='__main__': main()
