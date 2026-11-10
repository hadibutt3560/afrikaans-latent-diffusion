"""Diffusion latents against simpler latent priors, Phase A (PROTOCOL.md, approved 30 October 2026).

Inference only. `--dry-run` builds the C1-C3 latents, the H selection and the C2 record and prints
them without loading B0 or mGPT or writing anything; the full run writes to a new outputs directory.
Analysis is a separate CPU step: `analyze.py <output directory>`.
"""
import os, sys, json, hashlib, random, time, re, argparse, datetime, math
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'outputs/longrun-local-inputs'
CACHE=BUNDLE/'cache'
AE=ROOT/'outputs/cluster-results/afrikaans-56531/best'
TRAIN=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'
VALID=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl'
STAGE1=ROOT/'outputs/reconstruction-audit-20261022'
REFERENCE=ROOT/'outputs/longrun-local-evaluation/validation-reference-mgpt.json'
RUNS=ROOT/'outputs/longrun-local-evaluation/runs'
MGPT_DEFAULT=Path.home()/'.cache/huggingface/hub/models--ai-forever--mGPT/snapshots/40897bd7c8b47a76802c411108ca6220438b8b40'
# SHA-256 of the approved PROTOCOL.md with line endings normalised to LF.
PROTOCOL_SHA='14319d16a373e1d358ace2f779f4f1181260bbb3f3a6cd0033978c184c088994'
AE_SHA='123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'
TRAIN_SHA='2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab'
VALID_SHA='bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e'
REFERENCE_SHA='51cca0b60c0d8ece2a7b6e0f39b98177cf60726d36e3defe9f97d9dbd8d52498'
CACHE_MANIFEST_SHA='84febf0732cbb5e6b0cb85a607600b97c37cf1108e1137d7de55248ff795b541'
INPUTS_SHA='d7f2aeb8f348ad360f39d5d11b2e08fc4ec64667d66f0861a75b1fd1cdba3772'
STAGE1_EXAMPLES_SHA='f0ea392ab0cf17c06c531f00131fdc80ac3b1c6d2f530b7a2866409105a21d3b'
MGPT_SHA='c45c522411e64dbed6da1ed0ce0839bf03931d3a17b80182d06d4dab8d8b2fbb'
MGPT_REVISION='40897bd7c8b47a76802c411108ca6220438b8b40'
BANK_SHA={  # PROTOCOL.md Appendix A
    'small-seed42':'ef98bc5d50b63ae04317471d66f1468b42967f13c623e434357a5e77fe978e74',
    'small-seed43':'777cf891ad04e9b6dc0901b9084f1d8ef3bfd88643b1cd4aefaf8f7da5b9fa42',
    'small-seed44':'f038c2e44fe6eeef90b52f41ad37a1f99018dbbbaa931cb75c93a5feba262743',
    'small-seed45':'40ea866a625c8766fd1b98ba445bb46bbfabaec4ba1a0b8cc014dd79c5494099',
    'small-seed46':'7dd1b769cbffd209f33fe32242eabe2a450abdb5ef6ce0b97052820cdca4a281',
    'published-seed42':'cb3fb2d8d50199a5144d4463c9b91457de8aada604f4983d41cce2c1c153e8a2',
    'published-seed43':'1c26a7b6a74f7cd0cfed7c0ef0e629d91bfec73942c748efc540ba32f1205cc9',
    'published-seed44':'699fce97c8a5e16960f90bd934ea2b270c1e81a9276a86406120825275e791e4',
    'published-seed45':'cf06811aa709413330eef9d5dc8730c814c0ea0fea888c20db71b63aaa86a661',
    'published-seed46':'69d15ffa310178f707d71245993da1b40404f7a32fddc45d41c904353528225d'}
SEED_C1, SEED_C2, SEED_C3 = 30000, 40000, 50000
SEED_VALID_DOCS=91427   # same call and seed as the 22 October audit
N_CONTROL=1000
SLOTS, DIM, NORM = 32, 64, 8.0
MAX_TOKENS=128
H_ROWS, H_DOCS = 811, 58
REFERENCE_BAND_COUNTS=[140,223,349,288]   # gives the fixed weights 0.140, 0.223, 0.349, 0.288
BANDS=[(1,10),(11,20),(21,35),(36,10**9)]
BEAM=dict(max_length=128,min_length=5,num_beams=4,do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2,length_penalty=1.0)
NORM_TOL=1e-4
SCORE_TOL=1e-4
GPU_SECONDS_CAP=4*3600
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
def band(words): return next(i for i,(a,b) in enumerate(BANDS) if a<=words<=b)
def log(event,**fields): print(json.dumps(dict(event=event,**fields),ensure_ascii=False),flush=True)

def check_inputs():
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools')); from as_run import as_run_sha256, as_run_matches  # protocols are checked as they were at tag as-run
    assert as_run_sha256(HERE/'PROTOCOL.md')==PROTOCOL_SHA, 'PROTOCOL.md at tag as-run differs from the approved version'
    for p,h in ((TRAIN,TRAIN_SHA),(VALID,VALID_SHA),(REFERENCE,REFERENCE_SHA),(CACHE/'manifest.json',CACHE_MANIFEST_SHA),
                (BUNDLE/'inputs.json',INPUTS_SHA),(STAGE1/'examples.jsonl',STAGE1_EXAMPLES_SHA)):
        assert digest(p)==h, p
    for run,h in BANK_SHA.items(): assert digest(RUNS/run/'validation-mgpt.json')==h, run
    original=read(BUNDLE/'inputs.json')
    for p,h in original['code'].items(): assert digest(BUNDLE/p)==h, p
    return original

def train_latents(torch):
    from tools.longrun_training import load_checkpoint
    manifest=read(CACHE/'manifest.json')
    parts=[]
    for shard in manifest['splits']['train']['shards']:
        assert digest(CACHE/shard['file'])==shard['sha256'], shard['file']
        parts.append(load_checkpoint(CACHE/shard['file'])['latents'].float())
    z=torch.cat(parts)
    assert list(z.shape)==[manifest['splits']['train']['records'],SLOTS,DIM]==[47767,SLOTS,DIM]
    assert torch.isfinite(z).all()
    return z

def ledoit_wolf(x, chunk=4096):
    """Ledoit-Wolf (2004) shrinkage towards (trace(S)/p)I, closed-form intensity, float64.
    Same formula as scikit-learn's ledoit_wolf_shrinkage with assume_centered=False."""
    n,p=x.shape
    mu=x.mean(0)
    s=np.zeros((p,p)); beta_=0.
    for i in range(0,n,chunk):
        c=x[i:i+chunk]-mu
        s+=c.T@c
        c2=c*c
        beta_+=float((c2.T@c2).sum())
    s/=n
    m=np.trace(s)/p
    delta_=float((s*s).sum())        # = sum((X^T X)^2) / n^2
    beta=(beta_/n-delta_)/(p*n)
    delta=(delta_-2*m*np.trace(s)+p*m*m)/p
    beta=min(beta,delta)
    shrinkage=0. if beta==0 else beta/delta
    cov=(1-shrinkage)*s
    cov[np.diag_indices(p)]+=shrinkage*m
    return mu,cov,shrinkage,m

def project(z):
    """Scale each 64-dimensional slot to norm 8."""
    return z/np.linalg.norm(z,axis=-1,keepdims=True)*NORM

def slot_norms(z): n=np.linalg.norm(z,axis=-1); return dict(mean=float(n.mean()),sd=float(n.std(ddof=1)),min=float(n.min()),max=float(n.max()))

def build_controls(torch, latents):
    F=torch.nn.functional
    c1=[]
    for i in range(N_CONTROL):
        g=torch.Generator(device='cpu').manual_seed(SEED_C1+i)
        c1.append(F.normalize(torch.randn(1,SLOTS,DIM,generator=g),dim=-1)[0]*NORM)
    c1=torch.stack(c1)
    x=latents.reshape(len(latents),SLOTS*DIM).double().numpy()
    mu,cov,shrinkage,target=ledoit_wolf(x)
    chol=np.linalg.cholesky(cov)
    pre=np.stack([mu+chol@np.random.Generator(np.random.PCG64(SEED_C2+i)).standard_normal(SLOTS*DIM) for i in range(N_CONTROL)])
    post=project(pre.reshape(N_CONTROL,SLOTS,DIM))
    def moments(flat):
        return dict(mean_distance=float(np.linalg.norm(flat.mean(0)-mu)),
                    covariance_relative_frobenius=float(np.linalg.norm(np.cov(flat,rowvar=False)-cov)/np.linalg.norm(cov)))
    record=dict(estimator='Ledoit-Wolf 2004, target (trace(S)/p)I, closed-form intensity, float64, NumPy',
                shrinkage_intensity=shrinkage,target_scale=target,n_training=len(x),dimensions=SLOTS*DIM,
                training_slot_norms=slot_norms(x.reshape(-1,SLOTS,DIM)),c2_slot_norms_after_projection=slot_norms(post),
                c2_after_projection=moments(post.reshape(N_CONTROL,-1)),
                supplementary_c2_before_projection=dict(slot_norms=slot_norms(pre.reshape(N_CONTROL,SLOTS,DIM)),**moments(pre)),
                note='Covariance differences from 1,000 draws in 2,048 dimensions include sampling error; the before-projection value is the reference for that error.')
    c2=torch.from_numpy(post).float()
    donors=[np.random.Generator(np.random.PCG64(SEED_C3+i)).integers(0,len(latents),size=SLOTS).tolist() for i in range(N_CONTROL)]
    slots=torch.arange(SLOTS)
    c3=torch.stack([latents[torch.tensor(d),slots] for d in donors])
    for name,z in (('C1',c1),('C2',c2),('C3',c3)):
        assert list(z.shape)==[N_CONTROL,SLOTS,DIM] and torch.isfinite(z).all(), name
        dev=float((z.norm(dim=-1)-NORM).abs().max())
        assert dev<=NORM_TOL, f'G1: {name} slot norms deviate from 8 by {dev}'
    return dict(C1=c1,C2=c2,C3=c3),donors,record

def select_reference(tok):
    valid=rows_of(VALID)
    docs=sorted({r['document_id'] for r in valid}); random.Random(SEED_VALID_DOCS).shuffle(docs)
    reserve=set(docs[64:128])
    assert reserve=={r['document_id'] for r in read(STAGE1/'manifest.json')['reserved']}, 'Reserve documents are not positions 64-127'
    ref=read(REFERENCE)['rows']
    assert len(ref)==1000
    assert [sum(band(len(r['text'].split()))==k for r in ref) for k in range(len(BANDS))]==REFERENCE_BAND_COUNTS
    kept=[r for r in ref if r['document_id'] not in reserve]
    assert len(kept)==H_ROWS and len({r['document_id'] for r in kept})==H_DOCS
    h,excluded=[],[]
    for r in kept:
        full=tok(r['text'],truncation=False)['input_ids']
        if len(full)>MAX_TOKENS: excluded.append(dict(id=r['id'],full_tokens=len(full))); continue
        ids=tok(r['text'],truncation=True,max_length=MAX_TOKENS)['input_ids']
        h.append(dict(id=r['id'],document_id=r['document_id'],text=r['text'],full_tokens=len(full),
                      model_view=tok.decode(ids,skip_special_tokens=True),mgpt=r['mgpt']))
    anchors=kept[:3]   # G2(b): the first three non-reserve reference rows
    return h,excluded,anchors,reserve

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--mgpt',type=Path,default=MGPT_DEFAULT)
    args=ap.parse_args()
    original=check_inputs()
    sys.path.insert(0,str(BUNDLE))
    import torch
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BUNDLE/'models/mt5-base',local_files_only=True)
    latents=train_latents(torch)
    controls,donors,c2_record=build_controls(torch,latents)
    h,excluded,anchors,reserve=select_reference(tok)
    sizes=dict(C1=N_CONTROL,C2=N_CONTROL,C3=N_CONTROL,H=len(h),R=len(h),H_documents=len({r['document_id'] for r in h}),
               H_excluded_over_128=len(excluded),H_band_counts=[sum(band(len(r['text'].split()))==k for r in h) for k in range(len(BANDS))],
               H_largest_document=max(Counter(r['document_id'] for r in h).values()),
               C3_outputs_with_repeated_donor=sum(len(set(d))<SLOTS for d in donors))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,sizes=sizes,c2_record=c2_record),indent=2)); return

    import transformers, sacrebleu
    from transformers import AutoModelForCausalLM
    from transformers.modeling_outputs import BaseModelOutput
    from sacrebleu.metrics import CHRF
    from tools.longrun_training import load_ae
    from tools.compare_afrikaans_generators import score_text
    torch.set_num_threads(4)
    torch.manual_seed(20261030)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    assert digest(AE/'model.pt')==AE_SHA
    assert digest(args.mgpt/'pytorch_model.bin')==MGPT_SHA, 'mGPT weights differ from the pinned revision'
    OUT=ROOT/f"outputs/latent-controls-{datetime.date.today():%Y%m%d}"
    OUT.mkdir(exist_ok=False)
    def save(name,obj):
        with (OUT/name).open('x',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    def append(name,row):
        with (OUT/name).open('a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    def fail(gate,detail):
        save('failure.json',dict(gate=gate,detail=detail)); raise SystemExit(f'{gate} failed: {detail}')
    latent_sha={}
    for name,z in controls.items():
        path=OUT/f'latents-{name}.pt'
        assert not path.exists()
        torch.save(z,path); latent_sha[name]=digest(path)
    save('manifest.json',dict(sizes=sizes,reserve_documents=sorted(reserve),H=[{k:r[k] for k in ('id','document_id','full_tokens')} for r in h],
        H_excluded=excluded,C3_donors=donors,anchors_mgpt=[r['id'] for r in anchors],
        seeds=dict(C1=SEED_C1,C2=SEED_C2,C3=SEED_C3,valid_docs=SEED_VALID_DOCS),reference_band_counts=REFERENCE_BAND_COUNTS))
    save('c2-record.json',c2_record)
    analyzer=HERE/'analyze.py'
    save('contract.json',dict(protocol_sha256_lf=PROTOCOL_SHA,protocol_sha256_raw=digest(HERE/'PROTOCOL.md'),script_sha256=digest(__file__),
        analysis_script_sha256=digest(analyzer) if analyzer.exists() else None,manifest_sha256=digest(OUT/'manifest.json'),
        c2_record_sha256=digest(OUT/'c2-record.json'),latent_sha256=latent_sha,ae_sha256=AE_SHA,train_sha256=TRAIN_SHA,valid_sha256=VALID_SHA,
        reference_sha256=REFERENCE_SHA,cache_manifest_sha256=CACHE_MANIFEST_SHA,bank_sha256=BANK_SHA,mgpt=dict(revision=MGPT_REVISION,weights_sha256=MGPT_SHA),
        source_code=original['code'],packages=dict(torch=torch.__version__,transformers=transformers.__version__,sacrebleu=sacrebleu.__version__,numpy=np.__version__),
        gpu=torch.cuda.get_device_name(),precision='FP32',beam=BEAM,python=sys.version))
    log('manifest_frozen',sizes=sizes)

    start=time.monotonic()
    def check_memory():
        if torch.cuda.max_memory_allocated()>MEMORY_CAP: fail('G3','peak allocation above 10 GiB')
    def load_mgpt():
        mtok=AutoTokenizer.from_pretrained(args.mgpt,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(args.mgpt,local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
        return model,mtok

    # G2(b) and (c): the scorer reproduces stored scores. Also gives the scoring rate for the G3 projection.
    model,mtok=load_mgpt(); checks=[]
    bank=read(RUNS/'small-seed42/validation-mgpt.json')['rows'][:20]
    score_start=time.monotonic()
    for kind,rows in (('reference',anchors),('small-seed42',bank)):
        for r in rows:
            value=score_text(model,mtok,r['text'],'cuda')
            ok=value is not None and value['tokens']==r['mgpt']['tokens'] and abs(value['nll']-r['mgpt']['nll'])/max(1.,abs(r['mgpt']['nll']))<SCORE_TOL
            checks.append(dict(kind=kind,id=r.get('id',r.get('seed')),stored=r['mgpt'],recomputed=value,ok=ok))
    score_rate=(time.monotonic()-score_start)/len(checks)
    save('checks-mgpt.json',checks)
    if not all(c['ok'] for c in checks): fail('G2',dict(part='b/c',failed=[c['id'] for c in checks if not c['ok']]))
    del model; torch.cuda.empty_cache()
    log('scorer_reproduced',n=len(checks),seconds_per_row=score_rate)

    ae,loaded_tok=load_ae(AE,BUNDLE/'models/mt5-base',dict(latent_dim=DIM,latent_length=SLOTS,max_length=MAX_TOKENS))
    assert loaded_tok.get_vocab()==tok.get_vocab()
    chrf=CHRF(char_order=6,word_order=0,beta=2)
    def metrics(ref,pred):
        a,b=Counter(re.findall(r'\d+(?:[.,]\d+)*',ref)),Counter(re.findall(r'\d+(?:[.,]\d+)*',pred))
        return dict(exact=int(ref==pred),similarity=SequenceMatcher(None,ref,pred,autojunk=False).ratio(),chrf=chrf.sentence_score(pred,[ref]).score,
            number_expected=sum(a.values()),number_predicted=sum(b.values()),number_matched=sum((a&b).values()))
    def decode(z):
        """Benchmark decoding, identical to generate() in the execution source."""
        if not torch.isfinite(z).all(): fail('G1','non-finite latent')
        ids=ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z)),**BEAM)[0]
        eos=(ids[1:]==tok.eos_token_id).nonzero(as_tuple=True)[0]
        n=int(eos[0])+2 if len(eos) else len(ids)
        raw=tok.decode(ids[:n],skip_special_tokens=False)
        return dict(text=tok.decode(ids[:n],skip_special_tokens=True).strip(),decoder_tokens=n,eos=bool(len(eos)),
                    length_cap=n>=MAX_TOKENS,sentinel='<extra_id_' in raw,generated_ids=ids.tolist())

    remaining_scores=3*N_CONTROL+len(h)
    def project_runtime(done_here,left_here,later,rate):
        projected=(time.monotonic()-start)+rate*(left_here+later)+score_rate*remaining_scores
        log('projection',done=done_here,projected_seconds=projected)
        if projected>GPU_SECONDS_CAP: fail('G3',f'projected runtime {projected:.0f} s exceeds {GPU_SECONDS_CAP} s')

    with torch.inference_mode():
        # G2(a): B0 reproduces the stored 22 October correct-arm outputs exactly, through the 22 October path.
        stage1=read(STAGE1/'manifest.json')['selected']
        stored=rows_of(STAGE1/'examples.jsonl')
        assert [s['id'] for s in stored]==[r['id'] for r in stage1]
        mismatched=[]
        for r,s in zip(stage1,stored):
            ids=tok(r['text'],truncation=True,max_length=MAX_TOKENS)['input_ids']
            assert ids==r['ids'], 'Anchor tokenisation differs from 22 October'
            x=torch.tensor([ids],device='cuda'); mask=torch.ones_like(x)
            z=ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
            out=ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z)),**BEAM)[0].tolist()
            if out!=s['correct']['generated_ids']: mismatched.append(r['id'])
            append('anchor-b0.jsonl',dict(id=r['id'],generated_ids=out,reproduced=out==s['correct']['generated_ids']))
        if mismatched: fail('G2',dict(part='a',mismatched=len(mismatched),ids=mismatched))
        log('anchor_reproduced',n=len(stored),seconds=time.monotonic()-start)

        order=['C1','C2','C3']
        for k,name in enumerate(order):
            began=time.monotonic(); later=N_CONTROL*(len(order)-k-1)+len(h)
            for i in range(N_CONTROL):
                row=dict(condition=name,index=i,**decode(controls[name][i:i+1].cuda()))
                if name=='C1': row['seed']=SEED_C1+i
                if name=='C2': row['seed']=SEED_C2+i
                if name=='C3': row.update(seed=SEED_C3+i,donors=donors[i])
                append(f'decodes-{name}.jsonl',row); check_memory()
                if i+1==32: project_runtime(i+1,N_CONTROL-32,later,(time.monotonic()-began)/32)
            log('decoded',condition=name,seconds=time.monotonic()-start)

        began=time.monotonic()
        for i,r in enumerate(h):
            # Same encoding path as the latent cache (prepare_cache), at batch one.
            batch=tok([r['text']],return_tensors='pt',padding='max_length',truncation=True,max_length=MAX_TOKENS).to('cuda')
            z=ae.get_diffusion_latent(ae.get_encoder()(**batch),batch.attention_mask).float()
            out=decode(z)
            append('decodes-R.jsonl',dict(condition='R',id=r['id'],document_id=r['document_id'],source=r['text'],model_view=r['model_view'],
                full_tokens=r['full_tokens'],**out,model_metrics=metrics(r['model_view'],out['text']),source_metrics=metrics(r['text'],out['text'])))
            check_memory()
            if i+1==32: project_runtime(i+1,len(h)-32,0,(time.monotonic()-began)/32)
        log('decoded',condition='R',seconds=time.monotonic()-start)
        del ae; torch.cuda.empty_cache()

        model,mtok=load_mgpt()
        for name in order+['R']:
            for row in rows_of(OUT/f'decodes-{name}.jsonl'):
                row['mgpt']=score_text(model,mtok,row['text'],'cuda') if row['text'] else None
                append(f'scored-{name}.jsonl',row); check_memory()
            log('scored',condition=name,seconds=time.monotonic()-start)
    save('runtime.json',dict(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,strict_load=True,
        scorer_seconds_per_row=score_rate,chrf_signature=str(chrf.get_signature())))
    print('INFERENCE_COMPLETE',flush=True)

if __name__=='__main__': main()
