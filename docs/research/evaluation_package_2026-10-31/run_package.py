"""Runner for the evaluation package (PROTOCOL.md, approved 1 November 2026). Never trains or reads the test split or the reserve.

Stages run in the protocol's order (section 11). Every stage resumes from its saved files; gate failures are written to status.json."""
import argparse, hashlib, importlib.util, json, math, os, random, re, subprocess, sys, time, unicodedata, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import analyze_package as A
OUT=A.OUT
E=Path('E:/afrikaans-eval-20261031')
BUNDLE=ROOT/'outputs/longrun-local-inputs'
DATA=ROOT/'data/nchlt_afrikaans/processed/stage_5_7'
PHASE_A=A.PHASE_A
REFERENCE=ROOT/'outputs/longrun-local-evaluation/validation-reference-mgpt.json'
RUNS_DIR=ROOT/'outputs/longrun-local-evaluation/runs'
AR_DIR=ROOT/'outputs/cluster-results/ar-baseline-57759'
AR_HIST=ROOT/'outputs/ar-validation-audit-20261022/ar-cap127-scored.jsonl'
AR_HIST_SAMPLES=ROOT/'outputs/ar-validation-audit-20261022/ar-cap127-samples.jsonl'
MGPT=Path.home()/'.cache/huggingface/hub/models--ai-forever--mGPT/snapshots/40897bd7c8b47a76802c411108ca6220438b8b40'
HASHES={  # section 2 and Appendix A
    DATA/'train.jsonl':'2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab',
    DATA/'valid.jsonl':'bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e',
    REFERENCE:'51cca0b60c0d8ece2a7b6e0f39b98177cf60726d36e3defe9f97d9dbd8d52498',
    AR_DIR/'best_model/model.safetensors':'ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076',
    AR_DIR/'best_model/tokenizer.json':'b5d0e18b86936c25d0717f6717f0c87a0576aaa261e4dc1818ac723ca2da021a',
    AR_HIST_SAMPLES:'1fb2a4f732d1beb6c5bd11c3a7ab1ffda680d8c4a13067e08508a904d59f1229',
    AR_HIST:'e0232e7d575288d0e14edb9ae7817376dc49e8056b5bb369cc97b55b4d80667d',
    MGPT/'pytorch_model.bin':'c45c522411e64dbed6da1ed0ce0839bf03931d3a17b80182d06d4dab8d8b2fbb',
    **{PHASE_A/f:h for f,h in {
        'decodes-C1.jsonl':'6450fafbf3d08828923d2be9ee4da455fad88af7f23b8ccba74c6d0cedc16313','decodes-C2.jsonl':'db9189183732a79011d9fe950a3b81be93a7b7a05d98b0126760eb1083a4f00f',
        'decodes-C3.jsonl':'786291cfef5c13eb281f368db9996890d2aa5809f21f1b183b0cc0a53a3eedc9','decodes-R.jsonl':'2f103d415ae2acf5d875225cb5b2d78a05c530fee4672dc87e47ad544c39b5cd',
        'scored-C1.jsonl':'88fb70d6139b98bfaabf58a0d11269a4531e138e0db305819cd03239796533b4','scored-C2.jsonl':'b0679a76672895e41f40ce2621e0b7df8a5696248cc3df90ee8ed4583fe3111d',
        'scored-C3.jsonl':'ce6cdae265aee2df8d4d5469fe9d85aae0bb17981ecf350245cd3efa4b34c292','scored-R.jsonl':'cc4d29396a09857baf5e2dbae6ecd3a3071d17e355961f88783d9bdf780d5cfa',
        'manifest.json':'a171e638639e0a50cb754b014032d149fe29122243d00e476c4c7d1db2153d82','analysis.json':'f2b706fb9ac51bc69716572b84311901573ef6563b14cb02955f59f335007404'}.items()}}
GRID=[(t,p,c) for t in (0.8,0.9,1.0) for p in (0.90,0.95,1.00) for c in ('off','on')]   # section 8 grid order
def cell_name(t,p,c): return f't{t:.1f}-p{p:.2f}-{c}'
HIST_CELL=cell_name(1.0,0.95,'off')
MEMORY_CAP=10*1024**3
PORT=8099

os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf-8')
read,rows_of=A.read,A.rows_of

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def lf_digest(p): return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def log(event,**k): print(json.dumps(dict(event=event,time=time.strftime('%H:%M:%S'),**k),ensure_ascii=False,default=str),flush=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=1,ensure_ascii=False,default=float)+'\n',encoding='utf8')
def write_rows(name,rows): (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
def append(path,row):
    with open(path,'a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
def done(name): return (OUT/name).exists()

class GateFailure(Exception): pass
def fail(gate,detail):
    status=read(OUT/'status.json') if done('status.json') else {}
    status[gate]=dict(failed=True,detail=detail); save('status.json',status); raise GateFailure(f'{gate}: {detail}')
def note(key,value):
    status=read(OUT/'status.json') if done('status.json') else {}
    status[key]=value; save('status.json',status)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

load_train=A.load_train

# ---------------------------------------------------------------- setup and G1
def setup():
    if not OUT.exists(): OUT.mkdir(parents=True,exist_ok=False)
    import importlib.metadata as md, torch
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    pins=read(E/'pins.json')
    contract=dict(protocol_sha256_lf=lf_digest(HERE/'PROTOCOL.md'),runner_sha256=digest(__file__),analysis_sha256=digest(HERE/'analyze_package.py'),
        pins_sha256=digest(E/'pins.json'),inputs={str(p.relative_to(ROOT)):h for p,h in HASHES.items() if ROOT in p.parents},
        mgpt_weights=HASHES[MGPT/'pytorch_model.bin'],bank_sha256=read(PHASE_A/'contract.json')['bank_sha256'],
        packages={d.metadata['Name']:d.version for d in md.distributions()},python=sys.version,gpu=torch.cuda.get_device_name(),
        test_accessed=False,reserve_accessed=False)
    if done('contract.json'):
        if read(OUT/'contract.json')!=contract: raise RuntimeError('Contract changed; use a new output directory')
        return
    for p,h in HASHES.items():
        if digest(p)!=h: fail('G1',f'hash mismatch {p}')
    for run,h in contract['bank_sha256'].items():
        if digest(RUNS_DIR/run/'validation-mgpt.json')!=h: fail('G1',f'bank hash {run}')
    for key,group in pins.items():
        base=Path(group['path']) if 'path' in group else E/'llama'/group['tag'] if 'tag' in group else E/key
        for f,h in group['files'].items():
            p=base/f
            if digest(p)!=h: fail('G1',f'pinned file mismatch {p}')
    save('contract.json',contract); log('G1 passed')

# ---------------------------------------------------------------- banks
def reserve_and_hl():
    valid=rows_of(DATA/'valid.jsonl')
    docs=sorted({r['document_id'] for r in valid}); random.Random(91427).shuffle(docs); reserve=set(docs[64:128])
    assert reserve==set(read(PHASE_A/'manifest.json')['reserve_documents']),'reserve differs from Phase A'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BUNDLE/'models/mt5-base',local_files_only=True)
    hl=[dict(id=r['id'],document_id=r['document_id'],text=r['text']) for r in valid if r['document_id'] not in reserve]
    assert len(hl)==4827
    kept=[r for r in hl if len(tok(r['text'],truncation=False)['input_ids'])<=128]
    return reserve,kept,len(hl)-len(kept)

def build_existing_banks():
    if done('bank-HL.jsonl'): return
    reserve,hl,excluded=reserve_and_hl()
    ref={r['id']:r for r in read(REFERENCE)['rows']}
    h=[dict(id=x['id'],document_id=ref[x['id']]['document_id'],text=ref[x['id']]['text'],mgpt=ref[x['id']]['mgpt']) for x in read(PHASE_A/'manifest.json')['H']]
    assert len(h)==811 and not {r['document_id'] for r in h}&reserve
    write_rows('bank-H.jsonl',h)
    write_rows('bank-R.jsonl',[dict(id=r['id'],document_id=r['document_id'],text=r['text'],mgpt=r['mgpt']) for r in rows_of(PHASE_A/'scored-R.jsonl')])
    for c in ('C1','C2','C3'): write_rows(f'bank-{c}.jsonl',[dict(index=r['index'],seed=r.get('seed'),text=r['text'],mgpt=r['mgpt']) for r in rows_of(PHASE_A/f'scored-{c}.jsonl')])
    for run in A.RUNS: write_rows(f'bank-{run}.jsonl',[dict(seed=r['seed'],text=r['text'],mgpt=r['mgpt']) for r in read(RUNS_DIR/run/'validation-mgpt.json')['rows']])
    write_rows('bank-AR-hist.jsonl',[dict(seed=r['seed'],text=r['text'],mgpt=r['mgpt']) for r in rows_of(AR_HIST)])
    write_rows('bank-HL.jsonl',hl)
    save('manifest.json',dict(HL=len(hl),HL_excluded_over_128=excluded,HL_documents=len({r['document_id'] for r in hl}),
        HL_largest_document=max(sum(r['document_id']==d for r in hl) for d in {r['document_id'] for r in hl}),reserve_documents=sorted(reserve)))
    log('banks built',HL=len(hl),excluded=excluded)

def banks_present(): return A.bank_files()

# ---------------------------------------------------------------- scorers
def score_module():
    sys.path.insert(0,str(BUNDLE)); from tools.compare_afrikaans_generators import score_text; return score_text

def load_scorer(kind):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path=MGPT if kind=='mgpt' else Path(read(E/'pins.json')['goldfish']['path'])
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    if kind=='goldfish': assert tok.bos_token=='[CLS]', f'Goldfish BOS is {tok.bos_token}'
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32).cuda().eval().requires_grad_(False)
    return model,tok

def check_memory(torch,where):
    if torch.cuda.max_memory_allocated()>MEMORY_CAP: fail('G3',f'peak allocation above 10 GiB at {where}')

def score_file(kind,model,tok,src_rows,path,limit=None):
    """Appends {'<kind>': score} per row to path, resuming."""
    score_text=score_module(); have=len(rows_of(path)) if path.exists() else 0
    for r in src_rows[have:limit]:
        append(path,{kind:score_text(model,tok,r['text'],'cuda') if r['text'].strip() else None})

# ---------------------------------------------------------------- AR generation
_AR=None
def ar():
    global _AR
    if _AR is None:
        m=load_module('run_audit',ROOT/'docs/research/ar_validation_audit_2026-10-22/run_audit.py'); tok,model,_=m.load_ar(); _AR=(tok,model)
    return _AR

def generate(cell,seeds,path,limit=None):
    import torch
    tok,model=ar(); t,p,c=next(g for g in GRID if cell_name(*g)==cell)
    have=rows_of(path) if path.exists() else []
    assert [r['seed'] for r in have]==list(seeds[:len(have)])
    extra=dict(min_length=5,repetition_penalty=1.2,no_repeat_ngram_size=3) if c=='on' else dict(min_new_tokens=0,repetition_penalty=1.,no_repeat_ngram_size=0)
    with torch.inference_mode():
        for seed in list(seeds)[len(have):limit]:
            torch.manual_seed(seed); random.seed(seed)
            ids=torch.tensor([[0]],device='cuda')
            tokens=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=True,top_k=0,top_p=p,temperature=t,num_beams=1,
                max_new_tokens=127,use_cache=True,pad_token_id=0,bos_token_id=0,eos_token_id=1,**extra)[0].tolist()
            raw=tok.decode(tokens[1:],skip_special_tokens=False)
            append(path,dict(seed=seed,token_ids=tokens,text=tok.decode(tokens[1:],skip_special_tokens=True).strip(),decoder_tokens=len(tokens),
                generated_tokens=len(tokens)-1,eos=1 in tokens[1:],length_cap=len(tokens)-1>=127,sentinel='<extra_id_' in raw))

# ---------------------------------------------------------------- G2
def gate_g2():
    if done('checks-g2.json'): return
    import torch
    cell=OUT/f'ar-grid-{HIST_CELL}.jsonl'
    generate(HIST_CELL,range(30000,31000),cell,limit=256)
    hist=rows_of(AR_HIST_SAMPLES); new=rows_of(cell)[:256]
    a_ok=[h['token_ids']==n['token_ids'] for h,n in zip(hist,new)]
    if not all(a_ok): fail('G2',dict(part='a',mismatched_seeds=[h['seed'] for h,o in zip(hist,a_ok) if not o][:20]))
    model,tok=load_scorer('mgpt'); score_text=score_module(); b=[]
    for r in rows_of(AR_HIST)[:20]:
        v=score_text(model,tok,r['text'],'cuda'); b.append(v['tokens']==r['mgpt']['tokens'] and abs(v['nll']-r['mgpt']['nll'])/max(1.,abs(r['mgpt']['nll']))<1e-4)
    del model; torch.cuda.empty_cache()
    if not all(b): fail('G2',dict(part='b',failed=[i for i,o in enumerate(b) if not o]))
    model,tok=load_scorer('goldfish'); h=rows_of(OUT/'bank-H.jsonl')
    s1=[score_text(model,tok,r['text'],'cuda') for r in h[:20]]; s2=[score_text(model,tok,r['text'],'cuda') for r in h[:20]]
    if s1!=s2: fail('G2',dict(part='c'))
    rt=[' '.join(tok.decode(tok.encode(r['text'],add_special_tokens=False),skip_special_tokens=True).split())==' '.join(r['text'].split()) for r in h[:100]]
    del model; torch.cuda.empty_cache()
    if sum(rt)<99: fail('G2',dict(part='d',round_trip=sum(rt)))
    save('checks-g2.json',dict(a=256,b=20,c=20,d=sum(rt))); log('G2 passed')

# ---------------------------------------------------------------- G3 budget pilot
def gate_g3():
    if done('budget-pilot.json'): return
    import torch
    torch.cuda.reset_peak_memory_stats(); t0=time.monotonic()
    first,last=cell_name(*GRID[0]),cell_name(*GRID[-1])
    for c in (first,last): generate(c,range(30000,31000),OUT/f'ar-grid-{c}.jsonl',limit=50)
    gen=(time.monotonic()-t0)/100
    pilot=rows_of(OUT/f'ar-grid-{first}.jsonl')[:50]+rows_of(OUT/f'ar-grid-{last}.jsonl')[:50]
    rate={}
    for kind in ('mgpt','goldfish'):
        model,tok=load_scorer(kind); score_text=score_module(); t=time.monotonic()
        for r in pilot: score_text(model,tok,r['text'],'cuda') if r['text'].strip() else None
        rate[kind]=(time.monotonic()-t)/len(pilot)
        if kind=='goldfish':
            t=time.monotonic()
            for r in read(RUNS_DIR/'small-seed42/validation-mgpt.json')['rows'][:100]: score_text(model,tok,r['text'],'cuda')
            rate['goldfish_bank']=(time.monotonic()-t)/100
        check_memory(torch,f'pilot {kind}'); del model; torch.cuda.empty_cache()
    st=load_labse(); hl=rows_of(OUT/'bank-HL.jsonl'); t=time.monotonic(); st.encode([r['text'] for r in hl[:100]],batch_size=32,normalize_embeddings=True)
    rate['labse']=(time.monotonic()-t)/100; del st; torch.cuda.empty_cache()
    n_existing=811+811+3000+10000+256
    projection=dict(generation=gen*(19000-100),mgpt=rate['mgpt']*19000,goldfish=rate['goldfish_bank']*n_existing+rate['goldfish']*19000,labse=rate['labse']*(len(hl)+15000))
    total=sum(projection.values())
    save('budget-pilot.json',dict(per_item=dict(generation=gen,**rate),projection_seconds=projection,total_hours=total/3600,peak_gib=torch.cuda.max_memory_allocated()/2**30))
    log('G3',hours=round(total/3600,2))
    if total>6*3600: fail('G3',f'projected {total/3600:.2f} h above 6 h')

def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(read(E/'pins.json')['labse']['path'],device='cuda')

# ---------------------------------------------------------------- Part A and B rows
_lid=_dic=None
def part_a_rows(name):
    global _lid,_dic
    if done(f'dict-{name}.jsonl'): return
    import fasttext
    from spylls.hunspell import Dictionary
    if _lid is None:
        _lid=fasttext.load_model(str(Path(read(E/'pins.json')['glotlid']['path'])/'model.bin'))
        _dic=(Dictionary.from_files(str(E/'dict/af_ZA')),{})
    dic,cache=_dic
    def ok(s):
        if s not in cache: cache[s]=bool(dic.lookup(s))
        return cache[s]
    lid,dct=[],[]
    for r in rows_of(OUT/f'bank-{name}.jsonl'):
        text=' '.join(r['text'].split())
        if text:
            labels,probs=_lid.predict(text,k=-1,threshold=0.0); p=dict(zip(labels,(float(x) for x in probs)))
            lid.append(dict(top=labels[0],p_top=float(probs[0]),p_afr=p.get('__label__afr_Latn',0.),p_nld=p.get('__label__nld_Latn',0.)))
        else: lid.append(dict(top=None))
        words,acr,acr_bad,rep=[],0,0,0
        for s,kind in A.dictionary_tokens(r['text']):
            if kind=='article': continue
            look=s.replace('’',"'"); rep+=look!=s
            if kind=='acronym': acr+=1; acr_bad+=not ok(look); continue
            words.append((s,ok(look)))
        dct.append(dict(checked=len(words),unrecognised=sum(not o for _,o in words),acronyms=acr,acronyms_unrecognised=acr_bad,apostrophe_replacements=rep,words=words))
    write_rows(f'lid-{name}.jsonl',lid); write_rows(f'dict-{name}.jsonl',dct)

def goldfish_rows(names):
    import torch
    todo=[n for n in names if not done(f'goldfish-{n}.jsonl') or len(rows_of(OUT/f'goldfish-{n}.jsonl'))<len(rows_of(OUT/f'bank-{n}.jsonl'))]
    if not todo: return
    model,tok=load_scorer('goldfish')
    for n in todo: score_file('goldfish',model,tok,rows_of(OUT/f'bank-{n}.jsonl'),OUT/f'goldfish-{n}.jsonl'); log('goldfish',bank=n)
    check_memory(torch,'goldfish'); del model; torch.cuda.empty_cache()

# ---------------------------------------------------------------- Part D
def make_ar_bank(name,src):
    if done(f'bank-{name}.jsonl'): return
    gen=rows_of(OUT/f'{src}.jsonl'); sc=rows_of(OUT/f'{src}-mgpt.jsonl'); assert len(gen)==len(sc)==1000
    write_rows(f'bank-{name}.jsonl',[dict(seed=g['seed'],text=g['text'],eos=g['eos'],length_cap=g['length_cap'],sentinel=g['sentinel'],mgpt=s['mgpt']) for g,s in zip(gen,sc)])

def part_d(train):
    import torch
    cells=[cell_name(*g) for g in GRID]
    for c in cells: generate(c,range(30000,31000),OUT/f'ar-grid-{c}.jsonl'); log('generated',cell=c)
    model,tok=load_scorer('mgpt')
    for c in cells: score_file('mgpt',model,tok,rows_of(OUT/f'ar-grid-{c}.jsonl'),OUT/f'ar-grid-{c}-mgpt.jsonl')
    check_memory(torch,'mgpt grid'); del model; torch.cuda.empty_cache()
    for c in cells: make_ar_bank(f'AR-grid-{c}',f'ar-grid-{c}'); part_a_rows(f'AR-grid-{c}')
    goldfish_rows([f'AR-grid-{c}' for c in cells])
    if not done('ar-tuned-selection.json'):
        runs={r:A.diversity(rows_of(OUT/f'bank-{r}.jsonl'),train) for r in A.RUNS}
        floors=dict(distinct2=min(v['distinct2'] for v in runs.values()),entropy=min(v['unigram_entropy_bits'] for v in runs.values()),copies=max(v['training_copies'] for v in runs.values()))
        table=[]
        for c in cells:
            rows=rows_of(OUT/f'bank-AR-grid-{c}.jsonl'); d=A.diversity(rows,train); p=A.ppl(rows,'mgpt')
            table.append(dict(cell=c,mgpt_std=p['standardised'],support=p['support'],distinct2=d['distinct2'],entropy=d['unigram_entropy_bits'],copies=d['training_copies'],
                eligible=p['support']=='ok' and d['distinct2']>=floors['distinct2'] and d['unigram_entropy_bits']>=floors['entropy'] and d['training_copies']<=floors['copies']))
        el=[x for x in table if x['eligible']]
        tuned=min(el,key=lambda x:(x['mgpt_std'],cells.index(x['cell'])))['cell'] if el else None
        save('ar-tuned-selection.json',dict(floors=floors,table=table,tuned=tuned)); log('tuned AR',cell=tuned)
    tuned=read(OUT/'ar-tuned-selection.json')['tuned']
    if tuned is None: note('part_d','no eligible cell; no tuned-AR comparison'); return
    generate(tuned,range(31000,32000),OUT/'ar-fresh.jsonl')
    model,tok=load_scorer('mgpt'); score_file('mgpt',model,tok,rows_of(OUT/'ar-fresh.jsonl'),OUT/'ar-fresh-mgpt.jsonl'); del model; torch.cuda.empty_cache()
    make_ar_bank('AR-fresh','ar-fresh'); part_a_rows('AR-fresh'); goldfish_rows(['AR-fresh'])

# ---------------------------------------------------------------- Part E
def part_e():
    if done('mauve.json'): return
    import torch, mauve
    st=load_labse(); emb={}
    def embed(name):
        p=OUT/f'emb-{name}.npy'
        if not p.exists(): np.save(p,st.encode([r['text'] for r in rows_of(OUT/f'bank-{name}.jsonl')],batch_size=32,normalize_embeddings=True,convert_to_numpy=True))
        return np.load(p)
    hl=rows_of(OUT/'bank-HL.jsonl'); H=embed('HL')
    by=defaultdict(list)
    for i,r in enumerate(hl): by[r['document_id']].append(i)
    docs=sorted(by)
    def m(p,q,seed): return float(mauve.compute_mauve(p_features=p,q_features=q,seed=seed,verbose=False,device_id=-1).mauve)
    e1=[]
    for r in range(10):
        rng=random.Random(20261110+r); d=docs[:]; rng.shuffle(d); a=[i for x in d[:len(d)//2] for i in by[x]]; b=[i for x in d[len(d)//2:] for i in by[x]]
        n=min(1000,len(a),len(b)); e1.append(dict(n=n,mauve=m(H[rng.sample(a,n)],H[rng.sample(b,n)],1)))
    refs=[]
    for k in range(5):
        rng=np.random.default_rng(20261120+k); pool=[i for j in rng.integers(0,len(docs),len(docs)) for i in by[docs[j]]]
        idx=rng.choice(len(pool),1000,replace=False); refs.append([pool[i] for i in idx])
    small=embed('small-seed42'); c1=embed('C1')
    e2=dict(clustering=[m(small,H[refs[0]],s) for s in range(1,6)],reference=[m(small,H[refs[k]],1) for k in range(5)],c1=m(c1,H[refs[0]],1),
            reference_duplicates=[len(r)-len(set(r)) for r in refs])
    hh=[x['mauve'] for x in e1]
    gate=dict(median=float(np.median(hh))>=.90,range=max(hh)-min(hh)<=.10,clustering_sd=float(np.std(e2['clustering'],ddof=1))<=.03,
              reference_range=max(e2['reference'])-min(e2['reference'])<=.10,c1_below=e2['c1']<min(hh))
    out=dict(E1=e1,E2=e2,gate=gate,passed=all(gate.values()))
    if out['passed']:
        e3={}
        for name in A.RUNS+['C1','C2','C3','AR-fresh',f'AR-grid-{HIST_CELL}']:
            if not done(f'bank-{name}.jsonl'): continue
            q=embed(name); ref=[m(q,H[refs[k]],1) for k in range(5)]; clu=[m(q,H[refs[0]],s) for s in range(1,6)]
            e3[name]=dict(reference_draws=ref,reference_mean=float(np.mean(ref)),reference_range=[min(ref),max(ref)],clustering=clu,clustering_range=[min(clu),max(clu)])
        out['E3']=e3
    save('mauve.json',out); del st; torch.cuda.empty_cache(); log('MAUVE',passed=out['passed'])

# ---------------------------------------------------------------- Part F
def judge_prompt():
    text=(HERE/'PROTOCOL.md').read_text(encoding='utf8').split('## Appendix B.')[1]
    return text.split('\n---\n')[1].strip('\n')

PARSE=re.compile(r'J1\s*=\s*([01])\s+J2\s*=\s*([01])\s+J3\s*=\s*([01])\s+J4\s*=\s*([01])\s+J5\s*=\s*([01])\s+J6\s*=\s*([01])\s+J7\s*=\s*([01])')

class Judge:
    def __init__(self,name):
        pins=read(E/'pins.json'); self.name=name
        gguf=next(Path(pins[f'judge-{name}']['path']).glob('*.gguf'))
        bindir=E/'llama'/pins['llama.cpp']['tag']/'bin'
        if not (bindir/'llama-server.exe').exists():
            import zipfile
            for z in (E/'llama'/pins['llama.cpp']['tag']).glob('*.zip'): zipfile.ZipFile(z).extractall(bindir)
        self.base=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
        server=str(bindir/'llama-server.exe'); helptext=subprocess.run([server,'--help'],capture_output=True,text=True,errors='replace').stdout
        flags=[f for f in (['--reasoning-budget','0'],['--no-webui']) if f[0] in helptext]
        note(f'judge_{name}_server_flags',[x for f in flags for x in f])
        self.proc=subprocess.Popen([server,'-m',str(gguf),'-c','2048','-ngl','999','--host','127.0.0.1','--port',str(PORT),
            '-np','1']+[x for f in flags for x in f],stdout=open(OUT/f'llama-{name}.log','w'),stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                if urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2).status==200: break
            except Exception: time.sleep(1)
        else: raise RuntimeError('llama-server did not start')
        self.peak=0; self.prompt=judge_prompt()
    def rate(self,text):
        body=dict(messages=[dict(role='user',content=self.prompt.replace('{TEXT}',text))],temperature=0,max_tokens=64,seed=0,
                  chat_template_kwargs=dict(enable_thinking=False))
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        reply=json.loads(urllib.request.urlopen(req,timeout=300).read())['choices'][0]['message'].get('content') or ''
        used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])
        self.peak=max(self.peak,used-self.base)
        for line in reply.splitlines():
            m=PARSE.search(line)
            if m: return dict(raw=reply,flags=[int(x) for x in m.groups()])
        return dict(raw=reply,flags=None)
    def close(self): self.proc.terminate(); self.proc.wait(60)

def ud_sentences(fname):
    return [l[len('# text = '):].strip() for l in (E/'ud'/fname).read_text(encoding='utf8').splitlines() if l.startswith('# text = ')]

def corruptions(hl):
    """Validation suite (section 10): two document halves, 150 passages each, six versions per passage."""
    by=defaultdict(list)
    for r in hl: by[r['document_id']].append(r)
    docs=sorted(by); random.Random(20261130).shuffle(docs); halves=dict(selection=docs[:len(docs)//2],confirmation=docs[len(docs)//2:])
    def long_words(ws): return [i for i,w in enumerate(ws) if sum(ch.isalpha() for ch in w)>=6 and pairs(w)]
    def pairs(w): return [i for i in range(1,len(w)-2) if w[i].isalpha() and w[i+1].isalpha() and w[i]!=w[i+1]]
    nl,en=ud_sentences('nl_alpino-ud-test.conllu'),ud_sentences('en_ewt-ud-test.conllu')
    suite={}; skipped={}
    for half,ds in halves.items():
        pool=[r for d in ds for r in by[d]]; random.Random(20261131).shuffle(pool)
        chosen=[]; skip=0
        for r in pool:
            ws=r['text'].split()
            if len(ws)>=11 and len(long_words(ws))>=2: chosen.append(r)
            elif len(ws)>=11: skip+=1
            if len(chosen)==150: break
        skipped[half]=skip; items=[]
        for j,r in enumerate(chosen):
            ws=r['text'].split(); n=len(ws)
            def rng(v): return random.Random(f'20261132-{r["id"]}-{v}')
            g=rng('shuffle'); s=ws[:]
            while s==ws: g.shuffle(s)
            g=rng('morphology'); m=ws[:]; lw=long_words(ws); k=max(2,math.ceil(.3*len(lw)))
            for i in g.sample(lw,min(k,len(lw))):
                w=m[i]; q=g.choice(pairs(w)); m[i]=w[:q]+w[q+1]+w[q]+w[q+2:]
            g=rng('repeat'); L=g.randint(4,min(6,n)); st=g.randint(0,n-L); rep=ws[:st+L]+ws[st:st+L]*2+ws[st+L:]
            g=rng('truncate'); cut=g.randint(math.ceil(.4*n),math.floor(.7*n)); tr=ws[:cut]; tr[-1]=tr[-1].rstrip('.,;:!?')
            g=rng('splice'); src=nl if j%2==0 else en; third=n//3
            sent=g.choice([x for x in src if len(x.split())>=third]).split()[:third]; sp=ws[:third]+sent+ws[2*third:]
            for v,t in (('clean',ws),('shuffle',s),('morphology',m),('repeat',rep),('truncate',tr),('splice',sp)):
                items.append(dict(id=r['id'],document_id=r['document_id'],version=v,splice_language=('nl' if j%2==0 else 'en') if v=='splice' else None,text=' '.join(t)))
        suite[half]=items
    return suite,skipped

TARGET=dict(shuffle=3,morphology=2,repeat=1,truncate=4,splice=5)   # J4, J3, J2, J5, J6 (0-based)
def fault(flags): return flags is not None and any(flags[:6])

def run_judge(judge,items,path):
    have=rows_of(path) if path.exists() else []
    for it in items[len(have):]: append(path,dict(**it,**judge.rate(it['text'])))
    return rows_of(path)

def gate_j(rated,c1):
    out={}; clean=[r for r in rated if r['version']=='clean']
    kc=sum(fault(r['flags']) for r in clean); out['clean_flag_rate']=kc/len(clean)
    out['detection']={v:sum(fault(r['flags']) for r in rated if r['version']==v)/sum(r['version']==v for r in rated) for v in TARGET}
    out['target_item']={v:sum(r['flags'] is not None and r['flags'][i]==1 for r in rated if r['version']==v)/sum(r['version']==v for r in rated) for v,i in TARGET.items()}
    allr=rated+c1; out['unparsed_rate']=sum(r['flags'] is None for r in allr)/len(allr)
    k1=sum(fault(r['flags']) for r in c1); out['c1_flag_rate']=k1/len(c1); out['c1_minus_clean']=A.newcombe(k1,len(c1),kc,len(clean))
    out['conditions']=dict(detection=min(out['detection'].values())>=.80,clean=out['clean_flag_rate']<=.20,unparsed=out['unparsed_rate']<=.02,c1=out['c1_minus_clean'][0]>0)
    out['passed']=all(out['conditions'].values()); out['score']=float(np.mean(list(out['detection'].values()))-out['clean_flag_rate'])
    return out

def part_f():
    if done('judge-application.json') or done('judge-selection.json') and read(OUT/'judge-selection.json').get('final')=='not reported': return
    hl=rows_of(OUT/'bank-HL.jsonl'); suite,skipped=corruptions(hl)
    c1=[dict(id=r['index'],version='C1',text=r['text']) for r in rows_of(OUT/'bank-C1.jsonl')[:150]]
    sel=read(OUT/'judge-selection.json') if done('judge-selection.json') else dict(skipped=skipped,candidates={})
    planned=1050*2+1050+5656
    for name in ('gemma','qwen'):
        if name in sel['candidates']: continue
        j=Judge(name); t=time.monotonic()
        rated=run_judge(j,suite['selection'][:20],OUT/f'judge-validation-{name}-selection.jsonl')
        per=(time.monotonic()-t)/20 if len(rated)==20 else None
        if j.peak>10*1024: j.close(); sel['candidates'][name]=dict(dropped=f'peak {j.peak} MiB above 10 GiB'); save('judge-selection.json',sel); continue
        if per and per*planned>8*3600:
            j.close(); sel['final']='not reported'; sel['reason']=f'budget: projected {per*planned/3600:.1f} h'; save('judge-selection.json',sel); note('part_f','not conducted: budget'); return
        rated=run_judge(j,suite['selection'],OUT/f'judge-validation-{name}-selection.jsonl')
        c1r=run_judge(j,c1,OUT/f'judge-validation-{name}-selection-c1.jsonl'); j.close()
        sel['candidates'][name]=dict(seconds_per_rating=per,peak_mib=j.peak,**gate_j(rated,c1r)); save('judge-selection.json',sel); log('judge selection',name=name,passed=sel['candidates'][name]['passed'])
    ok=[(v['score'],n) for n,v in sel['candidates'].items() if v.get('passed')]
    if not ok: sel['final']='not reported'; sel['reason']='no candidate passed the selection half'; save('judge-selection.json',sel); return
    chosen=sorted(ok,key=lambda x:(-x[0],x[1]!='qwen'))[0][1]   # ties go to the smaller model (Qwen3.5 9B)
    sel['chosen']=chosen; save('judge-selection.json',sel)
    j=Judge(chosen)
    rated=run_judge(j,suite['confirmation'],OUT/f'judge-validation-{chosen}-confirmation.jsonl')
    c1r=run_judge(j,c1,OUT/f'judge-validation-{chosen}-confirmation-c1.jsonl')
    sel['confirmation']=gate_j(rated,c1r); save('judge-selection.json',sel)
    if not sel['confirmation']['passed']:
        j.close(); sel['final']='not reported'; sel['reason']='chosen judge failed the confirmation half'; save('judge-selection.json',sel); return
    used={r['id'] for h in suite.values() for r in h}
    h=rows_of(OUT/'bank-H.jsonl'); hs=random.Random(20261135).sample(range(len(h)),300); hids=[h[i]['id'] for i in hs]
    rmap={r['id']:r for r in rows_of(OUT/'bank-R.jsonl')}
    hlpool=[r for r in hl if r['id'] not in used]; hlsample=random.Random(20261133).sample(hlpool,300)
    plan={'H':[h[i] for i in hs],'R':[rmap[i] for i in hids],'HL-sample':hlsample}
    for n in ['C1','C2','C3']+A.RUNS+['AR-hist','AR-fresh',f'AR-grid-{HIST_CELL}']:
        if done(f'bank-{n}.jsonl'): plan[n]=rows_of(OUT/f'bank-{n}.jsonl')[:300]
    app={}
    for n,rows in plan.items():
        rated=run_judge(j,[dict(id=r.get('id',r.get('seed',r.get('index'))),document_id=r.get('document_id'),version=n,text=r['text']) for r in rows],OUT/f'judge-{n}.jsonl')
        app[n]=summarise_judge(rated,n in ('H','HL-sample'))
    j.close(); sel['final']='reported'; save('judge-selection.json',sel)
    app['diffusion_summary']={size:dict(fault_rate=A.mean_sd([app[r]['fault_rate'] for r in runs])) for size,runs in A.SIZES.items()}
    app['cross_check']=cross_check(plan)
    save('judge-application.json',app); log('judge applied')

def summarise_judge(rated,documents):
    ok=[r for r in rated if r['flags'] is not None]; n=len(ok); k=sum(fault(r['flags']) for r in ok)
    out=dict(n=len(rated),unparsed=len(rated)-n,fault_rate=k/n if n else None,items={f'J{i+1}':sum(r['flags'][i] for r in ok)/n for i in range(7)} if n else None)
    out['interval']=A.doc_bootstrap(ok,lambda rs: sum(fault(r['flags']) for r in rs)/len(rs),20261134) if documents else A.wilson(k,n)
    return out

def cross_check(plan):
    j3_with=j3=unk=unk_j3=0
    for n in plan:
        if not done(f'dict-{n}.jsonl') or n in ('H','R','HL-sample'): continue
        d=rows_of(OUT/f'dict-{n}.jsonl'); jr=rows_of(OUT/f'judge-{n}.jsonl')
        for x,r in zip(d,jr):
            if r['flags'] is None: continue
            has=x['unrecognised']>0; f3=r['flags'][2]==1
            j3+=f3; j3_with+=f3 and has; unk+=has; unk_j3+=has and f3
    return dict(j3_flagged_with_unrecognised=j3_with/j3 if j3 else None,unrecognised_rows_j3_flagged=unk_j3/unk if unk else None,
                note='generated banks only; rows with at least one dictionary-unrecognised token')

# ---------------------------------------------------------------- manual-check sheet (section 4)
def manual_sheet():
    if done('manual-check-sheet.csv'): return
    import csv
    groups=dict(HL=['HL'],diffusion=A.RUNS,AR=['AR-hist']); rng=random.Random(20261101); items=[]
    for g,names in groups.items():
        occ=[]
        for n in names:
            for i,(b,d) in enumerate(zip(rows_of(OUT/f'bank-{n}.jsonl'),rows_of(OUT/f'dict-{n}.jsonl'))):
                occ+=[(n,i,s) for s,ok in d['words'] if not ok]
        rng.shuffle(occ); per=defaultdict(int); pick=[]
        for n,i,s in occ:
            t=unicodedata.normalize('NFC',s).casefold()
            if per[t]<2: per[t]+=1; pick.append((g,n,i,s))
            if len(pick)==50: break
        items+=pick
    rng.shuffle(items)
    texts={n:rows_of(OUT/f'bank-{n}.jsonl') for n in {x[1] for x in items}}
    with open(OUT/'manual-check-sheet.csv','w',newline='',encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['item','token','context','label'])
        for k,(g,n,i,s) in enumerate(items): w.writerow([k,s,texts[n][i]['text'],''])
    with open(OUT/'manual-check-key.csv','w',newline='',encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['item','group','bank','row'])
        for k,(g,n,i,s) in enumerate(items): w.writerow([k,g,n,i])

# ---------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    if args.dry_run:
        reserve,hl,excluded=reserve_and_hl()
        print(json.dumps(dict(HL=len(hl),excluded=excluded,documents=len({r['document_id'] for r in hl}),grid=[cell_name(*g) for g in GRID],
            prompt_chars=len(judge_prompt()),suite={k:len(v) for k,v in corruptions(hl)[0].items()},skipped=corruptions(hl)[1])))
        return
    t0=time.monotonic()
    if os.name=='nt':   # keep the PC from idle-sleeping while this process runs; released when it exits
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|0x00000001)
    try:
        setup(); build_existing_banks(); gate_g2(); gate_g3()
        existing=[n for n in banks_present() if not n.startswith('AR-grid') and n!='AR-fresh']
        for n in existing: part_a_rows(n)
        log('parts A and B rows done'); manual_sheet()
        goldfish_rows([n for n in existing if n!='HL'])
        train=A.train_sets(load_train())
        part_d(train)
    except GateFailure as e:
        log('stopped',reason=str(e)); return 1
    for name,stage in (('part_e',part_e),('part_f',part_f)):
        try: stage()
        except GateFailure as e: log('stage gate',stage=name,reason=str(e))
        except Exception as e: note(name,f'error: {type(e).__name__}: {e}'); log('stage error',stage=name,error=repr(e))
    A.main(); save('runtime.json',dict(seconds=time.monotonic()-t0)); log('finished')

if __name__=='__main__':
    sys.exit(main())
