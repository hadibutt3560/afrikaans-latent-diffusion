"""Step 1 only: immutable validation reconstruction diagnostic."""
import os, sys, json, hashlib, random, time, re
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'outputs/longrun-local-inputs'
OUT=ROOT/'outputs/reconstruction-audit-20261022'
AE=ROOT/'outputs/cluster-results/afrikaans-56531/best'
DATA=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl'
sys.path.insert(0,str(BUNDLE))
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf8')
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=OUT/name
    with p.open('x',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def main():
    import torch, transformers, sacrebleu, numpy as np
    from transformers import AutoTokenizer
    from transformers.modeling_outputs import BaseModelOutput
    from sacrebleu.metrics import CHRF
    from tools.longrun_training import load_ae
    torch.set_num_threads(4)
    torch.manual_seed(91427)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    assert digest(AE/'model.pt')=='123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'
    assert digest(DATA)=='bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e'
    original=read(BUNDLE/'inputs.json')
    for p,h in original['code'].items(): assert digest(BUNDLE/p)==h,p
    tok=AutoTokenizer.from_pretrained(BUNDLE/'models/mt5-base',local_files_only=True)
    rows=[json.loads(s) for s in DATA.read_text(encoding='utf8').splitlines() if s.strip()]
    groups=defaultdict(list)
    appendix=[]
    for r in rows:
        ids=tok(r['text'],truncation=False)['input_ids']
        r['full_tokens']=len(ids)
        r['bin']='short' if len(ids)<=32 else 'medium' if len(ids)<=64 else 'long'
        r['ids']=tok(r['text'],truncation=True,max_length=128)['input_ids']
        r['model_view']=tok.decode(r['ids'],skip_special_tokens=True)
        groups[r['document_id']].append(r)
        if len(ids)>128: appendix.append(r)
    rng=random.Random(91427)
    docs=sorted(groups); rng.shuffle(docs)
    def select(docids,cap):
        result=[]
        for d in docids:
            candidates=groups[d].copy(); rng.shuffle(candidates)
            picked=[]; bins=set()
            for r in candidates:
                if r['bin'] not in bins and len(picked)<cap: picked.append(r); bins.add(r['bin'])
            for r in candidates:
                if len(picked)<cap and r not in picked: picked.append(r)
            result.extend(picked)
        return result
    selected=select(docs[:64],2)
    reserved=select(docs[64:128],4)
    donors={}
    for label in ('short','medium','long'):
        ix=[i for i,r in enumerate(selected) if r['bin']==label]
        assert len(ix)>1
        for attempt in range(100000):
            perm=ix.copy(); rng.shuffle(perm)
            if all(selected[i]['document_id']!=selected[j]['document_id'] for i,j in zip(ix,perm)): break
        else: raise ValueError('No cross-document derangement')
        donors.update(zip(ix,perm))
    manifest=dict(selected=selected,donor_indices=donors,reserved=reserved,truncation_appendix=appendix)
    save('manifest.json',manifest)
    settings=dict(max_length=128,min_length=5,num_beams=4,do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2,length_penalty=1.0)
    save('contract.json',dict(ae_sha256=digest(AE/'model.pt'),data_sha256=digest(DATA),protocol_sha256=digest(HERE/'PROTOCOL.md'),script_sha256=digest(__file__),manifest_sha256=digest(OUT/'manifest.json'),source_code=original['code'],packages=dict(torch=torch.__version__,transformers=transformers.__version__,sacrebleu=sacrebleu.__version__),gpu=torch.cuda.get_device_name(),precision='FP32',settings=settings,selected=len(selected),documents=64,reserved=len(reserved)))
    print(json.dumps(dict(event='manifest_frozen',n=len(selected),bins=dict(Counter(r['bin'] for r in selected)))),flush=True)
    ae,loaded_tok=load_ae(AE,BUNDLE/'models/mt5-base',dict(latent_dim=64,latent_length=32,max_length=128))
    assert loaded_tok.get_vocab()==tok.get_vocab()
    start=time.monotonic()
    chrf=CHRF(char_order=6,word_order=0,beta=2)
    def metrics(ref,pred):
        a,b=Counter(re.findall(r'\d+(?:[.,]\d+)*',ref)),Counter(re.findall(r'\d+(?:[.,]\d+)*',pred))
        return dict(exact=int(ref==pred),similarity=SequenceMatcher(None,ref,pred,autojunk=False).ratio(),chrf=chrf.sentence_score(pred,[ref]).score,number_expected=sum(a.values()),number_predicted=sum(b.values()),number_matched=sum((a&b).values()))
    def encode(r):
        x=torch.tensor([r['ids']],device='cuda'); mask=torch.ones_like(x)
        z=ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
        assert torch.isfinite(z).all()
        return z
    def evaluate(r,z):
        labels=torch.tensor([r['ids']],device='cuda')
        enc=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z))
        answer=ae(encoder_outputs=enc,labels=labels)
        assert torch.isfinite(answer.loss)
        ce=float(answer.loss); correct=int((answer.logits.argmax(-1)==labels).sum()); del answer
        ids=ae.generate(encoder_outputs=enc,**settings)[0].tolist()
        pred=tok.decode(ids,skip_special_tokens=True).strip()
        return dict(text=pred,ce=ce,nll=ce*len(r['ids']),tokens=len(r['ids']),correct_tokens=correct,generated_ids=ids,eos=tok.eos_token_id in ids,at_cap=len(ids)>=128,model_metrics=metrics(r['model_view'],pred),source_metrics=metrics(r['text'],pred))
    results=[]
    with torch.inference_mode():
        latents=[encode(r).cpu() for r in selected]
        for i,r in enumerate(selected):
            row=dict(id=r['id'],document_id=r['document_id'],bin=r['bin'],source=r['text'],model_view=r['model_view'],full_tokens=r['full_tokens'],fertility=(r['full_tokens']-1)/max(1,len(r['text'].split())),donor_id=selected[donors[i]]['id'],correct=evaluate(r,latents[i].cuda()),shuffled=evaluate(r,latents[donors[i]].cuda()))
            results.append(row)
            with (OUT/'examples.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if (i+1)%16==0:
                elapsed=time.monotonic()-start
                print(json.dumps(dict(done=i+1,total=len(selected),seconds=elapsed)),flush=True)
                assert elapsed/(i+1)*len(selected)<7200
                assert torch.cuda.max_memory_allocated()<10*1024**3
        extra=[dict(id=r['id'],source=r['text'],model_view=r['model_view'],full_tokens=r['full_tokens'],correct=evaluate(r,encode(r))) for r in appendix]
    save('truncation-appendix.json',extra)
    save('runtime.json',dict(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,strict_load=True,chrf_signature=str(chrf.get_signature())))
    print('INFERENCE_COMPLETE',flush=True)
if __name__=='__main__': main()
