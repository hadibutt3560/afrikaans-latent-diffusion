"""One-factor reconstruction test; no training or test data."""
import os, sys, json, hashlib, time, re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BUNDLE=ROOT/'outputs/longrun-local-inputs'
PARENT=ROOT/'outputs/reconstruction-audit-20261022'
OUT=ROOT/'outputs/stage3-decoder-20261022'
AE=ROOT/'outputs/cluster-results/afrikaans-56531/best'
sys.path.insert(0,str(BUNDLE))
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf8')
def read(p): return json.loads(p.read_text(encoding='utf8'))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    import torch, transformers, sacrebleu, numpy as np
    from transformers.modeling_outputs import BaseModelOutput
    from sacrebleu.metrics import CHRF
    from tools.longrun_training import load_ae
    torch.set_num_threads(4); torch.manual_seed(91427)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    parent=read(PARENT/'contract.json'); manifest=read(PARENT/'manifest.json')
    previous=[json.loads(s) for s in (PARENT/'examples.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(previous)==len(manifest['selected'])==81
    assert digest(AE/'model.pt')==parent['ae_sha256']
    assert digest(ROOT/'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl')==parent['data_sha256']
    assert digest(PARENT/'manifest.json')==parent['manifest_sha256']
    for p,h in parent['source_code'].items(): assert digest(BUNDLE/p)==h,p
    original=read(BUNDLE/'inputs.json')
    for p,h in original['model_cache'].items(): assert digest(BUNDLE/'models/mt5-base'/p)==h,p
    OUT.mkdir(exist_ok=False)
    baseline=parent['settings']; treatment={**baseline,'no_repeat_ngram_size':0}
    save('contract.json',dict(protocol_sha256=digest(HERE/'PROTOCOL.md'),script_sha256=digest(__file__),parent_contract_sha256=digest(PARENT/'contract.json'),parent_examples_sha256=digest(PARENT/'examples.jsonl'),parent_manifest_sha256=digest(PARENT/'manifest.json'),ae_sha256=parent['ae_sha256'],baseline=baseline,treatment=treatment,packages=dict(torch=torch.__version__,transformers=transformers.__version__,sacrebleu=sacrebleu.__version__),gpu=torch.cuda.get_device_name(),precision='FP32'))
    assert transformers.__version__==parent['packages']['transformers'] and sacrebleu.__version__==parent['packages']['sacrebleu']
    ae,tok=load_ae(AE,BUNDLE/'models/mt5-base',dict(latent_dim=64,latent_length=32,max_length=128))
    metric=CHRF(char_order=6,word_order=0,beta=2)
    def score(ref,ids):
        text=tok.decode(ids,skip_special_tokens=True).strip()
        a,b=Counter(re.findall(r'\d+(?:[.,]\d+)*',ref)),Counter(re.findall(r'\d+(?:[.,]\d+)*',text))
        words=text.split()
        rep={str(n):sum(v-1 for v in Counter(tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))).values()) for n in (3,4)}
        return dict(text=text,generated_ids=ids,chrf=metric.sentence_score(text,[ref]).score,exact=int(ref==text),similarity=SequenceMatcher(None,ref,text,autojunk=False).ratio(),number_expected=sum(a.values()),number_predicted=sum(b.values()),number_matched=sum((a&b).values()),eos=1 in ids,cap=len(ids)>=128,repeated_word_ngrams=rep)
    start=time.monotonic(); results=[]
    with torch.inference_mode():
        for i,(r,old) in enumerate(zip(manifest['selected'],previous)):
            assert r['id']==old['id']
            x=torch.tensor([r['ids']],device='cuda'); mask=torch.ones_like(x)
            z=ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
            decoded=ae.get_decoder_input(z); assert torch.isfinite(decoded).all()
            arms={}
            for name,settings in [('baseline',baseline),('treatment',treatment)]:
                ids=ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=decoded),**settings)[0].tolist()
                arms[name]=score(r['model_view'],ids)
            assert arms['baseline']['generated_ids']==old['correct']['generated_ids'],f'Baseline mismatch at {i}'
            results.append(dict(index=i,id=r['id'],document_id=r['document_id'],source=r['text'],model_view=r['model_view'],**arms))
            with (OUT/'pairs.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(results[-1],ensure_ascii=False,allow_nan=False)+'\n')
            if (i+1)%16==0:
                elapsed=time.monotonic()-start
                print(json.dumps(dict(done=i+1,total=81,seconds=elapsed)),flush=True)
                assert elapsed/(i+1)*81<7200 and torch.cuda.max_memory_allocated()<10*1024**3
    docs=sorted({r['document_id'] for r in results})
    stats=np.array([[sum(r['treatment']['chrf']-r['baseline']['chrf'] for r in results if r['document_id']==d),sum(r['document_id']==d for r in results)] for d in docs])
    rng=np.random.default_rng(91428); boot=stats[rng.integers(0,len(docs),(10000,len(docs)))].sum(1)
    interval=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
    arms={}
    for arm in ('baseline','treatment'):
        items=[r[arm] for r in results]
        expected=sum(x['number_expected'] for x in items); matched=sum(x['number_matched'] for x in items); pred=sum(x['number_predicted'] for x in items)
        arms[arm]=dict(mean_chrf=float(np.mean([x['chrf'] for x in items])),exact_count=sum(x['exact'] for x in items),mean_similarity=float(np.mean([x['similarity'] for x in items])),number_expected=expected,number_matched=matched,number_predicted=pred,number_recall=matched/expected,number_precision=matched/pred if pred else None,eos_count=sum(x['eos'] for x in items),cap_count=sum(x['cap'] for x in items),repeated_word_trigrams=sum(x['repeated_word_ngrams']['3'] for x in items),repeated_word_fourgrams=sum(x['repeated_word_ngrams']['4'] for x in items))
    delta=arms['treatment']['mean_chrf']-arms['baseline']['mean_chrf']
    gate=delta>=2 and interval[0]>0 and arms['treatment']['number_recall']>=arms['baseline']['number_recall']-.02
    save('analysis.json',dict(arms=arms,mean_chrf_difference=delta,paired_document_bootstrap_95=interval,automatic_gate_pass=gate,content_gate='requires AI-assisted inspection; no human assessment',changed_indices=[r['index'] for r in results if r['baseline']['generated_ids']!=r['treatment']['generated_ids']],count=81,documents=len(docs),baseline_all_token_sequences_reproduced=True,seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,chrf_signature=str(metric.get_signature())))
    print(json.dumps(dict(status='complete',delta=delta,interval=interval,automatic_gate=gate)),flush=True)
if __name__=='__main__': main()
