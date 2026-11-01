"""Frozen shared-latent generation and pinned mGPT evaluation; no training."""
from collections import Counter
import csv
import gc
import math
import random
import time
import unicodedata

import engine as e

COUNT=256
SEED=20261024

def normalized(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())

def pooled_ppl(rows):
    scored=[r['mgpt'] for r in rows if r.get('mgpt')]
    tokens=sum(r['tokens'] for r in scored)
    return math.exp(sum(r['nll'] for r in scored)/tokens) if tokens else None

def metrics(rows,train):
    words=[r['text'].split() for r in rows];norm=[normalized(r['text']) for r in rows]
    result=dict(n=len(rows),ppl=pooled_ppl(rows),scored=sum(bool(r.get('mgpt')) for r in rows),
        mean_words=sum(map(len,words))/len(rows),empty=sum(not t for t in norm),
        duplicate_excess=len(norm)-len(set(norm)),train_matches=sum(t in train for t in norm),
        eos=sum(r['eos'] for r in rows),cap=sum(r['length_cap'] for r in rows),sentinel=sum(r['sentinel'] for r in rows))
    for n in (1,2,3,4):
        counts=[Counter(tuple(w[i:i+n]) for i in range(max(0,len(w)-n+1))) for w in words]
        total=Counter()
        for c in counts:total.update(c)
        result[f'distinct_{n}']=len(total)/sum(total.values()) if total else 0.
        if n>=3:result[f'mean_repeat_{n}']=sum(1-len(c)/sum(c.values()) if c else 0. for c in counts)/len(rows)
    result['length_bands']={}
    for lo,hi in ((1,10),(11,20),(21,35),(36,1000000)):
        subset=[r for r in rows if lo<=len(r['text'].split())<=hi]
        result['length_bands'][f'{lo}-{hi}']=dict(n=len(subset),ppl=pooled_ppl(subset))
    return result

def transfer_gate(old,new):
    n=new['n']
    return bool(new['empty']==0 and new['scored']==n and old['scored']==old['n']
        and new['cap']/n<=old['cap']/old['n']+.02
        and new['train_matches']/n<=old['train_matches']/old['n']+.02
        and new['duplicate_excess']/n<=old['duplicate_excess']/old['n']+.02
        and new['mean_repeat_3']<=old['mean_repeat_3']+.005
        and new['distinct_2']>=old['distinct_2']*.95
        and .9*old['mean_words']<=new['mean_words']<=1.1*old['mean_words']
        and old['ppl'] is not None and new['ppl'] is not None and new['ppl']<=old['ppl']*1.05)

def paired_intervals(old,new,np):
    assert [r['seed'] for r in old]==[r['seed'] for r in new]
    def array(rows):return np.array([[r['mgpt']['nll'],r['mgpt']['tokens']] if r.get('mgpt') else [0.,0.] for r in rows])
    a,b=array(old),array(new);rng=np.random.default_rng(20261025);ratios=[];lengths=[]
    delta=np.array([len(y['text'].split())-len(x['text'].split()) for x,y in zip(old,new)])
    for _ in range(100):
        ix=rng.integers(len(old),size=(100,len(old)));aa=a[ix].sum(1);bb=b[ix].sum(1)
        ok=(aa[:,1]>0)&(bb[:,1]>0)
        ratios.extend(np.exp(bb[ok,0]/bb[ok,1]-aa[ok,0]/aa[ok,1]).tolist())
        lengths.extend(delta[ix].mean(1).tolist())
    return dict(candidate_over_B0_ppl=np.percentile(ratios,[2.5,97.5]).tolist() if ratios else None,
                candidate_minus_B0_mean_words=np.percentile(lengths,[2.5,97.5]).tolist(),replicates=10000,
                interpretation='Paired generated-latent resampling; conditional on one AE, denoiser and training seed; no training-seed uncertainty. Empty outputs remain counted and fail the gate; PPL uses scored tokens only.')

class Transfer:
    def __init__(self,exp):
        self.exp=exp;self.torch=exp.torch;self.folder=exp.out/'generation'
        self.folder.mkdir(exist_ok=False)
        self.config=e.read(e.ROOT/'evaluation/diffusion-config.json')
        self.contract=e.read(e.ROOT/'evaluation/contract.json')
        self.latents=[]

    def diffusion(self):
        from tools.longrun_training import make_model
        saved=self.exp.load(e.ROOT/'evaluation/small-seed42-best.pt')
        assert saved['contract']==self.config
        model=make_model(self.config['config'],'small')
        model.load_state_dict(saved['ema'],strict=True)
        return model.eval().requires_grad_(False)

    def scorer(self):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        path=e.ROOT/'evaluation/mgpt'
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=self.torch.float32).cuda().eval().requires_grad_(False)
        return model,tokenizer

    def sample(self,model,seed):
        from tools.longrun_training import isolated_rng,seed_all
        self.exp.guard()
        with isolated_rng(seed),self.torch.no_grad():
            seed_all(seed)
            z,_=model.sample(1,self.torch.tensor([32],device='cuda'))
        assert z.shape==(1,32,64) and self.torch.isfinite(z).all()
        return z.detach().cpu()

    def decode(self,z,seed):
        from transformers.modeling_outputs import BaseModelOutput
        self.exp.guard()
        self.exp.ae.eval()
        with self.torch.no_grad():
            enc=BaseModelOutput(last_hidden_state=self.exp.ae.get_decoder_input(z.cuda()))
            ids=self.exp.ae.generate(encoder_outputs=enc,**e.SETTINGS)[0].tolist()
        endings=[i for i,t in enumerate(ids[1:],1) if t==self.exp.tokenizer.eos_token_id]
        n=endings[0]+1 if endings else len(ids)
        ids=ids[:n];raw=self.exp.tokenizer.decode(ids,skip_special_tokens=False)
        return dict(seed=seed,text=self.exp.tokenizer.decode(ids,skip_special_tokens=True).strip(),
                    generated_ids=ids,decoder_tokens=n,eos=bool(endings),length_cap=n>=128,sentinel='<extra_id_' in raw)

    def prepare(self):
        """Exercise all later GPU dependencies before admitting training."""
        from tools.compare_afrikaans_generators import score_text
        start=time.monotonic();model=self.diffusion();sampling=[];decoding=[];probes=[]
        self.exp.restore(self.exp.union)
        for i in range(2):
            t=time.monotonic();z=self.sample(model,SEED+i);sampling.append(time.monotonic()-t);self.latents.append(z)
            t=time.monotonic();probes.append(self.decode(z,SEED+i));decoding.append(time.monotonic()-t)
        del model;gc.collect();self.torch.cuda.empty_cache()
        for arm in ('D0','D1'):
            for i,z in enumerate(self.latents):
                assert self.decode(z,SEED+i)['generated_ids']==probes[i]['generated_ids'],'Preflight identity decode failed: '+arm
        model,tok=self.scorer();anchors=[]
        for r in e.read(e.ROOT/'evaluation/scorer-anchors.json'):
            self.exp.guard();score=score_text(model,tok,r['text'],'cuda');old=r['mgpt']
            assert score['tokens']==old['tokens'] and abs(score['nll']-old['nll'])/max(1,abs(old['nll']))<1e-4,'mGPT anchor mismatch'
            anchors.append(dict(seed=r['seed'],old=old,new=score))
        times=[]
        for r in probes:
            t=time.monotonic();r['mgpt']=score_text(model,tok,r['text'],'cuda');times.append(time.monotonic()-t)
        del model,tok;gc.collect();self.torch.cuda.empty_cache()
        self.exp.save(self.folder/'preflight-latents.pt',self.latents)
        prediction=COUNT*(max(sampling)+3*max(decoding)+3*max(times))
        reserve=max(10800.,prediction*2+1200.)
        result=dict(status='passed',scorer_anchors=anchors,seconds=time.monotonic()-start,
                    predicted_transfer_seconds=prediction,reserve_seconds=reserve,contract=self.contract,
                    sampling_seconds=sampling,decoding_seconds=decoding,scoring_seconds=times)
        e.write(self.folder/'preflight.json',result)
        e.event('evaluation_preflight_passed',seconds=result['seconds'],reserved_hours=reserve/3600)
        return reserve

    def run(self,candidate,count=COUNT,smoke=False):
        from tools.compare_afrikaans_generators import score_text
        start=time.monotonic();model=self.diffusion()
        while len(self.latents)<count:
            self.latents.append(self.sample(model,SEED+len(self.latents)))
            if len(self.latents)%16==0:
                self.exp.save(self.folder/'shared-latents.pt',self.latents)
                e.event('generation_latents',done=len(self.latents),total=count)
        self.exp.save(self.folder/'shared-latents.pt',self.latents[:count])
        del model;gc.collect();self.torch.cuda.empty_cache()
        outputs={}
        for arm in (('B0',candidate) if smoke else ('B0','D0',candidate)):
            self.exp.restore(self.exp.union if arm=='B0' or smoke else self.exp.load(self.exp.best[arm]['file'])['model'])
            rows=[]
            for i,z in enumerate(self.latents[:count]):
                rows.append(self.decode(z,SEED+i))
                if (i+1)%16==0:e.write(self.folder/f'{arm}-samples.json',rows)
            outputs[arm]=rows;e.write(self.folder/f'{arm}-samples.json',rows)
        if smoke:
            assert all(a['generated_ids']==b['generated_ids'] for a,b in zip(outputs['B0'],outputs[candidate])),'Identity decode smoke failed'
        # Free decoder GPU allocation during external scoring.
        self.exp.ae.cpu();gc.collect();self.torch.cuda.empty_cache()
        model,tok=self.scorer()
        for arm,rows in outputs.items():
            for i,r in enumerate(rows):
                self.exp.guard();r['mgpt']=score_text(model,tok,r['text'],'cuda')
                if (i+1)%32==0:
                    e.write(self.folder/f'{arm}-scored.json',rows[:i+1]);e.event('external_scoring',arm=arm,done=i+1,total=count)
            e.write(self.folder/f'{arm}-scored.json',rows)
        del model,tok;gc.collect();self.torch.cuda.empty_cache()
        train={normalized(r['text']) for r in self.exp.rows}
        summary={a:metrics(rs,train) for a,rs in outputs.items()}
        self.review(outputs,candidate,train)
        control_result={}
        if not smoke:
            original_folder=self.folder
            self.folder=original_folder/'control-comparison';self.folder.mkdir()
            self.review(outputs,candidate,train,baseline='D0')
            self.folder=original_folder
            control_result=dict(candidate_over_control_intervals=paired_intervals(outputs['D0'],outputs[candidate],self.exp.np),
                control_guardrails_passed=transfer_gate(summary['D0'],summary[candidate]))
        result=dict(**control_result,status='completed',smoke_only=smoke,candidate=candidate,n=count,metrics=summary,
                    intervals=paired_intervals(outputs['B0'],outputs[candidate],self.exp.np),
                    automatic_transfer_guardrails_passed=transfer_gate(summary['B0'],summary[candidate]),
                    researcher_review='pending; automatic checks do not establish linguistic quality',
                    hypothesis_established=False,seconds=time.monotonic()-start,contract=self.contract)
        e.write(self.folder/'summary.json',result)
        e.event('generation_evaluation_finished',count=count,guardrails=result['automatic_transfer_guardrails_passed'])
        return result

    def review(self,outputs,candidate,train,baseline='B0'):
        rng=random.Random(20261025);sheet=[];key=[];flagged=[]
        for i,(old,new) in enumerate(zip(outputs[baseline],outputs[candidate])):
            if i<64:
                flip=rng.choice((False,True));texts=[old['text'],new['text']]
                if flip:texts.reverse()
                sheet.append(dict(item=i+1,seed=old['seed'],A=texts[0],B=texts[1],
                    A_clause_structure='',B_clause_structure='',A_coherence='',B_coherence='',
                    A_lexical_corruption='',B_lexical_corruption='',preference='',notes=''))
                key.append(dict(item=i+1,A=candidate if flip else baseline,B=baseline if flip else candidate))
            for arm,r in ((baseline,old),(candidate,new)):
                norm=normalized(r['text']);reasons=[]
                if not norm:reasons.append('empty')
                if r['length_cap']:reasons.append('length_cap')
                if r['sentinel']:reasons.append('sentinel')
                if norm in train:reasons.append('exact_train_match')
                if sum(normalized(x['text'])==norm for x in outputs[arm])>1:reasons.append('duplicate')
                words=r['text'].split();grams=list(zip(words,words[1:],words[2:]))
                if len(set(grams))<len(grams):reasons.append('repeated_trigram')
                if reasons:flagged.append(dict(arm=arm,seed=r['seed'],reasons=','.join(reasons),text=r['text']))
        rng.shuffle(sheet)
        with (self.folder/'researcher-generation-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sheet[0]));w.writeheader();w.writerows(sheet)
        e.write(self.folder/'review-key.json',key);e.write(self.folder/'flagged-samples.json',flagged)

def should_transfer(summary):
    confirmation=summary.get('confirmation')
    return bool(summary.get('candidate') and isinstance(confirmation,dict)
                and confirmation.get('automatic_reconstruction_gate_passed'))
