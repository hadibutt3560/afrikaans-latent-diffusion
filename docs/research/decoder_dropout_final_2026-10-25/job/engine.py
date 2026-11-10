"""Bounded D0/D1 decoder adaptation. User-submitted job; historical inputs read-only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT / 'vendor'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
STOP = False
SETTINGS = dict(max_length=128, min_length=5, num_beams=4, do_sample=False,
                no_repeat_ngram_size=3, repetition_penalty=1.2, length_penalty=1.)


def stop_signal(*_):
    global STOP
    STOP = True


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def event(name, **kwargs):
    print(json.dumps(dict(event=name, utc=datetime.now(timezone.utc).isoformat(), **kwargs)), flush=True)


def configure_mask(ae, arm):
    ae.eval().requires_grad_(False)
    blocks = ae.decoder.block[-1:] if arm == 'D0' else ae.decoder.block[-2:]
    for block in blocks:
        block.requires_grad_(True)
    ae.decoder.final_layer_norm.requires_grad_(True)
    names = [n for n, p in ae.named_parameters() if p.requires_grad]
    allowed = ('decoder.block.11.', 'decoder.final_layer_norm.')
    if arm == 'D1':
        allowed += ('decoder.block.10.',)
    assert names and all(n.startswith(allowed) for n in names)
    assert not any(m.training for m in ae.modules())
    return names


def metrics(ref, text):
    from sacrebleu.metrics import CHRF
    source = Counter(re.findall(r'\d+(?:[.,]\d+)*', ref))
    pred = Counter(re.findall(r'\d+(?:[.,]\d+)*', text))
    words = text.split()
    grams = list(zip(words, words[1:], words[2:]))
    return dict(chrf=CHRF(char_order=6, word_order=0, beta=2).sentence_score(text, [ref]).score,
                exact=int(ref == text), numeric_source=sum(source.values()),
                numeric_pred=sum(pred.values()), numeric_match=sum((source & pred).values()),
                repeat3=1-len(set(grams))/len(grams) if grams else 0., empty=int(not text))


def aggregate(rows):
    n = len(rows)
    totals = {k: sum(r[k] for r in rows) for k in
              ('exact','numeric_source','numeric_pred','numeric_match','empty','cap')}
    return dict(n=n, chrf=sum(r['chrf'] for r in rows)/n, exact=totals['exact'],
                numeric_recall=totals['numeric_match']/max(1,totals['numeric_source']),
                numeric_precision=totals['numeric_match']/max(1,totals['numeric_pred']),
                repeat3=sum(r['repeat3'] for r in rows)/n, **{k:v for k,v in totals.items() if k!='exact'},
                ce=sum(r['nll'] for r in rows)/sum(r['tokens'] for r in rows))


def eligible(a, b):
    return (a['empty']==0 and a['cap']==0 and a['numeric_recall']>=b['numeric_recall']-.02
            and a['numeric_precision']>=b['numeric_precision']-.02 and a['repeat3']<=b['repeat3']+.005)


def dev_pass(a, b):
    return eligible(a,b) and a['chrf']>=b['chrf']+3 and a['exact']>=b['exact']+3


def choose_candidate(best, baseline):
    c, t = best['D0']['metrics'], best['D1']['metrics']
    if dev_pass(t,baseline) and t['chrf']>=c['chrf']+1:
        return 'D1'
    return 'D0' if dev_pass(c,baseline) else None


def choose_horizon(seconds1, seconds2, validation_seconds, remaining_seconds, max_steps=3000):
    # Keep two hours for reserve confirmation/export, and 25% timing headroom.
    for n in reversed(range(1000, max_steps+1, 500)):
        train = n*(seconds1+seconds2)*1.25
        total = train + 2*(n//500)*validation_seconds*1.25 + 7200
        if train<=18*3600 and total<=remaining_seconds:
            return n
    return 0


def tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


class Experiment:
    def __init__(self, args):
        import torch
        import numpy as np
        import transformers
        from tools.longrun_training import load_ae, atomic_save, load_checkpoint
        self.torch, self.np, self.save, self.load = torch, np, atomic_save, load_checkpoint
        self.args = args
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.training_seconds = 0.
        self.deadline = datetime.fromisoformat(args.deadline).timestamp()
        self.begin_limit = datetime.fromisoformat(args.start_before).timestamp()
        if not args.verify_only and time.time()>self.begin_limit:
            raise RuntimeError('Start deadline passed; do not consume the writing buffer.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required inside an allocation.')
        if not args.verify_only and 'RTX PRO 6000 Blackwell' not in torch.cuda.get_device_name():
            raise RuntimeError('This package is admitted for the checked RTX PRO 6000 Blackwell on mscluster110 only.')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261024)
        torch.cuda.manual_seed_all(20261024)
        random.seed(20261024)
        np.random.seed(20261024)
        self.ae, self.tokenizer = load_ae(Path(args.ae), Path(args.model_cache),
            dict(latent_dim=64,latent_length=32,max_length=128))
        # A tied output projection would silently invalidate mT5's checkpoint.
        assert self.ae.lm_head.weight.data_ptr()!=self.ae.shared.weight.data_ptr()
        assert len(self.ae.decoder.block)==12
        configure_mask(self.ae, 'D1')
        self.union = {n:p.detach().cpu().clone() for n,p in self.ae.named_parameters() if p.requires_grad}
        self.frozen_hash = self.hash_frozen()
        self.rows = [json.loads(line) for line in (ROOT/'data/train.jsonl').read_text(encoding='utf-8').splitlines()]
        self.order = list(range(len(self.rows)))
        random.Random(20261024).shuffle(self.order)
        self.labels = self.tokenizer([r['text'] for r in self.rows], truncation=True,max_length=128)['input_ids']
        cache = read(ROOT/'cache/manifest.json')
        self.latents = torch.cat([self.load(ROOT/'cache'/s['file'])['latents'] for s in cache['splits']['train']['shards']])
        assert self.latents.shape==(len(self.rows),32,64) and self.latents.dtype==torch.float32
        assert len(self.rows)==47767 and torch.isfinite(self.latents).all()
        assert torch.allclose(self.latents.norm(dim=-1),torch.full((len(self.rows),32),8.),atol=2e-4,rtol=2e-4)
        self.panel = read(ROOT/'panel.json')
        self.check_cache()
        self.dev = self.encode_panel(self.panel['selected'])
        contract = dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
                        input_manifest_sha256=digest(ROOT/'inputs.json'),script_sha256=digest(__file__),
                        protocol_sha256=digest(ROOT/'PLAN.md'),seed=20261024,precision='FP32',decode=SETTINGS,
                        trainable={arm:configure_mask(self.ae,arm) for arm in ('D0','D1')},
                        args=vars(args),frozen_sha256=self.frozen_hash,cache_check=self.cache_check)
        write(self.out/'contract.json',contract)
        event('effective_trainable_parameters',arms=contract['trainable'])
        self.states={}
        self.best={}
        self.history={arm:[] for arm in ('D0','D1')}
        self.degenerate={arm:0 for arm in ('D0','D1')}

    def remaining(self):
        return min(self.deadline-time.time(),24*3600-(time.monotonic()-self.started))

    def guard(self, training=False):
        if STOP or self.remaining()<=120 or (training and self.training_seconds>=18*3600):
            raise InterruptedError('Signal, absolute deadline or wall-time ceiling reached.')
        limit=10 if self.args.verify_only else 16
        if self.torch.cuda.max_memory_allocated()>limit*1024**3:
            raise RuntimeError('Unexpected memory use; inspect frozen mask before proceeding.')

    def hash_frozen(self):
        h=hashlib.sha256()
        for n,p in self.ae.named_parameters():
            if n not in self.union:
                h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def restore(self, values):
        with self.torch.no_grad():
            for n,p in self.ae.named_parameters():
                if n in values:
                    p.copy_(values[n].to(p.device))

    def weights(self):
        return {n:p.detach().cpu().clone() for n,p in self.ae.named_parameters() if n in self.union}

    def check_cache(self):
        checks=[]
        with self.torch.no_grad():
            for i in self.order[:16]:
                batch=self.tokenizer(self.rows[i]['text'],return_tensors='pt',padding='max_length',truncation=True,max_length=128).to('cuda')
                z=self.ae.get_diffusion_latent(self.ae.get_encoder()(**batch),batch.attention_mask).float().cpu()[0]
                delta=float((z-self.latents[i]).abs().max())
                if not self.torch.allclose(z,self.latents[i],atol=2e-4,rtol=2e-4):
                    raise RuntimeError(f'Frozen cache mismatch at {i}: {delta}')
                checks.append(dict(index=i,id=self.rows[i]['id'],max_abs_error=delta))
        self.cache_check=checks

    def encode_panel(self, rows):
        result=[]
        with self.torch.no_grad():
            for r in rows:
                ids=self.tokenizer(r['text'],truncation=True,max_length=128)['input_ids']
                assert ids==r['ids']
                x=self.torch.tensor([ids],device='cuda'); mask=self.torch.ones_like(x)
                z=self.ae.get_diffusion_latent(self.ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
                result.append((r,z.detach().cpu()))
        return result

    def evaluate(self, panel, tag):
        from transformers.modeling_outputs import BaseModelOutput
        start=time.monotonic(); rows=[]
        with self.torch.no_grad():
            for r,z in panel:
                self.guard()
                enc=BaseModelOutput(last_hidden_state=self.ae.get_decoder_input(z.cuda()))
                labels=self.torch.tensor([r['ids']],device='cuda')
                answer=self.ae(encoder_outputs=enc,labels=labels,use_cache=False)
                ce=float(answer.loss)
                if not math.isfinite(ce): raise FloatingPointError('Nonfinite validation loss')
                del answer
                ids=self.ae.generate(encoder_outputs=enc,**SETTINGS)[0].tolist()
                text=self.tokenizer.decode(ids,skip_special_tokens=True).strip()
                ref=self.tokenizer.decode(r['ids'],skip_special_tokens=True)
                rows.append(dict(id=r['id'],document_id=r['document_id'],source=r['text'],model_view=ref,
                                 text=text,generated_ids=ids,nll=ce*len(r['ids']),tokens=len(r['ids']),
                                 cap=int(len(ids)>=128),eos=int(self.tokenizer.eos_token_id in ids),**metrics(ref,text)))
        elapsed=time.monotonic()-start
        write(self.out/f'{tag}.json',dict(rows=rows,metrics=aggregate(rows),seconds=elapsed))
        event('validation',tag=tag,seconds=elapsed,metrics=aggregate(rows))
        return rows,aggregate(rows),elapsed

    def optimizer(self, arm):
        names=configure_mask(self.ae,arm)
        opt=self.torch.optim.AdamW([p for p in self.ae.parameters() if p.requires_grad],lr=2e-5,
                                  betas=(.9,.999),weight_decay=.01,foreach=False)
        return opt,names

    def update(self,opt,step):
        from transformers.modeling_outputs import BaseModelOutput
        self.guard(training=True)
        ids=self.order[step*8:(step+1)*8]
        assert len(ids)==8
        total=sum(len(self.labels[i]) for i in ids)
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group['lr']=2e-5*min(1.,(step+1)/100)
        value=0.
        for i in ids:
            # No gradient path into the frozen reconstruction/Perceiver mapping.
            with self.torch.no_grad():
                hidden=self.ae.get_decoder_input(self.latents[i:i+1].cuda())
            labels=self.torch.tensor([self.labels[i]],device='cuda')
            loss=self.ae(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),labels=labels,use_cache=False).loss
            if not self.torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            weight=len(self.labels[i])/total
            value+=float(loss.detach())*weight
            (loss*weight).backward()
        params=[p for p in self.ae.parameters() if p.requires_grad]
        norm=self.torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if float(norm)==0: raise RuntimeError('Zero training gradient')
        opt.step()
        self.torch.cuda.synchronize()
        return value

    def snapshot(self,opt,step):
        return dict(model=self.weights(),optimizer=opt.state_dict(),step=step,
                    torch_rng=self.torch.get_rng_state(),cuda_rng=self.torch.cuda.get_rng_state_all())

    def restore_snapshot(self,arm,state):
        self.restore(state['model']); opt,_=self.optimizer(arm)
        opt.load_state_dict(state['optimizer'])
        self.torch.set_rng_state(state['torch_rng'].cpu())
        self.torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        return opt

    def roundtrip(self,arm,opt):
        # Two identical disposable updates from the same state, then restore it.
        before=self.snapshot(opt,0)
        path=self.out/f'{arm}-roundtrip.pt'
        self.save(path,before)
        opt=self.restore_snapshot(arm,self.load(path))
        self.update(opt,0); one=self.snapshot(opt,1)
        opt=self.restore_snapshot(arm,self.load(path))
        self.update(opt,0); two=self.snapshot(opt,1)
        assert tree_equal(one,two), 'Update differs after save/load'
        assert any(not self.torch.equal(one['model'][n],before['model'][n]) for n in one['model']), 'No parameter update'
        opt=self.restore_snapshot(arm,self.load(path))
        assert self.hash_frozen()==self.frozen_hash
        if arm=='D0':
            assert all(self.torch.equal(one['model'][n],self.union[n]) for n in self.union if n.startswith('decoder.block.10.'))
        write(self.out/f'{arm}-roundtrip.json',dict(exact=True,weights_changed=True,frozen_unchanged=True))
        return opt

    def train_to(self,arm,target,calibrate=False):
        if arm in self.states:
            state=self.load(self.states[arm]); opt=self.restore_snapshot(arm,state); step=state['step']
        else:
            self.restore(self.union); opt,_=self.optimizer(arm); step=0
            smoke_start=time.monotonic(); opt=self.roundtrip(arm,opt)
            self.training_seconds+=time.monotonic()-smoke_start
        samples=[]; timer=time.monotonic()
        try:
            while step<target:
                start=time.monotonic(); value=self.update(opt,step)
                elapsed=time.monotonic()-start
                self.training_seconds+=elapsed
                if step>=20: samples.append(elapsed)
                step+=1
                if step%20==0:
                    event('train',arm=arm,step=step,target=target,ce=value,training_seconds=self.training_seconds)
        finally:
            p=self.out/f'{arm}-latest.pt'
            self.save(p,self.snapshot(opt,step)); self.states[arm]=p
            write(self.out/'progress.json',dict(states={k:str(v) for k,v in self.states.items()},
                  training_seconds=self.training_seconds,current_arm=arm,step=step,stop_requested=STOP))
        assert self.hash_frozen()==self.frozen_hash
        if arm=='D0':
            assert all(self.torch.equal(self.weights()[n],self.union[n]) for n in self.union if n.startswith('decoder.block.10.'))
        event('block_finished',arm=arm,step=step,seconds=time.monotonic()-timer)
        return float(self.np.percentile(samples,90)) if calibrate else step

    def confirmation(self,candidate,baseline):
        panel=self.encode_panel(self.panel['reserved'])
        outputs={}
        for arm in ('B0','D0','D1'):
            self.restore(self.union if arm=='B0' else self.load(self.best[arm]['file'])['model'])
            outputs[arm]=self.evaluate(panel,f'reserve-{arm}')[0]
        def interval(a,b):
            docs=sorted({r['document_id'] for r in a})
            values={d:[i for i,r in enumerate(a) if r['document_id']==d] for d in docs}
            differences=self.np.array([x['chrf']-y['chrf'] for x,y in zip(a,b)])
            sums=self.np.array([differences[values[d]].sum() for d in docs]); counts=self.np.array([len(values[d]) for d in docs])
            rng=self.np.random.default_rng(20261025)
            picks=rng.integers(len(docs),size=(10000,len(docs)))
            estimates=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
            return self.np.percentile(estimates,[2.5,97.5]).tolist()
        a,b=aggregate(outputs[candidate]),aggregate(outputs['B0'])
        ci=interval(outputs[candidate],outputs['B0'])
        passes=eligible(a,b) and a['chrf']-b['chrf']>=3 and ci[0]>0 and (a['exact']-b['exact'])/a['n']>=.03
        control_ci=interval(outputs['D1'],outputs['D0'])
        t,c=aggregate(outputs['D1']),aggregate(outputs['D0'])
        specific=eligible(t,c) and t['chrf']-c['chrf']>=1 and control_ci[0]>0
        sheet=[]; key=[]; rng=random.Random(20261025)
        for i,(old,new) in enumerate(zip(outputs['B0'],outputs[candidate])):
            flip=rng.choice([False,True]); texts=[old['text'],new['text']]
            if flip: texts.reverse()
            sheet.append(dict(item=i+1,id=old['id'],source=old['source'],A=texts[0],B=texts[1],
                              A_content_errors='',B_content_errors='',A_clause_errors='',B_clause_errors='',
                              A_unsupported_additions='',B_unsupported_additions='',notes=''))
            key.append(dict(item=i+1,A=candidate if flip else 'B0',B='B0' if flip else candidate))
        rng.shuffle(sheet)
        with (self.out/'researcher-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sheet[0]));w.writeheader();w.writerows(sheet)
        write(self.out/'review-key.json',key)
        return dict(candidate=candidate,metrics={arm:aggregate(rows) for arm,rows in outputs.items()},
                    candidate_minus_baseline_chrf_ci=ci,two_minus_one_chrf_ci=control_ci,
                    automatic_reconstruction_gate_passed=bool(passes),additional_block_effect_supported=bool(specific),
                    researcher_review='pending',unconditional_transfer='not run; requires successful content review',
                    hypothesis_established=False)

    def run(self):
        self.restore(self.union)
        if self.args.verify_only:
            rows,_,_=self.evaluate(self.dev[:4],'verification-baseline')
            reference=read(ROOT/'baseline.json')
            assert all(r['generated_ids']==ref['correct']['generated_ids'] for r,ref in zip(rows,reference[:4]))
            assert self.hash_frozen()==self.frozen_hash
            write(self.out/'verification.json',dict(baseline_token_sequences_match=True,cache_alignment=True,
                    frozen_unchanged=True,optimizer_updates=0,peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3))
            return
        rows,baseline,v=self.evaluate(self.dev,'development-B0')
        reference=read(ROOT/'baseline.json')
        assert all(r['generated_ids']==ref['correct']['generated_ids'] for r,ref in zip(rows,reference)), 'Baseline reproduction differs; inspect environment before training'
        self.save(self.out/'baseline-delta.pt',dict(model=self.union,step=0))
        for arm in ('D0','D1'):
            self.best[arm]=dict(step=0,metrics=baseline,file=str(self.out/'baseline-delta.pt'))
        rates={arm:self.train_to(arm,120,calibrate=True) for arm in ('D0','D1')}
        if time.monotonic()-self.started>7200:
            write(self.out/'summary.json',dict(status='not_admitted',reason='Preflight exceeded two hours',training_seconds=self.training_seconds))
            return
        n=choose_horizon(rates['D0'],rates['D1'],v,self.remaining(),self.args.max_steps)
        write(self.out/'calibration.json',dict(p90_seconds_per_update=rates,validation_seconds=v,horizon=n,
                    peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3,remaining_seconds=self.remaining()))
        if not n:
            write(self.out/'summary.json',dict(status='not_admitted',reason='Measured workload cannot fit',training_seconds=self.training_seconds))
            return
        reason='planned_horizon'
        milestones={arm:baseline['chrf'] for arm in ('D0','D1')}; stale={arm:0 for arm in milestones}
        stopped=set()
        for target in range(500,n+1,500):
            for arm in ('D0','D1'):
                if arm in stopped: continue
                self.train_to(arm,target)
                _,m,_=self.evaluate(self.dev,f'development-{arm}-{target:04d}')
                self.history[arm].append(dict(step=target,metrics=m))
                best=self.best[arm]
                if eligible(m,baseline) and (m['chrf'],m['exact'],-target)>(best['metrics']['chrf'],best['metrics']['exact'],-best['step']):
                    file=self.out/f'{arm}-best.pt'; self.save(file,self.load(self.states[arm]))
                    self.best[arm]=dict(step=target,metrics=m,file=str(file))
                current=self.best[arm]['metrics']['chrf']
                if current>=milestones[arm]+.5: milestones[arm]=current; stale[arm]=0
                else: stale[arm]+=1
                bad=m['numeric_recall']<baseline['numeric_recall']-.05 or m['repeat3']>.02
                self.degenerate[arm]=self.degenerate[arm]+1 if bad else 0
                if self.degenerate[arm]>=2:
                    stopped.add(arm);event('degeneration_stop',arm=arm,step=target)
                write(self.out/'selection.json',dict(best=self.best,history=self.history,stopped=sorted(stopped)))
            if target>=1000 and (all(stale[a]>=2 for a in stale) or len(stopped)==2):
                reason='early_stop';break
            if stopped:
                # Do not give one arm a larger continuation budget after an arm fails.
                reason='paired_stop_after_degeneracy';break
        candidate=choose_candidate(self.best,baseline)
        write(self.out/'candidate-locked.json',dict(candidate=candidate,best=self.best,reason=reason))
        summary=dict(status='completed_training',reason=reason,horizon=n,best=self.best,candidate=candidate,
                     training_seconds=self.training_seconds,peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3)
        if candidate:
            summary['confirmation']=self.confirmation(candidate,baseline)
        else:
            summary['confirmation']='not admitted; neither candidate passed development gate'
        assert self.hash_frozen()==self.frozen_hash
        summary['total_seconds']=time.monotonic()-self.started
        write(self.out/'summary.json',summary)
        event('finished',candidate=candidate,training_hours=self.training_seconds/3600)


def preflight(args):
    import torch, transformers, sacrebleu
    if sacrebleu.__version__!='2.6.0':
        raise RuntimeError('SacreBLEU 2.6.0 required; run the documented dependency preparation before submission.')
    for line in (ROOT/'bundle-manifest.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        if digest(ROOT/name)!=expected: raise RuntimeError(f'Bundle hash mismatch: {name}')
    contract=read(ROOT/'inputs.json')
    for name,key in [('model.pt','ae'),('args.json','ae_args')]:
        if digest(Path(args.ae)/name)!=contract[key]: raise RuntimeError(f'AE mismatch: {name}')
    for name,expected in contract['model_cache'].items():
        if digest(Path(args.model_cache)/name)!=expected: raise RuntimeError(f'Model cache mismatch: {name}')
    for split in ('train','valid'):
        assert digest(ROOT/f'data/{split}.jsonl')==contract['data'][split]
    cache=read(ROOT/'cache/manifest.json')
    assert cache['contract']['ae']==contract['ae'] and cache['contract']['data']==contract['data']
    for shard in cache['splits']['train']['shards']:
        assert digest(ROOT/'cache'/shard['file'])==shard['sha256']
    panel=read(ROOT/'panel.json')
    assert len(panel['selected'])==81 and len(panel['reserved'])==103
    assert not ({r['document_id'] for r in panel['selected']} & {r['document_id'] for r in panel['reserved']})
    event('cpu_preflight_passed',torch=torch.__version__,transformers=transformers.__version__,sacrebleu=sacrebleu.__version__)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ae',required=True);parser.add_argument('--model-cache',required=True)
    parser.add_argument('--output',default=str(ROOT/'outputs/check'))
    parser.add_argument('--cpu-only',action='store_true');parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--max-steps',type=int,default=3000,choices=(1000,1500,2000,2500,3000))
    parser.add_argument('--deadline',default='2026-10-25T15:00:00+02:00')
    parser.add_argument('--start-before',default='2026-10-24T15:00:00+02:00')
    args=parser.parse_args()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop_signal)
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,stop_signal)
    preflight(args)
    if args.cpu_only:return
    exp=None
    try:
        exp=Experiment(args);exp.run()
    except Exception as exc:
        if exp is not None:
            write(exp.out/'failure.json',dict(error=repr(exc),training_seconds=exp.training_seconds,stop_requested=STOP))
        raise


if __name__=='__main__':
    main()
