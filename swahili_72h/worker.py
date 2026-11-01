"""Bounded offline stages used by the single-allocation supervisor."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from common import read,write,rows,sha,identity,append
import engine as e


def token_array(root,split):return np.load(root/'tokens'/f'{split}.npy',mmap_mode='r')
def latent_array(out,split):return np.load(out/'cache'/f'{split}.npy',mmap_mode='r')


def equal(a,b):
    if isinstance(a,e.torch.Tensor):return e.torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def resume_smoke(factory,train,valid,c,kind,folder,contract):
    c=copy.deepcopy(c);c[kind].update(micro_batch=1,accumulation=1,validate_every=100,max_steps=10,warmup=2)
    c.update(validation_examples=2,validation_repeats=1,checkpoint_steps=1)
    # Always fresh independent checks per hardware/job. Both AE and real diffusion architectures.
    folder.mkdir(parents=True,exist_ok=True)
    for mode,n in [('continuous',4),('resumed',2),('resumed',2)]:
        e.seed(713);model=factory()
        e.train(model,train,valid,c,kind,folder/mode,713,contract,steps=n)
        del model;e.clean()
    x=e.load(folder/'continuous/latest.pt');y=e.load(folder/'resumed/latest.pt')
    comparison={k:equal(x[k],y[k]) for k in ('model','optimizer','ema','rng','scheduler')}
    if not all(comparison.values()):raise AssertionError(f'{kind} resume differs: {comparison}')
    result=dict(comparison=comparison,step=y['state']['step']);del x,y;e.clean();return result


def reference_rows(root,split,count,c):
    # Full exact canonical text, selected by a fixed shuffled row index before test outcomes exist.
    n=read(root/'data/audit.json')['record_counts'][split]
    selected=set(np.random.default_rng(c['validation_seed']).permutation(n)[:count].tolist())
    return [r for i,r in enumerate(rows(root/'data'/f'{split}.jsonl')) if i in selected]


def smoke(a,c):
    out=a.out;root=a.root;hardware=e.setup(c)
    write(out/'hardware.json',dict(**hardware,python=platform.python_version(),host=platform.node(),job=os.getenv('SLURM_JOB_ID'),
                                environment=subprocess.run([__import__('sys').executable,'-m','pip','freeze'],capture_output=True,text=True).stdout))
    report={};train=token_array(root,'train')[:8];valid=token_array(root,'valid')[:4]
    tok=e.tokenizer(root)
    for split in ('train','valid'):
        array=token_array(root,split)
        for i,row in enumerate(rows(root/'data'/f'{split}.jsonl')):
            if i>=128:break
            ids=tok.encode(row['text'],add_special_tokens=True);ids=(ids[:191]+[tok.eos_token_id]) if len(ids)>192 else ids
            if not np.array_equal(array[i],ids+[0]*(192-len(ids))):raise ValueError('Tokenizer version changed exported model views')
    report['tokenizer_export_parity']=True
    folder=out/'smoke'/str(int(time.time()))
    report['ae_resume']=resume_smoke(lambda:e.ae_model(root,c),train,valid,c,'ae',folder/'ae',a.contract)
    write(out/'smoke-progress.json',report)
    e.seed(42);synthetic=e.torch.nn.functional.normalize(e.torch.randn(8,32,64),dim=-1).numpy()*8
    for name in c['models']:
        report[name+'_resume']=resume_smoke(lambda:e.diffusion(c,name),synthetic[:6],synthetic[6:],c,'diffusion',folder/name,a.contract)
        write(out/'smoke-progress.json',report)
    # Measure real pretrained baseline and free-running Swahili reconstruction, validation only.
    ae=e.ae_model(root,c);tok=e.tokenizer(root)
    tick=time.monotonic();report['initial_validation']=e.validate(ae,token_array(root,'valid'),c,'ae',count=32)
    report['initial_reconstruction']=e.reconstruct(ae,token_array(root,'valid'),c,tok,count=16)
    report['ae_validation_seconds']=time.monotonic()-tick
    del ae;e.clean()
    model,tok=e.scorer(root);tick=time.monotonic()
    scores=[e.score(model,tok,r['text']) for r in reference_rows(root,'valid',16,c)]
    report['scorer_seconds_per_text']=(time.monotonic()-tick)/len(scores);report['scorer_smoke_scores']=scores
    del model;e.clean()
    write(out/'smoke.json',report)


def ae_train(a,c):
    ae=e.ae_model(a.root,c)
    e.train(ae,token_array(a.root,'train'),token_array(a.root,'valid'),c,'ae',a.out/'ae',c['seed'],a.contract)
    if e.STOP:raise InterruptedError('AE training signal')


def ae_gate(a,c):
    ae=e.ae_model(a.root,c,a.out/'ae/best.pt');tok=e.tokenizer(a.root)
    result=e.validate(ae,token_array(a.root,'valid'),c,'ae')
    recon=e.reconstruct(ae,token_array(a.root,'valid'),c,tok,count=64)
    initial=read(a.out/'smoke.json')['initial_validation']['loss'];s=c['ae']
    gate=dict(validation=result,reconstruction=recon,initial_nll=initial,
        checks=dict(accuracy=result['token_accuracy']>=s['gate_token_accuracy'],chrf=recon['mean_chrf']>=s['gate_chrf'],
                    nll_reduction=result['loss']<=initial*s['gate_nll_ratio']),best_sha256=sha(a.out/'ae/best.pt'))
    gate['passed']=all(gate['checks'].values());write(a.out/'ae-gate.json',gate)
    if not gate['passed']:raise RuntimeError('Swahili reconstruction gate failed; retain diagnostic results, do not train diffusion')


def cache_split(a,c,ae,split):
    folder=a.out/'cache';folder.mkdir(exist_ok=True);path=folder/f'{split}.npy';meta=folder/f'{split}.json'
    values=token_array(a.root,split);n=len(values);contract=dict(ae=sha(a.out/'ae/best.pt'),tokens=sha(a.root/'tokens'/f'{split}.npy'))
    if meta.exists():
        state=read(meta)
        if state['contract']!=contract:raise ValueError('Stale latent cache')
        if state.get('complete'):
            if sha(path)!=state['sha256']:raise ValueError('Corrupted latent cache')
            return
        if path.exists() and not Path(str(path)+'.partial').exists():
            array=np.load(path,mmap_mode='r')
            if array.shape!=(n,c['latent_length'],c['latent_dim']):raise ValueError('Cache finalization shape')
            for i in range(0,n,1024):
                if not np.isfinite(array[i:i+1024]).all():raise ValueError('Cache finalization non-finite')
            del array;write(meta,dict(contract=contract,cursor=n,complete=True,sha256=sha(path),records=n,recovered_finalization=True));return
        start=state['cursor'];array=np.lib.format.open_memmap(str(path)+'.partial',mode='r+')
    else:
        start=0;array=np.lib.format.open_memmap(str(path)+'.partial',mode='w+',dtype='float32',shape=(n,c['latent_length'],c['latent_dim']))
        array.flush();write(meta,dict(contract=contract,cursor=0,complete=False))
    began=time.monotonic();last_save=began;batch=8;ae.eval()
    with e.torch.no_grad():
        for i in range(start,n,batch):
            e.check(30);x,mask,_=e.token_batch(values,np.arange(i,min(n,i+batch)))
            # Cache in FP32, independent of diffusion mixed precision.
            z=ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask).float()
            if not e.torch.isfinite(z).all():raise FloatingPointError('Cache latent')
            array[i:i+len(z)]=z.cpu().numpy()
            if time.monotonic()-last_save>30:
                array.flush();write(meta,dict(contract=contract,cursor=i+len(z),complete=False));last_save=time.monotonic()
                print('cache',split,i+len(z),'/',n,flush=True)
    array.flush();del array;Path(str(path)+'.partial').replace(path)
    write(meta,dict(contract=contract,cursor=n,complete=True,sha256=sha(path),seconds=time.monotonic()-began,records=n))


def cache(a,c):
    ae=e.ae_model(a.root,c,a.out/'ae/best.pt').eval().requires_grad_(False)
    for split in ('train','valid'):cache_split(a,c,ae,split)


def benchmark(a,c):
    result=dict(score=read(a.out/'smoke.json')['scorer_seconds_per_text'],models={})
    train=latent_array(a.out,'train');valid=latent_array(a.out,'valid');folder=a.out/'benchmark'/str(int(time.time()))
    for name in c['models']:
        e.seed(713);model=e.diffusion(c,name);small=copy.deepcopy(c);small['validation_examples']=128;small['validation_repeats']=1
        state=e.train(model,train,valid,small,'diffusion',folder/name,713,a.contract,steps=32)
        tick=time.monotonic();e.validate(model,valid,c,'diffusion',count=128);val=(time.monotonic()-tick)/128
        ae=e.ae_model(a.root,c,a.out/'ae/best.pt').eval().requires_grad_(False);tok=e.tokenizer(a.root)
        tick=time.monotonic();samples=e.generate(model,ae,tok,c,4,c['sample_seed']);sample=(time.monotonic()-tick)/4
        # Also time encoding outside startup so reserve reflects the adapted AE.
        tick=time.monotonic()
        with e.torch.no_grad():
            for i in range(0,32,8):
                x,mask,_=e.token_batch(token_array(a.root,'valid'),np.arange(i,i+8));ae.get_diffusion_latent(ae.get_encoder()(input_ids=x,attention_mask=mask),mask)
        e.torch.cuda.synchronize();encode=(time.monotonic()-tick)/32
        result['models'][name]=dict(step=float(np.quantile(state['timings'][4:],.9)),sample=sample,validate=val,encode=encode)
        write(a.out/f'benchmark-{name}-samples.json',samples);del model,ae;e.clean()
    result.update({key:max(v[key] for v in result['models'].values()) for key in ('sample','validate','encode')})
    # Generated outputs can tokenize much longer than real Swahili. Measure both,
    # sequentially after unloading AE/diffusion to avoid inflated peak VRAM.
    scorer,tok=e.scorer(a.root);durations=[]
    benchmark_texts=reference_rows(a.root,'valid',8,c)
    for name in c['models']:benchmark_texts+=read(a.out/f'benchmark-{name}-samples.json')
    for item in benchmark_texts:
        tick=time.monotonic();e.score(scorer,tok,item['text'],c['scorer']['context']);durations.append(time.monotonic()-tick)
    result['score']=max(result['score'],max(durations));result['scoring_timing_seconds']=durations
    del scorer;e.clean()
    write(a.out/'benchmark.json',result)


def diffusion_train(a,c):
    name,seed=a.run.rsplit('-',1);e.seed(int(seed));model=e.diffusion(c,name)
    e.train(model,latent_array(a.out,'train'),latent_array(a.out,'valid'),c,'diffusion',a.out/'runs'/a.run,int(seed),a.contract)
    if e.STOP:raise InterruptedError('Diffusion training signal')


def selection(a,c):
    plan=read(a.out/'plan.json');results=[]
    for run in plan['admitted_runs']:
        path=a.out/'runs'/run/'best.pt'
        if not path.exists():raise RuntimeError(f'No validated checkpoint for admitted run {run}')
        name,_=run.rsplit('-',1);model=e.inference_checkpoint(e.diffusion(c,name),path)
        result=e.validate(model,latent_array(a.out,'valid'),c,'diffusion',count=len(latent_array(a.out,'valid')))
        results.append(dict(run=run,checkpoint_sha256=sha(path),validation=result));del model;e.clean()
    winner=min(results,key=lambda r:r['validation']['loss'])['run']
    # Freeze before first test-model access. No selection/science changes after this file.
    write(a.out/'selection.json',dict(criterion=c['selection'],winner=winner,runs=results,configuration_sha256=sha(a.root/'config.json')))


def evaluate(a,c):
    selected=read(a.out/'selection.json');ae=e.ae_model(a.root,c,a.out/'ae/best.pt').eval().requires_grad_(False);tok=e.tokenizer(a.root)
    cache_split(a,c,ae,'test')
    if not (a.out/'ae-test.json').exists():
        write(a.out/'ae-test.json',dict(validation=e.validate(ae,token_array(a.root,'test'),c,'ae',count=1024),
            reconstruction=e.reconstruct(ae,token_array(a.root,'test'),c,tok,count=64)))
    for selected_run in selected['runs']:
        run=selected_run['run'];folder=a.out/'runs'/run;name,_=run.rsplit('-',1);path=folder/'best.pt'
        if sha(path)!=selected_run['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        model=e.inference_checkpoint(e.diffusion(c,name),path)
        if not (folder/'test.json').exists():
            write(folder/'test.json',e.validate(model,latent_array(a.out,'test'),c,'diffusion',count=len(latent_array(a.out,'test'))))
        for split,offset in [('valid',0),('test',100000)]:
            dest=folder/f'{split}-samples.json';partial=folder/f'{split}-samples.partial.json'
            if dest.exists():continue
            samples=read(partial) if partial.exists() else []
            while len(samples)<c['samples_per_split']:
                samples+=e.generate(model,ae,tok,c,1,c['sample_seed']+offset+len(samples));write(partial,samples)
            os.replace(partial,dest)
        del model;e.clean()
    for split in ('valid','test'):write(a.out/f'{split}-references.json',reference_rows(a.root,split,c['samples_per_split'],c))


def scoring(a,c):
    model,tok=e.scorer(a.root);groups={}
    for split in ('valid','test'):groups[f'real-{split}']=read(a.out/f'{split}-references.json')
    for item in read(a.out/'selection.json')['runs']:
        for split in ('valid','test'):groups[f'{item["run"]}-{split}']=read(a.out/'runs'/item['run']/f'{split}-samples.json')
    out=a.out/'scores';out.mkdir(exist_ok=True)
    for name,texts in groups.items():
        path=out/f'{name}.json';partial=out/f'{name}.partial.json'
        if path.exists():continue
        scored=read(partial) if partial.exists() else []
        for r in texts[len(scored):]:
            e.check(30);scored.append(dict(text=r['text'],id=r.get('id'),seed=r.get('seed'),score=e.score(model,tok,r['text'],c['scorer']['context'])))
            write(partial,scored)
        os.replace(partial,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','ae_train','ae_gate','cache','benchmark','diffusion_train','selection','evaluate','scoring'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--run',default='');a=p.parse_args();a.root=a.root.resolve();a.out=a.out.resolve();c=read(a.root/'config.json')
    a.contract=read(a.out/'contract.json');e.DEADLINE=a.deadline;hardware=e.setup(c)
    append(a.out/'hardware-history.jsonl',dict(stage=a.stage,run=a.run,job=os.getenv('SLURM_JOB_ID'),time=time.time(),hardware=hardware))
    globals()[a.stage](a,c)


if __name__=='__main__':
    try:main()
    except InterruptedError as exc:print(str(exc),flush=True);raise SystemExit(75)
