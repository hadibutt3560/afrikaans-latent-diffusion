"""One allocation, bounded stages, persisted admission plan, no test-driven adaptation."""
import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
from common import read,write,sha,identity,budget,append

STOP=False
CHILD=None


def stop(*_):
    global STOP
    STOP=True
    if CHILD and CHILD.poll() is None:CHILD.terminate()


for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,stop)


def verify_bundle(root):
    manifest=read(root/'bundle-manifest.json')
    for name,digest in manifest['files'].items():
        if sha(root/name)!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    return sha(root/'bundle-manifest.json')


class Runner:
    def __init__(self,root,out,allocation_end):
        self.root=root;self.out=out;self.end=allocation_end;self.start=time.time();self.c=read(root/'config.json')
        self.state=read(out/'stages.json') if (out/'stages.json').exists() else {}

    def stage(self,name,seconds,key=None,run=''):
        global CHILD
        key=key or name
        if self.state.get(key,{}).get('complete'):return
        prior=self.state.get(key,{}).get('seconds',0)
        remaining=max(0,seconds-prior)
        if STOP or remaining<180:raise InterruptedError('Stage budget exhausted: '+key)
        # Always preserve the final hour for report/archive, even on estimate failure.
        deadline=min(time.time()+remaining,self.end-3600)
        if deadline-time.time()<180:raise InterruptedError('Evaluation/shutdown deadline reached')
        command=[sys.executable,'-u',str(self.root/'worker.py'),name,'--root',str(self.root),'--out',str(self.out),'--deadline',str(deadline)]
        if run:command+=['--run',run]
        self.state[key]=dict(complete=False,seconds=prior,started=time.time(),deadline=deadline)
        write(self.out/'stages.json',self.state);print('STAGE',key,'budget seconds',remaining,flush=True)
        began=time.monotonic();last=began
        with (self.out/'logs'/f'{key}.log').open('a',encoding='utf8') as log:
            CHILD=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            while CHILD.poll() is None:
                if STOP or time.time()>=deadline:
                    CHILD.terminate()
                    try:CHILD.wait(timeout=120)
                    except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
                    break
                if time.monotonic()-last>15:
                    self.state[key]['seconds']=prior+time.monotonic()-began;write(self.out/'stages.json',self.state);last=time.monotonic()
                time.sleep(1)
            code=CHILD.returncode;CHILD=None
        self.state[key].update(seconds=prior+time.monotonic()-began,returncode=code,complete=code==0)
        write(self.out/'stages.json',self.state)
        if code:raise RuntimeError(f'{key} returned {code}; inspect logs/{key}.log; latest checkpoint retained')

    def run(self):
        c=self.c
        self.stage('smoke',3600)
        self.stage('ae_train',c['ae']['hours']*3600)
        self.stage('ae_gate',1800)
        self.stage('cache',c['budget']['cache_max_hours']*3600)
        self.stage('benchmark',1800)
        measured=read(self.out/'benchmark.json');counts=read(self.root/'data/audit.json')['record_counts']
        # Validation + held-out test are both charged to evaluation, with a 2.5x timing safety factor.
        n_eval=counts['valid']+counts['test']
        plan_path=self.out/'plan.json'
        if plan_path.exists():plan=read(plan_path)
        else:
            elapsed=sum(x.get('seconds',0) for x in self.state.values());plan=None
            for n in (4,2,1):
                b=budget(elapsed,measured,n,c['samples_per_split'],n_eval)
                available=min(b['remaining_train_seconds'],self.end-time.time()-3600-b['evaluation_reserve_seconds'])
                if available>=n*c['budget']['minimum_run_hours']*3600:
                    candidates=[f'{model}-{seed}' for seed in c['candidate_seeds'] for model in c['models']]
                    plan=dict(**b,admitted_runs=candidates[:n],candidates=candidates,run_seconds=min(c['budget']['maximum_run_hours']*3600,available/n),
                        total_training_pool_seconds=available,
                        samples_per_split=c['samples_per_split'],measured=measured,rule='admit 4, else 2, else 1 complete run; no test access before selection')
                    break
            if plan is None:raise RuntimeError('Insufficient budget for a scientifically complete diffusion run and evaluation')
            plan['run_budgets']={name:plan['run_seconds'] for name in plan['admitted_runs']}
            write(plan_path,plan)
        frozen=(self.out/'selection.json').exists() or plan.get('training_frozen',False)
        for run in ([] if frozen else list(plan['admitted_runs'])):
            available=self.end-time.time()-3600-plan['evaluation_reserve_seconds']
            used=sum(v.get('seconds',0) for k,v in self.state.items() if k.startswith('train-'))
            available=min(available,plan['total_training_pool_seconds']-used)
            prior=self.state.get('train-'+run,{}).get('seconds',0)
            if self.state.get('train-'+run,{}).get('complete'):continue
            if available<180:raise InterruptedError('Training stopped to preserve evaluation reserve')
            self.stage('diffusion_train',min(plan['run_budgets'][run],prior+available),key='train-'+run,run=run)
        # If convergence saved time, spend it on the next preregistered independent seed.
        # Admission uses timing only and increases the reserve BEFORE the run starts.
        for run in ([] if frozen else plan['candidates']):
            if run in plan['admitted_runs']:continue
            b=budget(0,measured,len(plan['admitted_runs'])+1,c['samples_per_split'],n_eval)
            available=self.end-time.time()-3600-b['evaluation_reserve_seconds']
            used=sum(v.get('seconds',0) for k,v in self.state.items() if k.startswith('train-'))
            available=min(available,plan['total_training_pool_seconds']-used-(b['evaluation_reserve_seconds']-plan['evaluation_reserve_seconds']))
            if available<c['budget']['minimum_run_hours']*3600:break
            plan['total_training_pool_seconds']-=b['evaluation_reserve_seconds']-plan['evaluation_reserve_seconds']
            plan['admitted_runs'].append(run);plan['evaluation_reserve_seconds']=b['evaluation_reserve_seconds']
            plan['run_budgets'][run]=min(plan['run_seconds'],available);write(plan_path,plan)
            self.stage('diffusion_train',plan['run_budgets'][run],key='train-'+run,run=run)
        # Allocate explicit downstream reserves; selection cannot consume scoring time.
        n=len(plan['admitted_runs']);samples=c['samples_per_split']
        reserves=budget(0,measured,n,samples,n_eval)['stage_reserves']
        score_reserve=reserves['scoring'];generation_reserve=reserves['evaluate']
        plan['evaluation_stage_reserves']=dict(scoring=score_reserve,evaluate=generation_reserve)
        plan['training_frozen']=True
        write(plan_path,plan)
        for stage,downstream in [('selection',generation_reserve+score_reserve),('evaluate',score_reserve),('scoring',0)]:
            remaining=self.end-time.time()-3600-downstream
            self.stage(stage,self.state.get(stage,{}).get('seconds',0)+remaining)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('The full workflow must run inside its one Slurm allocation')
    if out.exists() and not a.resume:raise FileExistsError('Use a fresh output or --resume for interruption recovery')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    # Launcher supplies scheduler EndTime; no 72-hour reset after slow startup.
    allocation_end=float(os.environ['SW_ALLOCATION_END'])
    try:
        manifest_hash=verify_bundle(root)
        if shutil.disk_usage(out).free < 150*1024**3:raise RuntimeError('Need at least 150 GiB free persistent storage for checkpoints, caches, smoke and results archive; also check user quota')
        decision=read(root/'data-decision.json')
        if decision['status']!='retain_supplied_benchmark':raise RuntimeError('Dataset decision pending; see DATA_AUDIT.md')
        tok=read(root/'tokens/audit.json');c=read(root/'config.json')
        for split in ('train','valid'):
            for metric in ('unknown','truncated'):
                if tok['splits'][split][metric+'_fraction']>c['tokenizer_gate']['maximum_'+metric+'_fraction']:raise RuntimeError('Tokenizer gate failed')
        contract=dict(bundle_sha256=manifest_hash,config_sha256=sha(root/'config.json'),decision=decision)
        if (out/'contract.json').exists() and read(out/'contract.json')!=contract:raise ValueError('Cannot resume changed source/data/config/assets')
        write(out/'contract.json',contract);write(out/'configuration.json',c)
        allocation=dict(job=os.getenv('SLURM_JOB_ID'),start=time.time(),end=allocation_end,resume=a.resume)
        write(out/'allocation.json',allocation);append(out/'allocation-history.jsonl',allocation)
        Runner(root,out,allocation_end).run()
        write(out/'completion.json',dict(status='complete',finished=time.time()))
    except BaseException as exc:
        write(out/'completion.json',dict(status='partial',error=repr(exc),traceback=traceback.format_exc(),finished=time.time()))
        raise
    finally:
        # Reporting is independent of GPU/model imports, including negative/partial outcomes.
        result=subprocess.run([sys.executable,str(root/'report.py'),'--root',str(root),'--out',str(out)],check=False,
                              timeout=max(1,min(1800,allocation_end-time.time()-180)))
        if result.returncode:
            status=read(out/'completion.json');status.update(status='partial',reporting_exitcode=result.returncode);write(out/'completion.json',status)
        if read(out/'completion.json')['status']!='complete':raise RuntimeError('PARTIAL outcome; see completion.json and stage logs')


if __name__=='__main__':main()
