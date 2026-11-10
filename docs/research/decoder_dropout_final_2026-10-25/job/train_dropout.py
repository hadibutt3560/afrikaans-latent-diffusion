"""One controlled decoder-dropout experiment, with all conditional evaluation."""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import signal
import random
import time
import training_support as support
import engine as e

ARMS = ('D0', 'D1')
MAX_STEPS = 26620
INTERVAL = 1200

def configure_mask(ae, arm):
    if arm not in ARMS: raise ValueError(arm)
    ae.eval().requires_grad_(False)
    for block in ae.decoder.block: block.requires_grad_(True)
    ae.decoder.final_layer_norm.requires_grad_(True)
    ae.decoder.train(arm == 'D1')
    # This embedding module is shared by encoder and decoder; keep its mode frozen.
    ae.shared.eval()
    names=[n for n,p in ae.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(('decoder.block.','decoder.final_layer_norm.')) for n in names)
    assert not ae.shared.weight.requires_grad and not ae.lm_head.weight.requires_grad
    assert not any(m.training for m in ae.encoder.modules())
    assert not any(m.training for m in ae.perceiver_ae.modules())
    return names

support.configure_mask = e.configure_mask = configure_mask

def choose_candidate(best, baseline):
    control,treatment=best['D0']['metrics'],best['D1']['metrics']
    return 'D1' if e.dev_pass(treatment,baseline) and treatment['chrf']>=control['chrf']+1 else None

e.choose_candidate = choose_candidate

def plateau_update(best, milestones, stale):
    for arm in ARMS:
        value=best[arm]['metrics']['chrf']
        if value>=milestones[arm]+.5:
            milestones[arm]=value;stale[arm]=0
        else: stale[arm]+=1
    return all(stale[a]>=6 for a in ARMS)

def projected_seconds(rates, validation, overhead, current=120):
    checks=math.ceil((MAX_STEPS-current)/INTERVAL)
    return ((MAX_STEPS-current)*sum(rates.values())+checks*2*(validation+overhead))*1.15+180

class Experiment(support.Experiment):
    def __init__(self,args):
        super().__init__(args)
        # Reserve five minutes for the wrapper's hash/report/archive export.
        self.hard_end=min(self.started+args.wall_hours*3600-300,
                          time.monotonic()+self.deadline-time.time()-300)
        self.phase_end=min(self.started+args.train_phase_hours*3600,self.hard_end-10800)
        assert self.ae.config.dropout_rate==.1
        import torch
        drops=[(n,float(m.p)) for n,m in self.ae.decoder.named_modules() if isinstance(m,torch.nn.Dropout)]
        assert drops and all(p==.1 for _,p in drops)
        contract=e.read(self.out/'contract.json')
        contract.update(entrypoint_sha256=e.digest(Path(__file__)),support_sha256=e.digest(Path(support.__file__)),
                        dropout=dict(D0=0.,D1=.1,evaluation=0.),decoder_dropout_modules=drops,
                        selection_seed_count=1,max_steps=MAX_STEPS,validation_interval=INTERVAL,
                        export_buffer_seconds=300,plateau_stopping=True)
        e.write(self.out/'contract.json',contract)

    def assert_control_frozen(self):
        assert not self.ae.shared.weight.requires_grad and not self.ae.lm_head.weight.requires_grad
        assert self.ae.decoder.training==(self.arm=='D1')
        assert not any(m.training for m in self.ae.encoder.modules())
        assert not any(m.training for m in self.ae.perceiver_ae.modules())

    def evaluate(self,panel,tag):
        mode=self.ae.decoder.training
        self.ae.eval()
        try: return super().evaluate(panel,tag)
        finally:
            self.ae.decoder.train(mode)
            self.ae.shared.eval()

    def train_to(self,arm,target,calibrate=False):
        if arm not in self.states:
            # Each arm starts from the same RNG state; dropout then advances D1's stream.
            self.torch.manual_seed(20261024);self.torch.cuda.manual_seed_all(20261024)
        return super().train_to(arm,target,calibrate=calibrate)

    def mode_probe(self):
        from transformers.modeling_outputs import BaseModelOutput
        torch=self.torch;self.restore(self.union)
        i=self.order[0:1][0];labels=torch.tensor([self.labels[i]],device='cuda')
        results={};rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        with torch.no_grad():
            hidden=self.ae.get_decoder_input(self.latents[i:i+1].cuda())
            def loss():
                return float(self.ae(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),labels=labels,use_cache=False).loss)
            for arm in ARMS:
                configure_mask(self.ae,arm)
                values=[loss(),loss(),loss()]
                assert all(math.isfinite(x) for x in values)
                assert (len(set(values))==1)==(arm=='D0'),(arm,values)
                self.ae.eval();ev=[loss(),loss()];assert ev[0]==ev[1]
                results[arm]=dict(training_losses=values,evaluation_losses=ev)
        assert results['D0']['evaluation_losses']==results['D1']['evaluation_losses']
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        self.ae.eval();assert self.hash_frozen()==self.frozen_hash
        e.write(self.out/'mode-verification.json',dict(results=results,optimizer_updates=0))

    def confirmation(self,candidate,baseline):
        # Reuse audited paired-document evaluation, with renamed arm semantics.
        result=e.Experiment.confirmation(self,candidate,baseline)
        result['dropout_minus_control_chrf_ci']=result.pop('two_minus_one_chrf_ci')
        result['dropout_effect_supported']=result.pop('additional_block_effect_supported')
        # Separate blinded source-output review against the matched trained control.
        control=e.read(self.out/'reserve-D0.json')['rows']
        treated=e.read(self.out/'reserve-D1.json')['rows']
        rng=random.Random(20261025);sheet=[];key=[]
        for i,(old,new) in enumerate(zip(control,treated)):
            assert old['id']==new['id']
            flip=rng.choice((False,True));texts=[old['text'],new['text']]
            if flip:texts.reverse()
            sheet.append(dict(item=i+1,id=old['id'],source=old['source'],A=texts[0],B=texts[1],
                A_content_errors='',B_content_errors='',A_clause_errors='',B_clause_errors='',
                A_unsupported_additions='',B_unsupported_additions='',notes=''))
            key.append(dict(item=i+1,A='D1' if flip else 'D0',B='D0' if flip else 'D1'))
        rng.shuffle(sheet)
        with (self.out/'researcher-control-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sheet[0]));w.writeheader();w.writerows(sheet)
        e.write(self.out/'control-review-key.json',key)
        return result

    def run(self):
        self.mode_probe()
        if self.args.verify_only:
            e.Experiment.run(self)
            if self.args.gradient_check:self.gradient_verification()
            self.ae.eval()
            if self.args.evaluation_smoke:
                from transfer_eval import Transfer
                transfer=Transfer(self);transfer.prepare()
                # Exercise all three production banks with identical weights.
                self.save(self.out/'identity.pt',dict(model=self.union,step=0))
                for arm in ARMS:self.best[arm]=dict(file=str(self.out/'identity.pt'))
                result=transfer.run('D1',count=4)
                rows=[e.read(self.out/f'generation/{a}-scored.json') for a in ('B0',*ARMS)]
                assert all([r['generated_ids'] for r in x]==[r['generated_ids'] for r in rows[0]] for x in rows)
                result['smoke_only']=True;result['status']='verification_only_no_training'
                e.write(self.out/'generation/summary.json',result)
            return
        self.restore(self.union)
        rows,baseline,v=self.evaluate(self.dev,'development-B0')
        reference=e.read(e.ROOT/'baseline.json')
        assert len(rows)==len(reference)==81
        assert all(r['generated_ids']==ref['correct']['generated_ids'] for r,ref in zip(rows,reference)), 'Baseline reproduction failed'
        self.save(self.out/'baseline-delta.pt',dict(model=self.union,step=0))
        for arm in ARMS:self.best[arm]=dict(step=0,metrics=baseline,file=str(self.out/'baseline-delta.pt'))
        from transfer_eval import Transfer,should_transfer
        transfer=Transfer(self);reserve=transfer.prepare()
        self.phase_end=min(self.phase_end,self.hard_end-reserve)
        rates={arm:self.train_to(arm,120,calibrate=True) for arm in ARMS}
        left=self.phase_end-time.monotonic()
        predicted=projected_seconds(rates,v,max(self.block_overheads,default=0.))
        admitted=predicted<=left and time.monotonic()-self.started<=7200
        e.write(self.out/'calibration.json',dict(p90_seconds_per_update=rates,validation_seconds=v,
                max_steps=MAX_STEPS,validation_interval_updates=INTERVAL,phase_remaining_seconds=left,
                projected_remaining_seconds=predicted,reserved_evaluation_seconds=reserve,admitted=admitted,
                allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3,
                reserved_gib=self.torch.cuda.max_memory_reserved()/1024**3))
        if not admitted:
            e.write(self.out/'summary.json',dict(status='not_admitted',reason='Fixed paired workload cannot fit measured budget or preflight exceeded two hours',
                    best=self.best,final_steps=self.completed_steps,training_seconds=self.training_seconds,
                    total_seconds=time.monotonic()-self.started,candidate=None,confirmation='not admitted',
                    transfer=dict(status='skipped',reason='Timing admission failed')))
            return
        e.event('calibrated',rates=rates,projected_remaining_hours=predicted/3600,phase_hours_remaining=left/3600)
        current=120;reason='maximum_updates';bad={a:0 for a in ARMS}
        milestones={a:baseline['chrf'] for a in ARMS};stale={a:0 for a in ARMS}
        while current<MAX_STEPS:
            target=min(current+INTERVAL,MAX_STEPS)
            need=((target-current)*sum(rates.values())+2*v+2*max(self.block_overheads,default=0.))*1.15+180
            if need>self.phase_end-time.monotonic():reason='training_phase_budget';break
            for arm in ARMS:
                rates[arm]=self.train_to(arm,target)
                _,m,duration=self.evaluate(self.dev,f'development-{arm}-{target:06d}')
                v=max(v,duration);self.history[arm].append(dict(step=target,metrics=m))
                best=self.best[arm]
                if e.eligible(m,baseline) and (m['chrf'],m['exact'],-target)>(best['metrics']['chrf'],best['metrics']['exact'],-best['step']):
                    p=self.out/f'{arm}-best.pt';self.save(p,self.load(self.states[arm]))
                    self.best[arm]=dict(step=target,metrics=m,file=str(p))
                bad[arm]=bad[arm]+1 if support.collapse(m) else 0
            current=target;plateau=plateau_update(self.best,milestones,stale)
            e.write(self.out/'selection.json',dict(best=self.best,history=self.history,collapse_counts=bad,
                    improvement_milestones=milestones,stale_checks=stale))
            e.event('paired_checkpoint',step=current,stale_checks=stale,phase_hours_remaining=max(0,self.phase_end-time.monotonic())/3600)
            if any(count>=3 for count in bad.values()):reason='persistent_severe_collapse';break
            if current>=12000 and plateau:reason='paired_plateau';break
        assert len(set(self.completed_steps.values()))==1
        candidate=choose_candidate(self.best,baseline)
        e.write(self.out/'candidate-locked.json',dict(candidate=candidate,best=self.best,reason=reason,paired_final_steps=self.completed_steps))
        summary=dict(status='completed_training',reason=reason,best=self.best,candidate=candidate,
                     final_steps=self.completed_steps,training_seconds=self.training_seconds,
                     elapsed_before_confirmation_seconds=time.monotonic()-self.started,
                     peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3)
        e.write(self.out/'summary.json',summary)
        summary['confirmation']=self.confirmation(candidate,baseline) if candidate else 'not admitted; development gate not passed'
        e.write(self.out/'summary.json',summary)
        if should_transfer(summary):
            summary['transfer']=dict(status='running',researcher_review='pending');e.write(self.out/'summary.json',summary)
            self.ae.eval();summary['transfer']=transfer.run(candidate)
            summary['confirmation']['unconditional_transfer']='automatic evaluation completed; researcher review pending'
        else:summary['transfer']=dict(status='skipped',reason='No candidate passed automatic reserve reconstruction gate')
        assert self.hash_frozen()==self.frozen_hash
        summary['total_seconds']=time.monotonic()-self.started;e.write(self.out/'summary.json',summary)
        e.event('finished',candidate=candidate,steps=self.completed_steps,total_hours=summary['total_seconds']/3600)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ae',required=True);p.add_argument('--model-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--gradient-check',action='store_true')
    p.add_argument('--evaluation-smoke',action='store_true');p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--wall-hours',type=float,default=16.)
    p.add_argument('--train-phase-hours',type=float,default=13.)
    p.add_argument('--memory-limit-gib',type=float,default=36.)
    p.add_argument('--deadline',default='2026-10-26T18:00:00+02:00')
    p.add_argument('--start-before',default='2026-10-26T02:00:00+02:00')
    args=p.parse_args();args.wall_started=time.monotonic()
    if not (args.wall_hours==16 and args.train_phase_hours==13):p.error('Approved budget is fixed at 16 total / 13 phase hours')
    if (args.gradient_check or args.evaluation_smoke) and not args.verify_only:p.error('Verification flags require --verify-only')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,e.stop_signal)
    exp=None
    try:
        if not args.verify_only and time.time()>datetime.fromisoformat(args.start_before).timestamp():
            raise RuntimeError('Latest start passed; preserve writing buffer')
        e.event('preflight_started',output=args.output);e.preflight(args)
        if args.cpu_only:return
        exp=Experiment(args);exp.run()
    except Exception as exc:
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        e.write(out/'failure.json',dict(error=repr(exc),training_seconds=getattr(exp,'training_seconds',0),
                completed_steps=getattr(exp,'completed_steps',{}),stop_requested=e.STOP))
        raise
    finally:
        if not args.cpu_only:
            from export_report import render
            render(Path(args.output))

if __name__=='__main__':main()
