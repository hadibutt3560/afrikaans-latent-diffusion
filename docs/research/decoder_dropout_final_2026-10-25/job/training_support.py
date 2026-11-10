"""Time-budgeted paired decoder adaptation; no quality-plateau early stopping."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import signal
import time

import engine as e

ARMS = ('D0', 'D1')


def configure_mask(ae, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    ae.eval().requires_grad_(False)
    blocks = ae.decoder.block[-2:] if arm == 'D0' else ae.decoder.block
    for block in blocks:
        block.requires_grad_(True)
    ae.decoder.final_layer_norm.requires_grad_(True)
    names = [n for n,p in ae.named_parameters() if p.requires_grad]
    prefixes = ('decoder.block.', 'decoder.final_layer_norm.') if arm == 'D1' else (
        'decoder.block.10.', 'decoder.block.11.', 'decoder.final_layer_norm.')
    assert names and all(n.startswith(prefixes) for n in names)
    assert not ae.shared.weight.requires_grad and not ae.lm_head.weight.requires_grad
    assert not any(m.training for m in ae.modules())
    return names


e.configure_mask = configure_mask


class EpochStream:
    """Concatenated full permutations: retain the tail at an epoch boundary."""
    def __init__(self, size, seed=20261024):
        if size <= 0: raise ValueError('Empty training corpus')
        self.size, self.seed, self.cache = size, seed, OrderedDict()

    def permutation(self, epoch):
        if epoch not in self.cache:
            order = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(order)
            self.cache[epoch] = order
            if len(self.cache)>3: self.cache.popitem(last=False)
        return self.cache[epoch]

    def __getitem__(self, key):
        assert isinstance(key,slice) and key.step is None and key.start>=0
        return [self.permutation(i//self.size)[i%self.size] for i in range(key.start,key.stop)]


def plan_block(seconds_left, rates, validation_seconds, current_step, interval):
    """Schedule from timing alone; reserve validation, saves and shutdown margin."""
    unit = sum(rates.values()) * 1.15
    budget = seconds_left - 2*validation_seconds*1.15 - 180
    fit = max(0, math.floor(budget / unit / 100)*100)
    advance = min(interval, fit)
    return current_step+advance if advance>=100 else None


def collapse(metrics):
    # Fidelity regressions disqualify checkpoints, but do not trigger a plateau stop.
    return metrics['empty']/metrics['n']>=.25 or metrics['repeat3']>=.20


class Experiment(e.Experiment):
    def __init__(self,args):
        super().__init__(args)
        self.started = args.wall_started
        self.hard_end = min(self.started+args.wall_hours*3600,
                            time.monotonic()+self.deadline-time.time())
        self.phase_end = min(self.started+args.train_phase_hours*3600,self.hard_end-3600)
        self.order = EpochStream(len(self.rows))
        self.completed_steps = {a:0 for a in ARMS}
        self.grad_checked = set()
        self.arm = None
        self.block_rates = {}
        self.block_overheads = []
        e.event('budget',wall_hours=args.wall_hours,train_phase_hours=args.train_phase_hours,
                training_phase_remaining_seconds=self.phase_end-time.monotonic(),
                deadline=args.deadline,plateau_stopping=True)

    def remaining(self):
        if hasattr(self,'hard_end'): return self.hard_end-time.monotonic()
        return min(self.deadline-time.time(),self.args.wall_hours*3600-(time.monotonic()-self.args.wall_started))

    def guard(self,training=False):
        if e.STOP or self.remaining()<=180:
            raise InterruptedError('Signal, job wall clock or absolute deadline reached')
        if training and hasattr(self,'phase_end') and time.monotonic()>=self.phase_end:
            raise InterruptedError('Training-phase budget reached; current state will be saved')
        if self.torch.cuda.max_memory_allocated()>self.args.memory_limit_gib*1024**3:
            raise RuntimeError('Memory admission limit exceeded')

    def assert_control_frozen(self):
        if self.arm=='D0':
            names=set(configure_mask(self.ae,'D0'))
            assert all(self.torch.equal(p.detach().cpu(),self.union[n])
                       for n,p in self.ae.named_parameters() if n in self.union and n not in names)

    def optimizer(self,arm):
        self.arm=arm
        opt,names=super().optimizer(arm)
        return opt,names

    def inspect_gradients(self):
        blocks = range(12)
        totals={i:0. for i in blocks}
        for n,p in self.ae.named_parameters():
            if not p.requires_grad: continue
            if p.grad is None or not self.torch.isfinite(p.grad).all():
                raise RuntimeError('Missing/nonfinite gradient: '+n)
            if n.startswith('decoder.block.'):
                totals[int(n.split('.')[2])]+=float(p.grad.detach().abs().sum())
        assert all(x>0 for x in totals.values()), totals
        e.event('gradient_path_verified',arm=self.arm,block_gradient_abs_sums=totals)
        return totals

    def update(self,opt,step):
        # A pre-step hook verifies the whole gradient path before the first real update.
        if self.arm not in self.grad_checked:
            def check(_opt,_args,_kwargs):
                self.inspect_gradients();self.grad_checked.add(self.arm)
            hook=opt.register_step_pre_hook(check)
            try: return super().update(opt,step)
            finally: hook.remove()
        return super().update(opt,step)

    def roundtrip(self,arm,opt):
        before=self.snapshot(opt,0)
        self.save(self.out/f'{arm}-roundtrip.pt',before)
        opt=self.restore_snapshot(arm,self.load(self.out/f'{arm}-roundtrip.pt'))
        self.update(opt,0); one=self.snapshot(opt,1)
        opt=self.restore_snapshot(arm,self.load(self.out/f'{arm}-roundtrip.pt'))
        self.update(opt,0); two=self.snapshot(opt,1)
        assert e.tree_equal(one,two),'Save/load continuation differs'
        assert any(not self.torch.equal(one['model'][n],before['model'][n]) for n in one['model'])
        self.assert_control_frozen()
        opt=self.restore_snapshot(arm,self.load(self.out/f'{arm}-roundtrip.pt'))
        assert self.hash_frozen()==self.frozen_hash
        e.write(self.out/f'{arm}-roundtrip.json',dict(exact=True,frozen_unchanged=True))
        return opt

    def train_to(self,arm,target,calibrate=False):
        block_start=time.monotonic()
        if arm in self.states:
            state=self.load(self.states[arm]);opt=self.restore_snapshot(arm,state);step=state['step']
            del state
        else:
            self.restore(self.union);opt,_=self.optimizer(arm);step=0
            start=time.monotonic();opt=self.roundtrip(arm,opt)
            self.training_seconds+=time.monotonic()-start
        samples=[];update_seconds=0.
        try:
            while step<target:
                start=time.monotonic();value=self.update(opt,step)
                duration=time.monotonic()-start;update_seconds+=duration
                self.training_seconds+=duration
                if step>=20:samples.append(duration)
                step+=1
                if step%100==0 or step==target:
                    e.event('train',arm=arm,step=step,target=target,ce=value,
                            epoch_exposure=step*8/len(self.rows),training_seconds=self.training_seconds,
                            phase_hours_remaining=max(0,self.phase_end-time.monotonic())/3600)
        finally:
            p=self.out/f'{arm}-latest.pt';self.save(p,self.snapshot(opt,step));self.states[arm]=p
            self.completed_steps[arm]=step
            e.write(self.out/'progress.json',dict(states={k:str(v) for k,v in self.states.items()},
                    steps=self.completed_steps,training_seconds=self.training_seconds,
                    elapsed_seconds=time.monotonic()-self.started,stop_requested=e.STOP))
        assert self.hash_frozen()==self.frozen_hash
        self.assert_control_frozen()
        self.block_overheads.append(time.monotonic()-block_start-update_seconds)
        self.block_rates[arm]=float(self.np.percentile(samples,90))
        e.event('block_finished',arm=arm,step=step,seconds=time.monotonic()-block_start)
        del opt
        self.ae.zero_grad(set_to_none=True)
        return self.block_rates[arm]

    def gradient_verification(self):
        from transformers.modeling_outputs import BaseModelOutput
        result={}
        for arm in ARMS:
            self.restore(self.union);self.arm=arm;configure_mask(self.ae,arm)
            self.ae.zero_grad(set_to_none=True)
            i=self.order[0:1][0]
            with self.torch.no_grad():
                hidden=self.ae.get_decoder_input(self.latents[i:i+1].cuda())
            labels=self.torch.tensor([self.labels[i]],device='cuda')
            loss=self.ae(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),labels=labels,use_cache=False).loss
            loss.backward()
            result[arm]=self.inspect_gradients()
            self.ae.zero_grad(set_to_none=True)
        assert all(self.torch.equal(p.detach().cpu(),self.union[n]) for n,p in self.ae.named_parameters() if n in self.union)
        assert self.hash_frozen()==self.frozen_hash
        e.write(self.out/'gradient-verification.json',dict(gradients=result,optimizer_updates=0,
                    all_weights_unchanged=True,peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3))

    def confirmation(self,candidate,baseline):
        result=super().confirmation(candidate,baseline)
        result['full_minus_two_chrf_ci']=result.pop('two_minus_one_chrf_ci')
        result['full_decoder_effect_supported']=result.pop('additional_block_effect_supported')
        return result

    def run(self):
        if self.args.verify_only:
            super().run()
            if self.args.gradient_check:self.gradient_verification()
            if self.args.evaluation_smoke:
                from transfer_eval import Transfer
                transfer=Transfer(self);transfer.prepare()
                transfer.run('SMOKE_IDENTITY',count=4,smoke=True)
            return
        self.restore(self.union)
        rows,baseline,v=self.evaluate(self.dev,'development-B0')
        reference=e.read(e.ROOT/'baseline.json')
        assert len(rows)==len(reference)==81
        assert all(r['generated_ids']==ref['correct']['generated_ids'] for r,ref in zip(rows,reference))
        self.save(self.out/'baseline-delta.pt',dict(model=self.union,step=0))
        for arm in ARMS:self.best[arm]=dict(step=0,metrics=baseline,file=str(self.out/'baseline-delta.pt'))
        from transfer_eval import Transfer,should_transfer
        transfer=Transfer(self)
        reserve=transfer.prepare()
        self.phase_end=min(self.phase_end,self.hard_end-reserve)
        rates={arm:self.train_to(arm,120,calibrate=True) for arm in ARMS}
        interval=max(100,min(5000,math.floor(1800/(sum(rates.values())*1.15)/100)*100))
        left=self.phase_end-time.monotonic()
        admitted=left>=6*3600 and time.monotonic()-self.started<=7200
        e.write(self.out/'calibration.json',dict(p90_seconds_per_update=rates,validation_seconds=v,
                validation_interval_updates=interval,phase_remaining_seconds=left,admitted=admitted,
                expected_remaining_updates=int(max(0,left)/(sum(rates.values())*1.15)),
                allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3,
                reserved_gib=self.torch.cuda.max_memory_reserved()/1024**3))
        if not admitted:
            e.write(self.out/'summary.json',dict(status='not_admitted',reason='Insufficient six-hour window or preflight exceeded two hours'))
            return
        e.event('calibrated',rates=rates,validation_interval_updates=interval,phase_hours_remaining=left/3600)
        current=120;reason='training_phase_budget';bad={a:0 for a in ARMS}
        while True:
            # Include observed save/load cost; only timing can adjust exposure.
            available=self.phase_end-time.monotonic()-2*max(self.block_overheads,default=0.)
            target=plan_block(available,rates,v,current,interval)
            if target is None:break
            for arm in ARMS:
                rates[arm]=self.train_to(arm,target)
                _,m,duration=self.evaluate(self.dev,f'development-{arm}-{target:06d}')
                v=max(v,duration)
                self.history[arm].append(dict(step=target,metrics=m))
                best=self.best[arm]
                if e.eligible(m,baseline) and (m['chrf'],m['exact'],-target)>(best['metrics']['chrf'],best['metrics']['exact'],-best['step']):
                    p=self.out/f'{arm}-best.pt';self.save(p,self.load(self.states[arm]))
                    self.best[arm]=dict(step=target,metrics=m,file=str(p))
                bad[arm]=bad[arm]+1 if collapse(m) else 0
                e.write(self.out/'selection.json',dict(best=self.best,history=self.history,collapse_counts=bad))
            current=target
            e.event('paired_checkpoint',step=current,phase_hours_remaining=max(0,self.phase_end-time.monotonic())/3600,
                    rates=rates,plateau_stopping=True)
            if any(count>=3 for count in bad.values()):
                reason='persistent_severe_collapse';break
        assert len(set(self.completed_steps.values()))==1
        candidate=e.choose_candidate(self.best,baseline)
        e.write(self.out/'candidate-locked.json',dict(candidate=candidate,best=self.best,reason=reason,
                paired_final_steps=self.completed_steps))
        summary=dict(status='completed_training',reason=reason,best=self.best,candidate=candidate,
                     final_steps=self.completed_steps,training_seconds=self.training_seconds,
                     elapsed_before_confirmation_seconds=time.monotonic()-self.started,
                     peak_allocated_gib=self.torch.cuda.max_memory_allocated()/1024**3)
        e.write(self.out/'summary.json',summary)
        summary['confirmation']=self.confirmation(candidate,baseline) if candidate else 'not admitted; development gate not passed'
        e.write(self.out/'summary.json',summary)
        if should_transfer(summary):
            summary['transfer']=dict(status='running',researcher_review='pending')
            e.write(self.out/'summary.json',summary)
            summary['transfer']=transfer.run(candidate)
            summary['confirmation']['unconditional_transfer']='automatic evaluation completed; researcher review pending'
        else:
            summary['transfer']=dict(status='skipped',reason='No candidate passed automatic reserve reconstruction gate')
        assert self.hash_frozen()==self.frozen_hash
        summary['total_seconds']=time.monotonic()-self.started
        e.write(self.out/'summary.json',summary)
        e.event('finished',candidate=candidate,steps=self.completed_steps,
                training_hours=self.training_seconds/3600,total_hours=summary['total_seconds']/3600)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ae',required=True);p.add_argument('--model-cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--gradient-check',action='store_true')
    p.add_argument('--evaluation-smoke',action='store_true')
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--wall-hours',type=float,default=12.)
    p.add_argument('--train-phase-hours',type=float,default=10.5)
    p.add_argument('--memory-limit-gib',type=float,default=36.)
    p.add_argument('--deadline',default='2026-10-26T12:00:00+02:00')
    p.add_argument('--start-before',default='2026-10-26T00:00:00+02:00')
    args=p.parse_args();args.wall_started=time.monotonic()
    if not (6<=args.train_phase_hours<=10.5 and args.train_phase_hours+1<=args.wall_hours<=12):
        p.error('Budget must leave >=1 hour for final checks within 12 hours')
    if args.gradient_check and not args.verify_only:p.error('Gradient check requires --verify-only')
    if args.evaluation_smoke and not args.verify_only:p.error('Evaluation smoke requires --verify-only')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,e.stop_signal)
    e.event('preflight_started',output=args.output)
    e.preflight(args)
    if args.cpu_only:return
    exp=None
    try:
        exp=Experiment(args);exp.run()
    except Exception as exc:
        if exp is not None:
            e.write(exp.out/'failure.json',dict(error=repr(exc),training_seconds=exp.training_seconds,
                    completed_steps=exp.completed_steps,stop_requested=e.STOP))
        raise
    finally:
        if exp is not None and not args.verify_only:
            from export_report import render
            render(exp.out)


if __name__=='__main__':main()
