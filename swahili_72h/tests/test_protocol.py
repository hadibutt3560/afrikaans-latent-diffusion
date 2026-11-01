import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import budget,read,write
from audit import spans
from metrics import chrf,aggregate
import run


class ProtocolTests(unittest.TestCase):
    def test_exact_text_spans_never_cross_lines(self):
        t='  '+('neno\t'*101)+'\r\nHabari za leo.\n'
        ranges=list(spans(t));self.assertEqual(''.join(t[a:b] for a,b in ranges),t)
        self.assertTrue(all(len(t[a:b].split())<=48 for a,b in ranges))
        self.assertTrue(all('\n' not in t[a:b].rstrip('\r\n') for a,b in ranges))

    def test_slow_sampling_shrinks_training_not_evaluation(self):
        fast=dict(sample=1,score=.05,encode=.01,validate=.005)
        slow={**fast,'sample':30}
        a=budget(18*3600,fast,4,256,55000);b=budget(18*3600,slow,4,256,55000)
        self.assertGreater(b['evaluation_reserve_seconds'],a['evaluation_reserve_seconds'])
        self.assertLess(b['remaining_train_seconds'],a['remaining_train_seconds'])
        self.assertAlmostEqual(b['train_end_seconds']+b['evaluation_reserve_seconds']+b['shutdown_seconds'],72*3600)
        for score_cost in (0.01,1,10,100):
            plan=budget(0,{**fast,'score':score_cost},4,256,55000)
            self.assertGreaterEqual(plan['evaluation_reserve_seconds'],sum(plan['stage_reserves'].values()))

    def test_metrics_empty_visible_and_identical_reconstruction(self):
        self.assertEqual(chrf('Habari za leo','Habari za leo'),100)
        self.assertIsNone(aggregate([dict(tokens=0,nll=0,empty=True,perplexity=None)])['token_weighted_perplexity'])

    def scenario(self,converged=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'package';out=Path(temp)/'out';root.mkdir();out.mkdir();(root/'data').mkdir()
            write(root/'config.json',read(Path(__file__).resolve().parents[1]/'config.json'))
            write(root/'data/audit.json',dict(record_counts=dict(train=1000,valid=100,test=100)))
            clock=[0.];events=[]
            class Fake(run.Runner):
                def stage(self,name,seconds,key=None,run=''):
                    key=key or name
                    if self.state.get(key,{}).get('complete'):return
                    events.append(name)
                    spent=seconds
                    if name=='benchmark':write(self.out/'benchmark.json',dict(sample=1,score=.1,encode=.01,validate=.01));spent=600
                    if name in ('selection','evaluate','scoring'):spent=600
                    if name=='diffusion_train' and converged:spent=1800
                    clock[0]+=spent;self.state[key]=dict(complete=True,seconds=spent)
            with patch.object(run.time,'time',side_effect=lambda:clock[0]):
                runner=Fake(root,out,72*3600);runner.run()
            self.assertLessEqual(clock[0],71*3600)
            self.assertLess(events.index('selection'),events.index('evaluate'))
            self.assertLess(events.index('evaluate'),events.index('scoring'))
            plan=read(out/'plan.json');self.assertEqual(sum(x=='diffusion_train' for x in events),len(plan['admitted_runs']))
            return len(plan['admitted_runs'])

    def test_full_allocation_does_not_overspend(self):self.scenario()
    def test_convergence_admits_useful_followups(self):self.assertGreater(self.scenario(True),4)

    def test_stage_completion_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);write(p/'config.json',{});write(p/'stages.json',{'cache':{'complete':True,'seconds':100}})
            runner=run.Runner(p,p,99999999999)
            with patch.object(run.subprocess,'Popen',side_effect=AssertionError('must not rerun')):runner.stage('cache',500)

    def test_checkpoint_budget_exhaustion_is_not_success(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);write(p/'config.json',{});write(p/'stages.json',{'cache':{'complete':False,'seconds':490}})
            runner=run.Runner(p,p,99999999999)
            with self.assertRaises(InterruptedError):runner.stage('cache',500)
            self.assertFalse(runner.state['cache']['complete'])

    def test_real_subprocess_failure_then_stage_resume(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'logs').mkdir();write(p/'config.json',{})
            (p/'worker.py').write_text('raise SystemExit(75)\n')
            runner=run.Runner(p,p,__import__('time').time()+10000)
            with self.assertRaises(RuntimeError):runner.stage('cache',500)
            self.assertFalse(read(p/'stages.json')['cache']['complete'])
            (p/'worker.py').write_text('print("resumed stage success")\n')
            runner=run.Runner(p,p,__import__('time').time()+10000);runner.stage('cache',500)
            self.assertTrue(read(p/'stages.json')['cache']['complete'])

    def test_signal_forwarding_requests_child_stop(self):
        from unittest.mock import Mock
        child=Mock();child.poll.return_value=None
        with patch.object(run,'CHILD',child),patch.object(run,'STOP',False):
            run.stop();self.assertTrue(run.STOP);child.terminate.assert_called_once()

    def test_resume_after_selection_never_admits_more_training(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data').mkdir();write(p/'config.json',read(Path(__file__).resolve().parents[1]/'config.json'))
            write(p/'data/audit.json',dict(record_counts=dict(train=1000,valid=100,test=100)))
            write(p/'benchmark.json',dict(sample=1,score=.1,encode=.01,validate=.01))
            write(p/'selection.json',{'frozen':True})
            write(p/'plan.json',dict(admitted_runs=['small-42'],candidates=['small-42','published-42'],
                run_seconds=7200,run_budgets={'small-42':7200},total_training_pool_seconds=40000,evaluation_reserve_seconds=14400))
            write(p/'stages.json',{k:{'complete':True,'seconds':1} for k in ['smoke','ae_train','ae_gate','cache','benchmark','train-small-42']})
            events=[]
            class Fake(run.Runner):
                def stage(self,name,seconds,key=None,run=''):
                    if not self.state.get(key or name,{}).get('complete'):events.append(name)
            Fake(p,p,__import__('time').time()+72*3600).run()
            self.assertNotIn('diffusion_train',events)
            self.assertEqual(read(p/'plan.json')['admitted_runs'],['small-42'])


if __name__=='__main__':unittest.main()
