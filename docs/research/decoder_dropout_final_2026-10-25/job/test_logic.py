"""Synthetic tests of the changed scientific/control paths; no research training."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import torch
from torch import nn
import numpy as np
import train_dropout as t
import transfer_eval as transfer
import export_report

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Embedding(5,4);self.lm_head=nn.Linear(4,5,bias=False)
        self.encoder=nn.Module();self.encoder.embed_tokens=self.shared
        self.perceiver_ae=nn.Linear(4,4)
        self.decoder=nn.Module();self.decoder.embed_tokens=self.shared
        self.decoder.block=nn.ModuleList(nn.Sequential(nn.Linear(4,4),nn.Dropout(.1)) for _ in range(12))
        self.decoder.final_layer_norm=nn.LayerNorm(4)
    def forward(self):
        x=torch.tensor([[.1,.2,.4,.8]])
        for b in self.decoder.block:x=torch.tanh(b(x))
        return self.lm_head(self.decoder.final_layer_norm(x)).square().mean()

class Checks(unittest.TestCase):
    def test_dropout_masks_rng_replay_and_frozen_parameters(self):
        names_by_arm=[]
        for arm in t.ARMS:
            torch.manual_seed(42);m=Tiny();names=t.configure_mask(m,arm);names_by_arm.append(names)
            self.assertTrue(any(n.startswith('decoder.block.0.') for n in names))
            self.assertEqual(m.decoder.training,arm=='D1')
            self.assertFalse(m.encoder.training);self.assertFalse(m.perceiver_ae.training)
            self.assertFalse(m.shared.weight.requires_grad);self.assertFalse(m.lm_head.weight.requires_grad)
            values=[float(m().detach()) for _ in range(3)]
            self.assertEqual(len(set(values))==1,arm=='D0')
            initial=copy.deepcopy(m.state_dict())
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5,foreach=False)
            def step():opt.zero_grad();m().backward();opt.step()
            step()
            for n,p in m.named_parameters():
                if n not in names:self.assertTrue(torch.equal(p,initial[n]),n)
                else:self.assertIsNotNone(p.grad,n)
            state=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
            step();expected=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=True)
            m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);step()
            self.assertTrue(t.e.tree_equal(expected,dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())))
            m.eval();self.assertEqual(float(m()),float(m()))
        self.assertEqual(names_by_arm[0],names_by_arm[1])

    def test_validation_disables_and_restores_dropout_even_on_error(self):
        from unittest.mock import patch
        exp=object.__new__(t.Experiment);exp.ae=Tiny();t.configure_mask(exp.ae,'D1')
        def evaluate(_self,panel,tag):
            self.assertFalse(exp.ae.decoder.training)
            self.assertEqual(float(exp.ae()),float(exp.ae()))
            if tag=='error':raise ValueError('fixture')
            return 1
        with patch.object(t.support.Experiment,'evaluate',evaluate):
            self.assertEqual(exp.evaluate([], 'ok'),1);self.assertTrue(exp.ae.decoder.training)
            with self.assertRaises(ValueError):exp.evaluate([],'error')
            self.assertTrue(exp.ae.decoder.training)

    def test_epochs_and_fixed_timing_admission(self):
        stream=t.support.EpochStream(11);expected=[]
        for epoch in range(6):
            values=list(range(11));random.Random(20261024+epoch).shuffle(values);expected+=values
        self.assertEqual(sum([stream[i:i+8] for i in range(0,56,8)],[]),expected[:56])
        self.assertEqual(stream[0:8],expected[:8])
        self.assertGreater(t.projected_seconds({'D0':1.,'D1':1.},90,30),13*3600)
        self.assertLess(t.projected_seconds({'D0':.2,'D1':.3},30,10),13*3600)
        self.assertEqual(min(26520+t.INTERVAL,t.MAX_STEPS),26620)

    def test_selection_does_not_replace_failed_treatment_with_control(self):
        b=dict(empty=0,cap=0,numeric_recall=21/33,numeric_precision=.84,repeat3=0.,chrf=67.70053259219789,exact=11)
        best={'D0':{'metrics':dict(b,chrf=72.,exact=16)},'D1':{'metrics':dict(b,chrf=70.05388982678022,exact=13)}}
        self.assertIsNone(t.choose_candidate(best,b))
        best['D1']['metrics']=dict(b,chrf=73.,exact=14)
        self.assertEqual(t.choose_candidate(best,b),'D1')
        best['D1']['metrics']['numeric_recall']=.5;self.assertIsNone(t.choose_candidate(best,b))

    def test_plateau_requires_both_arms_and_accumulates_small_gains(self):
        best={a:dict(metrics=dict(chrf=67.)) for a in t.ARMS};milestones={a:67. for a in t.ARMS};stale={a:0 for a in t.ARMS}
        for _ in range(5):self.assertFalse(t.plateau_update(best,milestones,stale))
        self.assertTrue(t.plateau_update(best,milestones,stale))
        best['D1']['metrics']['chrf']=67.4;self.assertTrue(t.plateau_update(best,milestones,stale))
        best['D1']['metrics']['chrf']=67.5;self.assertFalse(t.plateau_update(best,milestones,stale))
        self.assertEqual(stale['D1'],0)

    def test_reserve_and_both_review_sheets(self):
        exp=object.__new__(t.Experiment);exp.np=np;exp.panel={'reserved':[]};exp.union={}
        exp.best={a:dict(file=a) for a in t.ARMS};exp.load=lambda _:dict(model={});exp.restore=lambda _:None;exp.encode_panel=lambda _:[]
        with tempfile.TemporaryDirectory() as folder:
            exp.out=Path(folder)
            def evaluate(_,tag):
                arm=tag.split('-')[1];score={'B0':67.,'D0':69.,'D1':71.}[arm];exact={'B0':10,'D0':12,'D1':16}[arm]
                rows=[dict(id=str(i),document_id=str(i//2),source='een twee',text='een twee',chrf=score,
                    exact=int(i<exact),numeric_source=1,numeric_pred=1,numeric_match=1,repeat3=0.,empty=0,cap=0,nll=1.,tokens=2) for i in range(103)]
                m=t.e.aggregate(rows);t.e.write(exp.out/(tag+'.json'),dict(rows=rows,metrics=m));return rows,m,1.
            exp.evaluate=evaluate
            result=exp.confirmation('D1',{})
            self.assertTrue(result['automatic_reconstruction_gate_passed']);self.assertTrue(result['dropout_effect_supported'])
            self.assertEqual(result['dropout_minus_control_chrf_ci'],[2.,2.])
            self.assertTrue((exp.out/'researcher-review.csv').exists());self.assertTrue((exp.out/'researcher-control-review.csv').exists())

    def test_generation_guards_and_uncertainty(self):
        self.assertFalse(transfer.should_transfer({'candidate':None}))
        self.assertFalse(transfer.should_transfer({'candidate':'D1','confirmation':{'automatic_reconstruction_gate_passed':False}}))
        self.assertTrue(transfer.should_transfer({'candidate':'D1','confirmation':{'automatic_reconstruction_gate_passed':True}}))
        row=dict(seed=1,text='een twee drie',mgpt=dict(nll=3.,tokens=3),eos=True,length_cap=False,sentinel=False)
        baseline=transfer.metrics([row],set());self.assertTrue(transfer.transfer_gate(baseline,baseline))
        empty=transfer.metrics([dict(row,text='',mgpt=None)],set());self.assertFalse(transfer.transfer_gate(baseline,empty))
        self.assertFalse(transfer.transfer_gate(baseline,transfer.metrics([row],{'een twee drie'})))
        self.assertEqual(transfer.paired_intervals([row],[row],np)['candidate_over_B0_ppl'],[1.,1.])

    def test_failure_export_contains_checksums_and_diagnostics(self):
        import tarfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs/job-fixture';out.mkdir(parents=True)
            t.e.write(out/'failure.json',dict(error='fixture failure before training'))
            export_report.archive(out,root,'fixture',1)
            target=out.parent/'review-job-fixture.tar.gz'
            self.assertTrue(target.with_name(target.name+'.sha256').exists())
            with tarfile.open(target) as archive:
                self.assertIn('job-fixture/REPORT.md',archive.getnames())
                self.assertIn('job-fixture/artifact-manifest.json',archive.getnames())
            self.assertIn('fixture failure',(out/'REPORT.md').read_text())

if __name__=='__main__':unittest.main()
