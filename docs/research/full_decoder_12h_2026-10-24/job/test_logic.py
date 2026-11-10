"""Synthetic CPU tests. No research checkpoint optimizer updates."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
import train_full as t
import transfer_eval as transfer
import export_report

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Embedding(5,4);self.lm_head=nn.Linear(4,5,bias=False)
        self.encoder=nn.Linear(4,4);self.perceiver_ae=nn.Linear(4,4)
        self.decoder=nn.Module();self.decoder.embed_tokens=self.shared
        self.decoder.block=nn.ModuleList(nn.Linear(4,4) for _ in range(12))
        self.decoder.final_layer_norm=nn.LayerNorm(4)
    def forward(self):
        x=torch.tensor([[.1,.2,.4,.8]])
        for b in self.decoder.block:x=torch.tanh(b(x))
        return self.lm_head(self.decoder.final_layer_norm(x)).square().mean()

class Checks(unittest.TestCase):
    def test_masks_gradient_and_resume(self):
        for arm in t.ARMS:
            torch.manual_seed(42);m=Tiny();names=t.configure_mask(m,arm)
            self.assertEqual(any(n.startswith('decoder.block.0.') for n in names),arm=='F12')
            self.assertFalse(m.shared.weight.requires_grad);self.assertFalse(m.lm_head.weight.requires_grad)
            initial=copy.deepcopy(m.state_dict())
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5,foreach=False)
            def step():opt.zero_grad();m().backward();opt.step()
            step()
            for n,p in m.named_parameters():
                if n not in names:self.assertTrue(torch.equal(p,initial[n]),n)
                else:self.assertIsNotNone(p.grad,n)
            state=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict()))
            step();expected=copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict()))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=True)
            m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step()
            self.assertTrue(t.e.tree_equal(expected,dict(model=m.state_dict(),optimizer=opt.state_dict())))

    def test_epochs_no_dropped_tail_and_reproducibility(self):
        stream=t.EpochStream(11)
        expected=[]
        for epoch in range(6):
            values=list(range(11));random.Random(20261024+epoch).shuffle(values);expected+=values
        actual=sum([stream[i:i+8] for i in range(0,56,8)],[])
        self.assertEqual(actual,expected[:56])
        self.assertEqual(stream[0:8],expected[:8]) # cache eviction does not alter replay
        self.assertEqual(t.EpochStream(11)[40:48],expected[40:48])
        real=t.EpochStream(47767)
        self.assertEqual(len(real[5999*8:6000*8]),8)
        self.assertEqual(set(real[0:47767]),set(range(47767)))

    def test_timing_not_quality_controls_horizon(self):
        rates={'C2':.44,'F12':1.3}
        step=120;remaining=10*3600;iterations=0
        while True:
            target=t.plan_block(remaining,rates,80,step,800)
            if target is None:break
            remaining-=(target-step)*sum(rates.values())+160+60
            step=target;iterations+=1
        self.assertGreater(step,6000)
        self.assertGreater(10*3600-remaining,9*3600)
        self.assertLess(iterations,100)
        self.assertIsNone(t.plan_block(200,rates,80,120,800))
        self.assertFalse(t.collapse(dict(n=81,empty=0,repeat3=0,numeric_recall=.5,chrf=60)))

    def test_selection_rejects_small_gain_or_fidelity_regression(self):
        b=dict(empty=0,cap=0,numeric_recall=.64,numeric_precision=.84,repeat3=0.,chrf=67.,exact=11)
        best={'C2':{'metrics':dict(b,chrf=68.,exact=12)},'F12':{'metrics':dict(b,chrf=75.,exact=18,numeric_recall=.5)}}
        self.assertIsNone(t.e.choose_candidate(best,b))
        best['F12']['metrics']=dict(b,chrf=72.,exact=16)
        self.assertEqual(t.e.choose_candidate(best,b),'F12')

    def test_transfer_gates_and_empty_outputs(self):
        self.assertFalse(transfer.should_transfer({'candidate':None}))
        self.assertFalse(transfer.should_transfer({'candidate':'F12','confirmation':{'automatic_reconstruction_gate_passed':False}}))
        self.assertTrue(transfer.should_transfer({'candidate':'F12','confirmation':{'automatic_reconstruction_gate_passed':True}}))
        row=dict(seed=1,text='een twee drie',mgpt=dict(nll=3.,tokens=3),eos=True,length_cap=False,sentinel=False)
        baseline=transfer.metrics([row],set())
        self.assertTrue(transfer.transfer_gate(baseline,baseline))
        empty=transfer.metrics([dict(row,text='',mgpt=None)],set())
        self.assertIsNone(empty['ppl']);self.assertFalse(transfer.transfer_gate(baseline,empty))
        copy=transfer.metrics([row],{'een twee drie'})
        self.assertFalse(transfer.transfer_gate(baseline,copy))
        import numpy as np
        ci=transfer.paired_intervals([row],[row],np)
        self.assertEqual(ci['candidate_over_B0_ppl'],[1.,1.])
        self.assertEqual(ci['candidate_minus_B0_mean_words'],[0.,0.])

    def test_report_on_negative_outcome(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'summary.json').write_text(json.dumps(dict(status='completed_training',candidate=None,confirmation='not admitted',transfer=dict(status='skipped'))))
            export_report.render(out)
            self.assertIn('Selected candidate: none',(out/'REPORT.md').read_text())
            self.assertTrue((out/'MANUSCRIPT_INSERT.md').exists())

if __name__=='__main__':unittest.main()
