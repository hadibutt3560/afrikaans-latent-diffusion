"""CPU tests with synthetic tensors, never updates the research checkpoint."""
import unittest
import tempfile
from pathlib import Path
import copy
import torch
from torch import nn
from train_decoder import configure_mask, tree_equal, eligible, choose_candidate, choose_horizon

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(2,2)
        self.perceiver_ae=nn.Linear(2,2)
        self.decoder=nn.Module()
        self.decoder.block=nn.ModuleList(nn.Linear(2,2) for _ in range(12))
        self.decoder.final_layer_norm=nn.LayerNorm(2)

class Checks(unittest.TestCase):
    def test_masks_and_save_resume(self):
        for arm in ('C1','T2'):
            torch.manual_seed(9); m=Tiny(); names=configure_mask(m,arm)
            self.assertEqual(any(n.startswith('decoder.block.10.') for n in names),arm=='T2')
            original=copy.deepcopy(m.state_dict())
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01,foreach=False)
            def step():
                opt.zero_grad(); x=torch.tensor([[1.,2.]])
                for b in m.decoder.block:x=b(x)
                loss=m.decoder.final_layer_norm(x).square().sum();loss.backward();opt.step()
            step()
            for n,p in m.named_parameters():
                if n not in names:self.assertTrue(torch.equal(p,original[n]))
            checkpoint=copy.deepcopy(dict(model=m.state_dict(),opt=opt.state_dict()))
            step(); continuous=copy.deepcopy(dict(model=m.state_dict(),opt=opt.state_dict()))
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'state.pt';torch.save(checkpoint,p);loaded=torch.load(p,weights_only=True)
            m.load_state_dict(loaded['model']);opt.load_state_dict(loaded['opt']);step()
            self.assertTrue(tree_equal(continuous,dict(model=m.state_dict(),opt=opt.state_dict())))

    def test_admission_and_selection(self):
        b=dict(empty=0,cap=0,numeric_recall=.64,numeric_precision=.84,repeat3=0.,chrf=67.,exact=11)
        bad=dict(b,chrf=75.,numeric_recall=.57,exact=18)
        self.assertFalse(eligible(bad,b))
        c=dict(b,chrf=70.2,exact=14);t=dict(b,chrf=70.5,exact=15)
        self.assertEqual(choose_candidate({'C1':{'metrics':c},'T2':{'metrics':t}},b),'C1')
        t['chrf']=72.
        self.assertEqual(choose_candidate({'C1':{'metrics':c},'T2':{'metrics':t}},b),'T2')
        self.assertEqual(choose_horizon(3,3,200,24*3600),3000)
        self.assertEqual(choose_horizon(100,100,200,3600),0)

if __name__=='__main__':unittest.main()
