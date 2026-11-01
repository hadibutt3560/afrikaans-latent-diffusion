import unittest
import torch
from tools.decoder_aware_cycle import auxiliary_objective


class DecoderAwareTests(unittest.TestCase):
    def test_zero_weight_is_unchanged_control(self):
        v=torch.tensor(.5,requires_grad=True)
        ce=torch.tensor(2.,requires_grad=True)
        loss=auxiliary_objective(v,ce,0.)
        self.assertEqual(float(loss),.5)
        loss.backward()
        self.assertEqual(float(v.grad),1.)
        self.assertEqual(float(ce.grad),0.)

    def test_frozen_decoder_transmits_gradient_to_latent(self):
        decoder=torch.nn.Linear(4,6).requires_grad_(False)
        latent=torch.randn(1,4,requires_grad=True)
        ce=torch.nn.functional.cross_entropy(decoder(latent),torch.tensor([2]))
        auxiliary_objective(latent.square().mean(),ce,.02).backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))


if __name__=='__main__':
    unittest.main()
