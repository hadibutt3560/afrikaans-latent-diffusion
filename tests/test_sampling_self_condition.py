import unittest
from types import SimpleNamespace
import torch
from diffusion.text_denoising_diffusion import GaussianDiffusion


class SelfConditionTests(unittest.TestCase):
    def check_case(self, enabled, sampling):
        seen = []
        def net(z, mask, t, condition, **kwargs):
            seen.append(condition)
            return torch.zeros_like(z)
        model = SimpleNamespace(normalize_sampling_self_condition=enabled,
            train_schedule=lambda t:t, sampling_schedule=lambda t:t,
            diffusion_model=net, objective='pred_noise',
            predict_start_from_noise=lambda z,t,n,sampling: z)
        z = torch.ones(1,2,4)
        condition = z * 3
        original = condition.clone()
        GaussianDiffusion.diffusion_model_predictions(model,z,None,torch.tensor([0.5]),
            x_self_cond=condition,sampling=sampling)
        self.assertTrue(torch.equal(condition,original))
        expected = z if enabled and sampling else original
        self.assertTrue(torch.allclose(seen[0],expected))

    def test_opt_in_sampling_normalizes_copy(self):
        self.check_case(True,True)

    def test_default_preserves_sampling(self):
        self.check_case(False,True)

    def test_training_unchanged(self):
        self.check_case(True,False)


if __name__ == '__main__':
    unittest.main()
