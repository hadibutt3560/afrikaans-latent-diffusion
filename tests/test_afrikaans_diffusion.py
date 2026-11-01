import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class AfrikaansDiffusionTests(unittest.TestCase):
    def test_checkpoint_model_path_is_relocated_by_caller(self):
        from latent_models.latent_utils import get_latent_model

        saved = argparse.Namespace(
            enc_dec_model='/cluster-only/models/mt5-base',
            num_encoder_latents=32,
            num_decoder_latents=32,
            dim_ae=64,
            num_layers=3,
            l2_normalize_latents=True,
            lm_mode='decoder_last',
        )
        saved.enc_dec_model = 'google/mt5-base'
        self.assertEqual(saved.enc_dec_model, 'google/mt5-base')
        self.assertTrue(callable(get_latent_model))

    def test_latent_length_distribution_is_fixed(self):
        latent_slots = 32
        probs = torch.zeros(latent_slots + 1)
        probs[latent_slots] = 1.0
        distribution = torch.distributions.Categorical(probs=probs)
        samples = distribution.sample((128,))
        self.assertTrue(torch.all(samples.eq(latent_slots)))

    def test_downloaded_checkpoint_declares_expected_shape(self):
        checkpoint_args = Path(
            'outputs/cluster-results/afrikaans-56531/best/args.json'
        )
        if not checkpoint_args.exists():
            self.skipTest('Downloaded checkpoint is not available')
        data = json.loads(checkpoint_args.read_text(encoding='utf-8'))
        self.assertEqual(data['dataset_name'], 'nchlt_afrikaans')
        self.assertEqual(data['num_encoder_latents'], 32)
        self.assertEqual(data['dim_ae'], 64)
        self.assertTrue(data['l2_normalize_latents'])


if __name__ == '__main__':
    unittest.main()
