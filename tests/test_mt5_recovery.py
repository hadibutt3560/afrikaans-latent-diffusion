import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import MT5Config
from latent_models.t5_latent_model import MT5ForConditionalGenerationLatent
from latent_models.latent_utils import get_latent_model, set_autoencoder_train_mode


class MT5RecoveryTests(unittest.TestCase):
    def test_backbone_eval_preserves_gradients(self):
        config = MT5Config(vocab_size=128, d_model=64, d_ff=128,
                           num_layers=1, num_decoder_layers=1, num_heads=2,
                           d_kv=32, decoder_start_token_id=0, pad_token_id=0)
        model = MT5ForConditionalGenerationLatent(config, 4, 4, 64, num_layers=1)
        set_autoencoder_train_mode(model, True)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.decoder.training)
        self.assertTrue(model.perceiver_ae.training)
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        set_autoencoder_train_mode(model, False)
        self.assertTrue(model.encoder.training)
        self.assertTrue(model.decoder.training)

    def test_selective_gradients_and_latent_connection(self):
        config = MT5Config(vocab_size=128, d_model=64, d_ff=128,
                           num_layers=2, num_decoder_layers=2, num_heads=2,
                           d_kv=32, decoder_start_token_id=0, pad_token_id=0)
        model = MT5ForConditionalGenerationLatent(config, 4, 4, 64, num_layers=1)
        args = SimpleNamespace(enc_dec_model='google/mt5-base', num_encoder_latents=4,
                               num_decoder_latents=4, dim_ae=64, num_layers=1,
                               l2_normalize_latents=False, lm_mode='decoder_last')
        with patch('latent_models.latent_utils.AutoConfig.from_pretrained', return_value=config), \
             patch('latent_models.latent_utils.AutoTokenizer.from_pretrained', return_value=None), \
             patch.object(MT5ForConditionalGenerationLatent, 'from_pretrained', return_value=model):
            model, _, _ = get_latent_model(args)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.decoder.block[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.decoder.block[-1].parameters()))
        model.eval()
        ids = torch.tensor([[4, 5, 6, 1], [7, 8, 9, 1]])
        mask = torch.ones_like(ids)
        encoded = model.get_encoder()(input_ids=ids, attention_mask=mask)
        latent = model.get_diffusion_latent(encoded, mask)
        self.assertEqual(tuple(latent.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(latent).all())
        self.assertFalse(torch.allclose(latent[0], latent[1]))
        encoded = model.encoder_output_to_decoder_input(encoded, mask)
        loss = model(encoder_outputs=encoded, labels=ids).loss
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.perceiver_ae.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.decoder.block[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
