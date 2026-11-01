import pytest


def test_perceiver_autoencoder_forward_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from latent_models.perceiver_ae import PerceiverAutoEncoder

    model = PerceiverAutoEncoder(
        dim_lm=16,
        dim_ae=8,
        num_encoder_latents=4,
        num_decoder_latents=6,
        depth=1,
    )
    encoder_outputs = torch.randn(2, 10, 16)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    encoded = model.encode(encoder_outputs, attention_mask)
    decoded = model.decode(encoded)
    assert tuple(encoded.shape) == (2, 4, 8)
    assert tuple(decoded.shape) == (2, 6, 16)


def test_diffusion_transformer_forward_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    from model.diffusion_transformer import DiffusionTransformer

    model = DiffusionTransformer(
        tx_dim=64,
        tx_depth=1,
        heads=1,
        latent_dim=8,
        max_seq_len=4,
    )
    inputs = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    timestep = torch.tensor([0.25, 0.75])
    outputs = model(inputs, mask, timestep)
    assert tuple(outputs.shape) == tuple(inputs.shape)
