import argparse

import pytest


@pytest.mark.integration
def test_cached_bart_tokenizer_dataset_and_latent_forward(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    from dataset_utils.text_dataset import get_dataloader, get_dataset
    from latent_models.latent_utils import get_latent_model

    try:
        transformers.AutoTokenizer.from_pretrained(
            "facebook/bart-base", local_files_only=True
        )
    except Exception:
        pytest.skip("facebook/bart-base is not cached")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    args = argparse.Namespace(
        dataset_name="roc",
        enc_dec_model="facebook/bart-base",
        train_batch_size=2,
        eval_batch_size=3,
        dataloader_num_workers=0,
        seed=42,
        max_seq_len=16,
        num_encoder_latents=4,
        num_decoder_latents=4,
        dim_ae=8,
        num_layers=1,
        l2_normalize_latents=False,
        lm_mode="freeze",
    )
    try:
        model, tokenizer, config = get_latent_model(args)
    except OSError:
        pytest.skip("complete facebook/bart-base weights are not cached")
    dataset = get_dataset("roc")
    loader = get_dataloader(
        args,
        dataset["valid"].select(range(3)),
        config,
        tokenizer,
        args.max_seq_len,
        shuffle=False,
        is_train=False,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        encoder = model.get_encoder()(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        latent = model.get_diffusion_latent(encoder, batch["attention_mask"])
    assert tuple(batch["input_ids"].shape) == (3, 16)
    assert tuple(latent.shape) == (3, 4, 8)
