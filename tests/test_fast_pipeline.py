import argparse
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rocstories_source_and_constructed_split_counts():
    train_path = REPO_ROOT / "datasets" / "ROCstory" / "roc_train.json"
    valid_path = REPO_ROOT / "datasets" / "ROCstory" / "roc_valid.json"
    train = train_path.read_text(encoding="utf-8").splitlines()
    valid_source = valid_path.read_text(encoding="utf-8").splitlines()
    assert len(train) == 93161
    assert len(valid_source) == 5000
    assert len(valid_source[:1000]) == 1000
    assert len(valid_source[1000:]) == 4000
    assert isinstance(json.loads(train[0]), list)


def test_dataloader_order_is_deterministic_and_batch_sizes_are_split_specific():
    torch = pytest.importorskip("torch")
    from dataset_utils.text_dataset import get_batch_size, make_dataloader_generator

    first = torch.randperm(100, generator=make_dataloader_generator(42))
    second = torch.randperm(100, generator=make_dataloader_generator(42))
    third = torch.randperm(100, generator=make_dataloader_generator(43))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)

    args = argparse.Namespace(train_batch_size=7, eval_batch_size=3)
    assert get_batch_size(args, is_train=True) == 7
    assert get_batch_size(args, is_train=False) == 3


def test_bart_collator_shapes_and_padding_mask():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM

    class Tokenizer:
        pad_token_id = 1

    collator = DataCollatorForBartDenoisingLM(Tokenizer(), decoder_start_token_id=2)
    batch = collator([
        {"input_ids": [0, 5, 2, 1], "attention_mask": [1, 1, 1, 0]},
        {"input_ids": [0, 9, 8, 2], "attention_mask": [1, 1, 1, 1]},
    ])
    assert tuple(batch["input_ids"].shape) == (2, 4)
    assert tuple(batch["labels"].shape) == (2, 4)
    assert batch["labels"][0, 3].item() == -100
    assert torch.equal(batch["attention_mask"], batch["input_ids"].ne(1).long())
    assert torch.equal(batch["decoder_attention_mask"], batch["decoder_input_ids"].ne(1).long())


def test_cosine_schedule_endpoints_and_midpoint():
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("datasets")
    pytest.importorskip("ema_pytorch")
    pytest.importorskip("evaluate")
    pytest.importorskip("transformers")
    pytest.importorskip("wandb")
    from diffusion.text_denoising_diffusion import cosine_schedule

    values = cosine_schedule(torch.tensor([0.0, 0.5, 1.0]))
    assert values[0].item() == pytest.approx(1.0, abs=1e-6)
    assert values[1].item() == pytest.approx(0.5, abs=1e-6)
    assert values[2].item() <= 1e-6


def test_argument_parsing_accepts_correct_and_legacy_l2_flag():
    pytest.importorskip("accelerate")
    pytest.importorskip("transformers")
    from train_latent_model import build_parser

    correct = build_parser().parse_args(["--l2_normalize_latents"])
    legacy = build_parser().parse_args(["--l2_normalize_latent"])
    assert correct.l2_normalize_latents
    assert legacy.l2_normalize_latents


def test_diffusion_requires_latent_model_when_guard_is_set():
    pytest.importorskip("accelerate")
    pytest.importorskip("ema_pytorch")
    pytest.importorskip("transformers")
    from train_text_diffusion import build_parser

    args = build_parser().parse_args(["--require_latent_model"])
    assert args.require_latent_model
    assert args.latent_model_path is None


def test_run_metadata_contains_reproducibility_fields(tmp_path):
    pytest.importorskip("torch")
    from utils.file_utils import write_run_metadata

    args = argparse.Namespace(
        seed=42,
        train_batch_size=4,
        eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_devices=1,
    )
    metadata = write_run_metadata(
        tmp_path,
        args,
        dataset_counts={"train": 10, "valid": 2, "test": 3},
        model_counts={"total": 7, "trainable": 5},
    )
    assert metadata["batching"]["global_train_batch_size"] == 32
    assert metadata["git"]["commit"]
    assert (tmp_path / "args.json").is_file()
    assert (tmp_path / "run_metadata.json").is_file()
    assert (tmp_path / "package_versions.txt").is_file()


def test_checkpoint_roundtrip_restores_all_training_state(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint = {
        "step": 1,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None,
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    restored = torch.load(path, map_location="cpu")
    assert restored["step"] == 1
    assert {"model", "opt", "scheduler", "scaler", "step"} <= restored.keys()


def test_metric_empty_cases_do_not_divide_by_zero():
    pytest.importorskip("torch")
    pytest.importorskip("evaluate")
    from evaluation.evaluation import (
        compute_diversity,
        compute_memorization,
        compute_unique_sentence_count,
    )

    metrics = compute_diversity(["", "one"])
    assert metrics["distinct_4"] == 0.0
    assert compute_memorization([""], ["reference text"]) == 0.0
    assert compute_unique_sentence_count(["", "same", "same"]) == 1
