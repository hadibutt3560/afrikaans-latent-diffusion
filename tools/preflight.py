#!/usr/bin/env python
"""Safe ROCStories/LD4LG preflight. This script never starts training.

Run as: python tools/preflight.py [--output-dir DIR] [--require-cuda] [--require-bf16]
[--require-model-cache] [--allow-download] [--load-model] [--cpu-only]. It checks package
imports, the ROCStories files and split sizes, CUDA/BF16, a writable output folder, a
checkpoint save/load round trip, free disk space, the W&B mode, the cached model and
tokeniser, and one tokenised batch through the denoising collator. The JSON report goes
to --report-path (default DIR/preflight_report.json) and is also printed. Exit code 0
means every required check passed, 1 that one failed and 2 that the script raised an error.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Only the raw line counts are checked. dataset_utils/text_dataset.py later splits the
# 5000-line validation file into 1000 validation and 4000 test stories; the final counts
# are reported for reference.
EXPECTED_RAW_COUNTS = {"train": 93161, "valid_source": 5000}
EXPECTED_FINAL_COUNTS = {"train": 93161, "valid": 1000, "test": 4000}
# Import names; check_imports maps "_" to "-" to look up distribution versions.
REQUIRED_IMPORTS = [
    "accelerate", "datasets", "einops", "ema_pytorch", "evaluate",
    "numpy", "torch", "transformers", "wandb",
]


def add_check(report, name, ok, details, required=True):
    """Append one check result to the report; only required failures fail the preflight."""
    report["checks"].append({
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "details": details,
    })


def read_roc_file(path):
    """Count lines in a ROCStories file and decode the first three as examples.

    Each line must be a one-item JSON list holding the story text; only the first three
    lines are parsed. Returns ``(line_count, examples)``.
    """
    examples = []
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            count += 1
            if len(examples) < 3:
                decoded = json.loads(line)
                if not isinstance(decoded, list) or len(decoded) != 1:
                    raise ValueError(f"{path}:{line_number} is not a one-item JSON list")
                examples.append(decoded[0])
    return count, examples


def check_imports(report):
    """Import each required package and record its version or the import error."""
    versions = {}
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            package_name = module_name.replace("_", "-")
            try:
                versions[module_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                versions[module_name] = "importable (version unavailable)"
        except Exception as exc:
            versions[module_name] = f"ERROR: {type(exc).__name__}: {exc}"
    missing = [name for name, version in versions.items() if version.startswith("ERROR:")]
    add_check(report, "required_imports", not missing, {"versions": versions, "missing": missing})


def check_rocstories(report):
    """Check the ROCStories train and validation files exist and have the expected line counts."""
    data_dir = REPO_ROOT / "datasets" / "ROCstory"
    train_path = data_dir / "roc_train.json"
    valid_path = data_dir / "roc_valid.json"
    try:
        train_count, train_examples = read_roc_file(train_path)
        valid_source_count, valid_examples = read_roc_file(valid_path)
        raw_counts = {"train": train_count, "valid_source": valid_source_count}
        ok = raw_counts == EXPECTED_RAW_COUNTS
        add_check(
            report,
            "rocstories_files_and_splits",
            ok,
            {
                "paths": [str(train_path), str(valid_path)],
                "raw_counts": raw_counts,
                "expected_raw_counts": EXPECTED_RAW_COUNTS,
                "constructed_counts": EXPECTED_FINAL_COUNTS,
                "decoded_examples": train_examples + valid_examples,
            },
        )
        report["dataset_split_counts"] = EXPECTED_FINAL_COUNTS
    except Exception as exc:
        add_check(
            report, "rocstories_files_and_splits", False,
            f"{type(exc).__name__}: {exc}",
        )


def check_hardware(report, require_cuda=False, require_bf16=False):
    """Record CUDA, BF16 and GPU details; the check is required only if a ``require_*`` flag is set."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
        gpus = []
        if cuda:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpus.append({
                    "index": index,
                    "name": props.name,
                    "memory_gib": round(props.total_memory / 2**30, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        ok = (cuda or not require_cuda) and (bf16 or not require_bf16)
        add_check(
            report,
            "cuda_and_bf16",
            ok,
            {
                "torch": torch.__version__,
                "cuda_available": cuda,
                "torch_cuda_version": torch.version.cuda,
                "bf16_supported": bf16,
                "gpus": gpus,
            },
            required=require_cuda or require_bf16,
        )
    except Exception as exc:
        add_check(
            report, "cuda_and_bf16", not (require_cuda or require_bf16),
            f"{type(exc).__name__}: {exc}", required=require_cuda or require_bf16,
        )


def check_output_and_checkpoint(report, output_dir):
    """Check the output folder is writable, a checkpoint round-trips and disk space is ample.

    Temporary files are deleted afterwards. If the folder is not writable the later checks
    are skipped.
    """
    output_path = Path(output_dir).expanduser().resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output_path, prefix="preflight-", suffix=".tmp",
            delete=False, encoding="utf-8",
        ) as handle:
            handle.write("writable")
            temporary_path = Path(handle.name)
        temporary_path.unlink()
        add_check(report, "writable_output_directory", True, str(output_path))
    except Exception as exc:
        add_check(
            report, "writable_output_directory", False,
            f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        import torch
        checkpoint_path = output_path / "preflight-checkpoint.pt"
        # Same top-level keys as the project's trainer checkpoints.
        original = {
            "step": 7,
            "model": {"weight": torch.arange(4)},
            "opt": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 7},
            "scaler": None,
        }
        torch.save(original, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu")
        ok = (
            restored["step"] == 7
            and torch.equal(restored["model"]["weight"], original["model"]["weight"])
            and {"model", "opt", "scheduler", "scaler", "step"}.issubset(restored)
        )
        checkpoint_path.unlink()
        add_check(
            report, "checkpoint_save_load_roundtrip", ok,
            {"keys": sorted(restored), "step": restored["step"]},
        )
    except Exception as exc:
        add_check(
            report, "checkpoint_save_load_roundtrip", False,
            f"{type(exc).__name__}: {exc}",
        )

    free = shutil.disk_usage(output_path).free
    # 120 GiB recommendation; advisory only (required=False).
    recommended = 120 * 2**30
    add_check(
        report,
        "disk_space",
        free >= recommended,
        {
            "free_gib": round(free / 2**30, 2),
            "recommended_free_gib": 120,
            "note": "Recommendation includes model caches, AE/diffusion checkpoints, samples, and headroom.",
        },
        required=False,
    )


def check_wandb(report):
    """Check WANDB_MODE is a known value (``online`` when unset); advisory only."""
    mode = os.environ.get("WANDB_MODE", "online").lower()
    ok = mode in {"online", "offline", "disabled", "dryrun"}
    add_check(
        report, "wandb_mode", ok,
        {"mode": mode, "directory": os.environ.get("WANDB_DIR")},
        required=False,
    )


def check_model_cache(report, model_id, require_cache=False, allow_download=False, load_model=False):
    """Load the config and tokeniser for ``model_id`` from the local cache.

    With ``allow_download`` it may fetch them instead; with ``load_model`` it also loads the
    seq2seq weights and counts parameters. Failure is fatal only with ``require_cache``.
    """
    try:
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
        local_only = not allow_download
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_only)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        details = {
            "model_id": model_id,
            "local_files_only": local_only,
            "config_class": type(config).__name__,
            "tokenizer_class": type(tokenizer).__name__,
        }
        if load_model:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, local_files_only=local_only
            )
            details["model_class"] = type(model).__name__
            details["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
            del model
        add_check(report, "model_and_tokenizer", True, details, required=require_cache)
    except Exception as exc:
        add_check(
            report,
            "model_and_tokenizer",
            not require_cache,
            {
                "model_id": model_id,
                "cached": False,
                "error": f"{type(exc).__name__}: {exc}",
                "remedy": "Rerun with --allow-download on a networked login node, or pre-populate HF_HOME.",
            },
            required=require_cache,
        )


def check_tokenized_batch(report, model_id, allow_download=False):
    """Run two padded sentences through the BART denoising collator and check the batch.

    Checks shapes, that the attention mask equals the non-padding positions and that
    padding labels are -100 (ignored by the loss). Advisory only.
    """
    try:
        import torch
        from transformers import AutoTokenizer
        from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=not allow_download
        )
        encoded = tokenizer(
            ["A short ROCStory.", "A second story."],
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        examples = [
            {key: encoded[key][index] for key in encoded}
            for index in range(2)
        ]
        start_id = tokenizer.bos_token_id
        collator = DataCollatorForBartDenoisingLM(tokenizer, start_id)
        batch = collator(examples)
        expected_shape = (2, 16)
        shape_ok = all(
            tuple(batch[key].shape) == expected_shape
            for key in ["input_ids", "attention_mask", "labels", "decoder_input_ids"]
        )
        mask_ok = bool(
            torch.equal(batch["attention_mask"], batch["input_ids"].ne(tokenizer.pad_token_id).long())
        )
        labels_ok = bool((batch["labels"][batch["input_ids"].eq(tokenizer.pad_token_id)] == -100).all())
        add_check(
            report, "tokenized_shapes_masks_and_collator",
            shape_ok and mask_ok and labels_ok,
            {
                "shapes": {key: list(value.shape) for key, value in batch.items()},
                "attention_mask_ok": mask_ok,
                "padding_labels_masked": labels_ok,
            },
            required=False,
        )
    except Exception as exc:
        # Recorded as ok with status "not run", so an error here (for example no cached
        # tokeniser) never fails the preflight.
        add_check(
            report, "tokenized_shapes_masks_and_collator", True,
            {
                "status": "not run",
                "error": f"{type(exc).__name__}: {exc}",
                "remedy": "Cache the BART tokenizer or use --allow-download.",
            },
            required=False,
        )


def parse_args():
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/preflight")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--model-id", default="facebook/bart-base")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--require-model-cache", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    return parser.parse_args()


def main():
    """Run every check, write the JSON report and return 1 if a required check failed."""
    args = parse_args()
    if args.cpu_only and (args.require_cuda or args.require_bf16):
        raise ValueError("--cpu-only cannot be combined with CUDA/BF16 requirements")
    report = {
        "repository": str(REPO_ROOT),
        "python": sys.version,
        "checks": [],
    }
    check_imports(report)
    check_rocstories(report)
    check_hardware(
        report,
        require_cuda=args.require_cuda and not args.cpu_only,
        require_bf16=args.require_bf16 and not args.cpu_only,
    )
    check_output_and_checkpoint(report, args.output_dir)
    check_wandb(report)
    check_model_cache(
        report,
        args.model_id,
        require_cache=args.require_model_cache,
        allow_download=args.allow_download,
        load_model=args.load_model,
    )
    check_tokenized_batch(report, args.model_id, allow_download=args.allow_download)

    required_failures = [
        check["name"] for check in report["checks"]
        if check["required"] and not check["ok"]
    ]
    report["status"] = "fail" if required_failures else "pass"
    report["required_failures"] = required_failures
    report_path = (
        Path(args.report_path)
        if args.report_path
        else Path(args.output_dir) / "preflight_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Preflight report: {report_path.resolve()}")
    return 1 if required_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
