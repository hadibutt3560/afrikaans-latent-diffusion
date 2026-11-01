from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys


def ensure_output_dir(output_dir):
    """Create an output directory and all parents, then return its Path."""
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_output_dir(args):
    run_id = (
        os.environ.get("LD4LG_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    model_dir = f'{Path(args.dataset_name).stem}/{run_id}'
    output_dir = os.path.join(args.save_dir, model_dir)
    ensure_output_dir(output_dir)
    print(f'Created {output_dir}')
    return output_dir


def save_text_samples(all_texts_list, save_path):
    ensure_output_dir(Path(save_path).parent)
    full_text = '\n'.join(all_texts_list)
    with open(save_path, 'w', encoding='utf-8') as fo:
        fo.write(full_text)


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy, PyTorch, and all visible CUDA devices."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def _run_git(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state():
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("branch", "--show-current")
    status = _run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def package_versions():
    packages = {}
    for name in [
        "accelerate", "datasets", "einops", "ema-pytorch", "evaluate",
        "mauve-text", "numpy", "rouge-score", "sacrebleu", "sentencepiece",
        "spacy", "torch", "transformers", "wandb", "x-transformers",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def hardware_state():
    state = {"cuda_available": False, "cuda_version": None, "gpus": []}
    try:
        import torch
        state["torch_version"] = torch.__version__
        state["cuda_available"] = torch.cuda.is_available()
        state["cuda_version"] = torch.version.cuda
        state["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                state["gpus"].append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                })
    except ImportError:
        state["torch_version"] = None
    return state


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(output_dir, args, dataset_counts=None, model_counts=None, extra=None):
    """Write complete arguments and machine-readable run provenance."""
    output_path = ensure_output_dir(output_dir)
    arguments = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    num_devices = int(arguments.get("num_devices", 1))
    per_device_batch = int(arguments.get("train_batch_size", 1))
    accumulation = int(arguments.get("gradient_accumulation_steps", 1))
    arguments["per_device_train_batch_size"] = per_device_batch
    arguments["global_train_batch_size"] = per_device_batch * accumulation * num_devices

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "git": git_state(),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "hardware": hardware_state(),
        "random_seed": arguments.get("seed"),
        "dataset_split_counts": dataset_counts or {},
        "model_parameter_counts": model_counts or {},
        "batching": {
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": accumulation,
            "num_devices": num_devices,
            "global_train_batch_size": arguments["global_train_batch_size"],
            "eval_batch_size_per_device": arguments.get("eval_batch_size"),
        },
    }
    if extra:
        metadata.update(extra)

    with open(output_path / "args.json", "w", encoding="utf-8") as handle:
        json.dump(arguments, handle, indent=2, sort_keys=True)
    with open(output_path / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with open(output_path / "run_invocations.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "created_at_utc": metadata["created_at_utc"],
            "arguments": arguments,
            "git": metadata["git"],
        }, sort_keys=True) + "\n")
    with open(output_path / "package_versions.txt", "w", encoding="utf-8") as handle:
        for name, version in sorted(metadata["runtime"]["packages"].items()):
            handle.write(f"{name}=={version or 'NOT_INSTALLED'}\n")
    return metadata


def disk_free_bytes(path):
    return shutil.disk_usage(ensure_output_dir(path)).free
