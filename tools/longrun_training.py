"""Single-GPU training engine for the long-run protocol; reuse project models.

Imported by afrikaans_longrun.py and local_longrun_evaluation.py; not run directly. It
encodes the NCHLT text once into frozen-autoencoder latents (the cache), trains the
DiffusionTransformer denoiser on those cached latents with an EMA copy, validates and
samples from the EMA weights, and checkpoints so that an interrupted job resumes on the
same trajectory as an uninterrupted one.
"""
import contextlib
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from tools.longrun_common import digest, identity, lr_at_step, read_json, read_rows, steps_per_epoch, write_json

# Set by the signal handlers; long loops poll it and stop at a safe point.
STOP = False


def install_signals():
    """Make SIGTERM, SIGINT and (where defined) SIGUSR1 request a graceful stop via ``STOP``."""
    def stop(*_):
        """Signal handler: set the module-level stop flag."""
        global STOP
        STOP = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    # SIGUSR1 does not exist on Windows.
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop)


def seed_all(seed):
    """Seed Python, NumPy and Torch (CPU and all GPUs) with one value."""
    random.seed(seed)
    # NumPy's legacy seeding only accepts values below 2**32.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    """Capture every RNG state needed for an exact resume."""
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    """Restore states captured by ``rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints may be loaded with map_location="cuda", which moves these byte tensors to
    # the GPU; the Torch setters need them on the CPU.
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextlib.contextmanager
def isolated_rng(seed):
    """Run a block under ``seed`` and then restore the previous RNG states.

    Validation and sampling use this so they draw fixed random numbers without advancing
    the training RNG stream.
    """
    state = rng_state()
    seed_all(seed)
    try:
        yield
    finally:
        restore_rng(state)


def atomic_save(path, state):
    """``torch.save`` via a fsynced temporary file and rename, so a kill mid-write keeps the old file."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint or cache shard written by this project onto ``device``."""
    # Only load locally generated/trusted repository checkpoints.
    # weights_only=False because checkpoints also pickle RNG state (including NumPy arrays).
    return torch.load(str(path), map_location=device, weights_only=False)


def hardware():
    """Check that CUDA works and return a GPU/Torch identity dict.

    The dict is stored with the smoke test, benchmark and plan; later actions refuse to run
    when it differs, because the timing plan was measured on that setup.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A usable CUDA allocation is required; do not train on the login node")
    torch.cuda.init()
    # Small matmul as a sanity check that the allocated GPU actually computes.
    x = torch.randn(64, 64, device="cuda")
    if not torch.isfinite(x @ x.T).all():
        raise RuntimeError("CUDA arithmetic failed")
    return dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                torch=torch.__version__, cuda=torch.version.cuda)


def configure_torch(config):
    """Force deterministic kernels, disable TF32 and check the requested precision is usable."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Deterministic cuBLAS also needs CUBLAS_WORKSPACE_CONFIG, which the callers set.
    torch.use_deterministic_algorithms(True)
    if config["precision"] not in ("float32", "bf16"):
        raise ValueError("Only float32 or bf16 supported; FP16 needs a different scaler protocol")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; use float32 and repeat smoke and benchmark")


def precision(config, device="cuda"):
    """Return a BF16 autocast context, or a disabled one when precision is float32."""
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=config["precision"] == "bf16")


def make_model(config, name, device="cuda"):
    """Build the diffusion model for capacity arm ``name`` ("small" or "published").

    Width, depth and dense connections come from ``config["models"][name]``; the other
    settings are fixed here and shared by both arms.
    """
    from diffusion.text_denoising_diffusion import GaussianDiffusion
    from model.diffusion_transformer import DiffusionTransformer
    spec = config["models"][name]
    # Attention heads have a fixed size of 64 (width 256 gives 4 heads, 768 gives 12).
    net = DiffusionTransformer(tx_dim=spec["width"], tx_depth=spec["depth"], heads=spec["width"] // 64,
        latent_dim=config["latent_dim"], max_seq_len=config["max_length"],
        self_condition=config["self_condition"], scale_shift=True, dropout=.1,
        num_dense_connections=spec["dense_connections"])
    model = GaussianDiffusion(net, max_seq_len=config["latent_length"], objective="pred_v", loss_type="l2",
        train_schedule="cosine", sampler=config["sampler"], sampling_timesteps=config["sampling_steps"],
        train_prob_self_cond=.5)
    # The model is trained on cached AE latents, so these flags are set by hand here.
    # l2_normalize rescales the training-time self-conditioning estimate to norm
    # sqrt(latent_dim); the equivalent at sampling time is controlled separately by
    # normalize_sampling_self_condition.
    model.using_latent_model = True
    model.l2_normalize = True
    model.normalize_sampling_self_condition = config["normalize_sampling_self_condition"]
    return model.to(device)


def load_ae(ae_dir, model_cache, config):
    """Load the frozen mT5/Perceiver autoencoder and its tokeniser onto the GPU.

    Raises if the saved AE arguments do not match the latent shape, maximum length and L2
    normalisation the protocol expects. The backbone is read from the local ``model_cache``.
    """
    from latent_models.latent_utils import get_latent_model
    args = SimpleNamespace(**read_json(Path(ae_dir) / "args.json"))
    if (args.dim_ae, args.num_encoder_latents, args.max_seq_len, args.l2_normalize_latents) != (
            config["latent_dim"], config["latent_length"], config["max_length"], True):
        raise ValueError("Frozen AE architecture differs from protocol")
    # Point the AE at the local mt5-base copy instead of the name saved in args.json.
    args.enc_dec_model = str(model_cache)
    ae, tokenizer, _ = get_latent_model(args)
    saved = load_checkpoint(Path(ae_dir) / "model.pt")
    ae.load_state_dict(saved["model"], strict=True)
    del saved
    ae.cuda().eval().requires_grad_(False)
    return ae, tokenizer


@torch.no_grad()
def prepare_cache(config, data, ae_dir, model_cache, folder, contract, seconds):
    """Encode the train and valid splits into AE latents, saved in 1024-record shards.

    Resumable: existing shards are reloaded and checked instead of re-encoded. Writes
    ``manifest.json`` with shard hashes, truncation counts and token counts. Returns False if
    stopped or out of time (``seconds``), True once both splits are complete. The test split
    is not encoded here.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists() and read_json(manifest)["contract"] != contract:
        raise ValueError("Cache inputs changed; use a new workspace")
    metadata = read_json(manifest) if manifest.exists() else dict(contract=contract, splits={})
    write_json(manifest, metadata)
    ae, tokenizer = load_ae(ae_dir, model_cache, config)
    start = time.monotonic()
    for split in ("train", "valid"):
        rows = read_rows(Path(data) / f"{split}.jsonl")
        shards, truncated, tokens = [], 0, 0
        for offset in range(0, len(rows), 1024):
            path = folder / f"{split}-{offset:07d}.pt"
            if path.exists():
                item = load_checkpoint(path)
            else:
                values, capped, token_count = [], 0, 0
                for i in range(offset, min(offset + 1024, len(rows)), config["cache_batch"]):
                    if STOP or time.monotonic() - start >= seconds:
                        return False
                    texts = [row["text"] for row in rows[i:min(i+config["cache_batch"], offset+1024)]]
                    # Untruncated lengths are used only to count records longer than max_length.
                    lengths = tokenizer(texts, truncation=False)["input_ids"]
                    capped += sum(len(v) > config["max_length"] for v in lengths)
                    batch = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                      max_length=config["max_length"]).to("cuda")
                    token_count += int(batch.attention_mask.sum())
                    # Encode FP32 once: changing training precision does not change targets.
                    latent = ae.get_diffusion_latent(ae.get_encoder()(**batch), batch.attention_mask).float()
                    if not torch.isfinite(latent).all():
                        raise FloatingPointError("Non-finite encoded latent")
                    values.append(latent.cpu())
                item = dict(latents=torch.cat(values), truncated=capped, tokens=token_count)
                atomic_save(path, item)
            if list(item["latents"].shape) != [min(1024, len(rows)-offset), config["latent_length"], config["latent_dim"]]:
                raise ValueError(f"Invalid cache shard {path}")
            if not torch.isfinite(item["latents"]).all():
                raise ValueError(f"Non-finite cache {path}")
            truncated += item["truncated"]
            tokens += item["tokens"]
            shards.append(dict(file=path.name, sha256=digest(path)))
            print(f"cache {split}: {min(offset+1024,len(rows))}/{len(rows)}", flush=True)
        metadata["splits"][split] = dict(shards=shards, records=len(rows), truncated=truncated,
                                         truncation_fraction=truncated / len(rows), tokens=tokens)
        write_json(manifest, metadata)
    metadata["complete"] = True
    metadata["encoding_seconds_this_job"] = time.monotonic() - start
    write_json(manifest, metadata)
    return True


def load_cache(folder, contract):
    """Load the complete latent cache as ``{"train": tensor, "valid": tensor}``.

    Raises if the cache is incomplete, was built from different inputs, or any shard's
    SHA-256 differs from the manifest.
    """
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if not manifest.get("complete") or manifest["contract"] != contract:
        raise ValueError("Incomplete/stale cache; finish prepare first")
    result = {}
    for split in ("train", "valid"):
        values = []
        for shard in manifest["splits"][split]["shards"]:
            path = folder / shard["file"]
            if digest(path) != shard["sha256"]:
                raise ValueError(f"Cache checksum mismatch: {path}")
            values.append(load_checkpoint(path)["latents"])
        result[split] = torch.cat(values)
    return result


def epoch_batch(count, config, seed, step):
    """Return the training indices for optimiser update ``step``.

    Each epoch uses its own permutation seeded from the run seed and epoch number, so the
    batch for any step depends only on (seed, step) and resuming needs no data-loader state.
    The last batch of an epoch may be smaller than micro_batch * accumulation.
    """
    spe = steps_per_epoch(count, config)
    epoch, position = divmod(step, spe)
    # Epoch seed = run seed + epoch * 1000003; with run seeds a few apart, no two
    # (run, epoch) combinations share a permutation seed.
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch * 1000003))
    batch = config["micro_batch"] * config["accumulation"]
    return order[position*batch:(position+1)*batch]


@torch.no_grad()
def validation(model, cache, config, full=False, device="cuda"):
    """Return the mean diffusion loss on a fixed validation subset and a training probe.

    Subsets are fixed by ``validation_seed`` and each is evaluated ``validation_repeats``
    times with fixed seeds, because the loss samples random timesteps and noise.
    ``full=True`` uses every validation latent instead of ``validation_examples``. Also
    reports the valid-minus-train gap. Callers pass the EMA weights.
    """
    model.eval()
    metrics = {}
    with isolated_rng(config["validation_seed"]):
        for split, cap in (("train", config["train_probe_examples"]),
                           ("valid", len(cache["valid"]) if full else config["validation_examples"])):
            values = cache[split]
            order = torch.randperm(len(values), generator=torch.Generator().manual_seed(config["validation_seed"]))[:cap]
            losses = []
            for repeat in range(config["validation_repeats"]):
                seed_all(config["validation_seed"] + repeat)
                total = 0.
                for ids in order.split(config["micro_batch"]):
                    if STOP:
                        raise InterruptedError("Validation interrupted")
                    z = values[ids].to(device)
                    with precision(config, device):
                        # All-true mask: every cached latent has the full latent_length.
                        loss = model(z, torch.ones(z.shape[:2], device=device, dtype=torch.bool), class_id=None)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite validation loss")
                    total += float(loss) * len(ids)
                losses.append(total / len(order))
            metrics[split + "_loss"] = sum(losses) / len(losses)
            metrics[split + "_repeat_losses"] = losses
        metrics["generalization_gap"] = metrics["valid_loss"] - metrics["train_loss"]
    return metrics


@torch.no_grad()
def generate(model, ae, tokenizer, config, count, seed_start):
    """Sample ``count`` latents and decode them to text with the frozen AE decoder.

    Sample i is drawn alone under seed ``seed_start + i``, so any sample can be regenerated
    without the others. Each row records the text, decoder length, whether EOS was reached,
    whether the length cap was hit, and whether an mT5 ``<extra_id_*>`` sentinel appeared.
    """
    from transformers.modeling_outputs import BaseModelOutput
    model.eval()
    rows = []
    with isolated_rng(seed_start):
        for seed in range(seed_start, seed_start + count):
            if STOP:
                raise InterruptedError("Generation interrupted; repeat this evaluation on resume")
            seed_all(seed)
            latent, _ = model.sample(1, torch.tensor([config["latent_length"]], device="cuda"))
            if not torch.isfinite(latent).all():
                raise FloatingPointError("Non-finite generated latent")
            # Deterministic beam-search decoding; the same settings are used for every evaluation.
            ids = ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(latent)),
                max_length=config["max_length"], min_length=5, num_beams=4,
                do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.2)[0]
            # Decoder-start token may equal EOS in other backbones; skip it.
            eos = (ids[1:] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            # +2: one for the ids[1:] offset and one to keep the EOS token itself.
            n = int(eos[0]) + 2 if len(eos) else len(ids)
            raw = tokenizer.decode(ids[:n], skip_special_tokens=False)
            rows.append(dict(seed=seed, text=tokenizer.decode(ids[:n], skip_special_tokens=True).strip(),
                             decoder_tokens=n, eos=bool(len(eos)), length_cap=n >= config["max_length"],
                             sentinel="<extra_id_" in raw))
    return rows


def append_log(path, row):
    """Append one JSON row to a JSONL metrics log and flush it."""
    with Path(path).open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def train_run(config, spec, cache, folder, contract, ae=None, tokenizer=None,
              seconds=math.inf, max_new_steps=None, benchmark=False, device="cuda"):
    """Resume at an optimizer boundary with full RNG, EMA, optimizer and epoch cursor."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run_contract = dict(inputs=contract, config=config, spec=spec)
    if (folder / "config.json").exists():
        if read_json(folder / "config.json") != run_contract:
            raise ValueError("Run configuration/source/horizon changed; refusing incompatible resume")
    else:
        write_json(folder / "config.json", run_contract)
    # Seeded before construction so the initial weights depend only on the run seed.
    seed_all(spec["seed"])
    model = make_model(config, spec["model"], device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(.9, .999),
                            weight_decay=config["weight_decay"])
    spe = steps_per_epoch(len(cache["train"]), config)
    total = spec["epochs"] * spe
    state = dict(step=0, best_loss=None, best_step=0, stale_validations=0,
                 patience_loss=None, last_validated_step=-1, cumulative_seconds=0., status="running")
    latest = folder / "latest.pt"
    if latest.exists():
        saved = load_checkpoint(latest, device)
        if saved["contract"] != run_contract or saved["scheduler"]["total_steps"] != total:
            raise ValueError("Checkpoint contract/scheduler mismatch")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        state = saved["state"]
        restore_rng(saved["rng"])
        del saved
    if state["status"] in ("completed", "early_stopped"):
        return state
    initial, began, prior_seconds = state["step"], time.monotonic(), state["cumulative_seconds"]
    last_save = began
    timings = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save():
        """Write latest.pt (weights, EMA, optimiser, RNG, scheduler, state) and status.json."""
        nonlocal last_save
        state["cumulative_seconds"] = prior_seconds + time.monotonic() - began
        atomic_save(latest, dict(schema=1, contract=run_contract, model=model.state_dict(),
            ema=ema.state_dict(), optimizer=opt.state_dict(), rng=rng_state(), state=state,
            scheduler=dict(total_steps=total, next_step=state["step"], kind="warmup_cosine_floor")))
        write_json(folder / "status.json", state)
        last_save = time.monotonic()

    def evaluate():
        """Validate the EMA weights, update best.pt and the early-stopping counter.

        Returns the seconds the validation took.
        """
        started = time.monotonic()
        metrics = validation(ema, cache, config, device=device)
        value = metrics["valid_loss"]
        # Checkpoint selection: best.pt holds the EMA weights with the lowest validation
        # loss so far (any strict improvement counts).
        if state["best_loss"] is None or value < state["best_loss"]:
            state["best_loss"], state["best_step"] = value, state["step"]
            atomic_save(folder / "best.pt", dict(ema=ema.state_dict(), step=state["step"],
                        metrics=metrics, contract=run_contract))
        # Early-stopping patience resets only on a relative improvement larger than
        # early_stop_relative_delta, so tiny gains still count as stale validations.
        if state["patience_loss"] is None or value < state["patience_loss"] * (1-config["early_stop_relative_delta"]):
            state["patience_loss"], state["stale_validations"] = value, 0
        else:
            state["stale_validations"] += 1
        state["last_validated_step"] = state["step"]
        row = dict(kind="validation", step=state["step"], epoch=state["step"] / spe,
                   seconds=time.monotonic()-started, **metrics)
        append_log(folder / "metrics.jsonl", row)
        print(json.dumps(dict(run=spec["name"], **row)), flush=True)
        return row["seconds"]

    state["status"] = "running"
    # An initial checkpoint guarantees recovery even before the first scheduled save.
    if not latest.exists():
        save()
    try:
        # A time-limit signal may interrupt validation after its pre-validation save.
        # Replay that pending check before consuming any further training data.
        if (state["step"] > 0 and state["step"] % (spe*config["validate_every_epochs"]) == 0
                and state["last_validated_step"] != state["step"] and not STOP):
            evaluate()
            if state["step"] >= spe*config["early_stop_min_epochs"] and state["stale_validations"] >= config["early_stop_patience"]:
                state["status"] = "early_stopped"
            save()
        # Early stopping applies only after early_stop_min_epochs and once
        # early_stop_patience consecutive validations have been stale.
        while state["step"] < total and state["status"] != "early_stopped":
            if STOP or time.monotonic()-began >= seconds:
                break
            if max_new_steps is not None and state["step"]-initial >= max_new_steps:
                break
            tick = time.monotonic()
            ids = epoch_batch(len(cache["train"]), config, spec["seed"], state["step"])
            lr = lr_at_step(state["step"], total, config)
            for group in opt.param_groups:
                group["lr"] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            logged = 0.
            for micro in ids.split(config["micro_batch"]):
                z = cache["train"][micro].to(device)
                with precision(config, device):
                    loss = model(z, torch.ones(z.shape[:2], dtype=torch.bool, device=device), class_id=None)
                    # Weight by micro-batch share so the accumulated gradient is the mean over
                    # the whole effective batch, including a smaller final batch.
                    weighted = loss * len(micro) / len(ids)
                if not torch.isfinite(weighted):
                    raise FloatingPointError("Non-finite loss; last valid checkpoint retained")
                weighted.backward()
                logged += float(weighted.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
            opt.step()
            # Check after the optimizer as well: invalid state never replaces latest.
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("Non-finite parameter after optimizer; latest retained")
            with torch.no_grad():
                # EMA: ema += (1 - ema_decay) * (model - ema); buffers are copied, not averaged.
                for dest, source in zip(ema.parameters(), model.parameters()):
                    dest.lerp_(source, 1-config["ema_decay"])
                for dest, source in zip(ema.buffers(), model.buffers()):
                    dest.copy_(source)
            state["step"] += 1
            if device == "cuda":
                # Wait for queued GPU work so step_seconds measures the whole update.
                torch.cuda.synchronize()
            duration = time.monotonic()-tick
            timings.append(duration)
            if benchmark or state["step"] % config["log_every_steps"] == 0:
                row = dict(kind="train", step=state["step"], epoch=state["step"]/spe, loss=logged,
                    grad_norm=float(norm), learning_rate=lr, step_seconds=duration, examples_per_second=len(ids)/duration,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else 0)
                append_log(folder / "metrics.jsonl", row)
                print(json.dumps(dict(run=spec["name"], **row)), flush=True)
            # Save before and after validating; see the replay note at the top of this block.
            if state["step"] % (spe * config["validate_every_epochs"]) == 0 and not STOP:
                save()
                evaluate()
                if state["step"] >= spe*config["early_stop_min_epochs"] and state["stale_validations"] >= config["early_stop_patience"]:
                    state["status"] = "early_stopped"
                save()
            if (ae is not None and state["step"] % (spe*config["sample_every_epochs"]) == 0
                    and not STOP and time.monotonic()-began < seconds):
                # The same sample seeds are reused at every sampling point.
                rows = generate(ema, ae, tokenizer, config, config["validation_samples"], config["sample_seed"])
                write_json(folder / f"validation-samples-{state['step']:08d}.json", rows)
            if state["step"] % config["checkpoint_every_steps"] == 0 or time.monotonic()-last_save >= config["checkpoint_every_seconds"]:
                save()
            if state["status"] == "early_stopped":
                break
        if state["step"] == total:
            state["status"] = "completed"
        elif state["status"] != "early_stopped":
            state["status"] = "paused"
        # Smoke/benchmark must validate even when ending before an epoch boundary.
        if (not STOP and state["last_validated_step"] != state["step"]
                and (state["status"] == "completed" or max_new_steps is not None)):
            state["last_validation_seconds"] = evaluate()
        # The final save is timed; the benchmark action uses it as the checkpoint cost.
        started = time.monotonic()
        save()
        state["last_checkpoint_seconds"] = time.monotonic()-started
    except InterruptedError:
        state["status"] = "paused"
        save()
    except Exception as exc:
        write_json(folder / "failure.json", dict(step=state["step"], error=repr(exc),
                   recovery="latest.pt is the last successful save; no invalid state saved"))
        raise
    finally:
        del model, opt, ema
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    # Per-update timings from this invocation only; added after the last save, so not persisted.
    state["new_step_seconds"] = timings
    return state
