"""Dependency-free experiment contracts, integrity audit, and timing planner.

Imported by afrikaans_longrun.py and longrun_training.py; not run directly. Uses only the
standard library, so the audit and plan actions work without PyTorch installed. Provides
atomic JSON I/O, SHA-256 fingerprints, config validation, the train/valid/test leakage
audit, the learning-rate schedule and the planner that turns benchmark timings into
a seed-pair and epoch budget.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata


def read_json(path):
    """Load a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    """Write JSON atomically, creating parent folders.

    The value goes to a ``.tmp`` sibling, is fsynced and then renamed over the target, so an
    interrupted job leaves either the old file or the new one, never a partial file. NaN and
    infinity are rejected (``allow_nan=False``) so results files stay strict JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def digest(path):
    """Return the SHA-256 hex digest of a file, read in 8 MiB blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    """Return a SHA-256 fingerprint of a JSON-serialisable value.

    Keys are sorted so equal configs, plans or benchmarks give the same hash regardless of
    key order. Used to detect when a config changed after benchmarking or planning.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    """Raise ``ValueError`` if the long-run config breaks the protocol's fixed bounds."""
    positive = ("micro_batch", "accumulation", "min_epochs", "max_epochs", "warmup_steps",
                "validate_every_epochs", "validation_examples", "validation_repeats", "train_probe_examples",
                "sample_every_epochs", "validation_samples", "final_samples", "sampling_steps",
                "checkpoint_every_steps", "checkpoint_every_seconds", "log_every_steps", "cache_batch",
                "benchmark_warmup_steps", "benchmark_steps", "early_stop_patience")
    if any(not isinstance(config[k], int) or config[k] < 1 for k in positive):
        raise ValueError("Counts, batches and intervals must be positive integers")
    # Fixed protocol bounds: a 48-72 h productive budget and at least three seed pairs.
    if not 48 <= config["minimum_hours"] <= config["target_hours"] <= 72:
        raise ValueError("Requested productive budget must be between 48 and 72 hours")
    if not 3 <= config["min_seed_pairs"] <= config["max_seed_pairs"]:
        raise ValueError("At least three independent seed pairs required")
    if not config["min_epochs"] <= config["max_epochs"]:
        raise ValueError("Invalid epoch bounds")
    if not 0 < config["minimum_lr_ratio"] <= 1 or not 0 < config["ema_decay"] < 1:
        raise ValueError("Invalid LR floor or EMA decay")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if config["precision"] not in ("float32", "bf16"):
        raise ValueError("Choose float32 or bf16")
    if config["sampler"] not in ("ddpm", "ddim"):
        raise ValueError("Choose ddpm or ddim")
    if set(config["models"]) != {"small", "published"}:
        raise ValueError("This protocol requires the small and published capacity arms")


def normalize(text):
    """Normalise text for duplicate matching: NFC, casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def read_rows(path):
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def audit_data(folder):
    """Test is read only for split integrity, never for model selection."""
    sets, report, errors = {}, {"splits": {}, "overlap": {}}, []
    for split in ("train", "valid", "test"):
        path = Path(folder) / f"{split}.jsonl"
        texts, ids, docs, sources, shingles = set(), set(), set(), set(), set()
        count = 0
        for row in read_rows(path):
            count += 1
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("id", "document_id", "source_document", "text")):
                raise ValueError(f"{split} record {count}: missing/empty required field")
            text = normalize(row["text"])
            if row.get("split") != split or row.get("language") != "afr":
                errors.append(f"{split}: incorrect split/language at record {count}")
            if row["id"] in ids or text in texts:
                errors.append(f"{split}: duplicate id/text at record {count}")
            ids.add(row["id"])
            texts.add(text)
            docs.add(row["document_id"])
            sources.add(row["source_document"])
            words = re.findall(r"\w+", text)
            # Shared long phrases are advisory: boilerplate is not proof of leakage.
            # 12-word windows start every 4th word, so this samples rather than enumerates
            # all windows; 8-byte BLAKE2b hashes keep the per-split sets small.
            for i in range(0, len(words) - 11, 4):
                shingles.add(hashlib.blake2b(" ".join(words[i:i+12]).encode(), digest_size=8).digest())
        if not count:
            errors.append(f"{split}: empty split")
        sets[split] = dict(texts=texts, ids=ids, documents=docs, sources=sources, shingles=shingles)
        report["splits"][split] = dict(records=count, documents=len(docs), sha256=digest(path))
    # Only exact text, ID, document and source overlap fail the audit; shingle overlap is
    # reported but never added to errors.
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = {key: len(sets[a][key] & sets[b][key]) for key in sets[a]}
        report["overlap"][f"{a}-{b}"] = overlap
        if any(overlap[k] for k in ("texts", "ids", "documents", "sources")):
            errors.append(f"{a}-{b}: exact text, ID, document or source leakage")
    report.update(errors=errors, passed=not errors,
                  test_use="integrity only; test metrics remain sealed until final evaluation",
                  near_duplicate_limit="Sampled shared 12-word shingles are advisory; not exhaustive semantic deduplication")
    return report


def steps_per_epoch(count, config):
    """Return optimiser updates per epoch; a final partial effective batch counts as one."""
    return math.ceil(count / (config["micro_batch"] * config["accumulation"]))


def lr_at_step(step, total, config):
    """LR applied to zero-based update step; locked total on resume."""
    # Linear warm-up for warmup_steps, capped at 10% of the run's total updates.
    warmup = min(config["warmup_steps"], max(1, total // 10))
    if step < warmup:
        return config["learning_rate"] * (step + 1) / warmup
    # The "- 1" makes the last update (step total-1) land exactly on the floor.
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    floor = config["minimum_lr_ratio"]
    return config["learning_rate"] * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


def make_plan(config, benchmark, count):
    """Turn measured per-step timings into a fixed run plan for the multi-seed suite.

    ``benchmark`` is the benchmark.json written by the benchmark action and ``count`` is the
    number of training records. Returns the plan dict (saved as plan.json): the number of
    initial seed pairs, the shared epoch horizon, the initial and reserve run lists and time
    estimates. Raises if the best plan falls outside the configured hour budget.
    """
    validate_config(config)
    if benchmark["config_sha256"] != identity(config):
        raise ValueError("Benchmark config differs; rerun benchmark after changing settings")
    # Seconds per epoch per model: training steps, plus validation, sampling and checkpoint
    # costs amortised over their intervals. Checkpoint count per epoch is the larger of the
    # step-triggered and time-triggered counts.
    per_epoch = {}
    for name, timing in benchmark["models"].items():
        per_epoch[name] = (steps_per_epoch(count, config) * timing["step_seconds_median"]
                           + timing["validation_seconds"] / config["validate_every_epochs"]
                           + timing["generation_seconds"] / config["sample_every_epochs"]
                           + timing["checkpoint_seconds"] * max(
                               steps_per_epoch(count, config) / config["checkpoint_every_steps"],
                               steps_per_epoch(count, config) * timing["step_seconds_median"] / config["checkpoint_every_seconds"]))
    if set(per_epoch) != set(config["models"]):
        raise ValueError("Benchmark must cover every model")
    target = config["target_hours"] * 3600
    pair_epoch = sum(per_epoch.values())
    if pair_epoch <= 0 or not math.isfinite(pair_epoch):
        raise ValueError("Invalid benchmark timing")
    # For each allowed number of seed pairs, pick the clamped epoch count nearest the target,
    # then keep the pair count whose total is closest to target_hours (ties go to fewer pairs).
    candidates = []
    for pairs in range(config["min_seed_pairs"], config["max_seed_pairs"] + 1):
        epochs = max(config["min_epochs"], min(config["max_epochs"], round(target / pairs / pair_epoch)))
        seconds = pairs * epochs * pair_epoch
        candidates.append((abs(seconds - target), pairs, epochs, seconds))
    _, pairs, epochs, seconds = min(candidates)
    hours = seconds / 3600
    # Allows up to 10% over the target, none under the minimum.
    if not config["minimum_hours"] <= hours <= config["target_hours"] * 1.1:
        raise ValueError(f"Measured feasible plan is {hours:.1f}h; outside 48–72h intent. "
                         "Review scientific scope/epoch bounds, then rerun the planner; never pad runtime.")
    # Every seed up to max_seed_pairs is listed, both models per seed in adjacent order; the
    # first `pairs` seeds are the initial suite and the rest are held in reserve.
    runs = [dict(name=f"{name}-seed{seed}", model=name, seed=seed, epochs=epochs,
                 estimated_hours=epochs * per_epoch[name] / 3600)
            for seed in range(config["seed_start"], config["seed_start"] + config["max_seed_pairs"])
            for name in config["models"]]
    # Rough final-evaluation estimate per pair: generation time scaled from validation_samples
    # to final_samples for two splits, plus four validation-pass times per model.
    evaluation_hours = pairs * sum(t["generation_seconds"] * config["final_samples"] / config["validation_samples"] * 2
                                  + t["validation_seconds"] * 4 for t in benchmark["models"].values()) / 3600
    return dict(schema=1, config_sha256=identity(config), benchmark_sha256=identity(benchmark),
                hardware=benchmark["hardware"], train_records=count, steps_per_epoch=steps_per_epoch(count, config),
                initial_seed_pairs=pairs, epochs=epochs, expected_training_and_validation_hours=hours,
                estimated_additional_final_evaluation_hours=max(config["final_evaluation_reserve_hours"], evaluation_hours),
                initial_runs=runs[:pairs * len(per_epoch)], reserve_runs=runs[pairs * len(per_epoch):],
                policy="Run initial suite, then whole reserve seed pairs until minimum productive hours. "
                       "Early stopping may shorten runs; exhausted reserves below 48h is reported as under_budget.")
