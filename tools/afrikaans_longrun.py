"""Prepare, time, run, resume, and evaluate a sequential Afrikaans experiment.

Run as: python tools/afrikaans_longrun.py ACTION --workspace DIR [--model-cache DIR]
[--seconds N]; --model-cache is required except for audit, plan and summarize. Actions
are meant to run in this order: audit (leakage check of the NCHLT splits), prepare (encode train/valid text into frozen-AE latents), smoke (short
train/resume equivalence test), benchmark (time each model), plan (choose seed pairs and
epochs from the timings), run (train the planned suite), evaluate (select best.pt per
run on validation and seal the selection), final-test (score the sealed checkpoints on the
test split), score (mGPT perplexity and blinded review sheets) and summarize (tables and
curves). Inputs are the long-run config, the processed NCHLT JSONL splits, the frozen AE
checkpoint and a local mt5-base copy; every output is written inside the workspace. Exit
code 75 means the job stopped on a signal or its --seconds budget and the same command
should be rerun to resume.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.longrun_common import audit_data, digest, identity, make_plan, normalize, read_json, read_rows, validate_config, write_json

# Pinned Hugging Face commit of ai-forever/mGPT, so perplexity scores use fixed weights.
SCORER_REVISION = "40897bd7c8b47a76802c411108ca6220438b8b40"


def package_versions():
    """Return installed versions of the packages that affect training results."""
    return {p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "numpy", "einops", "accelerate", "datasets", "ema-pytorch")}


def code_hashes():
    """Return SHA-256 hashes of the project source files the experiment depends on.

    Part of the input contract, so editing any of these files blocks resuming in an existing
    workspace.
    """
    paths = []
    for folder in ("model", "diffusion", "latent_models", "dataset_utils", "evaluation", "utils"):
        paths.extend((ROOT / folder).glob("*.py"))
    paths += [ROOT / "CONSTANTS.py", ROOT / "tools/afrikaans_longrun.py",
              ROOT / "tools/longrun_common.py", ROOT / "tools/longrun_training.py",
              ROOT / "tools/compare_afrikaans_generators.py"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def input_contract(args, config):
    """Audit the data and fingerprint every input a run depends on.

    Writes integrity.json and returns a dict of hashes for the data splits, the frozen AE
    checkpoint and its args, the local model cache, the source code, plus the latent shape,
    maximum length and package versions. Raises if the audit fails, the model cache is
    empty or the AE checkpoint hash differs from ``expected_autoencoder_sha256``.
    """
    audit = audit_data(args.data)
    write_json(args.workspace / "integrity.json", audit)
    if not audit["passed"]:
        raise ValueError("Data audit failed; inspect integrity.json")
    files = [p for p in args.model_cache.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError("Local mt5-base model/tokenizer cache required")
    ae_hash = digest(args.ae / "model.pt")
    if ae_hash != config["expected_autoencoder_sha256"]:
        raise ValueError("Wrong frozen AE checkpoint; expected the audited job-56531 best weights")
    return dict(data={k: v["sha256"] for k, v in audit["splits"].items()},
                ae=ae_hash, ae_args=digest(args.ae / "args.json"),
                model_cache={p.relative_to(args.model_cache).as_posix(): digest(p) for p in sorted(files)},
                code=code_hashes(), latent_shape=[config["latent_length"], config["latent_dim"]],
                max_length=config["max_length"], packages=package_versions())


def provenance(args, hardware):
    """Return a per-invocation record (hardware, packages, git HEAD, command, Slurm job ID, time).

    Unlike the input contract this is only logged, never compared, so it may differ between jobs.
    """
    packages = package_versions()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return dict(hardware=hardware, packages=packages, python=sys.version, git_head=git.stdout.strip(),
                source_files=code_hashes(), command=sys.argv, job_id=os.environ.get("SLURM_JOB_ID"),
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def generation_metrics(rows, training_texts):
    """Summarise generated samples: emptiness, uniqueness, length, stopping and diversity.

    ``rows`` come from ``longrun_training.generate``; ``training_texts`` is the set of
    normalised training sentences used to count verbatim copies. Words are whitespace tokens.
    """
    tokens = [r["text"].split() for r in rows]
    result = dict(samples=len(rows), empty=sum(not r["text"] for r in rows),
                  unique=len({r["text"] for r in rows}),
                  mean_words=statistics.mean(map(len, tokens)),
                  mean_decoder_tokens=statistics.mean(r["decoder_tokens"] for r in rows),
                  eos_fraction=statistics.mean(r["eos"] for r in rows),
                  length_cap_fraction=statistics.mean(r["length_cap"] for r in rows),
                  sentinel_fraction=statistics.mean(r["sentinel"] for r in rows),
                  exact_training_matches=sum(normalize(r["text"]) in training_texts for r in rows))
    # distinct-n is pooled over all samples; the repeated-trigram fraction is per sample
    # (0 for samples under three words) and then averaged.
    for n in (1, 2, 3):
        grams = [tuple(w[i:i+n]) for w in tokens for i in range(len(w)-n+1)]
        result[f"distinct_{n}"] = len(set(grams))/len(grams) if grams else None
    fractions = []
    for words in tokens:
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        fractions.append(1-len(set(grams))/len(grams) if grams else 0.)
    result["mean_repeated_trigram_fraction"] = statistics.mean(fractions)
    return result


def evaluate_run(args, config, spec, contract, cache, ae, tok, split):
    """Evaluate one run's selected best.pt EMA weights on the full validation or test cache.

    Computes the denoising loss over every latent in ``cache["valid"]`` (which holds the test
    latents when ``split == "test"``), draws ``final_samples`` generations and writes
    selected-validation.json or final-test.json. An existing result file is returned as is.
    """
    from tools import longrun_training as engine
    import torch
    folder = args.workspace / "runs" / spec["name"]
    destination = folder / ("final-test.json" if split == "test" else "selected-validation.json")
    if destination.exists():
        return read_json(destination)
    saved = engine.load_checkpoint(folder / "best.pt", "cuda")
    expected = dict(inputs=contract, config=config, spec=spec)
    if saved["contract"] != expected:
        raise ValueError("Best checkpoint contract changed")
    model = engine.make_model(config, spec["model"])
    model.load_state_dict(saved["ema"], strict=True)
    model.eval().requires_grad_(False)
    metrics = engine.validation(model, cache, config, full=True)
    if split == "test":
        metrics = {key.replace("valid", "test"): value for key, value in metrics.items()}
    # Offsets keep these sample seeds apart from the in-training samples (sample_seed onwards)
    # and from each other.
    seeds = config["sample_seed"] + (100000 if split == "test" else 10000)
    sample_path = folder / f"{split}-selected-samples.json"
    rows = engine.generate(model, ae, tok, config, config["final_samples"], seeds)
    write_json(sample_path, rows)
    train_texts = {normalize(r["text"]) for r in read_rows(args.data / "train.jsonl")}
    result = dict(run=spec["name"], model=spec["model"], training_seed=spec["seed"], selected_step=saved["step"],
                  split=split, denoising=metrics, generation=generation_metrics(rows, train_texts),
                  sample_file=sample_path.name, selection="minimum fixed validation EMA velocity MSE",
                  test_reuse="Historical AE test evaluation exists; this is held out from the new diffusion tuning, not pristine project-wide data")
    write_json(destination, result)
    del model, saved
    torch.cuda.empty_cache()
    return result


def score_generations(args, config, specs, engine, remaining):
    """Pinned external LM scores, real-text reference, and blinded review sheets."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.compare_afrikaans_generators import score_text
    import torch
    tok = AutoTokenizer.from_pretrained("ai-forever/mGPT", revision=SCORER_REVISION, local_files_only=True)
    scorer = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT", revision=SCORER_REVISION,
        local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    groups = []
    for spec in specs:
        for split in ("validation", "test"):
            path = args.workspace / "runs" / spec["name"] / f"{split}-selected-samples.json"
            if path.exists():
                groups.append((spec["name"], split, read_json(path), path.with_name(f"{split}-mgpt.json")))
    if not groups:
        raise ValueError("No selected generations; run evaluate first")
    # Real-text reference: the first final_samples held-out sentences after a shuffle with
    # validation_seed, scored the same way as the generations.
    splits = sorted({split for _, split, _, _ in groups})
    for split in splits:
        source = "valid" if split == "validation" else "test"
        rows = read_rows(args.data / f"{source}.jsonl")
        random.Random(config["validation_seed"]).shuffle(rows)
        groups.append(("real-reference", split, rows[:config["final_samples"]], args.workspace / f"{split}-reference-mgpt.json"))
    blind = []
    for name, split, rows, destination in groups:
        if destination.exists():
            continue
        scored, nll, tokens = [], 0., 0
        for row in rows:
            if engine.STOP or remaining() <= 0:
                # Partial groups are not written; a rerun rescores this group from the start.
                return 75
            value = score_text(scorer, tok, row["text"], "cuda")
            scored.append(dict(**row, mgpt=value))
            if value is not None:
                nll += value["nll"]
                tokens += value["tokens"]
        empty = sum(v["mgpt"] is None for v in scored)
        # Empty text is an explicit failed generation, never silently omitted.
        result = dict(scorer="ai-forever/mGPT", revision=SCORER_REVISION, split=split,
                      count=len(rows), empty=empty, status="failed_empty" if empty else "scored",
                      token_weighted_perplexity=math.exp(nll/tokens) if tokens and not empty else None,
                      rows=scored, interpretation="Exploratory external LM likelihood; not a fluency or factuality judgment")
        write_json(destination, result)
    # Always rebuild reviews, including already-scored groups after resume.
    for name, split, rows, _ in groups:
        if name != "real-reference":
            # First 30 samples of each run and split; the review sheet hides run identity and
            # review-key.csv maps row IDs back to runs.
            blind.extend(dict(run=name, split=split, seed=r["seed"], text=r["text"]) for r in rows[:30])
    # Fixed shuffle seed so the review order is the same every time the sheet is rebuilt.
    random.Random(78321).shuffle(blind)
    with (args.workspace / "blind-review.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text", "grammar_1_to_5", "coherence_1_to_5", "notes"])
        for i, r in enumerate(blind, 1):
            writer.writerow([i, r["text"], "", "", ""])
    with (args.workspace / "review-key.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "run", "split", "sampling_seed"])
        for i, r in enumerate(blind, 1):
            writer.writerow([i, r["run"], r["split"], r["seed"]])
    return 0


def summarize(workspace):
    """Collect every run in the workspace into results tables and training curves.

    Writes results.json (per-run records and per-model seed means/sample SDs), results.csv,
    RESULTS.md, training-curves.svg/.png and plot-status.json. Seed aggregates use only runs
    that completed or stopped early.
    """
    workspace = Path(workspace)
    results, histories = [], {}
    for folder in sorted((workspace / "runs").glob("*")):
        if not (folder / "status.json").exists():
            continue
        state = read_json(folder / "status.json")
        contract = read_json(folder / "config.json")
        record = dict(run=folder.name, model=contract["spec"]["model"], seed=contract["spec"]["seed"],
                      status=state["status"], step=state["step"], best_step=state["best_step"],
                      best_validation_loss=state["best_loss"], hours=state["cumulative_seconds"]/3600)
        for name in ("selected-validation", "final-test"):
            path = folder / (name + ".json")
            if path.exists():
                record[name] = read_json(path)
        for split in ("validation", "test"):
            path = folder / f"{split}-mgpt.json"
            if path.exists():
                record[split + "_mgpt_perplexity"] = read_json(path)["token_weighted_perplexity"]
        results.append(record)
        if (folder / "metrics.jsonl").exists():
            # Discard log rows beyond committed state after an abrupt failure;
            # keep the latest repeated step after deterministic replay.
            history = {}
            for row in read_rows(folder / "metrics.jsonl"):
                if row["step"] <= state["step"]:
                    history[(row["kind"], row["step"])] = row
            histories[folder.name] = list(history.values())
    aggregates = {}
    for name in sorted({r["model"] for r in results}):
        values = [r["best_validation_loss"] for r in results if r["model"] == name
                  and r["status"] in ("completed", "early_stopped") and r["best_validation_loss"] is not None]
        if values:
            aggregates[name] = dict(completed_seeds=len(values), mean_best_validation_loss=statistics.mean(values),
                                    sample_std=statistics.stdev(values) if len(values)>1 else None)
            for split, key in (("selected-validation", "valid_loss"), ("final-test", "test_loss")):
                heldout = [r[split]["denoising"][key] for r in results if r["model"] == name and split in r]
                if heldout:
                    aggregates[name][split] = dict(seeds=len(heldout), mean=statistics.mean(heldout),
                                                   sample_std=statistics.stdev(heldout) if len(heldout)>1 else None)
    report = dict(runs=results, aggregates=aggregates, training_and_validation_hours=sum(r["hours"] for r in results),
                  language_quality="Denoising MSE and diversity do not establish fluent Afrikaans; blinded fluent-human review required")
    write_json(workspace / "results.json", report)
    with (workspace / "results.csv").open("w", newline="", encoding="utf8") as stream:
        fields = ["run", "model", "seed", "status", "step", "best_step", "best_validation_loss", "hours"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Afrikaans long-run results", "", f"Completed/paused run time: {report['training_and_validation_hours']:.2f} hours.",
             "", "| Run | State | Updates | Best validation MSE | Hours |", "|---|---|---:|---:|---:|"]
    for r in results:
        loss = "pending" if r["best_validation_loss"] is None else f"{r['best_validation_loss']:.5f}"
        lines.append(f"| {r['run']} | {r['status']} | {r['step']} | {loss} | {r['hours']:.2f} |")
    lines += ["", report["language_quality"], "", "See results.json for seed means, standard deviations and final evaluation metrics."]
    (workspace / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    # Standalone research diagnostics; tables remain usable if plotting is absent.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        write_json(workspace / "plot-status.json", dict(status="unavailable", reason="Install matplotlib to render curves"))
        print(json.dumps(report, indent=2))
        return
    panels = [("Validation and training probe loss", "validation", ["valid_loss", "train_loss"]),
              ("Learning rate", "train", ["learning_rate"]), ("Gradient norm", "train", ["grad_norm"]),
              ("Examples per second", "train", ["examples_per_second"]),
              ("Peak allocated GPU bytes", "train", ["peak_allocated_bytes"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 18), constrained_layout=True)
    for axis, (title, kind, keys) in zip(axes, panels):
        for idx, (name, rows) in enumerate(histories.items()):
            for key in keys:
                points = sorted((r["step"], r[key]) for r in rows if r["kind"] == kind and key in r)
                if points:
                    x, y = zip(*points)
                    axis.plot(x, y, label=f"{name}: {key}", color=plt.get_cmap("tab20")(idx % 20),
                              linestyle="--" if key == "train_loss" else "-", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=6, ncol=3)
    fig.savefig(workspace / "training-curves.svg")
    fig.savefig(workspace / "training-curves.png", dpi=150)
    plt.close(fig)
    write_json(workspace / "plot-status.json", dict(status="complete"))
    print(json.dumps(report, indent=2))


def main():
    """Parse the command line and run one protocol action.

    Returns the process exit code: 0 on success, 1 if the audit fails, 2 if the suite ran out
    of reserve runs below minimum_hours, and 75 if stopped early (rerun to resume).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["audit", "prepare", "smoke", "benchmark", "plan", "run", "evaluate", "final-test", "score", "summarize"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/afrikaans_longrun.json")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/nchlt_afrikaans/processed/stage_5_7")
    p.add_argument("--ae", type=Path, default=ROOT / "outputs/cluster-results/afrikaans-56531/best")
    p.add_argument("--model-cache", type=Path)
    p.add_argument("--seconds", type=float, default=math.inf, help="Useful work budget this job; scheduler must leave >=10 min margin")
    args = p.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    validate_config(config)
    if args.seconds <= 0:
        p.error("--seconds must be positive")
    if args.action == "audit":
        report = audit_data(args.data)
        write_json(args.workspace / "integrity.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if args.action == "summarize":
        summarize(args.workspace)
        return 0
    if args.action == "plan":
        benchmark = read_json(args.workspace / "benchmark.json")
        plan = make_plan(config, benchmark, read_json(args.workspace / "integrity.json")["splits"]["train"]["records"])
        if (args.workspace / "plan.json").exists() and read_json(args.workspace / "plan.json") != plan:
            raise ValueError("Existing plan differs; use a new workspace instead of changing an active horizon")
        write_json(args.workspace / "plan.json", plan)
        print(json.dumps(plan, indent=2))
        return 0
    if args.model_cache is None:
        p.error("--model-cache must be the existing local mt5-base directory")
    # Needed for deterministic cuBLAS under torch.use_deterministic_algorithms(True). Unless
    # already set, Hugging Face loads are kept offline and W&B logging is turned off.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    from tools import longrun_training as engine
    import torch
    engine.install_signals()
    started = time.monotonic()
    # Seconds left of this job's --seconds budget.
    remaining = lambda: max(0, args.seconds - (time.monotonic()-started))
    hw = engine.hardware()
    engine.configure_torch(config)
    contract = input_contract(args, config)
    path = args.workspace / "inputs.json"
    # Resuming is only allowed when data, AE, model cache, code and packages are unchanged.
    if path.exists() and read_json(path) != contract:
        raise ValueError("Source/data/AE changed; start a new workspace")
    write_json(path, contract)
    prov = provenance(args, hw)
    write_json(args.workspace / "invocations" / f"{time.time_ns()}.json", prov)
    if args.action == "prepare":
        return 0 if engine.prepare_cache(config, args.data, args.ae, args.model_cache,
                                         args.workspace / "cache", contract, remaining()) else 75
    cache = engine.load_cache(args.workspace / "cache", contract)
    if args.action == "smoke":
        ae, tok = engine.load_ae(args.ae, args.model_cache, config)
        # Real architecture, real training/validation data, short immutable horizon.
        # Train 3 updates, resume for 2 more, then compare against an uninterrupted 5-update
        # run: weights and EMA must match exactly. Seed 919 is not one of the planned
        # training seeds (42-53 with the current config).
        smoke = json.loads(json.dumps(config))
        smoke.update(validation_examples=32, validation_repeats=1, train_probe_examples=32,
                     checkpoint_every_steps=2, log_every_steps=1)
        reports = []
        for name in config["models"]:
            spec = dict(name=name+"-smoke", model=name, seed=919, epochs=config["min_epochs"])
            folder = args.workspace / "smoke" / name
            if folder.exists():
                raise FileExistsError("Smoke already exists; preserve it and use a new workspace for a new smoke")
            first = engine.train_run(smoke, spec, cache, folder, contract, max_new_steps=3, seconds=remaining())
            before = engine.load_checkpoint(folder / "latest.pt")
            if first["step"] != 3 or not (folder / "best.pt").exists():
                raise RuntimeError("Smoke failed before checkpoint")
            second = engine.train_run(smoke, spec, cache, folder, contract, max_new_steps=2, seconds=remaining())
            after = engine.load_checkpoint(folder / "latest.pt")
            if second["step"] != 5 or not any(not torch.equal(before["model"][k], after["model"][k]) for k in before["model"]):
                raise RuntimeError("Resume did not advance model updates")
            uninterrupted = args.workspace / "smoke" / (name + "-uninterrupted")
            engine.train_run(smoke, spec, cache, uninterrupted, contract, max_new_steps=5, seconds=remaining())
            reference = engine.load_checkpoint(uninterrupted / "latest.pt")
            if any(not torch.equal(after[group][key], reference[group][key])
                   for group in ("model", "ema") for key in after[group]):
                raise RuntimeError("GPU resume differs from uninterrupted weights/EMA")
            reports.append(dict(model=name, first_step=3, resumed_step=5, exact_resume_weights=True,
                                optimizer_entries=len(after["optimizer"]["state"])))
            del before, after, reference
        model = engine.make_model(config, "small")
        saved = engine.load_checkpoint(args.workspace / "smoke/small/best.pt", "cuda")
        model.load_state_dict(saved["ema"])
        samples = engine.generate(model, ae, tok, config, 2, config["sample_seed"])
        write_json(args.workspace / "smoke/samples.json", samples)
        write_json(args.workspace / "smoke.json", dict(passed=True, models=reports, hardware=hw, config_sha256=identity(config)))
        return 0
    if args.action == "benchmark":
        smoke = read_json(args.workspace / "smoke.json")
        if not smoke["passed"] or smoke["hardware"] != hw or smoke["config_sha256"] != identity(config):
            raise ValueError("Pass smoke on this hardware/config first")
        measurements = {}
        ae, tok = engine.load_ae(args.ae, args.model_cache, config)
        for name in config["models"]:
            folder = args.workspace / "benchmark" / name
            if folder.exists():
                raise FileExistsError("Benchmark exists; use new workspace or preserve and rename benchmark directory")
            spec = dict(name=name+"-benchmark", model=name, seed=919, epochs=config["min_epochs"])
            state = engine.train_run(config, spec, cache, folder, contract, ae, tok, seconds=remaining(), benchmark=True,
                max_new_steps=config["benchmark_warmup_steps"]+config["benchmark_steps"])
            # The first benchmark_warmup_steps updates are excluded from the timing.
            times = state["new_step_seconds"][config["benchmark_warmup_steps"]:]
            if len(times) != config["benchmark_steps"]:
                raise RuntimeError("Benchmark interrupted; no valid timing plan produced")
            model = engine.make_model(config, name)
            saved = engine.load_checkpoint(folder / "best.pt", "cuda")
            model.load_state_dict(saved["ema"])
            tick = time.monotonic()
            samples = engine.generate(model, ae, tok, config, config["validation_samples"], config["sample_seed"])
            generation_seconds = time.monotonic()-tick
            write_json(folder / "samples.json", samples)
            logs = read_rows(folder / "metrics.jsonl")
            measurements[name] = dict(step_seconds_median=statistics.median(times), step_seconds_mean=statistics.mean(times),
                step_seconds_p90=sorted(times)[int(.9*(len(times)-1))], measured_steps=len(times),
                validation_seconds=state["last_validation_seconds"], checkpoint_seconds=state["last_checkpoint_seconds"],
                generation_seconds=generation_seconds, peak_allocated_bytes=max(r.get("peak_allocated_bytes",0) for r in logs))
            del model, saved
            torch.cuda.empty_cache()
        write_json(args.workspace / "benchmark.json", dict(config_sha256=identity(config), hardware=hw, models=measurements))
        return 0
    # The remaining actions follow a plan made on the same config and hardware.
    plan = read_json(args.workspace / "plan.json")
    if plan["config_sha256"] != identity(config) or plan["hardware"] != hw:
        raise ValueError("Plan config/hardware differs; use the benchmarked GPU model and Torch stack")
    if args.action == "run":
        if (args.workspace / "selection.json").exists():
            raise ValueError("Selection sealed; use a new workspace for more training")
        ae, tok = engine.load_ae(args.ae, args.model_cache, config)
        completed, cumulative = [], 0.
        all_specs = plan["initial_runs"] + plan["reserve_runs"]
        for index, spec in enumerate(all_specs):
            # Reserve work starts only in whole pairs; complete both models for each seed.
            if index >= len(plan["initial_runs"]) and index % len(config["models"]) == 0 and cumulative >= config["minimum_hours"]*3600:
                break
            if engine.STOP or remaining() <= 0:
                return 75
            state = engine.train_run(config, spec, cache, args.workspace / "runs" / spec["name"], contract,
                                     ae, tok, seconds=remaining())
            if state["status"] not in ("completed", "early_stopped"):
                return 75
            # A run finished in an earlier job returns its saved state immediately, so its
            # hours still count after a resume.
            cumulative += state["cumulative_seconds"]
            completed.append(spec)
            write_json(args.workspace / "suite-status.json", dict(status="running", completed=completed, productive_hours=cumulative/3600))
        status = "completed" if cumulative >= config["minimum_hours"]*3600 else "under_budget"
        write_json(args.workspace / "suite-status.json", dict(status=status, completed=completed, productive_hours=cumulative/3600,
                   minimum_hours=config["minimum_hours"], target_hours=config["target_hours"]))
        summarize(args.workspace)
        return 0 if status == "completed" else 2
    suite = read_json(args.workspace / "suite-status.json")
    if suite["status"] not in ("completed", "under_budget"):
        raise ValueError("Finish the suite before selecting/final evaluation")
    specs = suite["completed"]
    if args.action == "score":
        result = score_generations(args, config, specs, engine, remaining)
        summarize(args.workspace)
        return result
    ae, tok = engine.load_ae(args.ae, args.model_cache, config)
    if args.action == "evaluate":
        for spec in specs:
            if engine.STOP or remaining() <= 0:
                return 75
            evaluate_run(args, config, spec, contract, cache, ae, tok, "validation")
        # Seal the selection: final-test refuses to run if any best.pt hash or the plan changes.
        selected = {s["name"]: digest(args.workspace / "runs" / s["name"] / "best.pt") for s in specs}
        write_json(args.workspace / "selection.json", dict(plan_sha256=identity(plan), checkpoints=selected,
                   rule="Each training seed/architecture uses its minimum fixed validation loss; all seeds reported, no seed cherry-picking"))
    elif args.action == "final-test":
        selection = read_json(args.workspace / "selection.json")
        if selection["plan_sha256"] != identity(plan):
            raise ValueError("Plan changed after selection")
        for spec in specs:
            if digest(args.workspace / "runs" / spec["name"] / "best.pt") != selection["checkpoints"][spec["name"]]:
                raise ValueError("Checkpoint changed after validation selection")
        # Test is encoded only after selection is sealed; never fed into optimizer.
        rows = read_rows(args.data / "test.jsonl")
        latents = []
        with torch.no_grad():
            for i in range(0, len(rows), config["cache_batch"]):
                if engine.STOP or remaining() <= 0:
                    return 75
                batch = tok([r["text"] for r in rows[i:i+config["cache_batch"]]], return_tensors="pt",
                            padding="max_length", truncation=True, max_length=config["max_length"]).to("cuda")
                latents.append(ae.get_diffusion_latent(ae.get_encoder()(**batch), batch.attention_mask).float().cpu())
        # validation() reads cache["valid"], so the test latents are placed under that key.
        test_cache = dict(train=cache["train"], valid=torch.cat(latents))
        for spec in specs:
            if engine.STOP or remaining() <= 0:
                return 75
            evaluate_run(args, config, spec, contract, test_cache, ae, tok, "test")
    summarize(args.workspace)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InterruptedError:
        raise SystemExit(75)
