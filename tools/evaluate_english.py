#!/usr/bin/env python
"""Strict, machine-readable English ROCStories evaluation."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FINAL_SEEDS = [42, 43, 44, 45, 46]


def read_roc_text(path):
    texts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            texts.append(value[0])
    return texts


def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_seeded_files(values):
    result = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"Expected SEED=PATH, got: {value}") from exc
        result[seed] = Path(path_text)
    return result


def clean_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_payload(value):
    if isinstance(value, dict):
        return {key: clean_payload(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [clean_payload(nested) for nested in value]
    return clean_json_value(value)


def flatten(prefix, value):
    rows = []
    if isinstance(value, dict):
        for key, nested in value.items():
            rows.extend(flatten(f"{prefix}.{key}" if prefix else key, nested))
    else:
        rows.append((prefix, clean_json_value(value)))
    return rows


def save_results(output_prefix, payload):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = clean_payload(payload)
    with open(output_prefix.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    with open(
        output_prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(flatten("", payload))


def evaluate_generation(args):
    from evaluation import evaluation

    seeded_files = parse_seeded_files(args.samples)
    if not args.allow_nonstandard_count and sorted(seeded_files) != FINAL_SEEDS:
        raise ValueError(f"Final evaluation requires exactly seeds {FINAL_SEEDS}")

    train = read_roc_text(REPO_ROOT / "datasets" / "ROCstory" / "roc_train.json")
    valid_source = read_roc_text(REPO_ROOT / "datasets" / "ROCstory" / "roc_valid.json")
    test = valid_source[1000:]
    evaluation.set_metric_policy(strict=not args.allow_missing_required_metrics)

    reference_samples = test[:1000]
    reference_metrics = {
        "sample_count": len(reference_samples),
        "gpt2_large_perplexity": evaluation.compute_perplexity(
            reference_samples, model_id="gpt2-large"
        ),
        "unique_word_count": evaluation.compute_wordcount(reference_samples),
        "unique_sentence_count": evaluation.compute_unique_sentence_count(reference_samples),
        "training_4gram_memorization": evaluation.compute_memorization(
            reference_samples, train, n=4
        ),
        **evaluation.compute_diversity(reference_samples),
    }

    per_seed = {}
    for seed, sample_path in sorted(seeded_files.items()):
        samples = read_nonempty_lines(sample_path)
        if not args.allow_nonstandard_count and len(samples) != 1000:
            raise ValueError(
                f"Seed {seed} has {len(samples)} non-empty samples; final evaluation requires 1000"
            )
        references = test[:len(samples)]
        metrics = {
            "sample_count": len(samples),
            "gpt2_large_perplexity": evaluation.compute_perplexity(
                samples, model_id="gpt2-large"
            ),
            "unique_word_count": evaluation.compute_wordcount(samples),
            "unique_sentence_count": evaluation.compute_unique_sentence_count(samples),
            "training_4gram_memorization": evaluation.compute_memorization(
                samples, train, n=4
            ),
            **evaluation.compute_diversity(samples),
        }
        if not args.skip_mauve:
            metrics["mauve_gpt2_large"], _ = evaluation.compute_mauve(
                samples, references, "gpt2-large", strict=False
            )
        per_seed[str(seed)] = metrics

    numeric_keys = sorted(
        set.intersection(*[
            {key for key, value in metrics.items() if isinstance(value, (int, float))}
            for metrics in per_seed.values()
        ])
    )
    aggregate = {}
    for key in numeric_keys:
        values = [
            float(metrics[key])
            for metrics in per_seed.values()
            if math.isfinite(float(metrics[key]))
        ]
        aggregate[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "valid_seed_count": len(values),
        }
    payload = {
        "experiment": "English ROCStories diffusion generation",
        "language": "English",
        "paper_reproduction_perplexity_model": "gpt2-large",
        "real_test_reference": reference_metrics,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "notes": "These English tokenisation/metric choices must be redesigned for Afrikaans.",
    }
    save_results(Path(args.output_prefix), payload)


def simple_token_accuracy(predictions, references):
    correct = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        total += len(reference_tokens)
        correct += sum(
            predicted == expected
            for predicted, expected in zip(prediction_tokens, reference_tokens)
        )
    return correct / total if total else 0.0


def evaluate_autoencoder(args):
    from evaluation import evaluation

    rows = []
    with open(args.examples, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    references = [row["reference"] for row in rows]
    predictions = [row[args.prediction_key] for row in rows]
    metrics = {
        "example_count": len(rows),
        "exact_reconstruction_rate": (
            sum(prediction == reference for prediction, reference in zip(predictions, references))
            / len(rows) if rows else 0.0
        ),
        "whitespace_token_accuracy": simple_token_accuracy(predictions, references),
        "bleu": evaluation.compute_bleu(predictions, references),
        "rouge_l": evaluation.compute_rouge(predictions, references)["rougeL"],
    }
    if args.validation_metrics:
        with open(args.validation_metrics, "r", encoding="utf-8") as handle:
            trainer_metrics = json.load(handle)
        metrics["validation_cross_entropy"] = trainer_metrics.get(
            "autoencoder/valid_cross_entropy",
            trainer_metrics.get("autoencoder/test_cross_entropy"),
        )
        metrics["model_token_accuracy"] = trainer_metrics.get(
            "autoencoder/valid_token_accuracy",
            trainer_metrics.get("autoencoder/test_token_accuracy"),
        )
    payload = {
        "experiment": "English ROCStories autoencoder reconstruction",
        "language": "English",
        "metrics": metrics,
    }
    save_results(Path(args.output_prefix), payload)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generation = subparsers.add_parser("generation")
    generation.add_argument(
        "--samples", action="append", required=True, metavar="SEED=PATH",
        help="Repeat once per generation seed.",
    )
    generation.add_argument("--output-prefix", required=True)
    generation.add_argument("--skip-mauve", action="store_true")
    generation.add_argument("--allow-nonstandard-count", action="store_true")
    generation.add_argument("--allow-missing-required-metrics", action="store_true")
    generation.set_defaults(func=evaluate_generation)

    autoencoder = subparsers.add_parser("autoencoder")
    autoencoder.add_argument("--examples", required=True)
    autoencoder.add_argument("--validation-metrics")
    autoencoder.add_argument("--prediction-key", default="beam_autoencoder")
    autoencoder.add_argument("--output-prefix", required=True)
    autoencoder.set_defaults(func=evaluate_autoencoder)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
