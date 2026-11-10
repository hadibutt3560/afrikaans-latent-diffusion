"""Audit mT5 tokenization over the immutable NCHLT Afrikaans splits.

Input: train/valid/test.jsonl from finalize_nchlt_afrikaans.py (default
data/nchlt_afrikaans/processed/stage_5_7); the files are read, never modified.

Output (default data/nchlt_afrikaans/analysis/mt5-base-tokenizer): report.json
with per-split token-length, fertility, unknown-token and truncation statistics,
<split>_length_histogram.json, and longest_examples.jsonl.

Usage: python tools/audit_afrikaans_tokenizer.py [--model-name google/mt5-base]
[--thresholds 64 96 128 192] [--local-files-only]. Requires transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "data" / "nchlt_afrikaans" / "processed" / "stage_5_7"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "nchlt_afrikaans" / "analysis" / "mt5-base-tokenizer"
)
DEFAULT_MODEL = "google/mt5-base"
# Candidate maximum sequence lengths; the report gives how many examples are
# longer than each one.
DEFAULT_THRESHOLDS = (64, 96, 128, 192)
# The masks inserted by stage 5 of the preprocessing; the report shows how the
# tokeniser splits each of them.
PLACEHOLDERS = ("<EMAIL>", "<PHONE>", "<URL>")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], percentage: float) -> float:
    """Return a percentile of already sorted values, with percentage in [0, 1].

    Interpolates linearly between the two nearest ranks (the same convention as
    NumPy's default); returns 0.0 for an empty list.
    """
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def distribution(values: list[float]) -> dict[str, float]:
    """Summarise values as max, mean, median, p90, p95 and p99, rounded to 6 decimals."""
    ordered = sorted(values)
    return {
        "max": round(max(ordered, default=0.0), 6),
        "mean": round(sum(ordered) / len(ordered), 6) if ordered else 0.0,
        "median": round(percentile(ordered, 0.5), 6),
        "p90": round(percentile(ordered, 0.9), 6),
        "p95": round(percentile(ordered, 0.95), 6),
        "p99": round(percentile(ordered, 0.99), 6),
    }


def read_jsonl_batches(
    path: Path,
    batch_size: int,
) -> Iterable[list[dict[str, object]]]:
    """Yield the rows of a JSONL file in lists of up to batch_size.

    Blank lines are skipped; a row without a string "text" field raises ValueError.
    """
    batch: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"Missing string text at {path}:{line_number}")
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def audit_split(
    path: Path,
    tokenizer,
    thresholds: tuple[int, ...],
    batch_size: int,
    longest_examples: int,
) -> tuple[dict[str, object], dict[str, int], list[dict[str, object]]]:
    """Tokenise one split file and summarise it.

    Returns the summary statistics, a histogram of token length to example
    count, and the longest_examples longest rows (longest first).
    """
    token_lengths: list[float] = []
    fertilities: list[float] = []
    histogram: Counter[int] = Counter()
    longest: list[tuple[int, int, dict[str, object]]] = []
    total_words = 0
    total_tokens = 0
    unknown_tokens = 0
    empty_examples = 0
    sequence_number = 0
    unk_token_id = tokenizer.unk_token_id

    for rows in read_jsonl_batches(path, batch_size):
        texts = [str(row["text"]) for row in rows]
        # Special tokens are included (for mT5, the closing </s>) and nothing is
        # truncated, so each length is the full encoded length of the example.
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for row, input_ids in zip(rows, encoded["input_ids"]):
            sequence_number += 1
            token_length = len(input_ids)
            word_count = len(str(row["text"]).split())
            # With an end-of-sequence token added, this stays 0 unless the
            # tokeniser adds no special tokens.
            empty_examples += int(token_length == 0)
            total_words += word_count
            total_tokens += token_length
            if unk_token_id is not None:
                unknown_tokens += input_ids.count(unk_token_id)
            token_lengths.append(float(token_length))
            histogram[token_length] += 1
            if word_count:
                fertilities.append(token_length / word_count)

            record = {
                "document_id": row.get("document_id"),
                "segment_id": row.get("id"),
                "source_document": row.get("source_document"),
                "text": row["text"],
                "token_count": token_length,
                "word_count": word_count,
            }
            # Fixed-size min-heap of the longest examples. sequence_number is
            # unique, so comparisons never reach the dict; when lengths tie at
            # the cut-off, the later row replaces the earlier one.
            heap_item = (token_length, sequence_number, record)
            if len(longest) < longest_examples:
                heapq.heappush(longest, heap_item)
            elif heap_item[:2] > longest[0][:2]:
                heapq.heapreplace(longest, heap_item)

    examples = len(token_lengths)
    summary: dict[str, object] = {
        "empty_tokenizations": empty_examples,
        "examples": examples,
        "fertility": {
            **distribution(fertilities),
            "weighted_tokens_per_word": (
                round(total_tokens / total_words, 6) if total_words else 0.0
            ),
        },
        "token_lengths_with_special_tokens": distribution(token_lengths),
        "total_tokens": total_tokens,
        "total_whitespace_words": total_words,
        # An example counts as over the limit only if strictly longer than it.
        "truncation": {
            str(threshold): {
                "examples_over_limit": sum(
                    length > threshold for length in token_lengths
                ),
                "fraction_over_limit": round(
                    sum(length > threshold for length in token_lengths) / examples,
                    6,
                )
                if examples
                else 0.0,
            }
            for threshold in thresholds
        },
        "unknown_token_rate": (
            round(unknown_tokens / total_tokens, 8) if total_tokens else 0.0
        ),
        "unknown_tokens": unknown_tokens,
    }
    longest_rows = [
        item[2] for item in sorted(longest, key=lambda item: (-item[0], item[1]))
    ]
    return summary, {str(key): value for key, value in sorted(histogram.items())}, longest_rows


def placeholder_report(tokenizer) -> dict[str, object]:
    """Return the token IDs, pieces and count for each placeholder, without special tokens."""
    report: dict[str, object] = {}
    for placeholder in PLACEHOLDERS:
        input_ids = tokenizer.encode(placeholder, add_special_tokens=False)
        report[placeholder] = {
            "ids": input_ids,
            "pieces": tokenizer.convert_ids_to_tokens(input_ids),
            "token_count": len(input_ids),
        }
    return report


def write_json(path: Path, value: object) -> None:
    """Write value as indented, key-sorted UTF-8 JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write one key-sorted JSON object per line, with LF line endings."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_audit(
    dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    batch_size: int = 256,
    longest_examples: int = 20,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Audit the train, valid and test splits and write the report files.

    Returns the report that is also written to report.json.
    """
    # Imported here so the module can be imported without transformers installed.
    from transformers import AutoTokenizer

    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    split_paths = {
        split: dataset_dir / f"{split}.jsonl"
        for split in ("train", "valid", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing processed split files: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, object] = {}
    longest_rows: list[dict[str, object]] = []
    for split, path in split_paths.items():
        summary, histogram, longest = audit_split(
            path,
            tokenizer,
            thresholds,
            batch_size,
            longest_examples,
        )
        split_reports[split] = summary
        write_json(output_dir / f"{split}_length_histogram.json", histogram)
        longest_rows.extend({**row, "split": split} for row in longest)

    report: dict[str, object] = {
        "input": {
            split: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for split, path in split_paths.items()
        },
        "interpretation": {
            "fertility_definition": (
                "mT5 tokens including special tokens divided by whitespace words"
            ),
            "important_caveat": (
                "mT5 uses its pretrained SentencePiece vocabulary; this audit does "
                "not train the proposal's originally specified corpus-specific BPE."
            ),
            "length_definition": "Token count includes model special tokens.",
        },
        "model": {
            "model_name": model_name,
            "model_max_length": tokenizer.model_max_length,
            "tokenizer_class": tokenizer.__class__.__name__,
            "unknown_token": tokenizer.unk_token,
            "unknown_token_id": tokenizer.unk_token_id,
            "vocabulary_size": len(tokenizer),
        },
        "parameters": {
            "batch_size": batch_size,
            "length_thresholds": list(thresholds),
            "longest_examples_per_split": longest_examples,
        },
        "placeholders": placeholder_report(tokenizer),
        "splits": split_reports,
    }
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / "longest_examples.jsonl", longest_rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the tokeniser audit."""
    parser = argparse.ArgumentParser(
        description="Audit mT5 tokenization over processed NCHLT Afrikaans splits."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--longest-examples", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    """Run the audit from the command line and print the report as JSON."""
    args = build_parser().parse_args()
    report = run_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        # Duplicate thresholds are dropped and the rest sorted ascending.
        thresholds=tuple(sorted(set(args.thresholds))),
        batch_size=args.batch_size,
        longest_examples=args.longest_examples,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote tokenizer audit to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
