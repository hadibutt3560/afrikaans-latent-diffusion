"""Prepare the NCHLT Afrikaans corpus through preprocessing stages 1-4.

This script is intentionally conservative. It inventories the untouched source
package, parses document boundaries, removes empty documents, and removes only
exact duplicates after Unicode/case/whitespace normalization. Later cleaning,
segmentation, and train/validation/test splitting are out of scope here.

Input: the unpacked NCHLT Afrikaans package (default data/nchlt_afrikaans/raw/af),
which must contain the files in KEY_ARTIFACTS and a 1.Source directory. Only the
CLEAN corpus is parsed; the other key files are checked and hashed.

Output (default data/nchlt_afrikaans/interim/stage_1_4): source_inventory.jsonl,
corpus_preamble.txt, documents.jsonl, duplicate_groups.jsonl,
empty_documents.jsonl, stats.json and manifest.json.

Usage: python tools/prepare_nchlt_afrikaans.py [--raw-root DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "nchlt_afrikaans" / "raw" / "af"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "nchlt_afrikaans" / "interim" / "stage_1_4"
)
README_RELATIVE_PATH = Path("Info.NCHLT.Corpora.Readme.txt")
CLEAN_CORPUS_RELATIVE_PATH = Path("2.Corpora/CORP.NCHLT.af.CLEAN.2.0.txt")
RAW_CORPUS_RELATIVE_PATH = Path("2.Corpora/CORP.NCHLT.af.RAW.2.0.txt")
KEY_ARTIFACTS = (
    README_RELATIVE_PATH,
    CLEAN_CORPUS_RELATIVE_PATH,
    RAW_CORPUS_RELATIVE_PATH,
    Path("3.Lexica/FREQ.LEX.NCHLT.af.txt"),
    Path("3.Lexica/NELIST.NCHLT.af.txt"),
    Path("3.Lexica/NELIST.NCHLT.all.txt"),
)
# In the compiled corpus each document starts with a line "<fn>filename</fn>";
# lines before the first marker form the corpus preamble.
MARKER_PATTERN = re.compile(r"^<fn>(.*?)</fn>\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceDocument:
    """One document from the compiled corpus: marker position, marker name and text."""

    source_index: int
    source_document: str
    text: str


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_compiled_corpus(text: str) -> tuple[str, list[SourceDocument]]:
    """Split an NCHLT compiled corpus while retaining its preamble."""
    preamble_lines: list[str] = []
    documents: list[SourceDocument] = []
    current_name: str | None = None
    current_lines: list[str] = []

    def finish_document() -> None:
        """Store the lines collected for the current marker as one document."""
        if current_name is None:
            return
        documents.append(
            SourceDocument(
                source_index=len(documents),
                source_document=current_name,
                text="\n".join(current_lines).strip(),
            )
        )

    for line in text.splitlines():
        marker = MARKER_PATTERN.fullmatch(line)
        if marker:
            finish_document()
            current_name = marker.group(1)
            current_lines = []
        elif current_name is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)
    finish_document()

    if not documents:
        raise ValueError("No <fn>document</fn> markers were found in the clean corpus")
    names = [document.source_document for document in documents]
    if len(names) != len(set(names)):
        raise ValueError("Document filenames in the clean corpus are not unique")
    return "\n".join(preamble_lines).strip() + "\n", documents


def normalize_for_deduplication(text: str) -> str:
    """Return a conservative comparison key without changing retained text."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_source_filename(filename: str) -> str:
    """Match compiled markers to source files despite inserted whitespace."""
    normalized = unicodedata.normalize("NFC", filename).casefold()
    return WHITESPACE_PATTERN.sub("", normalized)


def approximate_word_count(text: str) -> int:
    """Count whitespace-separated tokens (punctuation and numbers included)."""
    return len(text.split())


def stable_document_id(normalized_text: str) -> str:
    """Return a content-derived ID from the deduplication key.

    The ID depends only on the normalised text, so it is the same across runs
    and does not change if documents are reordered.
    """
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return f"nchlt-afr-{digest}"


def inventory_source_tree(raw_root: Path) -> list[dict[str, object]]:
    """List every file under raw_root with relative path, size and SHA-256, sorted by path."""
    inventory: list[dict[str, object]] = []
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": path.relative_to(raw_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def tree_sha256(inventory: Iterable[dict[str, object]]) -> str:
    """Hash the inventory (path, bytes, sha256 per line) into one fingerprint.

    The result depends on item order, so pass the sorted inventory.
    """
    digest = hashlib.sha256()
    for item in inventory:
        line = f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write value as indented, key-sorted UTF-8 JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    """Write one key-sorted JSON object per line."""
    # newline="\n" stops Windows from translating line endings to CRLF.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataset(raw_root: Path, output_dir: Path) -> dict[str, object]:
    """Run stages 1-4 and write all outputs to output_dir.

    Returns the statistics dictionary that is also written to stats.json.
    """
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    missing = [str(path) for path in KEY_ARTIFACTS if not (raw_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required NCHLT files are missing: {missing}")

    clean_path = raw_root / CLEAN_CORPUS_RELATIVE_PATH
    clean_text = clean_path.read_text(encoding="utf-8")
    preamble, source_documents = parse_compiled_corpus(clean_text)
    inventory = inventory_source_tree(raw_root)

    empty_documents: list[dict[str, object]] = []
    groups: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in source_documents:
        key = normalize_for_deduplication(document.text)
        if not key:
            empty_documents.append(
                {
                    "reason": "empty_after_whitespace_normalization",
                    "source_document": document.source_document,
                    "source_index": document.source_index,
                }
            )
            continue
        groups[key].append(document)

    retained_documents: list[dict[str, object]] = []
    duplicate_groups: list[dict[str, object]] = []
    exact_duplicate_copies = 0
    normalized_only_duplicate_copies = 0
    for key, members in groups.items():
        # Members are in corpus order, so the first occurrence is kept and
        # later copies are logged as removed.
        kept = members[0]
        document_id = stable_document_id(key)
        retained_documents.append(
            {
                "character_count": len(kept.text),
                "duplicate_count": len(members) - 1,
                "id": document_id,
                "language": "afr",
                "line_count": len(kept.text.splitlines()),
                "source_document": kept.source_document,
                "source_index": kept.source_index,
                "text": kept.text,
                "word_count": approximate_word_count(kept.text),
            }
        )
        if len(members) > 1:
            duplicate_groups.append(
                {
                    "canonical_id": document_id,
                    "count": len(members),
                    "kept_source_document": kept.source_document,
                    "kept_source_index": kept.source_index,
                    "normalization": "Unicode NFC + casefold + collapsed whitespace",
                    "removed": [
                        {
                            "match_type": (
                                "exact" if member.text == kept.text else "normalized"
                            ),
                            "source_document": member.source_document,
                            "source_index": member.source_index,
                        }
                        for member in members[1:]
                    ],
                }
            )
            exact_duplicate_copies += sum(
                member.text == kept.text for member in members[1:]
            )
            normalized_only_duplicate_copies += sum(
                member.text != kept.text for member in members[1:]
            )

    # Source alignment is reported only; it does not affect which documents are
    # kept. Marker names and 1.Source filenames are compared on normalised keys
    # (see normalize_source_filename), and a key collision is treated as an error.
    source_dir = raw_root / "1.Source"
    source_names = {path.name for path in source_dir.iterdir() if path.is_file()}
    marker_names = {document.source_document for document in source_documents}
    source_names_by_key: dict[str, str] = {}
    for name in source_names:
        key = normalize_source_filename(name)
        if key in source_names_by_key:
            raise ValueError(
                "Source filenames collide after filename normalization: "
                f"{source_names_by_key[key]!r} and {name!r}"
            )
        source_names_by_key[key] = name
    marker_names_by_key = {
        normalize_source_filename(name): name for name in marker_names
    }
    section_inventory: dict[str, dict[str, int]] = {}
    for item in inventory:
        # Section = top-level folder of the package (e.g. 1.Source, 2.Corpora).
        section = str(item["path"]).split("/", 1)[0]
        summary = section_inventory.setdefault(section, {"bytes": 0, "files": 0})
        summary["files"] += 1
        summary["bytes"] += int(item["bytes"])

    input_words = sum(approximate_word_count(doc.text) for doc in source_documents)
    retained_words = sum(int(doc["word_count"]) for doc in retained_documents)
    duplicate_copies = sum(len(members) - 1 for members in groups.values())
    stats: dict[str, object] = {
        "documents": {
            "duplicate_copies_removed": duplicate_copies,
            "duplicate_groups": len(duplicate_groups),
            "empty_removed": len(empty_documents),
            "exact_duplicate_copies_removed": exact_duplicate_copies,
            "markers": len(source_documents),
            "normalized_only_duplicate_copies_removed": (
                normalized_only_duplicate_copies
            ),
            "retained": len(retained_documents),
            "unique_marker_filenames": len(marker_names),
        },
        "source_alignment": {
            "exact_filename_matches": len(marker_names & source_names),
            "filename_matching": "Unicode NFC + casefold + removed whitespace",
            # Counts every marker name without an exact filename match, so it
            # also includes the names reported as missing in the next field.
            "marker_names_matched_after_normalization": len(marker_names - source_names),
            # Lists normalised keys, not the original marker names.
            "marker_names_missing_from_source_directory": sorted(
                marker_names_by_key.keys() - source_names_by_key.keys()
            ),
            "source_files_absent_from_compiled_markers": sorted(
                source_names_by_key[key]
                for key in source_names_by_key.keys() - marker_names_by_key.keys()
            ),
            "source_files": len(source_names),
        },
        "text": {
            "input_approximate_words": input_words,
            "retained_approximate_words": retained_words,
        },
    }
    manifest: dict[str, object] = {
        "corpus": {
            "language": "Afrikaans",
            "language_code": "afr",
            "name": "NCHLT Clean Afrikaans Text Corpus",
            "version": "2.0",
        },
        "input": {
            "key_artifacts": {
                path.as_posix(): {
                    "bytes": (raw_root / path).stat().st_size,
                    "sha256": sha256_file(raw_root / path),
                }
                for path in KEY_ARTIFACTS
            },
            "section_inventory": section_inventory,
            "total_bytes": sum(int(item["bytes"]) for item in inventory),
            "total_files": len(inventory),
            "tree_sha256": tree_sha256(inventory),
        },
        "licence": {
            "attribution": (
                "South African Department of Arts and Culture & Centre for Text "
                "Technology (CTexT, North-West University, South Africa)"
            ),
            "attribution_url": "http://www.nwu.ac.za/ctext",
            "name": "Creative Commons Attribution 2.5 South Africa",
            "url": "http://creativecommons.org/licenses/by/2.5/za/",
        },
        "provenance": {
            "citation_note": (
                "The supplied README specifies attribution but does not provide a "
                "scholarly citation."
            ),
            "official_record": (
                "https://repo.sadilar.org/items/"
                "289c350d-25a1-4c27-8285-c83b031318c7/full"
            ),
            "source_file": CLEAN_CORPUS_RELATIVE_PATH.as_posix(),
        },
        "processing": {
            "deduplication": "Unicode NFC + casefold + collapsed whitespace",
            "excluded": [
                "boilerplate filtering",
                "contact-information masking",
                "language identification",
                "sentence segmentation",
                "tokenization",
                "train/validation/test splitting",
            ],
            "stages": [
                "inventory and provenance capture",
                "clean-corpus source selection",
                "document-boundary parsing",
                "empty and conservative exact-duplicate removal",
            ],
            "text_policy": (
                "Retained document text is unchanged apart from removal of leading "
                "and trailing whitespace introduced by corpus concatenation."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "source_inventory.jsonl", inventory)
    (output_dir / "corpus_preamble.txt").write_text(preamble, encoding="utf-8")
    write_jsonl(output_dir / "documents.jsonl", retained_documents)
    write_jsonl(output_dir / "duplicate_groups.jsonl", duplicate_groups)
    write_jsonl(output_dir / "empty_documents.jsonl", empty_documents)
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "manifest.json", manifest)
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the stage 1-4 script."""
    parser = argparse.ArgumentParser(
        description="Prepare NCHLT Afrikaans preprocessing stages 1-4."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    """Run stages 1-4 from the command line and print the statistics as JSON."""
    args = build_parser().parse_args()
    stats = prepare_dataset(args.raw_root, args.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote stage 1-4 outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
