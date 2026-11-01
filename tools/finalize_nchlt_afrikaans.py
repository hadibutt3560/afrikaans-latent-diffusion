"""Prepare NCHLT Afrikaans preprocessing stages 5-7.

Stage 5 normalizes text and masks contact details. Stage 6 removes data-driven
boilerplate, creates bounded line-based segments, and removes exact segment
duplicates. Stage 7 creates deterministic document-grouped dataset splits.
Every removal and transformation remains auditable in the generated outputs.

Input: the stage 1-4 directory written by prepare_nchlt_afrikaans.py (default
data/nchlt_afrikaans/interim/stage_1_4); documents.jsonl is processed and its
manifest.json and stats.json are hashed for provenance. The NCHLT frequency
lexicon is optional and used only for a diagnostic coverage score.

Output (default data/nchlt_afrikaans/processed/stage_5_7): train/valid/test.jsonl,
normalised documents, audit logs for every removal, document_splits.jsonl,
stats.json and a manifest.json holding the parameters and output hashes.

Usage: python tools/finalize_nchlt_afrikaans.py [--input-dir DIR] [--output-dir DIR]
[--seed N] [--train-ratio R] [--valid-ratio R] (see --help for all options)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "nchlt_afrikaans" / "interim" / "stage_1_4"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "nchlt_afrikaans" / "processed" / "stage_5_7"
)
DEFAULT_LEXICON = (
    REPO_ROOT
    / "data"
    / "nchlt_afrikaans"
    / "raw"
    / "af"
    / "3.Lexica"
    / "FREQ.LEX.NCHLT.af.txt"
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# URL matches run to the next whitespace or <>{}; replace_url hands any
# trailing sentence punctuation back to the text.
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>{}]+")
# Deliberately broad (at least 7 characters of digits, spaces, brackets, "/",
# "." or "-", starting and ending with a digit); replace_phone then masks only
# candidates that look like South African numbers.
PHONE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d ()/.-]{5,}\d)(?!\w)"
)
# All whitespace except "\n", so collapsing spaces keeps the corpus line breaks
# that later become segment boundaries.
HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
COMPARISON_WHITESPACE_PATTERN = re.compile(r"\s+")
# Split after ".", "!" or "?" only when the next word (after optional quotes or
# an opening bracket) starts with a capital, including Latin-1 capitals such as É.
# A sentence that opens with a lower-case word such as Afrikaans "'n" is not
# split here; the word cap in split_long_line still bounds its length.
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[\"'(]*[A-ZÀ-ÖØ-Þ])"
)
# Letters only (no digits or "_"), allowing internal hyphens and straight or
# curly apostrophes. Used only for the lexicon coverage diagnostic.
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}"
SPLIT_NAMES = ("train", "valid", "test")
# Lower case because they are matched against comparison_key(), which casefolds;
# "<email>" and "<url>" therefore match the stage 5 masks. A line also has to
# pass the document-frequency and length thresholds before it is removed.
BOILERPLATE_CUES = (
    "<email>",
    "<url>",
    "| imibuzo",
    "| questions",
    "| vrae",
    "click on a service",
    "government jargon",
    "glossary",
    "just ask us",
    "kliek op 'n diens",
    "sibuze nje",
    "vra vir ons",
    "woordelys",
)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Return path relative to the repository root, or as an absolute path if outside it.

    Used in the manifest so that paths inside the repository do not record the
    local machine's directory layout.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file into a list of objects, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def comparison_key(text: str) -> str:
    """Return the matching key for boilerplate and duplicate detection.

    NFC, casefolded and with whitespace collapsed, the same normalisation as the
    stage 1-4 document deduplication. The stored text itself is not changed.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    return COMPARISON_WHITESPACE_PATTERN.sub(" ", normalized).strip()


def words(text: str) -> list[str]:
    """Return the letter-only word tokens of text (see WORD_PATTERN)."""
    return WORD_PATTERN.findall(text)


def replace_url(match: re.Match[str]) -> str:
    """Replace a URL match with <URL>, keeping trailing punctuation outside the mask."""
    value = match.group(0)
    trailing = value[len(value.rstrip(TRAILING_URL_PUNCTUATION)) :]
    return "<URL>" + trailing


def replace_phone(match: re.Match[str]) -> str:
    """Return <PHONE> if the candidate looks like a South African number, else the original text."""
    value = match.group(0)
    digits = "".join(character for character in value if character.isdigit())
    # "+27" followed by nine more digits (11 in total), or a ten-digit national
    # number starting with 0. Other candidates, such as long numbers or dates,
    # are left unchanged.
    is_south_african_phone = (
        (value.lstrip().startswith("+27") and len(digits) == 11)
        or (digits.startswith("0") and len(digits) == 10)
    )
    return "<PHONE>" if is_south_african_phone else value


def normalize_and_mask_text(text: str) -> tuple[str, dict[str, int]]:
    """Normalize layout and mask limited contact patterns, preserving wording."""
    audit = {
        "control_characters_removed": 0,
        "emails_masked": 0,
        "empty_lines_removed": 0,
        "phones_masked": 0,
        "unicode_nfc_changed": 0,
        "urls_masked": 0,
        "whitespace_normalized_lines": 0,
    }
    normalized = unicodedata.normalize("NFC", text)
    # A 0/1 flag per document, so the stage 5 total counts documents, not characters.
    audit["unicode_nfc_changed"] = int(normalized != text)

    # Drop control (Cc) and format (Cf, e.g. zero-width) characters except
    # newline and tab.
    retained_characters: list[str] = []
    for character in normalized:
        if character in "\n\t" or unicodedata.category(character) not in {"Cc", "Cf"}:
            retained_characters.append(character)
        else:
            audit["control_characters_removed"] += 1
    # "\r" is category Cc, so the loop above has already removed it: the
    # replace calls below find nothing, and a lone "\r" is dropped rather than
    # turned into a line break.
    normalized = "".join(retained_characters).replace("\r\n", "\n").replace("\r", "\n")

    clean_lines: list[str] = []
    for line in normalized.split("\n"):
        clean_line = HORIZONTAL_WHITESPACE_PATTERN.sub(" ", line).strip()
        if clean_line != line:
            audit["whitespace_normalized_lines"] += 1
        if clean_line:
            clean_lines.append(clean_line)
        else:
            audit["empty_lines_removed"] += 1
    normalized = "\n".join(clean_lines)

    # Order matters: emails are masked before URLs, so an address containing
    # "www." is masked whole as an email, and phones are masked last, so digits
    # inside already-masked emails and URLs are not considered.
    normalized, audit["emails_masked"] = EMAIL_PATTERN.subn("<EMAIL>", normalized)
    normalized, audit["urls_masked"] = URL_PATTERN.subn(replace_url, normalized)
    phone_count = 0

    def mask_valid_phone(match: re.Match[str]) -> str:
        """Apply replace_phone and count only the candidates actually masked."""
        nonlocal phone_count
        replacement = replace_phone(match)
        phone_count += int(replacement == "<PHONE>")
        return replacement

    normalized = PHONE_CANDIDATE_PATTERN.sub(mask_valid_phone, normalized)
    audit["phones_masked"] = phone_count
    return normalized, audit


def split_long_line(text: str, max_words: int) -> list[str]:
    """Split a corpus line without exceeding max_words in any output segment."""
    if len(text.split()) <= max_words:
        return [text]

    sentence_like_parts = SENTENCE_BOUNDARY_PATTERN.split(text)
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        """Emit the words collected so far as one chunk and reset the buffer."""
        if current:
            chunks.append(" ".join(current))
            current.clear()

    # Greedily pack whole sentence-like parts into chunks of at most max_words.
    # A part that alone exceeds the cap is cut into consecutive, non-overlapping
    # max_words windows; a short final window may later be removed by the
    # minimum segment length filter in finalize_dataset.
    for part in sentence_like_parts:
        part_words = part.split()
        if len(part_words) > max_words:
            flush_current()
            for start in range(0, len(part_words), max_words):
                chunks.append(" ".join(part_words[start : start + max_words]))
        elif len(current) + len(part_words) <= max_words:
            current.extend(part_words)
        else:
            flush_current()
            current.extend(part_words)
    flush_current()
    return chunks


def is_repeated_boilerplate(
    text: str,
    document_count: int,
    min_documents: int,
    min_words: int,
) -> bool:
    """Require repetition plus a conservative contact/navigation cue."""
    normalized = comparison_key(text)
    return (
        document_count >= min_documents
        and len(text.split()) >= min_words
        and any(cue in normalized for cue in BOILERPLATE_CUES)
    )


def load_lexicon(path: Path | None) -> set[str]:
    """Load the NCHLT frequency lexicon as a set of NFC, casefolded words.

    Returns an empty set when no lexicon file is available, which turns the
    coverage diagnostic off rather than failing.
    """
    if path is None or not path.is_file():
        return set()
    lexicon: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        # Skip the "Word<TAB>Freq" header; the word is the first tab-separated
        # column and the frequency is ignored.
        next(handle, None)
        for line in handle:
            word = line.split("\t", 1)[0].strip()
            if word:
                lexicon.add(unicodedata.normalize("NFC", word).casefold())
    return lexicon


def lexicon_coverage(text: str, lexicon: set[str]) -> float | None:
    """Return the share of word tokens found in the lexicon, or None if either is empty.

    Diagnostic only: the lexicon comes from the same corpus and is not used
    for filtering.
    """
    tokens = [unicodedata.normalize("NFC", token).casefold() for token in words(text)]
    if not lexicon or not tokens:
        return None
    return round(sum(token in lexicon for token in tokens) / len(tokens), 6)


def stable_segment_id(document_id: str, line_index: int, chunk_index: int) -> str:
    """Return a segment ID derived from its position (document, line, chunk).

    Unlike the stage 1-4 document IDs this is positional, not content-based, so
    identical text in two places gets two different IDs.
    """
    value = f"{document_id}:{line_index}:{chunk_index}"
    return "nchlt-afr-seg-" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_balanced_document_splits(
    documents: list[dict[str, object]],
    segments: list[dict[str, object]],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, str]:
    """Assign whole documents with proportional counts and balanced words."""
    ratios = {
        "train": train_ratio,
        "valid": valid_ratio,
        "test": 1 - train_ratio - valid_ratio,
    }

    def proportional_counts(total: int) -> dict[str, int]:
        """Divide total across splits by ratio using largest-remainder rounding.

        Remaining units go to the largest fractional parts; ties follow the
        train, valid, test order.
        """
        raw_counts = {name: total * ratios[name] for name in SPLIT_NAMES}
        counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
        remainder = total - sum(counts.values())
        ranked = sorted(
            SPLIT_NAMES,
            key=lambda name: (
                -(raw_counts[name] - counts[name]),
                SPLIT_NAMES.index(name),
            ),
        )
        for index in range(remainder):
            counts[ranked[index]] += 1
        return counts

    weights = {str(document["id"]): 0 for document in documents}
    for segment in segments:
        weights[str(segment["document_id"])] += int(segment["word_count"])

    total_words = sum(weights.values())
    word_targets = {
        split_name: total_words * ratios[split_name]
        for split_name in SPLIT_NAMES
    }
    document_capacities = proportional_counts(len(documents))
    current_words = {split_name: 0 for split_name in SPLIT_NAMES}
    current_documents = {split_name: 0 for split_name in SPLIT_NAMES}
    assignments: dict[str, str] = {}

    def seeded_order(document_id: str) -> str:
        """Return a seed-dependent hash used as a deterministic ordering key.

        Hashing rather than shuffling makes the order independent of the
        order in which documents are supplied.
        """
        return hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).hexdigest()

    # Largest documents (by word count) are placed first. The seed only breaks
    # ties between documents of equal size and orders the empty documents below.
    nonempty_documents = sorted(
        (document_id for document_id, weight in weights.items() if weight > 0),
        key=lambda document_id: (-weights[document_id], seeded_order(document_id)),
    )
    # Two document quotas: one for documents with words (used here) and one for
    # all documents (used when placing the empty ones afterwards).
    nonempty_capacities = proportional_counts(len(nonempty_documents))
    for document_id in nonempty_documents:
        weight = weights[document_id]
        candidate_scores: list[tuple[float, str]] = []
        # Among splits with document quota left, choose the one that minimises
        # the summed squared relative deviation from every split's word target.
        # max(..., 1.0) avoids division by zero; exact ties go to the
        # alphabetically first split name.
        for candidate in SPLIT_NAMES:
            if current_documents[candidate] >= nonempty_capacities[candidate]:
                continue
            word_error = 0.0
            for split_name in SPLIT_NAMES:
                candidate_words = current_words[split_name] + (
                    weight if split_name == candidate else 0
                )
                word_error += (
                    (candidate_words - word_targets[split_name])
                    / max(word_targets[split_name], 1.0)
                ) ** 2
            candidate_scores.append((word_error, candidate))
        split_name = min(candidate_scores)[1]
        assignments[document_id] = split_name
        current_words[split_name] += weight
        current_documents[split_name] += 1

    # Documents with no remaining segments carry no words, so they only fill
    # the overall document quotas, each going to the split with most room left.
    empty_documents = sorted(
        (document_id for document_id, weight in weights.items() if weight == 0),
        key=seeded_order,
    )
    for document_id in empty_documents:
        available = [
            name
            for name in SPLIT_NAMES
            if current_documents[name] < document_capacities[name]
        ]
        split_name = max(
            available,
            key=lambda name: document_capacities[name] - current_documents[name],
        )
        assignments[document_id] = split_name
        current_documents[split_name] += 1
    return assignments


def summarize_split(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return document, segment and word-count statistics for one split."""
    word_counts = [int(row["word_count"]) for row in rows]
    return {
        "documents": len({str(row["document_id"]) for row in rows}),
        "max_words": max(word_counts, default=0),
        "mean_words": round(sum(word_counts) / len(word_counts), 3) if rows else 0.0,
        "min_words": min(word_counts, default=0),
        "segments": len(rows),
        "words": sum(word_counts),
    }


def finalize_dataset(
    input_dir: Path,
    output_dir: Path,
    lexicon_path: Path | None = DEFAULT_LEXICON,
    boilerplate_min_documents: int = 20,
    boilerplate_min_words: int = 5,
    min_segment_words: int = 3,
    max_segment_words: int = 50,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> dict[str, object]:
    """Run stages 5-7 on the stage 1-4 output and write all files to output_dir.

    The test ratio is 1 - train_ratio - valid_ratio. Returns the statistics
    dictionary that is also written to stats.json.
    """
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Split ratios must leave positive train, valid, and test sets")
    if min_segment_words < 1 or max_segment_words < min_segment_words:
        raise ValueError("Invalid minimum or maximum segment length")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    documents_path = input_dir / "documents.jsonl"
    manifest_path = input_dir / "manifest.json"
    stats_path = input_dir / "stats.json"
    for required_path in (documents_path, manifest_path, stats_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing stage 1-4 input: {required_path}")

    input_documents = read_jsonl(documents_path)
    normalized_documents: list[dict[str, object]] = []
    normalization_audit: list[dict[str, object]] = []
    normalization_totals: Counter[str] = Counter()
    for document in input_documents:
        normalized_text, audit = normalize_and_mask_text(str(document["text"]))
        normalization_totals.update(audit)
        normalized_document = {
            "id": document["id"],
            "language": "afr",
            "source_document": document["source_document"],
            "source_index": document["source_index"],
            "text": normalized_text,
        }
        normalized_documents.append(normalized_document)
        # Only documents with at least one recorded change are logged.
        if any(audit.values()):
            normalization_audit.append(
                {
                    "document_id": document["id"],
                    "source_document": document["source_document"],
                    **audit,
                }
            )

    # Stage 6. Group every line in the corpus by its comparison key, so
    # boilerplate is judged by how many distinct documents contain the line.
    line_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    for document in normalized_documents:
        for line_index, line in enumerate(str(document["text"]).splitlines()):
            line_occurrences[comparison_key(line)].append(
                {
                    "document_id": document["id"],
                    "line_index": line_index,
                    "source_document": document["source_document"],
                    "text": line,
                }
            )

    boilerplate_keys: set[str] = set()
    boilerplate_patterns: list[dict[str, object]] = []
    for key, occurrences in line_occurrences.items():
        document_ids = {str(item["document_id"]) for item in occurrences}
        # The cue and word-count tests use the first occurrence's text.
        representative = str(occurrences[0]["text"])
        if is_repeated_boilerplate(
            representative,
            len(document_ids),
            boilerplate_min_documents,
            boilerplate_min_words,
        ):
            boilerplate_keys.add(key)
            boilerplate_patterns.append(
                {
                    "document_count": len(document_ids),
                    "occurrence_count": len(occurrences),
                    "sources": [
                        {
                            "document_id": item["document_id"],
                            "line_index": item["line_index"],
                            "source_document": item["source_document"],
                        }
                        for item in occurrences
                    ],
                    "text": representative,
                    "word_count": len(representative.split()),
                }
            )
    boilerplate_patterns.sort(
        key=lambda item: (-int(item["document_count"]), str(item["text"]))
    )

    lexicon = load_lexicon(lexicon_path)
    candidate_segments: list[dict[str, object]] = []
    removed_short_segments: list[dict[str, object]] = []
    documents_without_segments: list[dict[str, object]] = []
    for document in normalized_documents:
        before_count = len(candidate_segments)
        # Corpus lines are the base segment unit; long lines are split, and the
        # minimum-length filter is applied after splitting so every dropped
        # piece is logged.
        for line_index, line in enumerate(str(document["text"]).splitlines()):
            if comparison_key(line) in boilerplate_keys:
                continue
            for chunk_index, segment_text in enumerate(
                split_long_line(line, max_segment_words)
            ):
                word_count = len(segment_text.split())
                base_metadata = {
                    "document_id": document["id"],
                    "line_index": line_index,
                    "segment_index": chunk_index,
                    "source_document": document["source_document"],
                    "source_index": document["source_index"],
                }
                if word_count < min_segment_words:
                    removed_short_segments.append(
                        {
                            **base_metadata,
                            "reason": "below_min_segment_words",
                            "text": segment_text,
                            "word_count": word_count,
                        }
                    )
                    continue
                candidate_segments.append(
                    {
                        **base_metadata,
                        "afrikaans_lexicon_coverage": lexicon_coverage(
                            segment_text, lexicon
                        ),
                        "id": stable_segment_id(
                            str(document["id"]), line_index, chunk_index
                        ),
                        "language": "afr",
                        "text": segment_text,
                        "word_count": word_count,
                    }
                )
        if len(candidate_segments) == before_count:
            documents_without_segments.append(
                {
                    "document_id": document["id"],
                    "source_document": document["source_document"],
                }
            )

    # Deduplicate across the whole corpus before splitting, so the same
    # normalised segment cannot end up in two splits. The first occurrence in
    # document and line order is kept.
    unique_segments: list[dict[str, object]] = []
    canonical_by_text: dict[str, dict[str, object]] = {}
    duplicate_segments: dict[str, dict[str, object]] = {}
    for segment in candidate_segments:
        key = comparison_key(str(segment["text"]))
        canonical = canonical_by_text.get(key)
        if canonical is None:
            canonical_by_text[key] = segment
            unique_segments.append(segment)
            continue
        canonical_id = str(canonical["id"])
        group = duplicate_segments.setdefault(
            canonical_id,
            {
                "canonical_id": canonical_id,
                "kept_document_id": canonical["document_id"],
                "kept_source_document": canonical["source_document"],
                "removed": [],
                "text": canonical["text"],
            },
        )
        group["removed"].append(
            {
                "document_id": segment["document_id"],
                "segment_id": segment["id"],
                "source_document": segment["source_document"],
            }
        )

    # Stage 7. Whole documents are assigned to splits, so segments from one
    # source document never appear in more than one split.
    split_rows: dict[str, list[dict[str, object]]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    split_assignments = assign_balanced_document_splits(
        normalized_documents,
        unique_segments,
        seed,
        train_ratio,
        valid_ratio,
    )
    document_splits: dict[str, dict[str, object]] = {}
    for document in normalized_documents:
        document_id = str(document["id"])
        document_splits[document_id] = {
            "document_id": document_id,
            "source_document": document["source_document"],
            "split": split_assignments[document_id],
        }
    for segment in unique_segments:
        document_id = str(segment["document_id"])
        split_name = str(document_splits[document_id]["split"])
        split_rows[split_name].append({**segment, "split": split_name})

    # Unlike documents_without_segments (counted before deduplication), this
    # lists every document with no segment left after duplicate removal; such
    # documents are still assigned a split.
    documents_with_unique_segments = {
        str(segment["document_id"]) for segment in unique_segments
    }
    documents_without_unique_segments = [
        {
            "document_id": document_id,
            "source_document": assignment["source_document"],
            "split": assignment["split"],
        }
        for document_id, assignment in document_splits.items()
        if document_id not in documents_with_unique_segments
    ]

    # Safety check that no normalised segment text occurs twice across the
    # written splits.
    all_segment_keys = [
        comparison_key(str(row["text"]))
        for split_name in SPLIT_NAMES
        for row in split_rows[split_name]
    ]
    if len(all_segment_keys) != len(set(all_segment_keys)):
        raise RuntimeError("Exact normalized segment leakage remains after deduplication")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "normalized_documents.jsonl", normalized_documents)
    write_jsonl(output_dir / "normalization_audit.jsonl", normalization_audit)
    write_jsonl(output_dir / "boilerplate_patterns.jsonl", boilerplate_patterns)
    write_jsonl(output_dir / "removed_short_segments.jsonl", removed_short_segments)
    write_jsonl(output_dir / "segment_duplicate_groups.jsonl", duplicate_segments.values())
    write_jsonl(output_dir / "documents_without_segments.jsonl", documents_without_segments)
    write_jsonl(
        output_dir / "documents_without_unique_segments.jsonl",
        documents_without_unique_segments,
    )
    write_jsonl(
        output_dir / "document_splits.jsonl",
        sorted(document_splits.values(), key=lambda item: str(item["document_id"])),
    )
    for split_name in SPLIT_NAMES:
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows[split_name])

    stats: dict[str, object] = {
        "stage_5": {
            "documents": len(normalized_documents),
            "documents_with_recorded_changes": len(normalization_audit),
            **dict(sorted(normalization_totals.items())),
        },
        "stage_6": {
            "boilerplate_occurrences_removed": sum(
                int(item["occurrence_count"]) for item in boilerplate_patterns
            ),
            "boilerplate_patterns_removed": len(boilerplate_patterns),
            "candidate_segments": len(candidate_segments),
            "documents_without_segments": len(documents_without_segments),
            "documents_without_unique_segments": len(
                documents_without_unique_segments
            ),
            "duplicate_segment_copies_removed": (
                len(candidate_segments) - len(unique_segments)
            ),
            "short_segments_removed": len(removed_short_segments),
            "unique_segments": len(unique_segments),
        },
        "stage_7": {
            # Fixed at 0 because the leakage check above raises otherwise.
            "exact_normalized_segment_overlap_across_splits": 0,
            "splits": {},
        },
    }
    total_split_words = sum(
        int(row["word_count"])
        for split_name in SPLIT_NAMES
        for row in split_rows[split_name]
    )
    for split_name in SPLIT_NAMES:
        summary = summarize_split(split_rows[split_name])
        summary["assigned_documents"] = sum(
            assignment["split"] == split_name
            for assignment in document_splits.values()
        )
        summary["word_fraction"] = round(
            int(summary["words"]) / total_split_words, 6
        )
        stats["stage_7"]["splits"][split_name] = summary
    manifest = {
        "input": {
            "documents_jsonl_sha256": sha256_file(documents_path),
            "manifest_json_sha256": sha256_file(manifest_path),
            "stage_1_4_directory": portable_path(input_dir),
            "stats_json_sha256": sha256_file(stats_path),
        },
        "language": {"code": "afr", "name": "Afrikaans"},
        "lexicon_diagnostic": {
            "caveat": (
                "Coverage uses a lexicon derived from this corpus; it is diagnostic "
                "and is not an independent language-identification score."
            ),
            "entries": len(lexicon),
            "path": portable_path(lexicon_path) if lexicon_path else None,
            "used_for_filtering": False,
        },
        "parameters": {
            "boilerplate_min_documents": boilerplate_min_documents,
            "boilerplate_min_words": boilerplate_min_words,
            "max_segment_words": max_segment_words,
            "min_segment_words": min_segment_words,
            "seed": seed,
            "split_ratios": {
                "test": round(1 - train_ratio - valid_ratio, 10),
                "train": train_ratio,
                "valid": valid_ratio,
            },
        },
        "policies": {
            "boilerplate": (
                "Remove a normalized line only when it meets configured document-"
                "frequency and word-count thresholds and contains a conservative "
                "contact or navigation cue."
            ),
            "contact_masking": "Replace detected emails, URLs, and phone-like digit sequences.",
            "language_filtering": (
                "No automatic language filtering; the NCHLT label is retained and "
                "lexicon coverage is emitted for later review."
            ),
            "segmentation": (
                "Use NCHLT clean-corpus line boundaries; split long lines first at "
                "sentence punctuation and then by the configured word cap."
            ),
            "splitting": (
                "Globally remove normalized exact segment duplicates, then assign "
                "complete source documents with a seeded deterministic greedy "
                "balance of word and document counts."
            ),
        },
    }
    write_json(output_dir / "stats.json", stats)
    # Hash every file currently in output_dir except the manifest itself
    # (it cannot contain its own hash). Files left from an earlier run with a
    # different name would also be listed.
    output_artifacts = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest["output"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_artifacts
    }
    write_json(output_dir / "manifest.json", manifest)
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; defaults match finalize_dataset()."""
    parser = argparse.ArgumentParser(
        description="Prepare NCHLT Afrikaans preprocessing stages 5-7."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--boilerplate-min-documents", type=int, default=20)
    parser.add_argument("--boilerplate-min-words", type=int, default=5)
    parser.add_argument("--min-segment-words", type=int, default=3)
    parser.add_argument("--max-segment-words", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser


def main() -> int:
    """Run stages 5-7 from the command line and print the statistics as JSON."""
    args = build_parser().parse_args()
    stats = finalize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lexicon_path=args.lexicon,
        boilerplate_min_documents=args.boilerplate_min_documents,
        boilerplate_min_words=args.boilerplate_min_words,
        min_segment_words=args.min_segment_words,
        max_segment_words=args.max_segment_words,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote stage 5-7 outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
