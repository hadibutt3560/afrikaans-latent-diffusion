import json
from pathlib import Path

from tools.finalize_nchlt_afrikaans import (
    build_parser,
    comparison_key,
    finalize_dataset,
    normalize_and_mask_text,
    split_long_line,
)


def test_default_split_is_80_10_10():
    args = build_parser().parse_args([])
    assert args.train_ratio == 0.8
    assert args.valid_ratio == 0.1


def test_normalization_masks_contacts_without_losing_sentence_punctuation():
    text = (
        "Skryf aan vrae@example.org.  Besoek www.example.org/path.\n"
        "Bel 0860 00 4367 vir hulp.\u200b"
    )
    normalized, audit = normalize_and_mask_text(text)

    assert normalized == (
        "Skryf aan <EMAIL>. Besoek <URL>.\nBel <PHONE> vir hulp."
    )
    assert audit["emails_masked"] == 1
    assert audit["urls_masked"] == 1
    assert audit["phones_masked"] == 1
    assert audit["control_characters_removed"] == 1

    population, population_audit = normalize_and_mask_text(
        "Die fonds dek nou 1 000 000 lewens."
    )
    assert population == "Die fonds dek nou 1 000 000 lewens."
    assert population_audit["phones_masked"] == 0


def test_long_line_segmentation_is_bounded_and_lossless_in_words():
    text = "Een twee drie vier. Vyf ses sewe agt nege. Tien elf twaalf."
    segments = split_long_line(text, max_words=5)

    assert all(len(segment.split()) <= 5 for segment in segments)
    assert " ".join(segment for segment in segments).split() == text.split()


def write_stage_1_4_fixture(input_dir: Path) -> None:
    input_dir.mkdir(parents=True)
    documents = [
        {
            "id": "doc-a",
            "source_document": "a.txt",
            "source_index": 0,
            "text": (
                "Vra ons gerus by vrae@example.org vir hulp.\n"
                "Unieke Afrikaanse inhoud bly hier."
            ),
        },
        {
            "id": "doc-b",
            "source_document": "b.txt",
            "source_index": 1,
            "text": "Vra ons gerus by vrae@example.org vir hulp.\nNog unieke inhoud vir toetsing.",
        },
        {
            "id": "doc-c",
            "source_document": "c.txt",
            "source_index": 2,
            "text": "Unieke Afrikaanse inhoud bly hier.\nNet twee",
        },
        {
            "id": "doc-d",
            "source_document": "d.txt",
            "source_index": 3,
            "text": "Laaste dokument bevat genoeg nuttige woorde.",
        },
    ]
    with (input_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    (input_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (input_dir / "stats.json").write_text("{}\n", encoding="utf-8")


def test_finalization_audits_removals_and_prevents_split_leakage(tmp_path: Path):
    input_dir = tmp_path / "stage_1_4"
    output_dir = tmp_path / "stage_5_7"
    write_stage_1_4_fixture(input_dir)

    stats = finalize_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        lexicon_path=None,
        boilerplate_min_documents=2,
        boilerplate_min_words=6,
        min_segment_words=3,
        max_segment_words=8,
        train_ratio=0.5,
        valid_ratio=0.25,
    )
    rows = []
    for split_name in ("train", "valid", "test"):
        rows.extend(
            json.loads(line)
            for line in (output_dir / f"{split_name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )

    keys = [comparison_key(row["text"]) for row in rows]
    document_to_split = {}
    for row in rows:
        previous = document_to_split.setdefault(row["document_id"], row["split"])
        assert previous == row["split"]
    assert len(keys) == len(set(keys))
    assert stats["stage_6"]["boilerplate_patterns_removed"] == 1
    assert stats["stage_6"]["duplicate_segment_copies_removed"] == 1
    assert stats["stage_6"]["documents_without_unique_segments"] == 1
    assert stats["stage_6"]["short_segments_removed"] == 1
    assert stats["stage_7"]["exact_normalized_segment_overlap_across_splits"] == 0
    assert sum(
        split["assigned_documents"]
        for split in stats["stage_7"]["splits"].values()
    ) == 4
