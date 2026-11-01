import json
from pathlib import Path

from tools.prepare_nchlt_afrikaans import (
    normalize_for_deduplication,
    parse_compiled_corpus,
    prepare_dataset,
    stable_document_id,
)


def test_parser_retains_preamble_and_document_metadata():
    corpus = "Licence header\n\n<fn>a.txt</fn>\nEen reël.\n<fn>b.txt</fn>\nTwee.\n"
    preamble, documents = parse_compiled_corpus(corpus)

    assert preamble == "Licence header\n"
    assert [document.source_document for document in documents] == ["a.txt", "b.txt"]
    assert [document.source_index for document in documents] == [0, 1]
    assert documents[0].text == "Een reël."


def test_deduplication_key_is_unicode_case_and_whitespace_conservative():
    composed = "Dié   dokument\nIS dieselfde."
    decomposed = "Die\u0301 dokument is DIESELFDE."

    assert normalize_for_deduplication(composed) == normalize_for_deduplication(
        decomposed
    )
    assert stable_document_id(normalize_for_deduplication(composed)).startswith(
        "nchlt-afr-"
    )
    assert normalize_for_deduplication("woord!") != normalize_for_deduplication(
        "woord"
    )


def test_stage_1_4_outputs_are_auditable_and_preserve_retained_text(tmp_path: Path):
    raw_root = tmp_path / "af"
    source_dir = raw_root / "1.Source"
    corpora_dir = raw_root / "2.Corpora"
    lexica_dir = raw_root / "3.Lexica"
    source_dir.mkdir(parents=True)
    corpora_dir.mkdir()
    lexica_dir.mkdir()

    preamble = "Licence: test\n"
    clean = (
        preamble
        + "<fn>a.txt</fn>\nDié teks bly presies so.\n"
        + "<fn>b.txt</fn>\ndié   teks bly presies so.\n"
        + "<fn>empty.txt</fn>\n   \n"
    )
    (raw_root / "Info.NCHLT.Corpora.Readme.txt").write_text(
        "test readme", encoding="utf-8"
    )
    (corpora_dir / "CORP.NCHLT.af.CLEAN.2.0.txt").write_text(
        clean, encoding="utf-8"
    )
    (corpora_dir / "CORP.NCHLT.af.RAW.2.0.txt").write_text(
        clean, encoding="utf-8"
    )
    for name in ("FREQ.LEX.NCHLT.af.txt", "NELIST.NCHLT.af.txt", "NELIST.NCHLT.all.txt"):
        (lexica_dir / name).write_text("item\n", encoding="utf-8")
    for name in ("a.txt", "b.txt", "empty.txt", "uncompiled.txt"):
        (source_dir / name).write_text(name, encoding="utf-8")

    output_dir = tmp_path / "output"
    stats = prepare_dataset(raw_root, output_dir)
    documents = [
        json.loads(line)
        for line in (output_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    duplicate_groups = [
        json.loads(line)
        for line in (output_dir / "duplicate_groups.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    inventory = [
        json.loads(line)
        for line in (output_dir / "source_inventory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert stats["documents"] == {
        "duplicate_copies_removed": 1,
        "duplicate_groups": 1,
        "empty_removed": 1,
        "exact_duplicate_copies_removed": 0,
        "markers": 3,
        "normalized_only_duplicate_copies_removed": 1,
        "retained": 1,
        "unique_marker_filenames": 3,
    }
    assert documents[0]["text"] == "Dié teks bly presies so."
    assert documents[0]["duplicate_count"] == 1
    assert duplicate_groups[0]["removed"][0]["source_document"] == "b.txt"
    assert duplicate_groups[0]["removed"][0]["match_type"] == "normalized"
    assert stats["source_alignment"]["source_files_absent_from_compiled_markers"] == [
        "uncompiled.txt"
    ]
    assert len(inventory) == 10
    assert all(len(item["sha256"]) == 64 for item in inventory)
    assert (output_dir / "corpus_preamble.txt").read_text(encoding="utf-8") == preamble
