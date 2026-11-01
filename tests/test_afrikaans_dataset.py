import json
from pathlib import Path

import pytest

from tools.audit_afrikaans_tokenizer import audit_split


class FakeTokenizer:
    unk_token_id = 99

    def __call__(
        self,
        texts,
        add_special_tokens,
        padding,
        truncation,
        return_attention_mask,
    ):
        assert add_special_tokens
        assert not padding
        assert not truncation
        assert not return_attention_mask
        return {
            "input_ids": [
                list(range(len(text.split()))) + [1]
                for text in texts
            ]
        }


def write_rows(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            handle.write(
                json.dumps(
                    {
                        "id": f"segment-{index}",
                        "document_id": f"document-{index}",
                        "source_document": f"source-{index}.txt",
                        "text": text,
                    }
                )
                + "\n"
            )


def test_tokenizer_audit_reports_lengths_fertility_and_truncation(tmp_path: Path):
    split_path = tmp_path / "train.jsonl"
    write_rows(split_path, ["een twee", "drie vier vyf ses"])

    summary, histogram, longest = audit_split(
        split_path,
        FakeTokenizer(),
        thresholds=(3, 5),
        batch_size=2,
        longest_examples=1,
    )

    assert summary["examples"] == 2
    assert summary["total_tokens"] == 8
    assert summary["fertility"]["weighted_tokens_per_word"] == pytest.approx(8 / 6)
    assert summary["truncation"]["3"]["examples_over_limit"] == 1
    assert summary["truncation"]["5"]["examples_over_limit"] == 0
    assert histogram == {"3": 1, "5": 1}
    assert longest[0]["text"] == "drie vier vyf ses"


def test_nchlt_loader_reads_explicit_split_directory(tmp_path: Path):
    pytest.importorskip("datasets")
    from dataset_utils.text_dataset import get_dataset

    for split in ("train", "valid", "test"):
        write_rows(tmp_path / f"{split}.jsonl", [f"{split} voorbeeld"])

    dataset = get_dataset("nchlt_afrikaans", dataset_dir=tmp_path)

    assert set(dataset) == {"train", "valid", "test"}
    assert dataset["train"].column_names == ["text"]
    assert dataset["valid"][0]["text"] == "valid voorbeeld"
