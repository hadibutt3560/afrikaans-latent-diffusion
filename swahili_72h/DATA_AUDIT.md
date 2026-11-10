# Supplied Swahili benchmark

Source: `C:\Users\SCIENTIST\Downloads\cleaned-20261009T091046Z-1-001`.
Original manifest SHA256: `9dedc47978730f5651d703cdac413fa71346928118c753d9ddb55e7836fb139c`.

- All 738 listed text files present; no repeated membership or extra corpus files.
- 664 training, 37 validation, 37 test files, exactly as supplied. `val` is only renamed `valid` in filenames.
- 154,867,606 source bytes; 23,628,475 whitespace-delimited words. Strict UTF-8; no empty files, replacement characters, NULs, or BOMs.
- An extra `Open Notebook.onetoc2` is explicitly classified as non-corpus metadata and hashed in the audit.
- Every source file is a single line, including multi-megabyte concatenations. Original documents/articles and their boundaries cannot be reconstructed. A file must not be described as one original document.
- No full-file duplicates. Exported windows contain 11 exact duplicate groups, including 3 across splits; 14 normalized duplicate groups, including 3 across splits. A minimum-hash 13-word-shingle screen flags 2,975 repeated groups, including 480 across splits. This is an indicator, not exhaustive near-duplicate detection. See the SQLite index and JSON examples.

All text, duplicates, and supplied file assignments are retained. No cleaning, split repair, or benchmark filtering is applied. Any future correction requires a separately agreed version shared with the discrete experiment. Results from this version cannot establish document-independent generalization.

## Reproducible passage export

Nonoverlapping windows of at most 48 whitespace-delimited words, bounded by observed lines/files. Preserve exact text slices, including original whitespace. IDs are SHA256 of the filename, original file SHA256, and character offsets. Each row carries `source_file`, `source_sha256`, `char_start`, `char_end`, `text_sha256`, `split`, and `document_id: null`. Offsets are Python Unicode code-point offsets, not UTF-8 byte offsets.

Exports: 437,326 training, 29,761 validation, 25,537 test passages. A full verification joined every file's exported passages and reproduced its decoded source text exactly. Raw source bytes are also bundled. These windows and IDs are the explicit shared passage proposal; Jordan's file membership is unchanged. A discrete model should consume these same JSONL rows for passage-level comparisons. Until both pipelines do so, only file-level split alignment is established.

`data/audit.json` records all file hashes, membership, duplication evidence, policy, and JSONL hashes. `data/audit-index.sqlite` supports deeper review. `data/split_manifest.json` is copied byte-for-byte.

## Tokenizer audit

Pretrained mT5 SentencePiece tokenizer, legacy T5 behavior, EOS included, model cap 192. Full corpus audit: **zero truncated passages in every split**. Unknown tokens: training 212 / 42,466,274 (0.0004992%); validation 0 / 2,891,548; test 0 / 2,468,982. Tokenization of test text is an input-integrity check, not model evaluation or test-based protocol tuning.

`tokens/*-views.jsonl` maps canonical IDs to array rows and lengths. `tokens/audit.json` and NPY checksums freeze the model views. Unicode normalization/case folding are used only for duplicate detection, never to alter model text. A future corpus causing unknown fraction >0.1% or truncation >5% on train/validation stops before training.
