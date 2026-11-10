# NCHLT Afrikaans preprocessing: stages 1-8

The NCHLT Clean Afrikaans Text Corpus v2.0 is processed conservatively, without
changing the downloaded package, so that cleaning decisions can be measured and
reversed.

## Stages 1-4: `python tools/prepare_nchlt_afrikaans.py`

1. **Inventory.** Every file under `data/nchlt_afrikaans/raw/af` is recorded with
   path, size and SHA-256; the manifest adds licence, attribution, SADiLaR record,
   key checksums, section sizes and a tree checksum.
2. **Source.** Model text comes from `2.Corpora/CORP.NCHLT.af.CLEAN.2.0.txt`. The
   RAW corpus, lexicon and named-entity lists are hashed but not used as text.
3. **Documents.** Each `<fn>filename</fn>` marker starts a document, keeping
   filename and order. The licence preamble is saved separately.
4. **Removal.** Whitespace-only documents are removed. Exact duplicate bodies
   (NFC, case-folded, collapsed whitespace) keep the first body unchanged, with
   every removed filename audited. Punctuation and numbers are kept.

Filenames in `<fn>` markers often contain inserted whitespace, so provenance
matching reports exact matches, then NFC/case-folded matches with whitespace
removed.

Output, `data/nchlt_afrikaans/interim/stage_1_4/`: `manifest.json`,
`source_inventory.jsonl`, `corpus_preamble.txt`, `documents.jsonl`,
`duplicate_groups.jsonl`, `empty_documents.jsonl`, `stats.json`.

## Stages 5-7: `python tools/finalize_nchlt_afrikaans.py`

5. **Normalise.** NFC; control/format characters removed; horizontal whitespace
   collapsed without joining lines; email addresses, URLs and phone-like
   sequences masked as `<EMAIL>`, `<URL>` and `<PHONE>`. Wording, punctuation,
   case and diacritics are otherwise kept.
6. **Filter and segment.** Lines are sentence-like units. A line is boilerplate
   only if it occurs in at least 20 documents, has at least five words and
   contains a contact or site-navigation cue. Lines over 50 words are split at
   sentence punctuation where possible, then at the word cap. Segments under three words and
   exact normalised duplicates are removed. All removals are audited.
7. **Split.** Segments are deduplicated globally, so no exact text crosses
   splits; whole documents are then
   assigned by a seeded greedy procedure targeting 80%/10%/10%, balancing words
   and document counts. No document crosses splits.

Output, `data/nchlt_afrikaans/processed/stage_5_7/`: normalised documents,
removal audits, split assignments, `train.jsonl`, `valid.jsonl`, `test.jsonl`, a
manifest and statistics. Documents left with no unique segment keep their split
assignment and are listed separately. The NCHLT frequency lexicon, derived from the
same corpus, gives segment coverage as a diagnostic only, never as a filter.

## Stage 8: mT5 tokenisation: `python tools/audit_afrikaans_tokenizer.py`

Splits load with `--dataset_name nchlt_afrikaans` and optional `--dataset_dir`.
With `google/mt5-base`: 47,767/5,905/5,692 segments; fertility 1.86, 1.87 and 1.89
tokens per whitespace word (train/validation/test); no unknown tokens; `<EMAIL>`,
`<PHONE>` and `<URL>` take three tokens each.

| Token limit | Train | Validation | Test |
|---|---:|---:|---:|
| 64 | 27.67% | 29.96% | 35.96% |
| 96 | 2.29% | 2.51% | 4.36% |
| 128 | 0.20% | 0.03% | 0.11% |

128 tokens is the preregistered starting length; any reduction is reported with
its truncation rate. Amendment: mT5 keeps its SentencePiece vocabulary instead of
a new corpus BPE, preserving the decoder-vocabulary link, so the matched-tokenizer
baseline design is revised.

## Autoencoder smoke and pilot

```bash
bash scripts/research/afrikaans_autoencoder_smoke.sh
```

On an RTX 3060 with BF16, the one-step smoke run verified all 47,767/5,905/5,692
records, mT5 loading, the latent forward/backward path, held-out reconstruction,
checkpoint saving (2.64 GB) and resume from step 1 to 2. Token accuracy and exact
reconstruction were 0% after two steps.

```bash
bash scripts/research/afrikaans_autoencoder_pilot.sh
```

The 500-step pilot uses micro-batch 2 with four-way accumulation (effective batch
8), validates and saves atomically at steps 250 and 500 (`model.pt`), logs every
step to `training_metrics.jsonl` and resumes from an existing checkpoint. It is a
feasibility run, not the final model.

## Not included

Language detection, English spaCy, full autoencoder or diffusion training, and
mGPT evaluation.

The source README requires attribution to the South African Department of Arts and
Culture and the Centre for Text Technology (CTexT, North-West University) under the
Creative Commons Attribution 2.5 South Africa licence; it gives no scholarly
citation, as the manifest records.
