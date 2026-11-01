# NCHLT Afrikaans preprocessing: stages 1-7

This checkpoint prepares the NCHLT Clean Afrikaans Text Corpus v2.0 without
changing the downloaded package. It is deliberately conservative so that later
cleaning decisions can be measured and reversed.

## Included stages

1. **Preserve and inventory the source package.** Every file under
   `data/nchlt_afrikaans/raw/af` is recorded with its relative path, byte size,
   and SHA-256 digest. The package-level manifest records the licence,
   attribution, official SADiLaR record, key file checksums, section sizes, and
   a deterministic tree checksum.
2. **Select the modelling source.** The script reads
   `2.Corpora/CORP.NCHLT.af.CLEAN.2.0.txt`. The RAW corpus is retained and
   hashed for audit only. The lexicon and named-entity lists are retained and
   hashed, but are not mixed into the training text.
3. **Parse documents.** Each `<fn>filename</fn>` marker starts one source
   document. Its filename and original order are retained. The corpus licence
   preamble is saved separately and never becomes model text.
4. **Remove unusable documents.** Whitespace-only documents are removed. Exact
   duplicate bodies are grouped using Unicode NFC, case folding, and collapsed
   whitespace; the first body is retained unchanged and every removed source
   filename remains in the duplicate audit file. Punctuation and numbers are
   not removed.

The compiled corpus inserts whitespace into many `<fn>` filenames. Source-file
alignment therefore reports exact matches separately and then compares Unicode
NFC/case-folded filenames with whitespace removed. This normalization is used
only for provenance matching; the recorded source marker itself is unchanged.

## Run

From the repository root:

```powershell
python tools/prepare_nchlt_afrikaans.py
```

The generated checkpoint is written to
`data/nchlt_afrikaans/interim/stage_1_4/`:

- `manifest.json`: provenance, licence, processing policy, and package hashes.
- `source_inventory.jsonl`: path, size, and SHA-256 for every input file.
- `corpus_preamble.txt`: the non-training header before the first document.
- `documents.jsonl`: unique, non-empty documents with stable IDs and metadata.
- `duplicate_groups.jsonl`: retained document and every duplicate source.
- `empty_documents.jsonl`: every removed empty document.
- `stats.json`: document, word-count, and source-alignment totals.

## Stages 5-7

Run the second, independently auditable pass after stages 1-4:

```powershell
python tools/finalize_nchlt_afrikaans.py
```

5. **Normalize and mask limited contact data.** Text is converted to Unicode
   NFC, control/format characters are removed, horizontal whitespace is
   collapsed without joining lines, and detected email addresses, URLs, and
   phone-like sequences are replaced by `<EMAIL>`, `<URL>`, and `<PHONE>`.
   Afrikaans wording, punctuation, case, and diacritics are otherwise retained.
6. **Filter and segment conservatively.** NCHLT's clean-corpus line boundaries
   are treated as sentence-like units. A line is classified as boilerplate only
   when it occurs in at least 20 documents, contains at least five words, and
   contains a conservative contact or site-navigation cue. Frequency alone does
   not remove repeated substantive text.
   Lines longer than 50 words are split at sentence punctuation where possible,
   then by the word cap. Segments below three words and normalized exact segment
   duplicates are removed. Every pattern, short segment, and duplicate remains
   in an audit file.
7. **Create leakage-safe splits.** Exact normalized segments are deduplicated
   globally before assignment, preventing exact text overlap between splits.
   Complete source documents are then assigned to train/validation/test by a
   seeded deterministic greedy procedure targeting 80%/10%/10% while balancing
   both word totals and document counts. No document crosses split boundaries.

The generated stage 5-7 checkpoint is written to
`data/nchlt_afrikaans/processed/stage_5_7/`. It includes normalized documents,
all removal audits, document split assignments, `train.jsonl`, `valid.jsonl`,
`test.jsonl`, a manifest, and split statistics.

All retained stage 1-4 documents receive an immutable split assignment, even if
boilerplate and global segment deduplication leave a document with no unique
training segment. Those cases are listed separately rather than disappearing
from provenance.

The supplied NCHLT frequency lexicon is used only to emit per-segment coverage
as a review diagnostic. Because that lexicon was derived from the same corpus,
coverage is not an independent language-identification result and is never used
to filter examples.

## Stage 8: mT5 tokenisation audit

The processed splits are now loadable with `--dataset_name nchlt_afrikaans` and
an optional `--dataset_dir`. The tokenizer audit is reproducible with:

```powershell
python tools/audit_afrikaans_tokenizer.py
```

Using the pretrained `google/mt5-base` SentencePiece vocabulary, the 80/10/10
splits contain 47,767/5,905/5,692 segments. Weighted fertility is 1.86 tokens
per whitespace word on train, 1.87 on validation, and 1.89 on test. There are no
unknown tokens in this corpus. The `<EMAIL>`, `<PHONE>`, and `<URL>` placeholders
each use three tokens.

The original 64-token limit would truncate 27.67% of train, 29.96% of validation,
and 35.96% of test segments. At 96 tokens those values fall to 2.29%, 2.51%, and
4.36%. At 128 tokens they are 0.20%, 0.03%, and 0.11%. Therefore 128 tokens is
the preregistered starting length for the Afrikaans autoencoder smoke run; any
reduction for memory must be reported with its measured truncation rate.

This is a methodology amendment to the proposal: mT5 uses its existing
SentencePiece vocabulary rather than a newly trained corpus-specific BPE. This
preserves the pretrained decoder-vocabulary relationship but means the original
matched-tokenizer baseline design must be revised explicitly.

The one-step model-path check is:

```bash
bash scripts/research/afrikaans_autoencoder_smoke.sh
```

This smoke check completed on an RTX 3060 with BF16 and produced a resumable
2.64 GB checkpoint. It verified all 47,767/5,905/5,692 dataset records, mT5
loading, the latent forward/backward path, held-out reconstruction, checkpoint
saving, and resume from step 1 to step 2. As expected, two optimization steps
do not constitute training: validation token accuracy and exact reconstruction
were both 0%, and generated text was incoherent.

The next controlled run is a 500-step pilot with micro-batch 2 and four-way
gradient accumulation (effective batch 8). It saves and validates at steps 250
and 500, writes every step to `training_metrics.jsonl`, and resumes automatically
if its output checkpoint already exists:

```bash
bash scripts/research/afrikaans_autoencoder_pilot.sh
```

Checkpoint replacement is atomic, so an interrupted write does not destroy the
last complete `model.pt`. The pilot remains a tuning and feasibility run; it is
not the final Afrikaans model or a basis for reporting test-set performance.

## Explicitly deferred

These stages do not run an unvalidated language detector, use English spaCy,
complete Afrikaans autoencoder or diffusion training, or perform mGPT
evaluation. The mT5 tokenizer, dataset loader, smoke run, checkpoint resume, and
pilot command are integrated, but model adequacy still requires a successful
pilot followed by held-out reconstruction evidence.

The source README requires attribution to the South African Department of Arts
and Culture and the Centre for Text Technology (CTexT, North-West University)
under the Creative Commons Attribution 2.5 South Africa licence. The supplied
README gives required attribution but no scholarly citation; the manifest
records that limitation rather than inventing one.
