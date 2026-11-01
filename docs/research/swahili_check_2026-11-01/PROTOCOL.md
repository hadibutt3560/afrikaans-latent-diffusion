# Swahili checks: reconstruction, random latents, language and copying

**Status: APPROVED** by Hadi Butt in chat, 1 November 2026, before any model was run. Once approved, any change is made only by adding a dated note at the end.

## 1. Purpose and scope

The Swahili run (job 66417, completed 20 October 2026) used the same pipeline as the Afrikaans study and reported only denoising MSE and mGPT perplexity. This protocol applies the checks that changed how the Afrikaans results are read:
- passage-level reconstruction with a shuffled-latent control;
- a random-latent decoder control;
- language identification;
- copying from the training split.

**Scope limits:**
- **Inference only.** No training, no new seeds and no AR baseline.
- **Validation side for everything new.** The existing test samples and scores were produced and summarised earlier, so they are reported as they stand and are not reanalysed.
- **Exploratory.** All results are descriptive. With two seeds per size and a training budget that differed between sizes, nothing here compares architectures.

**Permitted paper use:** one short paragraph in the main text (§8).

## 2. Fixed inputs (hashes recorded before any model loads)

| Input | Location |
|---|---|
| Swahili autoencoder | `E:/swahili-check-20261101/handoff/.../results/swahili72/ae/best.pt` (unpacked from `handoff_2026-10-20/transfer/04-swahili-selected.tar`) |
| Diffusion checkpoints | `.../runs/{small,published}-{42,43}/best.pt`, used with EMA weights via the engine's `inference_checkpoint` |
| Code | `handoff_2026-10-20/github-research-diffusion/project/swahili_72h/engine.py` and its `vendor/`, imported unchanged |
| Model assets | `swahili_72h/assets/mt5-base`, `swahili_72h/assets/mgpt` |
| Data | `swahili_72h/data/{train,valid}.jsonl` (437,326 and 29,761 passages) |
| Existing samples and scores | `results/swahili72/runs/*/valid-samples.json`, `results/swahili72/scores/*` |
| Language identification | GlotLID, revision `baa20d1…`, the copy already pinned for the Afrikaans evaluation package |
| Environment | `E:/afrikaans-eval-20261031/env`; versions recorded in `contract.json` |

## 3. Integrity and reproduction gates (run first; any failure stops the run)

- **G1. Hashes.** Every input above is hashed into `contract.json`. The checkpoint hashes must match the handoff manifest wherever the manifest lists them.
- **G2. Scorer reproduction.** Re-score the first three `real-valid` references and the first three `small-42-valid` samples with the engine's `score` function. Each perplexity must match its stored value within a relative 1e-4.
- **G3. Autoencoder reproduction.** Recompute teacher-forced validation CE and token accuracy on the engine's fixed 1,024-passage validation draw (`validation_seed` 18371), using BF16 autocast as the engine does. They must lie within 2% (relative) of the stored values in `ae-test.json` (CE 0.5601, accuracy 0.8568).
- **G4. Generation reproduction (informational only).** Regenerate the first two `small-42` validation samples. An exact match is reported but not required, because beam decoding on a different GPU need not reproduce bit for bit.

## 4. S1: reconstruction (validation)

**Panel:** 200 validation passages, drawn with `numpy.random.default_rng(20261102)` after sorting by passage id, with at most 6 per source file. The panel ids are written to `panel.json` before any decoding.

**Correct arm:** encode each passage, then decode with the generation settings (beam 4, `min_length` 5, `max_length` 192, repetition penalty 1.2, no repeated trigrams). These match how diffusion samples are decoded and how the Afrikaans reconstructions were measured.

**Shuffled arm:** each passage receives the latent of a passage from a *different* source file in the same token-length bin (≤64, 65–128, 129–192 tokens). This is a seeded derangement (seed 20261103). If a bin has only one file, the nearest bin is used and the event is logged.

**Readouts per arm:**
- teacher-forced CE and token accuracy;
- exact match, comparing decoded output with the decoded model input;
- mean sentence chrF (SacreBLEU 2.6.0, character order 6, no word n-grams);
- numeral recall and precision, using the regex `\d+(?:[.,]\d+)*` on both texts;
- outputs at the length cap.

All readouts are also given by token-length bin.

**Intervals:** source-file bootstrap, 2,000 replicates, seed 20261104, for the exact-match rate, the mean chrF and the correct-minus-shuffled CE difference.

## 5. S2: random-latent decoder control

- **Latents:** 1,000 latents. For i = 0…999: `torch.Generator('cpu').manual_seed(40000 + i)`, then `randn(1, 32, 64)`, with each slot rescaled to the mean slot norm of the S1 panel's encoded latents (recorded; 8 is expected).
- **Decoding and scoring:** decode with the S1 generation settings, then score every output with the engine's `score` function (mGPT, FP32, window 512, stride 256).
- **Readouts:** token-weighted perplexity, mean words, distinct-2, empty outputs, and outputs containing `<extra_id_`.

**Wording fixed now:**
- If the random-latent perplexity is at most 1.20× the mean of the four diffusion validation perplexities: "random latents decoded by the same decoder reach similar raw perplexity".
- If it is above 1.20×: "random latents score clearly worse than diffusion samples".

## 6. S3: language identification

**Classifier:** GlotLID top label on each text, with line breaks replaced by spaces.

**Banks:**
- `real-valid` (256);
- the four diffusion validation sample sets (256 each);
- the S2 random latents (1,000);
- the S1 panel sources and reconstructions.

**Readouts:**
- the share whose top label is `swh_Latn`;
- the share labelled with any other Swahili code, recorded separately;
- the share labelled `eng_Latn`.

Each share is pooled and by word-length band (1–10, 11–20, 21–35, 36+). No Swahili dictionary check is made, because no pinned Swahili word list is available.

## 7. S4: length and copying

**Length:**
- per bank, the word-length distribution and mGPT perplexity by band (descriptive);
- no length-standardised index, because Swahili passages are fixed 48-word windows and the references fall almost entirely in one band.

**Copying,** after `norm()` (NFC, casefold, whitespace collapse):
- exact matches with any training passage;
- the share of word 4-grams found in the training split;
- high-overlap rows: outputs with at least four 4-grams, of which at least 90% occur in training.

`real-valid` is the baseline. The supplied split has documented cross-split overlap, so the baseline is itself inflated, and the report says so.

## 8. Reporting

**Files:** `contract.json`, `panel.json`, `s1-reconstruction.jsonl`, `s2-random.jsonl`, `s3-lid.jsonl`, `s4.json`, `summary.json` and `RESULTS.md`. They go in `outputs/swahili-check-20261101/`, with this protocol and the runner in `docs/research/swahili_check_2026-11-01/`.

**Paper paragraph** (about 150 words, Results §5.5). It reports:
- the existing test perplexities, with the training-budget caveat;
- the S1 exact match, chrF and numerals, against the shuffled arm;
- the S2 result, in the fixed wording;
- the S3 Swahili share for samples, references and random latents;
- the S4 overlap relative to references.

It also states: two seeds per size; unequal update counts; no AR baseline; cross-split overlap in the supplied data; all readings automatic.

**Not claimed:** Swahili quality, fluency, or any architecture comparison. An Afrikaans–Swahili comparison is made only as a ratio to each language's own human references.

## 9. Budget

The estimate is about 1.5 GPU hours on the RTX 3060:
- S1: 400 beam decodes;
- S2: 1,000 decodes and 1,000 scorings;
- G2 and G3.

S3 and S4 run on the CPU. The run stops if any stage exceeds three times its estimate.

## Dated notes

**1 November 2026, after approval, before any model run.** G3 named the wrong reference file. `ae-test.json` holds the engine's teacher-forced figures on the **test** split (CE 0.5601, accuracy 0.8568). The validation figures, computed by the same `validate` function on the same 1,024-passage draw, are in `ae-gate.json`: CE 0.4957, accuracy 0.8738. G3 therefore compares against `ae-gate.json`, with the same 2% relative tolerance. No test data are used.
