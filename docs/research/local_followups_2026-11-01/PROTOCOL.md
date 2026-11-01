# Local follow-ups to the 1 November review

Status: **descriptive and exploratory.** Every task below adds counts or repeats an existing measurement on data already used in the study. None tests a new hypothesis or changes a reported result. Seeds and rules are fixed here before the run; any change is a dated note.

Runner: `run_followups.py`. Hardware: one local RTX 3060 (12 GB). Autoencoder: the original B0 (job 56531, SHA-256 `123f4508…`). Scorer: mGPT, revision `40897bd7…`, FP32, as in the latent-control protocol. Code for loading, decoding and scoring is imported unchanged from `latent_controls_2026-10-30/run_controls.py`, whose input checks run first.

**Data discipline.** The test split and the reserve documents (positions 64–127 of the seed-91427 validation shuffle) are not read. The only new text evaluated is `extra_monitor.jsonl`, 681 passages never used for training or selection.

## CPU tasks (minutes)

| Task | Review item | What it reports |
|---|---|---|
| `numerals` | D1 | For the 81 original reconstructions and the 811 B0 reconstructions of the latent-control run: numerals expected, produced and matched (multiset), recall, precision, invented and dropped counts; a list of passages with invented numerals for human checking. Numerals use the regex `\d+(?:[.,]\d+)*`. Wrong entity–number associations need a human reader and are not counted. |
| `overlap` | D5 | Word 12-grams (after NFC, case folding and whitespace collapse) shared between training and non-reserve validation passages: passages and documents affected, longest shared span, and the share of shared 12-grams that occur in at least 5 training documents (boilerplate). |
| `lengths` | D8 | mT5 token lengths of training, non-reserve validation and `extra_monitor` passages: percentiles and the share over 128 tokens. |
| `figure1` | N2 | The per-band values behind Figure 1 as CSV, from the committed `length_bands.json`. |
| `partial_copies` | Copying | For every validation bank of the evaluation package: share of word 8-grams found in training, longest shared span, and near-copy rows (at least four 8-grams, at least 90% of them in training). |
| `scorers` | E (computational part) | For every bank with both scorers: per-band mGPT and Goldfish perplexity, the Spearman correlation of per-row loss per token between the scorers, and the share of rows Goldfish rates better than the median human row. |

## GPU tasks (about 4–5 hours)

| Task | Review item | Design |
|---|---|---|
| `extra_monitor` | Fresh evaluation | Encode and decode all 681 passages with B0, using the latent-cache encoding path and the benchmark beam settings. Exact match, chrF (character order 6, β 2) and numerals against the 128-token model view. |
| `shuffle_seeds` | Shuffle sensitivity | The 81 passages of the 22 October audit. The correct arm must first reproduce the stored generated IDs exactly (gate). Then, for seeds 20261111–20261115, each passage takes the latent of a passage from a different document in the same length bin, drawn by a seeded shuffle. Report exact match and chrF per seed. |
| `random_banks` | D3 | Four further banks of 1,000 latents: isotropic (C1 rule) with seeds 31000+i and 32000+i; fitted Gaussian (C2 rule, same Ledoit–Wolf fit) with seeds 41000+i and 42000+i. Decode, score with mGPT, and report raw and length-standardised perplexity (weights 0.140, 0.223, 0.349, 0.288; at least 20 rows per band), distinct-2 and sentinel counts, beside the original C1 and C2 banks. |

## Reporting

Outputs go to `outputs/local-followups/` under the data root, one file per task, with `status.json` and `SUMMARY.md`. A task that fails is recorded and the rest continue. Results are descriptive; any use in the paper says so.
