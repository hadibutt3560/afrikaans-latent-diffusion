# Language validity, scorer sensitivity, AR decoding sensitivity and a checklist judge: protocol

Status: **APPROVED.** After approval, the banks, grids, seeds, readouts, intervals, outcome rules and gates may not change; any change is a new protocol with a new date. The committed SHA-256 of this file (line endings normalised to LF) is the protocol hash in the run contract.

Approved by: Hadi Butt (in chat, approving every decision in §13; only the file hashes in Appendix C were added after approval) Date: 1 November 2026

Before approval, every downloaded file's SHA-256 is entered in Appendix C, and the decisions listed in §13 are confirmed or changed.

## 1. Question and scope

Three questions, plus a gated pilot:

- **A. Language validity.** Do the generated outputs look like Afrikaans to an external language identifier, and how often do they contain tokens an Afrikaans spelling dictionary does not recognise, compared with held-out human text?
- **B. Shared phrasing.** How much word-level phrasing do the outputs share with the training split, compared with held-out human text?
- **C. Scorer sensitivity.** Do the existing perplexity-based comparisons point the same way when the outputs are scored by a second language model, Goldfish, instead of mGPT?
- **D. AR decoding sensitivity.** How does the AR baseline's perplexity, diversity and length change across a fixed grid of sampling settings? Does the diffusion comparison still meet the fixed 1.20 criterion against the best eligible AR setting, when that setting is evaluated on fresh outputs not used to choose it?
- **E. MAUVE pilot (gated).** Can MAUVE be estimated stably enough with the available human text to be worth reporting? Only if it passes its gate is MAUVE reported for the generated banks.
- **F. Checklist judge (gated).** Can a local open-weight language model apply the proposal's sample-inspection checklist reliably enough, shown on deliberately corrupted human text, to be used? Only if it passes validation are its checklist flags reported for the generated banks.

This is evaluation and inference only. There is no autoencoder or diffusion training, no new diffusion sampling, no test-split access and no change to any earlier benchmark, threshold or artefact. The only new text generation is from the existing AR checkpoint in Part D. The Part F judge produces checklist ratings only.

What it **can** establish:
- how often each bank's outputs are identified as Afrikaans or Dutch by GlotLID, and how its rate of dictionary-unrecognised tokens compares with human text;
- how each bank's word 4-gram overlap with the training split compares with human text;
- whether three named comparisons have the same estimated direction under Goldfish as under mGPT, and whether each scorer's interval excludes zero;
- the shape of the AR baseline's perplexity–diversity trade-off over the grid, and how the diffusion runs sit against it;
- whether MAUVE is stable enough here to report;
- whether a model judge detects planted faults of the checklist's kinds, and, if so, how often it flags those faults in each bank.

What it **cannot** establish:
- grammaticality, coherence, factual accuracy or overall quality. All measures are automatic proxies;
- that an unrecognised token is an invented word. Afrikaans compounds freely, and dictionaries miss names, compounds and newer spellings. Such tokens are called **unrecognised by the dictionary**, never "non-words" or "invented words";
- memorisation. Shared n-grams also arise from formulaic government text. Overlap is called **shared phrasing with the training split**;
- that Goldfish is independent of the training data. Its training corpora are mainly web crawls (MADLAD-400, Glot500, NLLB, Wikipedia, OSCAR) that may include South African government pages, the source of NCHLT. Goldfish is called an **alternative scorer**, not an independent one;
- AR training-seed variability. There is still one AR checkpoint; Part D varies sampling only;
- that the checklist judge's flags match what a fluent human reader would flag. The judge is validated only against planted faults;
- that the comparison is between generation mechanisms. The AR model is small and trained from scratch; the diffusion system uses a pretrained decoder and beam search.

## 2. Fixed inputs

| Input | Identity |
|---|---|
| Training split | `data/nchlt_afrikaans/processed/stage_5_7/train.jsonl`, SHA-256 `2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab` (overlap and copy checks only) |
| Validation split | `data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl`, SHA-256 `bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e` |
| Reserve | The 64 validation documents at positions 64–127 of the seed-91427 shuffle of sorted validation document IDs, as asserted in `docs/research/latent_controls_2026-10-30/run_controls.py`. Never read. |
| Phase A outputs | `outputs/latent-controls-20261030/`: `decodes-{C1,C2,C3,R}.jsonl`, `scored-{C1,C2,C3,R}.jsonl`, `manifest.json` (SHA-256 in Appendix A) |
| Diffusion banks | The ten `outputs/longrun-local-evaluation/runs/{small,published}-seed{42..46}/validation-mgpt.json` files, hashes as in Appendix A of the Phase A protocol |
| Human reference (mGPT-scored) | `outputs/longrun-local-evaluation/validation-reference-mgpt.json`, SHA-256 `51cca0b60c0d8ece2a7b6e0f39b98177cf60726d36e3defe9f97d9dbd8d52498` |
| AR checkpoint | `outputs/cluster-results/ar-baseline-57759/best_model/`, `model.safetensors` SHA-256 `ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076`, `tokenizer.json` SHA-256 `b5d0e18b86936c25d0717f6717f0c87a0576aaa261e4dc1818ac723ca2da021a`. Loaded strictly with `load_ar()` from `docs/research/ar_validation_audit_2026-10-22/run_audit.py`, FP32. |
| Historical AR samples | `outputs/ar-validation-audit-20261022/ar-cap127-samples.jsonl`, SHA-256 `1fb2a4f732d1beb6c5bd11c3a7ab1ffda680d8c4a13067e08508a904d59f1229`; scored file `ar-cap127-scored.jsonl`, SHA-256 `e0232e7d575288d0e14edb9ae7817376dc49e8056b5bb369cc97b55b4d80667d` |
| mT5 tokeniser | From the execution source `outputs/longrun-local-inputs`, every file checked against `inputs.json` as in Phase A (used only to apply the 128-token rule) |
| mGPT | `ai-forever/mGPT`, revision `40897bd7c8b47a76802c411108ca6220438b8b40`, weights SHA-256 `c45c522411e64dbed6da1ed0ce0839bf03931d3a17b80182d06d4dab8d8b2fbb`, FP32, `score_text` from the execution source's `tools/compare_afrikaans_generators.py`, unchanged |
| Goldfish | `goldfish-models/afr_latn_1000mb` (GPT-2, 124.8M parameters, maximum 512 tokens, Apache-2.0). Revision `27555fdf876be2d273e82379d1e7dc42f8eed104`; file hashes in Appendix C. FP32, scored by the same `score_text`, unchanged, with context 512. The runner asserts that the tokeniser's BOS token is `[CLS]`, as the model card requires. |
| GlotLID | `cis-lmu/glotlid`, file `model.bin` (v3, unchanged since April 2024), revision `baa20d139151a26a71a20eddabf30d4de7329fc7`; SHA-256 in Appendix C. Loaded with a fastText-compatible loader; package and version in Appendix C. |
| Afrikaans dictionary | Hunspell `af_ZA.dic` and `af_ZA.aff` from the LibreOffice dictionaries repository, commit `892f0f402cf8347aa3273f774365c094c310cf1b`; SHA-256 of each file in Appendix C. Checked with `spylls` (version in Appendix C), which applies Hunspell's affix and compounding rules. |
| MAUVE encoder (Part E) | `sentence-transformers/LaBSE`, revision `836121a0533e5664b21c7aacc5d22951f2b8b25b`; file hashes in Appendix C; normalised sentence embeddings |
| MAUVE | `mauve-text`, version in Appendix C |
| Judge candidates (Part F) | **Gemma 4 12B instruction-tuned** (`google/gemma-4-12B-it`, Apache-2.0), as the 4-bit GGUF `gemma-4-12B-it-Q4_0.gguf` from `ggml-org/gemma-4-12b-it-GGUF`, revision `202658970530b31541cd0b645a9c198bed2a615e`. **Qwen3.5 9B** (`Qwen/Qwen3.5-9B`, Apache-2.0), as the 4-bit GGUF `Qwen3.5-9B-Q4_K_M.gguf` from `unsloth/Qwen3.5-9B-GGUF`, revision `3885219b6810b007914f3a7950a8d1b469d598a5`. Qwen publishes no official GGUF of this model, so this is a third-party quantisation; it is identified by its hash. Both run with llama.cpp release `b11630` (Windows CUDA 12.4 build, 15 October 2026) through `llama-server` on localhost. File hashes in Appendix C. |
| Splice sources (Part F) | `nl_alpino-ud-test.conllu` and `en_ewt-ud-test.conllu` from Universal Dependencies release tag `r2.18`; SHA-256 in Appendix C. Sentences are taken from their `# text =` lines. |
| Test split | Not read |
| Hardware | Local RTX 3060 12 GB, batch one |
| Software environment | A **new, separate Python environment** at `E:/afrikaans-eval-20261031/env`, with downloaded resources cached under `E:/afrikaans-eval-20261031/` (drive C has too little free space). Phase A's environment is not modified. The new environment pins the versions recorded in Phase A's `contract.json`: Python 3.13.14, PyTorch 2.11.0+cu128, Transformers 5.5.0, SacreBLEU 2.6.0 and NumPy 2.4.4. The historical AR audit used the same PyTorch and Transformers versions. SentencePiece and tokenizers are pinned to the versions installed in Phase A's environment, read before the new one is built. The new packages are then added. Every package version is recorded in the new contract. Gate G2 (a) and (b) is the evidence that the new environment reproduces the earlier AR outputs and mGPT scores. |

The NCHLT frequency lexicon used during preprocessing (`afrikaans_lexicon_coverage`) is **not** used as the dictionary. It was derived from the same corpus, so it would favour text copied from training.

Outputs go to a new directory, `outputs/evaluation-package-<run date>/`, created with `exist_ok=False`. No existing file is overwritten.

## 3. Banks

| Bank | Content | n |
|---|---|---|
| **H** | Phase A's human reference: non-reserve rows of the mGPT-scored reference file | 811 (58 documents) |
| **HL (large human pool)** | Every non-reserve validation passage of at most 128 untruncated mT5 tokens, EOS included. Before the 128-token rule this is 4,827 passages from 187 documents; the final count is recorded in the manifest. Used as the baseline for Parts A and B and as the source of human samples for Part E. | ≤ 4,827 |
| **R** | Phase A's B0 reconstructions of H | 811 |
| **C1, C2, C3** | Phase A's control decodes | 1,000 each |
| **Diffusion** | The ten validation banks, five small and five large | 10 × 1,000 |
| **AR-hist** | The historical AR audit's primary arm (top-p 0.95, temperature 1, 127 new tokens). Band counts 90, 84, 54, 28, so it meets band support. | 256 |
| **AR grid cells** | New, Part D, seeds 30000–30999. Used to select the tuned setting and to draw the frontier. | 18 × 1,000 |
| **AR fresh bank** | New, Part D: the tuned setting with seeds 31000–31999. The only AR bank used for the tuned-AR contrast. | 1,000 |

HL is the baseline for language and overlap measures because H is small and concentrated. HL is still document-clustered (its largest document has 947 passages), so its uncertainty is estimated by resampling documents.

## 4. Part A: language validity

**Language identification.** Each output's text, with line breaks replaced by single spaces and no other change, goes to GlotLID. Recorded per row: the top label and its probability, and the probabilities of `afr_Latn` and `nld_Latn` from the full label distribution. Empty outputs are counted and not classified.

Per bank: the share of rows whose top label is `afr_Latn`, `nld_Latn` or something else; the mean `afr_Latn` probability; and the same three shares within each length band (§7). Identification is less reliable for short texts, so band-level shares are always shown alongside the pooled share.

**Dictionary recognition.**
- **Tokenising.** Text is NFC-normalised. Tokens are matched left to right by `(?:['’][nN])(?![^\W\d_])|ŉ|[^\W\d_]+(?:['’\-][^\W\d_]+)*`. The first alternative catches the article written with a straight or a curly apostrophe, so `’n` is never split into a stray `n`. The article forms `'n`, `’n` and `ŉ` count as recognised. Inside a word, `’` is replaced by `'` before lookup, and the number of replacements is recorded.
- **Acronyms.** Tokens of two or more letters that are entirely upper case are counted as acronyms and left out of the main rate. Per bank, the number and fraction of tokens excluded this way are reported. As a sensitivity check, the token rate is also reported **with acronyms included**, so malformed upper-case output cannot leave the denominator unnoticed.
- **Lookup.** Each remaining token is checked as written with `spylls` against `af_ZA`. Hunspell's own case rules apply, so a capitalised sentence-initial word is accepted if its lower-case form is in the dictionary.
- **Types.** Types are casefolded after NFC. A type counts as unrecognised only if every one of its occurrences in the bank is unrecognised.
- **Per bank:**
  - the **token rate** (unrecognised tokens / checked tokens);
  - the **type rate** (unrecognised types / checked types);
  - the number of checked tokens and types;
  - the acronym counts above;
  - the 20 most frequent unrecognised types.
- The **excess over human text** is a bank's token rate minus HL's token rate.

**Blinded manual check.**
- **Sample.** 150 unrecognised **token occurrences** are drawn without replacement, with seed 20261101:
  - 50 from HL;
  - 50 pooled from the ten diffusion banks;
  - 50 from AR-hist.

  Within each group, at most two occurrences of the same casefolded type are drawn, so one frequent error cannot fill the sample. If a group has fewer eligible occurrences, all are used and the count is recorded.
- Each occurrence is shown in its containing output, in random order, with its source hidden.
- Each token is labelled as one of: valid Afrikaans (including compounds, names and abbreviations); misspelt or malformed; not Afrikaans; or cannot tell.
- The key is stored separately and opened only after every label is recorded.
- **Reader for this run:** an AI-assisted reading (§13, item 5). It is reported as such and never as human evaluation. The human manual check is reported as **not conducted**. A competent human reader can label the same sheet later; that reading would be reported separately, beside the AI reading.

## 5. Part B: shared phrasing with the training split

After `norm()` from `docs/research/ae_screen_2026-10-27/compare_generation.py` (NFC, casefold, whitespace), each output's word 4-grams are compared with the set of all word 4-grams in the training split.

Per bank:
- the pooled share of 4-grams that occur in training;
- the distribution of each row's share (quartiles);
- the number of **high-overlap rows**: rows with at least four 4-grams, of which at least 90% occur in training;
- rows with fewer than four words, which contribute no 4-grams (counted);
- exact training copies, as in Phase A.

Word 8-gram overlap is taken from Phase A for the banks it covers and computed in the same way for HL and the AR grid cells. HL is the baseline. A bank's overlap is described relative to HL's ("higher than held-out human text"), never as memorisation.

## 6. Part C: Goldfish as an alternative scorer

Goldfish scores every row of H, R, C1–C3, the ten diffusion banks, AR-hist, all AR grid cells and the AR fresh bank with `score_text`, unchanged. Per bank: raw perplexity and length-standardised perplexity under §7's definitions.

**Absolute perplexities are never compared across the two scorers.** They use different tokenisers. Only within-scorer comparisons are set side by side.

**Named comparisons.** Each is computed separately under mGPT and under Goldfish:

| ID | Comparison | Estimand and interval |
|---|---|---|
| C-1 | Large against small diffusion | Difference in mean log standardised perplexity, large − small, over the five runs of each size. Welch interval at 95% with Welch–Satterthwaite degrees of freedom. |
| C-2s, C-2l | Each size against AR-hist | Mean over the five runs of log standardised perplexity, minus log standardised perplexity of AR-hist. Interval ± t₄(0.9875) · √(s²_runs / 5 + V_AR), where V_AR is the bootstrap variance of AR-hist's log standardised perplexity (within-band row bootstrap, 10,000 replicates, seed 20261102). |
| C-3s, C-3l | Each size against C3 | Phase A's primary estimand and interval, unchanged (difference of standardised perplexity, t₄(0.9875), C3 bootstrap seed 20261030). Under mGPT the values are taken from Phase A's `analysis.json`; under Goldfish they are computed in the same way. |

**Reporting agreement.** For each comparison, two things are reported separately, never combined into one label:
1. **Direction:** whether the two scorers' point estimates have the same sign ("same direction" or "opposite directions").
2. **Interval status for each scorer:** whether its interval lies entirely below 0, entirely above 0, or contains 0.

A comparison is described as **confirmed under both scorers** only if the point estimates share a sign and both intervals exclude 0 on that side. Every other pattern is described by stating the direction and each scorer's interval status in turn, for example: "same direction; the mGPT interval excludes 0, the Goldfish interval contains 0". The words "agrees" and "partly agrees" are not used.

**Descriptive additions.**
- The Spearman rank correlation between the two scorers' standardised perplexities over all banks that meet band support.
- **Copies removed.** Rows that exactly match a training passage are removed from each diffusion bank and AR-hist, and raw and standardised perplexity are recomputed under both scorers. The number of rows removed is reported per bank.

## 7. Shared definitions

These are the Phase A definitions, unchanged:
- **Length bands:** whitespace words 1–10, 11–20, 21–35 and 36 or more.
- **Reference weights:** 0.140, 0.223, 0.349, 0.288.
- **Standardised perplexity:** exp(Σ_b w_b · NLL_b).
- **Band support:** at least 20 scored rows in every band, otherwise not estimable or insufficient support. Raw perplexity and band counts are always reported.
- **Distinct-2, unigram entropy and exact copies:** as in Phase A §5.

Diffusion values are summarised as mean and sample SD over the five runs of each size. For HL, 95% intervals for Part A and B rates come from 2,000 document-bootstrap replicates (seed 20261103). Parts A and B have no decision rules; they are reported descriptively with the wording limits in §1.

## 8. Part D: AR sampling grid

**Grid (18 cells).**
- Temperature: {0.8, 0.9, 1.0}.
- Top-p: {0.90, 0.95, 1.00}.
- Constraints: **off** or **on**.

Grid order, used for file names and tie-breaking: temperature ascending, then top-p ascending, then constraints off before on.

Off is the historical setting: no minimum length, repetition penalty 1.0, no n-gram ban. On copies the diffusion decoder's constraints: `min_length` 5 (counting the start token, as in the diffusion decoder), `repetition_penalty` 1.2, `no_repeat_ngram_size` 3.

Everything else is as in the historical audit's `generate()`:
- top-k 0, one beam, maximum 127 new tokens, FP32, batch one, cache on;
- BOS 0 and EOS 1;
- for each seed s in 30000–30999, `torch.manual_seed(s)` and `random.seed(s)` before generation.

The seeds are numerically equal to the diffusion sampling seeds but drive a different model, so outputs are never paired with diffusion outputs.

Each cell's 1,000 outputs are scored by mGPT and Goldfish and go through Parts A and B.

**Per cell:**
- raw and standardised perplexity under both scorers;
- mean words and band counts;
- distinct-2, unigram entropy and exact copies;
- 4-gram and 8-gram overlap;
- EOS, length-cap, sentinel and empty counts;
- GlotLID `afr_Latn` share and dictionary token rate.

**Frontier figure.** The horizontal axis is unigram entropy and the vertical axis is standardised mGPT perplexity. It shows all 18 cells (constraints off and on marked differently), the AR fresh bank (marked separately from its grid cell), the ten diffusion runs and C1–C3. H appears only as a labelled reference, because its entropy comes from 811 rows. A second panel uses distinct-2 on the horizontal axis. Every cell is shown; none is dropped.

**Selecting the tuned AR setting (on the grid outputs).**
- A cell is **eligible** if it meets band support and, compared with the ten diffusion runs, has:
  1. distinct-2 at least the lowest diffusion value;
  2. unigram entropy at least the lowest diffusion value;
  3. no more exact training copies than the highest diffusion count.
- This rule means the cell **clears the least diverse diffusion run**. It does not mean the cell matches diffusion's diversity. A permissive rule lets AR reach lower perplexity, which makes the diffusion comparison harder to pass, not easier.
- The eligible cell with the lowest standardised mGPT perplexity is the **tuned AR setting**. Ties are broken by the earlier cell in the grid order above.
- If no cell is eligible, this is reported and no tuned-AR comparison is made.

**Evaluation bank (separate from selection).** The grid outputs are used only to choose the setting; they never enter the contrast. After the choice, the tuned setting generates a **fresh bank** of 1,000 outputs with seeds 31000–31999, which do not overlap the grid seeds, using the same procedure. The fresh bank goes through mGPT, Goldfish and Parts A and B. Its own distinct-2, entropy, copy count and band counts are reported beside its perplexity. If the fresh bank no longer meets the eligibility conditions, that is reported with the result. The setting is not changed.

**Primary contrast (each size), on the fresh bank.** Log ratio = mean over the five runs of log standardised mGPT perplexity, minus log standardised mGPT perplexity of the fresh bank. Interval ± t₄(0.9875) · √(s²_runs / 5 + V_fresh), where V_fresh is the bootstrap variance of the fresh bank's log standardised perplexity (within-band row bootstrap, 10,000 replicates, seed 20261104). Ratios are reported as exp(log ratio) with the exponentiated interval. If the fresh bank fails band support, the contrast is classified as insufficient support.

The diffusion banks enter both the eligibility floors and the contrast. They are fixed in advance and not selected, so this does not make the selection depend on the outputs being compared.

For context only, the same ratio computed on the selected grid cell's own outputs is shown in the grid table, labelled **selection sample, no interval**. It has no classification.

**Outcome rules (each size), on the fresh bank, against the 1.20 criterion fixed before the original experiments:**

| Interval for the ratio | Classification | Permitted wording |
|---|---|---|
| Upper bound ≤ 1.20 | **Meets** | "[Size] diffusion met the fixed 1.20 perplexity criterion against the tuned AR setting." |
| Lower bound > 1.20 | **Does not meet** | "[Size] diffusion did not meet the fixed 1.20 perplexity criterion against the tuned AR setting." |
| Contains 1.20 | **Not established** | "The comparison did not establish whether [size] diffusion meets the 1.20 criterion against the tuned AR setting." |

The interval's position relative to 1.0 is reported descriptively, as "lower than", "higher than" or "not different under this procedure", without a classification. No result is called "diffusion beats AR" or a quality result. The historical comparison (ratios 0.6458 and 0.6087) is left unchanged and reported beside the new one.

## 9. Part E: MAUVE pilot (gated)

MAUVE (`compute_mauve` with precomputed features, package defaults for buckets, PCA and scaling) is computed on LaBSE embeddings. Human samples are drawn from HL, after the 128-token rule. Every MAUVE value compares **1,000 passages with 1,000 passages**. The larger pool improves how well the human side covers the validation documents. It does not make any single MAUVE value a comparison of more than 1,000 samples. Embeddings are computed once per passage and reused.

Two sources of variation are reported, under separate names:
- **Clustering variability:** the spread over k-means seeds with both samples fixed.
- **Reference-draw variability:** the spread over different human reference samples with the k-means seed fixed.

Neither is described as the overall uncertainty of MAUVE.

**Human reference draws.** Five reference samples, E-ref₀ to E-ref₄, are drawn in a way that respects documents. For k = 0…4 (seed 20261120 + k):
1. Resample HL's documents with replacement, keeping all passages of each chosen document.
2. Draw 1,000 passage instances uniformly without replacement from the resulting pool.

A passage from a document chosen twice can appear twice; this is recorded.

**E1. Human calibration.**
- For r = 0…9, split HL's documents into two halves at random (seed 20261110 + r).
- Draw the same number of passages, n_r = min(1,000, size of the smaller half), uniformly without replacement from each half. Record n_r.
- Compute MAUVE between the halves with k-means seed 1.

This gives ten human-against-human values.

**E2. Stability and discrimination.**
- MAUVE(small-seed42, E-ref₀) with k-means seeds 1–5 (clustering variability).
- MAUVE(small-seed42, E-ref_k) for k = 0…4 with k-means seed 1 (reference-draw variability).
- MAUVE(C1, E-ref₀) with k-means seed 1.

**Gate G-M.** Continue only if all five hold:
1. the median of the ten human-against-human values is at least 0.90;
2. their range (maximum − minimum) is at most 0.10;
3. the SD of the five clustering-variability values for small-seed42 is at most 0.03;
4. the range of the five reference-draw values for small-seed42 is at most 0.10;
5. MAUVE(C1, E-ref₀) is below the smallest human-against-human value.

These thresholds are operational choices fixed before any MAUVE value is seen. They are not a validated guarantee that MAUVE is reliable here, and passing them is described only as "passed the pilot's calibration gate".

If the gate fails, MAUVE is reported as **not reported: calibration gate failed**, with the E1 and E2 values in the appendix.

**E3 (only if G-M passes).** For each of these banks:
- the ten diffusion banks;
- C1, C2 and C3;
- the tuned AR fresh bank (§8);
- the historical-setting grid cell (temperature 1.0, top-p 0.95, constraints off);

compute MAUVE against E-ref₀ … E-ref₄ with k-means seed 1, reporting the mean and range as reference-draw variability. Also compute MAUVE against E-ref₀ with k-means seeds 1–5, reporting the range as clustering variability. Results are shown beside the human-against-human distribution. AR-hist (256 rows), H and R are not scored by MAUVE. MAUVE results are descriptive and have no decision rule.

## 10. Part F: checklist judge (validated before use)

**Purpose.** The research proposal promised researcher-side inspection of generated samples against a fixed error checklist. Part F applies that checklist automatically with a local open-weight language model, the **judge**. It is used only if it first passes a validation on deliberately corrupted human text.

It is an automatic diagnostic. It is not human evaluation and not a measure of quality. Because it replaces researcher-side inspection for the generated samples, it is listed as a deviation from the proposal.

**Candidates.** Two instruction-tuned open-weight models of roughly 8–12B parameters with stated broad multilingual coverage: Gemma 4 12B and Qwen3.5 9B (§2), each quantised to 4 bits and run locally on the RTX 3060 with llama.cpp.

The exact files and hashes are in §2 and Appendix C. A candidate that cannot run within 10 GiB peak memory is dropped at pinning, before any validation, and the drop is recorded. If both are dropped, Part F is reported as not conducted.

Any reasoning or "thinking" mode is switched off through the model's chat-template switch and llama.cpp's reasoning budget of 0. Decoding is greedy (temperature 0), with at most 64 new tokens, a context of 2,048 tokens and all layers on the GPU.

**Checklist items.** Every item is answered yes or no. The items come from the proposal's sample checklist and its reconstruction checks:

| Item | Question to the judge |
|---|---|
| J1 Unreadable | Is the text unreadable or incoherent as a whole? |
| J2 Repetition | Does it contain a fragment repeated in a way a writer would not intend? |
| J3 Morphology | Does it contain malformed Afrikaans words: wrong endings, broken compounds, or letter sequences that are not words? |
| J4 Sentence structure | Is the word order or clause structure broken? |
| J5 Truncation | Does it stop abruptly in the middle of a sentence or phrase? |
| J6 Not Afrikaans | Is any substantial part of it in Dutch, English or another language rather than Afrikaans? |
| J7 Domain | Is it clearly out of place in South African government text? |

The proposal's "excessive copying from the training corpus" is not asked of the judge, because the judge cannot see the training corpus. Part B measures it.

The **fault flag** for a text is J1–J6 answered yes for any item. J7 is reported but not part of the fault flag, because being off-domain is not an error.

**Prompt.**
- A single fixed English instruction states the domain (Afrikaans South African government text), lists J1–J7 with the wording above, and asks for one line in the form `J1=0 J2=1 … J7=0`.
- Each text is rated on its own. There are no pairwise comparisons, and nothing about the text's source is included.
- The exact prompt is fixed in Appendix B of this protocol and hashed in the contract. It is identical for every candidate apart from the model's own chat template.
- An answer that does not parse into seven 0/1 values counts as **unparsed** and is never retried.

**Validation suite.** No human labels are needed; the corruptions are the labels.

*Source passages.*
- HL's documents are split at random (seed 20261130) into a **selection half** and a **confirmation half**.
- From each half, 150 passages of at least 11 words are drawn uniformly without replacement (seed 20261131).
- Validation passages are excluded from every human sample used elsewhere in Part F.

*Versions.* Each source passage gives six versions. Each corruption is applied with a per-passage seed derived from seed 20261132:

| Version | Construction | Target item |
|---|---|---|
| Clean | The passage unchanged | None |
| Shuffle | All words randomly permuted | J4 |
| Morphology | In 30% of words of six or more letters (rounded up, at least two words), one pair of adjacent interior letters that differ from each other is swapped, so every chosen word changes. Passages with fewer than two such words are replaced by the next drawn passage, and the replacement is recorded. | J3 |
| Repeat | A span of four to six consecutive words is inserted twice more immediately after itself | J2 |
| Truncate | Cut after a word at a random point between 40% and 70% of the words, with final punctuation removed | J5 |
| Language splice | The middle third of the words is replaced by an equal number of words from one sentence drawn at random from a pinned Dutch or English sentence source (alternating by passage) | J6 |

*Splice sources.* Dutch and English sentences come from the Universal Dependencies treebanks `UD_Dutch-Alpino` and `UD_English-EWT` (test portions). Release and hashes are in §2 and Appendix C. No Afrikaans text is used as a splice source.

**Validation measures, per candidate and per half:**
- **Detection rate** per corruption type: the share of corrupted versions given the fault flag.
- **Target-item rate** per corruption type: the share given the intended item.
- **Clean flag rate:** the share of clean versions given the fault flag.
- **Unparsed rate** over all ratings.

**Gate G-J (applied per candidate per half).** A candidate passes a half only if all four hold:
1. the detection rate is at least 0.80 for every one of the five corruption types;
2. the clean flag rate is at most 0.20;
3. the unparsed rate is at most 0.02;
4. on the C1 control's first 150 rows, the fault-flag rate exceeds the clean flag rate, with the 95% Newcombe interval for the difference excluding 0.

The 0.20 clean threshold allows for the fact that NCHLT passages are segments of longer documents, so some genuine passages look truncated or list-like.

**Selection, then confirmation.**
1. Both candidates are rated on the selection half only.
2. Among candidates that pass G-J on the selection half, the one with the highest **mean detection rate minus clean flag rate** is chosen. Ties go to the smaller model.
3. The chosen judge alone is rated on the confirmation half.

Part F's results are reported only if the chosen judge also passes G-J on the confirmation half. Otherwise Part F is reported as **not reported: the judge failed validation**, with all validation figures in the appendix. If no candidate passes the selection half, Part F stops there and is reported the same way.

The validation suite shows that the judge detects faults of the kinds planted at the strength planted. It does not show that it detects subtler errors a fluent reader would notice. This is stated with every Part F result.

**Application (only after confirmation).** For generated banks, the judge rates the first 300 rows in stored order. Their sampling seeds are independent, so this is not a selected subset. H is stored in document order, so its 300 rows are drawn uniformly without replacement (seed 20261135), and R uses the same 300 passages. The banks are:
- H and R;
- C1, C2 and C3;
- the ten diffusion banks;
- AR-hist (256 rows, all used);
- the AR fresh bank;
- the historical-setting grid cell;
- a fixed sample of 300 HL passages (seed 20261133), excluding validation passages.

**Readouts per bank:**
- the rate of each item J1–J7;
- the fault-flag rate;
- the unparsed count.

**Uncertainty:**
- Diffusion: mean and sample SD over the five runs of each size.
- Single generated banks: 95% Wilson intervals.
- H and the HL sample: 95% document-bootstrap intervals (2,000 replicates, seed 20261134).

**Descriptive cross-check.** The share of J3-flagged rows that contain at least one dictionary-unrecognised token (Part A), and the share of rows with such a token that are J3-flagged.

**Wording.** Results are reported as "flagged by the validated checklist judge", always beside the judge's confirmation-half detection and clean flag rates. Part F has no decision rules. It is never described as human evaluation, fluency or quality, and no bank is said to be "better" because of it.

**Budget.** The judge has its own pilot within G3: 20 ratings per candidate are timed and the whole of Part F is projected. If the projection is above 8 hours, Part F is stopped and reported as not conducted for budget reasons. Sample sizes are not reduced without a new protocol.

Planned volume:
- 1,050 selection ratings per candidate (900 validation versions plus 150 C1 rows);
- 1,050 confirmation ratings;
- 5,656 application ratings.

## 11. Gates

| Gate | Requirement | If failed |
|---|---|---|
| G0. Pinning | Every resource in §2 has its SHA-256 in Appendix C before approval | Not approved |
| G1. Integrity | All hashes match; strict loads; no non-finite scores; every bank has its stated row count | Stop. Report. |
| G2. Reproduction | (a) The grid cell (temperature 1.0, top-p 0.95, constraints off) reproduces the `token_ids` of all 256 rows of `ar-cap127-samples.jsonl` for seeds 30000–30255 exactly. (b) mGPT re-scores the first 20 rows of `ar-cap127-scored.jsonl` with identical token counts and NLL within relative 1e-4. (c) Goldfish scores the first 20 H rows twice with identical results. (d) Goldfish's tokeniser reproduces at least 99% of the first 100 H texts exactly after encoding and decoding, compared after whitespace normalisation. | Pause. Nothing is interpreted. Investigate and report; resuming needs a dated note appended to this protocol. |
| G3. Budget pilot | Runs **before any part's main work**. It times: generating the first 50 outputs of the first grid cell (in grid order) and of the last grid cell; mGPT and Goldfish scoring of those 100 outputs; Goldfish scoring of the first 100 rows of small-seed42; and LaBSE embedding of 100 HL passages. From these timings it projects the GPU time for all remaining GPU work in Parts C, D and E. Requirements: projected total ≤ 6 hours and peak allocated memory ≤ 10 GiB. The pilot's outputs are kept and reused, because they are the same rows the main run would produce. | Stop before the main run. Report the projection. The grid and sample sizes are not reduced without a new protocol. |
| G4. Band support | §7's rule, per bank | As in §7 |
| G-M | §9 | As in §9 |
| G-J | §10, on the selection half and then the confirmation half | As in §10 |

Rough estimate, to be replaced by the G3 projection: AR generation took about 0.28 s per output in the historical audit, and mGPT about 0.08 s per row in Phase A. That gives about 1.5 hours of generation and 0.4 hours of mGPT scoring for the grid and the fresh bank, plus Goldfish scoring and LaBSE embedding.

**Execution order:**
1. G0, G1 and G2.
2. G3 budget pilot.
3. Parts A and B on the existing banks and HL (CPU).
4. Part C on the existing banks.
5. Part D: grid generation and scoring, then Parts A–C on the grid cells, then selection of the tuned setting, then the fresh bank and Parts A–C on it.
6. Part E.
7. Part F: selection half, confirmation half, then application.

Each part writes its outputs before the next starts. A failure later in the order leaves the earlier parts' results valid.

## 12. Implementation and reporting

- The runner and the analysis script are written **after** approval. They reuse `score_text`, `norm()`, `load_ar()`, the reserve assertion and the band definitions unchanged.
- `--dry-run` builds HL, the reserve check, the grid list, the seeds and the manual-check sample frame, then prints the counts without loading any model.
- Before any model loads, `contract.json` records this protocol's SHA-256, the runner's and the analysis script's SHA-256, all input hashes and the package versions.
- Files:
  - `manifest.json`, `contract.json`;
  - `lid-<bank>.jsonl`, `dict-<bank>.jsonl`, `overlap-<bank>.jsonl`;
  - `goldfish-<bank>.jsonl`;
  - `budget-pilot.json`;
  - `ar-grid-<cell>.jsonl`, `ar-grid-<cell>-mgpt.jsonl`;
  - `ar-tuned-selection.json` (eligibility table and the chosen cell, written before the fresh bank is generated);
  - `ar-fresh.jsonl`, `ar-fresh-mgpt.jsonl`;
  - `mauve.json`;
  - `manual-check-sheet.csv` (no source column), `manual-check-key.csv`;
  - `judge-validation-<candidate>-<half>.jsonl` (every version, its construction record and the raw judge answer), `judge-selection.json` (written before the confirmation half is rated), `judge-<bank>.jsonl`;
  - `analysis.json`, `frontier.pdf`, `ALL_OUTPUTS.md`, `runtime.json`.
- Examples quoted anywhere are rows 0, 250, 500 and 750 of each grid cell and of the fresh bank, labelled AI-assisted reading, not human evaluation.

**Later application to the new autoencoder's banks.** When job 77176 delivers validation banks generated through autoencoder D, Parts A, B and C may be applied to them with every definition above unchanged. Results go in a separate output directory and are reported separately. No new decision rule is added for them. That application needs only a dated note naming the new bank files and their hashes.

**Paper additions, after the run only:**
- Table 3: the language-identification cell (E4) and the copies-removed perplexity.
- Results (RQ3): one paragraph and one table for Parts A–C; the frontier figure and the tuned-AR contrast with its classification, using only §8's wording.
- Appendix: dictionary and language-ID details by band, the manual check (or "not conducted"), Goldfish details and the overlap caveat, the full grid table, MAUVE or the failed gate, and the judge's validation table with its application results or the failed gate.
- Deviations table: a row stating that a validated model judge applied the proposal's sample checklist in place of researcher-side inspection.
- `EVIDENCE_REGISTER.md` and `verify_revision.py`: new rows with paths and hashes.

## 13. Decisions to confirm before approval

1. **Grid.** 3 temperatures × 3 top-p values × constraints off/on, for an exploratory frontier. A wider grid costs about 5 minutes of generation per 1,000 outputs.
2. **Eligibility for the tuned AR setting.** Floors set by the least diverse diffusion run, and a ceiling set by the diffusion run with the most copies. This is permissive by design (§8). A stricter alternative uses the mean over runs.
3. **MAUVE gate thresholds.** Human median ≥ 0.90, human range ≤ 0.10, clustering SD ≤ 0.03, reference-draw range ≤ 0.10. These are operational choices, fixed before any MAUVE value is seen, not validated guarantees.
4. **MAUVE encoder.** LaBSE alone, following the advice to start with one encoder. XLM-R could be added only by a new protocol.
5. **Manual check reader.** The labels are produced by the AI assistant (Claude) that drafted this protocol, as an **AI-assisted reading**. No human reader is named, so the human manual check is reported as **not conducted**. The AI reading is reported under its own heading and never described as human evaluation. The assistant labels from `manual-check-sheet.csv` only. The key is joined by the analysis script after every label is saved, and the labels file's SHA-256 is recorded before the join. The assistant knows the study's aims, and the report says so.
6. **New packages.** A fastText loader, `spylls`, `mauve-text`, `sentence-transformers` and a 4-bit loader for the judge (`bitsandbytes` or `llama-cpp-python`), installed into a new environment, not Phase A's, with versions recorded.
7. **Judge gate thresholds.** Detection ≥ 0.80 per corruption type, clean flag rate ≤ 0.20, unparsed ≤ 0.02, and C1 flagged more than clean text. These are operational choices fixed before any judge rating is seen.
8. **Corruption strengths.** 30% of long words altered, spans of four to six words repeated twice, truncation at 40–70%, the middle third spliced. Milder corruptions would make the test harder and the gate less likely to pass.

## Appendix A. Phase A output hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `decodes-C1.jsonl` | `6450fafbf3d08828923d2be9ee4da455fad88af7f23b8ccba74c6d0cedc16313` |
| `decodes-C2.jsonl` | `db9189183732a79011d9fe950a3b81be93a7b7a05d98b0126760eb1083a4f00f` |
| `decodes-C3.jsonl` | `786291cfef5c13eb281f368db9996890d2aa5809f21f1b183b0cc0a53a3eedc9` |
| `decodes-R.jsonl` | `2f103d415ae2acf5d875225cb5b2d78a05c530fee4672dc87e47ad544c39b5cd` |
| `scored-C1.jsonl` | `88fb70d6139b98bfaabf58a0d11269a4531e138e0db305819cd03239796533b4` |
| `scored-C2.jsonl` | `b0679a76672895e41f40ce2621e0b7df8a5696248cc3df90ee8ed4583fe3111d` |
| `scored-C3.jsonl` | `ce6cdae265aee2df8d4d5469fe9d85aae0bb17981ecf350245cd3efa4b34c292` |
| `scored-R.jsonl` | `cc4d29396a09857baf5e2dbae6ecd3a3071d17e355961f88783d9bdf780d5cfa` |
| `manifest.json` | `a171e638639e0a50cb754b014032d149fe29122243d00e476c4c7d1db2153d82` |
| `analysis.json` | `f2b706fb9ac51bc69716572b84311901573ef6563b14cb02955f59f335007404` |

## Appendix B. Checklist judge prompt (Part F)

The text between the lines is sent as the user message, with `{TEXT}` replaced by the output being rated, inside each candidate's own chat template. No system message is used.

---
You will read one passage that is meant to be written in Afrikaans, in the style of South African government text. Judge only the passage itself.

Answer each question with 1 (yes) or 0 (no):
J1: Is the text unreadable or incoherent as a whole?
J2: Does it contain a fragment repeated in a way a writer would not intend?
J3: Does it contain malformed Afrikaans words: wrong endings, broken compounds, or letter sequences that are not words?
J4: Is the word order or clause structure broken?
J5: Does it stop abruptly in the middle of a sentence or phrase?
J6: Is any substantial part of it in Dutch, English or another language rather than Afrikaans?
J7: Is it clearly out of place in South African government text?

Passage:
"""
{TEXT}
"""

Reply with exactly one line in this form and nothing else:
J1=0 J2=0 J3=0 J4=0 J5=0 J6=0 J7=0
---

Parsing: the first line of the reply matching `J1=[01] J2=[01] J3=[01] J4=[01] J5=[01] J6=[01] J7=[01]`, allowing extra spaces, is used. Anything else is unparsed.

## Appendix C. Downloaded resources (SHA-256)

Recorded from `E:/afrikaans-eval-20261031/pins.json` after download, before any run. Package versions in the new environment: `{'fasttext-turbo': '0.1.0', 'spylls': '0.1.7', 'mauve-text': '0.4.0', 'faiss-cpu': '1.15.1', 'sentence-transformers': '5.3.0', 'matplotlib': '3.11.2'}`.

| Resource | File | SHA-256 |
|---|---|---|
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `.gitattributes` | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `added_tokens.json` | `402f9f8b3242fc55fa48680ea341203a1e174f9bbaf32a2d41efb0939512ca11` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `config.json` | `b3d5b32f91e9311426e53a9600ebf322561198659fee0ff35d0174dd2257d46f` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `pytorch_model.bin` | `977ff3dc1ca47ed3b831cd9534d6afecb6d1416082d0665151cee980195f1989` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `README.md` | `297c2cd644a9185ba711e2cce59aee282f285fc9e635d4aa171db140e9daa2ad` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `special_tokens_map.json` | `4d6025468ad1902a29170ab995e005ad9ddb5e9cbf935dd0e40766a260d189c7` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `spiece.model` | `21c4b3da6315fbdd0f3d8631929efe31d74d7fd5c2694a84008bfb49318a0e02` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `tokenizer.json` | `6e025416e21598defef5b61085575c374200282dc64189582e3ccebefca3baa3` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `tokenizer_config.json` | `94e7853872729c0674041f829ec55c7157e13751e95c0ce8e76a439fbd8ab9c0` |
| cis-lmu/glotlid @ baa20d139151 | `model.bin` | `a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e` |
| sentence-transformers/LaBSE @ 836121a0533e | `1_Pooling/config.json` | `c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b` |
| sentence-transformers/LaBSE @ 836121a0533e | `2_Dense/config.json` | `02a7646de76be592655235523e09b5315c115bb770d91de87f21a9b0746723b5` |
| sentence-transformers/LaBSE @ 836121a0533e | `2_Dense/model.safetensors` | `f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77` |
| sentence-transformers/LaBSE @ 836121a0533e | `config.json` | `b6bd83d561db8d2ccb7f44dafcd945e39bea16dc2819072ffb10adb45608b029` |
| sentence-transformers/LaBSE @ 836121a0533e | `config_sentence_transformers.json` | `b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8` |
| sentence-transformers/LaBSE @ 836121a0533e | `model.safetensors` | `77d8e1f2dbab6eb5d3c261ce9d3dbf1e3c69e02938c95f934f94f42c22dfa31f` |
| sentence-transformers/LaBSE @ 836121a0533e | `modules.json` | `41251094dfd3e7ff81a85ba97b3f4788df47e4039f1bc5037575b43c028beb40` |
| sentence-transformers/LaBSE @ 836121a0533e | `sentence_bert_config.json` | `fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7` |
| sentence-transformers/LaBSE @ 836121a0533e | `special_tokens_map.json` | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |
| sentence-transformers/LaBSE @ 836121a0533e | `tokenizer.json` | `d09046654dc5151fc313613f16768b6ecc1e5e3bddbc9c8aca209a5f23fe3cd3` |
| sentence-transformers/LaBSE @ 836121a0533e | `tokenizer_config.json` | `9c5867ec63310dc2958daae8ccfabb226a2b6d317f407ade84d50315e7f9e80b` |
| sentence-transformers/LaBSE @ 836121a0533e | `vocab.txt` | `7453247caa2cf23116c1da332bec0cc1c9ef896b7803591a62ad2007b7a33c33` |
| ggml-org/gemma-4-12b-it-GGUF @ 202658970530 | `gemma-4-12B-it-Q4_0.gguf` | `3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70` |
| unsloth/Qwen3.5-9B-GGUF @ 3885219b6810 | `Qwen3.5-9B-Q4_K_M.gguf` | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| dict | `af_ZA.aff` | `fdfcdd04d700fc4ac94be63930167fa942ee5ffb471b52a6f4e0c68362dd63d2` |
| dict | `af_ZA.dic` | `8d806f90146059ea18b49c2352d188bd27dcc571a06815794facaa176aa5a5de` |
| dict | `README_af_ZA.txt` | `e00c86cb0a1c0499380770750f545b5d0308d6cb35d25014cef6891c2fe5f255` |
| ud | `nl_alpino-ud-test.conllu` | `20d0f718cdc73b51dbb2744386c9d4ecb1a9df6a076efa81b4c008ace3801fc0` |
| ud | `en_ewt-ud-test.conllu` | `fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a` |
| llama.cpp b11630 | `cudart-llama-bin-win-cuda-12.4-x64.zip` | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
| llama.cpp b11630 | `llama-b11630-bin-win-cuda-12.4-x64.zip` | `e260e0f3dbc93928205005b91a69e3ab037519ea7b0aac3c73d8faa38a5c69ae` |
