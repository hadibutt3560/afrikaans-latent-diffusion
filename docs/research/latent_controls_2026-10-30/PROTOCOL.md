# Diffusion latents against simpler latent priors — protocol

Status: **APPROVED.** After approval, the conditions, samples, seeds, estimands, intervals, outcome rules and gates may not change; any change is a new protocol with a new date. The committed SHA-256 of this file (line endings normalised to LF) is the protocol hash in the run contract.

Approved by: Hadi Butt Date: 30 October 2026

## 1. Question and scope

Under the fixed B0 decoder and the benchmark decoding settings, how do the outputs of the trained diffusion models compare with outputs decoded from three simpler ways of producing latents?

This is inference only. There is no training, no test-split access, no new diffusion sampling in Phase A, and no change to any historical benchmark, threshold or artefact.

What it **can** establish:
- whether each diffusion configuration's validation outputs have lower, higher or not detectably different length-standardised mGPT perplexity than each control, under the stated procedure;
- how length, diversity and training-text overlap differ between the diffusion outputs and the controls;
- (Phase B) what applying per-slot normalisation to sampled diffusion latents does to their decodes, for two checkpoints.

What it **cannot** establish:
- linguistic quality. mGPT perplexity, distinct-2 and entropy are proxies;
- what the denoiser learned. Lower perplexity than C3 is consistent with useful cross-slot structure but does not demonstrate that mechanism;
- how much of the benchmark score is "due to" the decoder or the denoiser. This is a comparison against alternative latent priors, not a decomposition;
- anything about other autoencoders, or about autoencoder training-seed variance;
- uncertainty from fitting C2's Gaussian. C2 results are conditional on the single fitted Gaussian.

## 2. Fixed inputs

| Input | Identity |
|---|---|
| Autoencoder B0 | Job 56531, update 30,500, `outputs/cluster-results/afrikaans-56531/best/model.pt`, SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`. Strict load, FP32, evaluation mode, TF32 off. |
| Execution source and tokeniser | `outputs/longrun-local-inputs`, every file hash checked against `inputs.json` (SHA-256 `d7f2aeb8f348ad360f39d5d11b2e08fc4ec64667d66f0861a75b1fd1cdba3772`), as in `docs/research/reconstruction_audit_2026-10-22/run_audit.py` |
| Training latents | `outputs/longrun-local-inputs/cache`, manifest SHA-256 `84febf0732cbb5e6b0cb85a607600b97c37cf1108e1137d7de55248ff795b541`; 47 training shards, each checked against the manifest; 47,767 records of 32 × 64, encoded by B0 |
| Training split | `data/nchlt_afrikaans/processed/stage_5_7/train.jsonl`, SHA-256 `2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab` (for overlap measures only) |
| Validation split | `data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl`, SHA-256 `bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e` (for the reserve definition only) |
| Human reference | `outputs/longrun-local-evaluation/validation-reference-mgpt.json`, SHA-256 `51cca0b60c0d8ece2a7b6e0f39b98177cf60726d36e3defe9f97d9dbd8d52498`, 1,000 rows already scored by mGPT |
| Diffusion banks | `outputs/longrun-local-evaluation/runs/{small,published}-seed{42..46}/validation-mgpt.json`, 1,000 scored rows each, sampling seeds 30000–30999. SHA-256 per file recorded in Appendix A. |
| Scorer | `ai-forever/mGPT`, revision `40897bd7c8b47a76802c411108ca6220438b8b40`, weights SHA-256 `c45c522411e64dbed6da1ed0ce0839bf03931d3a17b80182d06d4dab8d8b2fbb`, FP32, `score_text` from the execution source's `tools/compare_afrikaans_generators.py` |
| Test split | Not read |
| Hardware | Local RTX 3060 12 GB, batch one, the same environment as the 27 October train-versus-held-out diagnostic |

Outputs go to a new directory, `outputs/latent-controls-<run date>/`, created with `exist_ok=False`. No existing file is overwritten.

## 3. Conditions

All decoded conditions use B0's decoder with the benchmark settings, identical to `generate()` in the execution source: beam 4, `max_length` 128, `min_length` 5, `no_repeat_ngram_size` 3, `repetition_penalty` 1.2, `length_penalty` 1.0, `do_sample=False`; truncation at the first EOS after the start token; `decoder_tokens`, `eos`, `length_cap` and `sentinel` recorded per row. Latent shape is 32 slots × 64.

| Condition | Construction | n |
|---|---|---|
| **C1 Isotropic** | For i = 0…999: `torch.Generator('cpu').manual_seed(30000 + i)`, `randn(1, 32, 64)`, each slot L2-normalised and scaled to norm 8. Identical to `random_latent_control()` in `docs/research/ae_screen_2026-10-27/job/diffusion_stage.py`. | 1,000 |
| **C2 Projected Gaussian** | Flatten all 47,767 training latents to 2,048 dimensions in float64. Mean μ; covariance Σ by Ledoit–Wolf (2004) shrinkage towards (trace(S)/2,048)·I with the closed-form intensity, implemented in NumPy and recorded with the runner hash. Cholesky factor L of Σ. For i = 0…999: `numpy.random.Generator(PCG64(40000 + i))`, z = μ + L·ε with ε standard normal, reshape to 32 × 64, then each slot scaled to norm 8. | 1,000 |
| **C3 Recombined slots** | For i = 0…999: `numpy.random.Generator(PCG64(50000 + i))` draws 32 donor indices independently and uniformly from the 47,767 training records, with replacement. Slot *j* of the latent is slot *j* of donor *j*. Slots are already norm 8. Donor indices stored per row. | 1,000 |
| **Diffusion** | The existing validation banks: five small and five large runs. Not regenerated. | 10 × 1,000 |
| **R Reconstruction** | The passages of H, encoded by B0 through the same encoding path as the latent cache, then decoded. | ≤ 811 |
| **H Human** | The 811 rows of the human reference whose document is not in the reserve (§4). Not decoded. Existing mGPT scores reused. | ≤ 811 |

**C2 post-projection record.** For C2's 1,000 latents and for the training latents, record: per-slot norm mean and SD; ‖mean(C2) − μ‖; relative Frobenius difference between the covariance of C2's latents and Σ; and the Ledoit–Wolf shrinkage intensity. C2 is called a projected Gaussian, never moment-matched.

**Seeds.** C1's seeds (30000–30999) are numerically equal to the diffusion validation sampling seeds but drive a different generator (CPU `torch.Generator` against CUDA sampling after `seed_all`). The outputs are not paired and are never analysed as pairs.

## 4. Reference sample (H and R)

- The reserve is the 64 validation documents at positions 64–127 of the seed-91427 shuffle of sorted validation document IDs (the call in `run_audit.py`). The runner asserts this list before reading the reference file.
- H keeps every human-reference row whose `document_id` is not a reserve document: 811 rows from 58 documents. Rows are not resampled, replaced or reweighted. No row is added to raise the count.
- Rows over 128 untruncated mT5 tokens (EOS included) are excluded from both H and R, so truncation cannot differ between them. The final count is recorded in the manifest.
- This sample is concentrated: the largest document contributes 158 of the 811 rows. All uncertainty for H and R is therefore estimated by resampling documents, with each source passage kept together with its reconstruction.
- Reserve passages are not encoded, decoded or scored. The 189 reserve-document rows of the reference file were scored by mGPT in earlier work; that existing exposure is not changed by this protocol. The reserve is described as **reserved from this analysis**, not as untouched.

## 5. Readouts

**Per row:** mGPT `nll` and `tokens`; word count (whitespace split); length band; EOS, length-cap, sentinel and empty flags. R additionally: exact match against the tokenised model view, sentence chrF (SacreBLEU 2.6.0, character order 6, word order 0, β 2) and numeral recall/precision, as in the 27 October diagnostic.

**Per bank** (each control, each diffusion run, R, H):

| Measure | Definition |
|---|---|
| Raw perplexity | exp(Σ nll / Σ tokens) over scored rows |
| Length bands | Words 1–10, 11–20, 21–35, ≥36 (the 27 October audit bands) |
| Reference weights | Fixed at the band shares of all 1,000 human-reference rows: **0.140, 0.223, 0.349, 0.288**. They are not recomputed from the 811-row subset, for continuity with earlier length-standardised results. These weights include information from the 189 reserve-document rows computed in earlier work; the reserve is excluded only from new passage-level evaluation here. |
| Standardised perplexity | exp(Σ_b w_b · NLL_b), where NLL_b = Σ nll / Σ tokens over the bank's rows in band b. Band perplexities are never averaged directly. |
| Band support | A bank's standardised perplexity is reported only if every band has ≥ 20 scored rows. A band with 0 rows makes it **not estimable**; 1–19 rows makes it **insufficient support under this protocol's reporting rule**. Raw perplexity and band counts are always reported. |
| Distinct-2 | Unique whitespace-token bigrams / all bigrams, pooled over the bank (as in `compare_generation.py`) |
| Unigram entropy | Shannon entropy in bits of the pooled word-unigram distribution after NFC, casefold and whitespace normalisation (`norm()` in `compare_generation.py`) |
| Training copies | Rows whose `norm()` text equals the `norm()` text of a training passage |
| 8-gram overlap | Share of all word 8-grams in the bank (after `norm()`) that occur in the training split. Rows under 8 words contribute none; their count is reported. |
| Lengths | Mean words; EOS, length-cap, sentinel and empty counts |
| Per-row distribution | Quartiles and 5th/95th percentiles of per-row NLL per token |

Diffusion measures are computed per run and summarised as mean and sample SD over the five runs of each size.

## 6. Estimands and intervals

**Primary contrasts (two).** For s ∈ {small, large}:

Δ_s = (mean over the five runs of size s of standardised perplexity) − (standardised perplexity of C3).

Interval: Δ_s ± t₄(0.9875) · √(s²_runs / 5 + V_C3), where s²_runs is the sample variance of the five run values and V_C3 is the bootstrap variance of C3's standardised perplexity. t₄(0.9875) = 3.4954 and t₄(0.975) = 2.7764, from the closed-form Student t distribution with 4 degrees of freedom.

These are **approximate t-based intervals** that combine between-run variability with the estimated C3 sampling variance. The statistic is not exactly t-distributed. Using two 97.5% intervals gives a Bonferroni **nominal** family-wise level of 95%; actual coverage is approximate, particularly with only five training runs. They are described this way wherever they are reported.

**Control bootstrap.** 10,000 replicates, seed 20261030. Rows are resampled with replacement **within each length band**, keeping the bank's observed band counts and the fixed reference weights. Every replicate therefore has the same band support as the original bank; no replicate is discarded. If the original bank fails band support, no interval is computed and the contrast is classified as insufficient support.

**Exploratory contrasts.** C1 and C2 against each size, by the same formula with t₄(0.975) and 95% intervals. Labelled exploratory everywhere they appear; they cannot produce a headline claim.

**H and R.** 10,000 document-bootstrap replicates (seed 20261031; documents resampled with replacement, all their rows kept, R paired with H). Raw perplexity always gets a 95% percentile interval. Standardised perplexity gets one only if **no** replicate has an empty band. If any replicate has an empty band, the standardised interval is withheld and the point estimate, the original band counts and the fraction of replicates with an empty band are reported instead. H and R are reference conditions, not bounds, and enter no primary contrast.

**Diversity and overlap** are reported as point values (controls) and mean ± SD (diffusion), without decision rules. H and R have at most 811 rows against 1,000 for every other bank. Distinct-2 and entropy depend on sample size, so H and R values are labelled descriptive and sample-size-sensitive and are not ranked against the other banks.

## 7. Outcome rules (each primary contrast)

| Interval for Δ_s | Classification | Permitted wording |
|---|---|---|
| Entirely below 0 | **Lower** | "[Size] diffusion outputs had lower length-standardised mGPT perplexity than recombined training slots under the B0 decoder." |
| Contains 0 | **Not established** | "The comparison did not establish a difference in length-standardised mGPT perplexity between [size] diffusion outputs and recombined training slots." |
| Entirely above 0 | **Higher** | "[Size] diffusion outputs had higher length-standardised mGPT perplexity than recombined training slots under the B0 decoder." |
| Any bank fails band support | **Insufficient support** | "Length-standardised comparison not reported: [bank] has fewer than 20 outputs in band [b]." Raw perplexity is reported instead, without a classification. |

**Interpretation limits, fixed now:**
- No classification is described as "better generation", "beats" or a quality result.
- A **Lower** classification is always reported together with the distinct-2, entropy, copy and 8-gram comparisons in the same paragraph. If diffusion's distinct-2 or entropy is below C3's, or its copy count is higher, the result is reported as lower perplexity together with that difference. It is not discounted and not called an improvement.
- A **Higher** classification is not attributed to the decoder, the scorer or any other cause. Candidate explanations (length composition, generic or repeated text, copying, scorer preference) may be listed as untested.
- These rules do not change after the results are seen. A near miss is reported as the classification it receives.

## 8. Gates

| Gate | Requirement | If failed |
|---|---|---|
| G1. Integrity | All hashes in §2 and Appendix A match; strict load; no non-finite latents or scores; C1/C2/C3 latents finite with per-slot norm 8 ± 1e-4 | Stop before decoding. Report. |
| G2. Reproduction | (a) B0 regenerates the stored `correct.generated_ids` for all 81 passages in `outputs/reconstruction-audit-20261022/examples.jsonl` (SHA-256 `f0ea392ab0cf17c06c531f00131fdc80ac3b1c6d2f530b7a2866409105a21d3b`), correct arm, exactly. (b) mGPT re-scores the first three non-reserve human-reference rows with identical token counts and NLL within relative 1e-4 of the stored values. (c) mGPT re-scores the first 20 rows of `small-seed42/validation-mgpt.json` under the same tolerance. | Pause. Nothing is interpreted. Investigate and report the cause; resuming requires a dated note appended to this protocol. |
| G3. Budget | Peak allocated memory ≤ 10 GiB. After the first 32 decodes of each condition, projected Phase A GPU time (encoding, decoding, scoring) ≤ 4 hours | Stop safely. Report. Sample sizes are not reduced without a new protocol. |
| G4. Band support | §5 rule, applied per bank | Classification as in §7 |

## 9. Phase B (optional): sampled latents and per-slot normalisation

Runs only if Phase A completes, and only after the EMA checkpoints for **small-seed42** and **published-seed42** are transferred from the cluster with SHA-256 recorded on the cluster before transfer and checked locally. If they are not available, Phase B is reported as not conducted.

For each checkpoint:
1. Sample 1,000 latents with seeds 30000–30999 through `generate()`'s code path and the saved run configuration (DDPM, 250 steps, `seed_all(seed)` per sample, batch one). Save every latent.
2. **As produced:** decode with the benchmark settings. Report the share of rows whose text equals the stored validation bank row for the same seed. This is a reproducibility observation across GPUs, not a gate.
3. **Normalised:** decode the same latents after scaling each slot to norm 8.
4. Report: per-slot norm distribution of the sampled latents (mean, SD, 1st/99th percentiles) against the training latents; the share of seeds whose two decodes are identical; sentence chrF between paired decodes; paired difference in per-row NLL per token (paired bootstrap over seeds, 10,000 replicates, seed 20261032); standardised perplexity of each arm; and cosine similarity of each slot to its nearest training slot at the same position, for the sampled latents and for C1–C3.

Phase B conclusions are phrased as the effect of applying per-slot normalisation to these sampled latents. They do not say the benchmark's unnormalised path is a defect. The benchmark results are unchanged.

Budget: after the first 8 samples, projected Phase B GPU time ≤ 4 hours, else stop and report.

## 10. Implementation and reporting

- The runner is written **after** approval and reuses `load_ae`, `generate`'s decoding block, `score_text`, `norm()` and the band definitions unchanged.
- `--dry-run` builds C1–C3 latents, the H selection and band weights, and prints counts and the C2 record without loading B0 or mGPT or writing outputs.
- Before B0 loads, `contract.json` records this protocol's SHA-256, the runner's SHA-256, the manifest SHA-256 and all input hashes.
- Files: `manifest.json`; `contract.json`; `latents-{C1,C2,C3}.pt` with hashes; `decodes-{C1,C2,C3,R}.jsonl`; `scored-{C1,C2,C3,R}.jsonl`; `c2-record.json`; `analysis.json`; `ALL_OUTPUTS.md` with every decoded text; `runtime.json`.
- Examples quoted anywhere are rows 0, ⌊n/4⌋, ⌊n/2⌋ and ⌊3n/4⌋ of each condition, where n is that condition's final row count (rows 0, 250, 500, 750 when n = 1,000), labelled AI-assisted reading, not human evaluation.

**Additions to the paper** (`docs/research/paper_revision_2026-10-26`), only after the run:
- `evaluation.tex`: one paragraph defining C1–C3, R and H. C1–C3, R and diffusion share B0's decoder and settings; H is undecoded human text.
- `results.tex` (RQ3): one table with rows H, R, small (mean ± SD), large (mean ± SD), C1, C2, C3 and columns raw perplexity, standardised perplexity, mean words, distinct-2, entropy, copies, 8-gram overlap; the two primary contrasts with 97.5% intervals and classifications; two to four sentences using only §7's wording.
- `appendix.tex`: construction details, C2 record, band counts, H/R document concentration, exploratory contrasts, Phase B if conducted, fixed-rule examples.
- `discussion.tex`: one sentence on what the comparison does not show (§1).
- `EVIDENCE_REGISTER.md` and `verify_revision.py`: new rows with paths and hashes; recomputation of the table from saved files.

## 11. Related issue, handled separately

`docs/research/ae_screen_2026-10-27/compare_generation.py` uses `np.nansum` over band NLLs, so an empty band drops out without renormalising the weights and lowers the standardised perplexity. This protocol does not use that function's standardisation; §5–6 define their own. The autoencoder-screen comparison is a separate approved protocol. Its handling (a dated correction if it has not run; a documented correction alongside the original output if it has and a band was empty) is decided outside this document.

## Appendix A. Diffusion bank hashes (SHA-256)

| Run | `validation-mgpt.json` |
|---|---|
| small-seed42 | `ef98bc5d50b63ae04317471d66f1468b42967f13c623e434357a5e77fe978e74` |
| small-seed43 | `777cf891ad04e9b6dc0901b9084f1d8ef3bfd88643b1cd4aefaf8f7da5b9fa42` |
| small-seed44 | `f038c2e44fe6eeef90b52f41ad37a1f99018dbbbaa931cb75c93a5feba262743` |
| small-seed45 | `40ea866a625c8766fd1b98ba445bb46bbfabaec4ba1a0b8cc014dd79c5494099` |
| small-seed46 | `7dd1b769cbffd209f33fe32242eabe2a450abdb5ef6ce0b97052820cdca4a281` |
| published-seed42 | `cb3fb2d8d50199a5144d4463c9b91457de8aada604f4983d41cce2c1c153e8a2` |
| published-seed43 | `1c26a7b6a74f7cd0cfed7c0ef0e629d91bfec73942c748efc540ba32f1205cc9` |
| published-seed44 | `699fce97c8a5e16960f90bd934ea2b270c1e81a9276a86406120825275e791e4` |
| published-seed45 | `cf06811aa709413330eef9d5dc8730c814c0ea0fea888c20db71b63aaa86a661` |
| published-seed46 | `69d15ffa310178f707d71245993da1b40404f7a32fddc45d41c904353528225d` |

"Published" is the existing artefact label for the large configuration, not a publication claim.
