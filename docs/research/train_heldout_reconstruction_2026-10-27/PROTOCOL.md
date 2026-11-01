# Train-versus-held-out reconstruction diagnostic — protocol

Status: **APPROVED.** The committed SHA-256 of this file is the protocol hash in the run contract. After approval, the panel rules, seeds, readouts and thresholds may not change. Any later change is a new protocol with a new date.

Approved by: Hadi Butt Date: 27 October 2026

## 1. Question and scope

Is the original autoencoder's poor held-out reconstruction mainly a failure to generalise? The alternative is that it reconstructs its own training passages no better than held-out ones, which would point to fit, capacity or optimisation.

This is inference only. There is no training, no tuning, no test-split access, no diffusion sampling and no change to any historical benchmark, threshold or artefact.

What it **can** establish:
- whether training passages reconstruct materially better than held-out passages from the same latent path;
- whether the next defensible step is autoencoder-side training (the fallback) or no further training.

What it **cannot** establish:
- semantic sufficiency of the latent;
- whether better reconstruction would improve unconditional generation;
- within a fit-limited result, whether the limit is capacity or optimisation. Length-band results are descriptive only;
- anything about autoencoder training-seed variance, since there is one autoencoder.

## 2. Fixed inputs

| Input | Identity |
|---|---|
| Autoencoder | Job 56531, update 30,500, `outputs/cluster-results/afrikaans-56531/best/model.pt`, SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`. Strict load. |
| Execution source and tokeniser | `outputs/longrun-local-inputs`, every file hash checked against `inputs.json`, as in `docs/research/reconstruction_audit_2026-10-22/run_audit.py` |
| Training split | `data/nchlt_afrikaans/processed/stage_5_7/train.jsonl`, SHA-256 `2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab` |
| Validation split | `data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl`, SHA-256 `bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e` |
| Test split | Not read |
| Hardware | Local RTX 3060 12 GB, FP32, TF32 off, evaluation mode, batch one, same environment as the 22 October reconstruction audit |

Outputs go to a new directory, `outputs/train-heldout-recon-<run date>/`, created with `exist_ok=False`. No existing file is overwritten.

## 3. Panels

**Shared definitions:**
- Token length is the untruncated mT5 tokenisation including EOS.
- Bins: short is ≤32 tokens, medium is 33–64, long is 65–128.
- Passages over 128 tokens are excluded from every panel, so truncation cannot differ between panels.
- The model view and labels are the tokenised source, as in the 22 October audit.

**Panel V (held-out validation)**
- Sorted validation document IDs are shuffled with `random.Random(91427)`. This is the same call and seed as the 22 October audit. Positions 0–63 are the dev panel and 64–127 the untouched 103-passage reserve. The run asserts that the dev documents equal positions 0–63.
- Panel V takes **positions 128–250**: 123 documents that neither the dev panel nor the reserve uses.
- In that order, it takes up to four passages per document. Distinct bins are preferred, using the `select()` rule from `run_audit.py`. Within-document passage order comes from a separate `random.Random(91434)`, so the result does not depend on replaying the dev and reserve draws.
- A pre-count without the model gives 194 passages before the over-128 exclusion. The final count is recorded in the manifest. Documents are not replaced to raise the count.
- The reserve (positions 64–127) is not read.

**Panel T (training)**
- Sorted training document IDs are shuffled with `random.Random(91430)`.
- Bin quotas equal panel V's realised bin counts.
- In shuffled order, for each document: shuffle its eligible passages with the same generator, take the first whose bin quota is unfilled, and take **at most one passage per document**. Stop when every quota is filled.
- If any quota cannot be filled, stop and report. There is no substitution.

**Panel A (anchor)**
- The existing 81 dev passages, decoded with correct latents only.
- This is a reproduction check, not part of the comparison.

**Freezing:** the manifest stores panel IDs, bins and donor assignments. It is written and hashed before the autoencoder is loaded.

## 4. Arms

For every passage in panels T and V:

1. **Correct latent, benchmark decoding.** Beam 4, `max_length` 128, `min_length` 5, `no_repeat_ngram_size` 3, `repetition_penalty` 1.2, `length_penalty` 1.0, `do_sample=False`. These are identical to the benchmark and to Stage 1.
2. **Shuffled latent, benchmark decoding.**
   - The donor is drawn within the same panel and bin, from a different document.
   - Donors are a deterministic derangement, as in `run_audit.py`.
   - Seeds: `random.Random(91432)` for panel T and `random.Random(91433)` for panel V.
3. **Correct latent, unconstrained greedy decoding.** `num_beams` 1, no n-gram ban, `repetition_penalty` 1.0, `min_length` 5, `max_length` 128. This arm is secondary and plays no part in the decision.

Teacher-forced CE and accuracy are computed under arms 1 and 2. They do not depend on decoding.

## 5. Readouts

Per passage:
- token-weighted teacher-forced CE and token accuracy;
- exact match against the model view (primary) and against the source;
- sentence chrF (SacreBLEU 2.6.0, character order 6, word order 0, β 2, mixed case, no whitespace, effective order);
- numeral multiset precision and recall, using the regex from `run_audit.py`;
- EOS present and length-cap hit;
- source and output lengths.

Per panel and per bin:
- the means of the above;
- exact-match rate;
- token-weighted CE.

**Primary statistic:** Δexact = exact rate on T − exact rate on V, under arm 1. The ratio R = exact_T / exact_V is reported alongside it.

**Uncertainty:**
- Panels T and V are independent samples, so each is resampled by document within its own panel. The bootstrap is not paired.
- 10,000 replicates, seed 91431, percentile 95% intervals for Δexact, R, ΔCE (V − T) and ΔchrF (T − V).
- **Bin reweighting:** panel T matches V's bin counts by construction. A bin-reweighted Δexact (weights = V's bin shares) is reported as a check.

## 6. Validity gates (checked before classification)

| Gate | Requirement | If failed |
|---|---|---|
| G1. Reproduction | Panel A correct-arm generated IDs equal the stored 22 October outputs for all 81 passages | Stop. Report an environment or code mismatch. No classification. |
| G2. Integrity | Hashes match, no non-finite values, valid derangements, peak allocation ≤10 GiB, projected runtime ≤2 GPU-hours after the first 32 passages | Stop safely. Report. |
| G3. Latent dependence on training passages | Panel T shuffled-arm exact ≤2%, **and** shuffled-arm chrF on T exceeds that on V by ≤5 points | Classify as **M** (decoder memorisation). The T–V gap cannot then be attributed to the latent. |

## 7. Decision thresholds (arm 1, model-view exact)

| Outcome | Rule |
|---|---|
| **G: generalisation-limited** | exact_T ≥ 3 × exact_V **and** the lower 95% bound of Δexact ≥ +10 percentage points **and** the ΔCE (V − T) interval lies above 0 |
| **F: fit-limited** | exact_T ≤ 1.5 × exact_V **and** the upper 95% bound of Δexact ≤ +10 percentage points |
| **I: indeterminate** | Anything else, including exact and CE pointing in opposite directions |
| **M: memorisation** | Gate G3 failed |

These thresholds are fixed now and must not be lowered or reinterpreted after the results are seen. A near miss is reported as the outcome it is.

**Bias note.** The autoencoder selected its checkpoint on a shuffled 128-passage validation subset, which may overlap panel V. Any such selection exposure would make V look slightly better, so it biases the result against G.

## 8. What follows from each outcome

- **G:**
  - No decoder continuation, no 32×128 widening and no autoencoder retraining before 2 November.
  - The manuscript reports a generalisation gap as the leading explanation, with this protocol's limits stated.
  - More in-domain data or regularisation is future work.
- **F:**
  - The fallback becomes *eligible*: one seed of a full or joint fine-tune of mT5 at 32×64, following Lovelace et al. 2023 (Table 13).
  - It still needs its own written protocol, a measured throughput preflight and a confirmed cluster allocation before it can run. Approving this protocol does not authorise it.
  - Its selection signal must be teacher-forced CE on at least 1,024 validation passages, not the 81-passage panel.
- **I or M:**
  - No training.
  - Report the result descriptively.
  - Prioritise the generation-side analysis controls (E1 random latents, E3 beam-decoded AR, length-standardised reporting).

In every case, the historical benchmark, its five-seed results and the AR comparison stay unchanged.

## 9. Budget

The 22 October audit decoded 81 passages × 2 arms in 225 s, about 1.4 s per passage-arm. Here there are about 390 passages × 3 decodes plus 81 anchor decodes, so roughly 30–40 minutes on the local RTX 3060.

The hard cap is 2 GPU-hours (gate G2). No cluster GPU-hours are used, and there is no cache regeneration or diffusion retraining.

## 10. Implementation and reporting

- The runner is written *after* approval. It extends `run_audit.py` by adding panel T, panel V, the greedy arm and the anchor check.
- Its SHA-256, this protocol's SHA-256 and the manifest's SHA-256 are written to `contract.json` before the autoencoder loads.
- A synthetic dry run of panel selection may be executed before the real run. It must not load the model or read any output text.
- Reported: every passage's source, model view, donor and all outputs; per-bin tables; all intervals; the outcome label; and any gate failure.
- Any example quoted in the manuscript is selected by a fixed rule (for example, the first stored row per outcome category) and is described as AI-assisted reading, not human evaluation.
