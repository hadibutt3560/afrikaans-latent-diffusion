# Autoencoder recipe, extra data and latent length: screen and diffusion protocol

Status: **APPROVED.** After approval, the arms, panels, selection rule, thresholds and stopping rules may not change; any change is a new protocol with a new date.

Approved by: Hadi Butt Date: 27 October 2026

## 1. Questions

1. Does training the autoencoder with the published full-mT5 recipe improve reconstruction over the current autoencoder (job 56531)?
2. Separately, does adding about 511k words of government-domain Afrikaans help, and do 64 latent slots help, especially on long passages?
3. Does diffusion trained on the best new autoencoder produce better unconditional generation than the existing five-seed benchmark under the same protocol?

**What it cannot establish:**
- Attribution of a *generation* change to recipe, data or latent length. Only the winner gets diffusion, so the generation comparison is winner against the old system as a whole.
- Training-seed variance of the autoencoder: there is one autoencoder per arm.
- Human-judged quality.

The old benchmark, its five-seed results, the AR comparison and every earlier threshold stay unchanged. Earlier test-set exposure remains disclosed.

## 2. Data

| Source | Use | Identity |
|---|---|---|
| NCHLT stage 5–7 `train.jsonl`, 47,767 passages | All arms | SHA-256 `2af76338…` |
| Gov-ZA cabinet statements (Afrikaans), `dsfsi/gov-za-monolingual` revision `12fc7ccc…` | Arms C, D | MIT on the dataset card; text from gov.za, GCIS |
| Vuk'uzenzele (Afrikaans), `dsfsi/vukuzenzele-monolingual` revision `b689fe84…` | Arms C, D | CC BY 4.0 |
| `extra_train.jsonl`, 15,269 passages, 511,155 words | Arms C, D | `data/extra_afrikaans/processed/manifest.json` |
| `extra_monitor.jsonl`, 681 passages (5% of documents) | Overfitting monitor only | Never used for training or selection |

**Extra-data processing** (`data/prepare_extra.py`):
- Gov-ZA lost its line breaks, so line breaks are restored where numbered headings and sentences run together.
- Then comes the unchanged NCHLT normalisation and contact masking, splitting into passages of at most 50 words at sentence punctuation, and removal of passages under three words.
- Exact duplicates are removed, both within the new text and against all NCHLT splits.
- Any document with a passage sharing a 12-word sequence with NCHLT validation or test is removed whole. This removed 3 documents.

**Known artefact:** some Vuk'uzenzele words are split by PDF extraction (for example "ty dens"). They are left in.

Diffusion trains on NCHLT only, in every case. Extra data reaches the autoencoder only.

## 3. Panels (NCHLT validation; `data/build_panels.py`)

The seed-91427 shuffle of the 251 validation documents, used since 22 October, defines three disjoint document groups:

| Panel | Documents | Passages | Use |
|---|---|---|---|
| Selection | Positions 0–63 (the former dev documents) | All 663 | Early stopping within an arm, and choosing between arms |
| V | Positions 128–250 | The 194 from the 27 October diagnostic | Descriptive comparison of all arms and B0 |
| Reserve | Positions 64–127 | The 103 fixed on 22 October, never evaluated | Evaluated **once**, for the winner and B0 only |

B0 is the job 56531 autoencoder.

## 4. Arms

| Arm | Data | Latent slots |
|---|---|---|
| A | NCHLT | 32 × 64 |
| B | NCHLT | 64 × 64 |
| C | NCHLT + extra | 32 × 64 |
| D | NCHLT + extra | 64 × 64 |

**Shared recipe** (`job/train_ae.py`), following Lovelace et al. 2023, Table 13 (mT5-base column):
- Initialisation: pretrained mT5-base, with a fresh, seeded Perceiver compression and reconstruction module of 3 layers, as in B0. Lovelace used 1 layer; keeping 3 avoids a second architectural change against B0.
- **All parameters trainable.** Transformers 5.x leaves the Perceiver uninitialised on load, so it is rebuilt explicitly and every weight is checked to be finite.
- Latents L2-normalised to norm 8, as in B0.
- bf16 autocast with FP32 weights; AdamW with learning rate 1e-4, betas (0.9, 0.999), weight decay 0.01, gradient clipping 1.0.
- Effective batch 128: micro-batch 32 × accumulation 4, or 16 × 8 if a 30-update preflight runs out of memory.
- 500 warm-up updates, then linear decay to zero at 10,000 updates. That is about 27 passes over NCHLT, or 21 for arms C and D.
- mT5 dropout at its configured 0.1 during training, off in evaluation.
- Teacher-forced, token-weighted cross-entropy (CE).
- 128-token cap.
- Seed 20261027.

**Early stopping:**
- Every 250 updates, compute token-weighted CE on the selection panel; the checkpoint with the lowest CE is kept.
- Stop once there have been at least 2,000 updates and 8 consecutive checks without a relative improvement of 0.5%.
- Each arm is also capped at 6 hours of training.
- CE on a fixed 1,024-passage training probe, and on the extra-data monitor slice, is logged to show overfitting. Neither is used for decisions.

## 5. Choosing the winner (`job/screen.py`, applied once)

1. **Eligibility.** The arm completed (early stop, update limit or time cap), and at most 2 of its V-panel decodes hit the length cap or lack EOS.
2. **Ranking.** Eligible arms are ranked by selection-panel CE, recomputed in FP32 at batch one with the same code used for B0.
3. **Parsimony.** If the best arm has 64 slots and its 32-slot partner (B→A, D→C) has a selection CE within 2% (relative), the 32-slot arm wins. 64 slots doubles the diffusion sequence length.
4. **Gate to diffusion.** Diffusion runs only if the winner's selection CE is lower than B0's. Otherwise the experiment ends after the screen and is reported as a negative result.

## 6. Reconstruction readouts and success thresholds

**Per autoencoder, per panel:** token CE and accuracy; beam-decoded exact match (against the tokenised model view), sentence chrF and numeral recall and precision; length-cap and EOS counts; all of these also by length bin (short ≤32, medium 33–64, long 65–128 tokens). Decoding uses the benchmark settings: beam 4, `min_length` 5, `max_length` 128, repetition penalty 1.2, no repeated trigrams, FP32, batch one.

**Contrasts.** Paired document bootstrap, 10,000 replicates, seed 20261028. All V-panel contrasts are descriptive:
- recipe: A − B0;
- latent slots: B − A and D − C;
- extra data: C − A and D − B;
- overall: winner − B0.

**Primary reconstruction result:** winner − B0 on the **reserve** panel. It counts as an improvement only if all three hold:

| Measure | Requirement |
|---|---|
| Exact match | At least +10 percentage points, with the 95% lower bound above 0 |
| chrF | At least +3, with the 95% lower bound above 0 |
| Numeral recall | Not more than 0.02 below B0 |

Anything less is reported as a miss, including near misses.

## 7. Diffusion and generation (`job/diffusion_stage.py`)

**Training.** Unchanged `tools/afrikaans_longrun.py` with `configs/afrikaans_longrun.json`. Only `expected_autoencoder_sha256` and `latent_length` change: same small and large denoisers, seeds from 42, early stopping, planner, EMA, 1,000 validation and 1,000 test samples per run, DDPM with 250 steps, and the same beam decoding. The NCHLT latents are re-encoded with the winner.

**Scoring.**
- The pinned mGPT (revision `40897bd7…`, FP32, 512-token window, stride 256), loaded from the copy already on the cluster.
- It must match the weights hash and reproduce three stored reference scores within a relative 1e-4 before anything is scored.

**Random-latent control (E1).** 1,000 Gaussian latents rescaled to norm 8 are decoded with each of B0 and the winner, then scored the same way. This shows how much of each system's score comes from the decoder alone.

**Comparison with the existing benchmark** (analysis after retrieval). For each denoiser size, compare test mGPT perplexity in two forms:
- raw;
- length-standardised to the human validation length mix, using the same bands and weights as the 27 October audit.

Also compare distinct-2, the exact training-copy count and mean length. Intervals come from Welch's test across seeds and a run bootstrap.

**Generation counts as improved for a denoiser size** only if all of the following hold:
- length-standardised test perplexity is lower than the old benchmark's, with the 95% Welch interval of the difference below 0;
- distinct-2 is at least 0.95 × the old value;
- exact training copies are at most double the old count;
- the random-latent control for the winner does not reach the winner's own diffusion perplexity. That would mean the decoder alone explains the score.

**Not a quality claim.** Perplexity is not quality. A fixed-rule sample (indices 0, 25, 50, 75 and 100 of seed 42) is read and reported as AI-assisted inspection, not human evaluation.

## 8. Jobs, budget and stopping

All three jobs run on mscluster111 (RTX PRO 6000 Blackwell, 96 GB). mscluster110 and mscluster112 failed a GPU check on 27 October. Each job first checks the GPU with `nvidia-smi` and a small CUDA calculation.

| Job | Contents | Slurm limit | Expected |
|---|---|---|---|
| 1 `screen.slurm` | B0 evaluation; four arms (preflight and training); evaluations; selection; reserve | 40 h | 10–28 h |
| 2 `diffusion_train.slurm` (after job 1 succeeds) | Re-encode cache, smoke, benchmark, plan, diffusion suite | 72 h | 55–65 h |
| 3 `diffusion_finish.slurm` (after job 2 ends) | Resume the suite if unfinished; evaluate; final test; random-latent control; scoring | 36 h | 8–12 h |

- Every stage resumes from its own outputs after a clean stop (exit 75).
- There is no automatic resubmission beyond this chain.
- Checkpoints stay on the cluster. Every job exports a results archive without them.
- The results are not complete until job 3 finishes. If the 2 November deadline arrives first, the paper reports what has finished and states what has not.

## 9. Reporting

- All four arms are reported, whatever the winner.
- Each training curve, the stopping reason and the hours used are reported.
- The protocol and code hashes are recorded in each job's outputs.
- AI-assisted reading of samples is labelled as such.
