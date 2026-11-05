# Full decoder experiment: a measured 6–12-hour allocation

Prepared 24 October 2026 at the author's request. **Not submitted or launched.** This supersedes the fixed 6,000-update proposal and the old 25 October training cutoff. Submission is understood as the end of 27 October 2026 SAST (00:00 on 28 October). No claim of guaranteed or drastic improvement is made.

Updated for the author's one-submission requirement: **training, reserve evaluation, conditional shared-latent generation, mGPT scoring, report creation and a downloadable review archive are now one job.** Package name: `afrikaans-full-decoder-allinone-20261024`. It includes the fixed small-seed42 checkpoint and pinned mGPT files, rather than assuming they will be available later. This replaces the earlier training-only archive.

## What changes and why

Job 73970 trained only the last one/two decoder blocks, stopped after 1,500 updates per arm, and selected no candidate. Best development chrF was 68.7525/68.8133 against 67.7005, with 12/81 exact reconstructions against 11/81. Its 32:54 duration reflects a narrow trainable mask, cached latents and a plateau rule. The initial hours estimate was inaccurate. More time alone is not evidence of improvement.

This run tests **all 12 mT5 decoder blocks plus final layer norm (F12)** against **the last two blocks plus final layer norm (C2)**. Both start from the unchanged AE job 56531 step 30500, SHA256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`. Do not initialize from selected job-73970 weights. The larger trainable scope permits earlier cross-attention and intermediate decoder representations to adapt. The reconstruction path is a demonstrated limitation, but a decoder-specific cause remains hypothetical.

Freeze encoder, entire Perceiver, shared embeddings, output projection and diffusion. Therefore latent caches and existing diffusion checkpoints remain compatible. This cannot restore information absent from the latent code, and overfitting remains possible. It is an Afrikaans reconstruction intervention inside the original unconditional-generation study, not a Swahili extension or replacement of the proposal hypothesis.

## Controlled settings

Original train split, tokenizer, 128-token model view, 32x64 norm-eight latents; token-weighted teacher-forced CE; FP32 with TF32 off; dropout disabled; AdamW LR 2e-5, betas 0.9/0.999, weight decay 0.01; 100-update warmup then constant; clip norm 1; microbatch 1 with eight accumulated passages/update. These match across arms. Original beam-4 decoding, min length 5, max length 128, repeated-trigram prohibition, repetition penalty 1.2 and length penalty 1 remain fixed.

Deterministic complete training permutations use seed 20261024 + epoch, preserving every tail record across an eight-record batch crossing an epoch boundary. Both arms see exactly the same order and update counts at paired checkpoints. Log passage presentations and equivalent corpus passes. Multiple epochs are explicitly supported; the previous one-permutation indexing is not reused beyond its length.

## Clock budget, not a small step cap

- Allocate one RTX 8000 on mscluster107, four CPUs and 64 GiB host RAM for **12 hours**. Direct existing Python; no Apptainer. This GPU/environment successfully ran job 73970. New allocation availability still depends on Slurm.
- Target a **10.5-hour training-and-development phase measured from Python entry, including preflight**. Reserve the remainder for candidate locking, confirmation and checkpoint export. No arbitrary 1,500/3,000/6,000-update ceiling; no sleep or idle padding to inflate runtime.
- Before training, load the frozen denoiser, sample two predetermined latent seeds, decode them with B0, load mGPT, and reproduce four historical validation scorer anchors. Reserve `max(1.5 hours, 2 * predicted 256-pair transfer time + 20 minutes)` using measured sample/decode/score times. Reduce the training phase if necessary; still require six hours remaining after all calibration. The two initial latents are reused, not selected by quality.
- Calibrate 20 warmup and 100 timed updates per arm; they count in the run. Disposable save/load checks also count in measured training costs. Set a fixed validation interval from measured rates to approximately a 30-minute paired training block (100–5,000 updates). Subsequently size the final blocks and project feasibility from timings only, never observed quality.
- Use observed p90 update times, observed checkpoint overhead, conservative validation times and 15% timing margin. Recompute available work at each completed pair to avoid a faster-than-expected run ending after a small preset workload. A last partial block can be shorter; both arms get the same target. A few unused minutes are preferable to an interrupted checkpoint.
- Require at least six hours remaining in the training phase after calibration, with preflight/calibration completed within two hours. Otherwise fail admission clearly; do not report a short pilot as the long run.
- **No plateau early stopping.** Ordinary chrF or numeric-fidelity regressions affect checkpoint eligibility, not time allocation. Preserve the best eligible checkpoint while continuing the exposure test.
- Abort on nonfinite loss/gradients, changed frozen parameters, alignment failure, unexpected memory beyond 36 GiB allocated, external termination or deadline. Stop after three consecutive validations with at least 25% empty outputs or mean repeated-trigram fraction at least 0.20. These severe collapse stops may end the job before six hours; wasting the allocation cannot be promised.
- Latest start: **26 October 00:00 SAST**; absolute end: **26 October 12:00 SAST**. An immediate 24 October start normally finishes 25 October morning. The later cutoff still leaves about 36 hours before submission. No automatic extension, new seed, hyperparameter sweep or retry.

The 10.5-hour phase includes validation and checkpointing, so it is not a promise of 10.5 hours of optimizer computation. Report actual training, setup, evaluation and total job durations separately. Expected allocated memory is provisionally 8–20 GiB; measure it. Local backward verification without Adam states does not establish full training memory. No multi-GPU speedup or speed extrapolation from the last-two-block run is assumed.

## Integrity and crash handling

The package includes hashed historical source and train/validation inputs, no test text/cache. Preflight strictly checks weights, model cache, input identities and the frozen latent cache. Reproduce all 81 baseline token sequences before training. Verify gradients reach every intended block and shared embeddings/output remain frozen. Save model delta, optimizer, step and RNG state; exercise exact update/save/load replay on the allocation. Verify control-only frozen decoder blocks as well as all shared frozen components.

Every paired block saves per-arm latest state, selected best state and progress/selection JSON. SIGTERM requests state preservation. If a crash leaves unequal updates, disclose it; do not claim a matched final exposure or automatically resume. Restart would require an explicit reviewed continuation utility and the same source contract. The package intentionally provides no silent automatic requeue or restart after a failed training run.

## Selection and evaluation

Development: the same previously inspected 81 passages. There has now been repeated adaptive use; this is exploratory selection, not independent confirmation. Select highest eligible mean sentence chrF, breaking ties by exact matches then earlier step. Eligibility: zero empty/capped reconstructions, numeric precision and recall no more than two percentage points below B0, and repeated-trigram fraction no more than 0.005 above B0. Falling training loss is never a selection rule.

Reserve admission requires at least +3 chrF and three additional exact reconstructions (at least 14/81). F12 also needs at least +1 chrF over the best eligible C2; otherwise C2 can be the single candidate if it passes B0. Lock one candidate before reserve inspection. The existing 103 passages/64 documents were not consumed by job 73970 according to the author-supplied summary. They are disjoint for this comparison, not an untouched historical validation corpus.

Confirmation evaluates B0 and the two selected arm checkpoints once, with 10,000 paired document-bootstrap replicates. Candidate success requires +3 chrF and a positive 95% interval, at least +3 percentage points exact reconstruction, and the same fidelity/degeneracy guardrails. F12 attribution requires an additional +1 chrF over C2, positive interval and no fidelity regression. These intervals omit training-seed variability. Save randomized researcher comparison sheets for names, numbers, negation, clause structure and unsupported additions. No recruited human study; AI assistance stays labelled.

After the automatic reserve gate passes, **the same job** runs unconditional transfer. This explicitly changes the earlier operational requirement to wait for researcher inspection: computation proceeds, but adoption and interpretation remain provisional until the researcher reviews the exported examples. No candidate reselection follows reserve inspection. If the gate fails, record why transfer was skipped and export the negative result.

Use the fixed small-seed42 EMA checkpoint, SHA256 `e09b5a4697cf6e963b419c460db121d9f8e8017b6e25eacde813927f870f268b`, unchanged DDPM-250 configuration and seeds 20261024–20261279. Save all 256 latent tensors and decode the identical tensors with B0 and the candidate. Pin mGPT revision `40897bd7c8b47a76802c411108ca6220438b8b40`, FP32, context 512/stride 256, and archived scorer source. Empty outputs remain counted, have no defined PPL, and fail the guardrail.

Report pooled-token PPL, whitespace word lengths and length-band PPL, Distinct-1/2, repeated trigrams/fourgrams, EOS/cap/sentinel counts, normalized exact train matches and within-bank duplicate excess using Stage 4 definitions. Produce 10,000 paired latent-sample bootstrap replicates for candidate/B0 PPL ratio and mean-word difference; these omit training-seed uncertainty. Generation guardrails: zero empties, all outputs scored, cap/copy/duplicate rates no more than two percentage points above B0, repeat-3 increase at most 0.005, Distinct-2 at least 95% of B0, mean length within 10%, and PPL ratio at most 1.05. These are regression checks, not proof of better language quality. Export the first 64 seed pairs blinded and all flagged samples for researcher review.

The same job writes `REPORT.md`, `MANUSCRIPT_INSERT.md`, CSV tables and JSON metrics, and exports a compressed review archive with samples, shared latents, logs, protocol/code and artifact hashes. Large model/optimizer checkpoints remain on the cluster with their hashes recorded. No second GPU submission is part of the normal successful workflow. Hardware failure, cancellation or failed preflight can still require intervention. The manuscript insert is a draft for author review, not an automatic edit to historical results. Better reconstruction is not itself better unconditional generation. Test exposure remains disclosed; historical benchmark tables are preserved.

## Writing and stopping

Archive job 73970 as the first restricted adaptation experiment. Describe this larger run as an adaptively chosen follow-up with a changed stopping protocol. Report both arms, measured compute and exposure, all validation checks, selected steps and negative outcomes. Retain the original model unless the confirmation and transfer evidence justify a limited new claim. Do not reinterpret a smaller improvement as passing the original thresholds.

Freeze exploratory results by 26 October evening, leaving 27 October for the manuscript and submission checks. No training should consume the final submission hours. The author may revise scheduling, but doing so is a new explicit decision, not an automatic timer extension.

Primary local evidence: [Stage 1 reconstruction audit](../reconstruction_audit_2026-10-22/RESULTS.md), [Stage 4](../stage4_final_evaluation_2026-10-23/FINAL_EVALUATION.md), [previous decision](../deadline_experiment_plan_2026-10-24/NEXT_RUN_DECISION_2026-10-24.md). Architecture context: [Lovelace et al. 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html), [mT5](https://aclanthology.org/2021.naacl-main.41/); neither guarantees this intervention's efficacy.
