# Decision after job 73970: full decoder adaptation

24 October 2026. **Proposed, not implemented or launched.** There is no evidence supporting a guaranteed or drastic improvement in 6–12 hours. This is one final, higher-capacity reconstruction experiment with a matched continuation control, contingent on approval and calibration. Afrikaans only. The historical benchmark and job 73970 remain unchanged.

## What the new evidence establishes

Author-supplied Slurm accounting records job 73970 as COMPLETED, exit 0:0, elapsed 00:32:54 on mscluster107. Its summary reports 1,220.797 training seconds, peak PyTorch-allocated memory 2.951 GiB, early stopping after 1,500 updates per arm, and no admitted candidate. This is evidence from pasted remote artifacts; the actual remote checkpoints and full output directory have not yet been retrieved or independently hashed locally.

Evidence: scheduler output and summary supplied by the author (not retained in the repository). Remote identifiers: `outputs/job-73970/summary.json`, `selection.json`, `C1-best.pt`, `T2-best.pt` under `/home-mscluster/hbutt/afrikaans-decoder-refine-20261024`.

| Development result | Original B0 | C1 best, step 500 | T2 best, step 500 |
|---|---:|---:|---:|
| Mean sentence chrF | 67.7005 | 68.7525 | 68.8133 |
| Exact reconstruction | 11/81 | 12/81 | 12/81 |
| Numeric recall | 21/33 | 21/33 | 21/33 |
| Numeric precision | 21/25 | 21/24 | 21/24 |

The best checkpoints preserved numeric recall; the later T2-1500 checkpoint fell to 18/33. Neither arm met the prospective +3 chrF and +3 exact-reconstruction gate. This excludes neither delayed improvement nor gains from broader adaptation. It supports only modest development gains under the tested masks, learning rate, exposure and seed. No reserve or unconditional-generation improvement was established.

The original AE was already trained through step 30,500; job 73970 was continuation, not a model learning Afrikaans from scratch. Its 1,500 updates exposed each arm to 12,000 passages, about one quarter of the 47,767-passage training corpus. Short elapsed time reflects cached latents and few trainable blocks. The earlier 6–15-hour planning estimate substantially overestimated this restricted workload; measured timing supersedes it.

## Bottleneck and intervention

[Stage 1](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/reconstruction_audit_2026-10-22/RESULTS.md) establishes that the reconstruction path uses source information but loses content: correct/shuffled CE 1.079/8.319, 11/81 exact reconstruction, and content errors in untruncated passages. It does not identify the responsible component. [Stage 4](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/FINAL_EVALUATION.md) shows that favourable diffusion MSE and aggregate external PPL do not resolve fidelity.

The strongest remaining bounded intervention is to allow **all 12 mT5 decoder blocks** to adapt to the same frozen latent-derived inputs. Earlier cross-attention and intermediate transformations can then learn; the previous run only modified the last one or two blocks. This is a plausible mechanism, not proof of a decoder bottleneck. If compression discarded a number or entity, a more flexible decoder cannot reliably reconstruct that missing information and may instead hallucinate or memorize.

Freeze the entire mT5 encoder, Perceiver encoder AND Perceiver reconstruction decoder, shared embeddings, output projection, and diffusion. The authoritative [model interface](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/longrun-local-inputs/latent_models/t5_latent_model.py) separates `get_diffusion_latent` from `get_decoder_input`. Neither will change in this experiment. This avoids retraining diffusion or regenerating its training cache. Do not use a generic full-model fine-tuning flag.

The autoencoding/latent-generation separation is supported by [Lovelace et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html); the pretrained backbone is [mT5, Xue et al. 2021](https://aclanthology.org/2021.naacl-main.41/). Neither paper predicts success for this intervention on this corpus.

## Concrete experiment

- B0: unchanged AE job 56531, step 30500; SHA256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`.
- C2 control: final two decoder blocks and final layer norm trainable.
- F12 treatment: all 12 decoder blocks and final layer norm trainable, including decoder block parameters such as relative attention bias. Embeddings and LM head remain frozen.
- Start both from B0 with fresh identical optimizer settings, not the development-selected step-500 models. One shared seed 20261024 and identical passage order. Compare same-step and selected-checkpoint outcomes; training-seed uncertainty remains unmeasured.
- Fixed settings: original 32-by-64 norm-eight cached latents; original tokenizer and length 128; teacher-forced token-weighted CE; FP32, TF32 off; dropout off; AdamW LR 2e-5, betas (0.9, 0.999), weight decay 0.01; 100-update linear warmup then constant LR; gradient clipping 1; microbatch 1, accumulation 8. Original deterministic beam-4 decoding and all penalties unchanged.
- **6,000 updates per arm**, roughly one complete training-corpus pass. Use deterministic reshuffling at epoch boundaries, retaining the tail instead of dropping records. The current trainer slices one permutation and cannot safely handle 6,000 updates without modification: 6,000 times 8 exceeds 47,767. This is not a command-line-only extension.
- Evaluate B0, then both arms every 1,000 updates on the same 81 development passages. Archive all metrics and sample outputs. Select highest eligible chrF, exact-match then earlier-step tie breaks; never select by training CE or PPL.
- The planned 6,000-update horizon tests the possibility of delayed recovery. **Do not reuse the previous plateau stop.** This deliberately changes the stopping protocol and must be labelled a new exploratory experiment selected after job 73970. Safety and serious fidelity deterioration still override the horizon.

## Resources and admission

Use one RTX 8000, 48 GiB, on mscluster107 with direct Python. User demonstrated Torch 2.7.1+cu128, Transformers 4.38.2 and successful CUDA computation there; job 73970 subsequently trained there. Availability is not guaranteed after release. Recheck allocation before scheduling; do not silently substitute another GPU. No paid compute or Swahili run.

Estimate, not measurement: F12 could take approximately 1–4 seconds/update versus the measured approximately 0.44 seconds/update for C2. Six thousand updates per arm imply about 2.4–7.4 hours of training combined. Validation, initialization, hashing and checkpointing add roughly 0.5–1.5 hours, with separate confirmation/transfer evaluation if admitted. A **6–12-hour scheduling allowance** is plausible; there is no minimum duration to fill. Full-decoder speed must be measured, not inferred by multiplying the earlier runtime by six.

Engineering memory estimate: 8–20 GiB with microbatch 1 and FP32, subject to measurement; the earlier 2.95 GiB figure does not apply to F12. Record allocated and reserved peaks and free VRAM. Allocate 64 GiB host RAM. Count warmup and save/load tests in the budget. Do not keep both optimizer states resident on GPU.

Before the main run, verify full trainable names/counts, unchanged frozen hashes, exact cached-input alignment, baseline reproduction, nonzero gradients in early decoder blocks, and optimizer/RNG save-resume. Measure 20 warmup + 100 updates per condition and one complete validation. Save these as part of the fixed run; do not tune the LR from their quality.

Admit only if `1.25 * (6000 * (p90_C2 + p90_F12) + 12 * validation_seconds + measured_setup_and_checkpoint_overhead)` plus a two-hour evaluation reserve fits both 12 hours and the remaining time to **25 October 2026, 15:00 SAST**. If not, do not start this design or quietly shorten its exposure. Implementation/preflight has a two-hour cap. All training must end by that absolute cutoff; final exploratory results freeze by 26 October 10:00 SAST. Keep 26 October for writing.

Abort on nonfinite loss/gradients, changes to frozen parameters or cache mismatch. Stop the paired experiment on two successive checks with numeric recall more than five percentage points below B0, or repeated-trigram fraction above 0.02, or persistent empty outputs; report stopped conditions. Save state at interruption and deadline. Do not extend beyond 6,000 updates or start a replacement sweep after failure.

## What counts as success

Retain the previous development eligibility and +3 chrF/+3 exact-reconstruction gate without lowering thresholds after seeing results. Numeric precision/recall may fall by at most two percentage points; no empty/capped reconstructions; repetition increase at most 0.005. F12 must also beat the eligible C2 control by at least one chrF point to support the expanded adaptation claim. If both improve equally, credit continued adaptation, not the additional trainable blocks.

Fix one candidate before consuming the existing 103-passage/64-document reserve. That panel was reported as not consumed by job 73970; verify this from retrieved artifacts. The development panel has repeated exposure and both experiments are adaptively selected. Reserve success requires +3 chrF with a positive paired-document-bootstrap 95% interval, +3 percentage points exact reconstruction, fidelity/degeneracy guardrails, and researcher review of names, numbers, negation and unsupported additions. There is no participant study. AI-assisted inspection remains separately labelled. After reserve inspection, no new candidate selection or tuning on it.

Only a confirmed reconstruction candidate proceeds to the original fixed small-seed42 denoiser's 256 shared-latent generation comparison, using the previous plan's seeds, scoring, length/diversity/copying checks and researcher inspection. Reconstruction improvements alone do not demonstrate unconditional-generation improvements. External PPL alone cannot establish either. No new test tuning; historical test exposure stays disclosed.

## Decision and manuscript

Recommend this single full-decoder experiment **if one further training attempt is desired**, not as a promised rescue. Do not run both the earlier proposed 3,000-update extension and this design; the C2 arm already addresses additional exposure with a control.

If successful, add a separately labelled exploratory decoder-adaptation section with B0/C2/F12, matched-update exposure, selected steps, full resource cost, reserve uncertainty and any generation transfer. If inconclusive or negative, report that broader decoder adaptation did not meet the declared criterion within the bounded budget. Neither outcome proves compression is the sole cause. Retain the original hypothesis and historical all-seed benchmark. Do not claim drastic improvement, novelty, a matched AR victory, or fulfillment of the composite proposal criterion without the missing evidence.

This file is a concrete protocol for review. No new trainer or job has been launched. Approval is still required before training under the author's explicit instruction.
