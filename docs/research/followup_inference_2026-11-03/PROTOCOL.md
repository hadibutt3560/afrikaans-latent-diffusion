# Follow-up inference checks: sampling audit, fitted prior and fresh reconstruction

**Status: DRAFT.** Not approved. Nothing in this protocol may be run until Hadi Butt approves it in chat. Once it is approved, any change is made only by adding a dated note at the end.

## 1. Purpose and scope

The new diffusion suite on autoencoder D (job 77176) is complete. Three questions remain that inference alone can answer:

- **F1. Sampling audit.** Training normalises the self-conditioning estimate to norm 8, but sampling does not (`normalize_sampling_self_condition = false`). Does aligning them change the outputs of the final systems?
- **F2. Fitted prior on D.** Does D's diffusion model produce better text than a Gaussian fitted to D's training latents? This repeats the C2 control already run for B0.
- **F3. Fresh reconstruction.** Does the reconstruction repair hold on the 681 `extra_monitor` passages, which were never used for training or selection?

**Scope limits:**
- Inference only: no training, no tuning, no test-split reads. The NCHLT test split is not touched.
- One Slurm job (`followup.slurm`). Every step skips work whose output already exists, so a resubmission resumes.
- All readings are automatic. Nothing here establishes fluency or meaning; that is for the separate human evaluation.

## 2. Fixed inputs (hashed into `contract.json` before any model loads)

| Input | Location on the cluster |
|---|---|
| Autoencoder D | `outputs/screen/arms/D/best/model.pt`; must hash to `14d9cae3…e9f2` (the screen's winner hash) |
| Autoencoder A | `outputs/screen/arms/A/best/model.pt`; hash recorded |
| Autoencoder B0 | `$OLD_AE/model.pt`; must hash to `123f4508…7f5c` |
| New denoisers | `outputs/screen/diffusion/runs/{small,published}-seed{42,43,44}/best.pt`, EMA weights; hashes must equal `diffusion/selection.json` |
| Historical denoisers | `outputs/b0-runs/{small,published}-seed{42,43,44}/best.pt`, uploaded from `outputs/longrun-local-inputs/runs/`; hashes recorded |
| Code | `tools/longrun_training.py`, `job/recon_eval.py`, `job/diffusion_stage.py`, imported unchanged; the historical and new manifests record identical hashes for every diffusion source file |
| Data | `data/extra_afrikaans/processed/extra_monitor.jsonl` (681 passages); D's training-latent cache `outputs/screen/diffusion/cache/` |
| Scorer | mGPT, pinned revision, weights hash `c45c5224…`; anchors must reproduce (as in job 77176) |

Every configuration (latent length, sampler, 250 DDPM steps, beam settings, FP32) comes from the configuration stored inside each checkpoint. Nothing is changed except the one flag under test in F1.

## 3. F1: self-conditioning audit

**Systems:**
- **Primary:** D small and D published, seeds 42–44 (6 checkpoints).
- **Secondary:** B0 small and B0 published, seeds 42–44 (6 checkpoints). These are run only if they have been uploaded; otherwise their absence is logged.

**Design:**
- Each checkpoint samples 100 latents twice, with the flag off and with it on.
- Sample i uses seed 70000 + i in both arms. The starting noise and the sampler's random stream are therefore identical, and the arms differ only in the flag.
- Seeds 70000–70099 belong to no earlier bank: in-training samples used 20000+, validation 30000+ and test 120000+.
- Decoding uses the engine's own `generate`, unchanged.

**Reproduction check (informational):** before the audit, each checkpoint regenerates the first two samples of its stored `validation-selected-samples.json` with the flag off. An exact match is recorded but not required.

**Readouts per run and arm:**
- token-weighted mGPT perplexity;
- mean per-sample NLL per token;
- mean words;
- distinct-2;
- EOS reached, length cap and sentinel counts.

**Primary outcome:** the paired difference, flag on minus off, in mean per-sample mGPT NLL per token. It is computed per run, with a 95% paired bootstrap interval over samples (2,000 replicates, seed 20261106).

**Decision rule (fixed now):**
- **No material effect:** every primary-system run's interval lies within ±0.10 nats per token.
- **Material effect:** the pooled mean is outside ±0.10, and at least 5 of 6 primary runs agree in sign. The final-system conclusions are then re-examined under a new protocol. The test banks are not replaced.
- **Inconclusive:** anything else. This is reported as such.

The audit compares the two settings. It does not choose a setting for the paper, and the test banks are not regenerated under either one.

## 4. F2: fitted prior on D

This is the C2 rule of `latent_controls_2026-10-30/PROTOCOL.md`, applied to D's 64 × 64 latents:
- Flatten all 47,767 D training latents to 4,096 dimensions in float64.
- Take the mean μ, and a Ledoit–Wolf covariance shrunk towards (trace(S)/4,096)·I, using the same NumPy implementation.
- Take the Cholesky factor L.
- Build three banks of 1,000. Bank b ∈ {40000, 41000, 42000} draws z = μ + L·ε, with ε from `PCG64(b + i)`. Each z is reshaped to 64 × 64, and every slot is scaled to norm 8.
- The record keeps the shrinkage intensity, slot norms before and after projection, and moment differences, exactly as for C2.

**Decoding:** D's decoder, with the benchmark beam settings and EOS truncation as in `generate`.

**Comparisons**, all already on disk except C2-D:
- D diffusion test banks (3 runs per size);
- D isotropic control (`controls/random-latents-winner`, seeds 30000+);
- B0's C1/C2 banks;
- human test text.

**Readouts:**
- raw and length-standardised mGPT perplexity (weights 0.140, 0.223, 0.349, 0.288; at least 20 rows per band);
- mean words;
- distinct-2;
- sentinel count;
- exact and word 4-gram copies against NCHLT train and against D's `extra_train`.

**No perplexity winner rule.**
- If C2-D's standardised perplexity is within the range of the three D published runs, the paper may not claim that the denoiser adds value beyond a fitted prior on this measure.
- If it is clearly worse (above 1.20× their mean), the paper may say that the denoiser improves on a fitted prior under mGPT, and nothing more.

## 5. F3: reconstruction on the 681 extra passages

**Systems:** B0, A and D. Each is evaluated through `job/recon_eval.py` unchanged:
- FP32, batch one, the 128-token model view;
- the benchmark beam settings.

B0 is re-run here, so that all three share the same code path. The earlier local B0 figure (13 exact) is reported beside it as a check, not a gate.

**Disclosure:** D was trained on `extra_train`, which comes from the same two sources but different documents. D's training also logged cross-entropy (CE) on `extra_monitor` as an overfitting monitor; it was never used for any decision. D is therefore in-domain and A and B0 are not. The paper must say this.

**Readouts:**
- exact match;
- mean sentence chrF (character order 6, β 2);
- numeral recall and precision;
- teacher-forced CE;
- length-cap count.

Each is reported pooled, and separately for Gov-ZA and Vuk'uzenzele.

**Intervals:** paired document bootstrap (2,000 replicates, seed 20261107) for D − B0 and A − B0, on exact-match rate and on mean chrF.

**Reading:** this is evidence of generalisation to unseen documents. It is not a fresh confirmation set for the whole project.

## 6. Order, budget and stopping

**Order:**
1. Gates.
2. F3 (about 1 hour).
3. F1 primary (about 2.5 hours).
4. F1 secondary (about 2.5 hours).
5. F2 decoding (about 1–2 hours).
6. mGPT scoring of every new text file (about 0.5 hours).

These estimates come from job 77176's measured rates on the RTX 8000: 8.0 s per published sample, 4.9 s per small sample, and about 1.2 s per reconstruction.

**Limits:**
- The job's time limit is 20 hours. Its internal deadline is 19.5 hours, minus ten minutes for export.
- A stage that cannot finish before the deadline stops cleanly with exit 75, and resubmission resumes it.

## 7. Outputs

Everything goes to `outputs/followup/`:
- `contract.json`;
- `recon681/{B0,A,D}.jsonl` and their summaries;
- `audit/{run}-{off,on}.json` and `-mgpt.json`;
- `audit/reproduction.json`;
- `prior/C2-D-{seed}.json` and `-mgpt.json`;
- `prior/c2-record.json`.

The analysis is a separate CPU step, run locally after the archive is copied back. It follows sections 3–5 exactly.

## 8. Permitted paper use

Each finding gets one or two sentences in the existing Results subsections, and the paper keeps its 8-page body. The paper must report:
- F1's decision under the fixed rule;
- F2's comparison in the fixed wording;
- F3's paired differences, with the in-domain disclosure.

No result here may be described as human-validated.

## Dated notes

**4 November 2026, record of approval status.** The user submitted job 81249 on 3 November at 19:44 UTC, after this draft was written and unchanged. Explicit approval was not given in chat. The job completed at 01:05 UTC on 4 November. `analyze.py` was written after the job completed but before any result was read; it implements sections 3–5 unchanged. F3's bootstrap resamples the 17 source documents in `extra_monitor`, so its intervals rest on few independent units.
