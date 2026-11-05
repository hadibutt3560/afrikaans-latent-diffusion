# Reproducibility index

This is the index for the manuscript `docs/research/manuscript_versions_2026-10-27/new/main.tex`. It points to where each method detail is fixed.

`outputs/` is not tracked by git. Summary results for the studies behind the manuscript are copied into each study's `results/` folder, with a `SHA256SUMS.txt` covering every file of the original output folder, including the sample banks and latents that are not committed. Items marked *(local)* exist only on the author's machine or the cluster. Model weights, raw corpora and full sample banks are not in the repository.

## Proposal

The research proposal, referred to in the manuscript as "the proposal", fixed the research questions and credibility criteria before the Afrikaans experiments. It is the version submitted to the University of the Witwatersrand on 24 May 2026, unaltered, and was added to the repository on 2 November 2026; the submission date rests on the university's record, not on this commit.

| Item | Where | SHA-256 |
|---|---|---|
| Submitted proposal | `docs/proposal/submitted_proposal_2026-05-24.pdf` | `6aa1ef1e4151877561562753af955870b6191e316351985acc5f218f22039493` |

## Training

| Item | Where |
|---|---|
| Method decisions (English reproduction, frozen backbone, schedules) | `docs/research/METHOD_DECISIONS.md` |
| Diffusion training configuration (denoisers, seeds, EMA, sampling) | `configs/afrikaans_longrun.json`, `tools/afrikaans_longrun.py` |
| Final evaluation protocol (frozen models, test-bank reanalysis) | `docs/research/stage4_final_evaluation_2026-10-23/PROTOCOL.md` |
| AR baseline protocol and budget | `docs/research/ar_scope_decision_2026-10-22/`, `outputs/cluster-results/ar-baseline-57759/protocol.json` *(local)* |
| Autoencoder screen (arms, recipe, panels, thresholds) | `docs/research/ae_screen_2026-10-27/PROTOCOL.md`, `screen_report_from_cluster.json` |
| Diffusion on the screen winner (jobs 77175 training, 77176 evaluation) | `docs/research/ae_screen_2026-10-27/job/diffusion_train.slurm`, `diffusion_finish.slurm`, `diffusion_stage.py`; outputs in `outputs/cluster-results/ae-diffusion-77176/` *(local)* |
| Generation comparison on the new autoencoder (empty-band correction) | `docs/research/ae_screen_2026-10-27/compare_generation.py`, `NOTE_2026-11-01_empty_bands.md` |
| Decoder interventions | `outputs/cluster-results/job-74177-review/job-74177/REPORT.md` *(local)* |

## Scorers and pinned revisions

| Item | Value | Where |
|---|---|---|
| mGPT | `ai-forever/mGPT` revision `40897bd7c8b47a76802c411108ca6220438b8b40`; FP32, 512-token window, stride 256, start token prepended | `tools/compare_afrikaans_generators.py` (`score_text`) |
| Goldfish | `goldfish-models/afr_latn_1000mb` revision `27555fdf876be2d273e82379d1e7dc42f8eed104` | `docs/research/evaluation_package_2026-10-31/PROTOCOL.md` §2 and Appendix C |
| GlotLID | `cis-lmu/glotlid` revision `baa20d139151a26a71a20eddabf30d4de7329fc7` | same |
| Afrikaans dictionary | LibreOffice `af_ZA`, commit `892f0f402cf8347aa3273f774365c094c310cf1b`, read with spylls 0.1.7 | same |
| chrF | SacreBLEU 2.6.0, character orders 1–6, no word n-grams | manuscript Evaluation section |

## Length standardisation and support

- **Bands:** 1–10, 11–20, 21–35 and 36 or more whitespace words.
- **Weights:** 0.140, 0.223, 0.349 and 0.288, the human length distribution.
- **Standardised perplexity:** exp(Σ w_b · mean NLL_b).
- **Band support:** a bank needs at least 20 outputs in every band.
- **Sources:** `docs/research/latent_controls_2026-10-30/PROTOCOL.md` and `docs/research/evaluation_package_2026-10-31/PROTOCOL.md` §7.

## Intervals

| Result | Procedure | Where |
|---|---|---|
| Large minus small test perplexity | Run bootstrap; Welch 95% interval; exact permutation over 252 splits | `docs/research/e12_uncertainty_2026-11-01/compute_e12.py` and `e12.json` |
| Decoder retraining chrF change | Paired document bootstrap, 10,000 replicates, seed 20261101 | same |
| Reconstruction contrasts | Paired document bootstrap, 10,000 replicates | `docs/research/reconstruction_audit_2026-10-22/`, `ae_screen_2026-10-27/PROTOCOL.md` §6 |
| Language, overlap and scorer comparisons | Document bootstrap (2,000 replicates, seed 20261103) for the human intervals; Welch interval and t₄(0.9875) with a bootstrap of the AR sample's variance (seed 20261102) for the scorer contrasts | `docs/research/evaluation_package_2026-10-31/PROTOCOL.md`, `analyze_package.py` |

## Results committed with each study

| Study | Protocol and code | Results |
|---|---|---|
| Train versus held-out reconstruction | `docs/research/train_heldout_reconstruction_2026-10-27/` | `results/analysis.json`, `results/ALL_RECONSTRUCTIONS.md` |
| Autoencoder screen | `docs/research/ae_screen_2026-10-27/` | `screen_report_from_cluster.json` |
| Latent-prior controls | `docs/research/latent_controls_2026-10-30/` (`run_controls.py`, `analyze.py`) | `results/analysis.json`, `results/ALL_OUTPUTS.md`; outcome note `NOTE_2026-11-01_C3_support_and_C2.md` |
| Evaluation package | `docs/research/evaluation_package_2026-10-31/` | `results/analysis.json`, `results/mauve.json` and the other JSON summaries; the blinding key of the manual check is not committed |
| Run-level uncertainty (E12) | `docs/research/e12_uncertainty_2026-11-01/` | `results/e12.json` |
| Swahili check | `docs/research/swahili_check_2026-11-01/` (`run_check.py`, `summarise.py`) | `results/summary.json`, `results/RESULTS.md`, `results/s4.json`, `results/gates.json` |
| Follow-up inference checks (job 81249: self-conditioning audit, fitted prior on D, 681-passage reconstruction) | `docs/research/followup_inference_2026-11-03/` (`followup.py`, `followup.slurm`, `analyze.py`) | `PROTOCOL.md` (with dated note on approval), `analysis.json` |
| AR baseline on the test split (RQ3 common-split check, run locally 5 Nov.) | `docs/research/ar_test_2026-11-05/` (`run_ar_test.py`; paths assume the `research` repository layout) | `PROTOCOL.md`, `results/analysis.json`, `results/anchors.json`, `results/contract.json`; samples in `outputs/ar-test-20261105/` *(local)* |
| English control (RQ1: BART on ROCStories, 5,000 updates) | `docs/research/english_control_2026-08-15/` (`README.md`) | `english_control_metrics.json`, extracted from the 20 September research audit |
| Tokenisation fertility from saved outputs (5 Nov.) | `docs/research/fertility_2026-11-05/` (`run_fertility.py`; paths assume the `research` repository layout) | `results/fertility.json` |

## Readings and records

| Item | Where |
|---|---|
| Content-check labels (AI-assisted) | `docs/research/reconstruction_checklist_2026-11-01/E7_ai_labels.csv` and its README |
| Literature search record | `docs/research/manuscript_versions_2026-10-27/new/LITERATURE_SEARCH_2026-11-01.md` |
| Number checks for the current manuscript | `docs/research/manuscript_versions_2026-10-27/new/verify_numbers.py` |
| Data-use history | Manuscript, Evaluation section ("Uncertainty and data use") |
