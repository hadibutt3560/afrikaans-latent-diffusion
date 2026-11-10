# Latent diffusion for Afrikaans text generation

Hadi Butt

Does latent diffusion work for a lower-resource language? We adapt the latent diffusion
model of [Lovelace et al. (2023)](https://arxiv.org/abs/2212.09462) to Afrikaans, using the
NCHLT corpus, an mT5-base encoder–decoder with a Perceiver bottleneck, and two diffusion
denoisers trained with five seeds each.

The diffusion models produce text that scores well, but the score hides a problem. Every
sample passes through the autoencoder, and the original autoencoder reconstructs only
11 of 81 held-out passages exactly. Random latents decoded by the same decoder reach
similar raw perplexity, so perplexity alone cannot show what the denoiser adds. Training
every autoencoder parameter fixes most of the reconstruction gap: 73 of 103 reserve
passages are reproduced exactly, against 19 before. Diffusion trained on that retrained
autoencoder, however, scores worse on perplexity and does not clearly meet the proposal's rule of
perplexity at most 1.20 times the autoregressive baseline's: the larger denoiser is
inconclusive on raw perplexity (1.21) and fails after length standardisation (1.38).

**Paper:** [PDF](docs/research/manuscript_versions_2026-10-27/new/main.pdf) ·
[LaTeX source](docs/research/manuscript_versions_2026-10-27/new/main.tex)

## Key results

| Result | Value | Details |
|---|---|---|
| Test mGPT perplexity, small / large denoiser (5 seeds each) | 51.31 / 48.61 | [Final evaluation](docs/research/stage4_final_evaluation_2026-10-23/) |
| AR baseline perplexity, test / validation (not a matched comparison) | 79.15 / 82.18 | [AR test check](docs/research/ar_test_2026-11-05/), [AR scope](docs/research/ar_scope_decision_2026-10-22/) |
| Ratio to AR on test, small / large (rule: at most 1.20), raw and length-standardised | 0.65 / 0.61 and 0.90 / 0.75 | [AR test check](docs/research/ar_test_2026-11-05/) |
| Length-standardised perplexity: small / large / Gaussian fitted to real latents / random latents | 71.3 / 59.6 / 63.6 / 107.6 | [Latent controls](docs/research/latent_controls_2026-10-30/) |
| Reserve reconstruction, original → retrained autoencoder: exact | 19 → 73 of 103 | [Autoencoder screen](docs/research/ae_screen_2026-10-27/) |
| Reserve reconstruction, original → retrained autoencoder: chrF | 68.32 → 98.70 | same |
| Unseen extra-data passages (681), exact: original / arm A / retrained (D) | 13 / 171 / 415 | [Follow-up checks](docs/research/followup_inference_2026-11-03/) |
| Diffusion on the retrained autoencoder: test perplexity, small / large (3 seeds each) | 288.45 / 95.90 | [Follow-up checks](docs/research/followup_inference_2026-11-03/) |
| Same, ratio to AR on test, raw / length-standardised: small / large | 3.64 / 1.21 and 3.87 / 1.38 | [AR test check](docs/research/ar_test_2026-11-05/) |
| Swahili run: reconstruction chrF, correct / shuffled latents | 74.8 / 24.3 | [Swahili check](docs/research/swahili_check_2026-11-01/) |

All readings are automatic. No human evaluation was carried out,
and readings made with AI assistance are labelled as such in the protocols and the paper.

## Where each result comes from

Folders are under `docs/research/`. Scripts and result files are inside the named folder unless a path is given.

| Paper | Folder | Script | Result file |
|---|---|---|---|
| Table 1, Appendix A: systems, model and training | — | `configs/afrikaans_longrun.json`, `tools/afrikaans_longrun.py` | — |
| §6.1, Table 2: reconstruction of 81 passages, shuffled latents | [reconstruction_audit_2026-10-22](docs/research/reconstruction_audit_2026-10-22/) | `run_audit.py`, `analyze.py` | `RESULTS.md`, `ALL_RECONSTRUCTIONS.md` |
| §6.1, Table 2: content checks (Claude) | [claude_labels_2026-11-07](docs/research/claude_labels_2026-11-07/) | — | `labels_81.csv`, `agreement.json` |
| §6.1: numerals, five shuffles, 681 extra-data passages | [local_followups_2026-11-01](docs/research/local_followups_2026-11-01/) | `run_followups.py` | `results/numerals.json`, `results/shuffle_seeds.json`, `results/extra_monitor_summary.json` |
| §6.1: test-split reconstruction loss | [stage4_final_evaluation_2026-10-23](docs/research/stage4_final_evaluation_2026-10-23/) | — | `FINAL_EVALUATION.md` |
| §6.2, Table 3: diffusion and human perplexity, diversity, denoiser size | [stage4_final_evaluation_2026-10-23](docs/research/stage4_final_evaluation_2026-10-23/) | `analyze.py` | `analysis.json`, `validation_comparison.csv` |
| Table 3: AR and human text on 256 validation samples | [ar_validation_audit_2026-10-22](docs/research/ar_validation_audit_2026-10-22/) | `run_audit.py`, `analyze_audit.py` | `AR_SUITABILITY_REPORT.md` |
| Table 3: random latents | [latent_controls_2026-10-30](docs/research/latent_controls_2026-10-30/) | `run_controls.py`, `analyze.py`, `first256.py` | `results/analysis.json`, `results/first256.json` |
| §6.2: seed-level intervals (Welch, permutation) | [e12_uncertainty_2026-11-01](docs/research/e12_uncertainty_2026-11-01/) | `compute_e12.py` | `results/e12.json` |
| §6.2: AR test comparison; Figure 1, Table 7 | [ar_test_2026-11-05](docs/research/ar_test_2026-11-05/) | `run_ar_test.py`; figure: `manuscript_versions_2026-10-27/new/figures/make_length_bands.py` | `results/analysis.json`, `results/length_bands_ar.json`; `stage4_final_evaluation_2026-10-23/length_bands.json` |
| §6.3, Table 4: language, shared phrasing, Goldfish, AR settings | [evaluation_package_2026-10-31](docs/research/evaluation_package_2026-10-31/) | `run_package.py`, `analyze_package.py` | `results/analysis.json`, `results/manual-check-result.json` |
| §6.3: tokenisation fertility | [fertility_2026-11-05](docs/research/fertility_2026-11-05/) | `run_fertility.py` | `results/fertility.json` |
| §6.3: random-latent banks, fitted Gaussian | [local_followups_2026-11-01](docs/research/local_followups_2026-11-01/) | `run_followups.py` | `results/random_banks.json`; fitted prior in `followup_inference_2026-11-03/analysis.json` |
| §6.4: decoder change 1, no trigram ban | [stage3_improvement_2026-10-22](docs/research/stage3_improvement_2026-10-22/) | `run_decoder_test.py` | `RESULTS.md` |
| §6.4: decoder change 2, last two blocks (job 73970) | [deadline_experiment_plan_2026-10-24](docs/research/deadline_experiment_plan_2026-10-24/) | `job/train_decoder.py` | results kept locally |
| §6.4: decoder change 3, all twelve blocks (job 74177) | [full_decoder_12h_2026-10-24](docs/research/full_decoder_12h_2026-10-24/) | `job/` | `RETRIEVED_JOB_74177.md` |
| §6.4: decoder change 4, dropout (job 75003) | [decoder_dropout_final_2026-10-25](docs/research/decoder_dropout_final_2026-10-25/) | `job/` | results kept locally |
| §6.4, Table 5: autoencoder screen | [ae_screen_2026-10-27](docs/research/ae_screen_2026-10-27/) | `job/screen.py`, `job/train_ae.py`, `data/` | `screen_report_from_cluster.json` |
| §6.5: diffusion on the retrained autoencoder | [ae_screen_2026-10-27](docs/research/ae_screen_2026-10-27/) | `job/diffusion_stage.py`, `compare_generation.py` | ratios in `ar_test_2026-11-05/results/analysis.json` |
| §6.5: random latents, fitted prior, 681 passages on the new decoder | [followup_inference_2026-11-03](docs/research/followup_inference_2026-11-03/) | `followup.py`, `analyze.py` | `analysis.json` |
| §7, RQ1: English control | [english_control_2026-08-15](docs/research/english_control_2026-08-15/) | — | `english_control_metrics.json` |
| Appendix B: Swahili | [swahili_check_2026-11-01](docs/research/swahili_check_2026-11-01/) | `run_check.py`, `summarise.py` | `results/summary.json`, `results/RESULTS.md` |

`docs/research/manuscript_versions_2026-10-27/new/verify_numbers.py` checks the numbers in the
paper against these result files. From a fresh clone, 52 of its 56 checks run; the 4 decoder
learning-curve checks need the raw cluster outputs and are skipped.

## Repository layout

| Path | Contents |
|---|---|
| `latent_models/`, `diffusion/`, `model/` | Autoencoder and diffusion model code (from the upstream release, extended for mT5) |
| `train_latent_model.py`, `train_text_diffusion.py` | Upstream training entry points |
| `tools/` | Afrikaans data preparation, multi-seed training, scoring and audits |
| `configs/afrikaans_longrun.json` | Denoiser sizes, seeds, schedule and sampling settings |
| `docs/research/<study>_<date>/` | One folder per study: protocol (fixed before the run), code and results |
| `cluster/wits/` | Slurm job scripts for the Wits cluster |
| `swahili_72h/` | The Swahili run of the same pipeline |
| `tests/` | Unit and integration tests |

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists where every method detail, scorer
version and interval procedure in the paper is fixed.

## Running the Afrikaans pipeline

The commands below run from the repository root. Long jobs are meant for a GPU cluster;
[the cluster guide](docs/research/WITS_CLUSTER_GUIDE.md) and
[the long-run guide](docs/research/AFRIKAANS_LONGRUN.md) give the Slurm versions.

**1. Environment.** Install the pinned packages:

```bash
pip install -r cluster/wits/longrun-requirements.txt
```

**2. Data.** Download the NCHLT Clean Afrikaans Text Corpus v2.0 from
[SADiLaR](https://repo.sadilar.org/) into `data/nchlt_afrikaans/raw/af`, then build the
splits ([details](docs/research/NCHLT_AFRIKAANS_PREPROCESSING.md)):

```bash
python tools/prepare_nchlt_afrikaans.py
python tools/finalize_nchlt_afrikaans.py
```

The follow-up autoencoder also trains on two government-domain Afrikaans sources:
[Gov-ZA cabinet statements](https://huggingface.co/datasets/dsfsi/gov-za-monolingual) and the
[Vuk'uzenzele newspaper](https://huggingface.co/datasets/dsfsi/vukuzenzele-monolingual)
(15,269 passages, 511,155 words after processing). Diffusion always trains on NCHLT only.
After the NCHLT splits exist, download both at pinned revisions, process them like NCHLT
(duplicates and documents overlapping NCHLT validation or test are removed) and build the
three validation panels ([details](docs/research/ae_screen_2026-10-27/PROTOCOL.md), §2–3):

```bash
python docs/research/ae_screen_2026-10-27/data/fetch_extra.py
python docs/research/ae_screen_2026-10-27/data/prepare_extra.py
python docs/research/ae_screen_2026-10-27/data/build_panels.py
```

`build_panels.py` also reads two earlier passage manifests kept under `outputs/`, which is
not in the repository.

**3. Autoencoder.** Train with every parameter trainable, as in the screen's winning arm
(64 latent slots, NCHLT plus the extra Afrikaans data and panels from step 2):

```bash
python docs/research/ae_screen_2026-10-27/job/train_ae.py --arm D --latents 64 --extra \
  --model-cache PATH_TO_HF_CACHE --output outputs/ae-D
```

**4. Diffusion.** Encode the corpus, time the hardware, plan the seed suite within the time
budget, then train the small and large denoisers:

```bash
python tools/afrikaans_longrun.py prepare --workspace outputs/longrun --ae PATH_TO_AUTOENCODER
python tools/afrikaans_longrun.py benchmark --workspace outputs/longrun --ae PATH_TO_AUTOENCODER
python tools/afrikaans_longrun.py plan --workspace outputs/longrun
python tools/afrikaans_longrun.py run --workspace outputs/longrun --ae PATH_TO_AUTOENCODER
```

The config pins the autoencoder by its SHA-256. For an autoencoder from step 3, set
`expected_autoencoder_sha256` and `latent_length` in a copy of the config and pass it with
`--config`; the screen's `job/screen.py` writes this copy automatically.

**5. Evaluation.** Generate the validation and test samples and score them with mGPT:

```bash
python tools/afrikaans_longrun.py evaluate --workspace outputs/longrun
python tools/afrikaans_longrun.py final-test --workspace outputs/longrun
python tools/afrikaans_longrun.py score --workspace outputs/longrun
```

Each study folder in `docs/research/` has its own runner and analysis script for the
controls and checks reported in the paper.

## What is not in the repository

Model weights, the raw corpus, latent caches and full sample banks are too large or not
ours to redistribute. Each study folder holds its summary results. Seven also record SHA-256 hashes of their
original outputs, so a rerun can be checked against them: `ar_validation_audit_2026-10-22` (`ARTIFACT_MANIFEST.json`), `e12_uncertainty_2026-11-01`, `evaluation_package_2026-10-31`, `latent_controls_2026-10-30`, `local_followups_2026-11-01`, `swahili_check_2026-11-01` and `train_heldout_reconstruction_2026-10-27`. The NCHLT corpus is distributed by SADiLaR under CC BY 2.5 South
Africa. Gov-ZA is released under the MIT licence and Vuk'uzenzele under CC BY 4.0, both by
DSFSI (Lastrucci et al., 2023, *Preparing the Vuk'uzenzele and ZA-gov-multilingual South
African Multilingual Corpora*, RAIL 2023); the texts come from the Government Communication
and Information System.

## Other documentation

- [English ROCStories reproduction](docs/ENGLISH_REPRODUCTION.md), the implementation check
  run before the Afrikaans study
- [Upstream README](docs/UPSTREAM.md): the original release's instructions and abstract
- [Method decisions](docs/research/METHOD_DECISIONS.md)

## Credit and citation

This code is a fork of the official release of *Latent Diffusion for Language Generation*
by Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman and Kilian Q. Weinberger,
which builds on implementations by [Lucidrains](https://github.com/lucidrains). Please cite
their paper if you use this code:

```bibtex
@article{lovelace2024latent,
  title={Latent diffusion for language generation},
  author={Lovelace, Justin and Kishore, Varsha and Wan, Chao and Shekhtman, Eliot and Weinberger, Kilian Q},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

To cite the Afrikaans study (GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff)):

```bibtex
@misc{butt2026afrikaans,
  title={A Representation-Centred Evaluation of Latent Diffusion for Afrikaans Text Generation},
  author={Butt, Hadi},
  year={2026},
  howpublished={\url{https://github.com/hadibutt3560/afrikaans-latent-diffusion}}
}
```

The upstream licence is in [LICENSE](LICENSE).
