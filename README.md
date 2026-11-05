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

**3. Autoencoder.** Train with every parameter trainable, as in the screen's winning arm
(64 latent slots, NCHLT plus extra Afrikaans). The panels and extra data are built by the
scripts in `docs/research/ae_screen_2026-10-27/data/`:

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
ours to redistribute. Each study's `results/` folder holds the summary files and a
`SHA256SUMS.txt` that records the hash of every original output, so a rerun can be
checked against them. The NCHLT corpus is distributed by SADiLaR under CC BY 2.5 South
Africa.

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

To cite the Afrikaans study:

```bibtex
@misc{butt2026afrikaans,
  title={A Representation-Centred Evaluation of Latent Diffusion for Afrikaans Text Generation},
  author={Butt, Hadi},
  year={2026},
  howpublished={\url{https://github.com/hadibutt3560/afrikaans-latent-diffusion}}
}
```

The upstream licence is in [LICENSE](LICENSE).
