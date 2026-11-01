# English ROCStories reproduction plan

## Why ROCStories

ROCStories is the correct first English corpus because it is the published
unconditional LD4LG task, is already implemented, has bounded short sequences,
and provides direct target settings. OpenWebText would answer a different,
large-scale language-modelling question and add substantial data engineering.
Avoiding it here removes approximately 1.5–2 weeks of work and reduces the risk
of debugging the data and model simultaneously.

## Fixed dataset construction

- Train: 93,161 ROCStories.
- Validation: first 1,000 examples of the supplied 5,000-example validation file
  after the fixed seed-42 processing.
- Test: remaining 4,000 examples.
- Maximum token length: 64.
- Random seed for training control: 42.
- Final generation seeds: 42, 43, 44, 45, and 46.

The preflight checks these counts before any job can be called valid.

## Experiment sequence and gates

### Gate 0 — preflight

Run imports, data counts/examples, token shapes/masks, BART cache/load, CUDA/BF16,
output write, checkpoint round-trip, W&B mode, and disk checks.

Pass: all required checks pass and a `preflight_report.json` is saved.

Fail: missing/changed ROCStories counts, checkpoint I/O failure, unavailable
required GPU capability, or missing required cached model.

### Gate 1 — autoencoder smoke (50 + 10 resumed steps)

Use BART-base, latent length 32, latent dimension 64, three AE layers, batch 2,
and no expensive generation metrics. Set `RUN_RESUME_TEST=1`.

Pass:

- `model.pt` contains model, optimizer, scheduler, scaler, and step;
- `args.json`, `run_metadata.json`, and package versions exist;
- loss is finite;
- resume advances from step 50 to step 60;
- reconstruction examples and validation metrics are saved.

Fail: out-of-memory, non-finite loss, missing state, no resumed step advance, or
shape mismatch.

### Gate 2 — autoencoder pilot (5,000 steps)

Enable reconstruction validation every 1,000 steps. Inspect fixed beam and
nucleus reconstructions and the validation curve.

Pass: validation cross-entropy is finite and improving/stable; token accuracy and
exact reconstruction are non-degenerate; saved samples are aligned with
references; resume works.

Fail: loss diverges, latent collapse is suspected, output is systematically
empty/truncated, or validation cannot complete.

### Gate 3 — full published-style autoencoder

Published-style settings:

- `facebook/bart-base`;
- latent length 32, dimension 64;
- three compression/reconstruction layers;
- maximum length 64;
- L2-normalised latents;
- AdamW, learning rate \(10^{-4}\), betas 0.9/0.999, weight decay \(10^{-2}\);
- linear decay, 1,000 warmup steps;
- target global batch 256;
- 50,000 optimiser steps.

On one GPU, the supplied default is batch 8 × accumulation 32 = 256. For multiple
GPUs, reduce accumulation so
`per_gpu_batch × accumulation × GPU_count = 256`.

Pass: complete checkpoint and provenance, finite reconstruction metrics, fixed
examples, and a validated checkpoint selected for diffusion.

Run strict held-out reconstruction evaluation after selecting the checkpoint:

```bash
python train_latent_model.py --eval_test \
  --resume_dir outputs/rocstories/autoencoder-full \
  --output_dir outputs/rocstories/autoencoder-full \
  --validation_max_examples 4000
```

### Gate 4 — latent-diffusion smoke (50 + 10 resumed steps)

The job must provide `LATENT_MODEL_PATH` and uses `--require_latent_model`.
Use a one-layer 64-dimensional Transformer and 10 DDPM sampling steps. Generated
quality is not a criterion.

Pass:

- `latent_model_provenance.json` says `using_latent_model: true`;
- latent length is 32 and latent dimension is 64;
- `args.json` records the autoencoder path;
- `model.pt` records `using_latent_model` and `latent_model_path`;
- finite training loss and resumed step advance.

Any missing proof above means this is not a latent-diffusion smoke.

### Gate 5 — latent-diffusion pilot (5,000–10,000 steps)

Use a four-layer, 256-dimensional Transformer, periodic samples, and validation
loss. The purpose is to detect optimisation, sampling, decoding, and metric
failures before the expensive model.

Pass: finite/improving loss, non-empty samples, strict evaluation path executes,
resume works, and no representation/provenance mismatch.

### Gate 6 — full published-style latent diffusion

- 12 Transformer layers, dimension 768, 12 heads;
- three dense connections;
- velocity prediction and L2 loss;
- cosine noise schedule;
- self-conditioning probability 0.5 and scale-shift conditioning;
- AdamW, learning rate \(2\times10^{-4}\), weight decay \(10^{-6}\);
- cosine LR decay and 1,000 warmup steps;
- target global batch 128;
- EMA 0.9999 and gradient clipping 1.0;
- 250,000 optimiser steps;
- 250-step DDPM sampling and four-beam decoding.

The one-GPU default is batch 8 × accumulation 16 = 128.

### Gate 7 — final English evaluation

Generate 1,000 non-empty samples for each seed 42–46. Report:

- GPT-2-Large perplexity for direct paper comparison;
- distinct/repetition statistics for 2-, 3-, and 4-grams;
- training-corpus 4-gram memorisation;
- unique words and unique sentences;
- MAUVE when dependency and compute availability permit;
- corresponding real test-text reference metrics;
- per-seed and aggregate JSON/CSV.

Required metric failure must terminate final evaluation. MAUVE is optional but a
missing value and reason must be recorded.

## Expected artifacts

Every training output contains:

- `args.json`;
- `run_metadata.json`;
- `run_invocations.jsonl` (preserves resume/evaluation invocation history);
- `package_versions.txt`;
- `model.pt`;
- reconstruction or generation samples;
- JSON metrics (and CSV for diffusion evaluation);
- W&B offline run files when enabled.

Diffusion outputs additionally contain `latent_model_provenance.json`.

## Verified command sequence

From the repository root:

```bash
conda env create -f environment.reproduction.yml
conda activate ld4lg-reproduction
python -m spacy download en_core_web_sm
python tools/preflight.py --cpu-only --output-dir outputs/preflight-local
pytest -m "not integration"
OUTPUT_DIR=outputs/rocstories/autoencoder-smoke RUN_RESUME_TEST=1 \
  bash scripts/reproduction/autoencoder_smoke.sh
LATENT_MODEL_PATH=outputs/rocstories/autoencoder-smoke \
  OUTPUT_DIR=outputs/rocstories/diffusion-smoke RUN_RESUME_TEST=1 \
  bash scripts/reproduction/diffusion_smoke.sh
```

Do not start either full script until its preceding gate is signed off.

## Cluster submission order

1. `00_preflight.slurm`
2. `01_autoencoder_smoke.slurm`
3. optional AE pilot via the canonical pilot script
4. `02_autoencoder_full.slurm`
5. `03_diffusion_smoke.slurm`
6. optional diffusion pilot via the canonical pilot script
7. `04_diffusion_full.slurm`
8. `05_evaluation.slurm`

Use Slurm `afterok` dependencies only after discovering the real Wits account,
partition, GPU resource syntax, time limits, and scratch rules.
