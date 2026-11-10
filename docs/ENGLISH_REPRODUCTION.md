# English ROCStories reproduction (implementation control)

The English ROCStories run with BART checks that the training code runs and learns
before it is applied to Afrikaans. It uses the upstream method unchanged.

1. **Create the environment (network required).**

   ```bash
   conda env create -f environment.reproduction.yml
   conda activate ld4lg-reproduction
   python -m spacy download en_core_web_sm
   ```

2. **Run preflight.** The CPU command works without starting training. Cached
   model/tokenizer checks are reported, not downloaded by default.

   ```bash
   python tools/preflight.py --cpu-only --output-dir outputs/preflight-local
   ```

   On a prepared GPU node:

   ```bash
   python tools/preflight.py --require-cuda --require-bf16 \
     --require-model-cache --load-model --output-dir outputs/preflight-gpu
   ```

3. **Run fast tests.**

   ```bash
   pytest -m "not integration"
   ```

   Cached-model integration tests are separate:

   ```bash
   pytest -m integration
   ```

4. **Run the English autoencoder smoke.** This requires BART-base and preferably
   a GPU. The resume test advances ten additional optimiser steps.

   ```bash
   OUTPUT_DIR=outputs/rocstories/autoencoder-smoke RUN_RESUME_TEST=1 \
     bash scripts/reproduction/autoencoder_smoke.sh
   ```

5. **Run the full English autoencoder only after smoke and pilot gates pass.**

   ```bash
   OUTPUT_DIR=outputs/rocstories/autoencoder-full \
     bash scripts/reproduction/autoencoder_full.sh
   ```

   Then run strict held-out reconstruction evaluation with
   `python train_latent_model.py --eval_test --resume_dir
   outputs/rocstories/autoencoder-full --output_dir
   outputs/rocstories/autoencoder-full --validation_max_examples 4000`.

6. **Run latent-diffusion smoke.** `LATENT_MODEL_PATH` is mandatory. The run
   fails instead of silently falling back to raw BART features.

   ```bash
   LATENT_MODEL_PATH=outputs/rocstories/autoencoder-smoke \
     OUTPUT_DIR=outputs/rocstories/diffusion-smoke RUN_RESUME_TEST=1 \
     bash scripts/reproduction/diffusion_smoke.sh
   ```

7. **Run full diffusion only after smoke and pilot gates pass.**

   ```bash
   LATENT_MODEL_PATH=outputs/rocstories/autoencoder-full \
     OUTPUT_DIR=outputs/rocstories/diffusion-full \
     bash scripts/reproduction/diffusion_full.sh
   ```

8. **Evaluate.** Final test evaluation is strict for required metrics and
   generates 1,000 samples for each seed 42–46.

   ```bash
   python train_text_diffusion.py --eval_test \
     --resume_dir outputs/rocstories/diffusion-full \
     --output_dir outputs/rocstories/diffusion-full \
     --num_samples 1000 --sampling_timesteps 250 \
     --sampler ddpm --sampling_schedule cosine
   ```

9. **Submit to Wits.** First edit the generic Slurm templates with verified
   account/partition settings, then follow
   `docs/research/WITS_CLUSTER_GUIDE.md`.

10. **Inspect artifacts and resume.** Each run saves complete arguments, Git and
    environment provenance, invocation history, dataset/model counts, batch
    accounting, package versions, and a resumable `model.pt`. Diffusion also saves
    `latent_model_provenance.json`; it must state `using_latent_model: true`.
    Resume with `--resume_training --resume_dir PATH --output_dir PATH` and use
    `--resume_to_step N` to extend the saved horizon.

The data split is fixed: 93,161 training stories, the first 1,000 of the
supplied 5,000-example validation file after seed-42 processing, and the
remaining 4,000 as test; maximum token length 64. The full gate definitions were
in `docs/research/ENGLISH_REPRODUCTION_PLAN.md`, removed on 10 November 2026 and
kept in the git history. The study used a reduced 5,000-update run, not the full 50,000- or 250,000-step
schedules.

