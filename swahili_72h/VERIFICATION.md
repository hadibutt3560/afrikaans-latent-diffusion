# Verification record

Local checks, 2026-10-09. No Slurm job was submitted and held-out test text was not used for model evaluation. All new files are under `swahili_72h`.

## Passed locally

- **Data:** membership, strict UTF-8, completeness, empty/invalid characters, duplicates, cross-split overlap and boundaries audited (`data/audit.json`; shared-phrase detection is indicative). Passage checksums verified; joining all exported spans reproduced all 738 decoded source files exactly. Raw byte hashes and the original manifest are retained.
- **Tokenizer:** 492,624 passages, zero truncation, zero unknown tokens on validation/test; text and token-array checksums exported.
- **Assets:** mT5-base and mGPT weights match upstream Hugging Face LFS SHA256 values and are in the package with tokenizers, configurations and the fixed scorer. Thirty Linux CPython 3.10 wheels, frozen in `requirements.lock`; CUDA PyTorch and the SIF are cluster prerequisites.
- **GPU smoke (RTX3060):** mT5 reconstruction forward/backward; both real diffusion architectures trained; continuous and interrupted/resumed four-step runs matched **exactly** in model, optimizer, EMA, RNG and LR scheduler state (`verification/gpu-smoke.json`). mGPT loaded offline and scored Swahili text (torch 2.11.0+cu128, Transformers 5.5.0, Python 3.13.14, 12 GB).
- **Production batches:** one finite step at AE microbatch 2 x accumulation 8 and diffusion microbatch 16 x accumulation 4; peak VRAM about 5.12 / 1.20 / 4.54 GB (AE/small/published). Published-size DDPM250 with beam-4 decoding passed. Not a steady-state benchmark.
- **Compatibility:** torch 2.7.1+cpu with Transformers 4.38.2 passed miniature mT5 forward/backward/generation, causal-LM forward and tokenizer parity on 128 rows (Windows CPU only).
- **End-to-end fixture:** AE caching, small diffusion training, validation selection, test MSE, DDPM250 with beam decoding and mGPT scoring. **All fake splits used eight validation passages; actual test data was not used.** A software test only.
- **Reporting:** JSON/Markdown, blinded review CSV with separate key, PNG and PDF plots and an archive, every member matching its SHA256 manifest.
- **Code:** ten orchestration tests (segmentation, deadline budgeting, reserve growth, follow-up admission, stage skip and rejection, subprocess failure/recovery, signal forwarding, no training after selection on resume); Bash syntax and Python compilation; nineteen extracted diffusion AST bodies identical to the original repository (only the import shell reduced).

## Problems found and fixed

- Transformers 5.5 overrides mT5's serialised untied-head setting. The loader restores the saved architecture and checks embedding and output-head tensors; the cluster pins 4.38.2.
- Missing imports in the reduced diffusion module were fixed after real training and DDPM250 sampling exposed them.
- With an untrained bottleneck the model had validation NLL 49.6033, token accuracy 0% and initial chrF 9.1257, so it is **not** an off-the-shelf Swahili AE and must pass the reconstruction gate after adaptation.
- Windows Application Control blocked an optional font DLL, so the PDF uses standard fonts. No security controls were disabled.

## Requires the cluster

- mscluster110 health, account access, scheduler policy, Apptainer bindings, signal propagation, CUDA/Blackwell kernels, quota, filesystem throughput and production VRAM/throughput.
- AE convergence and gate, cache time, diffusion, generation and scoring benchmarks (including generated text, which can tokenize much longer) and the number of seeds viable in 72 hours.
- Validation/test results, language quality and human review.
- Confirmation that the discrete pipeline uses the exported passage rows.

The allocation repeats GPU, resume, scorer and tokenizer checks first. `completion.json`, not Slurm success, distinguishes complete, partial and failed runs. `package.py` verifies the transfer archive (`transfer/bundle-verification.json`).
