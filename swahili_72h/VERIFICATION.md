# Verification record

Prepared locally on 2026-10-09. No Slurm job submitted and no model evaluation of the actual held-out test text performed.

## Passed locally

- Read the saved handoff and preserved existing repository work; all new experiment files are under `swahili_72h`.
- All supplied file membership, strict UTF-8, completeness, empty/invalid-character checks, whole-file and passage duplicates, cross-split overlap, and observable boundaries audited. `data/audit.json` is the detailed record. Shared phrase detection is indicative rather than exhaustive.
- Every exported passage text checksum verified; joining all exported spans reproduced all 738 decoded source files exactly. Raw byte hashes and the original manifest are retained.
- Full tokenizer audit over 492,624 passages. Zero truncation; zero unknown tokens on validation/test. Canonical text and token-array checksums exported.
- Cached mT5-base and mGPT weights matched upstream Hugging Face LFS SHA256 values. Both model weight files, tokenizer files, configurations and the fixed scorer are physically present in the package.
- Thirty Linux CPython 3.10 dependency wheels downloaded; versions frozen in `requirements.lock`. CUDA PyTorch and the SIF are documented existing cluster prerequisites, not bundled.
- Real RTX3060 GPU smoke: mT5 reconstruction forward/backward and initialization reconstruction; both actual diffusion architectures trained; continuous versus interrupted/resumed four-step runs matched **exactly** for model, optimizer, EMA, RNG and LR scheduler state. `verification/gpu-smoke.json` records the comparisons. No tiny substitute architecture was used for these GPU resume comparisons.
- Real mGPT loaded offline and scored Swahili reference text. Local scope: torch 2.11.0+cu128, Transformers 5.5.0, Python 3.13.14, RTX3060 12 GB; these are not cluster throughput measurements.
- Configured production batches passed a finite forward/backward/optimizer/EMA step: AE microbatch 2 x accumulation 8, both diffusion models microbatch 16 x accumulation 4. Observed peak allocated VRAM was about 5.12 / 1.20 / 4.54 GB for AE/small/published respectively in this one-step local check. Published-size DDPM250 plus beam-4 decoding also passed. This is a memory/functionality smoke, not a steady-state cluster benchmark.
- Separate compatibility check: torch 2.7.1+cpu and Transformers 4.38.2; miniature mT5 forward/backward/generation, causal-LM forward, and real-tokenizer parity for 128 exported rows passed. This is an API check on Windows CPU, not Linux CUDA validation.
- Real-model end-to-end fixture: AE latent caching, small diffusion training, full fixture validation selection, fixture test MSE, DDPM250 plus beam decoding, external mGPT scoring. **All fake splits used eight validation passages; actual test data was not used.** The fixture is a software test, not a research result or language-quality result.
- Report created JSON/Markdown, blinded review CSV and separate key, PNG and PDF plots, and a portable results archive. Every archived member verified against its SHA256 manifest. PNG visually inspected. PDF uses standard PDF fonts to avoid dependence on an optional Windows-blocked font component.
- Ten standard-library orchestration tests passed: exact segmentation, conservative deadline budgeting, slow-sampling reserve growth, useful follow-up admission after convergence, completed-stage skip, exhausted-stage rejection, real subprocess failure/recovery, signal forwarding, and no new training after validation selection on resume.
- Bash syntax checks passed for launcher/setup. Python files compiled. Nineteen extracted diffusion helper/class AST bodies match the original repository exactly; only the import shell was reduced.

## Problems discovered and addressed

The local Transformers 5.5 mT5 configuration forcibly changes the serialized untied-head setting. The loader now restores the saved architecture explicitly and verifies both the input embedding and output-head tensors against pretrained weights. The cluster version is pinned to 4.38.2. Missing imports in the reduced diffusion module were exposed by real training and DDPM250 sampling, corrected, and followed by a successful integration run.

The pretrained multilingual model with an untrained bottleneck had validation NLL 49.6033, teacher-forced token accuracy 0%, and mean initial free-running character chrF 9.1257 on the small initialization probes. It is **not** usable as an off-the-shelf Swahili AE. The full allocation must adapt it and pass the preregistered reconstruction gate. A four-step fixture is intentionally not expected to pass that gate.

An initial PDF attempt failed because Windows Application Control blocked an optional font-library DLL. The standard-font PDF path subsequently succeeded. A temporary approval-review usage-limit failure delayed that rerun; it was completed after the author requested continuation. No security controls were disabled.

## Still requires cluster hardware / real training

- Current mscluster110 availability/health, account access, scheduler policy, Apptainer bindings and signal propagation, exact CUDA/Blackwell kernels, quota, filesystem throughput, and full production-batch VRAM/throughput.
- Actual AE convergence and validation gate; full-corpus cache time; real-cache diffusion, generation and scoring benchmarks; viable number of seeds within 72 hours. Benchmark scoring covers generated text as well as real references, since generated text can tokenize much longer.
- Scientific validation/test results, language quality, and human review. No results are invented in advance. Failed gates or exhausted deadlines produce an honest partial outcome.
- Agreement that the discrete pipeline consumes the exact exported passage rows. The original file split is preserved now; passage-level alignment cannot be claimed for an unseen external pipeline.

The first allocation repeats GPU/resume/scorer/tokenizer checks before expensive training. `completion.json` distinguishes a complete run from a partial or failed run; Slurm success alone is insufficient evidence. The transfer archive is independently read back and verified by `package.py`; its final record is `transfer/bundle-verification.json`.
