# Diffusion autocast repair — 10 October 2026

Job 66417 passed resume equality checks, but its diffusion smoke gradients were zero. Reproduced with PyTorch 2.7.1 CPU and BF16: the no-grad self-conditioning forward populates autocast's weight cache, suppressing gradients through the later training forward. With the cache disabled, gradients and output-weight updates are nonzero. The original checks were insufficient: matching resumptions do not prove learning.

This repair disables autocast's weight cache only for diffusion loss computation. It retains BF16, self-conditioning, all scientific configuration, data and split membership. The running AE process and AE computation path remain unchanged. No test-set outcomes motivated this repair.

The installer requires the exact original checksums and refuses after benchmarking or downstream training has started. Original sources are backed up. Original bundle-manifest.json, contract.json, checkpoints and the original transfer tar are retained. runtime-patch.json records old/new hashes and installation-stage evidence; patched and original source files are included in the results provenance archive. The resume verifier checks both the baseline and recorded repair. Diffusion checkpoint contracts include the new implementation identity.

Before benchmarking each diffusion architecture, the existing benchmark stage now repeats real-model continuous/resumed checks, requires changed output weights, and verifies model/optimizer/EMA/RNG/scheduler equality. Every diffusion optimizer step rejects a zero gradient. Checks remain inside the original 30-minute benchmark budget and 72-hour allocation; failure stops the workflow with a partial archive. Inspect results/diffusion-learning-check.json. This on-cluster repaired check is not yet verified locally on cluster hardware.

Install from the login node with system Python 3 (standard library only):

    python3 /datasets/hbutt/swahili72/hotfix-20261010/install.py --root /datasets/hbutt/swahili72/package --out /datasets/hbutt/swahili72/results

Keep job 66417 running; do not submit another job for this repair. If the installer refuses because stages advanced, preserve the output and inspect before attempting recovery. For a later interruption after a successful repair, the existing job.slurm --resume command remains valid and checks the repair receipt.
