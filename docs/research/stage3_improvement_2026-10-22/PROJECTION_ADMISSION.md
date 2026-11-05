# Terminal latent projection: do not repeat in this bounded stage

The prior primary metadata identifies diffusion step50000, AE step30500, DDPM, and decoder-only projection: projection250 metadata (`outputs/cluster-results/inference-ablation-59061/projection-250/metadata.json` in the research repository, not included here). Its AE hash is `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`; its diffusion hash is `09a888768393ee379690d28b164771d2fa818cbde7084b8dd16617744a3a604f`, an older model, not a final multi-seed benchmark checkpoint.

The existing verified audit.json (`outputs/nchlt-afrikaans/ablation-59061-audit-20260926/audit.json` in the research repository, not included here), `arms.projection-50` and `arms.projection-250`, reports:

| Projection/control external PPL ratio | Point estimate | Paired sample-bootstrap95% interval | Samples per arm |
|---|---:|---|---:|
| DDPM50 | 1.0115 | 0.8753–1.1572 | 20 |
| DDPM250 | 1.0241 | 0.8401–1.1960 | 20 |

Neither point estimate improves PPL; neither interval excludes1. This is insufficient evidence for improvement or for equivalence/no possible benefit. Mean final norms of7.7748 and7.7664 do not describe the full norm distribution. These are previously exposed exploratory results, not new validation trials. The previous audit description (`outputs/nchlt-afrikaans/ablation-59061-audit-20260926/AUDIT.md` in the research repository, not included here) records source-score/checkpoint checks; this stage read that audit and primary metadata but did not rescore or rerun all trajectories.

A filename inventory including ignored files under `outputs/longrun-local-inputs` and `outputs/longrun-local-evaluation` found checkpoint weights and encoded-data caches, but no saved generated latent banks. The archived generation function (`outputs/longrun-local-inputs/tools/longrun_training.py:245` in the research repository, not included here) saves decoded text and termination flags rather than latent tensors or latent norm summaries. Encoded training/validation caches cannot establish the distribution of sampled diffusion latents. The test-latent filename was visible in inventory; its contents were not opened.

**Decision:** defer fresh final-model sampling/projection in Stage3. The existing negative pilot and the absence of new final-model mismatch evidence do not justify another lower-priority intervention under the author's bounded scope. This does not claim the final models have matching norms, nor does it claim the earlier proposed1% norm gate has been tested/passed. A future study could measure those distributions; no such study is initiated here. The only new treatment in this stage is the fixed decoder-constraint experiment.
