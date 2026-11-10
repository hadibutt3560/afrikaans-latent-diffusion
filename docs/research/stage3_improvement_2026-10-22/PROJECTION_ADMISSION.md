# Terminal latent projection: deferred in Stage 3

The earlier decoder-only projection run (inference ablation 59061; autoencoder hash `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`, diffusion hash `09a888768393ee379690d28b164771d2fa818cbde7084b8dd16617744a3a604f`, step 50000, an older model than the final benchmark checkpoints) was read from its verified audit (`outputs/nchlt-afrikaans/ablation-59061-audit-20260926/audit.json` in the research repository); nothing was rescored. Neither point estimate improves perplexity and neither interval excludes 1. Mean final norms were 7.7748 and 7.7664. These are earlier exploratory results, not new trials.

| Projection/control external PPL ratio | Point estimate | Paired sample-bootstrap 95% interval | Samples per arm |
|---|---:|---|---:|
| DDPM50 | 1.0115 | 0.8753–1.1572 | 20 |
| DDPM250 | 1.0241 | 0.8401–1.1960 | 20 |

No saved generated latent banks or latent-norm summaries exist for the final models (the generation function saves decoded text only), so any final-model norm mismatch remains unmeasured and the proposed 1% norm gate was not tested. Fresh projection work was deferred; this makes no claim that projection cannot help.
