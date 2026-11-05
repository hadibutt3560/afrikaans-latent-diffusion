# AR baseline on the test split: common-split check of the 1.20 rule

**Status: APPROVED** by Hadi Butt in chat, 5 November 2026, before any sample was generated. Any later change is made only by adding a dated note at the end.

## 1. Purpose

The proposal's RQ3 rule says diffusion's mGPT perplexity must be at most 1.20 × the AR baseline's. So far, AR has been scored only on 256 validation samples (perplexity 82.18), while every diffusion system has 1,000 test samples per run. This check puts AR on the test split, so every system can be compared on one split. It covers:
- the original diffusion (B0 autoencoder, 5 runs per size);
- the retrained diffusion (D autoencoder, 3 runs per size);
- human test text.

Inference only. No training, retuning or checkpoint reselection. No test text is read, except the already-scored human test references.

## 2. Fixed inputs

- **AR model:** `outputs/cluster-results/ar-baseline-57759/best_model`, the selected update-60,000 checkpoint. Its weights must hash to `ed12edbd…1d076`, the value recorded by the 22 October audit.
- **Sampling:** exactly as in that audit's `ar-cap127` arm, which gave the paper's 82.18:
  - start token 0;
  - nucleus sampling with p = 0.95, top-k off and temperature 1;
  - up to 127 new tokens, no repetition constraints;
  - FP32, batch one.
- **Seeds:** sample i uses seed 120000 + i, for i = 0…999. The same seed numbers were used for diffusion test sampling, but the generators are unrelated, so this pairs nothing. It only keeps the banks the same size.
- **Scorer:** the frozen `compare_afrikaans_generators.score_text` from the evaluation bundle, with mGPT at its pinned revision. Before AR is scored, it must reproduce stored values on four anchor rows to within a relative 1e-4.

## 3. Comparison banks (already on disk)

| Bank | Location |
|---|---|
| Human test (1,000) | `outputs/longrun-local-evaluation/test-reference-mgpt.json` |
| B0 diffusion, small and large, seeds 42–46 | `outputs/longrun-local-evaluation/runs/*/test-mgpt.json` |
| D diffusion, small and large, seeds 42–44 | `outputs/cluster-results/ae-diffusion-77176/screen/diffusion/runs/*/test-mgpt.json` |

## 4. Readouts

**For every bank:**
- token-weighted raw perplexity;
- length-standardised perplexity (weights 0.140, 0.223, 0.349, 0.288; at least 20 outputs per band, otherwise not reported);
- mean words and distinct-2;
- exact training copies;
- share of word 4-grams found in the NCHLT training split.

**Ratios to AR:** for each diffusion system, the ratio is the mean of its run perplexities divided by AR's perplexity, for both raw and standardised perplexity.

**95% interval:** on the log scale.
- The diffusion part is the run-to-run variance divided by the number of runs.
- The AR part is the bootstrap variance of AR's log perplexity, from 2,000 resamples of outputs with seed 20261105.
- The interval uses t with (runs − 1) degrees of freedom.
- Like the paper's existing intervals, it omits AR and autoencoder training variation.

## 5. Decision wording (fixed now)

**Passes the 1.20 rule on test:** the point ratio is at most 1.20.
- **Clearly passes:** the upper interval limit is also at most 1.20.
- **Clearly fails:** the lower limit is above 1.20.
- **Inconclusive:** anything else.

**Required wording:** whatever the outcome, the paper keeps the existing qualification. This compares systems, not generation mechanisms: the pretrained decoder with beam search against an AR model trained from scratch with nucleus sampling. The test split was exposed earlier, so the check is not fresh confirmation.

## 6. Outputs

Everything goes to `outputs/ar-test-20261105/`:
- `contract.json`;
- `anchors.json`;
- `ar-test-samples.jsonl`;
- `ar-test-scored.jsonl`;
- `analysis.json`.

## Dated notes
