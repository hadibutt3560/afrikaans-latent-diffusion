# Dated note, 1 November 2026: empty length bands in `compare_generation.py`

Written before any generation output from the new autoencoder exists (job 77175 was still training and job 77176 had not started). It corrects an implementation defect; the protocol's measures, comparisons and decision rule in §7 are unchanged.

## The defect

`run_metrics` computed length-standardised perplexity as `exp(nansum(w_b · NLL_b))`. A run with no outputs in a band had that band's term silently dropped, while the other weights were not renormalised. The result is biased low, so a run with a missing band looked *better*. The local latent-control run of 30 October showed this is a realistic case: one control produced 2 of 1,000 outputs in the 1–10-word band.

## The correction

- Standardised perplexity is computed only when every band has at least 20 scored outputs, the rule fixed in the latent-control protocol (`latent_controls_2026-10-30/PROTOCOL.md` §5). Otherwise it is recorded as `None`.
- If any old run, new run or random-latent control fails that rule, the standardised Welch test is replaced by `status: insufficient_support`, and `improved_by_protocol` is `null`, not `true` or `false`. The report then gives raw perplexity, distinct-2, copies and band counts without a verdict.
- The raw-perplexity Welch test, distinct-2 and copy rules are unchanged.

For reference, the ten existing benchmark test banks have at least 129 outputs in every band, so the correction cannot change their values.
