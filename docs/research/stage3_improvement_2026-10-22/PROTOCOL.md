# Stage 3: bounded decoder improvement investigation

Frozen before treatment inference. Scope: one new one-factor decoder test; an artifact-only decision on whether to repeat terminal latent projection. No training, new tokenizer, new sampler, test evaluation or decoding grid. The completed benchmark remains unchanged.

## Decoder experiment

Hypothesis: prohibiting repeated token trigrams obstructs faithful reconstruction. Use Stage1's final AE job56531/step30500, archived source and exact81 diagnostic validation passages across64 documents. Use the fixed103-passage/64-document reserve only if the diagnostic gate passes. Validation has been inspected previously; the reserve is procedural, not an untouched dataset.

Baseline: FP32, beam4, max_length128, min_length5, repetition_penalty1.2, length_penalty1, deterministic decoding, no_repeat_ngram_size3. Treatment changes only no_repeat_ngram_size to0. Batch1. Reproduce all baseline output token sequences against Stage1 before interpreting treatment effects. Same encoded latent for both arms; no teacher-forced rescoring is needed because this decoding-only change does not affect that quantity.

Primary endpoint: mean sentence chrF (SacreBLEU2.6.0, char_order6,word_order0,beta2). Paired recipient-document bootstrap10000, seed91428. Also report exact matches, SequenceMatcher, numeric-string precision/recall, EOS/caps and repeated word3/4-grams. Inspect every changed source/baseline/treatment triple for omissions, unsupported additions and clause/negation changes; explicitly AI-assisted, not human ratings. Unchanged triples inherit the Stage1 inspection. Report uncertainty conditional on one AE and selected validation documents, not AE-training uncertainty.

Advance only if mean chrF improves by at least2 points, its paired95% interval is entirely positive, numeric recall does not fall by more than2 percentage points, and inspection reveals no material content-preservation regression. If the automatic gate fails, stop without consuming the reserve or testing unconditional transfer. If the automatic gate passes but content assessment is ambiguous, label inconclusive rather than assuming a pass. A passing candidate is confirmed once on the fixed reserve with the same gates before any claim of reconstruction improvement. Unconditional transfer is a separate gated check on fixed small/large seed42 latent banks under the earlier bounded experiment plan; never infer generation improvement solely from reconstruction.

Estimated diagnostic resource use:5–15 minutes on RTX3060 including loading, below4GiB allocated memory based on Stage1 measurements; estimate only. Stop at2 GPU-hours,10GiB allocated memory, nonfinite values or baseline/identity mismatch. Save all new artifacts separately and hash protocol, source and parent outputs.

## Terminal projection admission

Review the primary audit of inference-ablation59061. It already tested decoder-only norm projection at50 and250 DDPM steps. Do not repeat this negative intervention without new final-model mismatch evidence. Search available final-model artifacts for stored sampled latents/norm summaries; absent those, report that final-model norm mismatch is unmeasured and defer new sampling/projection. Absence of cached evidence is not proof that norms match. This is a deliberate resource-priority decision, not a claimed successful norm gate. This Stage3 task does not require running every optional candidate in the earlier programme.

Completion: baseline reproduced, treatment analysed against the frozen gate, all changed outputs inspected, projection admission documented, result and manuscript paragraph saved. Failure to improve is a completed outcome. Do not proceed to final evaluation or dissertation-wide edits.
