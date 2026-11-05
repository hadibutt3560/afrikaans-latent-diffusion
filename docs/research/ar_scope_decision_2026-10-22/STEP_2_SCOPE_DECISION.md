# Step 2: AR comparison scope decision

> **Superseded recommendation:** The recommendation below to narrow the dissertation before reconciling budgets was premature. Read [STAGE_2_COMPLETED.md](STAGE_2_COMPLETED.md) for the corrected decision and verified budget reconciliation. This earlier document is retained as decision history, not the current scope instruction. No scope amendment was approved.

Status: technical assessment and recommended scope completed; supervisor agreement pending. This document proposes an amendment; it does not record approval or replace the submitted research questions. No training or additional inference is part of Step 2.

## Recommended decision

Complete the dissertation as a representation-centred diagnostic study of unconditional Afrikaans generation, retaining the completed five-seed-per-configuration diffusion benchmark and the verified historical AR model as a contextual comparator. Do not commission additional AR training for this scope. Retain the original matched-AR question and its 20% criterion visibly, with status **not established under the proposed matched design**.

This recommendation is based on evidence and bounded completion value, not an assumption that cluster access is absent. RTX 3060 and cluster access are confirmed; allocation, queue, deadline and training budget are not. Extra compute alone would not eliminate pretraining asymmetry or the reconstruction limitation. Negative and qualified findings are legitimate outcomes; a missing matched comparison must be described as missing evidence rather than a failed experiment.

## Evidence supporting the decision

1. **The historical AR checkpoint is usable.** Run `ar-baseline-57759`, selected step 60000, has 67,218,176 parameters and strict-load/tensor-identity verification. Its exported weights SHA256 is `ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076`. Reproduced selection-subset CE differs from the recorded value by approximately 0.000032; full-validation CE is 3.594901. There is no technical reason to retrain merely to replace a supposedly broken checkpoint. Sources: [AR suitability audit](../ar_validation_audit_2026-10-22/AR_SUITABILITY_REPORT.md), primary preflight (`outputs/ar-validation-audit-20261022/preflight.json` in the research repository, not included here).
2. **The existing comparison is informative but unmatched.** Equal-size validation banks give external mGPT PPL 82.1786 for the single historical AR, versus means 53.0745 and 50.0238 for the small and large diffusion configurations. Their descriptive ratios are 0.64584 and 0.60872. Different lengths and generation procedures affect interpretation; the small diffusion model has worse PPL than AR in both shorter word-length bands despite its aggregate advantage. Source: primary analysis (`outputs/ar-validation-audit-20261022/analysis.json` in the research repository, not included here), keys and exact tables documented in the suitability audit.
3. **Reconstruction is a substantive limitation.** Final AE job56531, step30500, retains source-specific information, but only 11/81 diagnostic passages reconstruct exactly and 21/33 numeric strings are retained. AI-assisted inspection identifies meaning changes even without truncation. This does not prove a formal adequacy threshold failed, because such a threshold is not operationally established; it prevents claiming adequacy has been demonstrated. Source: [Step 1 report](../reconstruction_audit_2026-10-22/RESULTS.md), primary analysis (`outputs/reconstruction-audit-20261022/analysis.json` in the research repository, not included here).

## What is and is not matched

| Dimension | Historical AR | Completed diffusion | Permitted interpretation |
|---|---|---|---|
| Data identity | Same frozen NCHLT training/validation split hashes | Same split hashes | Corpus identity is matched; exposure is not. |
| Tokenizer | Inherited mT5 vocabulary, 250100 entries | Same inherited tokenizer | Encodings matched on all 5905 validation passages; neither tokenizer was trained solely on this corpus. |
| Pretraining | Random generator weights | Pretrained mT5 encoder/decoder plus downstream adaptation | Different inherited knowledge; not a pure generative-mechanism comparison. |
| Model capacity | 67.218M total, four layers/width256 | 8.098M or 187.978M denoiser plus shared AE/decoder | Denoiser-only counts are not full-system counts. Full AE trainable/total/inference-active counts are not newly established by this decision and must not be invented or treated as matched. |
| Training view | Up to128 source tokens plus BOS; effective batch8, BF16 | AE view128; denoiser trains on compressed representations, effective batch up to64, FP32 | Similar source view, different objectives and optimisation. |
| Exposure/selection | Selected step60000, completed157529; 128-passage probe checked630 times | Five seeds each; maximum381 epochs; 2048-passage probe every two epochs, early stopping; different realised selected steps | Neither data exposure nor validation opportunity is matched. AE adaptation is additional diffusion-system exposure. |
| Compute | Historical allowance16266 seconds based on an earlier diffusion run; RTX PRO6000 Blackwell | Later multi-seed capacity benchmark, plus AE adaptation | Old wall-time allowance does not match the final suite or FLOPs; pretraining excluded historically. |
| Decoding | BOS-only top-p0.95, temperature1, one beam, 127 new-token cap | Unconditional latent sampling followed by constrained beam4 decoding, cap128 | Both unconditional but different system generation procedures. Equal token cap is not equal decoding or realised length. |
| Replication | One training seed; 256 fresh samples | Five training seeds per configuration; 256 saved validation samples per seed in contextual table | Sample bootstrap cannot replace AR training-seed replication. |
| Scoring | Pinned mGPT, FP32, context512/stride256 | Same scorer and function | External scoring is aligned, but it does not establish linguistic quality. |

Primary design sources: historical AR protocol (`outputs/cluster-results/ar-baseline-57759/protocol.json` in the research repository, not included here), AR summary (`outputs/cluster-results/ar-baseline-57759/summary.json` in the research repository, not included here), locked diffusion plan (`outputs/longrun-local-inputs/plan.json` in the research repository, not included here), methodology evidence (`docs/research/tacl_methodology_2026-10-22/METHODOLOGY.md` in the research repository, not included here), and the AR suitability report. Full-system parameter accounting remains a reporting item, not grounds to call this comparison matched.

## Scope and claim boundary proposed for approval

- Retain RQ1's implementation-control account and its documented deviations; this AR decision does not resolve those deviations.
- Answer RQ2 with measured input dependence and qualified reconstruction fidelity. No claim of adequate linguistic preservation based solely on token accuracy or overlap.
- Keep original RQ3 and the original hypothesis in the proposal-to-study mapping. State that the matched-AR component remains incompletely tested.
- Add the following explicitly amended question only if approved: **“Under the implemented representation and generation pipeline, how do the two diffusion configurations differ across training seeds, and what contextual automatic-evaluation differences are observed relative to the available historical AR model?”** This concerns implemented systems; it does not establish intrinsic diffusion superiority.
- Report the completed diffusion benchmark separately from the exploratory validation-only AR and reconstruction diagnostics. Preserve all historical test exposure. Do not perform a new test evaluation merely to make the contextual comparator look final.

The original provisional criterion is generated-text mGPT PPL no more than 20% above a matched trained AR baseline, alongside adequate reconstruction and no severe degeneracy. Here the numerical ratio is below1.20 **only for the unmatched validation comparison**. Matching is absent; reconstruction adequacy is not established; absence of severe degeneracy is not established by EOS or a small AI review. Consequently the composite criterion is **not demonstrated**, not “passed” and not a formally established rejection. New matched AR runs alone would not settle the other conditions.

## If the supervisor requires the original matched comparison

Do not silently relabel the historical comparison. A new prospective design is necessary unless equivalent independently trained artifacts are recovered. Minimum requirements for an acceptable implementation-focused comparison are:

1. Agree one matching target: recommend **maximum downstream data exposure and validation opportunity**, with a common frozen corpus/view, 381-epoch ceiling, common selection-probe IDs and every-two-epoch check schedule. Report realised exposure and AE adaptation separately. This would remain an unequal-pretraining system comparison and requires explicit acceptance of that interpretation. If equal compute or equal inherited knowledge is mandatory, a different design is required; the historical suite cannot retroactively satisfy it.
2. Fix architecture and training policy before observing new outcomes. Use at least three new independent AR training seeds, with five preferred only if the predeclared budget supports all five. Historical seed42 remains contextual if the protocol changes. This replication recommendation is not a power guarantee. Use each model's own validation loss; never equate AR likelihood with diffusion MSE.
3. Align evaluation counts, scorer revision and content caps; predeclare generation procedures and a bounded validation selection allowance. Report length, diversity, copying and linguistic limitations alongside PPL. Identical numeric generation seeds across systems do not imply paired semantic outputs.
4. Admit training only after an authorised throughput/learning pilot on the actual allocated GPU, an agreed resource cap and a feasible completion date. No reliable current GPU-hour estimate follows from the old4.52-hour run. No pilot, cluster job or training is authorised by this document.
5. Preserve existing benchmark outputs and explicitly label any subsequent final evaluation given historical test exposure. Do not tune on previously inspected test scores or claim an untouched test.

These are admission conditions for a separately authorised design, not an executable experiment specification. Developing and running that alternative is outside the recommended Step2 scope. If equal pretraining is required, a pretrained AR comparison would also need a justified model, tokenizer/exposure policy and budget; merely adding seeds to the current random AR cannot solve that issue.

## Supervisor decision: ready for review

**Requested decision:** approve the diagnostic scope above, using the existing AR as a contextual comparator and recording original RQ3 as incompletely tested, or require a prospective matched-system comparison with an explicit matching target and resource/deadline allocation.

Suggested text for discussion with the supervisor:

> The historical AR checkpoint has been verified and scored with the same external evaluator as the diffusion models. Its comparison is not matched in pretraining, training exposure, compute, validation selection or decoding. A new frozen-autoencoder diagnostic also shows clear input dependence but substantive reconstruction errors. I propose retaining the completed diffusion-capacity benchmark and representation diagnosis, presenting AR as contextual evidence, and explicitly marking the original matched-AR hypothesis as incompletely tested. May we adopt the amended question above? If the matched comparison must remain essential, please confirm the required matching target and the available time/compute before we prepare new training.

| Decision record | Status |
|---|---|
| Technical recommendation | Narrow diagnostic scope; reuse verified historical AR; no new training for this claim. |
| Supervisor agreement/date | Pending; no approval inferred from “complete Step2”. |
| Original hypothesis changed | No. Original remains visible and incompletely assessed. |
| Training authorised/launched | No / no. |
| Technical Step2 definition of done | Evidence reviewed, mismatch table completed with unknowns explicit, scope and claims specified, conditional alternative and approval request prepared. Complete. |
| Formal scope decision definition of done | Supervisor accepts amendment or specifies required matched design. Pending external decision. |

## Manuscript wording for the recommended route

The autoregressive model was retained as a historical contextual comparator after verification of checkpoint integrity, tokenizer compatibility and validation-loss reproduction. Although the diffusion configurations attained lower aggregate external-language-model perplexity in the equal-count validation comparison, this result does not test the originally proposed matched autoregressive criterion. The systems differ in inherited pretraining, realised data exposure, validation selection, compute, parameter allocation and generation procedures. Furthermore, the reconstruction diagnostic demonstrates input dependence but also content-preservation errors. We therefore do not claim that the original composite hypothesis has been established. **[SUPERVISOR AGREEMENT PENDING: the amended diagnostic scope and question must be accepted before being presented as the approved dissertation scope.]**
