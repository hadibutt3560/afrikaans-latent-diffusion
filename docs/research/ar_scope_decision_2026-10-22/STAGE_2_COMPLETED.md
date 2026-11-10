# Stage 2 completed: retain the existing AR comparison with explicit budget limits

## Decision

Keep the existing trained AR baseline and retain the original research question. Do not automatically narrow the dissertation or launch replacement training. The AR is the promised *type* of baseline, and it already supports a useful empirical comparison. The remaining limitation is how closely the implemented conditions satisfy the proposed matching requirement; it is not absence of a baseline.

Use the final **small diffusion configuration as the primary resource-context comparison** with AR, and large diffusion as a separate higher-cost comparison. This is a retrospective reporting choice based on the audited resource records, not a prospectively budget-matched experiment or a new model-selection result. Report all five small and all five large seeds; do not pick the best seed or mix validation results with historical test-bank results.

No research question has been rewritten, no hypothesis silently relaxed, and no supervisor approval assumed. A qualified answer to the original question is permissible; claiming the original composite criterion has been demonstrated is not supported by the present evidence.

## What the proposal actually required

Submitted proposal (`handoff_2026-10-20/github-research-diffusion/proposal/submitted_proposal.pdf` in the research repository, not included here), §3.4–3.5, PDF p17, specifies a lightweight GPT-style autoregressive Transformer with the same Afrikaans split, SentencePiece BPE tokenizer, maximum sequence length and compute budget. It does not specify an exact GPT architecture, a particular AR parameter count or a mandatory number of AR training seeds in these clauses. Its hypothesis uses a provisional 20% external-mGPT-perplexity tolerance plus adequate reconstruction and no severe degeneracy (PDF pp12–13).

The implemented baseline is randomly initialized, four layers, width256, four heads and 67,218,176 parameters. Its validity is established in the [AR audit](../ar_validation_audit_2026-10-22/AR_SUITABILITY_REPORT.md). Both systems share the actual NCHLT splits and inherited mT5 tokenizer. Replacing the proposed custom BPE with that tokenizer is a common methodological deviation requiring justification; it is not a tokenizer mismatch between the compared systems. AR adds a causal BOS position to the source-token view; generation caps were explicitly reconciled in the earlier validation diagnostic.

More training seeds, equal parameter counts and identical decoding are desirable design questions, not additional proposal obligations established by the cited clauses. Different pretraining and decoding must be disclosed because they restrict interpretation; they do not invalidate the existence or usefulness of the trained baseline.

## Reconciled primary records

Derived values and exact source hashes are saved in [budget_reconciliation.json](budget_reconciliation.json), [diffusion_budgets.csv](diffusion_budgets.csv) and [budget_sources.json](budget_sources.json). Reproduce using [reconcile_budgets.ps1](reconcile_budgets.ps1), which reads logs/configurations on CPU and writes only derived files in this documentation folder.

| Run/group | Recorded training-loop time | Selected checkpoint | Exposure at selected checkpoint |
|---|---|---|---|
| AR `ar-baseline-57759` | 16277.03s = 4.521h; declared allowance16266s | step60000; final run step157529 | 480000 passage draws, approximately10.05 corpus-equivalent draws |
| Small diffusion seeds42–46 | 2.930, 3.126, 3.115, 3.100, 3.122h (rounded; see CSV for exact values) | selected epochs288,306,304,302,306 | Complete epoch permutations, 288–306 corpus passes |
| Large diffusion seeds42–46 | approximately7.19–7.85h | selected epochs180,176,176,174,172; stopped at192,178,184,186,180 | 172–180 corpus passes at selection |

AR samples with replacement; its corpus-equivalent draws are not complete coverage passes. At run end it had1260232 draws, approximately26.38 corpus equivalents. The diffusion epoch accounting uses47767 training records and747 updates per epoch, including a smaller final batch; it does not multiply every update by64 and overcount the last batch.

**Timing definitions matter.** AR's timer includes training, periodic validation and checkpointing after preprocessing; its final evaluation/export/sampling raises the recorded total to16336.73s. The selected-step log reads6292.06s (about1.748h), but that timestamp precedes the step60000 validation/save operation. Finding and selecting the best model required the whole4.52-hour run; report both facts, not just the cheaper selected-step timestamp.

Diffusion `status.json.cumulative_seconds` records accumulated loop time, including validation, checkpoint operations and any in-loop sample generation. It excludes initial model/cache preparation and subsequent final evaluation. Timestamp placement can omit the final save's duration. Exact elapsed time to each selected diffusion checkpoint is not established by these status files; do not interpolate it from step fractions. The planned381-epoch estimates in `plan.json` are not measured completed runtimes.

Sources: AR protocol (`outputs/cluster-results/ar-baseline-57759/protocol.json` in the research repository, not included here), summary (`outputs/cluster-results/ar-baseline-57759/summary.json` in the research repository, not included here), training log (`outputs/cluster-results/ar-baseline-57759/training_metrics.jsonl` in the research repository, not included here); diffusion `outputs/longrun-local-inputs/runs/{small,published}-seed{42,43,44,45,46}/status.json`; archived loop implementation (`outputs/longrun-local-inputs/tools/longrun_training.py:309` in the research repository, not included here) and AR loop (`tools/train_afrikaans_baseline.py:163` in the research repository, not included here).

### The older budget link is real, but does not establish equality

The AR protocol explicitly derives its16266-second allowance from the reported elapsed time of earlier diffusion job57231, associated with `diffusion-50k-retry-20260920`. Thus a budget-matching attempt was made; it was not invented after scoring.

However, the associated diffusion run metadata (`outputs/cluster-results/diffusion-50k-retry-20260920/run_metadata.json` in the research repository, not included here) lists Quadro RTX8000 devices, while AR's protocol records an RTX PRO6000 Blackwell. The older diffusion configuration uses one device and resumes from another checkpoint directory. Visible device count is not evidence of two-GPU training. Original job57231 scheduler accounting and full preceding-resume costs were not located in this bounded audit. Therefore the metadata prevents asserting verified same-hardware, complete-run cost equality; it does not establish which hardware ran every phase of that historical job. The recorded allowance is traceable, but not a calibrated compute match.

The **final** diffusion suite's plan (`outputs/longrun-local-inputs/plan.json` in the research repository, not included here) records the same GPU model as AR: RTX PRO6000 Blackwell. Final diffusion uses FP32; AR uses BF16. Same GPU model is not proof of equal utilisation or FLOPs. Small diffusion's recorded loop cost is lower than AR's, while large diffusion's is higher. Frozen-AE adaptation, latent caching, inherited pretraining and evaluation remain additional cost categories, so this does not establish lower total pipeline cost or greater computational efficiency.

## What the existing result supports

Use the already-scored equal-count **validation** banks:256 AR outputs and256 per diffusion training seed. The primary AR cap127 bank has pooled external mGPT PPL82.1786; small diffusion has mean53.0745 across five seeds; large diffusion has mean50.0238. Sources: comparison.csv (`outputs/ar-validation-audit-20261022/comparison.csv` in the research repository, not included here) and analysis.json (`outputs/ar-validation-audit-20261022/analysis.json` in the research repository, not included here). Scoring uses the same pinned mGPT revision `40897bd7c8b47a76802c411108ca6220438b8b40`.

Supported statement: **the implemented small diffusion system attained lower aggregate external PPL than the verified AR baseline in the reported validation banks, with lower recorded denoiser-loop time than AR-loop time on the same GPU model.** Add the cost exclusions, precision differences and uncertainty limits whenever using this statement. The AR is a single training run; diffusion variation across five runs and AR sample-bootstrap uncertainty are different quantities. Numeric generation-seed alignment does not create cross-system semantic pairs.

Unsupported statements: equal-FLOP comparison; equal end-to-end compute; matched inherited knowledge; intrinsically better diffusion; better linguistic quality; or passing the original composite hypothesis. Output lengths differ and small diffusion's aggregate PPL advantage reverses in the two shorter length bands. Reconstruction is limited according to Stage1; no new adequacy or degeneracy conclusion is manufactured here.

## Is additional AR training necessary?

**No, for reporting this existing comparison and a qualified answer to RQ3.** The checkpoint is valid and the missing budget interpretation is now documented. There is no automatic justification for retraining or for three/five new seeds merely to satisfy a requirement the proposal did not explicitly state.

**Possibly, only if the final claim must assert a strictly matched compute-budget experiment.** The completed records cannot retroactively supply that control. First obtain an agreed accounting boundary (generator training only versus AE-inclusive system cost), hardware/precision policy and budget. Additional AR seeds alone would not fix the boundary or pretraining differences. Do not choose a new budget, retrieve a diffusion checkpoint or tune decoding based on which comparison produces the preferred PPL ratio.

The supervisor-facing issue is therefore **acceptance of disclosed implementation deviations and a qualified interpretation**, not automatic abandonment of RQ3. The current evidence supports the numerical part of the criterion descriptively (ratios0.64584 and0.60872), but the exact matched-design and reconstruction/degeneracy conditions remain unestablished. Report this as a limitation of the evidence, without declaring either full success or a definitive rejection.

## Manuscript-ready correction

We trained a lightweight GPT-style autoregressive baseline on the same Afrikaans split and inherited mT5 tokenizer as the latent-diffusion pipeline. Its training allowance of16266 seconds was based on an earlier diffusion job, although available metadata do not establish calibrated hardware or complete-run cost matching for that historical pair. The later small diffusion runs required approximately2.93–3.13 recorded loop hours each, compared with4.52 hours for AR, on the same recorded GPU model but at different numerical precisions. These times exclude the diffusion system's autoencoder adaptation and initial latent preparation. In equal-count validation generation banks, small diffusion had lower mean external mGPT perplexity than AR. We interpret this as evidence about the implemented systems under disclosed resource conditions, rather than an equal-compute or pretraining-controlled demonstration of generative-mechanism superiority. The original composite criterion remains unestablished because its matching and reconstruction/degeneracy conditions are not fully demonstrated.

## Completion record

Stage2's technical work is complete: proposal obligations checked, AR identity retained, older/final budgets distinguished, all ten final diffusion timings reconciled, exposure and timing boundaries documented, current comparison designated and earlier scope recommendation superseded. No model inference, training, paid compute, test evaluation or historical-output changes were performed. Supervisor review of the reported deviations remains part of dissertation approval; it is not represented as already obtained. No Stage3 work is included.
