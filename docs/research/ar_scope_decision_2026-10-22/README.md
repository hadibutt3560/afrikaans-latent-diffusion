# AR budget reconciliation (22 October 2026)

These files reconcile the recorded training budgets of the AR baseline (`ar-baseline-57759`) and the final diffusion runs.

- `reconcile_budgets.ps1`: reads the AR protocol, summary, selection and training log, the diffusion run plan and status files, and the older diffusion run metadata, and writes the three files below.
- `budget_reconciliation.json`: AR budget and timed phase, selected and final steps, passage draws, GPU and precision; per-run diffusion steps, epochs and loop times; caveats on what the times include.
- `diffusion_budgets.csv`: the per-run diffusion figures for small and published seeds 42–46.
- `budget_sources.json`: paths and SHA-256 hashes of the 16 source files read.
