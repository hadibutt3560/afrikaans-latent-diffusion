# Retrieved job 74177: inspection (25 October 2026)

`outputs/review-job-74177.tar.gz` was extracted to `outputs/cluster-results/job-74177-review` in the research repository; all 60 files in the artifact manifest passed SHA-256 verification, and six large checkpoints (including `F12-best.pt`) were left on the cluster. Slurm reported COMPLETED, exit 0:0, elapsed 10:09:29, with 26,620 updates per arm (about 4.46 corpus passes) and 8.720 training hours. The selected F12 checkpoint, step 26,520, reached chrF 70.0539 against 67.7005, 13/81 exact against 11/81 and 25/33 numeric matches against 21/33. It failed the development thresholds, so reserve and generation evaluation were skipped and the decision stands; any further evaluation of this checkpoint would need a labelled exploratory amendment.

An AI-assisted reading of the seven development rows (zero-based) whose exact-match or numeric counts changed between B0 and F12 found a modest, mixed gain, not a broad improvement:

- Rows 26, 28 and 73 became exact; row 78 stopped being exact (punctuation and conjunction), a net gain of two.
- The four extra numeric matches are in rows 4 (+1), 20 (+1) and 57 (+2). Row 57 still misstates opening hours; row 20 still changes the sporting category and dates; row 4 still changes the scholarship recipients.
