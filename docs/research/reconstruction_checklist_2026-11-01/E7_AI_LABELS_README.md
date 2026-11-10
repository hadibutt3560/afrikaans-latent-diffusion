# E7 labels: provenance

`E7_ai_labels.csv` holds the earlier labels: 26 of 81 passages pass all three checks, and 55 fail at least one.

**Superseded on 7 November 2026.** The manuscript now reports a relabelling by Claude Opus 5.5, in `../claude_labels_2026-11-07/`: 19 pass and 62 fail. That relabelling agrees with these labels on 74 of 81 passages (κ 0.79). These labels are kept only for the agreement comparison.

## Who produced them

- **Labeller:** an AI assistant working outside this repository. The author added its output to this repository on 1 November 2026. The model and version were not recorded.
- **What it saw:** each source passage beside its reconstruction, taken from `E7_reconstruction_checklist.xlsx`. It knew which text was the source.
- **Not verified:** neither the author nor an Afrikaans reader has checked the labels. Report them as an AI-assisted reading, never as human evaluation.

## Columns

- **Checks 1–3:** the AI labeller's answers, in the form Yes = pass. Its free-text notes were not copied into the CSV.
- **`strict_rule_check1`:** check 1 re-marked under a stricter rule. Any change to a named entity, number or key noun fails. The rule changes items 7, 9, 29 (Ministerie → Minister), 10, 16, 45 and 68. Under it, failures rise from 55 to 62.

## If the author reviews the labels

Record the reviewed labels in a new file, with the date and the reviewer. Then update the manuscript wording from "AI-assisted" to "AI-drafted, checked by the author".
