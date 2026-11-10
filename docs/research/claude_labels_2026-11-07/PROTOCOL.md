# Claude labels: protocol (fixed 7 November 2026, before labelling)

The author approved this protocol in chat on 7 November 2026.

## Labeller

- **Model:** Claude Opus 5.5 (Anthropic, model ID `claude-opus-5-5`), running in Claude Code.
- **Not blind:** before labelling, the labeller knew the study aims, the paper's wording and the earlier labeller's total (55 of 81 failing at least one check). It did not open the earlier label file (`../reconstruction_checklist_2026-11-01/E7_ai_labels.csv`) until all 81 labels were saved.
- **Not human evaluation.** No Afrikaans reader checks these labels. Report them as an AI-assisted reading by Claude.

## Task A: 81 reconstructions

- **Input:** the Review sheet of `../reconstruction_checklist_2026-11-01/E7_reconstruction_checklist.xlsx`. Each row has a source passage and the original autoencoder's (B0) reconstruction.
- **Checks:** the three checks from the plan, as defined on the workbook's Instructions sheet. Yes means pass.
  1. Major named entities and content words are preserved.
  2. The main clause structure is recognisable.
  3. There is no obvious hallucinated material.
- **Judging:** judge meaning, not spelling. A changed or dropped number, name, department or key noun fails check 1. A small inflection or punctuation change does not.
- **Exact reconstructions:** these pass all three checks.
- **Result:** Pass if all three checks are Yes. Use Unsure only when no decision is possible, and give a reason. Unsure counts as not passing.
- **Agreement:** after saving, compare against the earlier labels per check and on Pass/Fail. Report raw agreement and Cohen's kappa.

## Task B: screen protocol sample reading

- **Input:** for both retrained denoisers (small and published sizes), seed 42, test outputs at indices 0, 25, 50, 75 and 100. This is the fixed rule in `../ae_screen_2026-10-27/PROTOCOL.md` §7.
- **What to record:** for each output, record:
  - whether it reads as Afrikaans;
  - whether it is grammatical;
  - whether it is coherent;
  - a short note.
- **Status:** this is descriptive only. It makes no quality claim and is not counted in any criterion.

## Outputs

- `labels_81.csv`
- `samples_screen.md`
- `agreement.json`
- this protocol
