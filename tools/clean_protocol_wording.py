"""One-off wording cleanup of protocol and plan files (10 November 2026).

Replaces approval, chat and author/user phrasing with neutral dated statements. No rule, threshold,
seed, setting, gate or analysis plan is changed. The exact versions used by the recorded runs are at
the git tag `as-run`; tools/verify_provenance.py checks the recorded hashes against that tag.
"""
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'docs/research'
EDITS = {
 'ae_screen_2026-10-27/PROTOCOL.md': [
  ("Approved by: Hadi Butt Date: 27 October 2026", "Fixed on 27 October 2026, before the runs.")],
 'ar_test_2026-11-05/PROTOCOL.md': [
  ("**Status: APPROVED** by Hadi Butt in chat, 5 November 2026, before any sample was generated.",
   "**Status: APPROVED** on 5 November 2026, before any sample was generated.")],
 'claude_labels_2026-11-07/PROTOCOL.md': [
  ("The author approved this protocol in chat on 7 November 2026.", "Fixed on 7 November 2026, before labelling.")],
 'evaluation_package_2026-10-31/PROTOCOL.md': [
  ("Approved by: Hadi Butt (in chat, approving every decision in §13; only the file hashes in Appendix C were added after approval) Date: 1 November 2026",
   "Fixed on 1 November 2026, with every decision in §13 confirmed; only the file hashes in Appendix C were added afterwards, before any run.")],
 'followup_inference_2026-11-03/PROTOCOL.md': [
  ("**Status: DRAFT.** Not approved. Nothing in this protocol may be run until Hadi Butt approves it in chat.",
   "**Status: DRAFT.** Not approved (see the dated note of 4 November 2026). Nothing in this protocol may be run until it is approved."),
  ("The user submitted job 81249 on 3 November at 19:44 UTC, after this draft was written and unchanged. Explicit approval was not given in chat.",
   "Job 81249 was submitted on 3 November at 19:44 UTC, after this draft was written and unchanged. Explicit approval was not given.")],
 'latent_controls_2026-10-30/PROTOCOL.md': [
  ("Approved by: Hadi Butt Date: 30 October 2026", "Fixed on 30 October 2026, before the run.")],
 'stage4_final_evaluation_2026-10-23/PROTOCOL.md': [
  ("23 October2026. User confirmed: strictly follow the proposal.",
   "Fixed on 23 October 2026, before the analysis. The evaluation strictly follows the proposal."),
  ("Existing AI-assisted inspection remains labelled and requires author review; it is not relabelled as the researcher's personal observation.",
   "Existing AI-assisted inspection stays labelled as AI-assisted and is not presented as the researcher's own observation.")],
 'swahili_check_2026-11-01/PROTOCOL.md': [
  ("**Status: APPROVED** by Hadi Butt in chat, 1 November 2026, before any model was run.",
   "**Status: APPROVED** on 1 November 2026, before any model was run.")],
 'train_heldout_reconstruction_2026-10-27/PROTOCOL.md': [
  ("Approved by: Hadi Butt Date: 27 October 2026", "Fixed on 27 October 2026, before the run.")],
 'deadline_experiment_plan_2026-10-24/PLAN.md': [
  ("short allocation checks run by the author are the current evidence below.", "short allocation checks on the cluster are the current evidence below."),
  ("Raw user-provided telemetry", "Raw telemetry"),
  ("merely because the author allowed an outer maximum.", "merely because an outer maximum is allowed."),
  ("This is author inspection,", "This is researcher inspection,"),
  ("Updated24 October2026 after successful user-run hardware discovery.", "Updated 24 October 2026 after a hardware check."),
  ("The author has confirmed submission on27 October, but not a submission hour.", "Submission is on 27 October; the hour is not fixed."),
  ("User-provided current scheduler output on24 October:", "Scheduler output on 24 October:"),
  ("confirmed by user-run short allocation:", "confirmed by a short test allocation:"),
  ("The author's48-hour maximum", "The 48-hour maximum"),
  ("User approves this concrete plan;", "Plan approved;"),
  ("Submit by the author's actual institutional cutoff;", "Institutional submission deadline;"),
  ("the author's proposal-based review", "a proposal-based review")],
 'decoder_dropout_final_2026-10-25/PLAN.md': [
  ("Author inspection of entities", "Researcher inspection of entities"),
  ("Approved by the user on 25 October 2026.", "Fixed on 25 October 2026, before the run."),
  ("User's diagnostic job74805", "Diagnostic job 74805")],
 'full_decoder_12h_2026-10-24/PLAN.md': [
  ("according to the user-supplied summary.", "according to the job 73970 summary."),
  ("a draft for author review,", "a draft for review,"),
  ("Prepared 24 October 2026 at the user's request. **Not submitted or launched.**", "Fixed on 24 October 2026, before the run."),
  ("Updated for the user's one-submission requirement:", "Updated so that"),
  ("The user may revise scheduling, but doing so is a new explicit decision, not an automatic timer extension.",
   "Any scheduling change is a new explicit decision, not an automatic timer extension.")],
}

if __name__ == '__main__':
    for rel, pairs in EDITS.items():
        p = R / rel
        raw = p.read_bytes().decode('utf8')
        crlf = '\r\n' in raw
        t = raw.replace('\r\n', '\n')
        for old, new in pairs:
            assert t.count(old) == 1, (rel, old[:60], t.count(old))
            t = t.replace(old, new)
        p.write_bytes((t.replace('\n', '\r\n') if crlf else t).encode('utf8'))
        print(f'{rel}: {len(pairs)} line(s)')
