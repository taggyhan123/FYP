QUARANTINED ARTIFACTS — concurrent-client contamination
=======================================================
Cause: a second Claude Code session (`claude --resume`, pid 30496) ran its own
copy of this runbook against the SAME vLLM server (127.0.0.1:8000) and the SAME
result directory. Up to three audit clients ran at once. This violates the
runbook rule "Use one isolated replay client per server."

Quarantined here (NOT used for any reported result):
  original.json     validation.clean=false
                    cached_plus_computed_matches_prompt_per_request=false
                    measurement_consistent=false
  alphabetical.json written concurrently by two processes; raced
  random.json       written during the concurrent window
  *.err             stray stderr files from the racing drivers

Resolution: competing processes terminated (user-authorised, all owned by
taghan; no other user's process signalled). All 7 k64 audits rerun afterwards
with a single client. Only the reran audits are reported.

Also preserved here: k4-original-trial-1.json / .err — this session's
calibration replay at ~22:33, which returned counter_validation.clean=false
because two clients shared the counter window. The other session moved it here
and reran the trial cleanly at 22:34:58. The contaminated artifact IS preserved;
the reported k4/original-trial-1.json is the clean rerun.

k64-audit-frequency-orphan-run.json
  Reason: produced by an audit process that was orphaned when the harness
    killed its parent shell. Superseded by a clean rerun of the same
    condition. Preserved for completeness; not used.

k64-audit-original-contended.json / .stdout
  Reason: validation.clean=false
          cached_plus_computed_matches_prompt_per_request=false
          (tokenize_count_matches_completion_usage was true)
  Cause: this audit ran while harness-killed and orphaned audit processes
    were still issuing traffic to the same server, so the per-request
    /metrics window before and after each completion absorbed a neighbouring
    process's prefill work. This violated the one-client-per-server rule
    through no fault of the method.
  Evidence: once the sweep was run detached and undisturbed, all six other
    conditions validated per-request exactly (clean=true), and the rerun of
    original also validated clean.
  Action: preserved here and rerun into the declared path on an isolated
    server, per section 6 and the audit script's own instruction.
    Conditions unchanged.

parallel-session-pressure/
  Reason: PROVENANCE, not a measurement fault.
    These six section-7 pressure outputs (and their QUARANTINE.txt) appeared
    in this result directory between 01:15 and 01:36 on 2026-08-06 while this
    executor was idle waiting on the section-6 audits. This executor did not
    launch them. A second Claude session is running on this host with cwd
    /home/taghan/FYP (parent pid 3913631), which is the likely author.
  Verified independently by this executor before setting them aside:
    all 24 regime-runs report requirement_met=false, peak KV usage in
    0.036463-0.036882 against a required 0.90, and both counter_validation
    sub-checks true. Their stated cause is consistent with the raw data.
  Action: preserved unmodified. Section 7 was then re-executed by this
    executor into the declared paths so the handover reports only runs this
    executor produced. Conditions and the 0.90 threshold unchanged.
