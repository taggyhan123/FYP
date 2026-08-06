# GPU handover — initial research brief closure run

Execution record only. No scientific conclusions are drawn here; validating the
evidence and deciding whether the brief is closed belongs to the local analysis
session.

## Commit and environment

| item | value |
|---|---|
| repo | `/home/taghan/FYP` |
| branch executed from | `tooltrie-v0-workflow` |
| commit | `65b86bacfbcc227c2b21223682018a7b29dd3e94` |
| pin check | `EXPECTED_FYP_COMMIT` matched exactly |
| tests at that commit | `97 passed` |
| host | `gpuserver`, Linux 6.8.0-136-generic |
| GPU | index 0, RTX 3090, `GPU-87ff24b3-1ffa-2319-0737-a9fc422063fc` |
| vLLM / torch | 0.26.0 / 2.11.0+cu130 |
| model | `Qwen/Qwen3-0.6B` |
| live KV capacity | **190,896 tokens** (block_size 16 x 11,931 GPU blocks) |
| `enable_prefix_caching` | true (verified live, not assumed) |
| result stamp | `20260805-222246` |

The historical capacity figure of 188,912 was **not** reused; capacity was read
back from the running server, as §3 requires.

`NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` and
`cluster/initial-brief-closure-manifest.json` were both read in full before any
command was run.

`AGENTS.md` did **not** exist at the pinned commit `65b86ba`. It was added
afterwards by `917d560` on `tooltrie-v0-workflow` and pulled into this worktree
at 22:34, roughly eleven minutes into the run, then read in full and followed
for the remainder. Recorded so provenance is exact:

| item | value |
|---|---|
| measurements executed at | `65b86bacfbcc227c2b21223682018a7b29dd3e94` (pin matched at §1) |
| worktree advanced mid-run to | `917d560d675986b64fd70e6d0b438c76775f121c` |
| `git diff --name-only 65b86ba 917d560` | `AGENTS.md` — documentation only |
| parent of this handover commit | `917d560` |

No code, script, dataset, or configuration path differs between the two commits,
so no measurement in this run is affected by the drift. The earlier revision of
this file stated that `AGENTS.md` did not exist anywhere in the repo; that was
written before the pull and is corrected here.

## GPU occupancy discipline

At selection time GPU0 held 1 MiB / 24,576 MiB at 0% with no processes. GPU1
carried two other users' processes (`yongan`, pid 3578, LLMLingua-2; `chenhan`,
pid 2996507, VLLM::EngineCore) and was never touched. GPU2 and GPU3 were idle
and unused. Port 8000 was free; ports 8001/8002 belong to other users and were
left alone. No other user's process was signalled. No `sudo`. The server was
shut down by PID resolved from the listening socket, and GPU0 returned to
1 MiB / 0% afterwards.

The server was restarted once between §6 and §7 (the harness killed the first
instance after §6 had already completed). Both instances ran identical flags and
reported identical cache configuration; see `cache-config.json` and
`cache-config-section7.json`.

## Artifact counts

| stage | declared | produced | status |
|---|---|---|---|
| §4 workloads | 4 menu sizes x 7 conditions = 28 | 28 | complete |
| §5 primary replays | 4 x 7 x 3 trials = 84 | 84 | complete |
| §5 summaries | 4 | 4 | complete |
| §6 rendered-prefix audits (k64) | 7 | 7 | complete |
| §7 pressure runs | 6 orderings x 4 regimes = 24 | 24 | **threshold not met** |

Every replay carries 200 requests, `counter_validation.clean=true`, and
`cache_reset_before=true`. Every audit carries `validation.clean=true`,
including `tokenize_count_matches_completion_usage` and the per-request
cached-plus-computed identity.

## Acceptance criteria

| manifest criterion | result |
|---|---|
| `isolated_server` | pass (see caveat below) |
| `same_selected_tool_membership_across_orderings` | pass, all 4 menu sizes |
| `clean_engine_counters` | pass, 84/84 replays and 7/7 audits |
| `no_warm_start` | pass, reset before every declared run |
| `all_declared_trials_present` | pass, 84/84 |
| `pressure_threshold_met_by_every_declared_pressure_run` | **FAIL, 0/24** |
| `raw_results_preserved_outside_git` | pass, archived and checksummed |

Membership was additionally pre-checked independently of the summarizer: within
each menu size all seven conditions carry byte-identical sorted tool sets per
case, and all six non-`original` orderings differ from `original` on 200/200
requests, so the conditions are genuine permutations of one selected set.

## Required direct partial-reuse strata (§5)

Both predeclared k64 conditions have a populated `direct_reuse_buckets.partial_all`
stratum built from real requests. Nothing was interpolated and no substitute
condition was selected.

| condition | `partial_all` requests | cold | 
|---|---|---|
| `alphabetical` | 199 | 1 |
| `tooltrie_v0` | 199 | 1 |

## §7 pressure: failed condition

All 24 regime-runs completed with clean counters and resets but **none reached
the declared 0.90 peak KV occupancy**. Observed peak range 0.036379–0.036882.

Measured cause, from the run records: `estimated_peak_resident_tokens` = 7,041
(one request's prompt), peak `vllm:num_requests_running` = 1.0, and
`vllm:num_requests_waiting` = 0.0 at every sample. The replay client is strictly
sequential with `--max-tokens 1`, so occupancy is bounded by
7,041 / 190,896 = 3.69%.

The threshold was not lowered and concurrency was not added, per §7. A rerun
requires a newly predeclared pressure protocol decided before execution. Full
detail in `PRESSURE-QUARANTINE.md`.

Note for whoever writes this up: `memory_pressure.prompt_volume_over_capacity`
(7.23x) is cumulative prompt-token throughput, not resident occupancy, and does
not evidence KV-cache pressure.

## Quarantined runs

All preserved under `cluster/results/initial-brief-closure-20260805-222246/quarantine/`
and included in the archive. Nothing was deleted or hidden.

1. **`k4-original-trial-1`** — first traffic after server boot. Counter window
   absorbed vLLM's own first-request warmup: 167,808 prefix-cache queries
   against 127,961 prompt tokens actually sent. `metrics-before.txt` recorded
   the counter at 0.0 and vLLM logged no requests before that replay, so no
   other client was involved. Rerun into the declared path is clean and
   reproduces trials 2 and 3 exactly (queries = prompt_tokens = 127,961,
   hits = 20,304).
2. **`k64-audit-original-contended`** — ran while harness-killed and orphaned
   audit processes were still issuing traffic to the same server, breaking the
   one-client rule. Rerun on an undisturbed server is clean, as are the other
   six conditions.
3. **`k64-audit-frequency-orphan-run`** — produced by an audit process orphaned
   when the harness killed its parent shell. Superseded by a clean rerun.
4. **`parallel-session-pressure/`** — provenance, not a measurement fault. See
   below.

## Caveat on server isolation

A second Claude session is running on this host with cwd `/home/taghan/FYP`
(parent pid 3913631). While this executor was idle waiting on the §6 audits, that
session wrote a complete set of §7 pressure outputs plus a quarantine note into
**this executor's result directory** between 01:15 and 01:36, and later built its
own handover directory at `reports/initial-brief-closure/20260805-222246/` from
this executor's summaries.

Handling:

- Those artifacts were preserved unmodified, moved to
  `quarantine/parallel-session-pressure/`, and are in the archive.
- Their numbers were verified independently against the raw JSON before being
  set aside; they agree with this executor's own rerun.
- §7 was then re-executed by this executor into the declared paths, so every
  §7 number reported here was produced and verified by this executor.
- The parallel session's handover directory at
  `reports/initial-brief-closure/20260805-222246/` was **left untouched**, which
  is why this handover lives at the `-gpu-executor` sibling path: §8's
  `test ! -e "$TRACKED_HANDOVER"` guard forbids overwriting it.

§5 and §6 completed at 00:54, before that session's 01:15 activity began, and
every §5 and §6 run carries clean counters, so those stages are unaffected.

## Raw archive

| item | value |
|---|---|
| path | `/home/taghan/initial-brief-closure-20260805-222246.tar.gz` |
| size | 127,635,469 bytes |
| entries | 221 |
| SHA-256 | `568dedaea859056a4c3d2cb4773a810364b40b03858efc83092c5d0804d4a7d5` |

Post-write verification: the SHA-256 was recomputed from the file on disk and
matches, and `tar -tzf` read the whole archive successfully, listing 221 entries
— so the file is a valid archive, not merely an unchanged one. Contents
reconcile with the declared counts: 84 replay JSONs across four menu-size
subdirectories, 7 k64 audits, 6 pressure files each carrying all four regimes
under a `conditions` key (6 x 4 = 24 regime-runs), and 21 quarantine entries.

Raw token IDs and per-request records are intentionally not committed. The
archive still sits on the same physical disk (`/dev/sda2`) as the originals and
needs a copy elsewhere. Two large local filesystems exist (`/mnt/ssd0`,
`/mnt/ssd1`) but both are world-writable shared scratch holding other users'
data, so this executor did not copy research data there unasked; that placement
decision belongs to the project owner.

## Files in this directory

- `retrieved-k{4,16,64,128}-summary.json` — §5 paired matrices
- `audit-k64-validation-summary.json` — §6 validation blocks, no token IDs
- `pressure-bfcl-k64-summary.json` — §7 per-regime occupancy and counters
- `PRESSURE-QUARANTINE.md` — §7 failure record
- `quarantine-README.txt` — full quarantine log
- `environment.txt`, `cache-config.json`, `cache-config-section7.json`, `models.json`
- `raw-archive.sha256`
