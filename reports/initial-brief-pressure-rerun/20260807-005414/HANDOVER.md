# GPU handover — controlled-cache pressure rerun

Execution record only. No scientific conclusions are drawn here; validating the
evidence and deciding whether the initial brief is closed belongs to the local
analysis session.

**This run is a controlled-cache stress test.** The KV cache was deliberately
reduced to 480 blocks (7,680 tokens) to make the declared 0.90 occupancy
threshold reachable by a strictly sequential client. **Latency from this run is
not comparable with the earlier 190,896-token-capacity runs**, and nothing here
may be described as naturally occurring production pressure. The manifest sets
`compare_latency_across_cache_capacities: false`; that constraint is part of the
method, not a caveat added afterwards.

## Commit and environment

| item | value |
|---|---|
| repo | `/home/taghan/FYP` (source), detached worktree `/home/taghan/FYP-pressure-rerun-20260807-005414` |
| commit | `caf8ab576eb15d95265a53ec76c772c8af6c7929` |
| pin check | `EXPECTED_FYP_COMMIT` matched exactly; worktree never advanced |
| tests at that commit | `101 passed` |
| host | `gpuserver`, Linux 6.8.0-136-generic |
| GPU | index 0, RTX 3090, `GPU-87ff24b3-1ffa-2319-0737-a9fc422063fc`, driver 580.173.02 |
| vLLM / torch | 0.26.0 / 2.11.0+cu130 |
| model | `Qwen/Qwen3-0.6B` |
| result stamp | `20260807-005414` |

A separate detached worktree was used as instructed. The source repository at
`/home/taghan/FYP` was neither advanced nor cleaned, and its untracked files
were left in place.

### Deviation from the runbook's literal environment command

Runbook §2 records `torch` and `vllm` versions from `$FYP_PYTHON`. This host
splits the two roles: the server venv (`/home/taghan/tatm/.venv`) holds
`vllm`+`torch`, and the client venv (`/home/taghan/FYP/.venv`) holds `tatm`+
`pytest`. Running the line as written raises `ModuleNotFoundError: torch`.
Both interpreters were therefore recorded separately in `environment.txt`, with
that traceback left in the file and annotated rather than hidden. No package was
installed or modified.

## GPU occupancy discipline

At selection time GPU0 held 1 MiB at 0% with no compute processes. Both other
users' processes (pids 3578 and 2996507) were on GPU1 and were never touched.
GPU2 and GPU3 were idle and unused. Port 8000 was free; 8001 and 8002 belong to
other users and were left alone. No `sudo`. One server, one sequential client,
for the whole matrix — the server was never restarted mid-run.

## Live cache configuration (read back, not assumed)

| item | value |
|---|---|
| `enable_prefix_caching` | true |
| `block_size` | 16 |
| `num_gpu_blocks` | 480 (`num_gpu_blocks_override=480`) |
| live capacity | **7,680 tokens** — matches `expected_capacity_tokens` exactly |
| `max_model_len` | 7,168 |
| `kv_cache_max_concurrency` | 1.0714 |
| eviction histogram | `vllm:kv_block_idle_before_evict_seconds` present, count 0.0 before the run |

`--kv-cache-metrics --kv-cache-metrics-sample 1.0` means 100% of blocks are
sampled, so the eviction counts below are measured, not extrapolated from a 1%
default sample.

## Acceptance

| manifest criterion | result |
|---|---|
| `live_cache_capacity_tokens_equals: 7680` | pass |
| `all_24_regime_runs_present` | pass, 24/24 |
| `pressure_threshold_met_by_every_regime_run` | **pass, 24/24** |
| `eviction_metric_present_in_every_regime_run` | pass |
| `sampled_eviction_count_positive_in_every_regime_run` | pass, range 57,696–85,340 |
| `preemptions_in_every_regime_run: 0` | pass, 0 everywhere |
| `peak_running_requests_at_most: 1` | pass, 1.0 everywhere |
| `peak_waiting_requests: 0` | pass, 0.0 everywhere |
| `cache_reset_before_every_ordering_run` | pass |
| `clean_engine_counters` | pass |
| `isolated_server` | pass |
| `raw_results_preserved_outside_git` | pass |

`scripts/summarize_pressure_replays.py` exit code 0:

```
Accepted pressure regimes: 24/24
Validation checks: 384/384
```

384 = 24 regime-runs x 16 checks. **No failed checks.** Every ordering command
also returned rc=0 from its own `--require-peak-kv-usage 0.90` gate.

## 24-run acceptance table

Peak KV usage is a fraction of the 7,680-token capacity. Evictions are sampled
block-eviction events at 100% sampling.

| ordering | regime | peak KV | resident tok | evictions | reuse | preempt | run/wait | |
|---|---|---|---|---|---|---|---|---|
| original | empirical | 0.91858 | 7055 | 84,885 | 0.0118 | 0 | 1/0 | PASS |
| original | uniform | 0.91858 | 7055 | 85,340 | 0.0069 | 0 | 1/0 | PASS |
| original | skewed | 0.91441 | 7023 | 84,810 | 0.0124 | 0 | 1/0 | PASS |
| original | session_bursty | 0.91858 | 7055 | 85,309 | 0.0069 | 0 | 1/0 | PASS |
| alphabetical | empirical | 0.91441 | 7023 | 60,699 | 0.2921 | 0 | 1/0 | PASS |
| alphabetical | uniform | 0.91858 | 7055 | 60,600 | 0.2935 | 0 | 1/0 | PASS |
| alphabetical | skewed | 0.91441 | 7023 | 61,677 | 0.2806 | 0 | 1/0 | PASS |
| alphabetical | session_bursty | 0.91858 | 7055 | 61,948 | 0.2776 | 0 | 1/0 | PASS |
| random | empirical | 0.91441 | 7023 | 58,158 | 0.3216 | 0 | 1/0 | PASS |
| random | uniform | 0.91858 | 7055 | 58,948 | 0.3127 | 0 | 1/0 | PASS |
| random | skewed | 0.91441 | 7023 | 57,696 | 0.3267 | 0 | 1/0 | PASS |
| random | session_bursty | 0.91441 | 7023 | 59,854 | 0.3019 | 0 | 1/0 | PASS |
| frequency | empirical | 0.91858 | 7055 | 77,757 | 0.0944 | 0 | 1/0 | PASS |
| frequency | uniform | 0.91858 | 7055 | 78,176 | 0.0899 | 0 | 1/0 | PASS |
| frequency | skewed | 0.91441 | 7023 | 77,682 | 0.0951 | 0 | 1/0 | PASS |
| frequency | session_bursty | 0.91858 | 7055 | 78,144 | 0.0900 | 0 | 1/0 | PASS |
| schema_cost_weighted | empirical | 0.91858 | 7055 | 77,757 | 0.0944 | 0 | 1/0 | PASS |
| schema_cost_weighted | uniform | 0.91858 | 7055 | 78,176 | 0.0899 | 0 | 1/0 | PASS |
| schema_cost_weighted | skewed | 0.91023 | 6991 | 77,682 | 0.0951 | 0 | 1/0 | PASS |
| schema_cost_weighted | session_bursty | 0.91858 | 7055 | 78,144 | 0.0900 | 0 | 1/0 | PASS |
| fp_tree_global | empirical | 0.91858 | 7055 | 77,757 | 0.0944 | 0 | 1/0 | PASS |
| fp_tree_global | uniform | 0.91858 | 7055 | 78,176 | 0.0899 | 0 | 1/0 | PASS |
| fp_tree_global | skewed | 0.91441 | 7023 | 77,682 | 0.0951 | 0 | 1/0 | PASS |
| fp_tree_global | session_bursty | 0.91858 | 7055 | 78,144 | 0.0900 | 0 | 1/0 | PASS |

Peak KV usage range **0.91023 – 0.91858**, every value above the declared 0.90.
The threshold was not altered, capacity was not adjusted after inspection, and
no condition was substituted. Contrast with the superseded run at 190,896-token
capacity, which peaked at 0.036882 across all 24 regime-runs.

## Observation for the analysis session: three conditions are not independent

`frequency`, `schema_cost_weighted`, and `fp_tree_global` returned **identical
values on every field** in all four regimes — measured cached tokens (130,400 in
`empirical`), measured reuse (0.094445), sampled evictions, and, decisively, the
whole analytical `predicted` block (`cacheable_block_tokens` 97,408, predicted
`evictions` 11,664, `estimated_block_reuse_ratio` 0.087441). That block is
computed from the ordering itself before any request is served, so three
policies producing byte-identical predictions are emitting the same tool
sequence on this workload. `alphabetical` differs on every one of those fields
(403,328 cached, 348,320 cacheable, 9,103 predicted evictions).

Stated as fact, not interpretation: the matrix contains 24 valid regime-runs but
only **four distinct orderings**. This is consistent with the already-documented
Phase 2 finding that the fitted baselines collapse onto essentially one global
order (all within 0.01pp at 39.69% on BFCL). Whether that changes how the six
conditions should be reported is a decision for the analysis session.

Caveat on verification: `per_request` records in these run files carry only
`canonical_tool_tokens`, `index`, `measurement`, and `task_id` — not the tool
sequence. The identity above is therefore established from the analytical
prediction and every measured aggregate agreeing exactly, not from a direct
byte-comparison of emitted orderings, which these artifacts do not support.

## Raw archive

| item | value |
|---|---|
| path | `/home/taghan/initial-brief-pressure-rerun-20260807-005414.tar.gz` |
| size | 833,218 bytes |
| entries | 36 |
| SHA-256 | `d18c613958dbeadbb9114a7ec6c1418d1c062803075974940b470791e86a78e0` |

Verified readable, not merely unchanged: `tar -tzf` listed all 36 entries after
writing. Contents include the six ordering JSONs, their stdout/stderr, the
server log, `metrics-before.txt`, the driver scripts, and both summaries.

The archive sits on the same physical disk (`/dev/sda2`) as the originals and
needs a copy elsewhere. `/mnt/ssd0` and `/mnt/ssd1` have space but are
world-writable shared scratch holding other users' data, so this executor did
not place research data there unasked.

## Scope

This run replaces **only** the failed memory-pressure acceptance criterion. The
84 retrieved-menu replays, seven rendered-prefix audits, and their archive from
`20260805-222246` were not rerun, not overwritten, and not read for anything
except provenance. The superseded pressure outputs remain preserved in that
earlier archive.

`PROJECT_STATUS.md` and the scientific reports were not edited, per the runbook.

## Files in this directory

- `pressure-summary.json` — full validator output, 24 regime-runs x 16 checks
- `acceptance-table.txt` — the table above as generated
- `environment.txt`, `cache-config.json`, `models.json`
- `raw-archive.sha256`
