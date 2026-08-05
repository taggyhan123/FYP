# §7 demonstrated-pressure runs — QUARANTINED (threshold not reached)

Produced by this executor. All six declared orderings x four declared regimes
ran to completion with clean counters and a cache reset before each condition,
but **none reached the declared `--require-peak-kv-usage 0.90`**.

`locality_replay.py` exited non-zero for every ordering (rc=1), as designed.

## Observed peak KV occupancy (fraction of 190,896-token capacity)

| ordering | empirical | uniform | skewed | session_bursty |
|---|---|---|---|---|
| original | 0.036882 | 0.036882 | 0.036714 | 0.036882 |
| alphabetical | 0.036882 | 0.036882 | 0.036714 | 0.036882 |
| random | 0.036714 | 0.036630 | 0.036714 | 0.036714 |
| frequency | 0.036714 | 0.036882 | 0.036547 | 0.036882 |
| schema_cost_weighted | 0.036882 | 0.036882 | 0.036714 | 0.036882 |
| fp_tree_global | 0.036379 | 0.036882 | 0.036547 | 0.036882 |

Required: 0.90. Observed range: 0.036379 - 0.036882. `requirement_met=false`
for 24/24 regime-runs.

## Measured cause (from the run records, not inferred)

- `capacity_tokens` = 190,896
- `estimated_peak_resident_tokens` = 7,041, i.e. a single request's prompt
- peak `vllm:num_requests_running` = 1.0
- `vllm:num_requests_waiting` = 0.0 at every sample; nothing ever queued

The replay client is strictly sequential: it issues one request, waits for the
response, then issues the next, with `--max-tokens 1`. At most one request's
prefill is resident at any instant, so peak occupancy is bounded by
7,041 / 190,896 = 3.69%. Reaching 0.90 would require roughly 48 prompts of
this size resident concurrently, which this client cannot produce.

## Not done, deliberately

- The 0.90 threshold was **not** lowered after inspection.
- Concurrency was **not** added to force occupancy. Either change would alter
  the declared method after seeing results, which §7 of
  `NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md` and the manifest both forbid.
- No pressure run is reported as passing.
- `memory_pressure.prompt_volume_over_capacity` (7.23x) is **cumulative prompt
  token throughput**, not resident occupancy, and must not be quoted as
  evidence of KV-cache pressure.

## Required next step (belongs to the local analysis session, not the executor)

Per §7: "If pressure is not reached, quarantine that run and ask for a newly
predeclared rerun; do not lower the threshold after inspection."

A rerun needs a change to the pressure protocol declared **before** execution —
for example a declared concurrency level, a declared smaller KV cache, or a
declared different threshold.

## Manifest fields affected

- `memory_pressure.required_peak_kv_usage_fraction` = 0.9 -> not reached
- `acceptance.pressure_threshold_met_by_every_declared_pressure_run` = true
  -> **NOT SATISFIED**
