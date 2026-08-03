# ToolTrie-v0 — first causal run on the NUS GPU server

Runbook: `NUS_GPU_AGENT_INSTRUCTIONS.md` §10. All cached-token and latency claims
below come from the vLLM server's own Prometheus counters. The planner's
analytical `hinted_schema_tokens` is **not** used to support any cache claim.

## 1. Configuration

| Item | Value |
|---|---|
| Repository | `/home/taghan/FYP` (runbook's `/home/taghan/tatm` is not a git repo) |
| Commit | `558e923a6f6899221682728c9e66dc220c48ccc1` — "Add NUS GPU agent runbook" |
| Branch | `tooltrie-v0-workflow` (descendant of required `9866367`) |
| vLLM / torch | 0.26.0 / 2.11.0+cu130 |
| Platform | Linux-6.8.0-136-generic-x86_64-with-glibc2.39 |
| GPU | RTX 3090 24 GiB, driver 580.173.02, **physical GPU 2** |
| Systems model | `Qwen/Qwen3-0.6B` — block_size **16**, num_gpu_blocks **11931**, capacity **190,896** |
| Quality models | `Qwen/Qwen3-0.6B` (as above) and `Qwen/Qwen3-8B` — block_size 16, num_gpu_blocks **2791**, capacity **44,656** |
| Unit tests | 55 passed before any GPU work |

Exact server commands are in `server-command.txt`. The runbook's suggested
`--capacity-tokens 188912` was **not** used: this server reported 11,931 GPU
blocks, so the correct product is **190,896**, which matches the server's own
`kv_cache_size_tokens` exactly.

Environment note: no single virtualenv on this host has both vLLM and the
project. The server was launched from `/home/taghan/tatm/.venv` (vLLM) and every
client script ran from `/home/taghan/FYP/.venv` (project + pyarrow + tokenizers).
This is safe because `src/tatm/vllm_client.py` speaks HTTP over `urllib.request`;
server and clients are separate processes.

## 2. Systems results — 3 trials per condition

200 requests, 64-tool menus, matched tool sets, APC reset once immediately before
each replay. Mean ± 95% Student-t half-width (`src/tatm/stats.py::describe`).
Cached-token counts were bit-identical across all three trials, so no interval is
shown for them.

### BFCL (total prompt tokens 1,380,694 per replay)

| Metric | original | alphabetical | ToolTrie-v0 |
|---|---|---|---|
| requests | 200 | 200 | 200 |
| measured cached prompt tokens | 16,384 | 526,432 | **1,203,760** |
| cached ratio | 1.19% | 38.13% | **87.19%** |
| computed prefill tokens | 1,364,310 | 854,262 | **176,934** |
| aggregate prefill time (s) | 45.376 ±0.980 | 32.436 ±0.362 | **9.537 ±0.567** |
| aggregate TTFT (s) | 53.140 ±1.753 | 40.324 ±0.636 | **17.378 ±0.741** |
| wall time (s) | 84.59 ±2.47 | 73.37 ±0.65 | **49.88 ±0.75** |

### ToolRet (total prompt tokens 1,391,902 per replay)

| Metric | original | alphabetical | ToolTrie-v0 |
|---|---|---|---|
| requests | 200 | 200 | 200 |
| measured cached prompt tokens | 193,040 | 710,576 | **1,163,408** |
| cached ratio | 13.87% | 51.05% | **83.58%** |
| computed prefill tokens | 1,198,862 | 681,326 | **228,494** |
| aggregate prefill time (s) | 40.602 ±0.322 | 26.601 ±0.481 | **11.939 ±0.215** |
| aggregate TTFT (s) | 48.688 ±0.149 | 34.647 ±0.366 | **19.762 ±0.846** |
| wall time (s) | 82.73 ±0.18 | 68.99 ±0.71 | **54.15 ±1.13** |

All ToolTrie-vs-baseline differences on both partitions have **non-overlapping
95% intervals** for cached ratio and TTFT.

**Sanity cross-check passed.** Alphabetical BFCL measured 38.13% here against
38.15% recorded in `PROJECT_STATUS.md` from an earlier independent session. This
was the predeclared check that the setup had not drifted.

## 3. ToolTrie planner metadata

| Field | BFCL | ToolRet | BFCL quality (both models) |
|---|---|---|---|
| requests with a hinted prefix | 199 / 200 | 199 / 200 | 99 / 100 |
| hinted schema tokens | 979,144 | 944,850 | 474,200 |
| final node count | 1,584 | 2,011 | 600 |
| retained schema tokens | 134,837 | 176,264 | 44,472 |
| planner evictions | 0 | 0 | 424 (at 44,656 capacity) |
| capacity tokens | 190,896 | 190,896 | 190,896 / 44,656 |

Request 0 has no matched prefix in every workload, confirming causality. Median
matched-prefix length was 53/64 tools (BFCL) and 62/64 (ToolRet).

At the 8B capacity the planner evicted 424 nodes, yet the resulting orderings
were **byte-identical to the 190,896-capacity plan on all 100 records**, because
the evicted nodes were cold leaves off the reused path. The two models therefore
scored exactly the same ToolTrie ordering, so §4 is a clean model-size
comparison with no capacity confound.

## 4. BFCL quality — stratified, n=100 (20 per domain)

| Ordering | Qwen3-0.6B name / full / no-tool | Qwen3-8B name / full / no-tool |
|---|---|---|
| original | 78.8% / 52.5% / 90.0% | 87.5% / 78.8% / 90.0% |
| alphabetical | 67.5% / 47.5% / 95.0% | **91.2% / 81.2% / 95.0%** |
| ToolTrie-v0 | 75.0% / 53.8% / 95.0% | 87.5% / 75.0% / 95.0% |

**ToolTrie minus alphabetical:** at 0.6B **+7.50pp** name, **+6.25pp** full;
at 8B **−3.75pp** name, **−6.25pp** full. No-tool accuracy is identical (95.0%)
at both scales, which is expected since ToolTrie falls back to alphabetical.

**The sign flips with model scale.** The 0.6B run alone would have concluded
ToolTrie is quality-free or better. Running both was decisive, and it repeats the
pattern already documented in `PROJECT_STATUS.md`, where a 0.6B ordering-quality
gap vanished at 8B. Here the 8B result does not merely shrink the 0.6B gap, it
reverses it.

By-domain tables for both models are in `bfcl-quality-*-score.json` and
`8b-bfcl-quality-*-score.json`. At 8B the regression concentrates in
`multiple` (full 85%→75%) and `parallel_multiple` (full 80%→70%);
`parallel` is unchanged at 80%/80%.

Quality-run cached ratios (100 requests, `--max-tokens 128`): original 1.7%,
alphabetical 27.2–29.6%, ToolTrie **84.6%** at both model sizes.

## 5. Failed requests and unrelated traffic

Zero failed or retried requests across all 24 replays used in this report
(18 systems + 6 quality); every request returned a finish reason.

**Unrelated traffic did reach the server during the first systems attempt**, which
runbook §10.5 requires be reported. Between 01:27 and 01:43 two replay drivers
issued requests against the same server concurrently. Because
`vllm:prompt_tokens_cached` is a global counter, each run's metric window also
captured the other's hits, producing impossible cached ratios above 100% and
roughly doubled wall times.

Detection, quarantine, and recovery:

- A trial is accepted only if `vllm:prefix_cache_queries` for its window equals
  its own prompt-token total. Contaminated trials show roughly 2×.
- The 10 affected replays are quarantined in `contaminated/` with a README and
  are excluded from every number above.
- Those conditions were re-run with a single driver. **All 18 systems trials in
  the parent directory pass the equality check.**
- The 6 quality replays ran with a single driver and are clean.

## 6. Cautious comparison against the predeclared gate

**Systems: ToolTrie-v0 clearly wins, and the effect is large.** It improves
measured APC reuse over *both* baselines on *both* partitions — BFCL 87.19% vs
38.13% alphabetical (+49.1pp) and 1.19% original; ToolRet 83.58% vs 51.05%
(+32.5pp). Aggregate TTFT falls 2.3× versus alphabetical on BFCL (17.4s vs
40.3s) and 1.8× on ToolRet. Every comparison has separated 95% intervals across
3 trials, and cached-token counts were exactly reproducible.

The mechanism is the one the design predicts: the planner re-derives the shared
catalog order from history and demotes each request's novel tools to the tail via
the alphabetical fallback, so the varying part of the menu stops breaking the
common prefix. This is the same failure mode that made frequency ordering lose in
Task E, exploited in the opposite direction.

**Quality: the gate is not met at 8B.** The predeclared threshold was no more
than a one-percentage-point BFCL regression. Against alphabetical, ToolTrie
regresses **3.75pp on function-name and 6.25pp on full accuracy at Qwen3-8B**.
That exceeds the gate by a wide margin, so on the runbook's own criterion
**ToolTrie-v0 does not pass**, despite winning decisively on systems metrics.

Three cautions against over-reading the quality verdict, in both directions:

1. **The gate is below its own resolution.** n=100 means 1pp is one task; the
   observed 6.25pp is 6 tasks. A 1pp threshold was never measurable with this
   sample, and this run is single-pass with no repeats. The regression is real
   as measured but should be treated as directional.
2. **It is not a small-model artefact, but it is scale-dependent** — and in the
   direction that matters, since 8B is the more deployment-realistic model.
3. **ToolTrie still beats `original` at 8B on no-tool accuracy** (95.0% vs
   90.0%) and ties it on function-name (87.5%). The regression is specifically
   against *alphabetical*, which is the strongest quality baseline at 8B.

**Recommendation.** The systems result is strong enough to justify continuing,
but the quality gate must be settled before ToolTrie-v0 is claimed as a win.
The cheapest next step is repeated quality trials at 8B with a larger
per-domain sample to establish whether the 4–6pp gap survives, since the current
evidence is one run of 100 tasks. Until then the honest statement is: *ToolTrie-v0
buys a 32–49pp increase in measured prefix-cache reuse and a 1.8–2.3× TTFT
reduction, at a measured but not yet replicated 4–6pp function-call accuracy cost
against alphabetical ordering on Qwen3-8B.*

## Scope limitations

- Only the empirical benchmark request order was replayed. Brief §4.5's uniform,
  skewed, and session-bursty replays were not run here; Task D covers locality
  separately.
- Quality is single-pass; only the systems replays have repeated trials.
- ToolRet is used for APC/TTFT only. Its retrieval labels are not BFCL call
  correctness and were not scored.
