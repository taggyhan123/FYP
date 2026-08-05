# ToolTrie-v0 — first causal run on the NUS GPU server

Runbook: `NUS_GPU_AGENT_INSTRUCTIONS.md` §10. All cached-token and latency claims
below come from the vLLM server's own Prometheus counters. The planner's
analytical `hinted_schema_tokens` is **not** used to support any cache claim.

> Historical scope note: this is the Phase 1 report. Its proposed next
> experiments and "open no-tool question" were subsequently completed and
> superseded by `reports/tooltrie-phase2/findings.md`, which confirmed the
> no-tool regression and measured causal ContextPilot plus the external engine
> comparison. The measurements below remain valid for their stated run.

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

**The sign flips with model scale**, so the 0.6B run alone would have concluded
ToolTrie is quality-free or better. **However, see §4a: at n=1000 the 8B
regression does not survive either.** The n=100 result above is retained because
it is what the runbook specified, and because the contrast is the finding.

Quality-run cached ratios (100 requests, `--max-tokens 128`): original 1.7%,
alphabetical 27.2–29.6%, ToolTrie **84.6%** at both model sizes.

## 4a. Follow-up at n=1000 — the 8B regression was sampling noise

The §4 gate rests on 100 tasks, where one percentage point is one task. The
evaluation was repeated at the **maximum balanced sample, 200 per domain
(n=1000)**, a nested superset of the 100 above since the builder takes a
deterministic per-domain prefix. Single pass per condition: generation runs at
`temperature=0, seed=0`, so repeated identical replays add almost no power and
the entire budget was spent on sample size instead.

| Qwen3-8B | function-name | full | no-tool |
| --- | --- | --- | --- |
| original | 83.13% | 77.75% | 78.50% |
| alphabetical | 81.50% | 75.62% | 87.50% |
| ToolTrie-v0 | 82.25% | 77.12% | 83.50% |

**ToolTrie minus alphabetical, n=100 → n=1000:**

| Metric | n=100 | n=1000 | 95% CI on the n=1000 difference |
| --- | --- | --- | --- |
| function-name | −3.75pp | **+0.75pp** | −3.03 … +4.53pp |
| full | −6.25pp | **+1.50pp** | −2.66 … +5.66pp |
| no-tool | +0.00pp | −4.00pp | −10.89 … +2.89pp |

Both headline metrics reverse sign and become slightly positive, and zero lies
inside all three intervals. **There is no detectable quality cost at 8B.**

Two things the larger sample also exposes:

1. **The n=100 sample was optimistic as well as noisy.** Absolute accuracy fell
   for every condition (alphabetical full 81.25%→75.62%, no-tool 95.0%→87.5%),
   so the first 20 tasks per domain were easier than the remaining 180.
2. **No-tool accuracy is the one metric where ToolTrie's point estimate is
   negative** (−4.00pp), and it is the same direction alphabetical won at 0.6B.
   It is not significant here (200 irrelevance tasks, CI ±6.9pp), but it has now
   appeared twice, and correctly declining irrelevant requests is
   safety-relevant. This deserves a targeted run rather than dismissal.

By domain (n=200 each):

| domain | metric | original | alphabetical | tooltrie |
| --- | --- | --- | --- | --- |
| simple_python | function name | 100.00% | 97.00% | 97.50% |
| simple_python | full | 96.00% | 92.50% | 93.50% |
| multiple | function name | 96.00% | 96.00% | 96.50% |
| multiple | full | 88.00% | 86.50% | 88.50% |
| parallel | function name | 75.00% | 73.00% | 74.50% |
| parallel | full | 71.50% | 70.50% | 72.50% |
| parallel_multiple | function name | 61.50% | 60.00% | 60.50% |
| parallel_multiple | full | 55.50% | 53.00% | 54.00% |
| irrelevance | no tool | 78.50% | 87.50% | 83.50% |

Run on two identically configured 8B servers (GPU2 :8000, GPU3 :8100, both
capacity 44,656) to halve wall time; each server served exactly one replay at a
time. Because this run is scored for accuracy only, its timings are not used for
any claim. `original` shows a 0.1% excess on `prefix_cache_queries`
(8,192 tokens, 999/1000 per-request windows clean) which affects cache
accounting only, not generated output; all 3000 requests returned valid
completions with zero failures.

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

**Quality: no cost is detectable at 8B, but the gate as written is unfalsifiable.**
At n=100 ToolTrie appeared to regress 3.75pp/6.25pp against alphabetical. At
n=1000 (§4a) both metrics reverse to **+0.75pp and +1.50pp**, with zero inside
every interval. The apparent regression was sampling noise.

The predeclared ≤1pp threshold, however, **cannot be settled at any sample size
this project can afford.** The 95% CI on the full-accuracy difference is ±4.2pp
at n=1000; resolving 1pp would need roughly n=15,000, or about 12 GPU-hours per
condition at 8B. The gate should be restated as an equivalence test with a
declared margin, not a point threshold.

What the evidence supports:

1. **No quality regression is detectable at n=1000**, and both headline point
   estimates favour ToolTrie. This is weaker than "ToolTrie passes" and stronger
   than "the cost is unknown."
2. **The n=100 verdict was wrong in sign, not just in magnitude.** Any claim
   from a 100-task BFCL sample should be treated as a pilot.
3. **No-tool accuracy remains the open question** (−4.00pp point estimate, not
   significant, but the same direction alphabetical won at 0.6B).

**Recommendation.** The systems result is strong and the quality blocker is
resolved to the resolution available. The honest summary is: *ToolTrie-v0 buys a
32–49pp increase in measured prefix-cache reuse and a 1.8–2.3× TTFT reduction,
with no detectable function-call accuracy cost against alphabetical ordering at
Qwen3-8B and n=1000.* The next experiment should be an external comparison
(CacheWeaver / SGLang RadixAttention) rather than further self-comparison, with
a targeted irrelevance-only run to close the no-tool question.

## Scope limitations

- Only the empirical benchmark request order was replayed. Brief §4.5's uniform,
  skewed, and session-bursty replays were not run here; Task D covers locality
  separately.
- Quality is single-pass at every sample size; only the systems replays have
  repeated trials. Justified by `temperature=0, seed=0`, but batching
  nondeterminism is not formally excluded.
- One model family, one menu size (64), one GPU type. The 87% reuse depends on
  menus drawn from a fixed shared catalog; that matches a connected MCP
  deployment but is an assumption the benchmarks do not themselves supply.
- ToolRet is used for APC/TTFT only. Its retrieval labels are not BFCL call
  correctness and were not scored.
