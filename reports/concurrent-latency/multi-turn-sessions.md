# Multi-turn sessions: does ToolTrie-v1 still beat ContextPilot?

Every earlier result was single-turn, and multi-turn traffic was the one
untested condition that could reverse the headline, because it is
ContextPilot's own design regime. This experiment tests it on real
coding-agent sessions.

**Short answer: yes, ToolTrie-v1 still beats ContextPilot in every
configuration, but in realistic multi-turn sessions the margin is essentially
zero.** The cache mostly reuses the growing conversation history, and the
history does not depend on tool order.
- **The feared reversal did not happen**: ContextPilot never wins.
- **Ordering matters again only in one setting**: the whole tool list changes
  every turn *and* sits at the front of the prompt. There, v1 recovers the
  most (20.3% hit rate against ContextPilot's 16.2%).
- **But a better design beats any ordering**: appending new tools after the
  history keeps the hit rate at 93%.

In multi-turn sessions, *where* tools go and how much cache there is matter
far more than tool order.

**Confirmed on the GPU** (13 runs on two RTX 3090s, section 5):
- The simulator matches vLLM within 0.11 points in every run, and exactly on
  every request whenever the cache isn't under pressure.
- With static tools, every policy has the same latency.
- Where v1 caches more (tools changing every turn), latency doesn't improve.
- A new effect: **with more than about 12 sessions sending at once, the cache
  thrashes.** Hits fall from 93% to 13%, and median latency rises from 0.5 s
  to 12–24 s.

---

## Setup: what is real and what is synthetic

| part | source | real or synthetic |
|---|---|---|
| conversations | 300 SWE-agent sessions from nebius/SWE-agent-trajectories, one attempt per issue, 5,058 model calls | **real** text and turn structure |
| system prompt | identical in every session (the SWE-agent scaffold) | **real** |
| mid-session tool changes | TraceLab (real Claude Code use): `ToolSearch` in 11.8% of sessions, usually once and early, about 2 tools each. Here: 34 sessions, 75 changes | **real** rates, positions and sizes |
| gaps between turns | TraceLab's measured tool latencies | **real** |
| which catalogue tools a session gets | 64 tools (16 as a check) retrieved from ToolRet by dense retrieval, using the issue text | **synthetic**: real coding agents barely use large catalogues (MCP tools are 0.03% of TraceLab's tool calls) |

**Three ways the tools change during a session:**
- **Static:** tools retrieved once, at the start.
- **Realistic:** occasional additions, at TraceLab's rates.
- **Every turn:** the top 64 re-retrieved on every model call (per-turn
  tool-RAG).

**Two places the changed tools can go:**
- **Front:** the tool block is re-rendered in the system prompt.
- **Append:** new tools are added after the history, Anthropic's approach,
  keeping every loaded tool.

**The rest of the setup:**
- **Policies:** the same six orderings, re-planned whenever a session's tools
  change.
- **Concurrency:** 1, 8 or 32 sessions at once, with rounds served in the order
  a closed-loop system would serve them.
- **Cache sizes:** those of the 0.6B (main), 4B and 8B servers, plus unlimited.

**How it was measured.** Every model call's prompt was rebuilt exactly as vLLM
would tokenise it, and replayed through the prefix-cache simulator validated
earlier ([`cache-hit-miss.md`](cache-hit-miss.md)).
- With Qwen3's template, each turn can reuse the *whole* previous prompt
  except its last 4 tokens. The model's reply is never reusable, because
  history renders it differently.
- Hypotheses were written down before any simulation ran.
- Re-running a configuration reproduces its numbers exactly.

---

## 1. In realistic sessions, tool order barely matters

Cache hit rate, static tools, 64 per session, 0.6B cache:

| policy | 1 session at once | 8 at once | 32 at once |
|---|---|---|---|
| no reordering | 92.95% | 91.62% | 84.34% |
| alphabetical | 92.95% | 91.62% | 84.34% |
| frequency | 92.95% | 91.62% | 84.35% |
| ToolTrie-v0 | 92.96% | 91.63% | 84.36% |
| ContextPilot | 92.95% | 91.62% | 84.35% |
| **ToolTrie-v1** | 92.96% | 91.64% | 84.39% |

- **All six policies are within 0.05 points.** v1 is highest in every column,
  by 0.01–0.04 points over ContextPilot. It's real, but it is too small to
  matter.
- **The history does the work.** At one session at a time, 93% of prompt
  tokens are reused, close to TraceLab's 95.7% for real coding agents.
- **Sharing between sessions is small.** Only about 7% of prompt tokens are
  reused from *other* sessions: the shared system prompt, plus the rare tool
  lists that start alike. v1 creates slightly more of this than ContextPilot
  (7.42% vs 7.26% at one session at a time).
- **The same holds with realistic tool changes and with 16 tools.** v1 is
  highest in every column among the policies run, by at most 0.08 points over
  ContextPilot.

---

## 2. What does matter: concurrency and cache size

Between two turns of a session, other sessions' turns can push its history out
of the cache. This is the same effect TraceLab reports when real users pause.
ToolTrie-v1, static tools:

| cache size | 1 session at once | 8 at once | 32 at once |
|---|---|---|---|
| 0.6B server (189,712 tokens) | 92.96% | 91.64% | 84.39% |
| 4B server (97,904 tokens) | 92.96% | 89.32% | 71.48% |
| 8B server (36,704 tokens) | 92.96% | 74.60% | **30.71%** |
| unlimited | 92.98% | 92.98% | 92.98% |

- **With 32 sessions on the 8B server's cache, the hit rate falls from 93% to
  31%.** Each model call then recomputes about 11,200 tokens instead of 1,100.
- **Every ordering policy moves identically.** In this sweep (no reordering,
  ContextPilot, v1), they stay within 0.1 points at every cache size and level
  of concurrency.
- **For multi-turn serving, the levers are memory and scheduling, not
  ordering.** That means keeping the sessions that are between turns in the
  cache.

---

## 3. Where new tools go matters far more than their order

ToolTrie-v1, hit rate:

| how tools change | where they go | 1 session | 8 | 32 | calls that keep their history cached (1 session) |
|---|---|---|---|---|---|
| static | — | 92.96% | 91.64% | 84.39% | 100% |
| realistic (TraceLab) | front | 92.60% | 91.24% | 83.84% | 98.4% |
| realistic (TraceLab) | **append** | **92.96%** | **91.63%** | **84.34%** | **100%** |
| every turn | front | 20.29% | 17.45% | 14.96% | 2.2% |
| every turn | **append** | **93.07%** | **90.55%** | **78.42%** | **100%** |

- **Re-retrieving tools every turn and putting them at the front destroys the
  cache.** Any change in the tool block invalidates everything after it,
  including the whole history. Only 2% of calls keep their history cached.
- **Appending new tools after the history keeps 93%.** It's the same tools and
  the same information, just placed where they don't disturb the cached
  prefix.
- **Realistic changes are rare**, so putting them at the front costs only about
  0.4 points. Policies that re-sort the whole block lose more: alphabetical
  and frequency lose about 1 point.
- **This answers brief RQ4 in the multi-turn setting.** Keeping already-loaded
  tools and appending new ones preserves the prefix. Rebuilding the tool list
  does not.

**The one setting where ordering matters:** every-turn re-retrieval at the
front.

| policy | hit, 1 session | 8 | 32 | tokens recomputed per call (1 session) |
|---|---|---|---|---|
| no reordering | 13.00% | 11.51% | 9.72% | 13,998 |
| alphabetical | 15.37% | 12.76% | 10.98% | 13,617 |
| frequency | 15.59% | 13.05% | 11.25% | 13,582 |
| ToolTrie-v0 | 16.41% | 13.76% | 11.76% | 13,451 |
| ContextPilot | 16.23% | 13.15% | 12.30% | 13,479 |
| **ToolTrie-v1** | **20.29%** | **17.45%** | **14.96%** | **12,826** |

- **v1 beats ContextPilot by 2.7–4.3 points (1.2–1.3x).** v1 keeps the tools
  that survive from one turn to the next in the same order, so consecutive
  prompts share more of their beginning. This is the single-turn result
  carried over.
- **But every policy here is 63–80 points below simply appending.** The
  ordering win is real, but it's a fix for a design no one should use.

---

## 4. ContextPilot is not stable when re-planning; v1 is

If the planner is asked again on every turn, even when the tool set has not
changed:

| change in hit rate from re-planning every turn | 1 session | 8 | 32 |
|---|---|---|---|
| ContextPilot | −0.60 | −0.71 | −0.70 |
| ToolTrie-v1 | −0.00 | −0.00 | +0.02 |

- **ContextPilot sometimes returns a different order for an identical tool
  list.** Each time, it breaks that session's cached history: the share of
  calls keeping their history falls from 100% to 99.3%.
- **v1 always returns the same order,** because its trie matches its own
  previous plan exactly.
- The main results re-plan only when tools change, which is how a deployed
  system would work. So this is a robustness property, not a headline number.

---

## 5. On the GPU: measured, not simulated

**Setup:**
- Two RTX 3090s, each running its own vLLM 0.26.0 server with Qwen3-0.6B. The
  flags are the same as every earlier run, plus per-request reporting of
  cached tokens.
- Every turn's exact token sequence was sent with `max_tokens=1`, so each
  request measures time to first token.
- N sessions ran at once. Each sent its next turn as soon as the previous one
  returned, with **no pause between turns**. That's harsher than real use,
  where tools run and people read.
- 13 runs and 56,010 requests; one request failed with a dropped connection.

**The simulator is exact on real vLLM.** Hit rates, measured vs simulated for
the same dispatch order:

| run | measured hit | simulated hit | requests matching exactly |
|---|---|---|---|
| static tools, 8 sessions (no reordering / ContextPilot / v1) | 92.95 / 92.96 / 92.97% | identical | 5,058 of 5,058 each |
| realistic changes, front / append, 8 sessions | 92.60 / 92.96% | identical | 5,058 of 5,058 each |
| every-turn changes at the front, 8 sessions (three policies) | 7.86 / 9.04 / 10.97% | within 0.01 | 1,809–1,810 of 1,810 |
| 16, 24, 32 sessions | 13.06 / 12.27 / 11.23% | within 0.01 | 99–100% |
| under memory pressure (12 sessions; every-turn append) | 83.64 / 84.30% | 83.53 / 84.23% | 86% / 77% |

- **Totals agree within 0.11 points in every run.**
- **Every request matches exactly** whenever the cache isn't under pressure.
- **Under pressure the per-request match loosens** to 77–86%, because requests
  running at the same time hold blocks the simulator treats as free, but the
  totals still agree.
- **So the simulated results in sections 1–4 are measurements in all but
  name.**

**Static tools, 8 sessions: the policies perform the same on the GPU too.**

| policy | hit rate | time to first token, p50 | p95 | p99 | requests/s |
|---|---|---|---|---|---|
| no reordering | 92.95% | 561 ms | 1,330 ms | 1,793 ms | 12.39 |
| ContextPilot | 92.96% | 559 ms | 1,294 ms | 1,822 ms | 12.36 |
| ToolTrie-v1 | 92.97% | 544 ms | 1,286 ms | 1,846 ms | 12.50 |

The differences are within run-to-run noise. Realistic tool changes are just as
cheap on the GPU: at the front, 92.60% hits and a 564 ms median; appended,
92.96% and 561 ms.

**New: a cliff when too many sessions are active at once.** Static tools,
ToolTrie-v1 (the 32-session run used no reordering, which is identical here):

| sessions sending at once | hit rate | time to first token, p50 | p95 | requests/s |
|---|---|---|---|---|
| 8 | 92.97% | 0.54 s | 1.3 s | 12.50 |
| 12 | 83.64% | 0.96 s | 9.1 s | 5.78 |
| 16 | **13.06%** | **12.2 s** | 14.8 s | 1.36 |
| 24 | 12.27% | 19.0 s | 22.1 s | 1.30 |
| 32 | 11.23% | 24.4 s | 27.4 s | 1.34 |

- **Why the cliff:** a session's prompt averages about 16,000 tokens, and the
  cache holds 190,000, which is about 12 sessions' histories. With more
  sessions than that sending back to back, each session's history is evicted
  before its next turn arrives. The cache thrashes: hits collapse between 12
  and 16 sessions, throughput falls about 9x, and median latency rises 22x at
  16 sessions and 45x at 32.
- **Why the simulation at 32 sessions gave 84% instead:** it used TraceLab's
  real pauses between turns (tool runs, user thinking), which keep most
  sessions idle at any moment.
- **Lesson for deployment:** real idle time is what keeps multi-turn caching
  working. When more sessions are active than the cache can hold, it
  collapses. Limiting active sessions to what fits in the cache matters far
  more than any ordering policy.

**Every-turn tool changes: v1 caches more, but it doesn't show in latency.**
The same 100 sessions, 8 at once:

| setup | hit rate | time to first token, p50 | p95 |
|---|---|---|---|
| front, no reordering | 7.86% | 6.30 s | 8.17 s |
| front, ContextPilot | 9.04% | 6.30 s | 8.20 s |
| front, ToolTrie-v1 | **10.97%** | 6.39 s | 8.29 s |
| appended, ToolTrie-v1 | **85.04%** | **1.50 s** | 8.78 s |

- **v1 still caches the most**, as the simulation predicted. But with about 90%
  of each prompt recomputed, the GPU is saturated, and 2–3 points less
  recomputation is invisible in latency.
- **Appending gives 85% hits and a 4x lower median.** Its p95 stays high,
  though: appended tools make prompts grow (about 24,000 tokens on average,
  against 16,000), so 8 sessions no longer fit in the cache and it partly
  thrashes. Appending is the right placement, but tools changing every turn
  is expensive either way.

---

## Pre-declared hypotheses: how they fared

| hypothesis | result |
|---|---|
| 1. Static tools, low concurrency: 85–95% hit rate for every ordering; history dominates | **supported**: 92.95–92.96% |
| 2. Ordering differences come only from cross-session reuse and are a few points at most; v1 ≥ ContextPilot | **supported**: at most 0.05 points; v1 ≥ ContextPilot in every cell |
| 3. Every-turn re-retrieval at the front collapses the hit rate; append keeps it | **supported**: 13–20% vs 93% |
| 4. More concurrency lowers hits through eviction, and v1's recency helps more there | **half supported**: hits fall with concurrency (93% at 1 session to 84% at 32 on the 0.6B cache, and 31% on the 8B cache); v1's margin does **not** grow with concurrency |

## What this means for the thesis

- **The multi-turn risk is resolved in v1's favour, but not as a win.** v1 is
  never worse than ContextPilot and is stable across turns. However, in
  realistic sessions the difference is too small to matter.
- **ToolTrie's value lies where the prompt changes per request:** single-turn
  retrieved menus, and cutting a large pool to a shortlist
  ([`top-findings.md`](top-findings.md)). It is not in long conversations.
- **On the GPU, v1's multi-turn edge doesn't reach latency.** With static
  tools every policy has the same latency. With tools changing every turn, v1
  caches the most but the GPU is saturated either way.
- **For multi-turn agents, the design advice is:**
  - append new tools after the history rather than rebuilding the tool block;
  - keep already-loaded tools;
  - keep the number of sessions actively sending within what the cache can
    hold. On one RTX 3090 with a 0.6B model that's about 12; beyond it, hits
    collapse from 93% to 13% and latency rises 20x or more.

  This matches how Claude Code's tool search is designed, and is consistent
  with TraceLab's findings.

## Limits

- **The tool layer is synthetic.** Real coding agents barely use large
  catalogues, so which tools each session gets is simulated. The conversation
  text, turn structure, change rates and waits are real.
- **SWE-agent sessions, not Claude Code.** Claude Code's text isn't public;
  TraceLab has only token counts.
- **Qwen3 template.** With other templates the model's reply might also be
  reusable, which would raise every hit rate equally.
- **Sessions are cut to the 32k context window.** 5,058 of 8,798 model calls
  are kept, and later turns of long sessions are dropped (median 14 calls per
  session). Later turns reuse more, so this lowers hit rates slightly.
- **Sections 1–4 are simulated; section 5 measures a subset on the GPU.** The
  simulator matched vLLM within 0.11 points in all 13 GPU runs.
- **The GPU runs have their own limits:**
  - they measure time to first token only (`max_tokens=1`, no decoding);
  - sessions send turns back to back, with no pauses;
  - the every-turn front runs use the first 100 sessions;
  - Qwen3-0.6B only.
- **The simulation serves model calls one at a time**, in the order a
  concurrent system would serve them. The GPU runs show this loosens
  per-request agreement only under memory pressure.
- **ContextPilot's ordering only**, as in all our comparisons. Its
  conversation-level de-duplication and scheduling are not tested.
- **ContextPilot needed a workaround at this scale.** At the pinned commit it
  crashes on long request streams (an unpicklable multiprocessing worker). It
  was run through `scripts/build_contextpilot_workload_serial.py`, which
  computes the same distances serially. Orderings were verified identical on a
  stream the unmodified builder completes.
- **The first sampling attempt was discarded.** The dataset has about 22
  attempts per issue, and sampling rows directly gave sessions with identical
  opening prompts. That inflated cross-session reuse to 36%. The run was
  superseded, and this one uses one attempt per issue.

## Reproduce

```
OUT=cluster/results/multiturn-sessions-<timestamp>
.venv/bin/python scripts/build_session_workload.py sample --out-dir $OUT
python3 scripts/build_session_workload.py tracelab --out-dir $OUT
CUDA_VISIBLE_DEVICES="" <tatm env>/python scripts/build_session_workload.py events --out-dir $OUT
bash $OUT/build_plans.sh $OUT        # existing ToolTrie / ContextPilot builders
CUDA_VISIBLE_DEVICES="" <tatm env>/python scripts/simulate_sessions.py --out-dir $OUT --workers 32
```

All of that is CPU only. The GPU replay (section 5) starts one vLLM server per
GPU and runs one driver per server:

```
bash $RUN/server_gpu1.sh &        # Qwen3-0.6B, same flags as before + --enable-prompt-tokens-details
<tatm venv>/python scripts/replay_sessions_vllm.py --out-dir $OUT --timeline D1-k64 \
    --ordering tooltrie_v1 --concurrency 8 --base-url http://127.0.0.1:8311 --output $RUN/replays/...
```

The simulation results are in
`cluster/results/multiturn-sessions-20260911-234046/` (with the
pre-declaration, plans and logs). The GPU results, server scripts and logs are
in `cluster/results/multiturn-gpu-20260912-001802/`.

**Data credits:**
- nebius/SWE-agent-trajectories (CC BY 4.0).
- TraceLab, UW SyFI Lab (CC BY 4.0), [paper](https://arxiv.org/abs/2606.30560).
