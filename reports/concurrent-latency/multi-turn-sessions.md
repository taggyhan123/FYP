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
- **For multi-turn agents, the design advice is:**
  - append new tools after the history rather than rebuilding the tool block;
  - keep already-loaded tools;
  - give the cache enough memory for the sessions that are between turns.

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
- **This is a simulation of hits, not a latency measurement.** A GPU replay is
  planned but has not been run.
- **Model calls are replayed one at a time**, in the order a concurrent system
  would serve them. Batching effects are not modelled.
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

All of it is CPU only. The results in this document are in
`cluster/results/multiturn-sessions-20260911-234046/`, together with the
pre-declaration, plans and logs.

**Data credits:**
- nebius/SWE-agent-trajectories (CC BY 4.0).
- TraceLab, UW SyFI Lab (CC BY 4.0), [paper](https://arxiv.org/abs/2606.30560).
