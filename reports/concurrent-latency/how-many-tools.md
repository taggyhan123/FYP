# How many tools should the evaluation present?

`findings.md` reports results at k = 4/16/64/128, where **k is the number of
tool definitions placed in the model's prompt**. Two objections apply to that
choice, and both are fair on their face:

1. A task typically needs **1.77 tools** (mean over 7,961 gold-labelled
   ToolRet tasks; 55% need exactly one). The model actually *calls* ~0.9-1.0
   tools per request. So at k128, roughly 126 of the 128 tools shown are
   irrelevant to that request. (The 200-task slice these experiments run on
   averages **1.54** — see §6, gap 3: it is an unrepresentative draw.)
2. **No published tool-use benchmark presents a menu that large.** They cap at
   2-11 tools, or retrieve top-5 to top-10 from a large catalog.

This document works out whether the large-k choice is defensible anyway, and
what k a valid evaluation should use. Every claim is a dated literature
citation or a run under `cluster/results/`. A plain-language version without
the statistics is in
[`how-many-tools-summary.md`](how-many-tools-summary.md).

**Answer in one paragraph.** The premise is right — tasks use very few tools —
but it argues for *better retrieval*, not a *smaller menu*, because the two
have different failure modes and shrinking the menu trades a recoverable
problem for an unrecoverable one. Measured end-to-end, accuracy is **flat**
across k=4 to k=128 (20.5 to 22.4), so there is no accuracy case for either
end. The real reason the project needs large k is that **the effect it studies
does not exist at small k**: at k=4 and k=10 every ordering policy ties on
latency. And the real reason large k is *legitimate* is that the field's small
menus come from a per-query router that most MCP deployments do not have —
150-tool composites are ordinary. But §5 finds the sharpest result of all in
the design that reconciles both: retrieve large, **present small**.

---

## 1. What the field presents, and what it retrieves from

Every benchmark separates two numbers this project collapses into one `k`: the
**catalog** the retriever searches, and the **menu** the model is shown.
Checked 2026-09-07.

| benchmark | catalog | menu shown to the model | gold tools/task |
|---|---|---|---|
| BFCL v4 `simple` | — (no retrieval) | **1** | 1 |
| BFCL v4 `multiple` | — | **2-4** | 1 |
| BFCL v4 `live_multiple` | — | **2-5** | 1 |
| τ²-bench (airline) | 11 per domain | **11**, fixed | multi-step |
| ToolBench / ToolLLM | 16k APIs | retriever **top-5 to 10** | — |
| **ToolRet** (this project's corpus) | **43k** | **k=10** | **2.17** |
| LiveMCPBench | 527 tools / 70 servers | router, **k=5** | — |
| MCP-Bench | 250 tools / 28 servers | retrieved subset | multi-hop |
| **this project** | **44,453** | **4 / 16 / 64 / 128** | **1.77** |

Two datapoints decide the "what is standard" question:

- **LiveMCPBench ablates menu size directly** on a 527-tool catalog:
  k=1 → 64.21%, **k=5 → 78.95%**, k=10 → no further gain (p=1.00). Retrieval
  errors are ~50% of all its failures — the dominant bottleneck.
- **ToolRet's own authors** — same corpus this project uses — run their
  downstream tool-use experiment at **k=10** (gold tools: 50.6% pass rate;
  retrieved@10: 39.2%, an 11.4-point drop).

**So the field's answer is unambiguous: catalog as large as you like, present
the model about 5-10.** This project's k=64/128 shows a menu 6-25x larger than
any benchmark hands a model. That is a genuine, previously unstated divergence.

---

## 2. What real deployments carry

Checked 2026-09-06, dated because this area moves quickly.

| finding | source |
|---|---|
| Best practice per MCP server: **~20-30 tools** | [Lunar.dev](https://www.lunar.dev/post/why-is-there-mcp-tool-overload-and-how-to-solve-it-for-your-ai-agents) |
| Common composite: 5 servers x 30 tools = **150 tools**, 30,000-60,000 tokens of metadata (25-30% of a context window) | [Lunar.dev](https://www.lunar.dev/post/why-is-there-mcp-tool-overload-and-how-to-solve-it-for-your-ai-agents) |
| Cloudflare's native MCP exposure: **~1.17M tokens** of tool definitions | [Agentive Futures](https://medium.com/agentive-futures/mcp-in-finance-is-great-until-you-need-1-000-tools-d09fc350a85e) |
| Cursor imposes a hard cap of **~40 tools**, specifically because of tool overload | [getunblocked.com](https://getunblocked.com/blog/mcp-tool-overload/) |
| Accuracy collapse from tool count alone: 78% → 13.62% (10 → 100+ tools); 43% → 2% (4 → 51 tools, BFCL calendar) | search-verified 2026-09-06 |

**This cuts against the benchmarks, not with them.** k=3-20 describes *one
well-scoped MCP server in isolation*. The moment a handful of ordinary
integrations are composed — GitHub + Slack + a database + a calendar, nothing
exotic — the injected tool count lands near **150**, which sits *between* k64
and k128. Cloudflare's disclosed case exceeds k128 by an order of magnitude.
Cursor's hard 40-cap exists precisely because overload is already an
operational problem in the field.

**The reconciliation between §1 and §2**, which neither literature nor this
project stated before: the benchmarks' small menus are the output of a
**per-query router**. LiveMCPBench needed a purpose-built MCP Copilot Agent to
get to k=5. Most MCP setups have no such router and inject the whole composite
catalogue. Both regimes are real; they are different problems, and an
evaluation should say which one it is testing.

---

## 3. Why a smaller menu is not simply "cleaner"

The intuition behind "most tasks use 1-2 tools, so why show 128" assumes the
system knows *which* 1-2. It does not — that is what retrieval is guessing at,
imperfectly. Cutting the menu therefore trades one failure mode for a worse
one.

**Retrieval hit-rate — whether the gold tool is in the menu at all — measured
on this project's dense retriever:**

| k | hit-rate | what it means |
|---|---|---|
| 4 | **52.5%** | nearly half of queries are unwinnable regardless of model or ordering |
| 16 | 67.5% | |
| 64 | 81.5% | |
| 128 | **85.5%** | |

At k=4 the correct tool is simply **absent** in 47.5% of requests. That is
unrecoverable — no model, however capable, and no ordering policy, however
good, can call a tool it was never shown. A large menu's failure mode
(the tool is present but buried among distractors) is recoverable by a better
model or better ordering; a small menu's failure mode is not.

**And the two effects very nearly cancel.** Reported accuracy in this project
is *ceiling-conditioned* — computed only over requests where the gold tool was
actually present. Multiplying that by the hit-rate above gives end-to-end
accuracy, which is what a deployment actually experiences:

| k | hit-rate | conditional F1 (`tooltrie_v1`, 0.6B) | **end-to-end** |
|---|---|---|---|
| 4 | 0.525 | 39.05 | **20.50** |
| 16 | 0.675 | 33.21 | **22.42** |
| 64 | 0.815 | 26.99 | **22.00** |
| 128 | 0.855 | 25.38 | **21.70** |

**Flat — 20.5 to 22.4 across a 32x range of menu size.** Smaller k finds the
tool more cleanly when it is there but misses it far more often; larger k
almost always has it but the model picks it out more messily. **There is no
end-to-end accuracy argument for choosing either end.** The choice must be
made on other grounds: latency and throughput cost (which favour small k), and
deployment realism (which, per §2, favours large).

**One clarification that matters for reading every number above.** F1 and
precision are *not* computed against menu size. Precision is correct calls ÷
calls made; recall is correct calls ÷ gold tools needed. Menu size appears in
neither. If a model correctly calls 2 tools out of a 3-tool menu and out of a
64-tool menu, F1 is 100% in both cases. The observed accuracy decline with k
(≈39 → 26, about 35% relative) is therefore a **real behavioural effect** —
the model gets distracted, calls the wrong tool, or calls nothing — not an
arithmetic artefact of a bigger denominator. A "correct ÷ menu size" reading
would predict a ~20x collapse; the measured decline is nothing like that.

---

## 4. Does the advantage hold at small k? (k=4 and k=16)

Before testing the field's menu size directly, the project's own shallow
depths already bracket ContextPilot's native k=3-20 range.

**Reuse and latency, both retrievers:**

| k | retriever | v1 reuse | CP reuse | v1 p50 | CP p50 | v1 max | CP max |
|---|---|---|---|---|---|---|---|
| 4 | BM25 | 19.47% | 18.72% | 46.1 | 46.3 | 93.0 | **90.0** |
| 4 | dense | 17.97% | **18.16%** | 46.2 | 46.3 | **78.5** | 87.7 |
| 16 | BM25 | **11.31%** | 9.93% | **92.1** | 98.2 | **323.4** | 336.3 |
| 16 | dense | **13.18%** | 11.58% | 89.0 | 89.0 | **283.9** | 285.2 |

**At k4 it is a wash** — the arms trade the lead across retrievers and
statistics, and ContextPilot wins `max` on BM25. **At k16 v1 has a real but
modest edge** (1.14x reuse on both retrievers), far from the 2.09-3.09x margin
at k64/k128. The decisive part of the dense result is a large-k phenomenon.

**Accuracy (`gold_hit_ceil`), dense, four arms, both models.** No accuracy run
had ever been done at k4 or k16 before this work — the prior accuracy batches
(`accuracy-gate-20260830-011416`, `dense-accuracy-20260903-002742`) both start
at k64:

| cell | v1 | ContextPilot | v0 | `original` | ceiling |
|---|---|---|---|---|---|
| k4 @ 0.6B | 42.86 | 41.90 | 42.86 | **43.81** | 0.525 |
| k4 @ 4B | 55.24 | 54.29 | 54.29 | **56.19** | 0.525 |
| k16 @ 0.6B | **38.52** | 34.81 | 28.15 | 37.78 | 0.675 |
| k16 @ 4B | **44.44** | 40.00 | 40.00 | **44.44** | 0.675 |

**v1 beats ContextPilot in all four cells** (+0.96, +0.95, +3.71, +4.44 pp), so
inside ContextPilot's own native range the accuracy advantage does hold. But
k4's margins are noise (0.14 sigma) and even k16 is under 1 SE alone.

**`original` — no reordering at all — wins outright at k4 on both models**
(43.81 vs 42.86; 56.19 vs 55.24). At that depth the retriever's own ranking is
already close to optimal and *any* reordering can only disturb it. k4 is the
one depth where not reordering is the right call on both the cache axis and
the accuracy axis.

---

## 5. Does the project's result survive at the field's menu size?

The decisive test. Two designs, dense retrieval, six arms, 0.6B for
reuse/latency (rate 10, uncapped) and 0.6B + 4B for accuracy. 36 runs,
0 failures. Runs: `cluster/results/eval-gaps-20260907-224433/`.

### 5.1 k=10, control intact — retrieve top-10, reorder those 10, present all 10

Same ten tools in every arm (ceiling 0.620 throughout), only the order differs.

| arm | reuse | p50 ms | p95 ms | 0.6B F1 | 0.6B end-to-end | 4B F1 | 4B end-to-end |
|---|---|---|---|---|---|---|---|
| `original` | 6.38% | 66.2 | 121.8 | 31.99 | 19.83 | 43.12 | 26.73 |
| `alphabetical` | 7.97% | 66.5 | 124.9 | 31.85 | 19.75 | 37.50 | 23.25 |
| `frequency` | 8.68% | 66.8 | 127.5 | 30.91 | 19.17 | **44.81** | **27.78** |
| `tooltrie_v0` | 9.94% | 68.1 | 127.9 | 31.85 | 19.75 | 37.50 | 23.25 |
| ContextPilot | 12.46% | 67.1 | 130.6 | **32.39** | **20.08** | 42.80 | 26.53 |
| **`tooltrie_v1`** | **13.71%** | 67.1 | **117.3** | 31.59 | 19.58 | 43.66 | 27.07 |

**The reuse ranking holds — v1 first, 2.15x `original`, 1.10x ContextPilot —
and it is worth nothing.** p50 is flat at 66-68 ms across all six arms: a
ten-tool prompt is too short for prefix reuse to move latency, and at this
rate no queue forms. Accuracy is a wash on both models (every gap under 1 SE).

**At the field's menu size the mechanism survives and the effect does not.**
This matches k=4 and k=16 exactly, and it is the honest cost of the
divergence identified in §1: had this project used the field's k, it would
have measured a null.

### 5.2 Retrieve 64, present 10 — the design that reconciles both regimes

Each arm reorders the same 64 retrieved tools; only the **first 10** are shown.
The tools shown now differ by arm, so **ceiling** — whether the gold tool
survives into the window — is the primary result, and end-to-end
(ceiling x F1) is the honest accuracy number. Conditional F1 over a tiny
surviving subset is shown only to make that point.

| arm | ceiling | reuse | p50 ms | 0.6B F1 | **0.6B end-to-end** | 4B F1 | **4B end-to-end** |
|---|---|---|---|---|---|---|---|
| `original` | 0.620 | 6.38% | 69.0 | 31.99 | 19.83 | 43.12 | 26.73 |
| **`tooltrie_v1`** | **0.615** | **27.64%** | **60.1** | 32.38 | **19.92** | 42.33 | 26.03 |
| ContextPilot | 0.470 | 20.16% | 64.1 | 34.57 | 16.25 | 45.64 | 21.45 |
| `frequency` | 0.385 | 9.39% | 72.8 | 52.16 | 20.08 | 63.29 | 24.37 |
| `tooltrie_v0` | 0.080 | 16.12% | 67.1 | 9.38 | 0.75 | 44.79 | 3.58 |
| `alphabetical` | 0.075 | 11.67% | 68.5 | 10.00 | 0.75 | 47.78 | 3.58 |

**This is the sharpest separation between the ordering policies anywhere in
the project, and it is at the field's menu size.** Truncation exposes what
each policy does to the retriever's ranking:

- **`tooltrie_v1` keeps the ceiling intact** — 0.615 against `original`'s
  0.620, a 0.5pp loss (0.1 sigma) — while gaining **4.3x the reuse and 13%
  lower p50**, with end-to-end accuracy unchanged on both models. Its rule
  (hoist only what the trie has evidence for, leave everything else in
  retriever order) is exactly what makes it safe under truncation.
- **ContextPilot drops the ceiling 15pp** (0.620 → 0.470, **3.0 sigma**). Its
  cluster hoist pulls cache-friendly tools into the window and pushes the gold
  tool out of it in **one request in seven**. End-to-end falls 3.6pp at 0.6B
  and **5.3pp at 4B**.
- **`tooltrie_v0` and `alphabetical` collapse** (ceiling 0.08). The
  alphabetical fallback puts arbitrary tools first, so the shown ten are
  almost never the relevant ones — 16% reuse bought at a 92% miss rate.
- **`frequency` shows why ceiling-conditioned F1 misleads**: the highest
  conditional F1 in the table (52/63), because the gold tools surviving its
  truncation are the common, easy ones — yet its ceiling is 0.385, so its
  end-to-end is no better than `original`'s.

---

## 6. What k should the evaluation use?

**Three answers, because there are three different questions.**

| question | right k | why |
|---|---|---|
| Comparability with the field | **k=10** | BFCL, ToolBench, ToolRet, LiveMCPBench all sit at 5-11. ToolRet's own downstream number (39.2%) is at k=10. |
| Deployment realism (no router) | **k≈150** | The commonly cited MCP composite; sits between the k64 and k128 already run. |
| Studying **ordering** at all | **large, or truncated** | §4.1 is a null at k=10. The effect requires either a long prompt (k64/128) or a truncation boundary (§4.2). |

**Recommended design, and it is the one this document validates: retrieve
large, present small.** §5.2 is the only condition that is simultaneously
comparable to the field (10 tools shown), realistic (64-tool candidate pool
behind it), and *discriminating* (a 15pp, 3-sigma ceiling gap between
policies). It also measures the thing that actually matters in a routed
deployment — not "does reordering slow the prompt down" but **"does
cache-oriented reordering push relevant tools out of the window the model
sees."**

**Concrete gaps this leaves:**

1. **No k=10 on BM25** — §5 is dense only, so the field-comparability claim
   rests on one retriever.
2. **No k≈150 point**, the single most commonly cited real composite.
3. **The 200-task slice is an unrepresentative draw, and by more than
   "slightly".** Taking `offset 0, limit 200` yields a mean of **1.54 gold
   tools per task**, against **1.77** corpus-wide over all 7,961 gold-labelled
   tasks. Random 200-task slices fall in **1.64-1.91** (5th-95th percentile,
   200 resamples), so 1.54 sits **below the 5th percentile** — the evaluated
   slice is materially easier than a representative sample, not marginally so.
   ToolRet's own paper states 2.17, further still. This does **not** bias any
   arm-vs-arm comparison, since every arm replays the identical slice, but it
   does mean the absolute accuracy numbers are optimistic and not directly
   comparable to ToolRet's published figures. Re-running the k64 cells on a
   random slice would settle the size of the inflation.
4. **Ceiling-conditioned accuracy is still the default** everywhere else in
   the report. §3 and §5.2 both show it can invert a conclusion; end-to-end
   (ceiling x F1) should be primary.
5. **Single-turn only.** BFCL v4, τ²-bench, LiveMCPBench and MCP-Bench are all
   multi-turn now; retrieval-error compounding across turns is untested here.

---

## Runs

| stage | directory | runs |
|---|---|---|
| k4/k16 dense accuracy, two models | `eval-validity-20260906-165826/` | 16 |
| k=10 and retrieve-64/present-10: reuse/latency (0.6B) + accuracy (0.6B, 4B) | `eval-gaps-20260907-224433/` | 36 |

Drivers: `scripts/replay_vllm_concurrent.py` (rate-controlled) and
`scripts/replay_vllm_workload.py` (serial, accuracy). Scoring:
`scripts/score_tool_selection.py` plus an F1/end-to-end re-scorer not yet
promoted to a script. Companion document:
[`metrics-and-latency-tradeoffs.md`](metrics-and-latency-tradeoffs.md).

Sources: [LiveMCPBench](https://arxiv.org/abs/2508.01780) ·
[ToolRet](https://arxiv.org/abs/2503.01763) ·
[BFCL v4 dataset guide](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/BFCL_v4_dataset_guide.md) ·
[τ²-bench](https://github.com/sierra-research/tau2-bench) ·
[MCP-Bench](https://arxiv.org/abs/2508.20453) ·
[Benchmarking the Benchmarks](https://arxiv.org/abs/2607.02577)
