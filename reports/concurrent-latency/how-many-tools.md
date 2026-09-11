# How many tools should the evaluation present?

`findings.md` reports results at k = 4/16/64/128, where **k is the number of
tool definitions placed in the model's prompt**. Two objections apply to that
choice, and both are fair on their face:

1. A task typically needs **1.77 tools** (mean over 7,961 gold-labelled
   ToolRet tasks; 55% need exactly one). The model actually *calls* 0.7-1.1
   tools per request — about one, at every menu size and model tested (§3).
   So at k128, roughly 126 of the 128 tools shown are irrelevant to that
   request. (The 200-task slice these experiments run on averages **1.54** —
   see §6, gap 3: it is an unrepresentative draw.)
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

### How many tools the model actually calls

The objection's premise, measured directly — not what tasks *need* (1.54 on
this slice) but what the model *does*. Tool calls per request, `tooltrie_v1`,
dense retrieval:

| menu | model | 0 calls | 1 call | 2 calls | 3+ | mean |
|---|---|---|---|---|---|---|
| 4 | 0.6B | 48% | 34% | 16% | 2% | 0.70 |
| 10 | 0.6B | 42% | 38% | 18% | 2% | 0.79 |
| 16 | 0.6B | 40% | 40% | 18% | 1% | 0.80 |
| 64 | 0.6B | 34% | 46% | 16% | 2% | 0.88 |
| 128 | 0.6B | 34% | 48% | 15% | 3% | 0.90 |
| 128 | 4B | 25% | 50% | 20% | 4% | 1.06 |
| 128 | 8B | 18% | 58% | 22% | 2% | 1.10 |

- **Menu size barely changes how many tools get called.** A 32x larger menu
  moves the mean from 0.70 to 0.90. Showing more tools does not make the model
  call more; the task decides that, not the menu.
- **Neither does the ordering policy.** At k64/0.6B the mean is 0.85-0.89
  across `original`, `tooltrie_v0`, ContextPilot and `tooltrie_v1`.
- **The model under-calls.** Tasks need 1.54 tools on average and even the
  best case (8B, 128 tools) calls 1.10. The largest single failure is calling
  nothing at all: 18-48% of requests.

**"Called nothing" changes meaning with menu size** — the most direct evidence
for the two failure modes above. Zero-call requests, split by whether the right
tool was in the menu at all:

| menu | model | zero-call requests | right tool **absent** | right tool **present** |
|---|---|---|---|---|
| 4 | 0.6B | 97 | **62%** | 38% |
| 16 | 0.6B | 81 | 44% | 56% |
| 64 | 0.6B | 69 | 10% | **90%** |
| 128 | 0.6B | 67 | 10% | **90%** |
| 4 | 4B | 79 | **56%** | 44% |
| 128 | 4B | 50 | 10% | **90%** |

At a small menu most "called nothing" requests had no correct tool to call:
declining was the sensible move, and the failure is retrieval's. At a large
menu, 90% had the right tool in front of the model and it went unused: the
failure is the model's, among the distractors. **The failure does not disappear
as the menu grows — it moves**, from "retrieval didn't find it" to "the model
didn't notice it." That is the mechanism behind the flat end-to-end accuracy
above: each end of the menu-size range fails, for a different reason.

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

### 5.3 The same design on BM25 — the result replicates and strengthens

§5.2 is dense-retrieval only, so the field-comparability claim rested on one
retriever. Repeated on BM25 at the identical design (retrieve 64, present 10;
six arms; 12 accuracy runs, 0 failures).
Runs: `cluster/results/eval-bm25trunc-20260908-000327/`.

**The ceiling needs no model to compute** — it is a property of which tools
survive truncation — so this half was answerable directly:

| arm | BM25 ceiling | dense ceiling (§5.2) |
|---|---|---|
| `original` | 0.590 | 0.620 |
| **`tooltrie_v1`** | **0.595 (+0.5pp)** | 0.615 (-0.5pp) |
| `frequency` | 0.415 | 0.385 |
| ContextPilot | **0.395 (-19.5pp)** | 0.470 (-15.0pp) |
| `tooltrie_v0` | 0.115 | 0.080 |
| `alphabetical` | 0.115 | 0.075 |

**ContextPilot's ceiling damage is larger on BM25, not smaller** — -19.5pp
against -15.0pp — and `tooltrie_v1` moves from -0.5pp to **+0.5pp**, marginally
*above* the unreordered baseline. The plausible mechanism is BM25's heavier hub
structure (its most-retrieved tool appears in 66 of 200 menus against dense's
21): ContextPilot's clustering has more cache-friendly hub material to hoist,
so it displaces more relevant tools when the list is cut.

**End-to-end (ceiling x F1), scored with `scripts/score_end_to_end.py`:**

| arm | ceiling | 0.6B F1 | **0.6B end-to-end** | 4B F1 | **4B end-to-end** |
|---|---|---|---|---|---|
| `original` | 0.590 | 38.98 | **23.00** | 49.01 | **28.92** |
| **`tooltrie_v1`** | **0.595** | 36.97 | **22.00** | 47.20 | **28.08** |
| `frequency` | 0.415 | 54.62 | 22.67 | 67.27 | 27.92 |
| ContextPilot | 0.395 | 41.98 | **16.58** | 56.54 | **22.33** |
| `tooltrie_v0` | 0.115 | 30.00 | 3.45 | 49.28 | 5.67 |
| `alphabetical` | 0.115 | 21.30 | 2.45 | 49.28 | 5.67 |

**Same conclusion, larger margins.** ContextPilot loses **6.4pp end-to-end at
0.6B and 6.6pp at 4B** against `original` (dense: 3.6 and 5.3). `tooltrie_v1`
tracks `original` within 1.0pp on both models while holding the ceiling.

**And BM25 sharpens the metric point.** `frequency` posts the highest
conditional F1 of any arm on either retriever (54.62 / 67.27) — it *looks* like
the best policy in the table — and its end-to-end is merely ordinary, because
its ceiling is 0.415. ContextPilot shows the same pattern more mildly: second-
highest conditional F1 at 4B (56.54), worst end-to-end of the four viable arms.
**Ranking these arms by conditional F1 would recommend the two policies that
throw away the most user requests.**

### 5.4 Under bursty arrival: the gap widens

§5.2-5.3 used the natural arrival order. Repeated under the upper-bound locality
ordering from `findings.md` §4.5 (adjacent requests share 17.5 of 64 tools
rather than 1.8), end-to-end F1:

| retriever / arrival | `original` | **`tooltrie_v1`** | ContextPilot | v1 - CP |
|---|---|---|---|---|
| dense, natural | 19.83 | 19.92 | 16.25 | +3.67 |
| dense, **locality_max** | 19.83 | 19.08 | **12.42** | **+6.66** |
| BM25, natural | 23.00 | 22.00 | 16.58 | +5.42 |
| BM25, **locality_max** | 23.00 | 20.92 | **15.08** | **+5.84** |

**ContextPilot's damage roughly doubles on dense** (-3.6 → -7.4pp against
`original`): with more shared material to cluster, it hoists more and pushes the
right tool out of the window more often. **v1 is no longer free** at the upper
bound — -0.75pp dense, -2.08pp BM25 — because high locality gives the trie more
matches to hoist, and some displace a relevant tool. "Nothing lost" becomes "a
little lost, far less than ContextPilot." The model-free ceiling puts v1 ahead
of ContextPilot by **+8.5 to +22.0pp** in all six arrival regimes, both
retrievers.

---

## 6. What k should the evaluation use?

**Three answers, because there are three different questions.**

| question | right k | why | what we found |
|---|---|---|---|
| Comparability with the field | **k=10** | BFCL, ToolBench, ToolRet, LiveMCPBench all sit at 5-11. ToolRet's own downstream number (39.2%) is at k=10. | Comparable, but ordering is a null there: p50 66-68 ms for all six arms, accuracy within 1 SE, while v1 still has the most reuse (13.71% vs ContextPilot 12.46%) (§5.1). |
| Deployment realism (no router) | **k≈150** | The commonly cited MCP composite; sits between the k64 and k128 already run. | Not run at 150 (gap 2). The bracketing k64/k128 both go to v1: 2.09x/3.09x ContextPilot's reuse on dense, F1 +5.52/+9.20 at 0.6B, tied at 8B (§3, `findings.md` §4.4). |
| Studying **ordering** at all | **large, or truncated** | §5.1 is a null at k=10. The effect requires either a long prompt (k64/128) or a truncation boundary (§5.2). | Retrieve-64/present-10 gives the sharpest separation: ceiling v1 0.615 vs ContextPilot 0.470 dense, 0.595 vs 0.395 BM25 (§5.2-5.4). |

**Recommended design, and it is the one this document validates: retrieve
large, present small.** §5.2 is the only condition that is simultaneously
comparable to the field (10 tools shown), realistic (64-tool candidate pool
behind it), and *discriminating* (a 15-20pp ceiling gap between policies —
3.0 sigma on dense, 4.0 on BM25 — which holds on both retrievers and widens
under bursty arrival, §5.3-5.4). It also measures the thing that actually matters in a routed
deployment — not "does reordering slow the prompt down" but **"does
cache-oriented reordering push relevant tools out of the window the model
sees."**

**Concrete gaps this leaves:**

1. ~~No k=10 on BM25~~ — **closed by §5.3.** The truncation design was
   repeated on BM25: the result replicates with *larger* margins
   (ContextPilot -19.5pp ceiling against dense's -15.0pp). The plain k=10
   control (§5.1) was not repeated, since its null is driven by prompt length
   rather than retriever and would replicate trivially.
2. **No k≈150 point**, the single most commonly cited real composite.
3. **The 200-task slice is an unrepresentative draw — now re-run; the
   headline comparison survives it, the full ranking does not.** Taking
   `offset 0, limit 200` yields a mean of **1.54 gold tools per task**, against
   **1.77** corpus-wide over all 7,961 gold-labelled tasks; random 200-task
   slices fall in **1.64-1.91** (5th-95th percentile, 200 resamples), so 1.54
   sits **below the 5th percentile**. ToolRet's own paper states 2.17, further
   still. The k64 cells were re-run on a random slice (`--sample-seed 2026`;
   `findings.md` §4.5). That draw is harder on both counts — 1.94 gold tools per
   task, retrieval hit@64 70.5% against 81.5% — and on it **end-to-end accuracy
   falls 30-35% for `original`, v1 and ContextPilot** (`tooltrie_v1` 22.0 →
   14.3), but only ~15% for `frequency` and v0/alphabetical, whose F1 barely
   moves. Random-slice end-to-end: `original` 14.38, `tooltrie_v1` 14.29,
   `frequency` 13.93, ContextPilot 12.20, `tooltrie_v0` and `alphabetical`
   11.35. **v1 still beats ContextPilot**, by about half its original margin,
   and v0/alphabetical stay last. But v1's small lead over `original` becomes a
   tie, its lead over `frequency` nearly vanishes, and **ContextPilot falls
   from ahead of `frequency` to behind it** — so accuracy comparisons among the
   middle arms are slice-dependent. The random draw sits just above the 95th
   percentile of difficulty, so the true inflation against a typical slice is
   smaller than a third — but every absolute figure in this document is
   optimistic, and so are the margins between the leading arms.
4. **Ceiling-conditioned accuracy is still the default in the older tables —
   partly addressed.** §3 and §5.2 both show it can invert a conclusion.
   `scripts/score_end_to_end.py` now emits end-to-end F1 (ceiling x F1), the
   reporting order is set in `metrics-and-latency-tradeoffs.md` §1.3, and every
   truncation result here (§5.2-5.4) uses it. **Still open:** the accuracy
   columns in `findings.md` §1.4 and the README are `gold_hit_ceil` under the
   plain label "accuracy". Those tables compare policies on identical menus,
   so the ceilings match and the *ranking* is unaffected — but the absolute
   numbers are ceiling-conditioned and the label does not say so.
5. **Single-turn only.** BFCL v4, τ²-bench, LiveMCPBench and MCP-Bench are all
   multi-turn now; retrieval-error compounding across turns is untested here.

---

## Runs

| stage | directory | runs drawn on |
|---|---|---|
| k64/k128 dense accuracy: all four arms at k64/0.6B, `tooltrie_v1` at k128 (§3) | `dense-accuracy-20260903-002742/` | 6 |
| k4/k16 dense accuracy, 0.6B and 4B (§3-4); `tooltrie_v1` at 8B/k128 (§3) | `eval-validity-20260906-165826/` | 17 |
| k=10 and retrieve-64/present-10, dense: reuse/latency (0.6B) + accuracy (0.6B, 4B) (§3, §5.1-5.2) | `eval-gaps-20260907-224433/` | 36 |
| retrieve-64/present-10 on BM25: accuracy (0.6B, 4B), six arms (§5.3) | `eval-bm25trunc-20260908-000327/` | 12 |
| retrieve-64/present-10 under bursty arrival (§5.4); random-slice re-run (§6, gap 3) | `eval-locality-20260910-231437/` | 12 |

Drivers: `scripts/replay_vllm_concurrent.py` (rate-controlled) and
`scripts/replay_vllm_workload.py` (serial, accuracy). Scoring:
`scripts/score_tool_selection.py` and `scripts/score_end_to_end.py` (the
latter emits ceiling, precision, recall, F1 and end-to-end together). Companion document:
[`metrics-and-latency-tradeoffs.md`](metrics-and-latency-tradeoffs.md).

Sources: [LiveMCPBench](https://arxiv.org/abs/2508.01780) ·
[ToolRet](https://arxiv.org/abs/2503.01763) ·
[BFCL v4 dataset guide](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/BFCL_v4_dataset_guide.md) ·
[τ²-bench](https://github.com/sierra-research/tau2-bench) ·
[MCP-Bench](https://arxiv.org/abs/2508.20453) ·
[Benchmarking the Benchmarks](https://arxiv.org/abs/2607.02577)
