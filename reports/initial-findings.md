# Initial research findings and recommendation

## Result in one paragraph

The public workloads contain a measurable analytical prefix-locality signal,
but it is workload- and evidence-dependent. Across 45,815 canonical
tools, median schema length is 70 Qwen tokens and P95 is
220. ToolRet's median relevance set and BFCL's median
exposed menu both contain one tool, which limits what reordering can achieve on
many tasks. On empirical dataset order, schema-cost-weighted ordering raises the
ToolRet tool-prefix block-reuse estimate from
26.91% to
31.80% (4.89
percentage points). Frequency ordering raises the BFCL menu estimate from
10.98% to
18.88% (7.90
points). These are analytical estimates, not vLLM hit-rate or latency results.

## Evidence obtained locally

- ToolRet: 7,961 tasks, 7,652
  tools appearing in gold relevance labels, with every label resolved.
- BFCL subset: 1,240 tasks and
  1,362 distinct canonical functions appearing in
  exposed menus.
- Schema diagnostics flag 19,921/45,815 tools. Most flags are
  empty parameter objects, which can be legitimate; malformed and missing-field
  counts are reported separately.
- Controlled support-skewed replay produces much more reuse than uniform or
  empirical access, confirming that any systems claim must state its workload
  locality rather than assume production-like skew.
- Classic FP-tree global order equals the frequency order in this baseline.
  Conditional pattern mining is still untested.

## Prefix-cache sanity status

Measured on an RTX 3090 (23.56 GiB) with vLLM 0.26.0 and `Qwen/Qwen3-0.6B`,
block size 16, 11,807 GPU blocks. Five trials per scenario with the prefix cache
reset before each trial (`cluster/results/`). Prompt length is 303 tokens except
for `changed_second_tool` at 309.

| Check | Cached prompt tokens | Reuse | TTFT on (ms) | TTFT off (ms) |
| --- | --- | --- | --- | --- |
| Cold prompt | 0 / 303 | 0.0% | 52.4 ± 28.8 | 51.5 ± 27.3 |
| Identical prompt reuse | 288 / 303 | 95.0% | 43.0 ± 1.2 | 40.5 ± 2.3 |
| Changed second tool | 128 / 309 | 41.4% | 41.4 ± 1.1 | 42.3 ± 2.2 |
| Reordered first two tools | 48 / 303 | 15.8% | 37.7 ± 5.7 | 43.0 ± 4.2 |
| Original restored | 288 / 303 | 95.0% | 41.1 ± 2.1 | 39.4 ± 2.8 |

All five Task B checks pass. Reuse behaves exactly as the exact-prefix model
predicts: an identical prompt reuses 95% of it, editing one tool description
cuts reuse to 41%, and reordering the first two tools cuts it to 16%. Cache-on
and cache-off produce identical output text and tool calls in every scenario,
with the control verified at `enable_prefix_caching=False` and 0 cached tokens
throughout. Peak KV-cache usage per request is 0.0019% of an 11,807-block
cache, and mean inter-token latency is ~2.5 ms across all scenarios.

**Prefix reuse produced no measurable TTFT benefit at this prompt size.**
Intervals above are 95% Student-t half-widths over five trials; no scenario
separates cache-on from cache-off, and reuse fraction does not order TTFT — the
15.8%-reuse case has the lowest mean. This is a measurement-floor result rather
than evidence against the hypothesis: 303 tokens prefill in ~40 ms end-to-end,
dominated by fixed per-request overhead, so eliminating 288 tokens of prefill
work is not resolvable above noise.

A single-trial version of this experiment appeared to show a 3x gain (69 ms cold
vs 23 ms warm). That was entirely the server's first-ever request: per-trial cold
TTFT is 93.9, 41.8, 42.7, 42.5, 41.3 ms. Any latency claim in this project needs
repeated trials and a discarded warmup.

## Recommendation

Proceed with the exact prompt-level ToolTrie baseline on the cluster, focusing
on multi-tool tasks and session-local replay. Use schema-cost-weighted ordering
as the strongest ToolRet candidate and frequency ordering as the strongest BFCL
candidate from this first analysis. Report the original order and fixed-random
controls alongside them.

The likely publishable refinement is not a generic “reorder context into a
trie” claim, because closely related cache-aware context ordering already
exists. The more defensible direction is tool-specific cache admission using
schema cost plus workflow co-occurrence, while preserving an active/authorized
tool manifest and measuring function-call quality.

Do not pursue arbitrary independent KV concatenation yet. First establish that
native exact APC converts the local token-reuse signal into repeatable TTFT
benefit without a BFCL quality regression. If it does not, narrow the project to
characterizing the crossover regimes or pivot toward retrieval/menu reduction.

Task B has now partly answered that, negatively, at small scale: 95% token reuse
bought no measurable TTFT at a 303-token prompt. Before any ordering experiment
can produce an interpretable latency number, Task E must first find the prompt
size at which schema prefill dominates TTFT. The concrete next measurement is a
prefill-cost sweep — TTFT against tool-catalog token count, cache-on and
cache-off, holding the request shape fixed — to locate the point where the
cache-on and cache-off curves separate beyond their confidence intervals. Every
ordering comparison should then be run above that threshold. Running the
existing workload orderings at 303-token prompts would produce differences
indistinguishable from noise and invite exactly the over-claim the single-trial
run already produced once.

## Immediate next GPU run

Done: model/vLLM/GPU/block size frozen and recorded (step 1), and the
cache-enabled and cache-disabled sanity probes both run and compared (step 2).
Remaining:

1. Sweep TTFT against tool-catalog token count, cache-on and cache-off, to find
   where schema prefill dominates TTFT. No ordering result is interpretable
   below that threshold.
2. Render and save complete prompt token IDs for each ordering.
3. Replay original, frequency, and schema-cost-weighted workloads above the
   threshold, with documented cold/warm policies and repeated trials.
4. Add BFCL name/argument/no-tool scores before interpreting latency.
