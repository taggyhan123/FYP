# Middle-overlap sweep — stated before the runs

## Why

Every workload in this report sits at one of two extremes. Padded menus share
**98.4%** of their tools between any two requests; BM25-retrieved menus share
**0%** outright and overlap 21–33% with their best partner. Nothing exists
between, and the report lists that gap as its top unsettled item.

It matters because the two methods have different mechanisms. ContextPilot
clusters on **set overlap** and hoists the intersection, so it finds a shared
core in one step. ToolTrie needs **exact prefix agreement**, a stricter
condition, so it has to accumulate one. Padded menus hand everyone total
agreement (clustering wins on speed, request 2 against 222); retrieved menus
give almost none (nobody agrees, ContextPilot wins on every depth). The middle
is the only untested place where the stricter mechanism might pay.

## Construction

Built from `bfcl-padded64`, changing one variable. Each request keeps its own
task-specific tool and a menu of 64. Of the remaining 63, **C** come from a fixed
core shared by every request and **63−C** are drawn per-request from the
45,815-tool corpus, disjoint across requests. So tools shared by all = C/64.

C ∈ {16, 32, 48} → **25%, 50%, 75%** overlap, filling the empty band.

Replacements are drawn to match the `schema_tokens` of the core tool they
replace, so prompt size stays ~5,534 tokens and reuse differences cannot be a
side-effect of menu length.

`original` is a per-request deterministic shuffle — genuinely unordered, rather
than the padded workload's "differing tool first", which is a worst case rather
than a neutral one.

## Predictions

1. Reuse ceiling is ~C/64 for any policy that hoists the shared core.
2. `original` near 0%; `alphabetical` low, because sorting by name interleaves
   core and unique tools so the shared prefix dies early.
3. ToolTrie and ContextPilot both approach C/64.
4. **ContextPilot converges faster at every C**, for the same reason it does on
   padded menus — one intersection versus accumulated agreement.
5. **They tie once converged, at every C.** If this holds, the middle band does
   not discriminate either, and the honest conclusion is that this whole family
   of workloads — one shared core plus per-request noise — cannot separate the
   two mechanisms. A trie would need *structured* tails (groups sharing
   sub-cores) to have a regime, which is a different experiment.

Prediction 5 is the one worth being wrong about.
