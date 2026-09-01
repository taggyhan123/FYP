# How ToolTrie-v1 works

`src/tatm/tooltrie_v1.py`. It subclasses `ToolTrie` and overrides **one method**.
Everything else — the trie, the walk, the selection key, the budget, eviction,
causality — is [v0](tooltrie-v0-design.md) unchanged.

## What both versions do

Every request arrives as a menu of tools in some order. The planner emits a
permutation of that menu; membership never changes. vLLM's prefix cache reuses
KV state only for an **exact** shared leading token prefix, so the goal is to
make request *N* begin with the same tools, in the same order, as something
already served.

`plan()` walks a trie whose root-to-node paths are sequences actually served
before, choosing at each step the child that leads to the most already-computed
tokens:

```python
while remaining:
    candidates = [child for tool_id, child in node.children.items()
                  if tool_id in remaining and self._is_resident_hint(child)]
    if not candidates: break
    chosen = min(candidates, key=self._selection_key)   # reachable cached cost
    matched.append(chosen.tool_id); remaining.remove(chosen.tool_id); node = chosen

fallback_ids = self._fallback_order(remaining)
ordered_ids  = (*matched, *fallback_ids)
```

The output is always **matched prefix + everything else**. `observe()` inserts
the emitted ordering afterwards, never before, so no request influences its own
plan.

## The one difference

What order do the tools the trie *could not* match go in?

```python
# v0 — sort by tool name
key = lambda item: (self.tools[item].name.casefold(), item)

# v1 — keep the order they arrived in
key = lambda t: incoming.get(t, len(incoming))
```

v1 captures the incoming order in `plan()` (the parent hands `_fallback_order` a
*set*, which has no order) and releases it afterwards:

```python
def plan(self, tool_ids):
    self._incoming = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    try:    return super().plan(tool_ids)
    finally: self._incoming = {}
```

## Why one line matters this much

**The fallback governs almost the whole menu.** The trie matches 1.19 of 64 tools
at k64 under v0 — so ~93% of every ordering is decided by the fallback, not the
trie.

**Tool name correlates with nothing.** Not relevance, not commonality. So v0
permuted ~99% of each menu on no information: it displaced the correct tool from
mean position 11.5 to 62.8 at k128 and returned 1.13% reuse for it — **37
positions of displacement per point of reuse, against ContextPilot's 8.0**.

v1 moves tools **1.1 positions on average instead of 41.9**, and leaves **120 of
200 requests byte-identical** to their input.

### Worked example

Menu `[A, B, C, D, E]` in retriever order; the trie has already served `[C, A, …]`.

```
walk       C matched (child of root, in menu, resident)
           A matched (child of C)
           no further child qualifies -> stop
unmatched  B, D, E
v0 emits   C A  +  B D E sorted by tool NAME     -> relevance below the prefix destroyed
v1 emits   C A  +  B D E in arrival order        -> relevance below the prefix preserved
```

Both get the same 2-tool cache hit. Only v0 also scrambles the three underneath.

## How much is the trie, how much the fallback?

Removing the trie from v1 leaves *exactly* the `original` ordering, so the split
is exact:

| depth | fallback alone | v1 | trie adds | trie's share | tools the trie places |
|---|---|---|---|---|---|
| k4 | 15.87% | 19.47% | +3.60pp | 18% | 0.57 / 4 |
| k16 | 6.12% | 11.31% | +5.19pp | 46% | 1.60 / 16 |
| k64 | 0.91% | 4.96% | +4.05pp | **82%** | 3.40 / 64 |
| k128 | 0.37% | 2.21% | +1.84pp | **83%** | 2.55 / 128 |

At the deep menus the trie places **2–5% of the tools and supplies ~82% of the
reuse** — because the tools it places sit at the *front*, and a leading prefix is
what the cache reuses. The fallback's job is not to generate reuse; it is to
avoid destroying the ordering the retriever already produced.

## The boundary condition

v1 never overrides its input, so an adversarial input ordering defeats it. On the
`padded-64` workload — whose `original` arm puts the one differing tool at
position 0 of *every* request — v1 preserves that and scores **1.19% against
v0's 87.19%**.

That is the construction, not high overlap: on menus sharing 25/50/75% of their
tools with a *shuffled* input order, v1 reaches 80–95% of ContextPilot's reuse
and ~3x v0's. Use v1 where the input ordering carries relevance information.

## Where to look next

| | |
| --- | --- |
| Implementation | `src/tatm/tooltrie_v1.py` |
| Base class | `src/tatm/tooltrie.py`, [design note](tooltrie-v0-design.md) |
| Workload builder | `scripts/build_tooltrie_v1_workload.py` |
| Tests | `tests/test_tooltrie_v1.py` |
| Measurements | `reports/concurrent-latency/findings.md` §5 |
