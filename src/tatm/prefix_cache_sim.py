"""Token-exact simulation of vLLM's automatic prefix cache.

Reproduces how many prompt tokens vLLM 0.26 serves from its prefix cache for a
sequence of requests, given their exact prompt token ids:

* the cache works in full blocks (16 tokens by default); a block is reusable
  only if every token before it also matches, so block identity is a hash
  chained over all preceding blocks;
* at least one prompt token is always computed, so an exact repeat of a cached
  prompt hits at most ``(len - 1) // block_size`` blocks;
* with finite capacity, freed blocks are queued for eviction in reverse order
  (a request's last blocks are evicted before its first ones), and new blocks
  evict from the front of that queue.

Validated against measured per-request cached tokens on 36 one-at-a-time
replays (six policies x six menu designs, Qwen3-0.6B): 199-200 of 200 requests
match exactly in every run. `tatm.analysis.bounded_trie_metrics` estimates the
same quantity at whole-tool granularity; this module is the token-exact one.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence


def block_hashes(token_ids: Sequence[int], block_size: int = 16) -> list[int]:
    """Chained hash per full block: block k's hash covers blocks 0..k."""
    hashes: list[int] = []
    previous: int | None = None
    for index in range(len(token_ids) // block_size):
        block = tuple(token_ids[index * block_size:(index + 1) * block_size])
        previous = hash((previous, block))
        hashes.append(previous)
    return hashes


def simulate_prefix_cache(
    prompts: Sequence[Sequence[int]],
    completion_tokens: Sequence[int],
    capacity_blocks: int | None = None,
    block_size: int = 16,
) -> list[int]:
    """Cached prompt tokens for each request, served one at a time in order.

    ``capacity_blocks=None`` models an unlimited cache. Output blocks occupy
    capacity but are never matched by another prompt.
    """
    if len(prompts) != len(completion_tokens):
        raise ValueError("prompts and completion_tokens differ in length")
    cache: OrderedDict[object, None] = OrderedDict()  # front = evicted first
    cached: list[int] = []
    for request_index, (ids, completion) in enumerate(zip(prompts, completion_tokens)):
        hashes = block_hashes(ids, block_size)
        limit = min(len(hashes), max(len(ids) - 1, 0) // block_size)
        hit = 0
        while hit < limit and hashes[hit] in cache:
            hit += 1
        cached.append(hit * block_size)
        if capacity_blocks is None:
            for value in hashes:
                cache[value] = None
            continue
        for value in hashes[:hit]:
            cache.pop(value, None)  # in use: not evictable while running
        total_blocks = -(-(len(ids) + completion) // block_size)
        while cache and len(cache) + total_blocks > capacity_blocks:
            cache.popitem(last=False)
        for extra in range(total_blocks - len(hashes)):
            cache[("output", request_index, extra)] = None
        for value in reversed(hashes):
            cache.pop(value, None)
            cache[value] = None
        while len(cache) > capacity_blocks:
            cache.popitem(last=False)
    return cached
