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
    cached, _ = simulate_block_cache(
        [block_hashes(ids, block_size) for ids in prompts],
        [len(ids) for ids in prompts],
        completion_tokens,
        capacity_blocks,
        block_size,
    )
    return cached


def simulate_block_cache(
    prompt_block_hashes: Sequence[Sequence[int]],
    prompt_lengths: Sequence[int],
    completion_tokens: Sequence[int],
    capacity_blocks: int | None = None,
    block_size: int = 16,
    owners: Sequence[object] | None = None,
) -> tuple[list[int], list[int]]:
    """`simulate_prefix_cache` on precomputed block hashes, with provenance.

    Hashing a prompt once and replaying it under many cache sizes or request
    orders is much cheaper than re-hashing every time. With ``owners`` (one
    label per request, e.g. a session id), the second list reports how many of
    each request's cached tokens came from blocks first written by a request
    with a different owner; without it, that list is all zeros.
    """
    count = len(prompt_block_hashes)
    if len(prompt_lengths) != count or len(completion_tokens) != count:
        raise ValueError("inputs differ in length")
    if owners is not None and len(owners) != count:
        raise ValueError("owners differ in length")
    cache: OrderedDict[object, None] = OrderedDict()  # front = evicted first
    writer: dict[object, object] = {}
    cached: list[int] = []
    foreign: list[int] = []
    for index in range(count):
        hashes = prompt_block_hashes[index]
        length = prompt_lengths[index]
        owner = owners[index] if owners is not None else None
        limit = min(len(hashes), max(length - 1, 0) // block_size)
        hit = 0
        while hit < limit and hashes[hit] in cache:
            hit += 1
        cached.append(hit * block_size)
        foreign.append(
            sum(block_size for value in hashes[:hit] if writer.get(value) != owner)
            if owners is not None else 0
        )
        if capacity_blocks is None:
            for value in hashes:
                if value not in cache:
                    cache[value] = None
                    writer[value] = owner
            continue
        for value in hashes[:hit]:
            cache.pop(value, None)  # in use: not evictable while running
        total_blocks = -(-(length + completion_tokens[index]) // block_size)
        while cache and len(cache) + total_blocks > capacity_blocks:
            writer.pop(cache.popitem(last=False)[0], None)
        for extra in range(total_blocks - len(hashes)):
            cache[("output", index, extra)] = None
        for position, value in enumerate(reversed(hashes)):
            if position < len(hashes) - hit:
                writer[value] = owner  # newly computed by this request
            cache.pop(value, None)
            cache[value] = None
        while len(cache) > capacity_blocks:
            writer.pop(cache.popitem(last=False)[0], None)
    return cached, foreign
