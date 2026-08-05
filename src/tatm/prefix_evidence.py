from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def token_blocks(token_ids: Sequence[int], block_size: int) -> list[dict[str, Any]]:
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    blocks = []
    for block_index, start in enumerate(range(0, len(token_ids), block_size)):
        values = tuple(int(token) for token in token_ids[start : start + block_size])
        digest = hashlib.sha256(
            ",".join(str(token) for token in values).encode("ascii")
        ).hexdigest()
        blocks.append(
            {
                "block_index": block_index,
                "start_token": start,
                "end_token_exclusive": start + len(values),
                "token_count": len(values),
                "full_block": len(values) == block_size,
                "sha256": digest,
            }
        )
    return blocks


def prefix_pair_evidence(
    previous: Sequence[int],
    current: Sequence[int],
    block_size: int,
) -> dict[str, int | None]:
    shared = common_prefix_length(previous, current)
    cacheable = (shared // block_size) * block_size
    diverges = shared < min(len(previous), len(current))
    return {
        "common_prefix_tokens": shared,
        "cacheable_full_block_tokens": cacheable,
        "shared_full_blocks": cacheable // block_size,
        "divergence_token_index": shared if diverges else None,
    }


def best_prior_prefix(
    prior_prompts: Sequence[Sequence[int]],
    current: Sequence[int],
    block_size: int,
) -> dict[str, int | None]:
    if not prior_prompts:
        return {
            "best_prior_index": None,
            "common_prefix_tokens": 0,
            "cacheable_full_block_tokens": 0,
            "shared_full_blocks": 0,
            "divergence_token_index": None,
        }
    candidates = [
        prefix_pair_evidence(previous, current, block_size)
        for previous in prior_prompts
    ]
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            int(candidates[index]["cacheable_full_block_tokens"]),
            int(candidates[index]["common_prefix_tokens"]),
            -index,
        ),
    )
    return {"best_prior_index": best_index, **candidates[best_index]}


@dataclass
class _TokenTrieNode:
    children: dict[int, "_TokenTrieNode"] = field(default_factory=dict)
    first_prompt_index: int | None = None


class RenderedPrefixIndex:
    """Incremental exact-token trie for longest-prior-prefix queries."""

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self._root = _TokenTrieNode()
        self._prompt_lengths: list[int] = []

    def query(self, token_ids: Sequence[int]) -> dict[str, int | None]:
        node = self._root
        shared = 0
        best_index: int | None = None
        for token in token_ids:
            child = node.children.get(int(token))
            if child is None:
                break
            node = child
            shared += 1
            best_index = child.first_prompt_index
        cacheable = (shared // self.block_size) * self.block_size
        diverges = (
            best_index is not None
            and shared < min(self._prompt_lengths[best_index], len(token_ids))
        )
        return {
            "best_prior_index": best_index,
            "common_prefix_tokens": shared,
            "cacheable_full_block_tokens": cacheable,
            "shared_full_blocks": cacheable // self.block_size,
            "divergence_token_index": shared if diverges else None,
        }

    def observe(self, token_ids: Sequence[int]) -> int:
        prompt_index = len(self._prompt_lengths)
        self._prompt_lengths.append(len(token_ids))
        node = self._root
        for token in token_ids:
            token = int(token)
            child = node.children.get(token)
            if child is None:
                child = _TokenTrieNode(first_prompt_index=prompt_index)
                node.children[token] = child
            elif child.first_prompt_index is None:
                child.first_prompt_index = prompt_index
            node = child
        return prompt_index
