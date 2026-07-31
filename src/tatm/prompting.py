from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from tatm.models import CanonicalTool, TaskRecord


def order_tool_ids(
    tool_ids: Sequence[str],
    tools: dict[str, CanonicalTool],
    support: Counter[str],
    ordering: str,
    *,
    random_seed: int = 42,
) -> tuple[str, ...]:
    ids = tuple(tool_id for tool_id in dict.fromkeys(tool_ids) if tool_id in tools)
    if ordering == "original":
        return ids
    if ordering == "alphabetical":
        key = lambda item: (tools[item].name.casefold(), item)
    elif ordering in {"frequency", "fp_tree_global"}:
        key = lambda item: (-support[item], item)
    elif ordering == "schema_cost_weighted":
        key = lambda item: (
            -(support[item] * tools[item].schema_tokens),
            -support[item],
            item,
        )
    elif ordering == "random":
        key = lambda item: hashlib.sha256(
            f"{random_seed}:{item}".encode()
        ).hexdigest()
    else:
        raise ValueError(f"Unknown ordering: {ordering}")
    return tuple(sorted(ids, key=key))


def build_menu(
    gold_ids: Sequence[str],
    distractor_pool: Sequence[str],
    target_count: int,
    *,
    seed: int = 42,
) -> tuple[str, ...]:
    """Pad a task's gold tools with distractors up to `target_count` tools.

    Benchmark tasks expose a median of one tool (~124 schema tokens), which is
    both far below the size where schema prefill costs anything measurable and
    impossible to reorder. Real MCP clients load catalogs of tens to hundreds of
    tools, so menus are padded to reach that regime.

    The gold set is always retained in full; only padding is added, so tool
    selection stays answerable.

    Padding is drawn from a single global ranking fixed by `seed`, not resampled
    per task, so different tasks receive almost the same distractors. That is
    what a deployment looks like — one connected catalog served to every
    request — and it is also the only way padding can contribute shared prefix.
    Seeding per task instead makes every menu unique and drives cross-request
    reuse to nearly zero.

    Returns gold first, then padding; apply `order_tool_ids` afterwards to
    impose the ordering under test.
    """
    gold = tuple(dict.fromkeys(gold_ids))
    if target_count <= len(gold):
        return gold
    taken = set(gold)
    ranked = sorted(set(distractor_pool))
    random.Random(seed).shuffle(ranked)
    needed = target_count - len(gold)
    padding = tuple(
        tool_id for tool_id in ranked if tool_id not in taken
    )[:needed]
    return gold + padding


def openai_tool(tool: CanonicalTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def workload_record(
    task: TaskRecord,
    ordered_ids: Sequence[str],
    tools: dict[str, CanonicalTool],
    ordering: str,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "source": task.source,
        "domain": task.domain,
        "evidence_type": task.evidence_type,
        "ordering": ordering,
        "messages": [{"role": "user", "content": task.query}],
        "tool_ids": list(ordered_ids),
        "tools": [openai_tool(tools[item]) for item in ordered_ids],
        "canonical_tool_tokens": sum(
            tools[item].schema_tokens for item in ordered_ids
        ),
    }
