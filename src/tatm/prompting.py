from __future__ import annotations

import hashlib
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
