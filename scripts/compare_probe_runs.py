#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "type": call.get("type"),
            "function": call.get("function"),
        }
        for call in value
        if isinstance(call, dict)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cache-enabled/disabled prefix probe outputs."
    )
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--disabled", type=Path, required=True)
    args = parser.parse_args()

    enabled = load(args.enabled)
    disabled = load(args.disabled)
    enabled_by_name = {
        row["scenario"]: row for row in enabled.get("results", [])
    }
    disabled_by_name = {
        row["scenario"]: row for row in disabled.get("results", [])
    }
    common = sorted(enabled_by_name.keys() & disabled_by_name.keys())
    all_equal = True
    for scenario in common:
        left = enabled_by_name[scenario]
        right = disabled_by_name[scenario]
        content_equal = left.get("content") == right.get("content")
        tool_calls_equal = normalized_tool_calls(
            left.get("tool_calls")
        ) == normalized_tool_calls(right.get("tool_calls"))
        all_equal &= content_equal and tool_calls_equal
        print(
            f"{scenario}: content_equal={content_equal} "
            f"tool_calls_equal={tool_calls_equal} "
            f"enabled_wall={left.get('wall_seconds')} "
            f"disabled_wall={right.get('wall_seconds')}"
        )
    if not common:
        raise SystemExit("No matching scenarios found.")
    print(f"All projected outputs equal: {all_equal}")
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
