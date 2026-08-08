#!/usr/bin/env python3
"""Compare a cache-enabled probe run against a cache-disabled control.

Task B check 4 asks whether prefix caching changes generated output. That
question is only meaningful if the control genuinely ran with prefix caching
off, so this script refuses to report equality until it has verified the served
configuration of both runs.
"""

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
        {"type": call.get("type"), "function": call.get("function")}
        for call in value
        if isinstance(call, dict)
    ]


def scenario_rows(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Collect every trial's row per scenario, for v1 and v2 payloads."""
    rows: dict[str, list[dict[str, Any]]] = {}
    if run.get("format_version", 1) >= 2:
        trials = [trial["results"] for trial in run.get("trials", [])]
    else:
        trials = [run.get("results", [])]
    for trial in trials:
        for row in trial:
            rows.setdefault(row["scenario"], []).append(row)
    return rows


def cached_tokens(row: dict[str, Any]) -> float:
    return row.get("metric_delta", {}).get("vllm:prompt_tokens_cached", 0.0)


def verify_configurations(
    enabled: dict[str, Any], disabled: dict[str, Any]
) -> list[str]:
    problems: list[str] = []
    for label, run in (("Cache-enabled", enabled), ("Cache-disabled", disabled)):
        if run.get("prefix_cache_reset_between_trials") is not True:
            problems.append(
                f"{label} run does not prove a successful prefix-cache reset "
                "between trials; re-run the fail-closed probe."
            )
    enabled_flag = enabled.get("server_cache_config", {}).get(
        "enable_prefix_caching"
    )
    disabled_flag = disabled.get("server_cache_config", {}).get(
        "enable_prefix_caching"
    )

    if enabled_flag is None or disabled_flag is None:
        problems.append(
            "One or both runs predate configuration capture "
            "(no server_cache_config.enable_prefix_caching). Re-run the probe."
        )
    else:
        if enabled_flag is not True:
            problems.append(
                f"Cache-enabled run served enable_prefix_caching={enabled_flag}."
            )
        if disabled_flag is not False:
            problems.append(
                f"Control run served enable_prefix_caching={disabled_flag}. "
                "Restart it with --no-enable-prefix-caching; omitting "
                "--enable-prefix-caching does not disable the feature."
            )

    control_cached = [
        cached for rows in scenario_rows(disabled).values() for cached in map(cached_tokens, rows)
    ]
    if control_cached and max(control_cached) > 0:
        problems.append(
            f"Control run reports up to {max(control_cached):.0f} cached prompt "
            "tokens; a true cache-disabled run must report 0 everywhere."
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cache-enabled/disabled prefix probe outputs."
    )
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--disabled", type=Path, required=True)
    args = parser.parse_args()

    enabled = load(args.enabled)
    disabled = load(args.disabled)

    problems = verify_configurations(enabled, disabled)
    if problems:
        print("CONTROL INVALID - equality is not evaluated:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(2)

    enabled_rows = scenario_rows(enabled)
    disabled_rows = scenario_rows(disabled)
    common = sorted(enabled_rows.keys() & disabled_rows.keys())
    if not common:
        raise SystemExit("No matching scenarios found.")

    all_equal = True
    print(f"{'scenario':<22} {'equal':<7} {'cached(on)':>11} {'cached(off)':>12}")
    for scenario in common:
        left = enabled_rows[scenario]
        right = disabled_rows[scenario]
        # Within a run every trial must agree before cross-run comparison means
        # anything; a run that is internally nondeterministic is not evidence.
        left_outputs = {row.get("content") for row in left}
        right_outputs = {row.get("content") for row in right}
        stable = len(left_outputs) == 1 and len(right_outputs) == 1
        content_equal = left_outputs == right_outputs
        tool_calls_equal = normalized_tool_calls(
            left[0].get("tool_calls")
        ) == normalized_tool_calls(right[0].get("tool_calls"))
        equal = stable and content_equal and tool_calls_equal
        all_equal &= equal
        marker = "yes" if equal else ("UNSTABLE" if not stable else "NO")
        mean_on = sum(map(cached_tokens, left)) / len(left)
        mean_off = sum(map(cached_tokens, right)) / len(right)
        print(f"{scenario:<22} {marker:<7} {mean_on:>11.1f} {mean_off:>12.1f}")

    print()
    print(f"Control verified: enable_prefix_caching=False, 0 cached tokens.")
    print(f"All projected outputs equal: {all_equal}")
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
