"""Summarize a small tool-usage trace for the TATM research project."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


DATA_PATH = Path(__file__).with_name("sample_tool_usage.csv")


def summarize_tool_usage(csv_path: Path) -> list[dict[str, object]]:
    """Return per-tool usage and cache statistics from a CSV trace."""
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "schema_tokens": 0,
            "cache_hits": 0,
            "prefill_ms": 0.0,
        }
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            tool = row["tool_name"]
            totals[tool]["calls"] += 1
            totals[tool]["schema_tokens"] += int(row["schema_tokens"])
            totals[tool]["cache_hits"] += row["cache_hit"].lower() == "true"
            totals[tool]["prefill_ms"] += float(row["prefill_ms"])

    summary = []
    for tool, values in totals.items():
        calls = int(values["calls"])
        summary.append(
            {
                "tool": tool,
                "calls": calls,
                "schema_tokens": int(values["schema_tokens"]),
                "cache_hit_rate": float(values["cache_hits"]) / calls,
                "average_prefill_ms": float(values["prefill_ms"]) / calls,
            }
        )

    return sorted(summary, key=lambda item: (-int(item["calls"]), str(item["tool"])))


def main() -> None:
    print(f"{'tool':<18} {'calls':>5} {'tokens':>8} {'hit rate':>10} {'avg prefill':>12}")
    for row in summarize_tool_usage(DATA_PATH):
        print(
            f"{row['tool']:<18} "
            f"{row['calls']:>5} "
            f"{row['schema_tokens']:>8} "
            f"{row['cache_hit_rate']:>9.0%} "
            f"{row['average_prefill_ms']:>10.1f} ms"
        )


if __name__ == "__main__":
    main()
