"""Small-sample statistics for latency measurements.

Latency claims in this project must come with intervals. A single trial of the
prefix-cache probe once appeared to show a 3x TTFT gain that was entirely the
server's first-ever request, so point estimates are not reported anywhere.
"""

from __future__ import annotations

import statistics
from typing import Any

# Two-sided 95% Student-t critical values by degrees of freedom.
T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def describe(values: list[float]) -> dict[str, Any]:
    """Mean, spread, and a 95% Student-t half-width for a small sample."""
    n = len(values)
    mean = statistics.fmean(values)
    row: dict[str, Any] = {
        "n": n,
        "mean": round(mean, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }
    if n > 1:
        stdev = statistics.stdev(values)
        critical = T_95.get(n - 1, 1.96)
        row["stdev"] = round(stdev, 6)
        row["ci95_half_width"] = round(critical * stdev / (n**0.5), 6)
    return row


def separated(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when two `describe` results have non-overlapping 95% intervals."""
    left_half = left.get("ci95_half_width")
    right_half = right.get("ci95_half_width")
    if left_half is None or right_half is None:
        return False
    return abs(left["mean"] - right["mean"]) > (left_half + right_half)
