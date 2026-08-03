"""Paired uncertainty estimates for BFCL ordering comparisons.

The same BFCL task and tool menu are evaluated under two orderings, so an
independent two-proportion interval throws away useful pairing information.
When several fixed menu seeds reuse the same task, the task is the sampling
unit: the bootstrap resamples task clusters and keeps every menu realization
for a selected task together.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _exact_mcnemar_p(discordant_left: int, discordant_right: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""

    total = discordant_left + discordant_right
    if total == 0:
        return 1.0
    tail = min(discordant_left, discordant_right)
    numerator = 2 * sum(math.comb(total, value) for value in range(tail + 1))
    denominator = 2**total
    return min(1.0, numerator / denominator)


def compare_paired_binary(
    baseline: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 42,
    equivalence_margin_pp: float | None = None,
) -> dict[str, Any]:
    """Compare paired binary outcomes with a task-clustered bootstrap.

    Rows are paired by ``case_id``. Multiple case IDs may share ``task_id``
    when the same query is tested with several fixed menu seeds; those rows are
    kept in one bootstrap cluster.
    """

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if equivalence_margin_pp is not None and equivalence_margin_pp <= 0:
        raise ValueError("equivalence_margin_pp must be > 0")

    def index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id") or row.get("task_id"))
            if case_id in indexed:
                raise ValueError(f"Duplicate paired case_id: {case_id}")
            if metric not in row or row[metric] is None:
                raise ValueError(f"{case_id}: missing binary metric {metric}")
            if not isinstance(row[metric], bool):
                raise ValueError(f"{case_id}: {metric} must be boolean")
            indexed[case_id] = row
        return indexed

    baseline_by_case = index(baseline)
    candidate_by_case = index(candidate)
    if set(baseline_by_case) != set(candidate_by_case):
        missing_candidate = sorted(set(baseline_by_case) - set(candidate_by_case))
        missing_baseline = sorted(set(candidate_by_case) - set(baseline_by_case))
        raise ValueError(
            "Paired case IDs differ: "
            f"missing candidate={missing_candidate[:3]}, "
            f"missing baseline={missing_baseline[:3]}"
        )
    if not baseline_by_case:
        raise ValueError("No paired cases")

    clusters: dict[str, list[int]] = defaultdict(list)
    baseline_correct = 0
    candidate_correct = 0
    baseline_only = 0
    candidate_only = 0
    for case_id in sorted(baseline_by_case):
        left = baseline_by_case[case_id]
        right = candidate_by_case[case_id]
        left_task = str(left.get("task_id") or case_id)
        right_task = str(right.get("task_id") or case_id)
        if left_task != right_task:
            raise ValueError(f"{case_id}: task IDs differ across paired rows")
        left_ok = bool(left[metric])
        right_ok = bool(right[metric])
        baseline_correct += int(left_ok)
        candidate_correct += int(right_ok)
        baseline_only += int(left_ok and not right_ok)
        candidate_only += int(right_ok and not left_ok)
        clusters[left_task].append(int(right_ok) - int(left_ok))

    pair_count = len(baseline_by_case)
    point_difference = (candidate_correct - baseline_correct) / pair_count
    cluster_ids = sorted(clusters)
    rng = random.Random(bootstrap_seed)
    bootstrap_differences: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in clusters[cluster_id]]
        bootstrap_differences.append(sum(values) / len(values))

    lower = _quantile(bootstrap_differences, 0.025)
    upper = _quantile(bootstrap_differences, 0.975)
    interval_pp = [round(lower * 100, 4), round(upper * 100, 4)]
    output: dict[str, Any] = {
        "metric": metric,
        "paired_cases": pair_count,
        "task_clusters": len(cluster_ids),
        "baseline_accuracy": round(baseline_correct / pair_count, 6),
        "candidate_accuracy": round(candidate_correct / pair_count, 6),
        "difference_percentage_points": round(point_difference * 100, 4),
        "cluster_bootstrap_95_ci_percentage_points": interval_pp,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "discordant_pairs": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
        },
        "mcnemar_exact_two_sided_p": round(
            _exact_mcnemar_p(baseline_only, candidate_only), 8
        ),
        "mcnemar_independence_assumption_met": pair_count == len(cluster_ids),
        "mcnemar_note": (
            "Exact McNemar is secondary/descriptive when several menu cases "
            "share a task; the task-clustered bootstrap interval is primary."
        ),
    }
    if equivalence_margin_pp is not None:
        output["equivalence_margin_percentage_points"] = equivalence_margin_pp
        output["equivalent_within_margin"] = (
            interval_pp[0] > -equivalence_margin_pp
            and interval_pp[1] < equivalence_margin_pp
        )
    return output
