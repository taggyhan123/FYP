from __future__ import annotations

from typing import Any


PRESSURE_ORDERINGS = (
    "original",
    "alphabetical",
    "random",
    "frequency",
    "schema_cost_weighted",
    "fp_tree_global",
)
PRESSURE_REGIMES = (
    "empirical",
    "uniform",
    "skewed",
    "session_bursty",
)
EVICTION_COUNT_METRIC = "vllm:kv_block_idle_before_evict_seconds_count"


def summarize_pressure_replays(
    labeled_replays: list[tuple[str, dict[str, Any]]],
    *,
    expected_capacity_tokens: int,
    required_peak_fraction: float,
    expected_orderings: tuple[str, ...] = PRESSURE_ORDERINGS,
    expected_regimes: tuple[str, ...] = PRESSURE_REGIMES,
) -> dict[str, Any]:
    """Validate and compact a predeclared controlled-cache pressure matrix.

    Structural drift is rejected immediately. Experimental acceptance checks
    are retained in the returned summary so a failed rerun can still be
    archived and inspected without being mistaken for valid pressure evidence.
    """
    if expected_capacity_tokens <= 0:
        raise ValueError("expected_capacity_tokens must be positive")
    if not 0 <= required_peak_fraction <= 1:
        raise ValueError("required_peak_fraction must be between 0 and 1")

    payload_by_ordering: dict[str, dict[str, Any]] = {}
    for label, payload in labeled_replays:
        if label in payload_by_ordering:
            raise ValueError(f"duplicate pressure replay label: {label}")
        payload_by_ordering[label] = payload

    if set(payload_by_ordering) != set(expected_orderings):
        missing = sorted(set(expected_orderings) - set(payload_by_ordering))
        extra = sorted(set(payload_by_ordering) - set(expected_orderings))
        raise ValueError(
            f"pressure ordering matrix mismatch; missing={missing}, extra={extra}"
        )

    orderings: dict[str, Any] = {}
    total_checks = 0
    passed_checks = 0
    for label in expected_orderings:
        payload = payload_by_ordering[label]
        _validate_replay_envelope(
            label,
            payload,
            expected_capacity_tokens=expected_capacity_tokens,
            expected_regimes=expected_regimes,
        )
        regime_summaries: dict[str, Any] = {}
        for regime in expected_regimes:
            result = payload["conditions"][regime]
            pressure = result.get("memory_pressure", {})
            sampling = result.get("kv_usage_sampling", {})
            peak = sampling.get("peak", {})
            metrics = result.get("session_metric_delta", {})
            counter_validation = result.get("counter_validation", {})

            observed_threshold = pressure.get("required_peak_fraction")
            threshold_matches = (
                observed_threshold is not None
                and abs(float(observed_threshold) - required_peak_fraction) < 1e-12
            )
            peak_fraction = pressure.get("peak_kv_usage_fraction")
            eviction_count = metrics.get(EVICTION_COUNT_METRIC)
            preemptions = metrics.get("vllm:num_preemptions")
            expected_same_multiset = regime in {"empirical", "session_bursty"}
            sampler_peak = peak.get("vllm:kv_cache_usage_perc")
            if sampler_peak is None:
                sampler_peak = peak.get("vllm:gpu_cache_usage_perc")
            checks = {
                "requests_complete": result.get("requests") == 200,
                "cache_reset_before": result.get("cache_reset_before") is True,
                "matched_multiset_label_correct": result.get(
                    "same_task_multiset_as_empirical"
                )
                is expected_same_multiset,
                "counter_validation_clean": bool(counter_validation)
                and all(value is True for value in counter_validation.values()),
                "capacity_matches": pressure.get("capacity_tokens")
                == expected_capacity_tokens,
                "declared_threshold_matches": threshold_matches,
                "peak_threshold_met": pressure.get("requirement_met") is True
                and peak_fraction is not None
                and float(peak_fraction) >= required_peak_fraction,
                "reported_peak_matches_sampler": peak_fraction is not None
                and sampler_peak is not None
                and abs(float(peak_fraction) - float(sampler_peak)) < 1e-8,
                "kv_samples_present": int(sampling.get("samples", 0)) > 0,
                "kv_scrape_errors_zero": int(sampling.get("scrape_errors", -1)) == 0,
                "eviction_metric_present": eviction_count is not None,
                "eviction_events_observed": eviction_count is not None
                and float(eviction_count) > 0,
                "preemption_metric_present": preemptions is not None,
                "no_preemptions": preemptions is not None
                and float(preemptions) == 0.0,
                "scheduler_metrics_present": (
                    "vllm:num_requests_running" in peak
                    and "vllm:num_requests_waiting" in peak
                ),
                "sequential_client_preserved": float(
                    peak.get("vllm:num_requests_running", 2.0)
                )
                <= 1.0
                and float(peak.get("vllm:num_requests_waiting", 1.0)) == 0.0,
            }
            total_checks += len(checks)
            passed_checks += sum(checks.values())
            regime_summaries[regime] = {
                "accepted": all(checks.values()),
                "checks": checks,
                "capacity_tokens": pressure.get("capacity_tokens"),
                "peak_kv_usage_fraction": peak_fraction,
                "estimated_peak_resident_tokens": pressure.get(
                    "estimated_peak_resident_tokens"
                ),
                "measured_reuse_ratio": result.get("measured_reuse_ratio"),
                "eviction_events_sampled": eviction_count,
                "preemptions": preemptions,
                "kv_samples": sampling.get("samples"),
                "peak_num_requests_running": peak.get(
                    "vllm:num_requests_running"
                ),
                "peak_num_requests_waiting": peak.get(
                    "vllm:num_requests_waiting"
                ),
            }
        orderings[label] = regime_summaries

    accepted_runs = sum(
        result["accepted"]
        for regimes in orderings.values()
        for result in regimes.values()
    )
    total_runs = len(expected_orderings) * len(expected_regimes)
    return {
        "format_version": 1,
        "experiment": "initial-brief-controlled-cache-pressure-rerun",
        "dataset": "BFCL",
        "menu_size": 64,
        "expected_capacity_tokens": expected_capacity_tokens,
        "required_peak_kv_usage_fraction": required_peak_fraction,
        "latency_comparison_allowed": False,
        "all_regime_runs": total_runs,
        "accepted_regime_runs": accepted_runs,
        "all_regime_runs_accepted": accepted_runs == total_runs,
        "checks_passed": passed_checks,
        "checks_total": total_checks,
        "orderings": orderings,
    }


def _validate_replay_envelope(
    label: str,
    payload: dict[str, Any],
    *,
    expected_capacity_tokens: int,
    expected_regimes: tuple[str, ...],
) -> None:
    expected = {
        "format_version": 2,
        "partition": "bfcl",
        "ordering": label,
        "menu_size": 64,
        "task_count": 200,
        "cache_capacity_tokens": expected_capacity_tokens,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(
                f"pressure replay {label!r} has {field}={payload.get(field)!r}; "
                f"expected {expected_value!r}"
            )

    support = payload.get("ordering_support", {})
    if support.get("mode") != "disjoint" or support.get(
        "evaluation_overlap_tasks"
    ) != 0:
        raise ValueError(f"pressure replay {label!r} has invalid support provenance")

    cache_config = payload.get("cache_config", {})
    try:
        block_size = int(cache_config.get("block_size"))
        gpu_blocks = int(cache_config.get("num_gpu_blocks"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"pressure replay {label!r} lacks a readable cache configuration"
        ) from error
    if block_size * gpu_blocks != expected_capacity_tokens:
        raise ValueError(
            f"pressure replay {label!r} cache configuration does not match "
            f"{expected_capacity_tokens} tokens"
        )
    if cache_config.get("enable_prefix_caching") is not True:
        raise ValueError(f"pressure replay {label!r} did not enable prefix caching")

    conditions = payload.get("conditions", {})
    if set(conditions) != set(expected_regimes):
        raise ValueError(
            f"pressure replay {label!r} has regimes {sorted(conditions)}; "
            f"expected {sorted(expected_regimes)}"
        )
