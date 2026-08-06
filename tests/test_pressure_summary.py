import copy

import pytest

from tatm.pressure_summary import (
    PRESSURE_ORDERINGS,
    PRESSURE_REGIMES,
    summarize_pressure_replays,
)


def pressure_replay(ordering: str) -> dict:
    conditions = {}
    for regime in PRESSURE_REGIMES:
        conditions[regime] = {
            "requests": 200,
            "cache_reset_before": True,
            "same_task_multiset_as_empirical": regime
            in {"empirical", "session_bursty"},
            "measured_reuse_ratio": 0.2,
            "counter_validation": {
                "query_counter_matches_response_prompt_tokens": True,
                "cached_plus_computed_matches_queries": True,
            },
            "session_metric_delta": {
                "vllm:kv_block_idle_before_evict_seconds_count": 7,
                "vllm:num_preemptions": 0,
            },
            "kv_usage_sampling": {
                "samples": 20,
                "scrape_errors": 0,
                "peak": {
                    "vllm:kv_cache_usage_perc": 0.92,
                    "vllm:num_requests_running": 1,
                    "vllm:num_requests_waiting": 0,
                },
            },
            "memory_pressure": {
                "capacity_tokens": 7680,
                "peak_kv_usage_fraction": 0.92,
                "estimated_peak_resident_tokens": 7066,
                "required_peak_fraction": 0.9,
                "requirement_met": True,
            },
        }
    return {
        "format_version": 2,
        "partition": "bfcl",
        "ordering": ordering,
        "menu_size": 64,
        "task_count": 200,
        "cache_capacity_tokens": 7680,
        "cache_config": {
            "enable_prefix_caching": True,
            "block_size": "16",
            "num_gpu_blocks": "480",
        },
        "ordering_support": {
            "mode": "disjoint",
            "evaluation_overlap_tasks": 0,
        },
        "conditions": conditions,
    }


def matrix() -> list[tuple[str, dict]]:
    return [(ordering, pressure_replay(ordering)) for ordering in PRESSURE_ORDERINGS]


def test_pressure_matrix_is_accepted() -> None:
    summary = summarize_pressure_replays(
        matrix(), expected_capacity_tokens=7680, required_peak_fraction=0.9
    )
    assert summary["accepted_regime_runs"] == 24
    assert summary["all_regime_runs_accepted"] is True
    assert summary["checks_passed"] == summary["checks_total"]
    assert summary["latency_comparison_allowed"] is False


def test_failed_eviction_check_is_preserved_in_summary() -> None:
    payloads = matrix()
    modified = copy.deepcopy(payloads[0][1])
    modified["conditions"]["empirical"]["session_metric_delta"][
        "vllm:kv_block_idle_before_evict_seconds_count"
    ] = 0
    payloads[0] = (payloads[0][0], modified)

    summary = summarize_pressure_replays(
        payloads, expected_capacity_tokens=7680, required_peak_fraction=0.9
    )
    result = summary["orderings"]["original"]["empirical"]
    assert result["accepted"] is False
    assert result["checks"]["eviction_metric_present"] is True
    assert result["checks"]["eviction_events_observed"] is False
    assert summary["accepted_regime_runs"] == 23


def test_pressure_matrix_rejects_cache_capacity_drift() -> None:
    payloads = matrix()
    payloads[0][1]["cache_capacity_tokens"] = 190896
    with pytest.raises(ValueError, match="cache_capacity_tokens"):
        summarize_pressure_replays(
            payloads, expected_capacity_tokens=7680, required_peak_fraction=0.9
        )


def test_pressure_matrix_rejects_missing_ordering() -> None:
    with pytest.raises(ValueError, match="matrix mismatch"):
        summarize_pressure_replays(
            matrix()[:-1], expected_capacity_tokens=7680, required_peak_fraction=0.9
        )
