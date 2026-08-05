import pytest

from tatm.measurement import (
    project_request_measurement,
    reuse_bucket,
    selected_tool_text_condition,
    summarize_request_measurements,
)


def test_project_request_measurement_uses_server_counters() -> None:
    row = project_request_measurement(
        {
            "vllm:prompt_tokens_cached": 48,
            "vllm:request_prefill_kv_computed_tokens_sum": 16,
            "vllm:request_prefill_time_seconds_sum": 0.2,
            "vllm:time_to_first_token_seconds_sum": 0.3,
            "vllm:inter_token_latency_seconds_sum": 0.12,
            "vllm:inter_token_latency_seconds_count": 3,
        },
        {"prompt_tokens": 64},
    )
    assert row["cached_ratio"] == 0.75
    assert row["computed_prefill_tokens"] == 16
    assert row["mean_inter_token_latency_seconds"] == 0.04


def test_reuse_buckets_separate_cold_partial_and_full() -> None:
    assert reuse_bucket(0, 100) == "cold"
    assert reuse_bucket(10, 100) == "partial_0_25"
    assert reuse_bucket(25, 100) == "partial_25_50"
    assert reuse_bucket(50, 100) == "partial_50_75"
    assert reuse_bucket(75, 100) == "partial_75_100"
    assert reuse_bucket(100, 100) == "full"


def test_summary_reports_direct_partial_reuse_latency() -> None:
    rows = [
        {
            "prompt_tokens": 100,
            "cached_tokens": 50,
            "cached_ratio": 0.5,
            "ttft_seconds": 0.3,
            "prefill_seconds": 0.2,
            "wall_seconds": 0.4,
        },
        {
            "prompt_tokens": 100,
            "cached_tokens": 75,
            "cached_ratio": 0.75,
            "ttft_seconds": 0.2,
            "prefill_seconds": 0.1,
            "wall_seconds": 0.3,
        },
    ]
    summary = summarize_request_measurements(rows)
    assert summary["partial_all"]["requests"] == 2
    assert summary["partial_all"]["mean_cached_ratio"] == 0.625
    assert summary["partial_all"]["ttft_seconds"]["mean"] == 0.25


def test_empty_summary_does_not_call_statistics_on_empty_input() -> None:
    summary = summarize_request_measurements([])
    assert summary["partial_all"]["requests"] == 0
    assert summary["partial_all"]["ttft_seconds"] == {"n": 0}


def test_text_prefill_fallback_is_explicit_and_preserves_original_order() -> None:
    condition = selected_tool_text_condition(
        "ordinary_text_prefill_fallback", ["original", "original"]
    )
    assert condition == {
        "role": "ordinary_text_prefill_fallback",
        "materialization": "ordinary_selected_tool_text_prefill",
        "workload_orderings": ["original"],
        "selected_tool_membership": "input_workload",
        "preserves_input_order": True,
        "retains_inactive_tools": False,
        "modifies_kv_tensors": False,
    }


def test_text_prefill_fallback_rejects_a_reordered_workload() -> None:
    with pytest.raises(ValueError, match="requires an original-order workload"):
        selected_tool_text_condition(
            "ordinary_text_prefill_fallback", ["alphabetical"]
        )
