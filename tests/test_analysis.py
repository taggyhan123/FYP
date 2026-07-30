from collections import Counter

from tatm.analysis import (
    locality_metrics,
    ordering_functions,
    trie_metrics,
)
from tatm.models import CanonicalTool, TaskRecord
from tatm.vllm_client import metric_delta, parse_prometheus


def make_tool(tool_id: str, name: str, tokens: int) -> CanonicalTool:
    return CanonicalTool(
        tool_id=tool_id,
        source="test",
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        schema_tokens=tokens,
    )


def test_trie_counts_shared_prefix_and_blocks() -> None:
    tools = {
        "a": make_tool("a", "A", 16),
        "b": make_tool("b", "B", 8),
        "c": make_tool("c", "C", 8),
    }
    metrics = trie_metrics(
        [("a", "b"), ("a", "c"), ("a", "b")],
        tools,
        block_size=16,
    )
    assert metrics["naive_nodes"] == 6
    assert metrics["trie_nodes"] == 3
    assert metrics["reusable_tool_tokens"] == 40
    assert metrics["cacheable_block_tokens"] == 32
    assert metrics["node_compression_ratio"] == 0.5


def test_frequency_and_fp_tree_global_are_equivalent() -> None:
    tools = {
        "a": make_tool("a", "Zulu", 10),
        "b": make_tool("b", "Alpha", 20),
    }
    orderers = ordering_functions(tools, Counter({"a": 4, "b": 2}))
    assert orderers["frequency"](("b", "a")) == ("a", "b")
    assert orderers["fp_tree_global"](("b", "a")) == ("a", "b")
    assert orderers["alphabetical"](("a", "b")) == ("b", "a")


def test_locality_metrics_distinguish_domain_and_tools() -> None:
    tools = {
        "a": make_tool("a", "A", 10),
        "b": make_tool("b", "B", 10),
    }
    tasks = [
        TaskRecord("1", "test", "", ("a",), "gold", domain="x"),
        TaskRecord("2", "test", "", ("a", "b"), "gold", domain="x"),
        TaskRecord("3", "test", "", ("b",), "gold", domain="y"),
    ]
    result = locality_metrics(tasks, tools)
    assert result["same_domain_ratio"] == 0.5
    assert result["shared_tool_ratio"] == 1.0
    assert result["mean_tool_jaccard"] == 0.5


def test_vllm_prometheus_counters_are_summed_and_gauges_take_max() -> None:
    parsed = parse_prometheus(
        """
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="0"} 32
vllm:prefix_cache_hits_total{engine="1"} 16
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.20
unknown_metric 100
"""
    )
    assert parsed["vllm:prefix_cache_hits"] == 48
    change = metric_delta(
        {"vllm:prefix_cache_hits": 40, "vllm:kv_cache_usage_perc": 0.1},
        parsed,
    )
    assert change["vllm:prefix_cache_hits"] == 8
    assert change["vllm:kv_cache_usage_perc"] == 0.25
