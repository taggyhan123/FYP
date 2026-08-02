from collections import Counter

import pytest

from tatm.models import CanonicalTool
from tatm.tooltrie import ToolTrie


def make_tool(tool_id: str, name: str, tokens: int) -> CanonicalTool:
    return CanonicalTool(
        tool_id=tool_id,
        source="test",
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        schema_tokens=tokens,
    )


TOOLS = {
    "a": make_tool("a", "Zulu", 8),
    "b": make_tool("b", "Alpha", 16),
    "c": make_tool("c", "Mike", 32),
    "d": make_tool("d", "Bravo", 8),
    "x": make_tool("x", "Xray", 8),
}


def test_cold_plan_uses_alphabetical_fallback_and_preserves_set() -> None:
    planner = ToolTrie(TOOLS)
    plan = planner.plan(("a", "c", "b"))
    assert plan.matched_prefix_ids == ()
    assert plan.ordered_ids == ("b", "c", "a")
    assert set(plan.ordered_ids) == {"a", "b", "c"}


def test_plan_uses_only_paths_observed_before_current_request() -> None:
    planner = ToolTrie(TOOLS)
    cold = planner.plan(("a", "b"))
    assert cold.matched_prefix_ids == ()

    planner.observe(cold.ordered_ids)
    warm = planner.plan(("a", "b"))
    assert warm.matched_prefix_ids == cold.ordered_ids
    assert warm.hinted_schema_tokens == 24


def test_greedy_tie_break_prefers_greater_reachable_schema_cost() -> None:
    planner = ToolTrie(TOOLS)
    planner.observe(("a", "b"))
    planner.observe(("a", "c"))

    plan = planner.plan(("a", "b", "c", "d"))
    # Both branches start under a. The a->c branch retains more schema tokens
    # than a->b, so c is selected before the alphabetical fallback b,d.
    assert plan.matched_prefix_ids == ("a", "c")
    assert plan.ordered_ids == ("a", "c", "b", "d")


def test_frequency_fallback_requires_separate_fitted_support() -> None:
    with pytest.raises(ValueError, match="separate training workload"):
        ToolTrie(TOOLS, fallback="frequency")

    planner = ToolTrie(
        TOOLS,
        fallback="frequency",
        support=Counter({"a": 2, "b": 10, "c": 4}),
    )
    assert planner.plan(("a", "b", "c")).ordered_ids == ("b", "c", "a")


def test_recency_window_stops_stale_paths_from_being_hinted() -> None:
    planner = ToolTrie(TOOLS, recency_window=1)
    planner.observe(("a", "b"))
    assert planner.plan(("a", "b")).matched_prefix_ids == ("a", "b")

    planner.observe(("x",))
    stale = planner.plan(("a", "b"))
    assert stale.matched_prefix_ids == ()


def test_leaf_first_lru_eviction_keeps_metadata_within_budget() -> None:
    planner = ToolTrie(TOOLS, capacity_tokens=None, max_nodes=2)
    planner.observe(("a", "b"))
    assert planner.snapshot()["nodes"] == 2

    planner.observe(("x", "d"))  # forces the old a->b path out leaf-first
    state = planner.snapshot()
    assert state["nodes"] <= 2
    assert state["evictions"] == 2
    assert planner.plan(("a", "b")).matched_prefix_ids == ()
    assert planner.plan(("x", "d")).matched_prefix_ids == ("x", "d")


def test_schema_token_capacity_is_enforced() -> None:
    planner = ToolTrie(TOOLS, capacity_tokens=24, max_nodes=None)
    planner.observe(("a", "b"))
    planner.observe(("x", "d"))
    state = planner.snapshot()
    assert state["retained_schema_tokens"] <= 24
    assert state["evictions"] == 1


def test_duplicate_or_unknown_ids_are_rejected() -> None:
    planner = ToolTrie(TOOLS)
    with pytest.raises(ValueError, match="must not repeat"):
        planner.observe(("a", "a"))
    with pytest.raises(ValueError, match="unknown tool IDs"):
        planner.plan(("missing",))
