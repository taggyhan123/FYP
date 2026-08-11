"""ToolTrie-v1 reads visit_count; v0 must be unaffected."""

import random

from tatm.models import CanonicalTool
from tatm.tooltrie import ToolTrie
from tatm.tooltrie_v1 import WeightedToolTrie


def tools(n: int = 12, tokens: int = 40) -> dict[str, CanonicalTool]:
    return {
        f"t{i}": CanonicalTool(
            tool_id=f"t{i}",
            name=f"tool_{i:02d}",
            description="d",
            parameters={},
            source="test",
            schema_tokens=tokens,
        )
        for i in range(n)
    }


def test_v1_is_a_permutation_and_causal():
    planner = WeightedToolTrie(tools(), recency_window=8)
    rng = random.Random(7)
    for _ in range(30):
        menu = rng.sample(sorted(tools()), 6)
        before = planner.snapshot()
        ordered = planner.plan(menu).ordered_ids
        assert planner.snapshot() == before, "plan must not mutate the trie"
        assert sorted(ordered) == sorted(menu)
        planner.observe(ordered)


def test_v1_selection_reads_visit_count_and_v0_does_not():
    t = tools()
    v0, v1 = ToolTrie(t, recency_window=None), WeightedToolTrie(t, recency_window=None)
    for planner in (v0, v1):
        for _ in range(5):
            planner.observe(["t0", "t1"])
        planner.observe(["t0", "t2"])
    node_v1 = v1.root.children["t0"]
    assert node_v1.children["t1"].visit_count == 5
    assert node_v1.children["t2"].visit_count == 1
    # v0's key ignores visit_count entirely; v1's does not.
    remaining = frozenset({"t1", "t2"})
    assert v1._selection_key(node_v1.children["t1"], remaining)[1] == -5
    node_v0 = v0.root.children["t0"]
    assert v0._selection_key(node_v0.children["t1"], remaining)[1] == 0


def test_v1_evicts_the_least_visited_leaf_not_the_least_recent():
    t = tools(n=4, tokens=100)
    planner = WeightedToolTrie(t, recency_window=None, capacity_tokens=250)
    for _ in range(6):
        planner.observe(["t0", "t1"])
    planner.observe(["t0", "t2"])
    planner.observe(["t0", "t3"])
    live = {n.tool_id for n in planner._live_nodes}
    assert "t1" in live, "the most-visited leaf must survive the budget"
    assert planner.evictions > 0, "the budget must actually bind in this test"


def test_v0_selection_key_matches_the_expression_it_replaced():
    """_selection_key was extracted from an inline lambda; it must be identical.

    Verified separately against the pre-refactor module from git: 1,500
    orderings over 25 seeds, zero mismatches, identical snapshots.
    """
    t = tools()
    planner = ToolTrie(t, recency_window=16)
    for _ in range(6):
        planner.observe(["t0", "t3", "t7"])
    node = planner.root.children["t0"]
    remaining = frozenset({"t3", "t7"})
    for child in node.children.values():
        expected = (
            -planner._reachable_cached_cost(child, remaining),
            -planner.support.get(child.tool_id or "", 0),
            child.tool_id or "",
        )
        assert planner._selection_key(child, remaining) == expected
