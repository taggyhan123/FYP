"""ToolTrie-v1 leaves unmatched tools in the order they arrived; v0 sorts them."""

from tatm.models import CanonicalTool
from tatm.tooltrie import ToolTrie
from tatm.tooltrie_v1 import RelevancePreservingToolTrie


def tools() -> dict[str, CanonicalTool]:
    # names sort in the OPPOSITE order to the ids, so alphabetical order and
    # arrival order disagree and the difference between v0 and v1 is visible.
    return {
        f"t{i}": CanonicalTool(
            tool_id=f"t{i}",
            name=f"{chr(ord('z') - i)}_tool",
            description="d",
            parameters={},
            source="test",
            schema_tokens=10,
        )
        for i in range(5)
    }


def test_v1_preserves_arrival_order_when_nothing_matches():
    planner = RelevancePreservingToolTrie(tools(), recency_window=None)
    order = ("t3", "t0", "t4", "t1", "t2")
    plan = planner.plan(order)
    assert plan.ordered_ids == order, "unmatched tools must keep the incoming order"


def test_v0_does_not():
    planner = ToolTrie(tools(), recency_window=None)
    order = ("t3", "t0", "t4", "t1", "t2")
    assert planner.plan(order).ordered_ids != order


def test_v1_still_hoists_a_matched_prefix():
    planner = RelevancePreservingToolTrie(tools(), recency_window=None)
    first = ("t0", "t1", "t2", "t3", "t4")
    planner.observe(planner.plan(first).ordered_ids)
    plan = planner.plan(("t4", "t3", "t2", "t1", "t0"))
    assert set(plan.ordered_ids) == set(first)
    assert len(plan.matched_prefix_ids) >= 1, "the trie must still place a prefix"


def test_membership_is_never_changed():
    planner = RelevancePreservingToolTrie(tools(), recency_window=None)
    for order in (("t0", "t1"), ("t2", "t0", "t4"), ("t4", "t3", "t2", "t1", "t0")):
        plan = planner.plan(order)
        assert sorted(plan.ordered_ids) == sorted(order)
        planner.observe(plan.ordered_ids)
