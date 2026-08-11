import pytest

from tatm.baselines import (
    CacheWeaverPlanner,
    FittedOrderingPlanner,
    OnlineFrequencyPlanner,
    OnlinePairTriplePlanner,
)
from tatm.models import CanonicalTool


def make_tool(tool_id: str, tokens: int = 8) -> CanonicalTool:
    return CanonicalTool(
        tool_id=tool_id,
        source="test",
        name=tool_id,
        description="",
        parameters={"type": "object", "properties": {}},
        schema_tokens=tokens,
    )


TOOLS = {item: make_tool(item, index * 8) for index, item in enumerate("abcdef", 1)}


def test_cacheweaver_cold_fallback_preserves_original_order() -> None:
    planner = CacheWeaverPlanner(TOOLS, history_window=4)
    plan = planner.plan(("c", "a", "b"))
    assert plan.matched_prefix_ids == ()
    assert plan.ordered_ids == ("c", "a", "b")


def test_cacheweaver_checks_candidates_in_retrieval_order() -> None:
    planner = CacheWeaverPlanner(TOOLS, history_window=4)
    planner.observe(("a", "b"))
    planner.observe(("c", "d"))
    # Both a and c are cached root children. Algorithm 1 takes c because it is
    # encountered first in this request's retrieval order, not because of cost.
    plan = planner.plan(("c", "a", "b", "d"))
    assert plan.matched_prefix_ids == ("c", "d")
    assert plan.ordered_ids == ("c", "d", "a", "b")


def test_cacheweaver_history_window_removes_old_paths() -> None:
    planner = CacheWeaverPlanner(TOOLS, history_window=1)
    planner.observe(("a", "b"))
    planner.observe(("c", "d"))
    assert planner.plan(("a", "b")).matched_prefix_ids == ()
    assert planner.plan(("c", "d")).matched_prefix_ids == ("c", "d")
    assert planner.snapshot()["retained_paths"] == 1


def test_fitted_frequency_and_schema_cost_can_differ() -> None:
    training = [("a",), ("a",), ("a",), ("b",), ("b",)]
    frequency = FittedOrderingPlanner(TOOLS, training, policy="frequency_fitted")
    weighted = FittedOrderingPlanner(TOOLS, training, policy="schema_cost_fitted")
    assert frequency.plan(("a", "b")).ordered_ids == ("a", "b")
    # b has lower support but twice the schema-token cost (2*16 > 3*8).
    assert weighted.plan(("a", "b")).ordered_ids == ("b", "a")


def test_conditional_pair_uses_training_cooccurrence_after_first_item() -> None:
    training = [("a", "c"), ("a", "c"), ("b",), ("b",), ("b",)]
    planner = FittedOrderingPlanner(TOOLS, training, policy="conditional_pair")
    # b has greatest global support and is selected first. With no b pair,
    # remaining candidates fall back to global support: a then c.
    assert planner.plan(("a", "b", "c")).ordered_ids == ("b", "a", "c")


def test_fp_tree_conditional_preserves_selected_set() -> None:
    planner = FittedOrderingPlanner(
        TOOLS,
        [("a", "b", "c"), ("a", "b"), ("d", "e")],
        policy="fp_tree_conditional",
    )
    plan = planner.plan(("e", "d", "a"))
    assert set(plan.ordered_ids) == {"a", "d", "e"}
    assert len(plan.ordered_ids) == 3


def test_online_frequency_cold_start_is_alphabetical_by_name() -> None:
    planner = OnlineFrequencyPlanner(TOOLS)
    plan = planner.plan(("c", "a", "b"))
    # No request observed yet, so every count is zero and the name tiebreak decides.
    assert plan.ordered_ids == ("a", "b", "c")
    assert plan.matched_prefix_ids == ()


def test_online_frequency_orders_by_previously_served_presence() -> None:
    planner = OnlineFrequencyPlanner(TOOLS)
    for _ in range(3):
        planner.observe(("c", "a"))
    planner.observe(("b",))
    # c and a seen 3x, b seen 1x, d never.
    assert planner.plan(("d", "b", "a", "c")).ordered_ids == ("a", "c", "b", "d")


def test_online_frequency_places_never_seen_tools_last() -> None:
    planner = OnlineFrequencyPlanner(TOOLS)
    planner.observe(("e", "f"))
    ordered = planner.plan(("a", "e", "f")).ordered_ids
    assert set(ordered[:2]) == {"e", "f"}
    assert ordered[-1] == "a"


def test_online_frequency_plan_cannot_see_the_current_request() -> None:
    """plan() must depend only on strictly earlier requests."""
    planner = OnlineFrequencyPlanner(TOOLS)
    planner.observe(("a",))
    before = planner.plan(("b", "c"))
    # Planning twice without observing must be identical: no hidden state update.
    assert planner.plan(("b", "c")).ordered_ids == before.ordered_ids
    assert planner.snapshot()["requests_observed"] == 1


def test_online_frequency_is_always_a_permutation() -> None:
    planner = OnlineFrequencyPlanner(TOOLS)
    menus = [("a", "b", "c"), ("c", "d"), ("e", "f", "a"), ("b", "f", "d", "a")]
    for menu in menus:
        plan = planner.plan(menu)
        assert set(plan.ordered_ids) == set(menu)
        assert len(plan.ordered_ids) == len(menu)
        planner.observe(plan.ordered_ids)


def test_online_frequency_rejects_repeated_and_unknown_ids() -> None:
    planner = OnlineFrequencyPlanner(TOOLS)
    try:
        planner.plan(("a", "a"))
    except ValueError as error:
        assert "repeat" in str(error)
    else:
        raise AssertionError("expected ValueError for repeated tool IDs")
    try:
        planner.plan(("a", "zzz"))
    except ValueError as error:
        assert "unknown" in str(error).lower()
    else:
        raise AssertionError("expected ValueError for unknown tool IDs")


def _pt_tools():
    return {
        f"t{i}": CanonicalTool(
            tool_id=f"t{i}",
            name=f"tool_{i}",
            description="d",
            parameters={},
            source="test",
            schema_tokens=10,
        )
        for i in range(6)
    }


def test_online_pair_triple_plan_cannot_see_the_current_request():
    planner = OnlinePairTriplePlanner(_pt_tools())
    before = planner.snapshot()
    planner.plan(["t3", "t1", "t2"])
    assert planner.snapshot() == before, "plan must not mutate observed counts"


def test_online_pair_triple_cold_start_is_alphabetical_by_name():
    planner = OnlinePairTriplePlanner(_pt_tools())
    ordered = planner.plan(["t3", "t0", "t2"]).ordered_ids
    assert ordered == ("t0", "t2", "t3")


def test_online_pair_triple_is_always_a_permutation():
    planner = OnlinePairTriplePlanner(_pt_tools())
    menu = ["t4", "t1", "t0", "t5"]
    for _ in range(4):
        ordered = planner.plan(menu).ordered_ids
        assert sorted(ordered) == sorted(menu)
        planner.observe(ordered)


def test_online_pair_triple_prefers_a_partner_over_a_more_frequent_tool():
    """Pair support must outrank presence, or the policy is just frequency."""
    tools = _pt_tools()
    planner = OnlinePairTriplePlanner(tools, use_triples=False)
    for _ in range(5):
        planner.observe(["t0"])
    planner.observe(["t0", "t5"])
    for _ in range(3):
        planner.observe(["t1"])
    # presence: t0=6, t1=3, t5=1 -> t0 leads. Then t1 is the more frequent
    # candidate, but only t5 has ever co-occurred with t0.
    assert planner.plan(["t0", "t1", "t5"]).ordered_ids == ("t0", "t5", "t1")

    frequency = OnlineFrequencyPlanner(tools)
    for _ in range(5):
        frequency.observe(["t0"])
    frequency.observe(["t0", "t5"])
    for _ in range(3):
        frequency.observe(["t1"])
    # The same stream under a pure frequency counter orders t1 before t5.
    assert frequency.plan(["t0", "t1", "t5"]).ordered_ids == ("t0", "t1", "t5")


def test_online_pair_triple_counts_triples_only_when_enabled():
    with_triples = OnlinePairTriplePlanner(_pt_tools(), use_triples=True)
    without = OnlinePairTriplePlanner(_pt_tools(), use_triples=False)
    for planner in (with_triples, without):
        planner.observe(["t0", "t1", "t2"])
    assert with_triples.snapshot()["triples_seen"] == 1
    assert without.snapshot()["triples_seen"] == 0


def test_online_pair_triple_rejects_duplicate_ids():
    planner = OnlinePairTriplePlanner(_pt_tools())
    with pytest.raises(ValueError):
        planner.plan(["t0", "t0"])
