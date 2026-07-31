from collections import Counter

import pytest

from tatm.models import CanonicalTool, TaskRecord
from tatm.prompting import build_menu, order_tool_ids, workload_record


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
    "a": make_tool("a", "Zulu", 10),
    "b": make_tool("b", "alpha", 20),
    "c": make_tool("c", "Mike", 30),
}
SUPPORT = Counter({"a": 5, "b": 5, "c": 1})

ORDERINGS = (
    "original",
    "alphabetical",
    "random",
    "frequency",
    "schema_cost_weighted",
    "fp_tree_global",
)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_ordering_preserves_the_tool_set(ordering: str) -> None:
    """Ordering may permute the menu but must never change which tools it holds.

    This is the invariant the whole ToolTrie baseline rests on: only order
    changes, so model semantics cannot shift.
    """
    ordered = order_tool_ids(("c", "a", "b"), TOOLS, SUPPORT, ordering)
    assert sorted(ordered) == ["a", "b", "c"]
    assert len(ordered) == len(set(ordered))


def test_original_deduplicates_and_keeps_input_order() -> None:
    assert order_tool_ids(("c", "a", "c", "b"), TOOLS, SUPPORT, "original") == (
        "c",
        "a",
        "b",
    )


def test_unknown_tool_ids_are_dropped() -> None:
    assert order_tool_ids(("a", "missing"), TOOLS, SUPPORT, "frequency") == ("a",)


def test_alphabetical_sorts_case_insensitively_by_name() -> None:
    # Names are alpha/Mike/Zulu; a naive case-sensitive sort would put the
    # lowercase "alpha" last.
    assert order_tool_ids(("a", "b", "c"), TOOLS, SUPPORT, "alphabetical") == (
        "b",
        "c",
        "a",
    )


def test_frequency_ranks_by_support_then_breaks_ties_by_id() -> None:
    # "a" and "b" both have support 5, so the id decides.
    assert order_tool_ids(("c", "b", "a"), TOOLS, SUPPORT, "frequency") == (
        "a",
        "b",
        "c",
    )


def test_schema_cost_weighted_ranks_by_support_times_tokens() -> None:
    # a: 5*10=50, b: 5*20=100, c: 1*30=30 -> b, a, c.
    assert order_tool_ids(
        ("a", "b", "c"), TOOLS, SUPPORT, "schema_cost_weighted"
    ) == ("b", "a", "c")


def test_ordering_is_independent_of_input_order() -> None:
    """A global order must not depend on how the retriever happened to emit it."""
    for ordering in ("alphabetical", "frequency", "schema_cost_weighted", "random"):
        forward = order_tool_ids(("a", "b", "c"), TOOLS, SUPPORT, ordering)
        reverse = order_tool_ids(("c", "b", "a"), TOOLS, SUPPORT, ordering)
        assert forward == reverse, ordering


def test_random_is_stable_per_seed_and_varies_across_seeds() -> None:
    first = order_tool_ids(("a", "b", "c"), TOOLS, SUPPORT, "random", random_seed=7)
    assert first == order_tool_ids(
        ("c", "a", "b"), TOOLS, SUPPORT, "random", random_seed=7
    )
    seeds = {
        order_tool_ids(("a", "b", "c"), TOOLS, SUPPORT, "random", random_seed=seed)
        for seed in range(12)
    }
    assert len(seeds) > 1


def test_fp_tree_global_matches_frequency() -> None:
    ids = ("c", "b", "a")
    assert order_tool_ids(ids, TOOLS, SUPPORT, "fp_tree_global") == order_tool_ids(
        ids, TOOLS, SUPPORT, "frequency"
    )


def test_unknown_ordering_raises() -> None:
    with pytest.raises(ValueError):
        order_tool_ids(("a",), TOOLS, SUPPORT, "not_an_ordering")


def test_build_menu_keeps_gold_and_pads_to_target() -> None:
    pool = [f"d{i}" for i in range(50)]
    menu = build_menu(("a", "b"), pool, 16, seed=1)
    assert len(menu) == 16
    assert menu[:2] == ("a", "b")
    assert len(set(menu)) == 16
    assert set(menu) - {"a", "b"} <= set(pool)


def test_build_menu_never_pads_with_gold_tools() -> None:
    """A distractor that duplicates a gold tool would silently shrink the menu."""
    pool = ["a", "b", "c", "d"]
    menu = build_menu(("a", "b"), pool, 4, seed=1)
    assert sorted(menu) == ["a", "b", "c", "d"]


def test_build_menu_is_deterministic_and_seed_sensitive() -> None:
    pool = [f"d{i}" for i in range(50)]
    assert build_menu(("a",), pool, 10, seed=3) == build_menu(("a",), pool, 10, seed=3)
    seeds = {tuple(build_menu(("a",), pool, 10, seed=s)) for s in range(6)}
    assert len(seeds) > 1


def test_build_menu_shares_padding_across_different_tasks() -> None:
    """Padding must come from one global catalog, not be resampled per task.

    Per-task padding makes every menu unique, which drives cross-request prefix
    reuse to nearly zero and silently defeats the experiment.
    """
    pool = [f"d{i}" for i in range(200)]
    first = set(build_menu(("a",), pool, 64, seed=7))
    second = set(build_menu(("b",), pool, 64, seed=7))
    shared = first & second
    # Only the gold entries should differ.
    assert len(shared) >= 62


def test_build_menu_returns_gold_when_target_is_not_larger() -> None:
    assert build_menu(("a", "b"), ["c"], 1) == ("a", "b")
    assert build_menu(("a", "b", "a"), ["c"], 2) == ("a", "b")


def test_workload_record_preserves_order_and_token_sum() -> None:
    task = TaskRecord("t1", "test", "query text", ("a", "b"), "gold", domain="x")
    record = workload_record(task, ("b", "a"), TOOLS, "frequency")
    assert record["tool_ids"] == ["b", "a"]
    assert [tool["function"]["name"] for tool in record["tools"]] == ["alpha", "Zulu"]
    assert record["canonical_tool_tokens"] == 30
    assert record["messages"] == [{"role": "user", "content": "query text"}]
    assert record["ordering"] == "frequency"
