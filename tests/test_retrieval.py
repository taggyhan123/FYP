from __future__ import annotations

import pytest

from tatm.models import CanonicalTool
from tatm.retrieval import (
    BM25ToolRetriever,
    aggregate_retrieval_metrics,
    retrieval_metrics,
    tokenize_retrieval_text,
    tool_document_terms,
)


def tool(
    tool_id: str,
    name: str,
    description: str,
    properties: dict | None = None,
) -> CanonicalTool:
    return CanonicalTool(
        tool_id=tool_id,
        source="toolret:test",
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties or {}},
    )


TOOLS = {
    "weather": tool(
        "weather",
        "getCurrentWeather",
        "Return the weather forecast for a city.",
        {"city_name": {"type": "string", "description": "Target city"}},
    ),
    "calendar": tool(
        "calendar",
        "historical_events",
        "Find historical events on a calendar date.",
        {"date": {"type": "string"}},
    ),
    "sum": tool("sum", "add_numbers", "Add two numeric values."),
}


def test_tokenizer_splits_snake_hyphen_and_camel_case() -> None:
    assert tokenize_retrieval_text("getWeather city_name long-term") == (
        "get",
        "weather",
        "city",
        "name",
        "long",
        "term",
    )


def test_document_uses_semantic_schema_fields_but_not_json_noise() -> None:
    terms = tool_document_terms(TOOLS["weather"])
    assert terms.count("weather") >= 3
    assert "city" in terms
    assert "properties" not in terms
    assert "string" not in terms


def test_bm25_retrieves_by_query_without_gold_input() -> None:
    retriever = BM25ToolRetriever(TOOLS)
    result = retriever.retrieve("What is the weather in Paris?", k=2)
    assert result.tool_ids[0] == "weather"
    assert result.scores[0] > 0
    assert len(result.tool_ids) == 2


def test_bm25_zero_match_fallback_is_deterministic_and_fills_menu() -> None:
    retriever = BM25ToolRetriever(TOOLS)
    first = retriever.retrieve("zzzz qqqq", k=3)
    second = retriever.retrieve("zzzz qqqq", k=3)
    assert first == second
    assert first.tool_ids == ("sum", "weather", "calendar")
    assert first.fallback_count == 3


def test_retrieval_metrics_include_recall_hit_precision_and_mrr() -> None:
    row = retrieval_metrics(("x", "gold-a", "gold-b", "z"), ("gold-a", "gold-b"))
    assert row == {
        "retrieved": 4,
        "gold": 2,
        "relevant_retrieved": 2,
        "hit": 1,
        "recall": 1.0,
        "precision": 0.5,
        "reciprocal_rank": 0.5,
    }


def test_aggregate_retrieval_metrics_is_macro_averaged() -> None:
    rows = [
        retrieval_metrics(("a",), ("a",)),
        retrieval_metrics(("x",), ("b",)),
    ]
    summary = aggregate_retrieval_metrics(rows)
    assert summary["queries"] == 2
    assert summary["mean_recall"] == 0.5
    assert summary["hit_rate"] == 0.5
    assert summary["mean_reciprocal_rank"] == 0.5


def test_bm25_argument_validation() -> None:
    with pytest.raises(ValueError):
        BM25ToolRetriever({})
    with pytest.raises(ValueError):
        BM25ToolRetriever(TOOLS, b=2)
    with pytest.raises(ValueError):
        BM25ToolRetriever(TOOLS).retrieve("query", k=0)
