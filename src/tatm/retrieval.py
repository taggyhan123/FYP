from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from tatm.models import CanonicalTool


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SCHEMA_NOISE = {
    "$schema",
    "additionalproperties",
    "allof",
    "anyof",
    "array",
    "boolean",
    "default",
    "description",
    "enum",
    "integer",
    "items",
    "null",
    "number",
    "object",
    "oneof",
    "properties",
    "required",
    "string",
    "title",
    "type",
}


def tokenize_retrieval_text(value: str) -> tuple[str, ...]:
    """Tokenize retrieval text deterministically without a model dependency."""
    expanded = _CAMEL_BOUNDARY.sub(" ", value.replace("_", " ").replace("-", " "))
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(expanded))


def _schema_terms(value: Any, *, parent_key: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = key.casefold().replace("_", "")
            if normalized_key not in _SCHEMA_NOISE:
                yield key
            yield from _schema_terms(child, parent_key=key)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_terms(child, parent_key=parent_key)
    elif isinstance(value, str):
        normalized = value.casefold().replace("_", "")
        if normalized not in _SCHEMA_NOISE:
            yield value


def tool_document_terms(tool: CanonicalTool) -> tuple[str, ...]:
    """Build the BM25 document from fields available before evaluation.

    Function-name terms are repeated three times as a small, declared field
    weight. Descriptions and meaningful parameter names/values are included;
    generic JSON-Schema syntax is dropped because it appears in nearly every
    document and carries no retrieval signal.
    """
    name_terms = tokenize_retrieval_text(tool.name)
    description_terms = tokenize_retrieval_text(tool.description)
    schema_terms = tuple(
        token
        for value in _schema_terms(tool.parameters)
        for token in tokenize_retrieval_text(value)
    )
    return (*name_terms, *name_terms, *name_terms, *description_terms, *schema_terms)


@dataclass(frozen=True)
class RetrievalResult:
    tool_ids: tuple[str, ...]
    scores: tuple[float, ...]
    scored_candidates: int
    fallback_count: int


class BM25ToolRetriever:
    """A deterministic sparse BM25 baseline over canonical tool definitions."""

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if not tools:
            raise ValueError("tools must not be empty")

        self.k1 = k1
        self.b = b
        self._tools = dict(tools)
        self._fallback_order = tuple(
            sorted(
                self._tools,
                key=lambda tool_id: (
                    self._tools[tool_id].name.casefold(),
                    tool_id,
                ),
            )
        )
        self._document_lengths: dict[str, int] = {}
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for tool_id in sorted(self._tools):
            terms = tool_document_terms(self._tools[tool_id])
            frequencies = Counter(terms)
            self._document_lengths[tool_id] = len(terms)
            for term, frequency in frequencies.items():
                postings[term].append((tool_id, frequency))
        self._postings = dict(postings)
        self._average_document_length = fmean(self._document_lengths.values())

    @property
    def corpus_size(self) -> int:
        return len(self._tools)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    @property
    def average_document_length(self) -> float:
        return self._average_document_length

    def retrieve(self, query: str, *, k: int) -> RetrievalResult:
        if k < 1:
            raise ValueError("k must be >= 1")
        limit = min(k, self.corpus_size)
        query_terms = Counter(tokenize_retrieval_text(query))
        scores: dict[str, float] = defaultdict(float)
        corpus_size = self.corpus_size
        average_length = self.average_document_length or 1.0

        for term, query_frequency in query_terms.items():
            posting = self._postings.get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_document_frequency = math.log(
                1.0
                + (corpus_size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            query_weight = 1.0 + math.log(query_frequency)
            for tool_id, term_frequency in posting:
                document_length = self._document_lengths[tool_id]
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                scores[tool_id] += (
                    inverse_document_frequency
                    * query_weight
                    * (term_frequency * (self.k1 + 1.0) / denominator)
                )

        ranked_scored = sorted(scores, key=lambda item: (-scores[item], item))
        selected = ranked_scored[:limit]
        selected_set = set(selected)
        if len(selected) < limit:
            selected.extend(
                tool_id
                for tool_id in self._fallback_order
                if tool_id not in selected_set
            )
            selected = selected[:limit]

        scored_count = min(len(ranked_scored), limit)
        return RetrievalResult(
            tool_ids=tuple(selected),
            scores=tuple(round(scores.get(tool_id, 0.0), 8) for tool_id in selected),
            scored_candidates=len(ranked_scored),
            fallback_count=limit - scored_count,
        )


def retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> dict[str, int | float]:
    retrieved = tuple(dict.fromkeys(retrieved_ids))
    gold = set(gold_ids)
    relevant_ranks = [
        rank for rank, tool_id in enumerate(retrieved, start=1) if tool_id in gold
    ]
    relevant_retrieved = len(relevant_ranks)
    return {
        "retrieved": len(retrieved),
        "gold": len(gold),
        "relevant_retrieved": relevant_retrieved,
        "hit": int(bool(relevant_ranks)),
        "recall": round(relevant_retrieved / len(gold), 8) if gold else 1.0,
        "precision": round(relevant_retrieved / len(retrieved), 8)
        if retrieved
        else 0.0,
        "reciprocal_rank": round(1.0 / relevant_ranks[0], 8)
        if relevant_ranks
        else 0.0,
    }


def aggregate_retrieval_metrics(
    rows: Sequence[Mapping[str, int | float]],
) -> dict[str, int | float]:
    if not rows:
        return {
            "queries": 0,
            "mean_recall": 0.0,
            "hit_rate": 0.0,
            "mean_precision": 0.0,
            "mean_reciprocal_rank": 0.0,
        }
    return {
        "queries": len(rows),
        "mean_recall": round(fmean(float(row["recall"]) for row in rows), 8),
        "hit_rate": round(fmean(float(row["hit"]) for row in rows), 8),
        "mean_precision": round(
            fmean(float(row["precision"]) for row in rows), 8
        ),
        "mean_reciprocal_rank": round(
            fmean(float(row["reciprocal_rank"]) for row in rows), 8
        ),
    }
