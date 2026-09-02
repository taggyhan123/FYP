"""Dense (embedding) tool retrieval, as a counterpart to the BM25 baseline.

The project's retrieved workloads have all used ``BM25ToolRetriever``. That is a
lexical matcher, while most production tool routers select tools by embedding
similarity, and ContextPilot's own paper evaluates with a dense retriever
(gte-Qwen2-7B + FAISS) on three of its four datasets. Every "retrieved menu"
result therefore rests on one retriever type that no baseline here was designed
for.

This module supplies the other type behind the same interface, so a workload can
be rebuilt by swapping the retriever and nothing else. It deliberately reads the
same fields BM25 reads -- name, description, parameter names and values -- so the
two differ in *how* they match, not in what they are allowed to see.

Embeddings are cached on disk keyed by (model, corpus fingerprint), because the
corpus is ~46k tools and is re-embedded on every workload rebuild otherwise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tatm.models import CanonicalTool
from tatm.retrieval import RetrievalResult, _schema_terms

__all__ = ["DenseToolRetriever", "tool_document_text"]

# bge-en-v1.5 is trained with an instruction on the *query* side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def tool_document_text(tool: CanonicalTool, *, max_chars: int = 1024) -> str:
    """Natural-language document for one tool.

    Same source fields as ``tool_document_terms``, but left as prose: an encoder
    reads word order, so the triplicated function name BM25 uses as a field
    weight would only distort it.
    """
    schema_values = []
    seen: set[str] = set()
    for value in _schema_terms(tool.parameters):
        text = str(value).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            schema_values.append(text)
    parts = [tool.name, tool.description.strip()]
    if schema_values:
        parts.append("Parameters: " + ", ".join(schema_values))
    return " ".join(part for part in parts if part)[:max_chars]


class DenseToolRetriever:
    """Cosine-similarity retrieval over sentence-embedded tool definitions.

    Mirrors ``BM25ToolRetriever``: construct over a tool mapping, then call
    ``retrieve(query, k=...)``. Scores are cosine similarities in [-1, 1] rather
    than BM25 sums, so they are comparable within a query but not across
    retrievers -- only the returned *ordering* is used downstream.
    """

    def __init__(
        self,
        tools: Mapping[str, CanonicalTool],
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | None = None,
        device: str | None = None,
        batch_size: int = 256,
        max_length: int = 256,
    ) -> None:
        if not tools:
            raise ValueError("tools must not be empty")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        # Sorted so the corpus fingerprint and row order are reproducible.
        self._tool_ids = tuple(sorted(tools))
        self._tools = dict(tools)
        self._device = device
        self._cache_dir = cache_dir or (
            Path(__file__).resolve().parents[2] / "data" / "embeddings"
        )
        self._model = None
        self._tokenizer = None
        self._matrix = self._load_or_build_corpus()

    # -- fingerprinting -----------------------------------------------------
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        digest.update(str(self.max_length).encode())
        for tool_id in self._tool_ids:
            digest.update(tool_id.encode())
            digest.update(tool_document_text(self._tools[tool_id]).encode())
        return digest.hexdigest()[:16]

    # -- encoder ------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        self._model = model.to(self._device).eval()

    def _encode(self, texts: Sequence[str]):
        import torch

        self._ensure_model()
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            # bge pools the [CLS] position, then L2-normalizes.
            pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.to("cpu"))
        return torch.cat(out) if out else torch.empty(0)

    def _load_or_build_corpus(self):
        import numpy
        import torch

        fingerprint = self._fingerprint()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        slug = self.model_name.replace("/", "__")
        path = self._cache_dir / f"{slug}-{fingerprint}.npy"
        meta = path.with_suffix(".json")
        if path.exists():
            return torch.from_numpy(numpy.load(path))
        matrix = self._encode(
            [tool_document_text(self._tools[t]) for t in self._tool_ids]
        )
        numpy.save(path, matrix.numpy())
        meta.write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "corpus_size": len(self._tool_ids),
                    "dim": int(matrix.shape[1]),
                    "max_length": self.max_length,
                    "pooling": "cls",
                    "normalized": True,
                    "fingerprint": fingerprint,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return matrix

    # -- interface ----------------------------------------------------------
    @property
    def corpus_size(self) -> int:
        return len(self._tool_ids)

    @property
    def embedding_dim(self) -> int:
        return int(self._matrix.shape[1])

    def retrieve(self, query: str, *, k: int) -> RetrievalResult:
        import torch

        if k < 1:
            raise ValueError("k must be >= 1")
        limit = min(k, self.corpus_size)
        vector = self._encode([QUERY_INSTRUCTION + query])[0]
        scores = self._matrix @ vector
        top = torch.topk(scores, limit)
        # Ties broken by tool_id, matching BM25ToolRetriever's determinism.
        ranked = sorted(
            ((float(s), self._tool_ids[int(i)]) for s, i in zip(top.values, top.indices)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return RetrievalResult(
            tool_ids=tuple(tool_id for _, tool_id in ranked),
            scores=tuple(score for score, _ in ranked),
            scored_candidates=self.corpus_size,
            fallback_count=0,
        )
