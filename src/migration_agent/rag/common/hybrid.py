"""HybridRetriever:融合 BM25 与向量检索结果(Reciprocal Rank Fusion,可配权重),可选 rerank 钩子。

    retriever.retrieve(query, k, mode="hybrid" | "bm25" | "vector") -> list[Hit]

RRF:score(d) = Σ_r w_r / (rrf_k + rank_r(d)),只依赖名次,BM25 分数与余弦相似度不必同尺度。
`where` 是可选的元数据等值过滤(两路同时生效)。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bm25 import BM25Index
from .embeddings import EmbeddingProvider
from .schema import Document, Hit
from .vector_store import VectorStore

MODES = ("hybrid", "bm25", "vector")
Reranker = Callable[[str, list[Hit]], list[Hit]]


class HybridRetriever:
    def __init__(
        self,
        docs: dict[str, Document],
        bm25: BM25Index,
        store: VectorStore,
        embedder: EmbeddingProvider,
        *,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),  # (bm25, vector)
        candidates: int = 50,
        reranker: Reranker | None = None,
    ):
        self.docs, self.bm25, self.store, self.embedder = docs, bm25, store, embedder
        self.rrf_k, self.weights, self.candidates, self.reranker = rrf_k, weights, candidates, reranker

    def retrieve(
        self, query: str, k: int = 5, *, mode: str = "hybrid", where: dict[str, Any] | None = None
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = max(k, self.candidates)
        ranked: dict[str, list[tuple[str, float]]] = {}
        if mode in ("hybrid", "bm25"):
            rows = self.bm25.query(query, n * 4 if where else n)
            if where:
                rows = [r for r in rows if _match(self.docs.get(r[0]), where)]
            ranked["bm25"] = rows[:n]
        if mode in ("hybrid", "vector"):
            vector = self.embedder.embed([query], is_query=True)[0]
            ranked["vector"] = self.store.query(vector, n, where=where)

        if mode == "hybrid":
            fused: dict[str, float] = {}
            for name, weight in zip(("bm25", "vector"), self.weights, strict=True):
                for rank, (doc_id, _) in enumerate(ranked[name], 1):
                    fused[doc_id] = fused.get(doc_id, 0.0) + weight / (self.rrf_k + rank)
            order = sorted(fused.items(), key=lambda r: -r[1])
        else:
            order = ranked[mode]

        positions = {
            name: {doc_id: i for i, (doc_id, _) in enumerate(rows, 1)} for name, rows in ranked.items()
        }
        hits = [
            Hit(
                doc=self.docs[doc_id],
                score=score,
                source=mode,
                ranks={name: pos[doc_id] for name, pos in positions.items() if doc_id in pos},
            )
            for doc_id, score in order
            if doc_id in self.docs
        ]
        if self.reranker:
            hits = self.reranker(query, hits)
        hits = hits[:k]
        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits


def _match(doc: Document | None, where: dict[str, Any]) -> bool:
    return doc is not None and all(doc.metadata.get(key) == val for key, val in where.items())
