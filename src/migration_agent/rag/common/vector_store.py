"""VectorStore 协议及实现:`upsert(ids, vectors, metadatas, documents) / query(vector, k, where) / delete / count`。

    ChromaStore           持久化(chromadb PersistentClient),余弦空间;每个子系统一个 collection
    InMemoryVectorStore   暴力余弦,适合小型语料;也是 chromadb 不可用时的退路

query 返回 `[(id, similarity)]`,similarity 越大越相似(Chroma 的 cosine distance 转成 1 - d)。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None: ...

    def query(
        self, vector: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]: ...

    def delete(self, ids: list[str]) -> None: ...

    def ids(self) -> set[str]: ...

    def count(self) -> int: ...


class ChromaStore:
    BATCH = 1000

    def __init__(self, path: Path | None, collection: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        if path is None:
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=chroma_settings)
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def upsert(self, ids, vectors, metadatas=None, documents=None) -> None:
        for i in range(0, len(ids), self.BATCH):
            sl = slice(i, i + self.BATCH)
            self._col.upsert(
                ids=ids[sl],
                embeddings=vectors[sl],
                metadatas=metadatas[sl] if metadatas else None,
                documents=documents[sl] if documents else None,
            )

    def query(self, vector, k, where=None):
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[vector], n_results=min(k, n), where=where or None, include=["distances"]
        )
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def delete(self, ids) -> None:
        for i in range(0, len(ids), self.BATCH):
            self._col.delete(ids=ids[i : i + self.BATCH])

    def ids(self) -> set[str]:
        return set(self._col.get(include=[])["ids"])

    def count(self) -> int:
        return self._col.count()


class InMemoryVectorStore:
    """暴力余弦。无过滤条件时用 numpy 矩阵乘(代码库索引每次查询上千块);有过滤时走纯 Python。"""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._matrix = None

    def upsert(self, ids, vectors, metadatas=None, documents=None) -> None:
        for n, (i, v) in enumerate(zip(ids, vectors, strict=True)):
            self._rows[i] = (v, (metadatas[n] if metadatas else {}) or {})
        self._matrix = None

    def query(self, vector, k, where=None):
        if not where and self._rows:
            import numpy as np

            if self._matrix is None:
                ids = list(self._rows)
                mat = np.asarray([self._rows[i][0] for i in ids], dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self._matrix = (ids, mat / np.where(norms == 0, 1.0, norms))
            ids, mat = self._matrix
            q = np.asarray(vector, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            sims = mat @ q
            top = np.argsort(-sims)[:k]
            return [(ids[i], float(sims[i])) for i in top]

        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)

        rows = [
            (i, cos(vector, v))
            for i, (v, meta) in self._rows.items()
            if not where or all(meta.get(key) == val for key, val in where.items())
        ]
        return sorted(rows, key=lambda r: -r[1])[:k]

    def delete(self, ids) -> None:
        for i in ids:
            self._rows.pop(i, None)
        self._matrix = None

    def ids(self) -> set[str]:
        return set(self._rows)

    def count(self) -> int:
        return len(self._rows)
