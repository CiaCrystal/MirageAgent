"""CodebaseRetriever:混合检索 + 调用图扩展(命中块的上下游一跳)。

    retriever = CodebaseRetriever(index)
    retriever.retrieve(query, k=8, expand_graph=True, mode="hybrid") -> list[CodeHit]
    retriever.find_usages(symbol) -> list[Usage]

检索复用 rag/common 的 BM25 + 向量 + RRF;索引没有 embedder 时退化为纯 BM25。
图扩展:前 k 个命中里的函数/方法,其调用方与被调用方(仓库内、可静态解析的)作为附加结果追加在后面,"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast_chunker import CodeChunk
from .call_graph import Usage
from .indexer import CodebaseIndex


@dataclass
class CodeHit:
    chunk: CodeChunk
    score: float
    rank: int
    via: str = "search"  # search | caller | callee
    related_to: str = ""  # 图扩展:由哪个命中扩展而来
    ranks: dict[str, int] = field(default_factory=dict)


class CodebaseRetriever:
    def __init__(self, index: CodebaseIndex, *, expand_limit: int = 4, **hybrid_kwargs):
        self.index = index
        self.expand_limit = expand_limit
        self.hybrid_kwargs = hybrid_kwargs

    def retrieve(
        self, query: str, k: int = 8, *, expand_graph: bool = True, mode: str = "hybrid"
    ) -> list[CodeHit]:
        query = query.strip()
        if not query or not self.index.docs:
            return []
        if self.index.embedder is None:
            rows = self.index.bm25.query(query, k)
            hits = [
                CodeHit(self.index.chunks[i], s, n, ranks={"bm25": n}) for n, (i, s) in enumerate(rows, 1)
            ]
        else:
            found = self.index.hybrid(**self.hybrid_kwargs).retrieve(query, k, mode=mode)
            hits = [CodeHit(self.index.chunks[h.doc.id], h.score, h.rank, ranks=h.ranks) for h in found]
        if expand_graph:
            hits.extend(self._expand(hits))
        return hits

    def _expand(self, hits: list[CodeHit]) -> list[CodeHit]:
        graph = self.index.graph
        seen = {h.chunk.id for h in hits}
        extra: list[CodeHit] = []
        for h in hits:
            if h.chunk.kind not in ("function", "method"):
                continue
            key = f"{h.chunk.file}::{h.chunk.qualname}"
            for via, keys in (("callee", graph.callees(key)), ("caller", graph.callers(key))):
                for other in sorted(keys):
                    chunk = self.index.chunk_for(other)
                    if chunk is None or chunk.id in seen:
                        continue
                    seen.add(chunk.id)
                    extra.append(CodeHit(chunk, 0.0, len(hits) + len(extra) + 1, via, key))
                    if len(extra) >= self.expand_limit:
                        return extra
        return extra

    def find_usages(self, symbol: str) -> list[Usage]:
        return self.index.graph.usages_of(symbol)
