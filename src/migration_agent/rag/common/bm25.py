"""BM25Index:基于 rank_bm25 的稀疏检索,支持代码感知分词(驼峰/下划线拆分)。

    index = BM25Index(); index.build(docs); index.query(text, k) -> list[(id, score)]

分词同时保留整词与拆分后的子词:`int_parsing` → `int_parsing int parsing`,
`model_validator` / `modelValidator` → `model_validator model validator`。
报错码与 API 名在报错文本里原样出现,整词命中是混合检索相对纯向量的主要增益来源。
"""

from __future__ import annotations

import re

from .schema import Document

_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with"  # noqa: SIM905
    " you your can not no do does if when which we our i".split()  # noqa: SIM905
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower().strip("_")
        if not whole:
            continue
        parts = [p.lower() for piece in word.split("_") if piece for p in _CAMEL.findall(piece)]
        if whole not in _STOP:
            out.append(whole)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in _STOP and len(p) > 1)
    return out


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._bm25 = None

    def build(self, docs: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [d.id for d in docs]
        corpus = [tokenize(d.text) or ["<empty>"] for d in docs]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def __len__(self) -> int:
        return len(self._ids)
