"""通用数据结构:`Document(id, text, metadata)`、`Hit(doc, score, source)`。

metadata 只放标量(str/int/float/bool),以便直接写入 Chroma。
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """文本 + 元数据的哈希;增量索引据此判断是否需要重新嵌入。"""
        payload = self.text + "\x00" + repr(sorted(self.metadata.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Hit(BaseModel):
    doc: Document
    score: float
    source: str  # "bm25" | "vector" | "hybrid"
    rank: int = 0
    # 各路检索中的名次(1 起),便于分析融合效果
    ranks: dict[str, int] = Field(default_factory=dict)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
