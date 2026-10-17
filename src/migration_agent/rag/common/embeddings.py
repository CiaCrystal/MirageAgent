"""EmbeddingProvider 协议:`embed(texts, is_query=False) -> list[list[float]]`,`dim`,`name`。

实现:
    SentenceTransformerEmbeddings  本地 sentence-transformers(默认 BAAI/bge-small-en-v1.5,CPU 可跑)
    HashingEmbeddings              确定性特征哈希,不依赖模型下载;用于离线运行

`make_embedder(spec)` 按字符串切换:`hash` / `hash:<dim>` / `st:<model>` / 直接给模型名。
OpenAI / Voyage 等远程实现只需满足同一协议,目前不内置。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Protocol

# bge 系列对查询加指令前缀效果更好;文档侧不加
_QUERY_PREFIXES = {"bge": "Represent this sentence for searching relevant passages: "}


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...


class SentenceTransformerEmbeddings:
    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32, device: str | None = None
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"st:{model_name}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self._prefix = next((p for key, p in _QUERY_PREFIXES.items() if key in model_name.lower()), "")

        self._lock = threading.Lock()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if is_query and self._prefix:
            texts = [self._prefix + t for t in texts]
        with self._lock:
            vectors = self._model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return [v.tolist() for v in vectors]


class HashingEmbeddings:
    """词袋 + 特征哈希 + L2 归一化。无语义,只用于测试管线连通性。"""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9_]+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                vec[h % self.dim] += 1.0 if h & 1 << 31 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


_CACHE: dict[str, EmbeddingProvider] = {}
_CACHE_LOCK = threading.Lock()


def make_embedder(spec: str) -> EmbeddingProvider:
    """按 spec 构造(并进程内缓存)embedding provider。"""
    with _CACHE_LOCK:
        if spec not in _CACHE:
            if spec == "hash" or spec.startswith("hash:"):
                _CACHE[spec] = HashingEmbeddings(int(spec.split(":", 1)[1]) if ":" in spec else 256)
            else:
                model = spec[3:] if spec.startswith("st:") else spec
                _CACHE[spec] = SentenceTransformerEmbeddings(model)
        return _CACHE[spec]
