"""KnowledgeRetriever:输入报错信息或问题描述,输出相关文档片段。

    retriever = KnowledgeRetriever.load(index_dir)
    retriever.retrieve(query, error_type=None, k=5, mode="hybrid", source_kind=None) -> list[Hit]

内置报错归一化 `normalize_query`:输入像 traceback 时,只保留最后一个异常及其后续说明行,
去掉文件路径、行号、内存地址、`input_value=` 等输入值与模型名;
把报错里的文档链接还原成错误码(目标包 `errors.doc_url_code_pattern`,与文档小节锚点同名)。
自然语言问句原样使用。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from ..common.bm25 import BM25Index
from ..common.embeddings import EmbeddingProvider, make_embedder
from ..common.hybrid import HybridRetriever
from ..common.schema import Hit
from ..common.vector_store import ChromaStore
from .indexer import COLLECTION, load_docs, read_meta

_EXC = re.compile(r"^\s*(?:E\s+)?((?:[\w]+\.)*\w*(?:Error|Exception|Warning))(?::\s*(.*))?$")
_FRAME = re.compile(r'^\s*File "[^"]*", line \d+')
_ERR_URL_CACHE: dict[str, re.Pattern[str] | None] = {}


def _err_url() -> re.Pattern[str] | None:
    from ...migration.targets import active

    pat = active().spec.errors.doc_url_code_pattern
    if pat not in _ERR_URL_CACHE:
        _ERR_URL_CACHE[pat] = re.compile(pat) if pat else None
    return _ERR_URL_CACHE[pat]


_TYPE_CODE = re.compile(r"\[type=([\w]+)")


def looks_like_traceback(text: str) -> bool:
    return bool(
        "Traceback (most recent call last)" in text
        or re.search(r"(?m)^E\s{2,}", text)
        or (text.count("\n") >= 1 and any(_EXC.match(ln) for ln in text.splitlines()))
    )


def normalize_query(text: str, error_type: str | None = None) -> str:
    text = text.strip()
    if not looks_like_traceback(text):
        q = re.sub(r"\s+", " ", text)
        return f"{error_type} {q}" if error_type else q
    lines = [re.sub(r"^E\s+", "", ln) for ln in text.splitlines()]
    err_url = _err_url()
    codes = list(dict.fromkeys((err_url.findall(text) if err_url else []) + _TYPE_CODE.findall(text)))
    idx = [i for i, ln in enumerate(lines) if _EXC.match(ln)]
    tail = lines[idx[-1] : idx[-1] + 16] if idx else lines[-16:]
    kept = []
    for ln in tail:
        if _FRAME.match(ln) or not ln.strip():
            continue
        if err_url is not None:
            ln = err_url.sub(lambda m: m.group(1), ln)
        ln = re.sub(r"https?://\S+", "", ln)
        ln = re.sub(r"For further information visit\s*", "", ln)
        ln = re.sub(
            r",?\s*input_(?:value|type)=(?:'[^']*'|\"[^\"]*\"|\{[^}]*\}|\[[^\]]*\]|[^,\]\s]+)", "", ln
        )
        ln = re.sub(r"\b(\d+ validation errors? for)\s+\S+", "validation error", ln)
        ln = re.sub(r"(?:[A-Za-z]:\\|/)[^\s:'\"]+", "", ln)
        ln = re.sub(r"0x[0-9a-fA-F]+", "", ln)
        ln = re.sub(r"\bline \d+\b", "", ln)
        ln = re.sub(r"^(?:[\w]+\.)+(\w+(?:Error|Exception|Warning))", r"\1", ln.strip())
        kept.append(ln)
    q = re.sub(r"\s+", " ", " ".join(kept)).strip()
    missing = [c for c in codes if c not in q]
    if missing:
        q = f"{q} {' '.join(missing)}"
    if error_type and error_type not in q:
        q = f"{error_type} {q}"
    return q[:1000]


class KnowledgeRetriever:
    def __init__(self, hybrid: HybridRetriever, meta: dict):
        self.hybrid, self.meta = hybrid, meta

    @classmethod
    def load(
        cls, index_dir: Path, embedder: EmbeddingProvider | None = None, **hybrid_kwargs
    ) -> KnowledgeRetriever:
        meta = read_meta(index_dir)
        docs = {d.id: d for d in load_docs(index_dir)}
        bm25 = BM25Index()
        bm25.build(list(docs.values()))
        embedder = embedder or make_embedder(meta["embedder"])
        if embedder.name != meta["embedder"]:
            raise ValueError(f"index built with {meta['embedder']}, got {embedder.name}")
        store = ChromaStore(index_dir / "chroma", COLLECTION)
        return cls(HybridRetriever(docs, bm25, store, embedder, **hybrid_kwargs), meta)

    def retrieve(
        self,
        query: str,
        error_type: str | None = None,
        k: int = 5,
        *,
        mode: str = "hybrid",
        source_kind: str | None = None,
    ) -> list[Hit]:
        q = normalize_query(query, error_type)
        if not q:
            return []
        return self.hybrid.retrieve(
            q, k, mode=mode, where={"source_kind": source_kind} if source_kind else None
        )


_LOADED: dict[str, KnowledgeRetriever] = {}
_LOCK = threading.Lock()


def shared_retriever(index_dir: Path) -> KnowledgeRetriever:
    "进程内共享知识检索器，避免重复加载模型与索引。"
    key = str(index_dir.resolve())
    with _LOCK:
        if key not in _LOADED:
            _LOADED[key] = KnowledgeRetriever.load(index_dir)
        return _LOADED[key]
