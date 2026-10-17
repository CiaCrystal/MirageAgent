"""CodebaseIndexer:遍历仓库 → AST 切块 → BM25 + 向量索引 → 调用图;文件变更后增量重建。

    index = CodebaseIndex(embedder)          # embedder=None 时只建 BM25 与调用图(Planner / find_usages 够用)
    index.index(files)  -> IndexStats        # files: {rel_path: source}
    index.update({"pkg/a.py": new_source, "pkg/old.py": None})  -> IndexStats   # None 表示删除
    index.sync(files)   -> IndexStats        # 与当前内容比对,只更新变化的文件
    load_repo_files(root) / files_from_tree(ctx.files.tree())

增量:按文件内容哈希判断变化;变化文件重新切块,只为新出现的块文本计算嵌入(向量按块文本哈希缓存),
删除消失的块;BM25 在全量块上重建(纯 Python 但只是分词计数,千块级 < 1 s);调用图只重新解析该文件。
向量放在 `InMemoryVectorStore`:索引随 run 存在、不落盘,不同 run 互不影响。
"""

from __future__ import annotations

import base64
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..common.bm25 import BM25Index
from ..common.embeddings import EmbeddingProvider
from ..common.hybrid import HybridRetriever
from ..common.schema import Document
from ..common.vector_store import InMemoryVectorStore
from .ast_chunker import CodeChunk, chunk_file
from .call_graph import CallGraph

SKIP_PARTS = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", "build", "dist", ".migration_agent"}
)
MAX_FILE_BYTES = 400_000
EMBED_CHARS = 2000  # bge 系列最多 512 token,超出部分模型本身也会截断


def _wanted(rel: str) -> bool:
    parts = rel.split("/")
    return (
        rel.endswith(".py")
        and not SKIP_PARTS & set(parts)
        and not any(p.endswith(".egg-info") for p in parts)
    )


def load_repo_files(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if _wanted(rel) and p.is_file() and p.stat().st_size <= MAX_FILE_BYTES:
            out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def files_from_tree(tree: dict) -> dict[str, str]:
    """`RepositoryFiles.tree()` 的结果(base64 内容)→ {path: source},只保留 .py。"""
    out = {}
    for rel, entry in tree.items():
        if not _wanted(rel):
            continue
        data = base64.b64decode(entry["data"])
        if len(data) <= MAX_FILE_BYTES:
            out[rel] = data.decode("utf-8", errors="replace")
    return out


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class IndexStats:
    files: int
    chunks: int
    changed_files: int
    embedded: int
    seconds: float
    phases: dict[str, float] = field(default_factory=dict)


class CodebaseIndex:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        *,
        chunker: Callable[[str, str], list[CodeChunk]] = chunk_file,
        batch: int = 32,
    ):
        self.embedder = embedder
        self.chunker = chunker
        self.batch = batch
        self.files: dict[str, str] = {}
        self._file_hash: dict[str, str] = {}
        self.chunks_by_file: dict[str, list[CodeChunk]] = {}
        self.chunks: dict[str, CodeChunk] = {}
        self.docs: dict[str, Document] = {}
        self.bm25 = BM25Index()
        self.store = InMemoryVectorStore()
        self.graph = CallGraph()
        self._vec_cache: dict[str, list[float]] = {}
        self._by_qualname: dict[str, str] = {}

    # ---- 构建 ----
    def index(self, files: dict[str, str]) -> IndexStats:
        self.files, self._file_hash = {}, {}
        self.chunks_by_file, self.chunks, self.docs = {}, {}, {}
        self.store = InMemoryVectorStore()
        self.graph = CallGraph()
        return self.update(dict(files))

    def sync(self, files: dict[str, str]) -> IndexStats:
        changed: dict[str, str | None] = {
            rel: src for rel, src in files.items() if self._file_hash.get(rel) != _hash(src)
        }
        changed.update({rel: None for rel in self.files.keys() - files.keys()})
        return self.update(changed)

    def update(self, changed: dict[str, str | None]) -> IndexStats:
        t0 = time.perf_counter()
        phases: dict[str, float] = {}
        changed = {rel: src for rel, src in changed.items() if _wanted(rel)}
        removed_ids: list[str] = []
        new_chunks: list[CodeChunk] = []
        t = time.perf_counter()
        for rel, src in changed.items():
            for c in self.chunks_by_file.pop(rel, []):
                self.chunks.pop(c.id, None)
                self.docs.pop(c.id, None)
                removed_ids.append(c.id)
            if src is None:
                self.files.pop(rel, None)
                self._file_hash.pop(rel, None)
                continue
            self.files[rel] = src
            self._file_hash[rel] = _hash(src)
            chunks = self.chunker(rel, src)
            self.chunks_by_file[rel] = chunks
            for c in chunks:
                self.chunks[c.id] = c
                self.docs[c.id] = c.to_document()
            new_chunks.extend(chunks)
        phases["chunk"] = time.perf_counter() - t

        t = time.perf_counter()
        still = set(self.chunks)
        self.store.delete([i for i in removed_ids if i not in still])
        embedded = 0
        if self.embedder is not None and new_chunks:
            todo = [c for c in new_chunks if _hash(c.text[:EMBED_CHARS]) not in self._vec_cache]
            for i in range(0, len(todo), self.batch):
                part = todo[i : i + self.batch]
                vectors = self.embedder.embed([c.text[:EMBED_CHARS] for c in part])
                for c, v in zip(part, vectors, strict=True):
                    self._vec_cache[_hash(c.text[:EMBED_CHARS])] = v
            embedded = len(todo)
            self.store.upsert(
                [c.id for c in new_chunks],
                [self._vec_cache[_hash(c.text[:EMBED_CHARS])] for c in new_chunks],
            )
        phases["embed"] = time.perf_counter() - t

        t = time.perf_counter()
        self.bm25.build(list(self.docs.values()))
        phases["bm25"] = time.perf_counter() - t

        t = time.perf_counter()
        for rel, src in changed.items():
            self.graph.update_file(rel, src)
        self._by_qualname = {}
        for c in self.chunks.values():
            self._by_qualname.setdefault(f"{c.file}::{c.qualname}", c.id)
        phases["graph"] = time.perf_counter() - t
        return IndexStats(
            len(self.files), len(self.chunks), len(changed), embedded, time.perf_counter() - t0, phases
        )

    # ---- 查询辅助 ----
    def chunk_for(self, key: str) -> CodeChunk | None:
        """`file::qualname` → 该定义的(第一个)块。"""
        cid = self._by_qualname.get(key)
        return self.chunks.get(cid) if cid else None

    def hybrid(self, **kwargs) -> HybridRetriever:
        if self.embedder is None:
            raise RuntimeError("vector retrieval needs an embedder; build the index with one")
        return HybridRetriever(self.docs, self.bm25, self.store, self.embedder, **kwargs)
