"""KnowledgeIndexer:抓取 → 切块 → 嵌入 → 入库,支持增量更新。

    KnowledgeIndexer(index_dir, embedder).build(sources, source_manifest) -> IndexStats
    build_knowledge_index(settings, refresh=False)  一站式:Fetcher 抓取 + build

索引目录布局:
    docs.jsonl   全部切块(Document + content_hash),BM25 与结果展示都从这里读
    meta.json    embedder 名称/维度、各来源块数、语料哈希、来源清单(URL、commit、issue 抓取参数)
    chroma/      向量(collection `knowledge`)

增量:按块 id 比较 content_hash,只为新增/变化的块计算嵌入,删除已消失的块;"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..common.embeddings import EmbeddingProvider, make_embedder
from ..common.schema import Document
from ..common.vector_store import ChromaStore, VectorStore
from .chunking import chunk_source
from .sources import RawSource

COLLECTION = "knowledge"


@dataclass
class IndexStats:
    docs: int
    embedded: int
    deleted: int
    by_kind: dict[str, int]
    corpus_hash: str
    seconds: float


def load_docs(index_dir: Path) -> list[Document]:
    path = index_dir / "docs.jsonl"
    if not path.exists():
        return []
    return [
        Document.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def read_meta(index_dir: Path) -> dict:
    path = index_dir / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"knowledge index not built: {index_dir} (run `migration-agent index`)")
    return json.loads(path.read_text(encoding="utf-8"))


class KnowledgeIndexer:
    def __init__(self, index_dir: Path, embedder: EmbeddingProvider, store: VectorStore | None = None):
        self.index_dir = index_dir
        self.embedder = embedder
        self.store = store

    def _store(self) -> VectorStore:
        if self.store is None:
            self.store = ChromaStore(self.index_dir / "chroma", COLLECTION)
        return self.store

    def build(
        self, sources: list[RawSource], source_manifest: dict | None = None, batch: int = 64
    ) -> IndexStats:
        t0 = time.monotonic()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        docs: dict[str, Document] = {}
        for src in sources:
            for d in chunk_source(src):
                docs.setdefault(d.id, d)
        store = self._store()

        old_meta = {}
        if (self.index_dir / "meta.json").exists():
            old_meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        same_embedder = old_meta.get("embedder") == self.embedder.name
        old_hash = {d.id: d.metadata.get("_hash") for d in load_docs(self.index_dir)} if same_embedder else {}
        stored = store.ids()
        if not same_embedder and stored:
            store.delete(sorted(stored))
            stored = set()

        for d in docs.values():
            d.metadata["_hash"] = d.content_hash()
        todo = [d for d in docs.values() if d.id not in stored or old_hash.get(d.id) != d.metadata["_hash"]]
        stale = sorted(stored - docs.keys())
        if stale:
            store.delete(stale)
        for i in range(0, len(todo), batch):
            part = todo[i : i + batch]
            vectors = self.embedder.embed([d.text for d in part])
            store.upsert(
                [d.id for d in part],
                vectors,
                [
                    {k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))}
                    for d in part
                ],
                [d.text for d in part],
            )

        ordered = sorted(
            docs.values(), key=lambda d: (d.metadata["source_kind"], d.metadata["heading"], d.id)
        )
        corpus_hash = hashlib.sha256(
            "".join(d.id + d.metadata["_hash"] for d in ordered).encode()
        ).hexdigest()[:16]
        (self.index_dir / "docs.jsonl").write_text(
            "".join(d.model_dump_json() + "\n" for d in ordered), encoding="utf-8"
        )
        by_kind = dict(Counter(d.metadata["source_kind"] for d in ordered))
        meta = {
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "docs": len(ordered),
            "by_kind": by_kind,
            "corpus_hash": corpus_hash,
            "sources": source_manifest or old_meta.get("sources", {}),
        }
        (self.index_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return IndexStats(len(ordered), len(todo), len(stale), by_kind, corpus_hash, time.monotonic() - t0)


def build_knowledge_index(
    index_dir: Path, raw_dir: Path, embedder_spec: str, *, refresh: bool = False
) -> IndexStats:
    from .sources import Fetcher

    fetcher = Fetcher(raw_dir, refresh=refresh)
    sources, source_manifest = fetcher.fetch_all()
    return KnowledgeIndexer(index_dir, make_embedder(embedder_spec)).build(sources, source_manifest)
