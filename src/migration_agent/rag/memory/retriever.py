"""经验检索：按状态、目标和版本过滤后，使用签名/API 匹配、BM25 和可选向量检索排序。MemorySession 控制重复注入与 token 预算。"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..common.bm25 import BM25Index
from ..common.schema import Document
from .schema import Experience, api_tags_in, error_api_tags, signature_key
from .store import estimate_tokens, render_entry, search_text

MODES = ("basic", "hybrid")


@dataclass
class MemoryHit:
    exp: Experience
    score: float
    match: str  # signature | type+api | api | bm25
    tokens: int
    text: str


@dataclass
class MemorySession:
    """一次 run 内的记忆使用记录:已展示条目(不重复注入)、累计注入 token、每次检索的元数据。"""

    shown: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    version: str | None = None
    version_checked: bool = False


def _first_exception_line(text: str) -> str:
    """默认签名函数:最后一条 `XxxError: ...` 行(调用方通常注入 ErrorParser 归一化,rag 层不依赖 migration)。"""
    lines = [ln.removeprefix("E").strip() for ln in text.splitlines()]
    exc = [ln for ln in lines if re.match(r"^(?:[\w.]+\.)?\w+(?:Error|Exception|Warning):", ln)]
    line = exc[-1] if exc else (lines[0] if lines else "")
    head, sep, body = line.partition(":")
    return f"{head.rsplit('.', 1)[-1]}:{body}" if sep else line


def _version_ok(exp: Experience, version: str | None) -> bool:
    if not version:
        return True
    from ...migration.targets import active

    spec = exp.versions.get(active().spec.memory.version_key)
    if not spec:
        return True
    try:
        return Version(version) in SpecifierSet(spec)
    except (InvalidSpecifier, InvalidVersion):
        return False


class MemoryRetriever:
    def __init__(
        self,
        memory: Any,
        *,
        target: str | None = None,
        policy: Any = None,
        min_bm25: float = 2.0,
        signature_fn: Callable[[str], str] | None = None,
        mode: str = "basic",
        embedder: Any = None,
        rrf_k: int = 60,
        weights: tuple[float, float, float] = (2.0, 1.0, 1.0),  # (exact, bm25, vector)
        vector_min: float = 0.75,
        hybrid_min_bm25: float = 3.0,
        vector_strong: float = 0.90,
        utility_weight: float = 0.25,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.memory = memory
        if target is None:
            from ...migration.targets import active

            target = active().name
        self.target = target
        self.policy = policy
        self.min_bm25 = min_bm25
        self.signature_fn = signature_fn or _first_exception_line
        self._entries = memory.entries()
        self._bm25: BM25Index | None = None
        self.mode, self.embedder, self.rrf_k, self.weights = mode, embedder, rrf_k, weights
        self.vector_min, self.vector_strong, self.utility_weight = vector_min, vector_strong, utility_weight
        self.hybrid_min_bm25 = hybrid_min_bm25
        self._vectors: dict[str, list[float]] | None = None
        self._bm25_hybrid: BM25Index | None = None

    def identity(self) -> dict[str, Any]:
        return self.memory.identity()

    def _filtered(self, version: str | None) -> list[tuple[dict, Experience]]:
        out = []
        for meta, exp in self._entries:
            if exp.status != "approved" or exp.target != self.target or not _version_ok(exp, version):
                continue
            if self.policy is not None and not self.policy.allows(exp)[0]:
                continue
            out.append((meta, exp))
        return out

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index()
            self._bm25.build([Document(id=m["id"], text=m["text"]) for m, _ in self._entries])
        return self._bm25

    def retrieve(
        self,
        error_text: str,
        code_context: str = "",
        *,
        version: str | None = None,
        k: int = 3,
        max_tokens: int = 1500,
        exclude: set[str] | list[str] = (),
    ) -> list[MemoryHit]:
        exclude = set(exclude)
        pool = [(m, e) for m, e in self._filtered(version) if e.id not in exclude]
        if not pool or not error_text.strip():
            return []
        if self.mode == "hybrid":
            return self._pack(self._hybrid(pool, error_text, code_context), k, max_tokens)
        scored = self._exact(pool, error_text, code_context)
        q_type, err_tags, ctx_tags = self._query_features(error_text, code_context)
        if len(scored) < k:
            allowed = {e.id: (m, e) for m, e in pool}
            ranked = [
                (i, s)
                for i, s in self._bm25_index().query(error_text + "\n" + code_context, k=20)
                if i in allowed
            ]
            top = ranked[0][1] if ranked else 0.0
            for exp_id, score in ranked:
                if exp_id in scored or score < self.min_bm25 or score < top / 2:
                    continue
                meta, exp = allowed[exp_id]
                if exp.error_type == q_type or set(exp.api_tags) & (err_tags | ctx_tags):
                    scored[exp_id] = (min(score / max(top, 1e-9), 0.99), "bm25", exp)
        return self._pack(scored, k, max_tokens)

    def _query_features(self, error_text: str, code_context: str) -> tuple[str, set[str], set[str]]:
        sig = self.signature_fn(error_text)
        err_tags = set(error_api_tags(error_text))
        ctx_tags = set(api_tags_in(code_context)) if code_context else set()
        return sig.split(":", 1)[0].strip(), err_tags, ctx_tags

    def _exact(
        self, pool: list[tuple[dict, Experience]], error_text: str, code_context: str
    ) -> dict[str, tuple[float, str, Experience]]:
        sig = self.signature_fn(error_text)
        q_keys = {signature_key(sig), signature_key(error_text.strip().splitlines()[0])}
        q_type = sig.split(":", 1)[0].strip()
        err_tags = set(error_api_tags(error_text))
        ctx_tags = set(api_tags_in(code_context)) if code_context else set()
        scored: dict[str, tuple[float, str, Experience]] = {}
        for meta, exp in pool:
            tags = set(exp.api_tags)
            overlap = len(tags & (err_tags | ctx_tags))
            stored_key = signature_key(re.sub(r"^(?:failure|error):\s*", "", exp.error_signature))
            if meta["sig_key"] in q_keys or stored_key in q_keys:
                scored[exp.id] = (10.0 + overlap, "signature", exp)
            elif q_type and exp.error_type == q_type and tags & (err_tags | ctx_tags):
                scored[exp.id] = (5.0 + overlap, "type+api", exp)
            elif tags & err_tags or len(tags & ctx_tags) >= 2:
                scored[exp.id] = (1.0 + overlap, "api", exp)
        return scored

    def _pack(
        self, scored: dict[str, tuple[float, str, Experience]], k: int, max_tokens: int
    ) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        used = 0
        for score, match, exp in sorted(scored.values(), key=lambda t: (-t[0], t[2].id)):
            if len(hits) >= k:
                break
            text = render_entry(exp)
            tokens = estimate_tokens(text)
            if used + tokens > max_tokens:
                continue  # 整条跳过,不截断
            hits.append(MemoryHit(exp, score, match, tokens, text))
            used += tokens
        return hits

    # ---------------------------------------------------------------- hybrid

    def _bm25_hybrid_index(self) -> BM25Index:
        if self._bm25_hybrid is None:
            self._bm25_hybrid = BM25Index()
            self._bm25_hybrid.build([Document(id=e.id, text=search_text(e)) for _, e in self._entries])
        return self._bm25_hybrid

    def _entry_vectors(self) -> dict[str, list[float]]:
        if self._vectors is None:
            if self.embedder is None:
                self._vectors = {}
            else:
                texts = [search_text(e)[:2000] for _, e in self._entries]
                vecs = self.embedder.embed(texts) if texts else []
                self._vectors = {e.id: v for (_, e), v in zip(self._entries, vecs, strict=True)}
        return self._vectors

    def rank_lists(
        self, pool: list[tuple[dict, Experience]], error_text: str, code_context: str = ""
    ) -> dict[str, list[tuple[str, float]]]:
        "Runtime support."
        allowed = {e.id for _, e in pool}
        exact = sorted(
            ((i, sc) for i, (sc, _, _) in self._exact(pool, error_text, code_context).items()),
            key=lambda t: (-t[1], t[0]),
        )
        query = error_text + "\n" + code_context
        bm25 = [
            (i, sc)
            for i, sc in self._bm25_hybrid_index().query(query, k=20)
            if i in allowed and sc >= self.hybrid_min_bm25
        ]
        vectors = self._entry_vectors()
        vec: list[tuple[str, float]] = []
        if vectors:
            q = self.embedder.embed([error_text[-2000:]], is_query=True)[0]
            sims = [
                (i, sum(a * b for a, b in zip(q, v, strict=True))) for i, v in vectors.items() if i in allowed
            ]
            vec = sorted(((i, s) for i, s in sims if s >= self.vector_min), key=lambda t: (-t[1], t[0]))[:20]
        return {"exact": exact, "bm25": bm25, "vector": vec}

    def _hybrid(
        self, pool: list[tuple[dict, Experience]], error_text: str, code_context: str
    ) -> dict[str, tuple[float, str, Experience]]:
        lists = self.rank_lists(pool, error_text, code_context)
        by_id = {e.id: e for _, e in pool}
        fused: dict[str, float] = {}
        for (_name, rows), w in zip(lists.items(), self.weights, strict=True):
            for rank, (i, _) in enumerate(rows, 1):
                fused[i] = fused.get(i, 0.0) + w / (self.rrf_k + rank)
        in_ = {name: {i for i, _ in rows} for name, rows in lists.items()}
        strong = {i for i, s in lists["vector"] if s >= self.vector_strong}
        out: dict[str, tuple[float, str, Experience]] = {}
        for i, score in fused.items():
            if i in in_["exact"]:
                match = "exact"
            elif i in in_["bm25"] and i in in_["vector"]:
                match = "bm25+vector"
            elif i in strong:
                match = "vector"
            else:
                continue  # 只有一路弱命中:不可靠
            u = max(-1.0, min(1.0, by_id[i].utility))
            out[i] = (score * 1000 * (1 + self.utility_weight * u), match, by_id[i])
        return out


def run_search(
    retriever: MemoryRetriever,
    session: MemorySession,
    error_text: str,
    code_context: str = "",
    *,
    version: str | None,
    k: int,
    max_tokens: int,
    purpose: str,
) -> tuple[list[MemoryHit], dict[str, Any]]:
    """一次检索 + 会话去重与记账;返回命中与写入 trace 的元数据。"""
    t0 = time.perf_counter()
    hits = retriever.retrieve(
        error_text, code_context, version=version, k=k, max_tokens=max_tokens, exclude=session.shown
    )
    latency = (time.perf_counter() - t0) * 1000
    tokens = sum(h.tokens for h in hits)
    session.shown += [h.exp.id for h in hits]
    session.injected_tokens += tokens
    ident = retriever.identity()
    meta = {
        "memory_purpose": purpose,
        "memory_hits": len(hits),
        "memory_hit_ids": ",".join(h.exp.id for h in hits),
        "memory_matches": ",".join(h.match for h in hits),
        "memory_sources": ",".join(sorted({s.run_id for h in hits for s in h.exp.sources})),
        "memory_tokens": tokens,
        "memory_latency_ms": round(latency, 2),
        "memory_snapshot": ident.get("snapshot_hash") or ident.get("content_version", ""),
        "memory_version": version or "",
    }
    session.calls.append(meta)
    return hits, meta
