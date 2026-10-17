"""知识库、代码库与经验检索工具及初始上下文提供者。检索元数据写入运行记录。"""

from __future__ import annotations

from typing import Any, Protocol

from .base import Tool, ToolContext, ToolResult

SOURCE_KINDS = ["guide", "docs", "errors", "release_notes", "issue"]


class _Retriever(Protocol):
    def retrieve(self, query: str, error_type: str | None = None, k: int = 5, **kw: Any) -> list: ...


def _target(target: Any) -> Any:
    if target is not None:
        return target
    from ..migration.targets import active

    return active()


class SearchMigrationDocsTool(Tool):
    name = "search_migration_docs"
    # 通用模板;目标包 texts["tools.search_migration_docs"] 原样优先(实例化时取)
    description = (
        "Search the {Library} {old}->{new} migration knowledge base: the official migration guide, {new} docs, the "
        "error-code reference, the release notes and GitHub issues about {old}/{new} behaviour differences. Use it when a "
        "failure is not covered by the quick reference, when an error message or behaviour change is unfamiliar, or "
        "before guessing an API. `query` may be a natural-language question or a pasted error/traceback (paths and "
        "values are stripped automatically). Returns the top passages with their source URL."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or error message/traceback excerpt."},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "description": "Number of passages (default 5).",
            },
            "source": {
                "type": "string",
                "enum": SOURCE_KINDS,
                "description": "Optional: restrict to one source kind.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, retriever: _Retriever, default_k: int = 5, max_chars_per_hit: int = 1500, *, target: Any = None
    ):
        self.description = _target(target).text("tools.search_migration_docs", type(self).description)
        self.retriever = retriever
        self.default_k = default_k
        self.max_chars_per_hit = max_chars_per_hit

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult.error("query must be a non-empty string")
        k = max(1, min(int(args.get("k") or self.default_k), 8))
        source = args.get("source")
        if source is not None and source not in SOURCE_KINDS:
            return ToolResult.error(f"source must be one of {SOURCE_KINDS}")
        hits = self.retriever.retrieve(query, k=k, source_kind=source)
        if not hits:
            return ToolResult(content="No matching passages.", metadata={"hits": 0, "hit_ids": ""})
        blocks = []
        for h in hits:
            meta = h.doc.metadata
            text = h.doc.text
            if len(text) > self.max_chars_per_hit:
                text = text[: self.max_chars_per_hit] + "\n… [passage truncated]"
            blocks.append(f"[{h.rank}] ({meta.get('source_kind')}) {meta.get('url', '')}\n{text}")
        return ToolResult(
            content=ctx.truncate("\n\n---\n\n".join(blocks)),
            metadata={
                "hits": len(hits),
                "hit_ids": ",".join(h.doc.id for h in hits),
                "hit_kinds": ",".join(str(h.doc.metadata.get("source_kind")) for h in hits),
            },
        )


BRIEFING_HEADER = (
    "Migration notes retrieved from the knowledge base for the initial failures "
    "(official docs; may be partially relevant — verify against the tests):"
)


def knowledge_briefing(
    retriever: _Retriever,
    report: Any,
    *,
    max_groups: int = 4,
    per_group: int = 2,
    max_chars: int = 700,
) -> tuple[str, dict[str, Any]]:
    """为初始测试里最常见的几个失败根因各检索官方文档片段,拼成注入初始消息的简报。

    只取 guide / docs / errors(issue 噪声大,留给模型主动用工具查)。返回 (文本, trace 元数据)。
    """
    groups: dict[str, list] = {}
    for f in getattr(report, "failures", []) or []:
        groups.setdefault(f.signature, []).append(f)
    top = sorted(groups.values(), key=len, reverse=True)[:max_groups]
    seen: set[str] = set()
    lines: list[str] = []
    for items in top:
        f = items[0]
        query = (f.traceback or f.message)[-3000:]
        hits = [
            h
            for h in retriever.retrieve(query, k=per_group * 4)
            if h.doc.metadata.get("source_kind") in ("guide", "docs", "errors")
        ]
        picked = [h for h in hits if h.doc.id not in seen][:per_group]
        if not picked:
            continue
        lines.append(f"\nFor failures like `{f.signature[:160]}` ({len(items)} tests):")
        for h in picked:
            seen.add(h.doc.id)
            text = h.doc.text
            if len(text) > max_chars:
                text = text[:max_chars] + " …"
            lines.append(f"  - ({h.doc.metadata.get('source_kind')}) {h.doc.metadata.get('url', '')}")
            lines.append("    " + text.replace("\n", "\n    "))
    meta = {"groups": len(top), "hits": len(seen), "hit_ids": ",".join(sorted(seen))}
    if not lines:
        return "", meta
    return BRIEFING_HEADER + "\n".join(lines), meta


# ---------------------------------------------------------------- 代码库检索


class RepoCodebase:
    """一个 run 的代码库索引会话。经 `ctx.files.tree()`(既有 Executor 协议)读取沙箱内当前源码,
    每次调用前按文件哈希增量同步,agent 的编辑立即可见;向量部分在第一次语义检索时才计算。"""

    def __init__(self, embedder_spec: str | None = None, embedder=None):
        self.embedder_spec = embedder_spec
        self._embedder = embedder
        self._index = None
        self.last_stats = None

    def _files(self, ctx: ToolContext) -> dict[str, str]:
        from ..rag.codebase.indexer import files_from_tree, load_repo_files

        try:
            return files_from_tree(ctx.files.tree())
        except Exception:  # Executor 不支持文件 RPC 时退回读取宿主目录
            return load_repo_files(ctx.repo_path)

    def index(self, ctx: ToolContext, *, vectors: bool):
        from ..rag.codebase.indexer import CodebaseIndex

        if vectors and self._embedder is None and self.embedder_spec:
            from ..rag.common.embeddings import make_embedder

            self._embedder = make_embedder(self.embedder_spec)
        files = self._files(ctx)
        if self._index is None:
            self._index = CodebaseIndex(self._embedder if vectors else None)
            self.last_stats = self._index.index(files)
        else:
            if vectors and self._index.embedder is None and self._embedder is not None:
                # 先前只建了 BM25/调用图:补上向量,块不变时只嵌入一次
                self._index.embedder = self._embedder
                self._index.update(dict(self._index.files))
            self.last_stats = self._index.sync(files)
        return self._index

    def graph(self, ctx: ToolContext):
        return self.index(ctx, vectors=False).graph


class SearchCodebaseTool(Tool):
    name = "search_codebase"
    description = (
        "Search THIS repository's source code (all .py files, split into functions / classes / module blocks by "
        "the syntax tree) with combined keyword + semantic retrieval. Use it to find where something is "
        "implemented or which class/function is responsible for a behaviour before opening files. Returns the "
        "top chunks with file path, line range and code, followed by their direct callers/callees in the repo. "
        "The index follows your edits automatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Behaviour, identifier or error text to look for."},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of chunks (default 6).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, codebase: RepoCodebase, default_k: int = 6, max_chars_per_hit: int = 1200):
        self.codebase = codebase
        self.default_k = default_k
        self.max_chars_per_hit = max_chars_per_hit

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        from ..rag.codebase.retriever import CodebaseRetriever

        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult.error("query must be a non-empty string")
        k = max(1, min(int(args.get("k") or self.default_k), 10))
        index = self.codebase.index(ctx, vectors=True)
        hits = CodebaseRetriever(index).retrieve(query, k=k, expand_graph=True)
        if not hits:
            return ToolResult(content="No matching code.", metadata={"hits": 0, "hit_ids": ""})
        blocks, related = [], []
        for h in hits:
            c = h.chunk
            where = f"{c.file}:{c.start_line}-{c.end_line}  {c.qualname} ({c.kind})"
            if h.via != "search":
                caller = h.related_to.split("::", 1)[-1]
                related.append(f"  {'called by' if h.via == 'caller' else 'calls'} [{caller}]: {where}")
                continue
            text = c.text.split("\n", 1)[1] if "\n" in c.text else c.text
            if len(text) > self.max_chars_per_hit:
                text = text[: self.max_chars_per_hit] + "\n… [chunk truncated; read_file for the rest]"
            blocks.append(f"[{h.rank}] {where}\n{text}")
        content = "\n\n---\n\n".join(blocks)
        if related:
            content += "\n\nRelated (call graph, one hop):\n" + "\n".join(related)
        stats = self.codebase.last_stats
        return ToolResult(
            content=ctx.truncate(content),
            metadata={
                "hits": len(hits),
                "hit_ids": ",".join(h.chunk.id for h in hits),
                "hit_files": ",".join(sorted({h.chunk.file for h in hits})),
                "index_seconds": round(stats.seconds, 3) if stats else 0,
                "index_changed_files": stats.changed_files if stats else 0,
            },
        )


class FindUsagesTool(Tool):
    name = "find_usages"
    description = (
        "List every place in THIS repository that uses a symbol, resolved statically through imports, aliases "
        "and re-exports; comments and strings never match. Symbols: module-level names (e.g. {example_symbol}), "
        "`<Type>.<attr>` for attribute access on values inferred to be of a library type (e.g. "
        "{example_attr_symbol}), or a repo definition by dotted module path (e.g. mypkg.models.User). Static "
        "inference can miss dynamically typed receivers; re-run after edits to see what is left."
    )
    input_schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "Symbol to look up."}},
        "required": ["symbol"],
    }

    def __init__(self, codebase: RepoCodebase, max_rows: int = 200, *, target: Any = None):
        self.description = _target(target).text("tools.find_usages", type(self).description)
        self.codebase = codebase
        self.max_rows = max_rows

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        symbol = str(args.get("symbol") or "").strip()
        if not symbol:
            return ToolResult.error("symbol must be a non-empty string")
        usages = self.codebase.graph(ctx).usages_of(symbol)
        if not usages:
            return ToolResult(content=f"No usages of {symbol} found.", metadata={"usages": 0, "files": 0})
        lines, current = [], None
        for u in usages[: self.max_rows]:
            if u.file != current:
                current = u.file
                ro = "  (read-only test file)" if ctx.is_protected(u.file) else ""
                lines.append(f"{u.file}{ro}")
            lines.append(f"  L{u.line} [{u.kind}] {u.text[:140]}")
        if len(usages) > self.max_rows:
            lines.append(f"… {len(usages) - self.max_rows} more")
        files = len({u.file for u in usages})
        head = f"{len(usages)} usage line(s) of {symbol} in {files} file(s):"
        return ToolResult(
            content=ctx.truncate(head + "\n" + "\n".join(lines)),
            metadata={"usages": len(usages), "files": files},
        )


# ---------------------------------------------------------------- 经验记忆


def target_version(ctx: ToolContext) -> str | None:
    """沙箱内当前安装的目标库版本(经既有 Executor);失败返回 None(检索时只按 target 过滤)。"""
    pkg = _target(None).version_package
    try:
        res = ctx.executor.exec(
            ["python", "-I", "-c", f"import importlib.metadata as m; print(m.version({pkg!r}))"], timeout=60
        )
    except Exception:  # noqa: BLE001
        return None
    return (res.stdout.strip() or None) if res.ok else None


def _session_version(session: Any, ctx: ToolContext) -> str | None:
    if not session.version_checked:
        session.version, session.version_checked = target_version(ctx), True
    return session.version


class SearchExperienceTool(Tool):
    name = "search_experience"
    description = (
        "Search reviewed fix experiences distilled from successful {Library} {old}->{new} migrations of OTHER "
        "repositories. Each hit gives the normalised error, when the fix applies (and when not), the minimal diff "
        "that fixed it there and the test evidence. Use it with the exact error message or traceback excerpt when "
        "a failure is unfamiliar or persists. Hits are hints from other codebases: adapt names to this code and "
        "verify with the tests. Returns nothing when there is no reliable match; entries already shown in this "
        "run are not repeated."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Error message or traceback excerpt."},
            "code": {"type": "string", "description": "Optional: the code around the failure."},
            "k": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Max entries (default 3)."},
        },
        "required": ["query"],
    }

    def __init__(
        self, retriever: Any, session: Any, *, default_k: int = 3, max_tokens: int = 1500, target: Any = None
    ):
        self.description = _target(target).text("tools.search_experience", type(self).description)
        self.retriever, self.session = retriever, session
        self.default_k, self.max_tokens = default_k, max_tokens

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        from ..rag.memory.retriever import run_search

        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult.error("query must be a non-empty string")
        k = max(1, min(int(args.get("k") or self.default_k), self.default_k))
        hits, meta = run_search(
            self.retriever,
            self.session,
            query,
            str(args.get("code") or ""),
            version=_session_version(self.session, ctx),
            k=k,
            max_tokens=self.max_tokens,
            purpose="tool",
        )
        if not hits:
            return ToolResult(
                content="No reliable matching experience (or all matches were already shown).", metadata=meta
            )
        body = "\n\n---\n\n".join(h.text for h in hits)
        return ToolResult(content=ctx.truncate(experience_header() + "\n\n" + body), metadata=meta)


EXPERIENCE_HEADER = (
    "Fix experiences from other repositories (reviewed; hints only — adapt to this code and verify with the tests; "
    "never use {compat_layer} imports or patch {library} at runtime):"
)


def experience_header() -> str:
    return _target(None).text("tools.experience_header", EXPERIENCE_HEADER)


def experience_briefing(
    retriever: Any,
    session: Any,
    ctx: ToolContext,
    report: Any,
    *,
    k: int = 3,
    max_tokens: int = 1500,
    max_groups: int = 4,
) -> tuple[str, dict[str, Any]]:
    """按初始测试里最常见的失败根因检索经验,总计最多 k 条 / max_tokens;无可靠命中时不注入任何文本。"""
    from ..rag.memory.retriever import run_search

    groups: dict[str, list] = {}
    for f in getattr(report, "failures", []) or []:
        groups.setdefault(f.signature, []).append(f)
    top = sorted(groups.values(), key=len, reverse=True)[:max_groups]
    version = _session_version(session, ctx)
    hits, metas = [], []
    for items in top:
        left_k, left_tokens = k - len(hits), max_tokens - sum(h.tokens for h in hits)
        if left_k <= 0 or left_tokens <= 0:
            break
        f = items[0]
        found, meta = run_search(
            retriever,
            session,
            (f.traceback or f.message)[-3000:],
            version=version,
            k=left_k,
            max_tokens=left_tokens,
            purpose="initial",
        )
        hits += found
        metas.append(meta)
    meta = {
        "memory_groups": len(top),
        "memory_hits": len(hits),
        "memory_hit_ids": ",".join(h.exp.id for h in hits),
        "memory_tokens": sum(h.tokens for h in hits),
        "memory_latency_ms": round(sum(m["memory_latency_ms"] for m in metas), 2),
        "memory_snapshot": metas[0]["memory_snapshot"] if metas else "",
        "memory_version": version or "",
    }
    if not hits:
        return "", meta
    return experience_header() + "\n\n" + "\n\n---\n\n".join(h.text for h in hits), meta
