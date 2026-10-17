"""ACI 工具(harness 配置):grep / glob / multi_edit / read_output、编辑前新鲜度检查、超长输出落盘。
参考 SWE-agent ACI 与 Claude Code 的 Grep / Glob / MultiEdit、「编辑前必须读」。

工具(实现 tools/base.Tool,只在 settings.aci_tools=True 时注册):
    grep         args: pattern, path?, glob?, context?, max_results;输出「path:line: text」按文件分组,结果数有上限
    glob         args: pattern, max_results;按路径排序(RepositoryFiles 不提供修改时间)
    multi_edit   args: path, edits=[{old_string, new_string}];依次在内存中应用,全部唯一匹配才一起写入,回显受影响行
    read_output  args: id, offset?, limit?;分页读取落盘的完整输出

AciHook(挂在 HookBus 上):
    - 新鲜度:记录每个文件最近一次被 read_file 读取(或被本 run 的编辑工具写入)时的内容;编辑已有文件前若未读过或
      文件已变化(回滚、shell 修改、策略控制恢复检查点),拒绝编辑并提示先重读;
    - 超长输出落盘:ToolContext.spill 把超过 tool_output_max_chars 的完整输出写到沙箱 `.migration_agent/outputs/<n>.txt`
      (该目录不计入文件树、最终 diff 与 Validator 检查),截断提示里给出 read_output 的 id。
未实现:只读工具同一步并行执行(循环仍按顺序执行)。
"""

from __future__ import annotations

import hashlib
import re
from fnmatch import fnmatch
from typing import Any

from ..agent.hooks import AfterTool, BeforeTool, Deny, Hook, RunStart
from .base import Tool, ToolContext, ToolResult
from .fs import _excerpt

OUTPUT_DIR = ".migration_agent/outputs"
EDIT_TOOLS = ("edit_file", "write_file", "insert_line", "multi_edit")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a Python regular expression. Returns 'path:line: text' grouped by file, "
        "capped at max_results matches. Narrow with path (directory or file) and glob (e.g. '*.py')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "directory or file, default '.'"},
            "glob": {"type": "string", "description": "file name glob, e.g. '*.py'"},
            "context": {"type": "integer", "description": "lines of context around each match (0-3)"},
            "max_results": {"type": "integer", "description": "default 50, max 200"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            return ToolResult.error(f"invalid regex: {e}")
        base = ctx.canonical(args.get("path") or ".")
        base = "" if base == "." else base
        glob = args.get("glob") or "*"
        context = max(0, min(int(args.get("context") or 0), 3))
        limit = max(1, min(int(args.get("max_results") or 50), 200))
        hits, files, more = [], 0, False
        for rel in sorted(ctx.files.list()):
            if base and not (rel == base or rel.startswith(base + "/")):
                continue
            if not fnmatch(rel.rsplit("/", 1)[-1], glob):
                continue
            try:
                lines = ctx.files.read(rel).splitlines()
            except (RuntimeError, UnicodeDecodeError):
                continue
            found = [i for i, line in enumerate(lines) if rx.search(line)]
            if not found:
                continue
            files += 1
            hits.append(f"== {rel}")
            for i in found:
                if len([h for h in hits if not h.startswith("==")]) >= limit:
                    more = True
                    break
                for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                    mark = ":" if j == i else "-"
                    hits.append(f"{rel}:{j + 1}{mark} {lines[j][:300]}")
            if more:
                break
        if not hits:
            return ToolResult(content="no matches")
        tail = "\n[more matches omitted; narrow path/glob/pattern or raise max_results]" if more else ""
        return ToolResult(content=ctx.truncate("\n".join(hits) + tail), metadata={"files": files})


class GlobTool(Tool):
    name = "glob"
    description = "List repository files matching a glob such as 'src/**/*.py' (sorted by path, capped)."
    input_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pat = args["pattern"]
        limit = max(1, min(int(args.get("max_results") or 200), 1000))
        found = [p for p in sorted(ctx.files.list()) if fnmatch(p, pat) or fnmatch(p, pat.replace("**/", ""))]
        body = "\n".join(found[:limit]) or "no files"
        if len(found) > limit:
            body += f"\n[{len(found) - limit} more files omitted]"
        return ToolResult(content=body, metadata={"matches": len(found)})


class MultiEditTool(Tool):
    name = "multi_edit"
    description = (
        "Apply several exact string replacements to one file atomically. Each old_string must match exactly once "
        "(after the previous edits); if any edit fails nothing is written. Test files are read-only."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"old_string": {"type": "string"}, "new_string": {"type": "string"}},
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    mutates_repo = True

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = ctx.canonical(args["path"])
        if ctx.is_protected(rel):
            return ToolResult.error(f"'{rel}' is a test file and is read-only. Fix the source code instead.")
        if rel not in ctx.files.list():
            return ToolResult.error(f"not a file: {rel}")
        text = ctx.files.read(rel)
        starts = []
        for n, e in enumerate(args.get("edits") or [], 1):
            old, new = e.get("old_string", ""), e.get("new_string", "")
            count = text.count(old) if old else 0
            if count != 1:
                return ToolResult.error(
                    f"edit {n}: old_string matched {count} times (must be exactly once); nothing was written"
                )
            pos = text.index(old)
            starts.append(text.count("\n", 0, pos) + 1)
            text = text[:pos] + new + text[pos + len(old) :]
        ctx.files.write(rel, text)
        echo = "\n".join(_excerpt(text, max(1, s - 2), 8) for s in starts[:5])
        return ToolResult(content=f"applied {len(starts)} edit(s) to {rel}\n{echo}")


class ReadOutputTool(Tool):
    name = "read_output"
    description = "Read a page of a long tool output that was saved because it exceeded the output limit."
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "offset": {"type": "integer", "description": "character offset, default 0"},
            "limit": {"type": "integer", "description": "characters, default 8000"},
        },
        "required": ["id"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        oid = re.sub(r"[^0-9a-f]", "", str(args.get("id", "")))
        try:
            text = ctx.files.read(f"{OUTPUT_DIR}/{oid}.txt")
        except RuntimeError:
            return ToolResult.error(f"no saved output {args.get('id')!r}")
        off = max(0, int(args.get("offset") or 0))
        lim = max(1, min(int(args.get("limit") or 8000), ctx.tool_output_max_chars))
        page = text[off : off + lim]
        rest = len(text) - off - len(page)
        return ToolResult(
            content=page + (f"\n[{rest} more chars; offset={off + len(page)}]" if rest > 0 else "")
        )


def aci_tools() -> list[Tool]:
    return [GrepTool(), GlobTool(), MultiEditTool(), ReadOutputTool()]


class AciHook(Hook):
    name = "aci"

    def __init__(self, max_chars: int = 12_000):
        self.max_chars = max_chars
        self.known: dict[str, str] = {}  # 路径 → 最近一次读取/写入时的内容摘要
        self.stale_refusals = 0
        self.spilled = 0

    def on_run_start(self, ev: RunStart) -> None:
        self.known.clear()
        ctx = ev.ctx

        def spill(text: str) -> str:
            oid = _digest(text)[:12]
            ctx.files.write(f"{OUTPUT_DIR}/{oid}.txt", text)
            self.spilled += 1
            return f"; full output saved, read it with read_output(id='{oid}')"

        ctx.spill = spill

    def _current(self, ctx: ToolContext, rel: str) -> str | None:
        try:
            return _digest(ctx.files.read(rel)) if rel in ctx.files.list() else None
        except RuntimeError:
            return None

    def before_tool(self, ev: BeforeTool) -> Deny | None:
        if ev.name not in EDIT_TOOLS:
            return None
        try:
            rel = ev.ctx.canonical(str(ev.args.get("path") or ""))
        except (PermissionError, ValueError):
            return None
        cur = self._current(ev.ctx, rel)
        if cur is None:  # 新文件(write_file 创建)不需要先读
            return None
        if rel not in self.known:
            why = "you have not read it in this run"
        elif self.known[rel] != cur:
            why = "it changed since you last read it (rollback, shell command or another edit)"
        else:
            return None
        self.stale_refusals += 1
        return Deny(
            ToolResult.error(
                f"refusing to edit {rel}: {why}. Read the current file with read_file first.", stale=rel
            ),
            rule="aci-freshness",
        )

    def after_tool(self, ev: AfterTool) -> None:
        if ev.result.is_error:
            return None
        if ev.name == "read_file" or ev.name in EDIT_TOOLS:
            try:
                rel = ev.ctx.canonical(str(ev.args.get("path") or ""))
            except (PermissionError, ValueError):
                return None
            cur = self._current(ev.ctx, rel)
            if cur is not None:
                self.known[rel] = cur
        return None
