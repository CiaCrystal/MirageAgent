"""文件工具:`read_file / list_files / edit_file / write_file`。

- 所有路径相对仓库根,越界拒绝。
- 测试目录与测试文件(见 ToolContext.protected_globs)只读,写入直接报错——这是「禁改测试」硬规则的第一道闸门。
- 写操作把文件记入 `ctx.changed_files`,供循环决定是否自动跑测试、供 RunResult 汇报。
- `edit_file` 采用 old_string → new_string 精确替换(要求 old_string 在文件中唯一),
  而不是 unified diff:LLM 生成 diff 的行号/上下文经常对不上,替换式编辑失败率低得多。
  unified-diff 形式的 `apply_patch` 留到需要批量改动时再加。
"""

from __future__ import annotations

import difflib
import fnmatch
from typing import Any

from .base import Tool, ToolContext, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the repository. Returns the content with 1-based line numbers. "
        "Optionally restrict to a line range."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to repository root"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = ctx.canonical(args["path"])
        if rel not in ctx.files.list():
            return ToolResult.error(f"not a file: {args['path']}")
        lines = ctx.files.read(rel).splitlines()
        start = max(1, int(args.get("start_line") or 1))
        end = min(len(lines), int(args.get("end_line") or len(lines)))
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        header = f"# {args['path']} (lines {start}-{end} of {len(lines)})\n"
        return ToolResult(content=ctx.truncate(header + body), metadata={"lines": len(lines)})


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files under a directory (recursively), relative to repository root. "
        "Skips .git, caches and virtualenvs. Use glob to filter, e.g. '*.py'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "description": "fnmatch pattern on file name, e.g. '*.py'"},
        },
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel_root = ctx.canonical(args.get("path") or ".")
        prefix = "" if rel_root == "." else rel_root + "/"
        pattern = args.get("glob")
        out: list[str] = []
        for rel in sorted(ctx.files.list()):
            if rel.startswith(prefix) and (not pattern or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)):
                tag = "  [read-only: test]" if ctx.is_protected(rel) else ""
                out.append(rel + tag)
        return ToolResult(content=ctx.truncate("\n".join(out) or "(empty)"), metadata={"count": len(out)})


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact, unique occurrence of old_string with new_string in a file. "
        "old_string must match exactly (including whitespace) and appear exactly once; "
        "include enough surrounding lines to make it unique. Returns numbered affected lines on success "
        "and the closest fragment on a mismatch. Test files are read-only."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["path", "old_string", "new_string"],
    }
    mutates_repo = True

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = ctx.canonical(args["path"])
        if ctx.is_protected(rel):
            return ToolResult.error(f"'{rel}' is a test file and is read-only. Fix the source code instead.")
        ctx.resolve(rel)
        if rel not in ctx.files.list():
            return ToolResult.error(f"not a file: {rel}")
        text = ctx.files.read(rel)
        old, new = args["old_string"], args["new_string"]
        if not old:
            return ToolResult.error("old_string must not be empty; use insert_line instead")
        n = text.count(old)
        if n == 0:
            lines = text.splitlines()
            width = max(1, len(old.splitlines()))
            candidates = [
                (difflib.SequenceMatcher(None, old, "\n".join(lines[i : i + width])).ratio(), i)
                for i in range(len(lines))
            ]
            best = max(candidates, default=(0, 0))[1] + 1
            return ToolResult.error(
                "old_string not found in file (check exact whitespace/indentation). "
                "Closest fragment:\n" + _excerpt(text, best, width)
            )
        if n > 1:
            return ToolResult.error(f"old_string occurs {n} times; include more context to make it unique")
        start = text[: text.index(old)].count("\n") + 1
        edited = text.replace(old, new, 1)
        ctx.files.write(rel, edited)
        ctx.changed_files.add(rel)
        return ToolResult(
            content=ctx.truncate(f"edited {rel}\n" + _excerpt(edited, start, len(new.splitlines()))),
            metadata={"path": rel, "delta": len(new) - len(old)},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a file with the given content. Prefer edit_file for small changes. "
        "Test files are read-only."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    mutates_repo = True

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = ctx.canonical(args["path"])
        if ctx.is_protected(rel):
            return ToolResult.error(f"'{rel}' is a test file and is read-only. Fix the source code instead.")
        ctx.resolve(rel)
        existed = rel in ctx.files.list()
        ctx.files.write(rel, args["content"])
        ctx.changed_files.add(rel)
        return ToolResult(
            content=f"{'overwrote' if existed else 'created'} {rel} ({len(args['content'])} chars)",
            metadata={"path": rel, "existed": existed},
        )


def _excerpt(text: str, start: int, count: int) -> str:
    lines = text.splitlines()
    lo, hi = max(1, start - 3), min(len(lines), start + max(1, count) + 2)
    return "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(lo, hi + 1))


class InsertLineTool(Tool):
    """Harness insert semantics: insert after a 1-based line; zero means prepend."""

    name = "insert_line"
    description = "Insert new_string after insert_line (1-based, 0 prepends). Returns numbered affected lines. Tests are read-only."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "insert_line": {"type": "integer", "minimum": 0},
            "new_string": {"type": "string"},
        },
        "required": ["path", "insert_line", "new_string"],
    }
    mutates_repo = True

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = ctx.canonical(args["path"])
        ctx.resolve(rel)
        if ctx.is_protected(rel):
            return ToolResult.error(f"'{rel}' is a test file and is read-only. Fix the source code instead.")
        text = ctx.files.read(rel)
        lines = text.splitlines(keepends=True)
        at = args["insert_line"]
        if not isinstance(at, int) or isinstance(at, bool) or not 0 <= at <= len(lines):
            return ToolResult.error(f"insert_line must be between 0 and {len(lines)}")
        before, after = "".join(lines[:at]), "".join(lines[at:])
        if before and not before.endswith("\n"):
            before += "\n"
        inserted = args["new_string"]
        if inserted and not inserted.endswith("\n"):
            inserted += "\n"
        edited = before + inserted + after
        ctx.files.write(rel, edited)
        ctx.changed_files.add(rel)
        return ToolResult(
            content=ctx.truncate(
                f"inserted into {rel}\n" + _excerpt(edited, at + 1, len(inserted.splitlines()))
            ),
            metadata={"path": rel, "insert_line": at},
        )


def fs_tools() -> list[Tool]:
    return [ReadFileTool(), ListFilesTool(), EditFileTool(), WriteFileTool(), InsertLineTool()]
