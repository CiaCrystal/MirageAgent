"""ToolRegistry:注册工具、导出 Anthropic tool schema 列表、按名称分发调用。

接口:`registry.register(tool)`、`registry.schemas()`、`registry.get(name)`、
`registry.dispatch(name, args, ctx) -> ToolResult`(未知工具 / 异常统一转为 is_error 结果,不让循环崩溃)。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import Tool, ToolContext, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool '{name}'. Available: {', '.join(self._tools)}")
        try:
            return tool.run(args or {}, ctx)
        except Exception as e:  # noqa: BLE001 — 工具异常回填给 LLM 而不是中断循环
            return ToolResult.error(f"{type(e).__name__}: {e}", exception=repr(e))

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
