"""Planner:任务开始时用代码库 RAG 的 `usages_of` 扫描废弃符号,生成按文件/模块分组、带优先级的迁移计划。

    plan = Planner(target, is_protected).plan(graph)     # graph: rag.codebase.CallGraph
    plan.render() -> str                                  # 注入初始消息的计划摘要
    plan_briefing(planner, codebase, ctx)(report)         # AgentLoop.initial_context_provider 形式

计划只依赖静态分析,不调用 LLM:
    - 可编辑文件:含任一废弃符号用法、且不受测试只读保护的文件;
    - 只读文件:含废弃用法的测试文件,单独列出(不能改,只提示源码改动会影响它们);
    - 优先级:被其他计划内文件 import 的文件排前(先改底层模型/基类,牵连错误少),再按用法数;
    - 分组:按顶层包/目录,便于降级为「按组逐组迁移」。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..migration.targets.base import MigrationTarget


@dataclass
class FilePlan:
    path: str
    usages: list[Any]  # rag.codebase.call_graph.Usage
    editable: bool
    dependents: int = 0  # 计划内 import 了本文件的文件数

    @property
    def symbols(self) -> Counter:
        return Counter(u.symbol for u in self.usages)

    @property
    def group(self) -> str:
        parts = self.path.split("/")
        if parts[0] == "src" and len(parts) > 2:
            return parts[1]
        return parts[0] if len(parts) > 1 else "."


@dataclass
class MigrationPlan:
    symbols: list[str]
    files: list[FilePlan] = field(default_factory=list)  # 可编辑,按优先级排序
    read_only: list[FilePlan] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for f in self.files:
            out.setdefault(f.group, []).append(f.path)
        return out

    def render(self, max_files: int = 40, max_lines_per_file: int = 4) -> str:
        if not self.files and not self.read_only:
            return ""
        total = sum(len(f.usages) for f in self.files)
        lines = [
            f"Migration plan from static analysis ({len(self.files)} source files, {total} deprecated-API usages; "
            "edit in this order — files imported by others first). Usages of dynamically typed receivers may be "
            "missing; `find_usages` re-scans after your edits."
        ]
        for i, f in enumerate(self.files[:max_files], 1):
            sym = ", ".join(f"{_short(s)}×{n}" for s, n in f.symbols.most_common())
            dep = f", imported by {f.dependents} planned file(s)" if f.dependents else ""
            lines.append(f"{i}. {f.path}  [{sym}]{dep}")
            for u in f.usages[:max_lines_per_file]:
                lines.append(f"     L{u.line}: {u.text[:110]}")
            if len(f.usages) > max_lines_per_file:
                lines.append(f"     … {len(f.usages) - max_lines_per_file} more")
        if len(self.files) > max_files:
            lines.append(f"… {len(self.files) - max_files} more files (use find_usages to list them)")
        if self.read_only:
            ro = ", ".join(f.path for f in self.read_only[:12])
            more = f" … +{len(self.read_only) - 12}" if len(self.read_only) > 12 else ""
            lines.append(
                f"Read-only test files that also use deprecated APIs (do not edit; make the source compatible): {ro}{more}"
            )
        return "\n".join(lines)


def _short(symbol: str) -> str:
    """计划里显示的符号短名:完整名取最后一段;目标包的文本模式符号(`pattern:<写法>`)显示写法本身。"""
    return symbol.split(":", 1)[1] if symbol.startswith("pattern:") else symbol.split(".")[-1]


class Planner:
    def __init__(self, target: MigrationTarget, is_protected: Callable[[str], bool] | None = None):
        self.target = target
        self.is_protected = is_protected or (lambda _p: False)

    def plan(self, graph) -> MigrationPlan:
        symbols = list(self.target.deprecated_symbols)
        by_file: dict[str, list] = {}
        for symbol in symbols:
            for u in graph.usages_of(symbol):
                by_file.setdefault(u.file, []).append(u)
        plan = MigrationPlan(symbols)
        editable = {p for p in by_file if not self.is_protected(p)}
        for path, usages in by_file.items():
            usages.sort(key=lambda u: (u.line, u.symbol))
            fp = FilePlan(path, usages, path in editable)
            (plan.files if fp.editable else plan.read_only).append(fp)
        dependents = Counter(
            dep for path in editable for dep in graph.imported_files(path) if dep in editable and dep != path
        )
        for fp in plan.files:
            fp.dependents = dependents.get(fp.path, 0)
        plan.files.sort(key=lambda f: (-f.dependents, -len(f.usages), f.path))
        plan.read_only.sort(key=lambda f: f.path)
        return plan


def plan_briefing(planner: Planner, codebase, ctx) -> Callable[[Any], tuple[str, dict[str, Any]]]:
    """返回 AgentLoop.initial_context_provider:同步代码库索引(不嵌入)→ 生成计划 → 文本 + trace 元数据。"""

    def provider(_report) -> tuple[str, dict[str, Any]]:
        graph = codebase.graph(ctx)
        plan = planner.plan(graph)
        meta = {
            "plan_files": len(plan.files),
            "plan_read_only": len(plan.read_only),
            "plan_usages": sum(len(f.usages) for f in plan.files),
        }
        return plan.render(), meta

    return provider
