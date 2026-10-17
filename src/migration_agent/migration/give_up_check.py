"""放弃前检查剩余失败的报错帧。若经过可编辑源码，要求 Agent 检查或解释无法修复的原因；返回次数由配置限制。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..agent.hooks import Continue, GiveUpEvent, Hook, RunStart
from .strategy import source_frames

MESSAGE = (
    "[give-up check] You are about to stop while {failing} test(s) still fail. Before concluding that the remaining "
    "failures cannot be fixed from source, note that these failures pass through editable (non-test) repository "
    "code:\n{where}\nInspect these frames. If the fix belongs in one of them, make it and run the tests again; if "
    "not, state for each listed file why changing it cannot help, then stop. (Check {returns} of {max_returns}.)"
)


def editable_failures(report: Any, is_protected, repo_files: set[str]) -> list[tuple[str, int, list[str]]]:
    """[(签名, 失败数, 可编辑帧)],只含至少有一个可编辑帧的签名组,按失败数降序。"""
    groups: dict[str, list[Any]] = {}
    for f in getattr(report, "failures", None) or []:
        groups.setdefault(f.signature, []).append(f)
    out = []
    for sig, fs in groups.items():
        frames: list[str] = []
        for f in fs[:5]:
            for p in source_frames(f.traceback or "", is_protected, repo_files):
                if p not in frames:
                    frames.append(p)
        if frames:
            out.append((sig, len(fs), frames))
    return sorted(out, key=lambda x: -x[1])


# 不进入的目录:隐藏目录(.git、.venv、.migration_agent…)、虚拟环境与第三方包、缓存
_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env", "build", "dist"}


def _repo_py_files(root: Path) -> set[str]:
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
        rel = Path(dirpath).relative_to(root)
        out.update((rel / f).as_posix() for f in filenames if f.endswith(".py"))
    return out


class GiveUpCheck(Hook):
    name = "give_up_check"

    def __init__(self, max_returns: int = 1, tracer: Any = None):
        self.max_returns = max_returns
        self.returns = 0
        self.tracer = tracer
        self.history: list[dict[str, Any]] = []

    def on_run_start(self, ev: RunStart) -> None:
        self.returns = 0
        self.history = []

    def on_give_up_final(self, ev: GiveUpEvent) -> Continue | None:
        report = ev.report if ev.report is not None else getattr(ev.ctx, "last_report", None)
        if report is None or self.returns >= self.max_returns:
            return None
        found = editable_failures(report, ev.ctx.is_protected, _repo_py_files(Path(ev.ctx.repo_path)))
        entry = {"returns": self.returns, "groups": [[s[:200], n, fr[:6]] for s, n, fr in found[:8]]}
        self.history.append(entry)
        if self.tracer is not None:
            self.tracer.log("give_up_check", {"action": "continue" if found else "accept", **entry})
        if not found:
            return None
        self.returns += 1
        where = "\n".join(
            f"  - {n} test(s): {sig[:200]}\n    editable frames: {', '.join(fr[:6])}"
            for sig, n, fr in found[:8]
        )
        failing = len(getattr(report, "failures", None) or [])
        return Continue(
            MESSAGE.format(failing=failing, where=where, returns=self.returns, max_returns=self.max_returns),
            source=self.name,
        )
