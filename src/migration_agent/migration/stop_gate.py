"""Stop 质量闸门(harness 配置):测试全绿、即将以成功结束时,检查最终产出相对初始源码是否新增
目标库补丁或兼容层导入(规则来自 migration/quality.py 与目标包,与 Evaluator 的严格口径相同)。

发现问题时不直接判失败,而是把具体位置作为 `Continue(reason)` 退回给 agent 继续修,最多 `max_returns` 次;
次数用尽后仍有问题 → 改判 QUALITY_GATE_FAILED(严格口径失败)。参考 Claude Code 的 Stop 钩子。
"""

from __future__ import annotations

import base64
from typing import Any

from ..agent.hooks import Continue, Hook, RunStart, StopEvent
from .quality import compat_imports, patch_sites
from .task import EndReason

# 通用模板(目标包 texts 里有同名键时原样使用);{returns} {max_returns} {where} 由闸门填充
FAILED = (
    "Tests pass, but after {returns} returns the output still patches {library} or uses the "
    "compatibility layer:\n{where}"
)
RETURN = (
    "[quality gate] All tests pass, but the migration is not acceptable yet. These changes patch the "
    "{library} library at runtime or import the {compat_layer} compatibility layer instead of migrating the "
    "code:\n{where}\nRemove them and migrate the calling code to the {new_api} API, then run the tests "
    "again. (Return {returns} of {max_returns}; after that the run is judged failed.)"
)


def _target():
    from .targets import active

    return active()


def _texts(tree: dict[str, dict], is_protected) -> dict[str, str]:
    out = {}
    for rel, entry in tree.items():
        if rel.endswith(".py") and not is_protected(rel):
            out[rel] = base64.b64decode(entry["data"]).decode("utf-8", "replace")
    return out


def quality_issues(initial: dict[str, str], final: dict[str, str]) -> dict[str, list[str]]:
    """{规则: [位置]}:相对初始源码新增的库补丁与兼容层导入。"""
    issues: dict[str, list[str]] = {"library_patch": [], "compat_layer": []}
    for rel, new in sorted(final.items()):
        old = initial.get(rel, "")
        if new == old:
            continue
        before = {t for _, t in patch_sites(old)}
        issues["library_patch"] += [f"{rel}:{line}: {t}" for line, t in patch_sites(new) if t not in before]
        if compat_imports(new) > compat_imports(old):
            issues["compat_layer"].append(
                f"{rel}: " + _target().text("stop_gate.compat_location", "{compat_layer} import")
            )
    return {k: v for k, v in issues.items() if v}


class StopGate(Hook):
    name = "stop_gate"

    def __init__(self, max_returns: int = 2):
        self.max_returns = max_returns
        self.returns = 0
        self.initial: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []
        # orchestrator:第二次尝试沿用第一次尝试开始时的源码作为对照基线(否则第一次尝试引入的补丁会被当成「原有」)
        self.baseline: dict[str, str] | None = None

    def on_run_start(self, ev: RunStart) -> None:
        self.returns = 0
        self.history = []
        self.initial = (
            dict(self.baseline)
            if self.baseline is not None
            else _texts(ev.ctx.files.tree(), ev.ctx.is_protected)
        )

    def on_stop(self, ev: StopEvent) -> Continue | None:
        issues = quality_issues(self.initial, _texts(ev.ctx.files.tree(), ev.ctx.is_protected))
        self.history.append({"returns": self.returns, "issues": issues})
        if not issues:
            return None
        where = "\n".join(f"  - [{k}] {loc}" for k, locs in issues.items() for loc in locs[:10])
        if self.returns >= self.max_returns:
            ev.override = (
                False,
                EndReason.QUALITY_GATE_FAILED,
                _target().text("stop_gate.failed", FAILED).format(returns=self.returns, where=where),
            )
            return None
        self.returns += 1
        return Continue(
            _target()
            .text("stop_gate.return", RETURN)
            .format(returns=self.returns, max_returns=self.max_returns, where=where)
        )
