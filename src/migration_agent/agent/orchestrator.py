"""迁移会话编排：在共享沙箱中按预算运行多次尝试，失败后恢复最佳检查点并以修订计划和新上下文继续。沙箱最终统一发布，停止由调用方负责。"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..migration.task import EndReason, MigrationTask, RunResult
from ..tools.sandbox.protocol import Sandbox
from .dynamic_plan import _role_usage

RETRY_REASONS = {
    EndReason.MAX_STEPS,
    EndReason.MAX_TOKENS,
    EndReason.MAX_COST,
    EndReason.MAX_WALL_TIME,
    EndReason.AGENT_GAVE_UP,
    EndReason.QUALITY_GATE_FAILED,
    EndReason.COMPATIBILITY_SHIM_FOUND,
}
MIN_STEPS_SECOND = 3


class HeldSandbox:
    """把真实沙箱的生命周期交给 Session:start / stop / publish 为空操作,其余委托。"""

    def __init__(self, inner: Any):
        self.inner = inner
        self.workdir = inner.workdir

    def exec(self, cmd, *, cwd=None, timeout=300, env=None):
        return self.inner.exec(cmd, cwd=cwd, timeout=timeout, env=env)

    def snapshot(self, label: str) -> str:
        return self.inner.snapshot(label)

    def restore(self, snapshot_id: str) -> None:
        self.inner.restore(snapshot_id)

    def start(self) -> None:
        self.inner.start()  # 已启动时是空操作(DockerSandbox.start 幂等)

    def stop(self) -> None:
        return None

    def publish(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class Attempt:
    index: int
    budget: dict[str, Any]
    result: RunResult
    cost_usd: float = 0.0
    restored: dict | None = None
    postmortem: dict | None = None
    postmortem_tokens: dict | None = None


@dataclass
class SessionResult:
    attempts: list[Attempt] = field(default_factory=list)
    combined: RunResult | None = None
    skipped_second: str = ""

    def trace_paths(self) -> list[Any]:
        return [a.result.trace_path for a in self.attempts if a.result.trace_path]


class Session:
    def __init__(
        self,
        task: MigrationTask,
        settings: Any,
        *,
        build: Callable[[MigrationTask, int], tuple[Any, Any]],
        executor: Any = None,
        max_attempts: int | None = None,
        first_fraction: float | None = None,
    ):
        self.task, self.settings, self.build = task, settings, build
        self.executor = executor
        self.max_attempts = max_attempts or settings.orchestrator_attempts
        self.first_fraction = first_fraction if first_fraction is not None else settings.first_attempt_budget
        self.loops: list[Any] = []

    def _total(self, loop: Any) -> dict[str, Any]:
        b = loop.budget
        return {
            "max_steps": b.max_steps,
            "max_total_tokens": b.max_total_tokens,
            "max_wall_seconds": b.max_wall_seconds,
            "max_cost_usd": b.max_cost_usd,
        }

    def run(self) -> SessionResult:
        out = SessionResult()
        held = HeldSandbox(self.executor) if isinstance(self.executor, Sandbox) else None
        t0 = time.monotonic()
        total: dict[str, Any] | None = None
        carry: dict[str, int] = {}
        prev_hook = None
        notes = ""
        used = {"steps": 0, "tokens": 0, "cost": 0.0}
        try:
            for i in range(1, self.max_attempts + 1):
                task = self.task.model_copy(update={"run_id": f"{self.task.run_id or 'run'}-a{i}"})
                loop, ctx = self.build(task, i)
                if held is not None:
                    ctx.executor = held
                total = total or self._total(loop)
                last = i == self.max_attempts
                budget = self._attempt_budget(total, i, used, time.monotonic() - t0)
                if budget is None:
                    out.skipped_second = "not enough budget left for another attempt"
                    break
                b = loop.budget
                b.max_steps, b.max_total_tokens = budget["max_steps"], budget["max_total_tokens"]
                b.max_wall_seconds, b.max_cost_usd = budget["max_wall_seconds"], budget["max_cost_usd"]
                hook = loop.hooks.get("dynamic_plan") if loop.hooks is not None else None
                if hook is not None:
                    hook.carry = carry
                    hook.restore_best_on_failure = not last
                    hook.reserve_postmortem = 0 if last else 1  # 给本次尝试之后的 postmortem 留 1 次重规划
                    if prev_hook is not None:
                        hook.initial_plan = prev_hook.state
                gate = loop.hooks.get("stop_gate") if loop.hooks is not None else None
                if gate is not None and prev_hook is not None and self._gate_baseline is not None:
                    gate.baseline = self._gate_baseline
                if notes:
                    loop.extra_initial_context = "\n\n".join(
                        x for x in (loop.extra_initial_context, notes) if x
                    )
                loop.tracer.log("attempt", {"attempt": i, "of": self.max_attempts, "budget": budget})
                self.loops.append(loop)
                result = loop.run(task, ctx)
                cost = self._cost(loop, result)
                att = Attempt(i, budget, result, cost, getattr(hook, "restored", None))
                out.attempts.append(att)
                used["steps"] += result.steps
                used["tokens"] += self._tokens(loop, result)
                used["cost"] += cost
                if gate is not None and i == 1:
                    self._gate_baseline = dict(gate.initial)
                loop.tracer.log(
                    "attempt",
                    {
                        "attempt": i,
                        "end": str(result.reason),
                        "success": result.success,
                        "restored": att.restored,
                    },
                )
                if result.success or result.reason not in RETRY_REASONS or last:
                    break
                # 复盘(沙箱仍可用:HeldSandbox 未停止)
                if hook is not None:
                    before = _role_usage(hook.router, "planner") if hook.router is not None else {}
                    res = hook.replan("postmortem")
                    att.postmortem = res.trace() if res is not None else None
                    if hook.router is not None:  # 复盘调用发生在 attempt 1 的 RunResult 之后,单独计入会话总账
                        after = _role_usage(hook.router, "planner")
                        delta = {
                            k: after[k] - before.get(k, 0) for k in ("in", "cache_read", "cache_write", "out")
                        }
                        att.postmortem_tokens = delta
                        used["tokens"] += sum(delta.values())
                        p = loop.budget.pricing
                        if p is not None:
                            used["cost"] += p.cost(
                                delta["in"], delta["cache_read"], delta["cache_write"], delta["out"]
                            )
                    carry = hook.carry
                    carry["replan_calls"] = hook.replanner.calls if hook.replanner else 0
                    if hook.replanner is not None and hook.replanner.trigger_quota is not None:
                        carry["replan_used"] = dict(hook.replanner.used)
                    carry["diagnosis_used"] = int(hook._diagnosis_used)
                    notes = self._postmortem_notes(att, hook, res)
                    loop.tracer.log("attempt", {"attempt": i, "postmortem": att.postmortem})
                prev_hook = hook
            out.combined = self._combine(out.attempts)
            if held is not None and out.combined.reason != EndReason.VALIDATION_FAILED:
                self.executor.publish()
            return out
        finally:
            for lp in self.loops:
                with contextlib.suppress(Exception):
                    lp.tracer.close()

    _gate_baseline: dict | None = None

    def _attempt_budget(
        self, total: dict[str, Any], i: int, used: dict[str, Any], elapsed: float
    ) -> dict[str, Any] | None:
        if i == 1 and self.max_attempts > 1:
            f = self.first_fraction
            return {
                "max_steps": max(1, math.floor(total["max_steps"] * f)),
                "max_total_tokens": int(total["max_total_tokens"] * f),
                "max_wall_seconds": int(total["max_wall_seconds"] * f),
                "max_cost_usd": None
                if total["max_cost_usd"] is None
                else round(total["max_cost_usd"] * f, 6),
            }
        if i == 1:
            return dict(total)
        steps = total["max_steps"] - used["steps"]
        if steps < MIN_STEPS_SECOND:
            return None
        cost = None if total["max_cost_usd"] is None else round(total["max_cost_usd"] - used["cost"], 6)
        if cost is not None and cost <= 0:
            return None
        return {
            "max_steps": steps,
            "max_total_tokens": max(1, total["max_total_tokens"] - used["tokens"]),
            "max_wall_seconds": max(1, int(total["max_wall_seconds"] - elapsed)),
            "max_cost_usd": cost,
        }

    @staticmethod
    def _tokens(loop: Any, result: RunResult) -> int:
        st = getattr(loop, "state", None)
        if st is not None:
            return int(st.budget_tokens(loop.budget))
        return result.tokens_in + result.tokens_cached + result.tokens_out

    @staticmethod
    def _cost(loop: Any, result: RunResult) -> float:
        st, p = getattr(loop, "state", None), loop.budget.pricing
        return float(st.cost(p)) if st is not None and p is not None else 0.0

    @staticmethod
    def _postmortem_notes(att: Attempt, hook: Any, res: Any) -> str:
        r = att.result
        lines = [
            f"[orchestrator] This is a fresh attempt. The previous attempt ended with {r.reason} after {r.steps} "
            f"steps ({r.tests_passed}/{r.tests_total} tests passing at the end)."
        ]
        if att.restored and "passed" in att.restored:
            lines.append(
                f"The repository was restored to the best checkpoint of that attempt ({att.restored['passed']}/"
                f"{att.restored['total']} passing); the current test result below reflects it."
            )
        failed = []
        for sig, diffs in list(hook._attempts.items())[:5]:
            failed.append(f"  - `{sig[:140]}`: {len(diffs)} edit(s) did not fix it; last: {diffs[-1][:200]}")
        if failed:
            lines.append("Approaches that did not work in the previous attempt:")
            lines += failed
        if res is not None and res.ok:
            lines.append(res.notes())
        if hook.state is not None:
            lines.append("Revised plan (tracked automatically):\n" + hook.state.render())
        return "\n".join(lines)

    @staticmethod
    def _combine(attempts: list[Attempt]) -> RunResult:
        last = attempts[-1].result
        rs = [a.result for a in attempts]
        pm = [a.postmortem_tokens or {} for a in attempts]
        files = sorted({f for r in rs for f in r.files_changed})
        return last.model_copy(
            update={
                "run_id": last.run_id.rsplit("-a", 1)[0],
                "steps": sum(r.steps for r in rs),
                "tokens_in": sum(r.tokens_in for r in rs) + sum(int(t.get("in", 0)) for t in pm),
                "tokens_cached": sum(r.tokens_cached for r in rs)
                + sum(int(t.get("cache_read", 0) + t.get("cache_write", 0)) for t in pm),
                "tokens_out": sum(r.tokens_out for r in rs) + sum(int(t.get("out", 0)) for t in pm),
                "duration_s": sum(r.duration_s for r in rs),
                "files_changed": files,
                "violations": sum(r.violations for r in rs),
                "rollbacks": sum(r.rollbacks for r in rs),
                "compactions": sum(r.compactions for r in rs),
                "re_read_after_compaction": sum(r.re_read_after_compaction for r in rs),
                "message": (f"[{len(rs)} attempt(s)] " + last.message)[:2000],
            }
        )
