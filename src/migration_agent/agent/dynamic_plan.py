"""动态规划的装配与钩子:PlanState + update_plan/view_plan + 运行时用法 + 失败分类 + LLM 重规划 + 诊断子 agent。

只在 `dynamic_plan=True`、full 档、hooks 写法下装配(agent/assemble.py);全部能力以钩子 / 工具接入 HookBus,
不包装 LLM 或工具。需要调 LLM 的组件(replanner、LLM 分类、诊断子 agent)经 ModelRouter 的 planner 角色调用
(与 executor 共用同一个密钥和客户端);这些调用的 token 计入同一 run 的 AgentState 预算与 router 的 planner 账目。

DynamicPlanHook
    briefing(report)     initial_context_provider:静态计划(Planner)+ 运行时弃用用法 → PlanState;
                         对初始失败分类(规则优先,判不出时 LLM ≤ classify_llm_max_calls 次);
                         静态计划为空或覆盖不到初始失败 → 触发 start 重规划。不返回文本(计划块经 before_llm 注入)。
    on_test_report       全量报告 → PlanState.observe(签名消失 → done、牵连错误、锁定 → abandoned);
                         打带测试分数的检查点(owner="plan"),供 orchestrator 选最佳检查点。
    before_llm           预算到 50% / 75% 而通过数没超过上一个刻度时的最好成绩 → stalled 重规划;
                         计划有变化时把计划块追加到最后一条 user 消息末尾(重规划后或距上次完整注入 ≥ 5 步时注入完整计划,
                         否则只注入一行变更摘要),连同重规划结论。
    after_tool           统计编辑落在计划文件内的比例。
    on_give_up           模型不再调工具但测试未通过 → stop 重规划,结论附在催促消息之后。
    supervisor 阶梯       SupervisedLLM.replan_handler:repeat_error 重规划(换检索词 → 重规划 → 回滚 → 放弃)。
    on_run_end           写最终计划指标与 router 按角色账目;orchestrator 第一次尝试失败时恢复到最佳检查点。
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Any

from ..migration.plan_state import PlanState
from ..tools.plan_tool import UpdatePlanTool, ViewPlanTool
from .hooks import AfterTool, BeforeLLM, ContextBlock, GiveUpEvent, Hook, RunEnd, RunStart, TestReportEvent

EDIT_TOOLS = ("edit_file", "write_file", "insert_line", "multi_edit")
PLAN_HINT = (
    "A migration plan is tracked for you: item statuses follow the test results automatically and the current plan "
    "is appended to your messages when it changes. Use `view_plan` to see it and `update_plan` to add, split or "
    "annotate items; you cannot mark an item done while its linked failures still fail."
)
RETRY_FULL_EVERY = 5
MIN_STEPS_BETWEEN_REPLANS = 3


@dataclass
class PlanStats:
    replans: dict[str, int] = field(default_factory=dict)
    replans_ok: int = 0
    replan_tokens: dict[str, float] = field(default_factory=dict)
    diagnosis_runs: int = 0
    diagnosis_tokens: int = 0
    classify_llm_calls: int = 0
    edits: int = 0
    edits_in_plan: int = 0
    injections: int = 0
    injected_chars: int = 0


class DynamicPlanHook(Hook):
    name = "dynamic_plan"

    def __init__(
        self,
        settings: Any,
        ctx: Any,
        target: Any,
        *,
        planner: Any = None,
        codebase: Any = None,
        test_cmd: list[str] | None = None,
        tracer: Any = None,
        budget: Any = None,
        supervisor: Any = None,
        classifier: Any = None,
    ):
        self.settings, self.ctx, self.target = settings, ctx, target
        self.planner, self.codebase = planner, codebase
        self.test_cmd = list(test_cmd or [])
        self.tracer, self.budget, self.supervisor = tracer, budget, supervisor
        self.classifier = (
            classifier  # (report, repo_files, changed, llm) -> {签名: 类别};None 用 failure_taxonomy
        )
        self.router = None
        self.replanner = None
        self.diagnoser = None
        # orchestrator 设置:第二次尝试的起始计划、跨尝试共用的调用计数、第一次尝试失败时恢复最佳检查点
        self.initial_plan: PlanState | None = None
        self.restore_best_on_failure = False
        # orchestrator 还有下一次尝试时为 1:尝试内的重规划最多用 replan_max_calls − 1 次
        self.reserve_postmortem = 0
        self.carry: dict[str, int] = {}
        self.checkpoints = None  # AgentLoop.run 在 on_run_start 之前经 bind_checkpoints 绑定,_reset 不清空
        self._reset()

    # ------------------------------------------------------------------ wiring
    def attach(self, router: Any, registry: Any) -> None:
        """add_harness 之后调用:拿到 router(planner 角色)、注册计划工具、构造 replanner 与诊断子 agent。"""
        from ..migration.replanner import Replanner

        self.router = router
        registry.register(ViewPlanTool(self))
        registry.register(UpdatePlanTool(self))
        self.registry = registry
        self.replanner = Replanner(
            lambda prompt, schema: router.complete_json("planner", prompt, schema),
            self.target,
            max_calls=self.settings.replan_max_calls,
            trigger_quota=self.settings.replan_trigger_quota,
            file_ok=self.file_ok,
            is_protected=self.ctx.is_protected,
            usage=lambda: _role_usage(router, "planner"),
        )
        if self.settings.diagnosis_subagent:
            from .subagents import DiagnosisAgent

            self.diagnoser = DiagnosisAgent(
                router.client("planner"),
                registry,
                target=self.target,
                max_steps=self.settings.diagnosis_max_steps,
                max_tokens=self.settings.diagnosis_max_tokens,
                account=self._charge_resp,
                tracer=self.tracer,
            )

    def _reset(self) -> None:
        self.state: PlanState | None = None
        self.agent_state: Any = None
        self.categories: dict[str, str] = {}
        self.stats = PlanStats()
        self._tree: dict | None = None
        self._repo_files: set[str] = set()
        self._report_seq = 0
        self._reports: dict[str, Any] = {}  # 检查点 id → 报告
        self._attempts: dict[str, list[str]] = {}  # 签名 → 未奏效尝试的 diff 摘要
        self._initial_passed = 0
        self._best_passed = 0
        self._mark_best: float = -1
        self._marks_done: set[float] = set()
        self._injected_version = 0
        self._full_step = -RETRY_FULL_EVERY
        self._pending_notes: list[str] = []
        self._force_full = True
        self._last_report: Any = None
        self._stop_replans = 0
        self._diagnosis_used = False
        self._last_replan_step = -MIN_STEPS_BETWEEN_REPLANS
        self._t0 = time.monotonic()
        self.restored: dict | None = None

    # ------------------------------------------------------------------ helpers
    def file_ok(self, path: str) -> str:
        path = str(path).replace("\\", "/").removeprefix("./")
        if self.ctx.is_protected(path):
            return "read-only test file"
        if self._repo_files and path not in self._repo_files:
            return "file does not exist"
        locks = getattr(self.supervisor, "locks", {}) or {}
        if path in locks:
            return "locked by the strategy controller"
        return ""

    def record(self, source: str, **payload: Any) -> None:
        if self.tracer is not None and self.state is not None:
            self.tracer.log(
                "plan",
                {"source": source, "version": self.state.version, "counts": self.state.counts(), **payload},
            )

    def _charge(self, tokens_in=0, cache_read=0, cache_write=0, tokens_out=0) -> None:
        """附加调用计入同一 run 的 AgentState(预算与 RunResult 的 token)。"""
        st = self.agent_state
        if st is None:
            return
        st.tokens_in += int(tokens_in)
        st.cache_read += int(cache_read)
        st.cache_write += int(cache_write)
        st.tokens_cached += int(cache_read) + int(cache_write)
        st.tokens_out += int(tokens_out)

    def _charge_resp(self, resp: Any) -> None:
        self._charge(resp.tokens_in, resp.cache_read_tokens, resp.cache_write_tokens, resp.tokens_out)

    def _llm_json(self) -> Any:
        """LLM 分类用的 complete_json(planner 角色);每 run 上限 classify_llm_max_calls 次。"""
        if self.router is None:
            return None
        used = self.carry.get("classify_llm", 0) + self.stats.classify_llm_calls
        if used >= self.settings.classify_llm_max_calls:
            return None

        def call(prompt: str, schema: dict) -> dict:
            self.stats.classify_llm_calls += 1
            before = _role_usage(self.router, "planner")
            try:
                return self.router.complete_json("planner", prompt, schema)
            finally:
                after = _role_usage(self.router, "planner")
                self._charge(
                    *(after[k] - before.get(k, 0) for k in ("in", "cache_read", "cache_write", "out"))
                )

        return call

    def _defines(self) -> Any:
        import base64

        from ..migration.failure_taxonomy import defines_from_texts

        tree = self._tree or {}
        return defines_from_texts(
            {
                p: base64.b64decode(e.get("data", "") or b"").decode("utf-8", "replace")
                for p, e in tree.items()
                if p.endswith(".py")
            }
        )

    def _classify(self, report: Any, changed: set[str]) -> None:
        new = {f.signature for f in getattr(report, "failures", None) or []} - set(self.categories)
        if not new:
            return
        try:
            if self.classifier is not None:
                cats = self.classifier(report, self._repo_files, changed, self._llm_json())
            else:
                from ..migration.failure_taxonomy import classify_report

                cats = classify_report(
                    report,
                    target=self.target,
                    repo_files=self._repo_files,
                    changed_files=changed,
                    is_protected=self.ctx.is_protected,
                    llm=self._llm_json(),
                    signatures=new,
                    defines=self._defines(),
                )
        except Exception as e:  # noqa: BLE001 — 分类失败只影响重规划模板
            cats = {}
            if self.tracer is not None:
                self.tracer.log("classify", {"error": f"{type(e).__name__}: {e}"[:300]})
        for sig in new:
            self.categories[sig] = cats.get(sig, "unknown")
        if self.tracer is not None:
            self.tracer.log("classify", {"categories": {s[:160]: self.categories[s] for s in sorted(new)}})
        if self.state is not None:
            for it in self.state.items:
                if not it.category:
                    cs = {self.categories.get(s, "") for s in it.signatures} - {""}
                    it.category = sorted(cs)[0] if len(cs) == 1 else ("mixed" if cs else "")

    # ------------------------------------------------------------------ lifecycle
    def on_run_start(self, ev: RunStart) -> None:
        self._reset()
        self.agent_state = ev.state
        for k in ("replan_calls", "diagnosis_used"):
            self.carry.setdefault(k, 0)
        if self.replanner is not None:
            self.replanner.calls = self.carry["replan_calls"]
            if self.replanner.trigger_quota is not None:  # 各触发的已用次数跨尝试沿用
                self.replanner.used = dict(self.carry.get("replan_used", {}))
            self.replanner.reserved = self.reserve_postmortem
            self.replanner.history = []
        self._diagnosis_used = bool(self.carry["diagnosis_used"])

    def bind_checkpoints(self, service: Any) -> None:
        self.checkpoints = service

    def briefing(self, report: Any) -> tuple[str, dict[str, Any]]:
        """initial_context_provider:构建 PlanState(不返回文本)。"""
        meta: dict[str, Any] = {}
        tree = self.ctx.files.tree()
        self._tree, self._repo_files = tree, set(tree)
        self._initial_passed = self._best_passed = int(report.passed or 0)
        self._mark_best = self._best_passed
        self._last_report = report
        if self.initial_plan is not None:  # orchestrator 第二次尝试:沿用复盘后的计划
            self.state = self.initial_plan
            self.state.is_protected = self.ctx.is_protected
            self.state.failing = {f.signature for f in report.failures}
            meta["plan_source"] = "postmortem"
        else:
            plan = None
            if self.planner is not None and self.codebase is not None:
                try:
                    plan = self.planner.plan(self.codebase.graph(self.ctx))
                except Exception as e:  # noqa: BLE001
                    meta["static_plan_error"] = f"{type(e).__name__}: {e}"[:200]
            usages: list[Any] = []
            if self.settings.plan_runtime_usages:
                from ..migration.runtime_usages import probe

                pr = probe(self.ctx, self.target, self.test_cmd, timeout=self.settings.pytest_timeout_seconds)
                usages = pr.usages
                meta.update(pr.meta())
                after = self.ctx.files.tree()
                self._tree, self._repo_files = after, set(after)
            self.state = PlanState.from_static(
                plan, report, usages, is_protected=self.ctx.is_protected, repo_files=self._repo_files
            )
            meta["plan_source"] = "static"
            meta["plan_static_files"] = len(getattr(plan, "files", []) or [])
            meta["plan_uncovered_initial"] = len(self.state.uncovered_initial)
        self._classify(report, set())
        meta["plan_items"] = len(self.state.items)
        self.record("initial", items=len(self.state.items), uncovered=len(self.state.uncovered_initial))
        static_items = [i for i in self.state.items if i.origin in ("static", "runtime_usage")]
        if self.initial_plan is None and (not static_items or self.state.uncovered_initial):
            self.replan("start")
        return "", meta

    def on_test_report(self, ev: TestReportEvent) -> None:
        if ev.subset or self.state is None:
            return None
        rep = ev.report
        tree = self.ctx.files.tree()
        changed = {p for p in set(tree) | set(self._tree or {}) if tree.get(p) != (self._tree or {}).get(p)}
        before_fail = set(self.state.failing)
        self._note_attempts(changed, tree, before_fail, rep)
        self._tree, self._repo_files = tree, set(tree)
        self._report_seq += 1
        self._classify(rep, changed)
        locks = set(getattr(self.supervisor, "locks", {}) or {})
        changes = self.state.observe(rep, changed, locks, self._repo_files)
        if changes:
            self.record("tests", changes=changes[:20])
        self._best_passed = max(self._best_passed, int(rep.passed or 0))
        self._last_report = rep
        if self.checkpoints is not None and not rep.parse_error:
            try:
                cp = self.checkpoints.create(f"plan report {self._report_seq}", rep, owner="plan")
                self._reports[cp.id] = rep
            except Exception as e:  # noqa: BLE001
                self.record("checkpoint_failed", error=f"{type(e).__name__}: {e}"[:200])
        return None

    def _note_attempts(self, changed: set[str], tree: dict, before_fail: set[str], rep: Any) -> None:
        """修改后仍失败的签名:记下这次修改的 diff 摘要,作为 replanner 的「历次失败尝试」证据。"""
        if not changed or self._tree is None:
            return
        still = before_fail & {f.signature for f in rep.failures}
        if not still:
            return
        import base64

        parts = []
        for path in sorted(changed)[:4]:
            a = base64.b64decode((self._tree.get(path) or {}).get("data", "") or b"").decode(
                "utf-8", "replace"
            )
            b = base64.b64decode((tree.get(path) or {}).get("data", "") or b"").decode("utf-8", "replace")
            diff = "".join(
                ln
                for ln in difflib.unified_diff(a.splitlines(True), b.splitlines(True), n=0)
                if ln[:1] in "+-"
            )
            parts.append(f"{path}: {diff[:300]}")
        summary = " | ".join(parts)
        for sig in still:
            self._attempts.setdefault(sig, []).append(summary)

    def before_llm(self, ev: BeforeLLM) -> list[ContextBlock] | None:
        if self.state is None:
            return None
        self._check_stalled(ev.state)
        blocks: list[ContextBlock] = []
        if self._pending_notes:
            blocks.append(ContextBlock(self.name, "\n".join(self._pending_notes)))
            self._pending_notes = []
        if self.state.version != self._injected_version:
            full = self._force_full or ev.step - self._full_step >= RETRY_FULL_EVERY
            if full:
                text = self.state.render()
                self._full_step = ev.step
                self._force_full = False
            else:
                last = self.state.history[-1] if self.state.history else {}
                ch = last.get("changes") or []
                bits = [
                    f"{c['id']} {c['from']}→{c['to']}"
                    if "to" in c
                    else f"{c['id']} +{c.get('added') or 'cascade'}"
                    for c in ch[:8]
                ]
                c = self.state.counts()
                text = (
                    f"[plan v{self.state.version}] {c['done']}/{len(self.state.items)} done"
                    + (f"; changes: {', '.join(bits)}" if bits else "")
                    + " (view_plan for details)"
                )
            blocks.append(ContextBlock(self.name, text))
            self._injected_version = self.state.version
            self.stats.injections += 1
        self.stats.injected_chars += sum(len(b.text) for b in blocks)
        return blocks or None

    def _check_stalled(self, st: Any) -> None:
        b = self.budget
        if b is None or self._last_report is None or self._last_report.all_passed:
            return
        # 刻度只按步数与墙钟:token / 成本里含 planner 角色自己的花费,按它们计会出现「重规划花费 → 触发 stalled →
        # 再重规划」的循环。距上次重规划不足 3 步时先不判，刻度保留到之后。
        if st.step - self._last_replan_step < MIN_STEPS_BETWEEN_REPLANS:
            return
        fracs = [st.step / b.max_steps if b.max_steps else 0]
        if b.max_wall_seconds:
            fracs.append(st.elapsed_s / b.max_wall_seconds)
        frac = max(fracs)
        for mark in (0.5, 0.75):
            if frac >= mark and mark not in self._marks_done:
                self._marks_done.add(mark)
                stalled = self._best_passed <= self._mark_best
                if self.tracer is not None:
                    self.tracer.log(
                        "plan",
                        {
                            "source": "budget_mark",
                            "mark": mark,
                            "best": self._best_passed,
                            "prev": self._mark_best,
                            "stalled": stalled,
                        },
                    )
                self._mark_best = self._best_passed
                if stalled:
                    self.replan("stalled")
                break

    def after_tool(self, ev: AfterTool) -> None:
        if ev.name in EDIT_TOOLS and not getattr(ev.result, "is_error", False) and self.state is not None:
            path = str((ev.args or {}).get("path") or "").replace("\\", "/").removeprefix("./")
            if path:
                self.stats.edits += 1
                self.stats.edits_in_plan += path in self.state.plan_files()
        return None

    def on_give_up(self, ev: GiveUpEvent) -> list[ContextBlock] | None:
        if self.state is None or self._stop_replans >= 1:
            return None
        self._stop_replans += 1
        self._last_report = ev.report
        res = self.replan("stop")
        if not res or not res.ok:
            return None
        text = "\n".join(self._pending_notes + [self.state.render()])
        self._pending_notes = []
        self._injected_version = self.state.version
        self._full_step = ev.state.step
        return [ContextBlock(self.name, text)]

    def on_repeat_error(self, escalation: Any, report: Any) -> tuple[str, dict[str, Any]]:
        """SupervisedLLM 阶梯的 replan 级别。"""
        self._last_report = report
        res = self.replan("repeat_error", focus=escalation.signature)
        meta = {"replanned": bool(res and res.ok), "replan_error": (res.error or res.skipped) if res else ""}
        note = (
            f"[strategy controller] `{escalation.signature[:200]}` appeared {escalation.count} times; the planner "
            "re-diagnosed it — follow the revised plan below. One more unsuccessful attempt rolls back."
            if res and res.ok
            else ""
        )
        return note, meta

    def on_run_end(self, ev: RunEnd) -> None:
        r = ev.result
        restored = None
        if self.restore_best_on_failure and not r.success and self.checkpoints is not None:
            best = self.checkpoints.best()
            if best is not None and (best.passed or 0) > (r.tests_passed or 0):
                try:
                    self.checkpoints.restore(best, owner="plan")
                    restored = {"checkpoint": best.id, "passed": best.passed, "total": best.total}
                    self._last_report = self._reports.get(best.id, self._last_report)
                except Exception as e:  # noqa: BLE001
                    restored = {"error": f"{type(e).__name__}: {e}"[:200]}
        self.restored = restored
        if self.state is not None:
            self.record("final", **self.metrics(), restored=restored)
        if self.tracer is not None and self.router is not None:
            self.tracer.log("router", {role: vars(u) for role, u in self.router.usage().items()})
        self.carry["replan_calls"] = self.replanner.calls if self.replanner else 0
        if self.replanner is not None and self.replanner.trigger_quota is not None:
            self.carry["replan_used"] = dict(self.replanner.used)
        self.carry["diagnosis_used"] = int(self._diagnosis_used)
        self.carry["classify_llm"] = self.carry.get("classify_llm", 0) + self.stats.classify_llm_calls

    def metrics(self) -> dict[str, Any]:
        s = self.stats
        return {
            "completion_rate": round(self.state.completion_rate(), 4) if self.state else 0.0,
            "items": len(self.state.items) if self.state else 0,
            "edits": s.edits,
            "edits_in_plan": s.edits_in_plan,
            "replans": dict(s.replans),
            "replans_ok": s.replans_ok,
            "replan_tokens": dict(s.replan_tokens),
            "diagnosis_runs": s.diagnosis_runs,
            "diagnosis_tokens": s.diagnosis_tokens,
            "classify_llm_calls": s.classify_llm_calls,
            "injections": s.injections,
            "injected_chars": s.injected_chars,
        }

    # ------------------------------------------------------------------ replanning
    def replan(self, trigger: str, focus: str | None = None) -> Any:
        if self.replanner is None or self.state is None or self._last_report is None:
            return None
        self.stats.replans[trigger] = self.stats.replans.get(trigger, 0) + 1
        if self.agent_state is not None:
            self._last_replan_step = self.agent_state.step
        if self.replanner.remaining_for(trigger) == 0:
            res = self.replanner.replan(trigger, self.state, self._last_report)  # 记录 skipped,不调用
            self._trace_replan(res, focus, None)
            return res
        diag = None
        if (
            self.diagnoser is not None
            and not self._diagnosis_used
            and trigger in ("start", "repeat_error", "postmortem")
            and (trigger != "start" or self._start_needs_diagnosis())
        ):
            self._diagnosis_used = True
            d = self.diagnoser.diagnose(self.ctx, self._failure_pack(focus))
            self.stats.diagnosis_runs += 1
            self.stats.diagnosis_tokens += d.tokens
            diag = d.summary() or None
        shown = sorted(getattr(self.supervisor, "_shown_doc_ids", None) or set())
        res = self.replanner.replan(
            trigger,
            self.state,
            self._last_report,
            categories=self.categories,
            attempts=self._attempts,
            shown_ids=shown,
            diagnosis=diag,
            budget_left=self._budget_left(),
            repo_files=self._repo_files,
        )
        t = res.tokens or {}
        self._charge(t.get("in", 0), t.get("cache_read", 0), t.get("cache_write", 0), t.get("out", 0))
        for k, v in t.items():
            self.stats.replan_tokens[k] = round(self.stats.replan_tokens.get(k, 0) + v, 6)
        if res.ok:
            self.stats.replans_ok += 1
            touched = self.state.apply_replan(res.accepted_ops)
            for it in (self.state.get(i) for i in touched):
                if it is not None and not it.category:
                    cs = {self.categories.get(s, "") for s in it.signatures} - {""}
                    it.category = sorted(cs)[0] if len(cs) == 1 else ""
            if res.upgrade_dependencies:
                self.state.apply_replan(
                    [
                        {
                            "op": "add",
                            "goal": "Upgrade third-party dependencies: "
                            + ", ".join(res.upgrade_dependencies),
                            "approach": "pip install -U " + " ".join(res.upgrade_dependencies),
                            "category": "env_dependency",
                        }
                    ]
                )
            if res.test_impossible:
                self._block_impossible(res.test_impossible["tests"], res.test_impossible.get("reason", ""))
            self._pending_notes.append(res.notes())
            self._force_full = True
        self._trace_replan(res, focus, diag)
        return res

    def _start_needs_diagnosis(self) -> bool:
        """开局只在静态计划覆盖不到、且规则分类判不出的失败上调用诊断子 agent。"""
        return any(self.categories.get(s, "unknown") == "unknown" for s in self.state.uncovered_initial)

    def _block_impossible(self, tests: list[str], reason: str) -> None:
        sigs = {f.signature for f in self._last_report.failures if f.test_id in set(tests)}
        for it in self.state.items:
            if it.signatures and set(it.signatures) <= sigs and it.status not in ("done", "abandoned"):
                it.status, it.category = "blocked", "test_impossible"
                it.notes.append(f"verified impossible without editing tests: {reason[:200]}")
        self.state.version += 1

    def _trace_replan(self, res: Any, focus: str | None, diag: Any) -> None:
        if self.tracer is not None:
            self.tracer.log(
                "replan",
                {
                    **res.trace(),
                    "focus": (focus or "")[:200],
                    "diagnosis_used": bool(diag),
                    "plan_version": self.state.version if self.state else 0,
                },
            )

    def _failure_pack(self, focus: str | None) -> str:
        rep = self._last_report
        groups: dict[str, list[Any]] = {}
        for f in rep.failures:
            groups.setdefault(f.signature, []).append(f)
        order = sorted(groups, key=lambda s: (s != focus, -len(groups[s])))[:3]
        lines = [
            f"Diagnose why these tests fail after migrating to {self.target.describe()}. {rep.summary_line()}"
        ]
        for s in order:
            f = groups[s][0]
            lines += [
                f"- {len(groups[s])} test(s), category={self.categories.get(s, 'unknown')}: {s[:240]}",
                f"  e.g. {f.test_id}",
                "  " + (f.traceback or "")[-1200:].replace("\n", "\n  "),
            ]
        for s in order:
            for a in self._attempts.get(s, [])[-2:]:
                lines.append(f"Earlier edit that did not fix `{s[:80]}`: {a[:300]}")
        return "\n".join(lines)

    def _budget_left(self) -> dict[str, Any]:
        b, st = self.budget, self.agent_state
        if b is None or st is None:
            return {}
        out: dict[str, Any] = {"steps": b.max_steps - st.step}
        if b.max_cost_usd and b.pricing is not None:
            out["cost_usd"] = round(b.max_cost_usd - st.cost(b.pricing), 4)
        return out


def _role_usage(router: Any, role: str) -> dict[str, float]:
    u = router.usage().get(role)
    if u is None:
        return {"in": 0, "cache_read": 0, "cache_write": 0, "out": 0, "cost": 0.0, "calls": 0}
    return {
        "in": u.tokens_in,
        "cache_read": u.cache_read,
        "cache_write": u.cache_write,
        "out": u.tokens_out,
        "cost": u.cost,
        "calls": u.calls,
    }
