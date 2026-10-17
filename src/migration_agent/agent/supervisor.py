"""SupervisedLLM:经 `LLMClient` 协议接入的策略控制层。AgentLoop 一行不改。

    llm = SupervisedLLM(inner_llm, ctx, budget, tracer, knowledge=kb, codebase=cb)
    AgentLoop(llm=llm, ...)

每次 `complete()` 被循环调用(即每一步开始、LLM 调用之前):
  1. 读 `ctx.last_report`(run_tests 每次执行都会写入);是新报告时交给 RepeatErrorTracker 计数,并为该状态打检查点
     (RollbackManager,与循环自己的逐工具事务回滚相互独立)。
  2. 处理升级动作(同一 ErrorSignature 第 3 次出现起依次):
       requery  —— 换检索词:用归一化签名(而非初始预检索用的原始 traceback)重查知识库(含 issue,排除已展示过的块),
                   并在代码库索引里查与报错相关的函数/类,附在回填消息里;
       rollback —— 回滚:若此前有通过数更高的检查点,恢复到它;否则恢复到该签名首次出现时的状态(仅当不损失通过数),
                   都不满足时不回滚、只提示;
       abandon  —— 放弃该文件:签名对应的源文件(traceback 帧;取不到时不锁文件,只放弃签名)被锁定,编辑工具拒绝修改,
                   该签名不再计数;若剩余失败全部被放弃,直接返回不调用工具的合成回复(0 token),循环按既有规则结束。
  3. 预算控制:步数/token/时间任一用到 50% / 75% / 90% 时提示剩余预算与按影响用例数排序的失败根因;
     连续 N 步没有任何仓库改动时提示停止探索、动手修改。
  4. 所有提示追加到即将发送的最后一条 user 消息末尾(原地修改,后续轮次可见,不破坏缓存前缀),并写 trace。

预算上限本身仍由 AgentLoop 的 Budget 判定,这里不改变步数/token/时间参数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..migration.rollback import RollbackManager
from ..migration.strategy import LEVELS, Escalation, RepeatErrorTracker
from ..tools.base import ToolContext
from ..trace.tracer import Tracer
from .hooks import AfterLLM, BeforeLLM, ContextBlock, Hook, ShortCircuit
from .llm import LLMClient, LLMResponse
from .state import Budget


@dataclass
class SupervisorConfig:
    repeat_threshold: int = 3
    budget_marks: tuple[float, ...] = (0.5, 0.75, 0.9)
    exploration_nudge_steps: int = 8
    requery_docs: int = 3
    requery_code: int = 3
    max_chars: int = 700
    levels: tuple[str, ...] = LEVELS  # dynamic_plan:LEVELS_REPLAN(在换检索词与回滚之间插入重规划)


@dataclass
class _Checkpoint:
    report_index: int
    sha: str
    passed: int
    report: Any
    tree: dict


@dataclass
class SupervisorStats:
    requeries: int = 0
    rollbacks: int = 0
    rollbacks_skipped: int = 0
    abandoned_signatures: int = 0
    abandoned_files: list[str] = field(default_factory=list)
    budget_notices: int = 0
    exploration_nudges: int = 0
    synthetic_stops: int = 0
    lock_bypasses: int = 0


class SupervisedLLM:
    def __init__(
        self,
        inner: LLMClient,
        ctx: ToolContext,
        budget: Budget,
        tracer: Tracer,
        *,
        knowledge=None,
        codebase=None,
        config: SupervisorConfig | None = None,
        locks: dict[str, str] | None = None,
        rollback=None,
    ):
        self.inner = inner
        self.ctx = ctx
        self.budget = budget
        self.tracer = tracer
        self.knowledge = knowledge
        self.codebase = codebase
        self.config = config or SupervisorConfig()
        self.tracker = RepeatErrorTracker(self.config.repeat_threshold, ctx.is_protected, self.config.levels)
        # 升级到 replan 级别时调用 replan_handler(escalation, report) -> (提示, trace 元数据);由 DynamicPlanHook 设置
        self.replan_handler = None
        self.stats = SupervisorStats()
        self.locks: dict[str, str] = {} if locks is None else locks  # 与 FileLockTool 共享
        # 可注入共享的检查点(CheckpointService 的 ServiceRollback 适配);为空时自建 RollbackManager(包装器写法)
        self._rollback = rollback
        self._owns_rollback = rollback is None
        self._reset_run()

    @property
    def model(self) -> str:
        return self.inner.model

    def _reset_run(self) -> None:
        self._calls = 0
        self._t0: float | None = None
        self._tokens = 0
        self._last_report: Any = None
        self._report_tree: dict | None = None
        self._step_tree: dict | None = None
        self._idle_steps = 0
        self._marks_done: set[float] = set()
        self._checkpoints: list[_Checkpoint] = []
        self._shown_doc_ids: set[str] = set()

    def locked(self) -> dict[str, str]:
        return self.locks

    # ------------------------------------------------------------------ LLMClient
    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        notes, synthetic = self.prepare()
        if synthetic is not None:
            return synthetic
        if notes:
            _append_text(messages, "\n\n".join(notes))
        resp = self.inner.complete(system, messages, tools)
        self.account(resp)
        return resp

    def prepare(self) -> tuple[list[str], LLMResponse | None]:
        """每步 LLM 调用之前:观察新测试报告、执行升级动作、生成预算提示。返回 (提示, 合成停止回复或 None)。
        包装器写法(complete)与钩子写法(SupervisorHook.before_llm)共用,保证两者逐字一致。"""
        self._calls += 1
        if self._t0 is None:
            self._t0 = time.monotonic()
        notes: list[str] = []
        tree = self.ctx.files.tree()
        changed_step = _diff(self._step_tree, tree) if self._step_tree is not None else set()
        self._step_tree = tree
        if changed_step:
            self._idle_steps = 0
            bypass = sorted(changed_step & set(self.locks))
            if bypass:
                self.stats.lock_bypasses += len(bypass)
                self.tracer.log("strategy", {"action": "lock_bypass", "files": bypass})
        elif self._calls > 1:
            self._idle_steps += 1

        report = self.ctx.last_report
        if report is not None and report is not self._last_report:
            notes += self._observe(report, tree)
            report = self.ctx.last_report  # 回滚会换成检查点对应的报告

        if report is not None and not report.all_passed and self.tracker.abandoned():
            remaining = self.tracker.remaining(report)
            if not remaining and getattr(report, "failures", None):
                self.stats.synthetic_stops += 1
                text = (
                    "Stopping: every remaining failing root cause was abandoned by the strategy controller after "
                    "repeated failed attempts: " + "; ".join(s[:160] for s in self.tracker.abandoned())
                )
                self.tracer.log("strategy", {"action": "stop", "abandoned": self.tracker.abandoned()})
                return notes, LLMResponse(
                    content=[{"type": "text", "text": text}], stop_reason="end_turn", model=self.inner.model
                )

        notes += self._budget_notes(report)
        return notes, None

    def account(self, resp: LLMResponse) -> None:
        self._tokens += resp.tokens_in + resp.tokens_out + resp.cache_read_tokens + resp.cache_write_tokens

    # ------------------------------------------------------------------ reports
    def _observe(self, report: Any, tree: dict) -> list[str]:
        changed = _diff(self._report_tree, tree) if self._report_tree is not None else set()
        actions = self.tracker.observe(report, changed, set(tree))
        self._last_report, self._report_tree = report, tree
        try:
            sha = self._rollbacks().checkpoint(f"strategy report {self.tracker.reports_seen}")
            self._checkpoints.append(
                _Checkpoint(self.tracker.reports_seen, sha, int(report.passed or 0), report, tree)
            )
        except Exception as e:  # noqa: BLE001 — 检查点失败只影响回滚级别,不中断 run
            self.tracer.log("strategy", {"action": "checkpoint_failed", "error": f"{type(e).__name__}: {e}"})
        notes = []
        rolled = False
        for a in actions:
            handler = {
                "requery": self._requery,
                "replan": self._replan,
                "rollback": self._rollback_to,
                "abandon": self._abandon,
            }[a.level]
            if a.level == "rollback" and rolled:  # 同一份报告只回滚一次;其余签名由这次回滚一并覆盖
                self.tracer.log(
                    "strategy",
                    {"action": "rollback", "signature": a.signature[:300], "count": a.count, "covered": True},
                )
                continue
            try:
                note, meta = handler(a, report)
                rolled = rolled or bool(meta.get("rolled_back"))
            except Exception as e:  # noqa: BLE001 — 策略动作失败不中断 run,如实记录
                note, meta = "", {"error": f"{type(e).__name__}: {e}"[:300]}
            self.tracer.log(
                "strategy",
                {
                    "action": a.level,
                    "signature": a.signature[:300],
                    "count": a.count,
                    "files": a.files,
                    **meta,
                },
            )
            if note:
                notes.append(note)
        return notes

    def _rollbacks(self):
        if self._rollback is None:
            self._rollback = RollbackManager(self.ctx)
        return self._rollback

    def _requery(self, a: Escalation, report: Any) -> tuple[str, dict[str, Any]]:
        self.stats.requeries += 1
        ex = a.example
        exc, _, body = a.signature.partition(": ")
        query = f"{_query_prefix()} {body or a.signature}"
        if getattr(ex, "message", ""):
            query += "\n" + ex.message[:300]
        lines = [
            f"[strategy controller] The failure `{a.signature[:200]}` has now appeared {a.count} times "
            f"({a.count - 1} attempts did not fix it; {self.tracker.states[a.signature].tests} test(s) affected). "
            "The current approach is not working: re-read the full traceback and the code it points to, and change "
            "approach instead of repeating a similar edit."
        ]
        meta: dict[str, Any] = {"query": query[:300]}
        if self.knowledge is not None:
            try:
                hits = [
                    h for h in self.knowledge.retrieve(query, k=12) if h.doc.id not in self._shown_doc_ids
                ]
            except Exception as e:  # noqa: BLE001
                hits, meta["knowledge_error"] = [], str(e)[:200]
            hits = hits[: self.config.requery_docs]
            if hits:
                lines.append("Reference passages for this error (searched with the normalised error text):")
                for h in hits:
                    self._shown_doc_ids.add(h.doc.id)
                    text = h.doc.text
                    if len(text) > self.config.max_chars:
                        text = text[: self.config.max_chars] + " …"
                    lines.append(f"  - ({h.doc.metadata.get('source_kind')}) {h.doc.metadata.get('url', '')}")
                    lines.append("    " + text.replace("\n", "\n    "))
            meta["hit_ids"] = ",".join(h.doc.id for h in hits)
        if self.codebase is not None:
            try:
                from ..rag.codebase.retriever import CodebaseRetriever

                index = self.codebase.index(self.ctx, vectors=True)
                code_query = " ".join(x for x in (body, getattr(ex, "test_id", ""), " ".join(a.files)) if x)
                code_hits = CodebaseRetriever(index).retrieve(code_query, k=self.config.requery_code)
            except Exception as e:  # noqa: BLE001
                code_hits, meta["codebase_error"] = [], str(e)[:200]
            if code_hits:
                lines.append("Code in this repository most related to the error:")
                for h in code_hits:
                    c = h.chunk
                    lines.append(f"  - {c.file}:{c.start_line}-{c.end_line}  {c.qualname} ({c.kind})")
            meta["code_hits"] = ",".join(h.chunk.id for h in code_hits)
        return "\n".join(lines), meta

    def _replan(self, a: Escalation, report: Any) -> tuple[str, dict[str, Any]]:
        if self.replan_handler is None:
            return "", {"skipped": "no replanner"}
        return self.replan_handler(a, report)

    def _rollback_to(self, a: Escalation, report: Any) -> tuple[str, dict[str, Any]]:
        current = int(report.passed or 0)
        cps = self._checkpoints[:-1]  # 最后一个是当前状态
        best = max(cps, key=lambda c: (c.passed, c.report_index), default=None)
        start = next((c for c in cps if c.report_index == a.first_seen_report), None)
        target, why = None, ""
        if best is not None and best.passed > current:
            target, why = best, f"the best earlier state ({best.passed} passed vs {current} now)"
        elif start is not None and start.passed >= current and _diff(start.tree, self._report_tree or {}):
            target, why = start, "the state before your attempts at this error (no tests lost)"
        if target is None:
            self.stats.rollbacks_skipped += 1
            note = (
                f"[strategy controller] `{a.signature[:200]}` appeared {a.count} times. A rollback would lose "
                "passing tests, so your edits are kept — but the next unsuccessful attempt abandons this error and "
                + (f"locks {a.files[0]}" if a.files else "stops tracking it")
                + ". Try a clearly different fix."
            )
            return note, {"rolled_back": False}
        reverted = sorted(_diff(target.tree, self._report_tree or {}))
        self._rollbacks().rollback(target.sha)
        self.stats.rollbacks += 1
        self.ctx.last_report = target.report
        self._last_report, self._report_tree, self._step_tree = target.report, target.tree, target.tree
        note = (
            f"[strategy controller] `{a.signature[:200]}` appeared {a.count} times. Rolled the repository back to "
            f"{why}; reverted files: {', '.join(reverted[:12]) or 'none'}. Re-read those files before editing — "
            f"their content changed. Test result at the restored state: {target.report.summary_line()}. "
            "Take a different approach to this error; one more unsuccessful attempt abandons it."
        )
        return note, {"rolled_back": True, "to_report": target.report_index, "reverted": reverted}

    def _abandon(self, a: Escalation, report: Any) -> tuple[str, dict[str, Any]]:
        self.stats.abandoned_signatures += 1
        # 只锁 traceback 帧里定位到的源文件;定位不到时(如报错发生在已安装的依赖里)只放弃签名,不锁文件,
        # 避免把「尝试期间碰巧改过」的无关文件锁死
        files = [f for f in a.files[:1] if not self.ctx.is_protected(f)]
        for f in files:
            self.locks.setdefault(f, f"repeated failure: {a.signature[:120]}")
            if f not in self.stats.abandoned_files:
                self.stats.abandoned_files.append(f)
        remaining = self.tracker.remaining(report)
        note = (
            f"[strategy controller] Abandoning `{a.signature[:200]}` after {a.count} occurrences"
            + (f"; {', '.join(files)} is now locked (edits will be refused)" if files else "")
            + ". "
            + (
                f"Move on to the other {len(remaining)} failing root cause(s)."
                if remaining
                else "No other failing root causes remain; stop and summarise."
            )
        )
        return note, {"locked": files, "remaining": len(remaining)}

    # ------------------------------------------------------------------ budget
    def _budget_notes(self, report: Any) -> list[str]:
        notes = []
        b = self.budget
        used_steps = self._calls - 1  # 本次调用之前已完成的步数
        frac = max(
            used_steps / b.max_steps if b.max_steps else 0,
            self._tokens / b.max_total_tokens if b.max_total_tokens else 0,
            (time.monotonic() - (self._t0 or time.monotonic())) / b.max_wall_seconds
            if b.max_wall_seconds
            else 0,
        )
        crossed = [m for m in self.config.budget_marks if frac >= m and m not in self._marks_done]
        if crossed and report is not None and not report.all_passed:
            self._marks_done.update(crossed)
            self.stats.budget_notices += 1
            groups = self.tracker.groups(report)
            top = sorted(
                ((s, len(v)) for s, v in groups.items() if s in self.tracker.remaining(report)),
                key=lambda x: -x[1],
            )[:5]
            lines = [
                f"[budget] {int(max(crossed) * 100)}% of the run budget is used: step {used_steps}/{b.max_steps}, "
                f"{self._tokens:,}/{b.max_total_tokens:,} tokens. {report.summary_line()}. "
                "Spend the remaining steps on the root causes that affect the most tests:"
            ]
            lines += [f"  - {n} test(s): {s[:160]}" for s, n in top]
            notes.append("\n".join(lines))
            self.tracer.log("budget", {"mark": max(crossed), "steps": used_steps, "tokens": self._tokens})
        n = self.config.exploration_nudge_steps
        if (
            n
            and self._idle_steps >= n
            and self._idle_steps % n == 0
            and report is not None
            and not report.all_passed
        ):
            self.stats.exploration_nudges += 1
            groups = self.tracker.groups(report)
            top = max(groups.items(), key=lambda kv: len(kv[1]), default=None)
            target = ""
            if top:
                files = self.tracker.states.get(top[0])
                where = f" (files: {', '.join(files.files[:3])})" if files and files.files else ""
                target = f" Start with the largest failing root cause: `{top[0][:160]}`{where}."
            notes.append(
                f"[budget] No code was changed in the last {self._idle_steps} steps. Stop exploring and make an "
                f"edit now.{target}"
            )
            self.tracer.log("budget", {"exploration_nudge": self._idle_steps})
        return notes

    def close(self) -> None:
        if self._rollback is not None and self._owns_rollback:
            self._rollback.close()
            self._rollback = None


def _diff(before: dict | None, after: dict) -> set[str]:
    before = before or {}
    return {p for p in set(before) | set(after) if before.get(p) != after.get(p)}


def _append_text(messages: list[dict[str, Any]], text: str) -> None:
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": text})
        return
    content = messages[-1]["content"]
    if isinstance(content, str):
        messages[-1]["content"] = content + "\n\n" + text
    else:
        content.append({"type": "text", "text": text})


class SupervisorHook(Hook):
    """钩子写法:before_llm = SupervisedLLM.prepare(提示作为一个上下文块;合成停止 → ShortCircuit),
    after_llm = account(预算 token 计数)。策略状态与提示文本和包装器写法完全相同。"""

    name = "supervisor"

    def __init__(self, supervisor: SupervisedLLM):
        self.sup = supervisor

    @property
    def stats(self) -> SupervisorStats:
        return self.sup.stats

    def before_llm(self, ev: BeforeLLM) -> list[ContextBlock] | ShortCircuit | None:
        notes, synthetic = self.sup.prepare()
        if synthetic is not None:
            return ShortCircuit(synthetic, source=self.name)
        return [ContextBlock(self.name, "\n\n".join(notes))] if notes else None

    def after_llm(self, ev: AfterLLM) -> None:
        self.sup.account(ev.response)

    def bind_checkpoints(self, service) -> None:
        """共用 AgentLoop 的 CheckpointService(owner=supervisor),不再自建 RollbackManager。"""
        from ..tools.sandbox.checkpoint_service import ServiceRollback

        self.sup._rollback = ServiceRollback(service, "supervisor")
        self.sup._owns_rollback = False

    def on_run_end(self, ev) -> None:
        self.sup.close()


def _query_prefix() -> str:
    """重查知识库的查询前缀:目标包 query_prefix;未声明时为「<library> <new> migration」。"""
    from ..migration.targets import active

    t = active()
    return t.spec.query_prefix or f"{t.label('library')} {t.label('new')} migration".strip()
