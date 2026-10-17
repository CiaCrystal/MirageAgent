"""AgentLoop:核心循环。

    loop = AgentLoop(llm, registry, test_tool, tracer, budget, target)
    result = loop.run(task, ctx)

一次 run:
  1. 先跑一遍测试得到初始失败(没有测试 / pytest 跑不起来 → NO_TESTS 直接结束)。
  2. 循环:预算检查 → LLM 调用 → 执行全部 tool_use → 回填 tool_result。
     - 任一「会修改仓库」的工具成功执行后,自动跑一次全量测试,结果作为额外文本块附在同一条回填消息里。
     - 测试全绿即刻结束(成功),不再花 token 让模型写总结。
     - LLM 不再调用工具但测试未通过:最多催促 `max_nudges` 次,再不动就判定 AGENT_GAVE_UP。
  3. 所有事件写 trace;run_end 事件带 success / reason。

不包含任何迁移领域知识;领域知识通过 target(提示)与 tools(能力)注入。

扩展点统一为 HookBus(agent/hooks.py)。`hooks` 为空时与包装器写法完全相同;非空时
  - before_llm 的上下文块追加到最后一条 user 消息,ShortCircuit 代替模型回复;after_llm 收到真实回复;
  - before_tool 的 Deny 代替工具执行(仍在逐工具事务内,Validator 与回滚照旧),Modify 改参数;
    after_tool / on_test_report 的块在下一次模型调用前注入;
  - run_tests 的每份报告(含 agent 自己调用的子集测试)经 on_test_report 广播;
  - 判定成功结束前经 on_stop,钩子可以 Continue(把理由交回模型继续修)或改判(StopEvent.override);
  - run 结束后 on_run_end。
Validator 与逐工具事务回滚是不可配置的硬闸门,留在循环里;检查点经 CheckpointService(与策略控制共用)。
消息变化同时写事件日志(agent/events.py);每次模型调用前校验 messages == 事件投影(不一致写 trace)。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..migration.targets.base import MigrationTarget
from ..migration.task import EndReason, MigrationTask, RunResult
from ..migration.validator import Validator, compat_shim_message
from ..tools.base import ToolContext, ToolResult
from ..tools.fs import ListFilesTool
from ..tools.pytest_tool import RunTestsTool, TestReport
from ..tools.registry import ToolRegistry
from ..tools.sandbox.checkpoint_service import CheckpointService, ServiceRollback
from ..tools.sandbox.protocol import Sandbox
from ..trace.tracer import Tracer
from . import prompts
from .compaction import ContextCompactor
from .events import EventLog, compaction_diff, project_messages
from .hooks import (
    AfterLLM,
    AfterTool,
    BeforeCompact,
    BeforeLLM,
    BeforeTool,
    Deny,
    GiveUpEvent,
    HookBus,
    Modify,
    RunEnd,
    RunStart,
    ShortCircuit,
    StopEvent,
    TestReportEvent,
)
from .llm import LLMClient, LLMError, LLMResponse
from .state import AgentState, Budget


@dataclass
class AgentLoop:
    llm: LLMClient
    registry: ToolRegistry
    test_tool: RunTestsTool
    tracer: Tracer
    budget: Budget
    target: MigrationTarget
    max_nudges: int = 2
    extra_system_context: str = ""
    extra_initial_context: str = ""
    allow_compatibility_shim: bool = False
    compactor: ContextCompactor | None = None
    # 可选:根据初始测试报告生成注入初始消息的上下文(知识库预检索),返回 (文本, trace 元数据)
    initial_context_provider: Callable[[TestReport], tuple[str, dict[str, Any]]] | None = None
    # 扩展点
    hooks: HookBus | None = None

    context_manager: Any = None
    # LLMError 后在同一进程内从最后一个完整事件(llm_request)重发的次数;0 = 直接结束
    llm_error_retries: int = 0
    # 事件日志写盘位置;None 时放在 trace 旁(<trace>.events.jsonl),tracer 无路径时只在内存
    events_path: Any = None

    def run(self, task: MigrationTask, ctx: ToolContext) -> RunResult:
        sandbox = ctx.executor if isinstance(ctx.executor, Sandbox) else None
        self.violations = 0
        self.rollback = None
        self.checkpoints = None
        self.log = None
        if self.compactor:
            self.compactor.reset()
        if self.context_manager is not None:
            self.context_manager.reset()
        listener = None
        try:
            if sandbox:
                sandbox.start()
            self.validator = Validator(ctx)
            self.checkpoints = CheckpointService(ctx, step=lambda: self.tracer.step)
            self.rollback = ServiceRollback(self.checkpoints, "loop")
            listener = self._emit_report
            self.test_tool.listeners.append(listener)
            if self.hooks is not None:
                self.hooks.tracer = self.tracer
                for h in self.hooks.hooks:
                    bind = getattr(h, "bind_checkpoints", None)
                    if bind:
                        bind(self.checkpoints)
            result = self._run(task, ctx)
            if sandbox and result.reason != EndReason.VALIDATION_FAILED:
                sandbox.publish()
            return result
        finally:
            if listener is not None:
                self.test_tool.listeners.remove(listener)
            if self.checkpoints:
                self.checkpoints.close()
            if sandbox:
                sandbox.stop()

    def _emit_report(self, rep: TestReport, subset: bool, ctx: ToolContext) -> None:
        if self.log is not None:
            self.log.append("test_report", {"subset": subset, "report": rep.model_dump(mode="json")})
        if self.hooks is not None:
            self.hooks.pending_blocks += self.hooks.test_report(TestReportEvent(rep, subset, ctx))

    def _run(self, task: MigrationTask, ctx: ToolContext) -> RunResult:
        path = self.events_path
        if path is None and getattr(self.tracer, "path", None):
            path = str(self.tracer.path).removesuffix(".jsonl") + ".events.jsonl"
        self.log = EventLog(path)
        state = AgentState(log=self.log)
        self.state = state
        t0 = time.monotonic()
        self.tracer.step = 0
        self.tracer.log(
            "run_start",
            {
                "repo": str(task.repo_path),
                "target": self.target.name,
                "model": self.llm.model,
                "tools": self.registry.names(),
                "budget": {k: v for k, v in self.budget.__dict__.items() if k != "pricing"},
            },
        )
        self.log.append("run_start", {"repo": str(task.repo_path), "tools": self.registry.names()})
        if self.hooks is not None:
            self.hooks.run_start(RunStart(task, ctx, state))

        # 1. 初始测试
        initial = self._run_tests(ctx, reason="initial")
        state.last_report = initial
        if initial.parse_error or initial.total == 0:
            return self._finish(
                state,
                ctx,
                t0,
                False,
                EndReason.NO_TESTS,
                f"test suite not runnable: {initial.parse_error or 'no tests collected'}\n{initial.stdout_tail}",
            )
        if initial.all_passed:
            return self._finish(
                state, ctx, t0, True, EndReason.TESTS_PASSED, "tests already passing; nothing to do"
            )

        system = prompts.build_system_prompt(self.target, self.extra_system_context)
        if self.allow_compatibility_shim:
            system += "\nConservative mode enabled by the user: compatibility shims are permitted for this run. The earlier shim prohibition is overridden; all other rules still apply."
        tree = ListFilesTool().run({"path": ".", "glob": "*.py"}, ctx).content
        initial_extra = self.extra_initial_context
        if self.initial_context_provider:
            with self.tracer.span("retrieval", purpose="initial_briefing") as sp:
                briefing, meta = self.initial_context_provider(initial)
                sp.payload.update(meta, chars=len(briefing))
            initial_extra = "\n\n".join(x for x in (initial_extra, briefing) if x)
        state.add_user(prompts.build_initial_message(tree, initial.render(), initial_extra))
        tool_schemas = self.registry.schemas()
        previous_input_tokens = 0

        # 2. 循环
        while True:
            exceeded = state.budget_exceeded(self.budget)
            if exceeded:
                return self._finish(state, ctx, t0, False, exceeded, "budget exhausted")

            state.step += 1
            self.tracer.step = state.step
            self.log.step = state.step
            self.log.append(
                "step_start",
                {
                    "step": state.step,
                    "tokens_in": state.tokens_in,
                    "tokens_cached": state.tokens_cached,
                    "tokens_out": state.tokens_out,
                    "nudges": state.nudges,
                },
            )

            self._compact(state, previous_input_tokens)

            try:
                with self.tracer.span("llm_call", model=self.llm.model, n_messages=len(state.messages)) as sp:
                    resp, short = self._call_llm(system, state, tool_schemas)
                    sp.tokens_in, sp.tokens_out = resp.tokens_in, resp.tokens_out
                    sp.payload.update(
                        stop_reason=resp.stop_reason,
                        tool_uses=[t.name for t in resp.tool_uses],
                        cache_read=resp.cache_read_tokens,
                        cache_write=resp.cache_write_tokens,
                        resolved_model=resp.model,
                        text=resp.text[:500],
                    )
                    if short:
                        sp.payload["short_circuit"] = short
            except LLMError as e:
                return self._finish(state, ctx, t0, False, EndReason.LLM_ERROR, str(e))

            state.tokens_in += resp.tokens_in
            state.tokens_cached += resp.cache_read_tokens + resp.cache_write_tokens
            state.cache_read += resp.cache_read_tokens
            state.cache_write += resp.cache_write_tokens
            state.tokens_out += resp.tokens_out
            previous_input_tokens = resp.tokens_in + resp.cache_read_tokens + resp.cache_write_tokens
            state.add_assistant(resp.content)

            tool_uses = resp.tool_uses
            if not tool_uses:
                # 模型停下来了
                if state.last_report is not None and state.last_report.all_passed:
                    done = self._stop(state, ctx, t0, resp.text)
                    if done:
                        return done
                    continue
                state.nudges += 1
                veto = None
                if state.nudges > self.max_nudges:
                    if self.hooks is not None:  # 放弃核验(give_up_check,默认不挂)可以再给一轮
                        veto = self.hooks.give_up_final(GiveUpEvent(state, ctx, state.last_report, resp.text))
                    if veto is None:
                        return self._finish(state, ctx, t0, False, EndReason.AGENT_GAVE_UP, resp.text)
                    state.nudges = self.max_nudges
                report = self._run_tests(ctx, reason="nudge")
                state.last_report = report
                if report.all_passed:
                    done = self._stop(state, ctx, t0, resp.text)
                    if done:
                        return done
                    continue
                self.tracer.log("note", {"nudge": state.nudges})
                state.add_user(prompts.NUDGE_TESTS_STILL_FAILING.format(report=report.render()))
                if veto is not None:
                    state.append_context(veto.reason, veto.source)
                if self.hooks is not None:  # 模型要结束而测试未过(无订阅者时不产生任何文本)
                    for block in self.hooks.give_up(GiveUpEvent(state, ctx, report, resp.text)):
                        state.append_context(block.text, block.source)
                continue

            # 3. 执行工具
            results = []
            mutated = False
            ran_tests = False
            for tu in tool_uses:
                tool = self.registry.get(tu.name)
                guarded = tool is not None and (tool.mutates_repo or tu.name == self.test_tool.name)
                before = ctx.files.tree() if guarded else None
                checkpoint = self.rollback.checkpoint(f"step {state.step}: {tu.name}") if guarded else None
                prior_report = state.last_report
                with self.tracer.span("tool_call", name=tu.name, args=tu.input) as sp:
                    res = self._dispatch(state.step, tu.name, tu.input, ctx)
                    if guarded:
                        validation = self.validator.validate(before)
                        # Direct fs refusals are also explicit invariant violations.
                        if res.is_error and "read-only" in res.content.lower():
                            validation.violations.append(res.content)
                        self.tracer.log("validation", validation.__dict__)
                        if not validation.ok:
                            self.violations += 1
                            self.rollback.rollback(checkpoint)
                            self.validator.restore_runtime()
                            ctx.last_report = prior_report
                            state.last_report = prior_report
                            self.tracer.log(
                                "rollback", {"checkpoint": checkpoint, "violations": validation.violations}
                            )
                            res = ToolResult.error(
                                "Validator rejected this action; rolled back. "
                                + "; ".join(validation.violations)
                            )
                        elif validation.changed_files:
                            mutated = True
                            ran_tests = False
                    sp.payload.update(
                        is_error=res.is_error,
                        result_chars=len(res.content),
                        result=res.content,
                        **_safe_meta(res.metadata),
                    )
                results.append({"tool_use_id": tu.id, "content": res.content, "is_error": res.is_error})
                if self.hooks is not None:
                    self.hooks.pending_blocks += self.hooks.after_tool(
                        AfterTool(state.step, tu.name, tu.input, res, ctx)
                    )
                if self.context_manager is not None:
                    self.context_manager.record_tool(tu.name, tu.input, res, self.tracer, tu.id)
                elif self.compactor:
                    path = self.compactor.record_read(tu.name, tu.input, res.is_error)
                    if path:
                        self.tracer.log("re_read_after_compaction", {"path": path, "tool_use_id": tu.id})
                if tu.name == self.test_tool.name:
                    ran_tests = True
                if ctx.last_report is not None:
                    state.last_report = ctx.last_report

            state.add_tool_results(results)

            # 4. 修改后自动测试
            if mutated or ran_tests:
                report = self._run_tests(ctx, reason="auto_after_edit")
                state.last_report = report
                state.append_context(f"[auto test run after your edit]\n{report.render()}", "auto_test")

            if state.last_report is not None and state.last_report.all_passed:
                done = self._stop(state, ctx, t0, "all tests pass")
                if done:
                    return done

    # ---- extension points ----
    def _compact(self, state: AgentState, previous_input_tokens: int) -> None:
        if self.context_manager is None and not self.compactor:
            return
        before = [
            {**m, "content": [dict(b) for b in m["content"]]} if isinstance(m["content"], list) else dict(m)
            for m in state.messages
        ]
        if self.hooks is not None:
            self.hooks.before_compact(BeforeCompact(state.step, state.messages))
        if self.context_manager is not None:
            event = self.context_manager.compact(state, previous_input_tokens, self.tracer)
        else:
            event = self.compactor.compact(state.messages, previous_input_tokens)
        if event:
            self.tracer.log("compaction", event)
            self.log.append(
                "compaction", {"replaced": compaction_diff(before, state.messages), "meta": event}
            )

    def _call_llm(self, system: str, state: AgentState, tool_schemas: list) -> tuple[LLMResponse, str]:
        short = ""
        resp = None
        if self.hooks is not None:
            d = self.hooks.before_llm(BeforeLLM(state.step, system, state.messages, state))
            if isinstance(d, ShortCircuit):
                resp, short = d.response, d.source or "hook"
            else:
                for block in d:
                    state.append_context(block.text, block.source)
        if resp is None and self.context_manager is not None:
            for block in self.context_manager.inject(state):
                state.append_context(block.text, block.source)
        if project_messages(self.log.events()) != state.messages:
            self.tracer.log("projection_mismatch", {"step": state.step})
        self.log.append("llm_request", {"n_messages": len(state.messages), "short_circuit": short})
        if resp is None:
            attempts = 0
            while True:
                try:
                    n_before, tail_before = len(state.messages), _tail(state.messages)
                    resp = self.llm.complete(system, state.messages, tool_schemas)
                    self._sync_external(state, n_before, tail_before)
                    break
                except LLMError as e:
                    self.log.errors.append({"step": state.step, "error": str(e)[:500]})
                    attempts += 1
                    if attempts > self.llm_error_retries:
                        raise
                    self.tracer.log("llm_resume", {"attempt": attempts, "error": str(e)[:300]})
            if self.hooks is not None:
                self.hooks.after_llm(AfterLLM(state.step, resp, state))
        self.log.append(
            "llm_response",
            {
                "content": resp.content,
                "tokens_in": resp.tokens_in,
                "tokens_cached": resp.cache_read_tokens + resp.cache_write_tokens,
                "tokens_out": resp.tokens_out,
                "stop_reason": resp.stop_reason,
            },
        )
        return resp, short

    def _sync_external(self, state: AgentState, n_before: int, tail_before: Any) -> None:
        """包装器写法(SupervisedLLM)在 complete() 内原地追加文本;把追加的部分补记为 context 事件,
        使事件投影在两种写法下都与 messages 一致。"""
        if len(state.messages) != n_before or not state.messages:
            return
        content = state.messages[-1]["content"]
        if isinstance(content, str) and isinstance(tail_before, str) and content != tail_before:
            self.log.append("context", {"text": content[len(tail_before) + 2 :], "source": "llm_wrapper"})
        elif isinstance(content, list) and isinstance(tail_before, int) and len(content) > tail_before:
            for b in content[tail_before:]:
                self.log.append("context", {"text": b.get("text", ""), "source": "llm_wrapper"})

    def _dispatch(self, step: int, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        # tool_call 记模型发出的调用(两种写法一致);钩子的拒绝 / 改参另记 decision
        self.log.append("tool_call", {"name": name, "args": args})
        if self.hooks is not None:
            d = self.hooks.before_tool(BeforeTool(step, name, args, ctx))
            if isinstance(d, Deny):
                self.log.append("decision", {"tool": name, "deny": d.rule})
                res = d.result
                self.log.append(
                    "tool_result", {"name": name, "is_error": res.is_error, "chars": len(res.content)}
                )
                return res
            if isinstance(d, Modify):
                self.log.append("decision", {"tool": name, "modify": d.args})
                args = d.args
        res = self.registry.dispatch(name, args, ctx)
        self.log.append("tool_result", {"name": name, "is_error": res.is_error, "chars": len(res.content)})
        return res

    def _stop(self, state: AgentState, ctx: ToolContext, t0: float, message: str) -> RunResult | None:
        """测试全绿、即将以成功结束:经 on_stop 闸门。返回 None 表示钩子要求继续(理由已交给模型)。"""
        if self.hooks is not None:
            ev = StopEvent(state, ctx, True, EndReason.TESTS_PASSED, message)
            d = self.hooks.stop(ev)
            if d is not None:
                self.tracer.log(
                    "stop_gate", {"action": "continue", "source": d.source, "reason": d.reason[:1000]}
                )
                state.append_context(d.reason, d.source)
                return None
            if ev.override:
                success, reason, msg = ev.override
                self.tracer.log(
                    "stop_gate", {"action": "override", "reason": str(reason), "message": msg[:500]}
                )
                return self._finish(state, ctx, t0, success, reason, msg)
        return self._finish(state, ctx, t0, True, EndReason.TESTS_PASSED, message)

    # ---- helpers ----
    def _run_tests(self, ctx: ToolContext, reason: str) -> TestReport:
        before = ctx.files.tree()
        checkpoint = self.rollback.checkpoint(f"test run: {reason}")
        with self.tracer.span("test_run", reason=reason) as sp:
            rep = self.test_tool.run_report(ctx)
            validation = self.validator.validate(before)
            if not validation.ok:
                self.violations += 1
                self.rollback.rollback(checkpoint)
                self.validator.restore_runtime()
                self.tracer.log("rollback", {"checkpoint": checkpoint, "violations": validation.violations})
                rep.parse_error = "Validator rejected test side effects: " + "; ".join(validation.violations)
                rep.returncode = 1
                ctx.last_report = rep
            sp.payload.update(
                passed=rep.passed,
                failed=rep.failed,
                errors=rep.errors,
                total=rep.total,
                parse_error=rep.parse_error,
                collection_blocked=rep.collection_blocked,
                n_root_causes=len({f.signature for f in rep.failures}),
            )
        return rep

    def _finish(
        self, state: AgentState, ctx: ToolContext, t0: float, success: bool, reason: EndReason, message: str
    ) -> RunResult:
        rep = state.last_report
        quality = self.validator.validate()
        ctx.changed_files = set(quality.changed_files)
        if not quality.ok:
            success, reason = False, EndReason.VALIDATION_FAILED
            message = "; ".join(quality.violations)
        elif success and quality.compatibility_import_count and not self.allow_compatibility_shim:
            success, reason = False, EndReason.COMPATIBILITY_SHIM_FOUND
            message = compat_shim_message(self.target)  # 文本来自目标包
        cm = self.context_manager or self.compactor
        result = RunResult(
            run_id=self.tracer.run_id,
            success=success,
            reason=reason,
            steps=state.step,
            tokens_in=state.tokens_in,
            tokens_cached=state.tokens_cached,
            tokens_out=state.tokens_out,
            duration_s=time.monotonic() - t0,
            tests_total=rep.total if rep else 0,
            tests_passed=rep.passed if rep else 0,
            files_changed=sorted(ctx.changed_files),
            trace_path=getattr(self.tracer, "path", None),
            message=message[:2000],
            violations=self.violations,
            rollbacks=self.rollback.rollbacks,
            deprecated_api_count=quality.deprecated_api_count,
            compatibility_import_count=quality.compatibility_import_count,
            compactions=cm.events if cm else 0,
            re_read_after_compaction=cm.re_read_after_compaction if cm else 0,
        )
        self.tracer.log("run_end", result.model_dump(mode="json", exclude={"trace_path"}))
        self.log.append("run_end", {"success": success, "reason": str(reason), "steps": state.step})
        if self.hooks is not None:
            self.hooks.run_end(RunEnd(result, ctx))
        return result


def _tail(messages: list[dict]) -> Any:
    """最后一条消息的内容长度标记:str → 原文;list → 块数。"""
    if not messages:
        return None
    c = messages[-1]["content"]
    return c if isinstance(c, str) else len(c)


def _safe_meta(meta: dict) -> dict:
    """metadata 可能含大对象;只保留标量和小 dict。"""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (int, float, str, bool)) or v is None or isinstance(v, dict) and len(str(v)) < 1000:
            out[k] = v
    return out
