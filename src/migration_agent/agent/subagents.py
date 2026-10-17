"""子 agent:上下文隔离,只把摘要交回主流程。参考 Claude Code subagent / Task 工具。

DiagnosisAgent(已实现)
    - 工具:主 run 注册表里的只读工具(read_file / list_files / grep / glob / search_codebase / find_usages /
      search_migration_docs / search_experience),不能编辑、不能跑命令或测试;
    - 预算:≤ max_steps 步、≤ max_tokens token(默认 8 / 60k);独立上下文,不进入主 run 的 messages;
      最后一步或 token 将尽时不再提供工具,要求直接给结论;
    - 输出(JSON,截断到约 500 tokens):root_cause、evidence(文件:行)、suggested_approach、confidence;
    - 由 DynamicPlanHook 在 start / repeat_error / postmortem 重规划之前调用(每 run 至多一次),结论作为 replanner 输入;
    - 使用 router 的 planner 角色(与 executor 同一密钥与客户端);每次调用经 `account(resp)` 计入父 run 的
      AgentState 预算,trace 记 subagent 事件(parent_run_id + subagent_id)。

GroupFixAgent(**只留接口,未实现**)
    设计:按 PlanState 分组,每组一个带编辑工具 + run_tests 的子 agent,独立预算与全新上下文,只交回 diff 摘要与通过数
    变化。未实现原因:它需要在同一沙箱里嵌套运行 AgentLoop(当前 AgentLoop.run 自己管理沙箱生命周期、检查点服务、
    Validator 快照与事件日志),要拆出「不拥有沙箱的子循环」并处理子循环之间的检查点与回滚归属,工作量超出本轮;"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..tools.registry import ToolRegistry

READ_ONLY_TOOLS = (
    "read_file",
    "list_files",
    "grep",
    "glob",
    "search_codebase",
    "find_usages",
    "search_migration_docs",
    "search_experience",
)
DIAGNOSIS_SCHEMA_TEXT = (
    '{"root_cause": str, "evidence": ["path:line — what it shows"], "suggested_approach": str, '
    '"confidence": 0..1}'
)
SYSTEM = (
    "You are a read-only diagnosis sub-agent inside an automated code migration ({target}). You cannot edit files "
    "or run commands. Use the tools to read the code that the failure points at, then reply with ONE JSON object "
    "and nothing else: "
    + DIAGNOSIS_SCHEMA_TEXT
    + ". Use at most 4 tool calls in total and keep the answer under 150 words. "
    "Be concrete (file paths, symbols, the new API to use). Never "
    "suggest compatibility layers, patching the library, editing tests or downgrading the target library."
)


@dataclass
class DiagnosisResult:
    data: dict[str, Any] | None = None
    steps: int = 0
    tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""

    def summary(self, max_chars: int = 1500) -> str:
        if not self.data:
            return ""
        d = self.data
        ev = "; ".join(str(e)[:160] for e in (d.get("evidence") or [])[:5])
        text = (
            f"root cause: {str(d.get('root_cause', ''))[:500]}\nevidence: {ev}\n"
            f"suggested approach: {str(d.get('suggested_approach', ''))[:500]}\nconfidence: {d.get('confidence', '?')}"
        )
        return text[:max_chars]


class DiagnosisAgent:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        *,
        target: Any,
        max_steps: int = 8,
        max_tokens: int = 60_000,
        account: Callable[[Any], None] | None = None,
        tracer: Any = None,
        max_tool_chars: int = 4000,
    ):
        self.llm = llm
        self.tools = ToolRegistry([t for n in READ_ONLY_TOOLS if (t := registry.get(n)) is not None])
        self.target = target
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.account = account or (lambda _r: None)
        self.tracer = tracer
        self.max_tool_chars = max_tool_chars
        self.runs = 0

    def diagnose(self, ctx: Any, failure_pack: str) -> DiagnosisResult:
        self.runs += 1
        sid = f"diagnosis-{self.runs}"
        res = DiagnosisResult()
        system = SYSTEM.replace("{target}", self.target.describe())
        messages: list[dict[str, Any]] = [{"role": "user", "content": failure_pack}]
        schemas = self.tools.schemas()
        try:
            last_call, added = 0, len(failure_pack) + len(system)
            for step in range(1, self.max_steps + 1):
                # 预估下一次调用的 token(上一次 × 1.5 + 新增文本 / 3):超出上限就不再调用;再下一次会超出时,本次就是
                # 最后一次，要求直接作答。
                est = int(last_call * 1.5 + added / 3)
                if res.tokens + est > self.max_tokens:
                    res.error = f"token cap {self.max_tokens} would be exceeded; stopped before step {step}"
                    break
                last = step == self.max_steps or res.tokens + 2 * est > self.max_tokens
                if last:  # 历史里有 tool_use 时仍需带工具定义;只用文字要求直接作答
                    note = "Budget exhausted: do not call tools. Reply now with the JSON object only."
                    if isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append({"type": "text", "text": note})
                    else:
                        messages[-1]["content"] += "\n\n" + note
                resp = self.llm.complete(system, messages, schemas)
                self.account(resp)
                res.steps = step
                last_call = (
                    resp.tokens_in + resp.cache_read_tokens + resp.cache_write_tokens + resp.tokens_out
                )
                res.tokens += last_call
                added = 0
                messages.append({"role": "assistant", "content": resp.content})
                if not resp.tool_uses:
                    res.data = _parse(resp.text)
                    if res.data is None:
                        res.error = "no JSON object in the final answer"
                    break
                results = []
                for tu in resp.tool_uses:
                    out = self.tools.dispatch(tu.name, tu.input, ctx)
                    res.tool_calls.append(tu.name)
                    content = out.content
                    added += min(len(content), self.max_tool_chars)
                    if len(content) > self.max_tool_chars:
                        content = content[: self.max_tool_chars] + "\n… [truncated]"
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": content,
                            **({"is_error": True} if out.is_error else {}),
                        }
                    )
                messages.append({"role": "user", "content": results})
                if last:
                    res.error = "no final answer within the step/token budget"
                    break
        except Exception as e:  # noqa: BLE001 — 诊断失败不影响主 run,replanner 照常进行
            res.error = f"{type(e).__name__}: {e}"[:300]
        if self.tracer is not None:
            self.tracer.log(
                "subagent",
                {
                    "subagent_id": sid,
                    "parent_run_id": getattr(self.tracer, "run_id", None),
                    "kind": "diagnosis",
                    "steps": res.steps,
                    "tokens": res.tokens,
                    "tool_calls": res.tool_calls,
                    "ok": res.data is not None,
                    "error": res.error,
                    "summary": res.summary(600),
                },
            )
        return res


class GroupFixAgent:
    """分组修复子 agent 的接口(未实现,原因见模块说明)。"""

    def __init__(self, *_: Any, **__: Any):
        raise NotImplementedError("GroupFixAgent is an interface only; see agent/subagents.py docstring")


def _parse(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
