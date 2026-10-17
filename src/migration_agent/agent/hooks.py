"""运行事件与钩子总线：在模型调用、工具执行、测试和运行结束时分发扩展行为。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMResponse

# ---------------------------------------------------------------- decisions


@dataclass
class ContextBlock:
    source: str
    text: str


@dataclass
class ShortCircuit:
    response: LLMResponse
    source: str = ""


@dataclass
class Allow:
    pass


@dataclass
class Deny:
    """`result`:返回给模型的 ToolResult(保留各规则原有的文本与 metadata);`rule`:规则 id。"""

    result: Any
    rule: str = ""


@dataclass
class Modify:
    args: dict[str, Any]


@dataclass
class Continue:
    reason: str
    source: str = ""


# ---------------------------------------------------------------- events


@dataclass
class RunStart:
    task: Any
    ctx: Any
    state: Any


@dataclass
class BeforeLLM:
    step: int
    system: str
    messages: list[dict[str, Any]]
    state: Any


@dataclass
class AfterLLM:
    step: int
    response: LLMResponse
    state: Any


@dataclass
class BeforeTool:
    step: int
    name: str
    args: dict[str, Any]
    ctx: Any


@dataclass
class AfterTool:
    step: int
    name: str
    args: dict[str, Any]
    result: Any
    ctx: Any


@dataclass
class TestReportEvent:
    report: Any
    subset: bool
    ctx: Any


@dataclass
class BeforeCompact:
    step: int
    messages: list[dict[str, Any]]


@dataclass
class StopEvent:
    state: Any
    ctx: Any
    success: bool
    reason: Any
    message: str
    continues: int = 0
    # 钩子可改判:(success, EndReason, message);例如 Stop 闸门退回次数用尽后按严格口径判失败
    override: tuple | None = None


@dataclass
class GiveUpEvent:
    state: Any
    ctx: Any
    report: Any
    message: str


@dataclass
class RunEnd:
    result: Any
    ctx: Any


class Hook:
    name: str = "hook"

    def on_run_start(self, ev: RunStart) -> None:
        return None

    def before_llm(self, ev: BeforeLLM) -> list[ContextBlock] | ShortCircuit | None:
        return None

    def after_llm(self, ev: AfterLLM) -> None:
        return None

    def before_tool(self, ev: BeforeTool) -> Allow | Deny | Modify | None:
        return None

    def after_tool(self, ev: AfterTool) -> list[ContextBlock] | None:
        return None

    def on_test_report(self, ev: TestReportEvent) -> list[ContextBlock] | None:
        return None

    def before_compact(self, ev: BeforeCompact) -> None:
        return None

    def on_stop(self, ev: StopEvent) -> Continue | None:
        return None

    def on_give_up(self, ev: GiveUpEvent) -> list[ContextBlock] | None:
        return None

    def on_give_up_final(self, ev: GiveUpEvent) -> Continue | None:
        return None

    def on_run_end(self, ev: RunEnd) -> None:
        return None


@dataclass
class HookBus:
    hooks: list[Hook] = field(default_factory=list)
    tracer: Any = None
    max_continues: int = 2
    continues: int = 0
    pending_blocks: list[ContextBlock] = field(default_factory=list)

    def add(self, hook: Hook) -> HookBus:
        self.hooks.append(hook)
        return self

    def get(self, name: str) -> Hook | None:
        return next((h for h in self.hooks if h.name == name), None)

    def reset(self) -> None:
        self.continues = 0
        self.pending_blocks = []

    def _call(self, hook: Hook, method: str, ev: Any) -> Any:
        try:
            return getattr(hook, method)(ev)
        except Exception as e:  # noqa: BLE001 — 钩子失败按无决策处理,不中断 run
            if self.tracer is not None:
                self.tracer.log(
                    "hook_error",
                    {"hook": hook.name, "event": method, "error": f"{type(e).__name__}: {e}"[:300]},
                )
            return None

    # ---- broadcast-only events
    def run_start(self, ev: RunStart) -> None:
        self.reset()
        for h in self.hooks:
            self._call(h, "on_run_start", ev)

    def after_llm(self, ev: AfterLLM) -> None:
        for h in self.hooks:
            self._call(h, "after_llm", ev)

    def before_compact(self, ev: BeforeCompact) -> None:
        for h in self.hooks:
            self._call(h, "before_compact", ev)

    def run_end(self, ev: RunEnd) -> None:
        for h in self.hooks:
            self._call(h, "on_run_end", ev)

    # ---- decision events
    def before_llm(self, ev: BeforeLLM) -> list[ContextBlock] | ShortCircuit:
        """返回 ShortCircuit(第一个)或按顺序拼接的上下文块(含 test_report / after_tool 积攒的块)。"""
        blocks, self.pending_blocks = list(self.pending_blocks), []
        for h in self.hooks:
            d = self._call(h, "before_llm", ev)
            if isinstance(d, ShortCircuit):
                d.source = d.source or h.name
                return d
            if d:
                blocks += d
        return blocks

    def before_tool(self, ev: BeforeTool) -> Allow | Deny | Modify:
        args = dict(ev.args)
        modified = False
        for h in self.hooks:
            d = self._call(h, "before_tool", BeforeTool(ev.step, ev.name, args, ev.ctx))
            if isinstance(d, Deny):
                d.rule = d.rule or h.name
                return d
            if isinstance(d, Modify):
                args, modified = {**args, **d.args}, True
        return Modify(args) if modified else Allow()

    def after_tool(self, ev: AfterTool) -> list[ContextBlock]:
        out: list[ContextBlock] = []
        for h in self.hooks:
            out += self._call(h, "after_tool", ev) or []
        return out

    def test_report(self, ev: TestReportEvent) -> list[ContextBlock]:
        out: list[ContextBlock] = []
        for h in self.hooks:
            out += self._call(h, "on_test_report", ev) or []
        return out

    def give_up(self, ev: GiveUpEvent) -> list[ContextBlock]:
        out: list[ContextBlock] = []
        for h in self.hooks:
            out += self._call(h, "on_give_up", ev) or []
        return out

    def give_up_final(self, ev: GiveUpEvent) -> Continue | None:
        """催促次数用尽、即将判 AGENT_GAVE_UP:第一个返回 Continue 的钩子再给一轮(各钩子自己限制次数)。"""
        for h in self.hooks:
            d = self._call(h, "on_give_up_final", ev)
            if isinstance(d, Continue):
                d.source = d.source or h.name
                return d
        return None

    def stop(self, ev: StopEvent) -> Continue | None:
        """超过 max_continues 后钩子仍被调用(可改判 override),但 Continue 不再生效。"""
        ev.continues = self.continues
        for h in self.hooks:
            d = self._call(h, "on_stop", ev)
            if isinstance(d, Continue) and self.continues < self.max_continues:
                d.source = d.source or h.name
                self.continues += 1
                return d
        return None


def append_text(messages: list[dict[str, Any]], text: str) -> None:
    """把文本追加到最后一条 user 消息末尾(与包装器写法 SupervisedLLM 逐字节相同)。"""
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": text})
        return
    content = messages[-1]["content"]
    if isinstance(content, str):
        messages[-1]["content"] = content + "\n\n" + text
    else:
        content.append({"type": "text", "text": text})
