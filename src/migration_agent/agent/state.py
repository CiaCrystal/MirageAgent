"""AgentState:消息历史、预算计数、最近测试报告、改动文件。

接口:`add_assistant(blocks)`、`add_user(blocks_or_text)`、`add_tool_results(results)`、
`budget_exceeded(limits) -> EndReason | None`。
snapshot/restore(供回滚使用)只需深拷贝 messages + 计数。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from ..migration.task import EndReason
from ..tools.pytest_tool import TestReport


@dataclass
class Budget:
    "折算成等价普通输入 token,需要 `pricing`);`max_cost_usd`:按 `pricing.cost` 计的成本上限(None 不限)。"

    max_steps: int
    max_total_tokens: int
    max_wall_seconds: int
    token_mode: str = "raw"
    max_cost_usd: float | None = None
    pricing: Any = None
    fallback: str = ""  # 配置要求 priced / max_cost 但没有价格表时的退回说明(写进 trace)


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    tokens_in: int = 0  # 未命中缓存的输入
    tokens_cached: int = 0  # 缓存命中(读)+ 写入的输入
    tokens_out: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_report: TestReport | None = None
    nudges: int = 0
    cache_read: int = 0  # tokens_cached 的拆分(成本口径用)
    cache_write: int = 0
    log: Any = None  # agent/events.EventLog;非空时消息变化同时写事件

    # ---- 消息 ----
    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        if self.log is not None:
            self.log.append(kind, payload)

    def append_context(self, text: str, source: str = "") -> None:
        """追加到最后一条 user 消息末尾(钩子上下文块、自动测试结果)。"""
        from .hooks import append_text

        append_text(self.messages, text)
        self._event("context", {"text": text, "source": source})

    def add_user(self, content: str | list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": content})
        self._event("user", {"content": content})

    def add_assistant(self, blocks: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": blocks})
        self._event("assistant", {"content": blocks})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """results: [{"tool_use_id", "content", "is_error"}]"""
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_use_id"],
                        "content": r["content"],
                        **({"is_error": True} if r.get("is_error") else {}),
                    }
                    for r in results
                ],
            }
        )
        self._event("tool_results", {"content": self.messages[-1]["content"]})

    # ---- 预算 ----
    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def budget_exceeded(self, b: Budget) -> EndReason | None:
        if self.step >= b.max_steps:
            return EndReason.MAX_STEPS
        if self.budget_tokens(b) >= b.max_total_tokens:
            return EndReason.MAX_TOKENS
        if b.max_cost_usd is not None and b.pricing is not None and self.cost(b.pricing) >= b.max_cost_usd:
            return EndReason.MAX_COST
        if self.elapsed_s >= b.max_wall_seconds:
            return EndReason.MAX_WALL_TIME
        return None

    def budget_tokens(self, b: Budget) -> float:
        if b.token_mode == "priced" and b.pricing is not None:
            p = b.pricing
            base = p.input_per_million or 1.0
            return (
                self.tokens_in
                + self.cache_read * p.cache_read_per_million / base
                + self.cache_write * p.cache_write_per_million / base
                + self.tokens_out
            )
        return self.tokens_in + self.tokens_cached + self.tokens_out

    def cost(self, pricing: Any) -> float:
        return pricing.cost(self.tokens_in, self.cache_read, self.cache_write, self.tokens_out)

    # ---- 快照 ----
    def snapshot(self) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(self.messages),
            "step": self.step,
            "tokens_in": self.tokens_in,
            "tokens_cached": self.tokens_cached,
            "tokens_out": self.tokens_out,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.messages = copy.deepcopy(snap["messages"])
        self.step = snap["step"]
        self.tokens_in = snap["tokens_in"]
        self.tokens_cached = snap["tokens_cached"]
        self.tokens_out = snap["tokens_out"]
