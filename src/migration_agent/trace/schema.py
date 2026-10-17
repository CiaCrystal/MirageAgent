"""TraceEvent 结构:`run_id, step, ts, kind(llm_call|tool_call|test_run|retrieval|checkpoint|rollback|strategy|budget|run_start|run_end), payload, tokens_in/out, duration_ms`。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "run_start",
    "run_end",
    "llm_call",
    "tool_call",
    "test_run",
    "retrieval",
    "checkpoint",
    "rollback",
    "validation",
    "note",
    "compaction",
    "re_read_after_compaction",
    "strategy",
    "budget",
    # harness
    "hook_error",
    "permission",
    "projection_mismatch",
    "stop_gate",
    "llm_resume",
    "context",
    "router",
    # 动态规划
    "plan",
    "replan",
    "subagent",
    "classify",
    "attempt",
    # 经验记忆
    "memory_summary",
    "memory_feedback",
    # 目标包 / 缓存 / skills
    "llm_cache",
    "skill",
    # 测试保护与放弃核验
    "test_protection",
    "give_up_check",
]


class TraceEvent(BaseModel):
    run_id: str
    step: int
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0

    def to_jsonl(self) -> str:
        return self.model_dump_json()
