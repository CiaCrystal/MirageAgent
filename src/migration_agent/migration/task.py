"""MigrationTask / RunResult 数据结构。

MigrationTask:仓库路径、迁移目标名、测试命令(默认 `pytest -q`)、预算覆盖。
RunResult:是否成功、终止原因、步数、token、耗时、最终测试报告摘要、trace 路径。
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class MigrationTask(BaseModel):
    repo_path: Path
    target: str = Field(default_factory=lambda: _default_target())
    test_cmd: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    max_steps: int | None = None
    max_total_tokens: int | None = None
    max_wall_seconds: int | None = None
    run_id: str | None = None


class EndReason(StrEnum):
    TESTS_PASSED = "tests_passed"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_WALL_TIME = "max_wall_time"
    AGENT_GAVE_UP = "agent_gave_up"
    LLM_ERROR = "llm_error"
    NO_TESTS = "no_tests"
    VALIDATION_FAILED = "validation_failed"
    COMPATIBILITY_SHIM_FOUND = "compatibility_shim_found"
    MAX_COST = "max_cost"  # harness:成本口径预算
    QUALITY_GATE_FAILED = "quality_gate_failed"  # harness:Stop 闸门退回次数用尽后仍有库补丁/兼容层


class RunResult(BaseModel):
    run_id: str
    success: bool
    reason: EndReason
    steps: int
    tokens_in: int  # 未命中缓存的输入 token
    tokens_cached: int = 0  # 缓存命中/写入的输入 token(计费更便宜,但占上下文)
    tokens_out: int
    duration_s: float
    tests_total: int
    tests_passed: int
    files_changed: list[str]
    trace_path: Path | None = None
    message: str = ""
    violations: int = 0
    rollbacks: int = 0
    deprecated_api_count: int = 0
    compatibility_import_count: int = 0
    compactions: int = 0
    re_read_after_compaction: int = 0

    @property
    def pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0


def _default_target() -> str:
    from .targets import default_target_name

    return default_target_name()
