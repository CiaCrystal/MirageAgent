"""trace 分析:读取 jsonl,输出步骤时间线、token 分布、工具调用频次。

接口:`load(path) -> list[TraceEvent]`、`summarize(events) -> TraceSummary`、`render_timeline(events) -> str`。
可在此扩展「检索 → 修改 → 测试通过」因果链分析。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from .schema import TraceEvent


class TraceSummary(BaseModel):
    run_id: str
    steps: int
    llm_calls: int
    tool_calls: int
    tool_call_counts: dict[str, int]
    test_runs: int
    tokens_in: int
    tokens_cached: int
    tokens_out: int
    wall_ms: float
    llm_ms: float
    tool_ms: float
    test_ms: float
    success: bool | None
    end_reason: str | None
    violations: int = 0
    rollbacks: int = 0
    deprecated_api_count: int = 0
    compatibility_import_count: int = 0
    compactions: int = 0
    re_read_after_compaction: int = 0


def load(path: Path | str) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(TraceEvent.model_validate_json(line))
    return events


def summarize(events: list[TraceEvent]) -> TraceSummary:
    if not events:
        raise ValueError("empty trace")
    tool_counts: Counter[str] = Counter()
    llm_ms = tool_ms = test_ms = 0.0
    llm_calls = tool_calls = test_runs = 0
    tin = tout = tcached = 0
    success = None
    end_reason = None
    quality = {}
    for ev in events:
        tin += ev.tokens_in
        tout += ev.tokens_out
        if ev.kind == "llm_call":
            llm_calls += 1
            llm_ms += ev.duration_ms
            tcached += int(ev.payload.get("cache_read") or 0) + int(ev.payload.get("cache_write") or 0)
        elif ev.kind == "tool_call":
            tool_calls += 1
            tool_ms += ev.duration_ms
            tool_counts[str(ev.payload.get("name", "?"))] += 1
        elif ev.kind == "test_run":
            test_runs += 1
            test_ms += ev.duration_ms
        elif ev.kind == "run_end":
            success = ev.payload.get("success")
            end_reason = ev.payload.get("reason")
            quality = {
                k: ev.payload.get(k, 0)
                for k in (
                    "violations",
                    "rollbacks",
                    "deprecated_api_count",
                    "compatibility_import_count",
                    "compactions",
                    "re_read_after_compaction",
                )
            }
    wall_ms = (events[-1].ts - events[0].ts).total_seconds() * 1000
    return TraceSummary(
        run_id=events[0].run_id,
        steps=max(e.step for e in events),
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        tool_call_counts=dict(tool_counts),
        test_runs=test_runs,
        tokens_in=tin,
        tokens_cached=tcached,
        tokens_out=tout,
        wall_ms=wall_ms,
        llm_ms=llm_ms,
        tool_ms=tool_ms,
        test_ms=test_ms,
        success=success,
        end_reason=end_reason,
        **quality,
    )


def render_timeline(events: list[TraceEvent]) -> str:
    """每个事件一行的紧凑时间线,供 `trace show` 输出。"""
    lines = []
    for ev in events:
        p = ev.payload
        if ev.kind == "llm_call":
            desc = f"stop={p.get('stop_reason')} tools={p.get('tool_uses')}"
        elif ev.kind == "tool_call":
            desc = f"{p.get('name')} {'ERR ' if p.get('is_error') else ''}{_short(p.get('args'))}"
        elif ev.kind == "test_run":
            desc = f"passed={p.get('passed')} failed={p.get('failed')} errors={p.get('errors')}"
        elif ev.kind in ("run_start", "run_end"):
            desc = _short(p, 120)
        else:
            desc = _short(p)
        tok = f" tok={ev.tokens_in}/{ev.tokens_out}" if ev.tokens_in or ev.tokens_out else ""
        lines.append(f"[{ev.step:>3}] {ev.kind:<9} {ev.duration_ms:>8.0f}ms{tok}  {desc}")
    return "\n".join(lines)


def _short(obj: object, n: int = 80) -> str:
    s = repr(obj) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[: n - 1] + "…"
