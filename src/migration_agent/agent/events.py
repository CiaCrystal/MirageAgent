"""完整会话事件日志及消息投影。模型回复、工具结果、压缩与工作笔记按事件记录，支持在每次模型调用前检查消息一致性。"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hooks import append_text


@dataclass
class Event:
    seq: int
    step: int
    kind: str
    payload: dict[str, Any]
    ts: float = 0.0


@dataclass
class EventLog:
    path: Path | None = None
    step: int = 0
    _events: list[Event] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def append(self, kind: str, payload: dict[str, Any]) -> Event:
        ev = Event(len(self._events), self.step, kind, copy.deepcopy(payload), time.time())
        self._events.append(ev)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False, default=str) + "\n")
        return ev

    def events(self, since: int = 0) -> list[Event]:
        return self._events[since:]

    def __len__(self) -> int:
        return len(self._events)

    @classmethod
    def load(cls, path: str | Path) -> EventLog:
        log = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break  # 崩溃时写了一半的最后一行
            log._events.append(Event(**d))
        log.path = None
        return log


def canonical(events: list[Event]) -> list[tuple[int, str, str]]:
    """比较用:去掉时间戳(与只为 trace 存在的耗时字段)。"""
    return [
        (e.step, e.kind, json.dumps(e.payload, sort_keys=True, ensure_ascii=False, default=str))
        for e in events
    ]


def project_messages(events: list[Event]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for e in events:
        p = e.payload
        if e.kind == "user":
            messages.append({"role": "user", "content": copy.deepcopy(p["content"])})
        elif e.kind == "assistant":
            messages.append({"role": "assistant", "content": copy.deepcopy(p["content"])})
        elif e.kind == "tool_results":
            messages.append({"role": "user", "content": copy.deepcopy(p["content"])})
        elif e.kind == "context":
            append_text(messages, p["text"])
        elif e.kind == "compaction":
            for mi, bi, content in p.get("replaced", []):
                messages[mi]["content"][bi]["content"] = content
    return messages


def compaction_diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[list[Any]]:
    """压缩前后(同长度)messages 的 tool_result 内容替换列表 [消息下标, 块下标, 新内容]。"""
    out = []
    for mi, (a, b) in enumerate(zip(before, after, strict=True)):
        if not isinstance(b["content"], list):
            continue
        for bi, (x, y) in enumerate(zip(a["content"], b["content"], strict=True)):
            if x.get("content") != y.get("content"):
                out.append([mi, bi, y.get("content")])
    return out


@dataclass
class ResumedState:
    messages: list[dict[str, Any]]
    step: int
    tokens_in: int
    tokens_cached: int
    tokens_out: int
    nudges: int
    last_report: dict[str, Any] | None
    complete_until: int  # 最后一个完整事件之后的 seq


def resume_state(log: EventLog) -> ResumedState:
    """从事件日志恢复循环状态。最后一个 `llm_request` 之后若没有 `llm_response`,视为未完成的请求(续跑时重发)。"""
    events = log.events()
    cut = len(events)
    for e in reversed(events):
        if e.kind == "llm_response":
            break
        if e.kind == "llm_request":
            cut = e.seq + 1
            break
    events = events[:cut]
    last_step = next((e for e in reversed(events) if e.kind == "step_start"), None)
    counters = dict(last_step.payload) if last_step else {}
    for e in events:
        if e.kind == "llm_response" and last_step and e.seq > last_step.seq:
            counters["tokens_in"] = counters.get("tokens_in", 0) + e.payload.get("tokens_in", 0)
            counters["tokens_cached"] = counters.get("tokens_cached", 0) + e.payload.get("tokens_cached", 0)
            counters["tokens_out"] = counters.get("tokens_out", 0) + e.payload.get("tokens_out", 0)
    report = next((e.payload["report"] for e in reversed(events) if e.kind == "test_report"), None)
    return ResumedState(
        messages=project_messages(events),
        step=int(counters.get("step", 0)),
        tokens_in=int(counters.get("tokens_in", 0)),
        tokens_cached=int(counters.get("tokens_cached", 0)),
        tokens_out=int(counters.get("tokens_out", 0)),
        nudges=int(counters.get("nudges", 0)),
        last_report=report,
        complete_until=cut,
    )
