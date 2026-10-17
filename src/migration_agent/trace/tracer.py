"""Tracer:追加写 jsonl(每个 run 一个文件),上下文管理器计时。

接口:
    tracer = Tracer(path, run_id)
    tracer.log("note", {"msg": "..."})
    with tracer.span("tool_call", name="read_file") as span:
        ...
        span.payload["result_len"] = 123        # 退出时自动写入,附带 duration_ms

`step` 由调用方通过 `tracer.step = n` 维护,表示当前 agent 循环轮次。

观察者(实时 UI 等)只读、不影响运行,也不改变 trace 文件内容:
    tracer.add_listener(fn)          # fn(event: TraceEvent, phase: "start" | "end" | "ephemeral")
    - log():写盘后以 phase="end" 通知;
    - span():进入时以 phase="start" 通知(不写盘),退出时照旧写盘并以 phase="end" 通知;
    - emit(kind, payload):只通知监听器、不写盘,kind 不受 EventKind 约束。
监听器抛出的异常一律吞掉(设置环境变量 MIGRATION_AGENT_DEBUG 时在 stderr 打印一次)。
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from .schema import EventKind, TraceEvent

Phase = Literal["start", "end", "ephemeral"]
Listener = Callable[[TraceEvent, Phase], None]


class Span:
    def __init__(self, kind: EventKind, payload: dict[str, Any]):
        self.kind = kind
        self.payload = payload
        self.tokens_in = 0
        self.tokens_out = 0


class Tracer:
    def __init__(self, path: Path | str, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.step = 0
        self._listeners: list[Listener] = []
        self._fh = self.path.open("a", encoding="utf-8")

    # ---- 观察者 ----
    def add_listener(self, fn: Listener) -> None:
        """注册监听器;同一个函数只注册一次。"""
        listeners = self._listener_list()
        if fn not in listeners:
            listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        listeners = self._listener_list()
        if fn in listeners:
            listeners.remove(fn)

    def _listener_list(self) -> list[Listener]:
        # 子类可能不调用 Tracer.__init__,这里按需补上
        listeners = self.__dict__.get("_listeners")
        if listeners is None:
            listeners = self.__dict__["_listeners"] = []
        return listeners

    def _notify(self, ev: TraceEvent, phase: Phase) -> None:
        listeners = self.__dict__.get("_listeners")
        if not listeners:
            return
        for fn in list(listeners):
            try:
                fn(ev, phase)
            except Exception as e:  # noqa: BLE001 — UI 绝不能影响运行
                if os.environ.get("MIGRATION_AGENT_DEBUG") and not self.__dict__.get("_listener_warned"):
                    self.__dict__["_listener_warned"] = True
                    print(f"[trace] listener error ignored: {type(e).__name__}: {e}", file=sys.stderr)

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> TraceEvent:
        """只通知监听器、不写盘的临时事件(如 assistant_text);kind 不受 EventKind 约束。"""
        ev = TraceEvent.model_construct(
            run_id=self.run_id, step=self.step, kind=kind, payload=dict(payload or {})
        )
        self._notify(ev, "ephemeral")
        return ev

    # ---- 写入 ----
    def log(
        self,
        kind: EventKind,
        payload: dict[str, Any] | None = None,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
    ) -> TraceEvent:
        ev = TraceEvent(
            run_id=self.run_id,
            step=self.step,
            kind=kind,
            payload=payload or {},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
        self._write(ev)
        self._notify(ev, "end")
        return ev

    def _write(self, ev: TraceEvent) -> None:
        self._fh.write(ev.to_jsonl() + "\n")
        self._fh.flush()

    @contextmanager
    def span(self, kind: EventKind, **payload: Any) -> Iterator[Span]:
        sp = Span(kind, dict(payload))
        if self.__dict__.get("_listeners"):  # 进入通知只给监听器,不写盘
            start = TraceEvent.model_construct(
                run_id=self.run_id, step=self.step, kind=kind, payload=dict(payload)
            )
            self._notify(start, "start")
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as e:  # 记录异常后继续抛出
            sp.payload["exception"] = repr(e)
            raise
        finally:
            self.log(
                kind,
                sp.payload,
                tokens_in=sp.tokens_in,
                tokens_out=sp.tokens_out,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullTracer(Tracer):
    """仅在内存中记录事件的 Tracer(同样支持监听器)。"""

    def __init__(self, run_id: str = "null"):
        self.run_id = run_id
        self.step = 0
        self.events: list[TraceEvent] = []
        self._listeners = []

    def _write(self, ev: TraceEvent) -> None:
        self.events.append(ev)

    def close(self) -> None:
        pass
