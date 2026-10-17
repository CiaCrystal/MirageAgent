"""Tracer:追加写 jsonl(每个 run 一个文件),上下文管理器计时。

接口:
    tracer = Tracer(path, run_id)
    tracer.log("note", {"msg": "..."})
    with tracer.span("tool_call", name="read_file") as span:
        ...
        span.payload["result_len"] = 123        # 退出时自动写入,附带 duration_ms

`step` 由调用方通过 `tracer.step = n` 维护,表示当前 agent 循环轮次。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import EventKind, TraceEvent


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
        self._fh = self.path.open("a", encoding="utf-8")

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
        self._fh.write(ev.to_jsonl() + "\n")
        self._fh.flush()
        return ev

    @contextmanager
    def span(self, kind: EventKind, **payload: Any) -> Iterator[Span]:
        sp = Span(kind, dict(payload))
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
    """仅在内存中记录事件的 Tracer。"""

    def __init__(self, run_id: str = "null"):
        self.run_id = run_id
        self.step = 0
        self.events: list[TraceEvent] = []

    def log(self, kind, payload=None, *, tokens_in=0, tokens_out=0, duration_ms=0.0):  # type: ignore[override]
        ev = TraceEvent(
            run_id=self.run_id,
            step=self.step,
            kind=kind,
            payload=payload or {},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
        self.events.append(ev)
        return ev

    def close(self) -> None:
        pass
