"""Tracer 监听器:抛异常的监听器不影响 log / span / emit,也不改变 trace 文件内容。"""

from __future__ import annotations

import json

import pytest

from migration_agent.trace.tracer import NullTracer, Tracer


def _boom(ev, phase):
    raise RuntimeError("listener exploded")


def test_null_tracer_ignores_failing_listener():
    seen: list[tuple[str, str]] = []
    tracer = NullTracer()
    tracer.add_listener(_boom)
    tracer.add_listener(lambda ev, phase: seen.append((ev.kind, phase)))

    ev = tracer.log("note", {"msg": "hi"})
    assert ev.kind == "note" and ev.payload == {"msg": "hi"}

    with tracer.span("tool_call", name="read_file") as sp:
        sp.payload["result_chars"] = 3
        sp.tokens_in = 7

    tracer.emit("assistant_text", {"text": "thinking out loud"})

    assert [e.kind for e in tracer.events] == ["note", "tool_call"]  # emit 和 span start 不入记录
    assert tracer.events[1].payload == {"name": "read_file", "result_chars": 3}
    assert tracer.events[1].tokens_in == 7
    assert seen == [
        ("note", "end"),
        ("tool_call", "start"),
        ("tool_call", "end"),
        ("assistant_text", "ephemeral"),
    ]


def test_span_exception_still_recorded_with_failing_listener():
    tracer = NullTracer()
    tracer.add_listener(_boom)
    with pytest.raises(ValueError), tracer.span("test_run", reason="initial"):
        raise ValueError("inner")
    assert tracer.events[-1].kind == "test_run"
    assert "exception" in tracer.events[-1].payload


def test_file_tracer_content_unchanged_by_listeners(tmp_path):
    path = tmp_path / "t.jsonl"
    with Tracer(path, run_id="r1") as tracer:
        tracer.add_listener(_boom)
        tracer.log("run_start", {"repo": "x"})
        with tracer.span("llm_call", model="m"):
            pass
        tracer.emit("assistant_text", {"text": "not on disk"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["run_start", "llm_call"]
    assert rows[1]["payload"] == {"model": "m"}
