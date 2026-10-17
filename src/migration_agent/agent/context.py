"""ContextManager + Condenser:可插拔的上下文压缩与工作笔记。参考 OpenHands condenser、Claude Code 自动摘要。

动机:主要失败原因是上下文过长(djantic 记忆运行 110k 普通输入 + 1.85M 缓存读取 + 61k 输出 ≈ 2.02M 撞上
max_total_tokens,实际只花 USD 0.12)。现有 compaction 只有一种策略。

三种 Condenser(settings.context_strategy):
                 摘要调用的 token 计入同一 run 预算(AgentState 计数)与 router 的 summarizer 角色
    notes        占位符压缩 + 程序从事件提取的工作笔记(已改文件、测试进展、已解决/仍存在的根因、被拒绝或回滚的做法),
                 每次压缩后与每 `notes_every` 步(笔记有变化时)重新注入,长度有上限

接口(AgentLoop 调用):reset()、compact(state, previous_input_tokens, tracer) -> trace 事件或 None、
inject(state) -> list[ContextBlock]、record_tool(name, args, result, tracer, tool_use_id);属性 events、re_read_after_compaction。
压缩只改 messages 投影(事件日志里是 compaction 替换事件,原始事件不变);压缩前后 token、被压缩块数、重读次数照常写 trace。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .compaction import ContextCompactor, estimate_tokens
from .hooks import ContextBlock

SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["done", "facts", "failed_attempts", "next"],
    "properties": {
        "done": {"type": "array"},
        "facts": {"type": "array"},
        "failed_attempts": {"type": "array"},
        "next": {"type": "array"},
    },
}
EDIT_TOOLS = ("edit_file", "write_file", "insert_line", "multi_edit")


@dataclass
class WorkingNotes:
    max_chars: int = 2500
    files: dict[str, int] = field(default_factory=dict)  # 路径 → 成功编辑次数
    rejected: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    last: str = ""

    def record_tool(self, name: str, args: dict[str, Any], result: Any) -> None:
        path = str(args.get("path") or "")
        if name in EDIT_TOOLS:
            if result.is_error:
                self.rejected.append(f"{name} {path}: {result.content[:160]}")
            elif path:
                self.files[path] = self.files.get(path, 0) + 1

    def record_report(self, report: Any) -> None:
        if report is None:
            return
        line = report.summary_line() if hasattr(report, "summary_line") else str(report)
        if not self.progress or self.progress[-1] != line:
            self.progress.append(line)

    def render(self, report: Any = None) -> str:
        lines = ["[working notes — maintained by the harness from the run's events]"]
        if self.files:
            lines.append(
                "Files you have modified: " + ", ".join(f"{p} (×{n})" for p, n in sorted(self.files.items()))
            )
        if self.progress:
            lines.append("Test progress: " + " → ".join(self.progress[-6:]))
        if report is not None and getattr(report, "failures", None):
            sigs: dict[str, int] = {}
            for f in report.failures:
                sigs[f.signature] = sigs.get(f.signature, 0) + 1
            top = sorted(sigs.items(), key=lambda kv: -kv[1])[:5]
            lines.append("Still failing (root cause: tests): " + "; ".join(f"{s[:120]}: {n}" for s, n in top))
        if self.rejected:
            lines.append("Refused or rolled-back attempts (do not repeat): " + " | ".join(self.rejected[-5:]))
        text = "\n".join(lines)
        return text if len(text) <= self.max_chars else text[: self.max_chars] + " …"


@dataclass
class ContextManager:
    strategy: str = "notes"
    compactor: ContextCompactor = field(default_factory=ContextCompactor)
    router: Any = None
    notes_every: int = 10
    summary_max_chars: int = 3000
    notes: WorkingNotes = field(default_factory=WorkingNotes)
    summaries: int = 0
    _pending: list[ContextBlock] = field(default_factory=list)
    _last_notes_step: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in ("summary", "notes"):
            raise ValueError(f"unknown context strategy {self.strategy!r}")
        if self.strategy == "summary" and self.router is None:
            raise ValueError("summary condenser requires a ModelRouter (summarizer role)")

    @property
    def events(self) -> int:
        return self.compactor.events

    @property
    def re_read_after_compaction(self) -> int:
        return self.compactor.re_read_after_compaction

    def reset(self) -> None:
        self.compactor.reset()
        self.notes = WorkingNotes(max_chars=self.notes.max_chars)
        self.summaries = 0
        self._pending = []
        self._last_notes_step = 0

    def compact(self, state: Any, previous_input_tokens: int, tracer: Any) -> dict | None:
        early = [json.loads(json.dumps(m)) for m in state.messages]
        event = self.compactor.compact(state.messages, previous_input_tokens)
        if not event:
            return None
        event["condenser"] = self.strategy
        if self.strategy == "summary":
            event.update(self._summarize(state, early, tracer))
        else:
            self._pending.append(ContextBlock("notes", self.notes.render(state.last_report)))
            self._last_notes_step = state.step
        event["tokens_after_with_injection"] = estimate_tokens(state.messages) + sum(
            len(b.text) // 3 for b in self._pending
        )
        return event

    def _summarize(self, state: Any, early: list[dict], tracer: Any) -> dict:
        turns = [i for i, m in enumerate(early) if m["role"] == "assistant"]
        boundary = (
            turns[-self.compactor.keep_recent_turns] if len(turns) > self.compactor.keep_recent_turns else 0
        )
        transcript = json.dumps(early[:boundary], ensure_ascii=False)[-60_000:]
        prompt = (
            "Summarise the early part of a code-migration agent's transcript so the agent can continue without it. "
            "List: done (edits made and why), facts (confirmed facts about the code and the tests), "
            "failed_attempts (approaches that did not work), next (open problems). Be specific: file paths, "
            "symbols, error messages.\n\nTRANSCRIPT:\n" + transcript
        )
        before = {k: getattr(v, "tokens_in", 0) for k, v in self.router.usage().items()}
        try:
            data = self.router.complete_json("summarizer", prompt, SUMMARY_SCHEMA)
        except Exception as e:  # noqa: BLE001 — 摘要失败退回占位符压缩
            return {"summary_error": f"{type(e).__name__}: {e}"[:300]}
        usage = self.router.usage().get("summarizer")
        if usage is not None:  # 摘要 token 计入同一 run 预算
            state.tokens_in += usage.tokens_in - before.get("summarizer", 0)
        self.summaries += 1
        parts = ["[summary of earlier turns — written by the summarizer model]"]
        for key in ("done", "facts", "failed_attempts", "next"):
            items = [str(x) for x in data.get(key, [])][:12]
            if items:
                parts.append(f"{key}:\n" + "\n".join(f"  - {x}" for x in items))
        text = "\n".join(parts)[: self.summary_max_chars]
        self._pending.append(ContextBlock("summary", text))
        return {"summary_chars": len(text), "summaries": self.summaries}

    def inject(self, state: Any) -> list[ContextBlock]:
        self.notes.record_report(state.last_report)
        out, self._pending = self._pending, []
        if (
            self.strategy == "notes"
            and not out
            and self.notes_every
            and state.step - self._last_notes_step >= self.notes_every
        ):
            text = self.notes.render(state.last_report)
            if text != self.notes.last:
                out.append(ContextBlock("notes", text))
                self._last_notes_step = state.step
        for b in out:
            if b.source == "notes":
                self.notes.last = b.text
        return out

    def record_tool(
        self, name: str, args: dict[str, Any], result: Any, tracer: Any, tool_use_id: str
    ) -> None:
        self.notes.record_tool(name, args, result)
        path = self.compactor.record_read(name, args, result.is_error)
        if path:
            tracer.log("re_read_after_compaction", {"path": path, "tool_use_id": tool_use_id})


def build_context_manager(settings: Any, router: Any = None) -> ContextManager:
    return ContextManager(
        strategy=settings.context_strategy,
        compactor=ContextCompactor(
            enabled=True,
            context_window=settings.context_window_tokens,
            threshold=settings.compaction_threshold,
            keep_recent_turns=settings.compaction_keep_recent_turns,
        ),
        router=router,
    )
