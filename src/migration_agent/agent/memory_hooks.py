"""MemoryHook:经验记忆的运行内钩子 —— 新签名补查 + 复用反馈。只在 memory_enabled=True 时挂到 HookBus。

    hook = MemoryHook(retriever, session, tracer=tracer, requery=True, feedback=True, k=3, max_tokens=1500)

新签名补查(on_test_report,`requery`)
    第一份全量报告的签名记为已见(开局预检索已覆盖);之后每份全量报告里**新出现**的签名(每份最多 2 个)各自查一次
    经验库(排除本 run 已展示条目),有可靠命中才把条目作为上下文块附在下一次模型调用前;不调用模型,不计步数。
复用反馈(`feedback`)
    hit       条目在本 run 中被展示过(初始预检索 / search_experience / 补查);
    adopted   展示之后,某次成功的编辑引入了条目 diff 新增的 v2 API,或原样包含条目 diff 的某条新增行(≥ 15 字符);
    resolved  采纳之后的全量报告里,条目对应的触发签名全部消失(触发签名 = 条目被展示时最近一份全量报告中、与条目
              签名归一化相同或 API 标签有重合的失败签名);
    rollback  采纳之后,同一步的事务回滚,或策略回滚撤销了采纳时编辑的文件(run 结束时从 trace 读取)。
    anti_pattern 条目:之后的编辑又出现了它记录的尝试改动行 → repeated(只写 trace)。
    run 结束写 trace(kind=memory_feedback);`on_outcome` 回调(在线模式的开发库)负责写回效用分,冻结快照只写 trace。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..rag.memory.schema import Experience, api_tags_in, changed_lines, error_api_tags, signature_key
from .hooks import AfterTool, ContextBlock, Hook, RunEnd, TestReportEvent

EDIT_TOOLS = ("edit_file", "write_file", "multi_edit")
MIN_LINE = 15
NEW_SIG_HEADER = "Experience matching a NEW failure that just appeared"


def _new_text(name: str, args: dict[str, Any]) -> str:
    if name == "edit_file":
        return str(args.get("new_string", ""))
    if name == "write_file":
        return str(args.get("content", ""))
    if name == "multi_edit":
        return "\n".join(str(e.get("new_string", "")) for e in args.get("edits") or [])
    return ""


class MemoryHook(Hook):
    name = "memory"

    def __init__(
        self,
        retriever: Any,
        session: Any,
        *,
        tracer: Any = None,
        requery: bool = True,
        feedback: bool = True,
        k: int = 3,
        max_tokens: int = 1500,
        max_new_per_report: int = 2,
        on_outcome: Callable[[str, str, list[str]], Any] | None = None,
    ):
        self.retriever, self.session, self.tracer = retriever, session, tracer
        self.requery, self.feedback = requery, feedback
        self.k, self.max_tokens, self.max_new = k, max_tokens, max_new_per_report
        self.on_outcome = on_outcome
        self.seen: set[str] | None = None
        self.entries: dict[str, Experience] = {}
        self.events: dict[str, set[str]] = {}
        self.triggers: dict[str, set[str]] = {}
        self.adopted: dict[str, dict[str, Any]] = {}  # id -> {step, files}
        self.requeries: list[dict[str, Any]] = []
        self._last_groups: dict[str, Any] = {}  # 最近一份全量报告(条目被展示时的触发签名从它取)

    # ---------------------------------------------------------------- helpers

    def _sync_hits(self) -> None:
        by_id = {e.id: e for _, e in self.retriever._entries}
        for exp_id in self.session.shown:
            if exp_id not in self.entries and exp_id in by_id:
                exp = self.entries[exp_id] = by_id[exp_id]
                self.events.setdefault(exp_id, set()).add("hit")
                key, tags = signature_key(exp.error_signature), set(exp.api_tags)
                self.triggers[exp_id] = {
                    s
                    for s, f in self._last_groups.items()
                    if signature_key(s) == key
                    or tags & set(error_api_tags(s + "\n" + (getattr(f, "message", "") or "")))
                }

    def _markers(self, exp: Experience) -> tuple[set[str], list[str]]:
        _, added = changed_lines(exp.diff)
        lines = [ln.strip() for ln in added if len(ln.strip()) >= MIN_LINE]
        return set(exp.api_added), lines

    # ---------------------------------------------------------------- events

    def on_test_report(self, ev: TestReportEvent) -> list[ContextBlock] | None:
        if ev.subset:
            return None
        groups: dict[str, Any] = {}
        for f in getattr(ev.report, "failures", []) or []:
            groups.setdefault(f.signature, f)
        current = set(groups)
        if self.feedback:
            self._sync_hits()
            for exp_id in self.entries:
                trig = self.triggers.get(exp_id, set())
                if exp_id in self.adopted and trig and not (trig & current):
                    self.events[exp_id].add("resolved")
        self._last_groups = groups
        if self.seen is None:
            self.seen = set(current)
            return None
        new = [s for s in groups if s not in self.seen]
        self.seen |= current
        if not self.requery or not new:
            return None
        from ..rag.memory.retriever import run_search

        blocks = []
        for sig in new[: self.max_new]:
            f = groups[sig]
            text = ((getattr(f, "traceback", "") or "") or (getattr(f, "message", "") or ""))[-3000:] or sig
            hits, meta = run_search(
                self.retriever,
                self.session,
                text,
                version=self.session.version,
                k=self.k,
                max_tokens=self.max_tokens,
                purpose="new_signature",
            )
            self.requeries.append({"signature": sig[:200], **meta})
            if self.tracer is not None:
                self.tracer.log("retrieval", {"purpose": "new_signature", "signature": sig[:300], **meta})
            if hits:
                blocks.append(
                    ContextBlock(
                        self.name,
                        f"[memory] {NEW_SIG_HEADER} (`{sig[:200]}`). Hints from other repositories' migrations; "
                        "adapt them to this code:\n\n" + "\n\n---\n\n".join(h.text for h in hits),
                    )
                )
        return blocks or None

    def after_tool(self, ev: AfterTool) -> list[ContextBlock] | None:
        if not self.feedback or ev.name not in EDIT_TOOLS or getattr(ev.result, "is_error", False):
            return None
        self._sync_hits()
        text = _new_text(ev.name, ev.args)
        if not text:
            return None
        v2 = set(api_tags_in(text, v1=False))
        path = str(ev.args.get("path", ""))
        for exp_id, exp in self.entries.items():
            apis, lines = self._markers(exp)
            matched = bool(apis & v2) or any(ln in text for ln in lines)
            if not matched:
                continue
            if exp.kind == "anti_pattern":
                self.events[exp_id].add("repeated")
                continue
            info = self.adopted.setdefault(exp_id, {"step": ev.step, "files": []})
            if path and path not in info["files"]:
                info["files"].append(path)
            self.events[exp_id].add("adopted")
        return None

    def _rollbacks(self) -> None:
        path = getattr(self.tracer, "path", None)
        if path is None or not self.adopted:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except OSError:
            return
        for r in rows:
            p, step = r.get("payload") or {}, r.get("step", 0)
            for exp_id, info in self.adopted.items():
                same_step = r.get("kind") == "rollback" and step == info["step"]
                strategy = (
                    r.get("kind") == "strategy"
                    and p.get("action") == "rollback"
                    and p.get("rolled_back")
                    and step >= info["step"]
                    and set(p.get("files") or []) & set(info["files"])
                )
                if same_step or strategy:
                    self.events[exp_id].add("rollback")

    def on_run_end(self, ev: RunEnd) -> None:
        if not self.feedback:
            return
        self._sync_hits()
        self._rollbacks()
        run_id = getattr(ev.result, "run_id", "")
        payload = {
            "run_id": run_id,
            "entries": {i: sorted(e) for i, e in sorted(self.events.items())},
            "adopted": self.adopted,
            "requeries": len(self.requeries),
            "requery_hits": sum(int(r.get("memory_hits", 0)) for r in self.requeries),
        }
        written = []
        if self.on_outcome is not None:
            for exp_id, events in sorted(self.events.items()):
                try:
                    if self.on_outcome(exp_id, run_id, sorted(events)):
                        written.append(exp_id)
                except Exception as e:  # noqa: BLE001 — 写回失败不影响 run
                    payload.setdefault("write_errors", []).append(f"{exp_id}: {type(e).__name__}: {e}"[:200])
        payload["written"] = written
        if self.tracer is not None:
            self.tracer.log("memory_feedback", payload)
