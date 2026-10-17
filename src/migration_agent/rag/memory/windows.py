"""证据窗口:规则定位「哪个窗口、哪段改动、哪个签名」,LLM 只对窗口做概括。不调用模型。

    windows = fix_windows(candidates) + event_windows(events, run_id, source, final_success=...)

两类来源:
  - fix:extractor.extract 产出的候选(签名、错误节选、最小 diff、测试前后变化、自动检查结果);
  - 事件:trace 事件(dict:kind / step / payload)里的
        rollback      策略回滚(相同签名反复出现)→ anti_pattern;尝试过的改动取窗口内编辑工具的参数原文
        stop_gate     Stop 质量闸门退回(库补丁 / 兼容层)→ anti_pattern
        permission    权限/命令策略拒绝 → anti_pattern
        abandon       放弃签名(锁定文件)→ strategy
        dependency    成功的 `pip install`(升级依赖)且下一次全量测试通过数上升 → env
    事件窗口的「自动检查」:来源允许、证据完整(找得到尝试的命令/改动与后果)。

每个窗口有稳定的 `ref`(evidence_refs 引用它)与 `text()`(落地性检查的检索范围,不截断);
`prompt_text(limit)` 只截取错误节选与事实,不截 diff。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import Experience, SourceRef

PRIORITY = {"fix:auto": 0, "fix:uncertain": 1, "rollback": 2, "stop_gate": 3, "dependency": 4}
EDIT_TOOLS = ("edit_file", "write_file", "multi_edit")


@dataclass
class EvidenceWindow:
    ref: str
    kind: str  # fix | anti_pattern | strategy | env
    origin: str  # fix | rollback | stop_gate | permission | abandon | dependency
    run_id: str
    steps: list[int]
    signature: str
    error_text: str = ""
    diff: str = ""  # 规则给出的改动原文(fix:最小 diff;anti_pattern:尝试过的编辑参数),永不截断
    facts: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceRef] = field(default_factory=list)
    auto_passed: bool = False
    rule_status: str = "candidate"  # candidate | rejected(规则硬性检查失败,不总结)
    issues: list[str] = field(default_factory=list)
    base: Experience | None = None  # fix 窗口对应的规则候选(diff / test_evidence / sources 原样沿用)

    @property
    def priority(self) -> int:
        if self.origin == "fix":
            return PRIORITY["fix:auto" if self.auto_passed else "fix:uncertain"]
        return PRIORITY.get(self.origin, 5)

    def _facts_text(self) -> str:
        return json.dumps(self.facts, ensure_ascii=False, sort_keys=True, indent=0)

    def text(self) -> str:
        """落地性检查的检索范围:签名、错误节选、改动、事实、文件名(全量,不截断)。"""
        files = sorted({f for s in self.sources for f in s.files})
        extra = ""
        if self.base is not None:
            extra = "\n".join([self.base.code_before, " ".join(self.base.api_tags), self.base.error_type])
        return "\n".join(
            [self.signature, self.error_text, self.diff, self._facts_text(), " ".join(files), extra]
        )

    def prompt_text(self, max_chars: int | None = None) -> str:
        """给模型看的证据。超出上限时先截错误节选、再截事实;diff 永远完整(放不下由调用方跳过窗口)。"""
        head = [f"[evidence ref={self.ref}] origin={self.origin} run={self.run_id} steps={self.steps}"]
        head.append(f"Error signature: {self.signature}")
        diff = f"Rule-extracted change (verbatim):\n```diff\n{self.diff}\n```" if self.diff else ""
        err, facts = self.error_text, self._facts_text()
        if max_chars is not None:
            fixed = len("\n".join(head)) + len(diff) + 200
            room = max(0, max_chars - fixed)
            if len(err) + len(facts) > room:
                err = err[: max(0, room // 2)]
                facts = facts[: max(0, room - len(err))]
        parts = head + ([f"Error excerpt:\n{err}"] if err else []) + ([diff] if diff else [])
        parts.append(f"Facts:\n{facts}")
        return "\n".join(parts)

    def min_prompt_chars(self) -> int:
        return len(self.prompt_text(0))


def _ref(prefix: str, run_id: str, *bits: Any) -> str:
    h = hashlib.sha256(json.dumps([run_id, *bits], default=str).encode()).hexdigest()[:8]
    return f"{prefix}:{run_id}:{h}"


# ------------------------------------------------------------------ fix windows (extractor)


def fix_windows(candidates: list[Experience]) -> list[EvidenceWindow]:
    out = []
    for c in candidates:
        ev = c.test_evidence[0] if c.test_evidence else None
        facts: dict[str, Any] = {
            "attribution": c.review.auto_checks.get("details", {}).get("attribution", ""),
            "files": sorted({f for s in c.sources for f in s.files}),
            "api_removed": c.api_removed,
            "api_added": c.api_added,
            "uncertain": c.review.uncertain,
        }
        if ev is not None:
            facts["tests"] = {
                "before": f"{ev.passed_before} passed",
                "after": f"{ev.passed_after}/{ev.total_after} passed",
                "failing_with_signature_before": ev.signature_tests_before,
                "resolved_tests": ev.resolved_tests,
                "signature_absent_at_end": ev.signature_absent_at_end,
            }
        steps = sorted({s for src in c.sources for s in src.steps})
        out.append(
            EvidenceWindow(
                ref=_ref("win", c.sources[0].run_id if c.sources else "", c.dedup_key, steps),
                kind="fix",
                origin="fix",
                run_id=c.sources[0].run_id if c.sources else "",
                steps=steps,
                signature=c.error_signature,
                error_text=c.error_text,
                diff=c.diff,
                facts=facts,
                sources=[s.model_copy(deep=True) for s in c.sources],
                auto_passed=c.review.auto_passed,
                rule_status=c.status,
                issues=[k for k, v in c.review.auto_checks.items() if v is False],
                base=c,
            )
        )
    return out


# ------------------------------------------------------------------ event windows (trace)


def _edit_text(name: str, args: dict[str, Any]) -> str:
    """编辑工具参数 → 规则文本(不是模型重写的 diff)。"""
    path = args.get("path", "")
    if name == "edit_file":
        return _pseudo_diff(path, args.get("old_string", ""), args.get("new_string", ""))
    if name == "multi_edit":
        return "\n".join(
            _pseudo_diff(path, e.get("old_string", ""), e.get("new_string", ""))
            for e in args.get("edits") or []
        )
    if name == "write_file":
        content = str(args.get("content", ""))
        return f"--- a/{path}\n+++ b/{path}\n" + "\n".join("+" + ln for ln in content.splitlines()[:60])
    return ""


def _pseudo_diff(path: str, old: str, new: str) -> str:
    lines = [f"--- a/{path}", f"+++ b/{path}", "@@"]
    lines += ["-" + ln for ln in str(old).splitlines()]
    lines += ["+" + ln for ln in str(new).splitlines()]
    return "\n".join(lines)


def _source(source: SourceRef, steps: list[int], files: list[str]) -> SourceRef:
    return source.model_copy(update={"steps": steps, "files": files})


def event_windows(
    events: list[dict[str, Any]],
    source: SourceRef,
    *,
    source_allowed: bool = True,
    max_diff_lines: int = 80,
) -> list[EvidenceWindow]:
    """从一个 run 的 trace 事件定位 anti_pattern / strategy / env 窗口。`events`:按顺序的 {kind, step, payload}。"""
    run_id = source.run_id
    out: list[EvidenceWindow] = []
    tests = [e for e in events if e["kind"] == "test_run"]
    edits = [
        e
        for e in events
        if e["kind"] == "tool_call"
        and e["payload"].get("name") in EDIT_TOOLS
        and not e["payload"].get("is_error")
    ]

    def report_before(step: int) -> dict | None:
        prior = [t for t in tests if t["step"] < step]
        return prior[-1]["payload"] if prior else None

    def report_after(step: int) -> dict | None:
        later = [t for t in tests if t["step"] >= step]
        return later[0]["payload"] if later else None

    def counts(p: dict | None) -> str:
        return f"{p.get('passed')}/{p.get('total')} passed" if p else "unknown"

    for e in events:
        p, step = e["payload"], e["step"]
        if e["kind"] == "strategy" and p.get("action") == "rollback" and p.get("rolled_back"):
            files = list(p.get("files") or [])
            # 回滚撤销的改动:上一次被回滚到的报告之后、回滚步骤之前,对这些文件的编辑
            start = 0
            if isinstance(p.get("to_report"), int) and p["to_report"] < len(tests):
                start = tests[p["to_report"]]["step"]
            tried = [
                x
                for x in edits
                if start < x["step"] <= step and x["payload"].get("args", {}).get("path") in files
            ]
            diff = "\n".join(_edit_text(x["payload"]["name"], x["payload"].get("args") or {}) for x in tried)
            n = len(diff.splitlines())
            issues = [] if tried else ["no edit tool call found for the rolled-back files"]
            if n > max_diff_lines:
                issues.append(f"attempted change has {n} lines (limit {max_diff_lines})")
            steps = sorted({x["step"] for x in tried} | {step})
            out.append(
                EvidenceWindow(
                    ref=_ref("ev", run_id, "rollback", step, p.get("signature")),
                    kind="anti_pattern",
                    origin="rollback",
                    run_id=run_id,
                    steps=steps,
                    signature=str(p.get("signature", "")),
                    diff=diff,
                    facts={
                        "event": "strategy controller rolled back the files after the same error repeated",
                        "repeat_count": p.get("count"),
                        "rolled_back_files": files,
                        "tests_before_attempts": counts(report_before(steps[0])),
                        "tests_at_rollback": counts(report_before(step + 1)),
                    },
                    sources=[_source(source, steps, files)],
                    auto_passed=source_allowed and not issues,
                    issues=issues,
                )
            )
        elif e["kind"] == "strategy" and p.get("action") == "abandon":
            files = list(p.get("files") or [])
            out.append(
                EvidenceWindow(
                    ref=_ref("ev", run_id, "abandon", step, p.get("signature")),
                    kind="strategy",
                    origin="abandon",
                    run_id=run_id,
                    steps=[step],
                    signature=str(p.get("signature", "")),
                    facts={
                        "event": "strategy controller abandoned this error and locked the files",
                        "repeat_count": p.get("count"),
                        "locked_files": list(p.get("locked") or files),
                        "tests_at_abandon": counts(report_before(step + 1)),
                    },
                    sources=[_source(source, [step], files)],
                    auto_passed=source_allowed,
                )
            )
        elif e["kind"] == "stop_gate" and p.get("action") == "continue":
            reason = str(p.get("reason", ""))
            issues = [ln.strip()[2:] for ln in reason.splitlines() if ln.strip().startswith("- [")][:6]
            out.append(
                EvidenceWindow(
                    ref=_ref("ev", run_id, "stop_gate", step),
                    kind="anti_pattern",
                    origin="stop_gate",
                    run_id=run_id,
                    steps=[step],
                    signature="stop gate: " + "; ".join(issues)[:200],
                    error_text=reason[:800],
                    facts={
                        "event": "tests passed but the stop quality gate returned the run",
                        "issues": issues,
                    },
                    sources=[_source(source, [step], [])],
                    auto_passed=source_allowed and bool(issues),
                    issues=[] if issues else ["stop gate event without issues"],
                )
            )
        elif e["kind"] == "permission":
            args = p.get("args") or {}
            what = args.get("command") or args.get("path") or ""
            out.append(
                EvidenceWindow(
                    ref=_ref("ev", run_id, "permission", step, what),
                    kind="anti_pattern",
                    origin="permission",
                    run_id=run_id,
                    steps=[step],
                    signature=f"permission denied: {p.get('rule', '')} ({p.get('tool', '')})",
                    error_text=str(p.get("message", ""))[:400],
                    facts={
                        "event": "the harness refused this tool call",
                        "rule": p.get("rule"),
                        "tool": p.get("tool"),
                        "attempted": what,
                    },
                    sources=[_source(source, [step], [])],
                    auto_passed=source_allowed and bool(what),
                )
            )
        elif (
            e["kind"] == "tool_call"
            and p.get("name") == "run_command"
            and re.search(r"\bpip\s+install\b", str((p.get("args") or {}).get("command", "")))
            and not p.get("is_error")
        ):
            before, after = report_before(step), report_after(step + 1)
            improved = bool(before and after and (after.get("passed", 0) > before.get("passed", 0)))
            cmd = str(p["args"]["command"])
            out.append(
                EvidenceWindow(
                    ref=_ref("ev", run_id, "dependency", step, cmd),
                    kind="env",
                    origin="dependency",
                    run_id=run_id,
                    steps=[step],
                    signature=f"dependency change: {cmd[:160]}",
                    error_text=str(p.get("result", ""))[-600:],
                    facts={
                        "command": cmd,
                        "tests_before": counts(before),
                        "tests_after": counts(after),
                        "improved": improved,
                    },
                    sources=[_source(source, [step], [])],
                    auto_passed=source_allowed and improved,
                    issues=[] if improved else ["no later full test run improved"],
                )
            )
    if not source_allowed:
        for w in out:
            w.rule_status = "rejected"
            w.issues.append("source not allowed")
    return out


def select(windows: list[EvidenceWindow], k: int) -> list[EvidenceWindow]:
    """每 run 最多 k 个窗口:规则拒绝的不选;按优先级(通过自动检查的 fix 最先)与步骤排序。"""
    eligible = [w for w in windows if w.rule_status != "rejected"]
    return sorted(eligible, key=lambda w: (w.priority, not w.auto_passed, w.steps[:1], w.ref))[:k]
