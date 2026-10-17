"""依赖方向
    migration 层不依赖 agent.llm;由上层注入 `complete_json(prompt: str, schema: dict) -> dict`
    (agent/router.py 的 planner 角色,与 executor 共用同一个密钥与客户端)。

触发条件(由 agent/dynamic_plan.py 的钩子判断,Replanner 只负责一次规划)
    start          静态计划为空,或覆盖不到初始失败(收集错误、依赖错误、行为差异)
    repeat_error   升级阶梯「换检索词 → 重规划 → 回滚 → 放弃」中的第二级
    stalled        预算到 50% / 75% 时通过数仍没超过历史最好
    stop           测试未通过而模型要结束
    postmortem     一次尝试失败结束后,由 agent/orchestrator.py 调用

输入(只放局部证据,不放整段 trace;见 `build_prompt`)
    按根因分组的失败(签名、影响用例数、分类、源文件帧、traceback 节选)、PlanState.render()、该签名历次失败尝试的
    diff 摘要、已展示过的文档 / 经验 id、可选的诊断子 agent 结论、剩余预算。
    不包含 oracle、上游已迁移版本或其他 case 的 trace(调用方只传本 run 的证据)。

程序校验(不通过的操作丢弃并写 trace;没有任何可用内容时本次视为未重规划)
    - JSON 解析失败 / 不符合 schema → 整体拒绝;
    - op 合法;update / split / block / abandon 引用的项存在;
    - files 存在且不受只读保护、未被锁定;
    - signatures 只保留当前报告里真实的失败签名(完整、唯一前缀或 F 编号),其余丢弃并记入 rejected_extra;
    - goal / approach 不含禁止做法:兼容层导入、给目标库打补丁、改测试或 pytest 配置、降级目标库、skip/xfail
      (`avoid` 字段本来就用来写禁止做法,不检查);
    - upgrade_dependencies 不能降级目标库;
    - test_impossible 由程序复核:所列用例当前确实失败,且异常由只读测试文件里的代码直接触发
      (目标包 failure.impossible_exceptions,如 ImportError / AttributeError / NameError 与库自己的导入错误);
    - 每 run 调用 ≤ max_calls 次(含失败的调用),token 由调用方计入同一 run 预算与 planner 角色账目;
      `reserved` 次只给 postmortem 用(orchestrator 还有下一次尝试时为 1),尝试内的触发最多用 max_calls − reserved 次。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .strategy import source_frames

OPS = ("add", "update", "split", "block", "abandon")

REPLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["diagnosis", "hypotheses", "items"],
    "properties": {
        "diagnosis": {"type": "string"},
        "hypotheses": {"type": "array"},
        "items": {"type": "array"},
        "upgrade_dependencies": {"type": "array"},
        "test_impossible": {},
    },
}

# goal / approach 中出现即拒绝的做法(与 Stop 闸门 / Evaluator 的严格口径一致)
_GENERIC_APPROACHES: list[tuple[str, str]] = [
    (
        "editing tests or pytest config",
        r"\b(?:edit|modify|change|rewrite|update|delete|remove)\w*\s+(?:the\s+)?(?:\w+\s+)?"
        r"(?:tests?|test files?|conftest(?:\.py)?)\b|pytest\.ini|tox\.ini|filterwarnings",
    ),
    ("skip/xfail", r"\b(?:pytest\.)?(?:skip|xfail)\b"),
]
_COMPILED: dict[str, list[tuple[str, re.Pattern[str]]]] = {}


def forbidden_approaches(target: Any = None) -> list[tuple[str, re.Pattern[str]]]:
    """goal / approach 中出现即拒绝的做法(与 Stop 闸门 / Evaluator 的严格口径一致)。
    规则来自目标包 quality.forbidden_approaches(名称与顺序原样);目标没写时由 patch_roots / compat_modules 生成。"""
    if target is None:
        from .targets import active

        target = active()
    if target.name not in _COMPILED:
        rules = target.quality_rules
        pairs = list(rules.forbidden_approaches)
        if not pairs:
            libs = "|".join(re.escape(r) for r in rules.patch_roots)
            pairs = [
                (
                    "compatibility layer",
                    "|".join(re.escape(m) + r"\b" for m in rules.compat_modules) or r"(?!x)x",
                ),
                (
                    "patching the target library",
                    rf"monkey[- ]?patch|patch(?:ing)?\s+(?:the\s+)?(?:{libs})|setattr\(\s*(?:{libs})",
                ),
                *_GENERIC_APPROACHES,
                ("downgrading the target library", rf"(?:{libs})\s*<|downgrad"),
            ]
        _COMPILED[target.name] = [(n, re.compile(p, re.I)) for n, p in pairs]
    return _COMPILED[target.name]


def _impossible_exc() -> tuple[str, ...]:
    from .targets import active

    return tuple(active().spec.failure.impossible_exceptions) or (
        "ImportError",
        "AttributeError",
        "NameError",
    )


RULES = (
    "Rules: migrate the code to the new API. Never propose compatibility layers ({compat_layer}), patching the "
    "library at runtime, editing tests / conftest / pytest config, skip/xfail, or downgrading the target "
    "library. Upgrading other third-party dependencies with pip is allowed. Only reference files listed as "
    "editable. If some tests can only pass by editing read-only test code, list them under test_impossible."
)


@dataclass
class ReplanResult:
    trigger: str
    ok: bool = False
    accepted_ops: list[dict[str, Any]] = field(default_factory=list)
    rejected_ops: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    diagnosis: str = ""
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    upgrade_dependencies: list[str] = field(default_factory=list)
    test_impossible: dict[str, Any] | None = None
    rejected_extra: list[str] = field(default_factory=list)
    error: str = ""
    skipped: str = ""
    prompt_chars: int = 0
    tokens: dict[str, float] = field(default_factory=dict)

    def notes(self) -> str:
        """交给模型的文本(计划块之外的部分)。"""
        if not self.ok:
            return ""
        lines = [f"[replanner · {self.trigger}] {self.diagnosis[:600]}"]
        for h in self.hypotheses[:3]:
            lines.append(f"  - hypothesis ({h.get('confidence', '?')}): {str(h.get('claim', ''))[:240]}")
        if self.upgrade_dependencies:
            lines.append(
                "  - dependencies to upgrade with `pip install -U`: " + ", ".join(self.upgrade_dependencies)
            )
        if self.test_impossible:
            lines.append(
                "  - verified: these tests cannot pass without editing read-only test code; do not spend steps "
                f"on them and never edit tests: {', '.join(self.test_impossible['tests'][:6])}"
            )
        return "\n".join(lines)

    def trace(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "ok": self.ok,
            "accepted": len(self.accepted_ops),
            "rejected": [[o.get("op"), o.get("id"), why] for o, why in self.rejected_ops][:12],
            "rejected_extra": self.rejected_extra[:10],
            "diagnosis": self.diagnosis[:300],
            "hypotheses": [str(h.get("claim", ""))[:160] for h in self.hypotheses[:3]],
            "upgrade_dependencies": self.upgrade_dependencies,
            "test_impossible": self.test_impossible,
            "error": self.error[:300],
            "skipped": self.skipped,
            "prompt_chars": self.prompt_chars,
            "tokens": self.tokens,
        }


def forbidden_reason(text: str) -> str:
    for name, pat in forbidden_approaches():
        if pat.search(text or ""):
            return f"forbidden approach: {name}"
    return ""


class Replanner:
    def __init__(
        self,
        complete_json: Callable[[str, dict[str, Any]], dict[str, Any]],
        target: Any,
        *,
        max_calls: int = 3,
        file_ok: Callable[[str], str] | None = None,
        is_protected: Callable[[str], bool] | None = None,
        usage: Callable[[], dict[str, float]] | None = None,
        trigger_quota: dict[str, int] | None = None,
    ):
        self.complete_json = complete_json
        self.target = target
        self.max_calls = max_calls
        self.file_ok = file_ok or (lambda _p: "")
        self.is_protected = is_protected or (lambda _p: False)
        self.usage = usage  # 返回 planner 角色累计 token / 成本(用于计算本次调用的增量)
        self.calls = 0
        self.reserved = 0  # 留给 postmortem 的调用数(orchestrator 还有下一次尝试时为 1)
        # 每个触发原因单独的额度(None = 共用 max_calls);额度里没写的触发不单独限制,只受总数限制
        self.trigger_quota = dict(trigger_quota) if trigger_quota is not None else None
        self.used: dict[str, int] = {}
        self.history: list[ReplanResult] = []

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls)

    def remaining_for(self, trigger: str) -> int:
        """尝试内的触发(start / repeat_error / stalled / stop)不能动用为 postmortem 预留的调用;
        设了 trigger_quota 时还受该触发自己的额度限制(stalled 用完不影响 stop 的额度)。"""
        left = self.remaining if trigger == "postmortem" else max(0, self.remaining - self.reserved)
        if self.trigger_quota is not None and trigger in self.trigger_quota:
            left = min(left, max(0, self.trigger_quota[trigger] - self.used.get(trigger, 0)))
        return left

    # ------------------------------------------------------------------ prompt
    def build_prompt(
        self,
        trigger: str,
        state: Any,
        report: Any,
        *,
        categories: dict[str, str] | None = None,
        attempts: dict[str, list[str]] | None = None,
        shown_ids: list[str] | None = None,
        diagnosis: dict[str, Any] | None = None,
        budget_left: dict[str, Any] | None = None,
        repo_files: set[str] | None = None,
    ) -> str:
        categories = categories or {}
        groups: dict[str, list[Any]] = {}
        for f in getattr(report, "failures", None) or []:
            groups.setdefault(f.signature, []).append(f)
        why = {
            "start": "The run is starting and the static plan does not cover all initial failures.",
            "repeat_error": "The same failure keeps coming back after several edits.",
            "stalled": "A large part of the budget is used and the number of passing tests has not improved.",
            "stop": "The agent wants to stop while tests still fail.",
            "postmortem": "The previous attempt failed; plan a fresh attempt that starts from the best checkpoint.",
        }.get(trigger, trigger)
        lines = [
            f"You are the planner of an automated code migration ({self.target.describe()}). {why}",
            "Diagnose the remaining failures from the evidence below and output a revised plan as JSON.",
            self.target.text("replanner.rules", RULES),
            "",
            f"TEST RESULT: {report.summary_line() if hasattr(report, 'summary_line') else ''}",
        ]
        for i, (sig, fs) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8], 1):
            frames = source_frames(fs[0].traceback or "", self.is_protected, repo_files)
            tb = (fs[0].traceback or "")[-900:]
            lines += [
                f"[F{i}] {len(fs)} test(s), category={categories.get(sig, 'unknown')}: {sig[:240]}",
                f"     e.g. {fs[0].test_id}; editable frames: {', '.join(frames[:4]) or 'none'}",
                "     " + tb.replace("\n", "\n     "),
            ]
            for a in (attempts or {}).get(sig, [])[-3:]:
                lines.append(f"     earlier attempt that did not fix it: {a[:400]}")
        lines += ["", "CURRENT PLAN:", state.render(max_chars=2000)]
        if diagnosis:
            lines += ["", "DIAGNOSIS FROM A READ-ONLY SUB-AGENT:", str(diagnosis)[:1500]]
        if shown_ids:
            lines.append(f"Documents already shown to the agent (ids): {', '.join(shown_ids[:20])}")
        if budget_left:
            lines.append("Remaining budget: " + ", ".join(f"{k}={v}" for k, v in budget_left.items()))
        lines += [
            "",
            'Output JSON: {"diagnosis": str, "hypotheses": [{"claim": str, "confidence": 0..1, "evidence": [str]}], '
            '"items": [{"op": "add|update|split|block|abandon", "id": "pN (for update/split/block/abandon)", '
            '"goal": str, "files": [editable paths], "signatures": [failure signatures], "approach": str, '
            '"avoid": [str]}], "upgrade_dependencies": [package names], '
            '"test_impossible": {"tests": [test ids], "reason": str} or null}. Keep at most 6 items, ordered by '
            "expected impact. Be brief: diagnosis under 80 words, each approach under 40 words, at most 3 "
            "hypotheses; output only the JSON object.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ call + validate
    def replan(self, trigger: str, state: Any, report: Any, **kw: Any) -> ReplanResult:
        res = ReplanResult(trigger)
        if self.remaining_for(trigger) == 0:
            quota = (self.trigger_quota or {}).get(trigger)
            if quota is not None and self.used.get(trigger, 0) >= quota and self.remaining:
                res.skipped = f"quota for {trigger} reached ({quota})"
            else:
                res.skipped = f"call limit reached ({self.max_calls})"
                if self.remaining:
                    res.skipped += f"; {self.reserved} reserved for postmortem"
            self.history.append(res)
            return res
        prompt = self.build_prompt(trigger, state, report, **kw)
        res.prompt_chars = len(prompt)
        self.calls += 1
        self.used[trigger] = self.used.get(trigger, 0) + 1
        before = self.usage() if self.usage else {}
        try:
            data = self.complete_json(prompt, REPLAN_SCHEMA)
        except Exception as e:  # noqa: BLE001 — 非法 JSON / 调用失败 → 不重规划
            res.error = f"{type(e).__name__}: {e}"[:300]
            data = None
        if self.usage:
            after = self.usage()
            res.tokens = {k: round(after.get(k, 0) - before.get(k, 0), 6) for k in after}
        if data is not None:
            self.validate(data, state, report, res, repo_files=kw.get("repo_files"))
        self.history.append(res)
        return res

    def validate(
        self, data: dict[str, Any], state: Any, report: Any, res: ReplanResult, repo_files=None
    ) -> ReplanResult:
        res.diagnosis = str(data.get("diagnosis") or "")[:1200]
        res.hypotheses = [h for h in data.get("hypotheses") or [] if isinstance(h, dict)][:5]
        for raw in data.get("items") or []:
            if not isinstance(raw, dict):
                res.rejected_ops.append(({"op": "?"}, "item is not an object"))
                continue
            why = self._check_item(raw, state)
            if why:
                res.rejected_ops.append((raw, why))
            else:
                res.accepted_ops.append(self._real_signatures(raw, report, res))
        target_lib = self.target.name.split("-")[0]
        for dep in data.get("upgrade_dependencies") or []:
            dep = str(dep).strip()
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+(?:\[[\w,\-]+\])?(?:\s*[<>=~!]=?\s*[\w.*]+)?", dep):
                res.rejected_extra.append(f"dependency {dep!r}: not a package spec")
            elif forbidden_reason(dep) or (dep.lower().startswith(target_lib) and "<" in dep):
                res.rejected_extra.append(f"dependency {dep!r}: downgrades the target library")
            else:
                res.upgrade_dependencies.append(dep)
        ti = data.get("test_impossible")
        if isinstance(ti, dict) and ti.get("tests"):
            ok, why = self.verify_test_impossible([str(t) for t in ti["tests"]], report, repo_files)
            if ok:
                res.test_impossible = {"tests": ok, "reason": str(ti.get("reason") or "")[:300]}
            if why:
                res.rejected_extra.append(f"test_impossible: {why}")
        res.ok = bool(res.accepted_ops or res.upgrade_dependencies or res.test_impossible)
        if not res.ok and not res.error:
            res.error = "no valid plan changes"
        return res

    def _check_item(self, item: dict[str, Any], state: Any) -> str:
        op = item.get("op")
        if op not in OPS:
            return f"unknown op {op!r}"
        if op != "add" and state.get(str(item.get("id") or "")) is None:
            return f"unknown plan item {item.get('id')!r}"
        if op in ("add", "split") and not str(item.get("goal") or "").strip():
            return "goal is required"
        files = item.get("files") or []
        if not isinstance(files, list):
            return "files must be a list"
        for f in files:
            why = self.file_ok(str(f))
            if why:
                return f"{f}: {why}"
        for key in ("goal", "approach"):
            why = forbidden_reason(str(item.get(key) or ""))
            if why:
                return f"{key}: {why}"
        return ""

    def _real_signatures(self, item: dict[str, Any], report: Any, res: ReplanResult) -> dict[str, Any]:
        """item 的 signatures 只保留当前报告里真实的失败签名(修复完成率虚高的缺陷)。

        接受:完整签名;提示里截断过的签名(唯一前缀,≥ 80 字符);提示里的编号 F1…F8。
        其余(模型自己编的文本)丢弃并记入 rejected_extra,避免 observe 因「关联签名全部消失」把项判成 done。
        """
        if "signatures" not in item:
            return item
        raw = item.get("signatures")
        raw = raw if isinstance(raw, list) else [raw]
        groups: dict[str, int] = {}
        for f in getattr(report, "failures", None) or []:
            groups[f.signature] = groups.get(f.signature, 0) + 1
        by_ref = {f"F{i}": s for i, s in enumerate(sorted(groups, key=lambda s: -groups[s])[:8], 1)}
        kept: list[str] = []
        label = item.get("id") or str(item.get("goal") or "")[:40]
        for s in raw:
            text = str(s).strip().strip("`").removesuffix("…").removesuffix("...").strip()
            ref = re.fullmatch(r"\[?(F\d)\]?", text)
            if text in groups:
                real = text
            elif ref and ref.group(1) in by_ref:
                real = by_ref[ref.group(1)]
            else:
                hits = [g for g in groups if len(text) >= 80 and g.startswith(text)]
                real = hits[0] if len(hits) == 1 else ""
            if real and real not in kept:
                kept.append(real)
            elif not real:
                res.rejected_extra.append(
                    f"{item.get('op')} {label!r}: signature {str(s)[:80]!r} is not failing"
                )
        return {**item, "signatures": kept}

    def verify_test_impossible(self, tests: list[str], report: Any, repo_files=None) -> tuple[list[str], str]:
        failing = {f.test_id: f for f in getattr(report, "failures", None) or []}
        ok, bad = [], []
        for t in tests:
            f = failing.get(t)
            if f is None:
                bad.append(f"{t} is not failing")
                continue
            # 签名形如 "error: ImportError: …" 或 "ImportError: …"
            exc_ok = any(p.strip() in _impossible_exc() for p in f.signature.split(":")[:2])
            raised_in_test = _raising_file(f.traceback or "")
            if exc_ok and raised_in_test and self.is_protected(raised_in_test):
                ok.append(t)
            else:
                bad.append(f"{t}: not raised by read-only test code using a removed API")
        return ok, "; ".join(bad[:4])


def _raising_file(traceback: str) -> str:
    """异常被触发的位置:最后一个不在 site-packages / 标准库里的帧对应的文件(相对路径)。"""
    frames = re.findall(r'File "([^"]+?\.py)", line \d+|^((?:[A-Za-z]:)?[^\s:]+?\.py):\d+:', traceback, re.M)
    paths = [(a or b).replace("\\", "/") for a, b in frames]
    paths = [
        p for p in paths if "site-packages" not in p and "/lib/python" not in p and not p.startswith("<")
    ]
    if not paths:
        return ""
    p = paths[-1]
    for marker in ("/workspace/",):
        if marker in p:
            p = p.split(marker, 1)[1]
    return p.removeprefix("./")
