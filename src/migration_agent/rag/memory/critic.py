"""经验审核：规则检查来源、保护边界、证据窗口与内容落地性，再由可选模型审核叙述。未配置评分阈值时保留为候选。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .llm_budget import AuxBudget, LLMChannel, SkipCall
from .schema import Experience, SourceRef
from .windows import EvidenceWindow

CRITIC_VERSION = "critic-v1"
SCORES = ("faithfulness", "generality", "safety")
# 落地性检查的通用名(不要求在证据里出现):库无关的部分在这里,库相关的(库名、基类、异常类)来自目标包
# memory.generic_names
GENERIC_BASE = frozenset({"pytest", "pip", "e.g", "i.e", "etc"})


def generic_names() -> frozenset[str]:
    from ...migration.targets import active

    return GENERIC_BASE | frozenset(active().spec.memory.generic_names)


PRESCRIPTIVE = ("recipe", "narrative.instead", "narrative.action")

_BACKTICK = re.compile(r"`([^`\n]{1,200})`")
_QUOTE = re.compile(r"(?<![A-Za-z0-9])'([^'\n]{3,160})'|\"([^\"\n]{3,160})\"")
_FILE = re.compile(r"[\w./-]*\w\.(?:py|toml|cfg|ini|txt|yaml|yml|json)\b")
_DOTTED = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_SNAKE = re.compile(r"\b_*[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*[A-Za-z0-9]\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b")
_DUNDER = re.compile(r"\b__\w+__\b")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\(")


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def identifiers(text: str) -> list[str]:
    """文本里的标识符与引用片段(去重,保持出现顺序)。"""
    found: list[str] = []
    for m in _BACKTICK.finditer(text):
        found.append(m.group(1).strip())
    for m in _QUOTE.finditer(text):
        found.append((m.group(1) or m.group(2)).strip())
    stripped = _QUOTE.sub(" ", _BACKTICK.sub(" ", text))
    for rx in (_FILE, _DOTTED, _DUNDER, _SNAKE, _CAMEL):
        found += rx.findall(stripped)
    found += [m.group(1) for m in _CALL.finditer(stripped)]
    seen, out = set(), []
    for f in found:
        f = f.strip().rstrip(".,;:")
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _grounded(token: str, evidence: str, evidence_ws: str) -> bool:
    generic = generic_names()
    if token in generic or token.lower() in generic:
        return True
    if token in evidence or _ws(token) in evidence_ws:
        return True
    bare = token.lstrip("@.").removesuffix("()").rstrip("(")
    if bare and (bare in evidence or bare in generic):
        return True
    if "/" in token or token.endswith(".py"):  # 文件:路径或文件名出现即可
        name = token.rsplit("/", 1)[-1]
        return name in evidence
    return False


def grounding_check(
    exp: Experience,
    windows: dict[str, EvidenceWindow],
    *,
    forbidden: Callable[[str], str],
    source_allowed: Callable[[SourceRef], tuple[bool, str]] = lambda s: (True, ""),
) -> list[str]:
    """返回问题列表(空 = 通过)。只看模型写的字段;规则字段(diff / sources / test_evidence)不在这里审。"""
    problems: list[str] = []
    if not exp.evidence_refs:
        problems.append("evidence_refs empty")
    unknown = [r for r in exp.evidence_refs if r not in windows]
    if unknown:
        problems.append(f"evidence_refs not provided to the summarizer: {unknown[:3]}")
    used = [windows[r] for r in exp.evidence_refs if r in windows]
    evidence = "\n".join(w.text() for w in used)
    evidence_ws = _ws(evidence)
    for name, text in exp.llm_text().items():
        missing = [t for t in identifiers(text) if not _grounded(t, evidence, evidence_ws)]
        if missing:
            problems.append(f"{name}: not found in evidence: {missing[:4]}")
        if name.startswith(PRESCRIPTIVE):
            why = forbidden(text)
            if why:
                problems.append(f"{name}: {why}")
    if not exp.sources:
        problems.append("no sources")
    bad = [why for ok, why in (source_allowed(s) for s in exp.sources) if not ok]
    if bad:
        problems.append("source not allowed: " + "; ".join(sorted(set(bad)))[:300])
    return problems


class ReplayVerifier(Protocol):
    """可选回放验证:在来源仓库窗口开始时的状态上应用最小 diff,确认对应用例由失败变通过(需要沙箱,外层编排执行)。
    返回 {"ok": bool, "tests_flipped": [...], "detail": str}。本轮只定义接口,未接入。"""

    def verify(self, exp: Experience) -> dict[str, Any]: ...


CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [*SCORES, "reasons"],
    "properties": {
        "faithfulness": {"type": "integer"},
        "generality": {"type": "integer"},
        "safety": {"type": "integer"},
        "reasons": {"type": "string"},
    },
}

# 通用模板;目标包 texts["critic.prompt"] 原样优先
CRITIC_PROMPT = """You are a strict reviewer of memory entries for an automated {migration} migration agent.
Another model wrote the entry below from the evidence below. Score each 1-5 (5 = best):
- faithfulness: every statement in the entry is supported by the evidence.
- generality: the lesson would help on a different repository with the same {migration} problem.
- safety: following it cannot lead to forbidden fixes ({compat_layer} imports, patching {library}, editing tests,
  skip/xfail, downgrading {library}) or to broad risky edits.
Reply with JSON only; "reasons" at most 300 characters.

Entry ({kind}):
{entry}

{evidence}
"""


@dataclass
class CriticVerdict:
    decision: str  # auto_approved | candidate | rejected
    problems: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    reasons: str = ""
    llm: str = ""  # "" | reviewed | skipped:<why> | error:<why> | invalid:<why>
    replay: dict[str, Any] | None = None
    prompt_hash: str = ""

    def trace(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "problems": self.problems[:6],
            "scores": self.scores,
            "llm": self.llm[:200],
            "reasons": self.reasons[:300],
        }


@dataclass
class Critic:
    channel: LLMChannel | None
    budget: AuxBudget | None
    forbidden: Callable[[str], str]
    source_allowed: Callable[[SourceRef], tuple[bool, str]] = lambda s: (True, "")
    thresholds: dict[str, int] | None = None  # 人工抽检冻结后再设定;None → 一律 candidate
    replay: ReplayVerifier | None = None
    replay_kinds: tuple[str, ...] = ("fix",)

    def build_prompt(self, exp: Experience, windows: list[EvidenceWindow]) -> str | None:
        entry = json.dumps(
            {
                "title": exp.summary_title,
                "narrative": exp.narrative,
                **{k: getattr(exp, k) for k in ("symptoms", "recipe", "pitfalls", "query_variants")},
            },
            ensure_ascii=False,
            indent=0,
        )

        from ...migration.targets import active

        template = active().text("critic.prompt", CRITIC_PROMPT)

        def render(max_chars: int | None) -> str:
            per = None if max_chars is None else max(0, max_chars // max(1, len(windows)))
            ev = "\n\n".join(
                w.prompt_text(per if per is None else w.min_prompt_chars() + per) for w in windows
            )
            return template.format(kind=exp.kind, entry=entry, evidence=ev)

        assert self.budget is not None
        limit = self.budget.limits.in_tokens["critic"]
        prompt = render(None)
        if self.budget.estimate_input(prompt, CRITIC_SCHEMA) <= limit:
            return prompt
        room = (limit - self.budget.estimate_input(render(0), CRITIC_SCHEMA)) * 3
        while room > 0:
            prompt = render(room)
            if self.budget.estimate_input(prompt, CRITIC_SCHEMA) <= limit:
                return prompt
            room //= 2
        prompt = render(0)
        return prompt if self.budget.estimate_input(prompt, CRITIC_SCHEMA) <= limit else None

    def review(self, exp: Experience, windows: dict[str, EvidenceWindow]) -> CriticVerdict:
        problems = grounding_check(exp, windows, forbidden=self.forbidden, source_allowed=self.source_allowed)
        if problems:
            v = CriticVerdict("rejected", problems=problems)
            self._record(exp, v)
            return v
        v = CriticVerdict("candidate")
        used = [windows[r] for r in exp.evidence_refs]
        if self.channel is None or self.budget is None:
            v.llm = "skipped:no critic model configured"
        else:
            prompt = self.build_prompt(exp, used)
            if prompt is None:
                v.llm = "skipped:rule-extracted change alone exceeds the critic input limit"
            else:
                v.prompt_hash = hashlib.sha256((CRITIC_VERSION + "\n" + prompt).encode()).hexdigest()[:16]
                try:
                    data = self.budget.call("critic", self.channel, prompt, CRITIC_SCHEMA)
                except SkipCall as e:
                    v.llm = f"skipped:{e}"
                except Exception as e:  # noqa: BLE001
                    v.llm = f"error:{type(e).__name__}: {e}"[:200]
                else:
                    bad = [k for k in SCORES if not (isinstance(data.get(k), int) and 1 <= data[k] <= 5)]
                    if bad:
                        v.llm = f"invalid:scores out of range {bad}"
                    else:
                        v.llm = "reviewed"
                        v.scores = {k: int(data[k]) for k in SCORES}
                        v.reasons = str(data.get("reasons", ""))[:300]
        if self.replay is not None and exp.kind in self.replay_kinds:
            try:
                v.replay = self.replay.verify(exp)
            except Exception as e:  # noqa: BLE001
                v.replay = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}
        if (
            self.thresholds is not None
            and v.llm == "reviewed"
            and exp.review.auto_passed
            and all(v.scores[k] >= self.thresholds.get(k, 6) for k in SCORES)
            and (v.replay is None or v.replay.get("ok"))
        ):
            v.decision = "auto_approved"
        self._record(exp, v)
        return v

    def _record(self, exp: Experience, v: CriticVerdict) -> None:
        exp.review.auto_checks["grounding"] = {"ok": not v.problems, "problems": v.problems[:10]}
        exp.review.auto_checks["critic"] = {
            "decision": v.decision,
            "llm": v.llm,
            "scores": v.scores,
            "reasons": v.reasons,
            "thresholds": self.thresholds,
            **({"replay": v.replay} if v.replay is not None else {}),
        }
        if v.decision == "rejected":
            exp.status = exp.review.status = "rejected"
            exp.review.reviewer = "critic"
            exp.review.reason = "grounding: " + "; ".join(v.problems)[:500]
        if self.channel is not None and v.llm == "reviewed":
            exp.generated_by["critic"] = {
                "role": self.channel.role,
                "model": self.channel.model,
                "prompt_version": CRITIC_VERSION,
                "prompt_hash": v.prompt_hash,
            }
