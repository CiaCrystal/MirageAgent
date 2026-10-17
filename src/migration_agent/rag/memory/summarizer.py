"""将规则定位的证据窗口总结为结构化经验；受来源策略、窗口数、调用次数、token 与成本预算约束。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .extractor import NOT_APPLICABLE, target_versions
from .llm_budget import AuxBudget, LLMChannel, SkipCall
from .schema import NARRATIVE_KEYS, Experience, Review, SourceRef, error_api_tags, experience_id
from .windows import EvidenceWindow

PROMPT_VERSION = "summary-v1"
MAX_ITEMS = 4
MAX_ITEM_CHARS = 200
MAX_TITLE_CHARS = 120
MAX_NARRATIVE_CHARS = 400

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "narrative", "symptoms", "recipe", "pitfalls", "query_variants", "evidence_refs"],
    "properties": {
        "title": {"type": "string"},
        "narrative": {"type": "object"},
        "symptoms": {"type": "array"},
        "recipe": {"type": "array"},
        "pitfalls": {"type": "array"},
        "query_variants": {"type": "array"},
        "evidence_refs": {"type": "array"},
    },
}

KIND_GUIDE = {
    "fix": "a FIX: the change shown resolved the error. narrative.root_cause = why {new_api} raises it (1-2 sentences).",
    "anti_pattern": (
        "an ANTI-PATTERN: something was tried and the harness undid or refused it. narrative.tried = what was "
        "attempted; narrative.consequence = what happened (from the facts); narrative.instead = what to do instead "
        "(only if the evidence shows it; otherwise say to re-diagnose)."
    ),
    "strategy": (
        "a STRATEGY lesson about the process. narrative.situation = when it applies; narrative.action = what to do; "
        "narrative.rationale = why, from the facts."
    ),
    "env": (
        "an ENVIRONMENT/dependency lesson. narrative.dependency = the package; narrative.symptom = how it showed up; "
        "narrative.action = what fixed it (from the facts)."
    ),
}

# 通用模板({migration} 等由目标包 labels 填充);目标包 texts["summarizer.prompt"] 原样优先
PROMPT = """You turn ONE evidence window from an automated {migration} migration run into a reusable memory entry.
The entry is {guide}

Rules:
- Use only facts in the evidence below. Write identifiers (class/function/attribute names, file names, error text) exactly
  as they appear in the evidence and never mention one that does not appear there. Do not invent version numbers.
- Do not write code or diffs: the rule-extracted change is attached to the entry by the system.
- recipe: at most {n} general steps, each at most {c} characters. Never recommend {compat_layer} imports, patching {library},
  editing tests or pytest config, skip/xfail, or downgrading {library}; put such warnings in pitfalls.
- symptoms / pitfalls / query_variants: at most {n} items each, each at most {c} characters. query_variants are other ways
  the same error or question could be phrased in a search (reword, shorten, drop file-specific names).
- title: at most {t} characters. narrative values: at most {nc} characters each; keys exactly: {keys}.
- evidence_refs: copy the ref id(s) of the evidence you used, exactly.
- Keep the whole reply under 350 words.

{evidence}
"""


def _target():
    from ...migration.targets import active

    return active()


@dataclass
class SummaryOutcome:
    ref: str
    status: str  # summarized | skipped | rejected_source | invalid | error
    reason: str = ""
    entry: Experience | None = None
    prompt_hash: str = ""
    raw: dict[str, Any] | None = None

    def trace(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "status": self.status,
            "reason": self.reason[:300],
            "id": self.entry.id if self.entry is not None else "",
            "prompt_hash": self.prompt_hash,
        }


@dataclass
class ExperienceSummarizer:
    channel: LLMChannel | None
    budget: AuxBudget
    source_allowed: Callable[[SourceRef], tuple[bool, str]] = lambda s: (True, "")
    target: str = field(default_factory=lambda: _target().name)
    outcomes: list[SummaryOutcome] = field(default_factory=list)
    restricted: str = ""  # 规划阶段因花费上限只总结通过自动检查的窗口时写明原因

    # ---------------------------------------------------------------- prompt

    def build_prompt(self, w: EvidenceWindow) -> str | None:
        """在输入上限内构造提示;只截错误节选与事实,不截 diff。diff 都放不下返回 None。"""
        keys = ", ".join(NARRATIVE_KEYS[w.kind])
        t = _target()
        template = t.text("summarizer.prompt", PROMPT)
        guide = t.text("summarizer.fix_guide", KIND_GUIDE["fix"]) if w.kind == "fix" else KIND_GUIDE[w.kind]

        def render(max_chars: int | None) -> str:
            return template.format(
                guide=guide,
                n=MAX_ITEMS,
                c=MAX_ITEM_CHARS,
                t=MAX_TITLE_CHARS,
                nc=MAX_NARRATIVE_CHARS,
                keys=keys,
                evidence=w.prompt_text(max_chars),
            )

        limit = self.budget.limits.in_tokens["summary"]
        prompt = render(None)
        if self.budget.estimate_input(prompt, SUMMARY_SCHEMA) <= limit:
            return prompt
        fixed = self.budget.estimate_input(render(0), SUMMARY_SCHEMA)
        if fixed > limit:
            return None
        room_chars = (limit - fixed) * 3  # 字节/3 估算;放不下就逐次减半
        while room_chars > 0:
            prompt = render(w.min_prompt_chars() + 200 + room_chars)
            if self.budget.estimate_input(prompt, SUMMARY_SCHEMA) <= limit:
                return prompt
            room_chars //= 2
        prompt = render(0)
        return prompt if self.budget.estimate_input(prompt, SUMMARY_SCHEMA) <= limit else None

    # ---------------------------------------------------------------- main

    def plan(self, windows: list[EvidenceWindow]) -> list[EvidenceWindow]:
        """规划:最坏情况(总结 + 审核各一次)放不下全部窗口时,只保留通过自动检查的窗口,并写明原因。"""
        k = self.budget.limits.max_summaries
        chosen = [w for w in windows if w.rule_status != "rejected"][:k]
        est = 0.0
        for w in chosen:
            prompt = self.build_prompt(w) or ""
            n_in = self.budget.estimate_input(prompt, SUMMARY_SCHEMA)
            est += self.budget.plan_cost("summary", n_in) + self.budget.plan_cost(
                "critic", self.budget.limits.in_tokens["critic"] // 2
            )
        if est > self.budget.cap and any(not w.auto_passed for w in chosen):
            self.restricted = (
                f"estimated worst case {est:.6f} USD for {len(chosen)} window(s) > cap {self.budget.cap:.6f}; "
                "only windows that passed the automatic checks are summarized"
            )
            chosen = [w for w in chosen if w.auto_passed]
        return chosen

    def summarize(self, windows: list[EvidenceWindow]) -> list[SummaryOutcome]:
        for w in self.plan(windows):
            self.outcomes.append(self.summarize_window(w))
        return self.outcomes

    def summarize_window(self, w: EvidenceWindow) -> SummaryOutcome:
        bad = [why for ok, why in (self.source_allowed(s) for s in w.sources) if not ok]
        if not w.sources:
            bad.append("no sources")
        if bad:  # B 来源 / 混合来源:调用模型之前拒绝
            return SummaryOutcome(w.ref, "rejected_source", "; ".join(bad))
        if w.rule_status == "rejected":
            return SummaryOutcome(w.ref, "skipped", "rule hard checks failed: " + ", ".join(w.issues))
        if self.channel is None:
            return SummaryOutcome(w.ref, "skipped", "no summarizer model configured")
        prompt = self.build_prompt(w)
        if prompt is None:
            return SummaryOutcome(
                w.ref, "skipped", "rule-extracted change alone exceeds the summary input limit"
            )
        prompt_hash = hashlib.sha256((PROMPT_VERSION + "\n" + prompt).encode()).hexdigest()[:16]
        try:
            data = self.budget.call("summary", self.channel, prompt, SUMMARY_SCHEMA)
        except SkipCall as e:
            return SummaryOutcome(w.ref, "skipped", str(e), prompt_hash=prompt_hash)
        except Exception as e:  # noqa: BLE001 — 模型失败只影响这一条
            return SummaryOutcome(w.ref, "error", f"{type(e).__name__}: {e}"[:300], prompt_hash=prompt_hash)
        problems = validate_output(data, w.kind)
        if problems:
            return SummaryOutcome(w.ref, "invalid", "; ".join(problems), prompt_hash=prompt_hash, raw=data)
        entry = self.to_experience(w, data, prompt_hash)
        return SummaryOutcome(w.ref, "summarized", entry=entry, prompt_hash=prompt_hash, raw=data)

    def to_experience(self, w: EvidenceWindow, data: dict[str, Any], prompt_hash: str) -> Experience:
        now = datetime.now(UTC).replace(microsecond=0).isoformat()
        if w.base is not None:  # fix:规则候选原样沿用(diff / test_evidence / sources / 自动检查)
            exp = w.base.model_copy(deep=True)
        else:
            exp = Experience(
                id="pending",
                target=self.target,
                versions=target_versions(),
                error_signature=w.signature,
                error_type=w.signature.split(":", 1)[0].strip() if ":" in w.signature else "",
                error_text=w.error_text[:800],
                api_tags=error_api_tags(w.signature + "\n" + w.error_text),
                not_applicable=[_target().text("memory.not_applicable", NOT_APPLICABLE)],
                diff=w.diff,
                explanation=f"Rule-located {w.origin} event in the source run (steps {w.steps}).",
                combined=False,
                sources=[s.model_copy(deep=True) for s in w.sources],
                review=Review(
                    auto_checks={"origin": w.origin, "issues": w.issues, "source_allowed": True},
                    auto_passed=w.auto_passed,
                    uncertain=list(w.issues),
                ),
            )
        exp.kind = w.kind  # type: ignore[assignment]
        exp.summary_title = str(data["title"]).strip()
        exp.narrative = {k: str(data["narrative"][k]).strip() for k in NARRATIVE_KEYS[w.kind]}
        for name in ("symptoms", "recipe", "pitfalls", "query_variants"):
            setattr(exp, name, [str(x).strip() for x in data[name] if str(x).strip()][:MAX_ITEMS])
        exp.evidence_refs = [str(x) for x in data["evidence_refs"]]
        exp.generated_by = {
            "role": self.channel.role if self.channel else "",
            "model": self.channel.model if self.channel else "",
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "at": now,
        }
        exp.status = "candidate"
        exp.review.status = "candidate"
        exp.updated_at = now
        exp.finalize()
        exp.id = experience_id(exp.dedup_key)
        return exp

    def trace(self) -> dict[str, Any]:
        return {
            "restricted": self.restricted,
            "outcomes": [o.trace() for o in self.outcomes],
        }


def validate_output(data: dict[str, Any], kind: str) -> list[str]:
    """结构与长度(提示里已限制;这里程序复核,超长不截断、整条判无效)。"""
    out = []
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        out.append("title missing")
    elif len(data["title"]) > MAX_TITLE_CHARS * 2:
        out.append(f"title longer than {MAX_TITLE_CHARS * 2} chars")
    nar = data.get("narrative")
    if not isinstance(nar, dict):
        out.append("narrative is not an object")
    else:
        for k in NARRATIVE_KEYS[kind]:
            v = nar.get(k)
            if not isinstance(v, str) or not v.strip():
                out.append(f"narrative.{k} missing")
            elif len(v) > MAX_NARRATIVE_CHARS * 2:
                out.append(f"narrative.{k} longer than {MAX_NARRATIVE_CHARS * 2} chars")
        extra = sorted(set(nar) - set(NARRATIVE_KEYS[kind]))
        if extra:
            out.append(f"unexpected narrative keys {extra}")
    for name in ("symptoms", "recipe", "pitfalls", "query_variants", "evidence_refs"):
        v = data.get(name)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            out.append(f"{name} is not a list of strings")
            continue
        if len(v) > MAX_ITEMS * 2:
            out.append(f"{name} has {len(v)} items (limit {MAX_ITEMS})")
        if any(len(x) > MAX_ITEM_CHARS * 2 for x in v):
            out.append(f"{name} has an item longer than {MAX_ITEM_CHARS * 2} chars")
    if isinstance(data.get("evidence_refs"), list) and not data["evidence_refs"]:
        out.append("evidence_refs empty")
    if kind == "fix" and isinstance(data.get("recipe"), list) and not data["recipe"]:
        out.append("recipe empty")
    return out


def prompt_digest(windows: list[EvidenceWindow]) -> str:
    return hashlib.sha256(json.dumps([w.ref for w in windows]).encode()).hexdigest()[:16]
