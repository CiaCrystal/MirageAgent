"""经验契约与 Markdown 往返。内容字段用于哈希与去重，来源、证据和审核记录用于追溯。可选总结字段仅在非默认值时写入。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

Status = Literal["candidate", "approved", "rejected", "retired"]
Kind = Literal["fix", "anti_pattern", "strategy", "env"]
KINDS: tuple[str, ...] = ("fix", "anti_pattern", "strategy", "env")
NARRATIVE_KEYS: dict[str, tuple[str, ...]] = {
    "fix": ("root_cause",),
    "anti_pattern": ("tried", "consequence", "instead"),
    "strategy": ("situation", "action", "rationale"),
    "env": ("dependency", "symptom", "action"),
}


def _default_target() -> str:
    from ...migration.targets import active

    return active().name


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class SourceRef(BaseModel):
    """一条经验的一个来源:规范仓库身份、家族、commit、run、步骤与证据引用。"""

    repo: str
    family: str
    commit: str
    run_id: str
    case: str = ""
    group: str = ""  # 抽取时的分组标签(A / dev / toy ...);是否允许由 SourcePolicy 判定,不看这个字段
    steps: list[int] = Field(default_factory=list)
    trace: str = ""
    evidence: str = ""
    files: list[str] = Field(default_factory=list)

    def key(self) -> tuple[str, tuple[int, ...]]:
        return self.run_id, tuple(self.steps)


class TestEvidence(BaseModel):
    """修复前后测试证据(来自同一 run 的相邻两次全量测试)。"""

    __test__ = False  # 防止 pytest 把它当测试类收集

    run_id: str
    step_before: int
    step_after: int
    passed_before: int
    passed_after: int
    total_after: int
    signature_tests_before: int  # 修复前带该签名的失败用例数
    resolved_tests: list[str] = Field(default_factory=list)  # 修复后通过的这些用例(最多 5 个示例)
    new_signatures_after: int = 0  # 这一窗口新出现的失败根因数
    final_passed: int = 0
    final_total: int = 0
    signature_absent_at_end: bool = False


class Review(BaseModel):
    status: Status = "candidate"
    reason: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    auto_checks: dict[str, Any] = Field(default_factory=dict)
    auto_passed: bool = False
    uncertain: list[str] = Field(default_factory=list)  # 证据不足/归因不清的说明,留待人工


class Experience(BaseModel):
    id: str
    target: str = Field(default_factory=lambda: _default_target())
    versions: dict[str, str] = Field(
        default_factory=dict
    )  # {<version_key>: ">=2.0,<3", "observed": "2.14.1"}
    error_signature: str
    error_type: str = ""
    error_text: str = ""
    api_tags: list[str] = Field(default_factory=list)
    api_removed: list[str] = Field(default_factory=list)
    api_added: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    not_applicable: list[str] = Field(default_factory=list)
    code_before: str = ""
    diff: str = ""
    explanation: str = ""
    explanation_source: str = "rule-template"
    combined: bool = False  # 多处改动共同修复,diff 为组合证据
    sources: list[SourceRef] = Field(default_factory=list)
    test_evidence: list[TestEvidence] = Field(default_factory=list)
    status: Status = "candidate"
    review: Review = Field(default_factory=Review)
    content_hash: str = ""
    dedup_key: str = ""
    reuse_success: int = 0
    reuse_failure: int = 0
    reuse_runs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    # ---- 扩展字段(默认值不写入文件,见模块说明)
    kind: Kind = "fix"
    summary_title: str = ""
    narrative: dict[str, str] = Field(default_factory=dict)
    symptoms: list[str] = Field(default_factory=list)
    recipe: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    generated_by: dict[str, Any] = Field(default_factory=dict)
    utility: float = 0.0
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)

    def finalize(self) -> Experience:
        """重新计算哈希与去重键(内容变化后调用)。"""
        self.api_tags = sorted(set(self.api_tags) | set(self.api_removed) | set(self.api_added))
        self.content_hash = content_hash(self)
        self.dedup_key = dedup_key(
            self.error_signature, self.diff, self.api_removed, self.api_added, kind=self.kind
        )
        return self

    def title(self) -> str:
        if self.summary_title:
            return self.summary_title.splitlines()[0][:120]
        return self.error_signature.splitlines()[0][:120] if self.error_signature else self.id

    def has_summary(self) -> bool:
        return bool(self.summary_title or self.narrative or self.recipe or self.symptoms)

    def llm_text(self) -> dict[str, str]:
        """LLM 生成的全部文本字段(落地性检查逐项核对)。键形如 narrative.root_cause / recipe[0]。"""
        out = {"summary_title": self.summary_title} if self.summary_title else {}
        out.update({f"narrative.{k}": v for k, v in self.narrative.items()})
        for name in ("symptoms", "recipe", "pitfalls", "query_variants"):
            out.update({f"{name}[{i}]": v for i, v in enumerate(getattr(self, name))})
        return out


CONTENT_FIELDS = (
    "target",
    "versions",
    "error_signature",
    "error_type",
    "error_text",
    "api_tags",
    "api_removed",
    "api_added",
    "preconditions",
    "not_applicable",
    "code_before",
    "diff",
    "explanation",
    "combined",
)
BODY_FIELDS = {
    "error_text": ("Error", "text"),
    "code_before": ("Code before", "python"),
    "diff": ("Fix (minimal diff)", "diff"),
    "explanation": ("Explanation", ""),
}


SUMMARY_CONTENT_FIELDS = (
    "kind",
    "summary_title",
    "narrative",
    "symptoms",
    "recipe",
    "pitfalls",
    "query_variants",
    "evidence_refs",
)
# 全部扩展字段(取默认值时不写入 front matter)
SUMMARY_FIELDS = (*SUMMARY_CONTENT_FIELDS, "generated_by", "utility", "feedback", "history")
_SUMMARY_DEFAULTS = {
    name: Experience.model_fields[name].get_default(call_default_factory=True) for name in SUMMARY_FIELDS
}


def summary_non_default(exp: Experience, names=SUMMARY_FIELDS) -> dict[str, Any]:
    return {n: v for n in names if (v := exp.model_dump(include={n}, mode="json")[n]) != _SUMMARY_DEFAULTS[n]}


def content_hash(exp: Experience) -> str:
    data = exp.model_dump(include=set(CONTENT_FIELDS), mode="json")
    data.update(summary_non_default(exp, SUMMARY_CONTENT_FIELDS))
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ------------------------------------------------------------------ normalisation

_QUOTED = re.compile(r"'([^'\n]{0,200})'|\"([^\"\n]{0,200})\"")
_NUM = re.compile(r"\b\d+\b")


def signature_key(signature: str) -> str:
    """跨仓库可比的签名:引号里的非 API 名替换为 <q>,数字 → <n>,小写,压缩空白。"""
    names = api_names()

    def q(m: re.Match) -> str:
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        return f"'{inner}'" if inner in names else "<q>"

    s = _QUOTED.sub(q, signature)
    s = _NUM.sub("<n>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def changed_lines(diff: str) -> tuple[list[str], list[str]]:
    removed, added = [], []
    for ln in diff.splitlines():
        if ln.startswith(("+++", "---", "@@")):
            continue
        if ln.startswith("-"):
            removed.append(ln[1:])
        elif ln.startswith("+"):
            added.append(ln[1:])
    return removed, added


_IDENT = re.compile(r"[A-Za-z_]\w*|'[^']*'|\"[^\"]*\"|\d+|\S")
_KEYWORDS = frozenset(
    "def class return import from as if else elif for in while with try except finally raise pass lambda not and "  # noqa: SIM905
    "or is None True False self cls async await yield".split()
)


def fix_pattern(diff: str, removed_api: list[str], added_api: list[str]) -> str:
    if removed_api or added_api:
        return "api:" + ",".join(sorted(set(removed_api))) + "->" + ",".join(sorted(set(added_api)))
    removed, added = changed_lines(diff)

    def norm(lines: list[str]) -> str:
        toks = []
        for ln in lines:
            for t in _IDENT.findall(ln.strip()):
                if t[0] in "'\"":
                    toks.append("S")
                elif t.isdigit():
                    toks.append("N")
                elif re.match(r"[A-Za-z_]", t) and t not in _KEYWORDS and t not in api_names():
                    toks.append("ID")
                else:
                    toks.append(t)
        return " ".join(toks)

    return "tok:" + norm(removed) + " => " + norm(added)


def dedup_key(
    signature: str, diff: str, removed_api: list[str], added_api: list[str], *, kind: str = "fix"
) -> str:
    raw = signature_key(signature) + "|" + fix_pattern(diff, removed_api, added_api)
    if kind != "fix":
        raw = kind + "|" + raw
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def experience_id(key: str) -> str:
    return f"exp-{key[:12]}"


# ------------------------------------------------------------------ API vocabulary (English tags)

# 词表来自目标包 target.yaml 的 memory 段(old_api / new_api:标签 → 正则;extra_api_names;tag_families)。
# 泛名(dict/json/copy/schema/Config/regex…)只匹配调用/关键字参数/类定义形式,降低误标(写在各目标包里)。


class _Vocab:
    def __init__(self, spec: Any):
        self.old = {k: re.compile(v, re.MULTILINE) for k, v in spec.old_api.items()}
        self.new = {k: re.compile(v, re.MULTILINE) for k, v in spec.new_api.items()}
        # 报错里以引号出现时保留原样的名字(signature_key 用)
        self.api_names = frozenset(
            re.sub(r"[^\w]", "", k.split("(")[0].split("=")[0].replace("class ", ""))
            for k in [*spec.old_api, *spec.new_api]
        ) | frozenset(spec.extra_api_names)
        # 同一迁移主题的新旧标签族(报错里的名字 → 相关改动块)。名字既含词表键也含报错中出现的裸名。
        self.families = [
            frozenset(f.split()) | ({"class Config"} if "Config" in f.split() else set())
            for f in spec.tag_families
        ]


_VOCABS: dict[str, _Vocab] = {}


def _vocab() -> _Vocab:
    from ...migration.targets import active

    t = active()
    if t.name not in _VOCABS:
        _VOCABS[t.name] = _Vocab(t.spec.memory)
    return _VOCABS[t.name]


def api_names() -> frozenset[str]:
    return _vocab().api_names


def tag_families() -> list[frozenset[str]]:
    return _vocab().families


def __getattr__(name: str) -> Any:
    if name == "API_NAMES":
        return api_names()
    if name == "TAG_FAMILIES":
        return tag_families()
    raise AttributeError(name)


def related_tags(tags: list[str] | set[str]) -> set[str]:
    """报错标签扩展到同族新旧标签;没有标签时返回空集(无法按 API 收窄)。"""
    out = set(tags)
    for fam in tag_families():
        if fam & out:
            out |= fam
    return out


def api_tags_in(text: str, *, v1: bool = True, v2: bool = True) -> list[str]:
    "Runtime support."
    voc = _vocab()
    out = []
    if v1:
        out += [k for k, rx in voc.old.items() if rx.search(text)]
    if v2:
        out += [k for k, rx in voc.new.items() if rx.search(text)]
    return sorted(set(out))


def error_api_tags(text: str) -> list[str]:
    """报错文本中的 API 标签:词表命中 + `cannot import name 'X'` / `has no attribute 'X'` 中属于词表的名字。"""
    names = api_names()
    tags = set(api_tags_in(text))
    for m in re.finditer(
        r"(?:cannot import name|has no attribute|unexpected keyword argument)\s+'(\w+)'", text
    ):
        name = m.group(1)
        if name in names:
            tags.add(name)
    for m in re.finditer(r"`@?(\w+)`", text):
        if m.group(1) in names:
            tags.add(m.group(1))
    return sorted(tags)


# ------------------------------------------------------------------ Markdown round trip


def _fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+|~+", text)), default=0)
    return "`" * max(3, longest + 1)


def to_markdown(exp: Experience) -> str:
    meta = exp.model_dump(mode="json", exclude=set(BODY_FIELDS) | set(SUMMARY_FIELDS))
    meta.update(summary_non_default(exp))
    head = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000)
    parts = ["---", head.rstrip("\n"), "---", "", f"# {exp.title()}", ""]
    for field, (heading, lang) in BODY_FIELDS.items():
        value = getattr(exp, field)
        parts.append(f"## {heading}")
        parts.append("")
        if lang:
            fence = _fence(value)
            parts += [f"{fence}{lang}", value, fence]
        else:
            parts.append(value)
        parts.append("")
    return "\n".join(parts)


_HEADINGS = {heading: field for field, (heading, _) in BODY_FIELDS.items()}


def from_markdown(text: str) -> Experience:
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError("experience file must start with YAML front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated YAML front matter")
    meta = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5 :]
    values: dict[str, str] = {}
    sections = re.split(r"^## (.+)\n", body, flags=re.MULTILINE)
    for heading, chunk in zip(sections[1::2], sections[2::2], strict=False):
        field = _HEADINGS.get(heading.strip())
        if field is None:
            continue
        lang = BODY_FIELDS[field][1]
        chunk = chunk.strip("\n")
        if lang:
            m = re.match(r"^(`{3,})" + re.escape(lang) + r"\n(.*)\n\1\s*$", chunk, flags=re.DOTALL)
            if m:
                values[field] = m.group(2)
            else:
                m = re.match(r"^(`{3,})" + re.escape(lang) + r"\n\1\s*$", chunk)
                values[field] = "" if m else chunk
        else:
            values[field] = chunk
    return Experience.model_validate({**meta, **values})
