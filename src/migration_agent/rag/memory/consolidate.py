"""记忆生命周期:入库合并、复用反馈与效用分。参考 Mem0 的 ADD / UPDATE / DELETE / NOOP。

    cons = Consolidator(store, channel=critic_channel, budget=aux_budget, online=True, run_id=run_id)
    exp_id, op, detail = cons.ingest(entry, verdict.decision)
    cons.run_outcome(exp_id, run_id, {"hit", "adopted", "resolved"})     # 复用反馈(同一 run 幂等)
    cons.retire_low_utility(threshold=-0.5, min_runs=2)

入库操作(程序先判断,只有「相似但不确定」时才调 LLM;判断依据与结果写进条目 history)
    ADD      新模式,新建条目
    UPDATE   同一条目(同 dedup_key、同一位置)有新来源 / 新证据 / 补上了 LLM 概括 → 并入,保留全部来源与测试证据
    MERGE    另一个已有条目与它是同一模式(LLM 判定)→ 并入那个条目(沿用 store._merge 的来源保留规则)
    NOOP     同一输入已经处理过,或没有任何新信息
    RETIRE   效用分持续为负 → status=retired(文件保留,移出索引)
    REJECTED 审核结论为 rejected:进 candidates/rejected,保留原因(不参与合并)
幂等:同一输入(内容哈希 + 来源 + 结论)处理过就 NOOP,不重复写。

放置
    auto_approved 且 online=True → experiences/(status=approved,reviewer=critic-auto,立即可检索);
    其余 → candidates/pending(等人工);rejected → candidates/rejected。
    MERGE 只在同一位置之间进行(未审核内容不并进已发布条目);合并后的来源必须全部被允许(不制造混合来源)。

复用反馈
    命中(检索返回并注入)→ 采纳(之后的编辑引用了条目里的 API / 改动行)→ 结果(相关签名消失 resolved / 回滚 rollback)。
    事件由 agent 侧钩子收集,run 结束时 `run_outcome` 一次写入(同一 run 幂等),并接上 store.record_reuse
    (采纳过才计:success = resolved 且无 rollback)。冻结快照只写 trace,不调用这里。
效用分
    utility = (resolved 的 run 数 − rollback 的 run 数) / (adopted 的 run 数 + 1)(平滑;未采纳的命中不计分)。
    检索排序时作为降权项;adopted ≥ min_runs 且 utility < 阈值 → RETIRE。阈值只在开发数据上设定。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from filelock import FileLock

from .llm_budget import AuxBudget, LLMChannel, SkipCall
from .schema import Experience, SourceRef, fix_pattern, signature_key
from .store import ExperienceStore, _merge

OPS = ("ADD", "UPDATE", "MERGE", "NOOP", "RETIRE", "REJECTED")
FEEDBACK_EVENTS = ("hit", "adopted", "resolved", "rollback")
LIST_FIELDS = ("symptoms", "recipe", "pitfalls", "query_variants")
MAX_LIST = 5

MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["same_pattern", "reason"],
    "properties": {"same_pattern": {"type": "boolean"}, "reason": {"type": "string"}},
}
# 通用模板;目标包 texts["consolidate.merge_prompt"] 原样优先
MERGE_PROMPT = """Two memory entries from automated {migration} migrations. Are they the SAME reusable pattern
(same cause and the same kind of fix / lesson, so one entry should hold both sources)? Answer false when unsure.
Reply with JSON only; "reason" at most 200 characters.

Entry A:
{a}

Entry B:
{b}
"""


def render_merge_prompt(a: str, b: str, target: Any) -> str:
    return target.text("consolidate.merge_prompt", MERGE_PROMPT).format(a=a, b=b)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def input_key(exp: Experience, decision: str) -> str:
    raw = json.dumps(
        [exp.content_hash, sorted(map(list, (s.key() for s in exp.sources))), decision], default=str
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _brief(exp: Experience) -> str:
    return json.dumps(
        {
            "kind": exp.kind,
            "error": exp.error_signature[:300],
            "title": exp.summary_title,
            "narrative": exp.narrative,
            "api_removed": exp.api_removed,
            "api_added": exp.api_added,
            "change": exp.diff[:1200],
        },
        ensure_ascii=False,
        indent=0,
    )


def _absorb(base: Experience, new: Experience) -> tuple[Experience, bool]:
    """把 new 的来源、测试证据、evidence_refs、LLM 列表字段并入 base;base 没有 LLM 概括时补上。"""
    merged, changed = _merge(base, new)
    merged = merged.model_copy(deep=True)
    refs = [r for r in new.evidence_refs if r not in merged.evidence_refs]
    if refs:
        merged.evidence_refs += refs
        changed = True
    if not merged.has_summary() and new.has_summary():
        for name in ("summary_title", "narrative", *LIST_FIELDS, "generated_by"):
            setattr(merged, name, getattr(new, name))
        merged.kind = new.kind
        changed = True
    elif new.has_summary():
        for name in LIST_FIELDS:
            cur = getattr(merged, name)
            extra = [x for x in getattr(new, name) if x not in cur]
            if extra and len(cur) < MAX_LIST:
                setattr(merged, name, (cur + extra)[:MAX_LIST])
                changed = True
    return merged, changed


def _similar(a: Experience, b: Experience) -> str:
    """程序判断「可能是同一模式」的理由;空串 = 不相似。"""
    if a.kind != b.kind or a.id == b.id or b.status in ("retired", "rejected"):
        return ""
    if signature_key(a.error_signature) == signature_key(b.error_signature):
        return "same normalized error signature"
    pa = fix_pattern(a.diff, a.api_removed, a.api_added)
    if pa.startswith("api:") and pa == fix_pattern(b.diff, b.api_removed, b.api_added):
        return f"same API change pattern ({pa})"
    return ""


class Consolidator:
    def __init__(
        self,
        store: ExperienceStore,
        channel: LLMChannel | None = None,
        budget: AuxBudget | None = None,
        *,
        online: bool = True,
        run_id: str = "",
        source_allowed: Callable[[SourceRef], tuple[bool, str]] = lambda s: (True, ""),
    ):
        self.store, self.channel, self.budget = store, channel, budget
        self.online, self.run_id, self.source_allowed = online, run_id, source_allowed
        self.log: list[dict[str, Any]] = []

    def _lock(self) -> FileLock:
        self.store.root.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.store.root / ".consolidate.lock"), timeout=120)

    def _hist(self, op: str, key: str, reason: str, *, by: str = "rule", other: str = "") -> dict[str, Any]:
        return {
            "at": _now(),
            "op": op,
            "by": by,
            "input": key,
            "run_id": self.run_id,
            "reason": reason[:300],
            **({"other": other} if other else {}),
        }

    def _allowed(self, exp: Experience) -> bool:
        return all(self.source_allowed(s)[0] for s in exp.sources)

    # ---------------------------------------------------------------- ingest

    def ingest(self, exp: Experience, decision: str) -> tuple[str, str, str]:
        exp = exp.model_copy(deep=True).finalize()
        key = input_key(exp, decision)
        with self._lock():
            out = self._ingest(exp, decision, key)
        self.log.append({"id": out[0], "op": out[1], "detail": out[2][:200], "decision": decision})
        return out

    def _ingest(self, exp: Experience, decision: str, key: str) -> tuple[str, str, str]:
        if decision == "rejected":
            where = "rejected"
            exp.status = exp.review.status = "rejected"
        elif decision == "auto_approved" and self.online:
            where = "experiences"
            exp.status = exp.review.status = "approved"
            exp.review.reviewer = "critic-auto"
            exp.review.reviewed_at = _now()
            exp.review.reason = (
                "auto-approved: grounding ok, rule checks passed, critic scores >= frozen thresholds"
            )
        else:
            where = "pending"
            exp.status = exp.review.status = "candidate"
        placed = dict(self.store.locate(exp.id))
        # 1) 同一条目(同 dedup_key)
        for loc in (where,) if where == "rejected" else (where, "experiences", "pending"):
            cur = placed.get(loc)
            if cur is None:
                continue
            if any(h.get("input") == key for h in cur.history):
                return exp.id, "NOOP", f"input already processed ({loc})"
            if loc == "pending" and where == "experiences":  # 同一条目这次自动通过:从候选区提升发布
                merged, _ = _absorb(exp, cur)
                if not self._allowed(merged):
                    return exp.id, "NOOP", "merge would mix in sources that are not allowed"
                merged.history = cur.history + [
                    self._hist("UPDATE", key, "promoted from pending (auto-approved)")
                ]
                self.store.write_entry(merged, "experiences")
                self.store.remove_entry(exp.id, "pending")
                return exp.id, "UPDATE", "promoted from pending"
            if loc != where:
                merged, changed = _absorb(cur, exp)
                if not changed:  # 已发布 / 已在候选区且没有新信息
                    return exp.id, "NOOP", f"no new information over the {loc} entry"
                continue
            merged, changed = _absorb(cur, exp)
            if not self._allowed(merged):
                return exp.id, "NOOP", "merge would mix in sources that are not allowed"
            merged.history.append(self._hist("UPDATE" if changed else "NOOP", key, f"same entry in {loc}"))
            self.store.write_entry(merged, loc)
            return exp.id, "UPDATE" if changed else "NOOP", f"same entry in {loc}"
        if where == "rejected":
            exp.history.append(self._hist("REJECTED", key, exp.review.reason or "rejected by review"))
            self.store.write_entry(exp, "rejected")
            return exp.id, "REJECTED", exp.review.reason
        # 2) 相似但 id 不同:同一位置之间才合并,由 LLM 判断
        notes = []
        for loc, other in self.store.lifecycle_entries():
            why = _similar(exp, other)
            if not why:
                continue
            if loc != where:
                notes.append(f"similar to {other.id} in {loc} ({why}); not merged across locations")
                continue
            same, reason = self._judge(exp, other)
            if same is None:
                notes.append(f"similar to {other.id} ({why}); undecided: {reason}")
                continue
            if not same:
                notes.append(f"similar to {other.id} ({why}); LLM: different pattern ({reason})")
                continue
            merged, _ = _absorb(other, exp)
            if not self._allowed(merged):
                notes.append(f"similar to {other.id}; merge would mix in sources that are not allowed")
                continue
            merged.history.append(self._hist("MERGE", key, f"{why}; LLM: {reason}", by="llm", other=exp.id))
            self.store.write_entry(merged, loc)
            return other.id, "MERGE", f"merged into {other.id}: {reason}"
        # 3) 新模式
        exp.history.append(self._hist("ADD", key, "; ".join(notes) or "new pattern"))
        self.store.write_entry(exp, where)
        return exp.id, "ADD", "; ".join(notes) or where

    def _judge(self, a: Experience, b: Experience) -> tuple[bool | None, str]:
        if self.channel is None or self.budget is None:
            return None, "no model for merge decisions"
        from ...migration.targets import active

        prompt = render_merge_prompt(_brief(a), _brief(b), active())
        try:
            data = self.budget.call("merge", self.channel, prompt, MERGE_SCHEMA)
        except SkipCall as e:
            return None, str(e)
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"[:200]
        if not isinstance(data.get("same_pattern"), bool):
            return None, "invalid merge answer"
        return data["same_pattern"], str(data.get("reason", ""))[:200]

    # ---------------------------------------------------------------- feedback / utility

    def run_outcome(self, exp_id: str, run_id: str, events: set[str] | list[str]) -> bool:
        """一次 run 对一条已发布经验的复用反馈;同一 run 已记录的事件不重复。返回是否有新事件写入。"""
        events = [e for e in FEEDBACK_EVENTS if e in set(events)]
        with self._lock():
            rows = dict(self.store.locate(exp_id))
            exp = rows.get("experiences")
            if exp is None:
                return False
            have = {(f.get("run_id"), f.get("event")) for f in exp.feedback}
            new = [e for e in events if (run_id, e) not in have]
            if not new:
                return False
            exp.feedback += [{"run_id": run_id, "event": e, "at": _now()} for e in new]
            exp.utility = utility(exp)
            exp.updated_at = _now()
            self.store.write_entry(exp, "experiences")
        if "adopted" in events:
            self.store.record_reuse(exp_id, run_id, success="resolved" in events and "rollback" not in events)
        return True

    def retire_low_utility(self, threshold: float, *, min_runs: int = 2) -> list[str]:
        retired = []
        with self._lock():
            for loc, exp in self.store.lifecycle_entries():
                if loc != "experiences" or exp.status != "approved":
                    continue
                adopted = len({f["run_id"] for f in exp.feedback if f.get("event") == "adopted"})
                if adopted >= min_runs and utility(exp) < threshold:
                    exp.status = exp.review.status = "retired"
                    exp.history.append(
                        self._hist(
                            "RETIRE", "", f"utility {utility(exp):.2f} < {threshold} after {adopted} runs"
                        )
                    )
                    self.store.write_entry(exp, "experiences")
                    retired.append(exp.id)
        return retired


def utility(exp: Experience) -> float:
    runs = {e: {f["run_id"] for f in exp.feedback if f.get("event") == e} for e in FEEDBACK_EVENTS}
    return round((len(runs["resolved"]) - len(runs["rollback"])) / (len(runs["adopted"]) + 1), 4)
