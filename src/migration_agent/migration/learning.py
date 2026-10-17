"""Candidate extraction and experience summarization after a migration run."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from ..migration.quality import changes_issues
from ..rag.memory.extractor import extract
from ..rag.memory.schema import SourceRef
from ..trace.analyze import load


def canonical_repo(url: str) -> str:
    m = re.match(r"(?:https?://|git@)([^/:]+)[/:](.+?)(?:\.git)?/?$", url.strip())
    return (f"{m.group(1)}/{m.group(2)}" if m else url).lower()


def run_facts(trace_path: Path | str, result: Any) -> dict[str, Any]:
    """抽取需要的 run 级事实:结果、兼容层计数、发生回滚的步骤、每步调用的工具名。"""
    rollback_steps: set[int] = set()
    tools_by_step: dict[int, list[str]] = {}
    for ev in load(trace_path):
        if ev.kind == "rollback":
            rollback_steps.add(ev.step)
        if ev.kind == "strategy" and ev.payload.get("action") == "rollback" and ev.payload.get("rolled_back"):
            rollback_steps.add(ev.step)
        if ev.kind == "tool_call":
            tools_by_step.setdefault(ev.step, []).append(ev.payload.get("name", ""))
    r = result if isinstance(result, dict) else result.model_dump(mode="json")
    return {
        "run_id": r.get("run_id"),
        "success": bool(r.get("success")),
        "reason": str(r.get("reason")),
        "compatibility_import_count": r.get("compatibility_import_count", 0),
        "rollback_steps": sorted(rollback_steps),
        "tools_by_step": tools_by_step,
    }


def extract_into(
    store: Any,
    evidence: dict,
    facts: dict,
    source: SourceRef,
    *,
    is_protected: Callable[[str], bool],
    policy: Any = None,
) -> dict[str, Any]:
    """抽取候选并导入库的候选区;返回计数(created/merged/unchanged × candidate/rejected)与 id 列表。"""
    cands = extract(
        evidence,
        facts,
        source,
        is_protected=is_protected,
        change_issues=partial(changes_issues, is_protected=is_protected),
        policy=policy,
    )
    actions: Counter = Counter()
    ids: list[str] = []
    for exp in cands:
        exp_id, action = store.add_candidate(exp)
        actions[f"{exp.status}:{action}"] += 1
        ids.append(exp_id)
    return {
        "windows": len(cands),
        "auto_passed": sum(c.review.auto_passed for c in cands),
        "rejected": sum(c.status == "rejected" for c in cands),
        "uncertain": sum(c.status == "candidate" and not c.review.auto_passed for c in cands),
        "actions": dict(actions),
        "ids": ids,
    }


def dev_source(repo: Path, run_id: str, trace: str, evidence: str) -> SourceRef:
    """运行来源:git remote/commit;无 remote 时使用目录名。"""

    def git(*args: str) -> str:
        res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else ""

    url = git("remote", "get-url", "origin")
    name = canonical_repo(url) if url else f"local/{repo.name.lower()}"
    return SourceRef(
        repo=name,
        family=f"dev:{name}",
        commit=git("rev-parse", "HEAD"),
        run_id=run_id,
        case=repo.name,
        group="dev",
        trace=trace,
        evidence=evidence,
    )


# ====================================================================== LLM 经验总结(外层编排)


class LearningSourcePolicy:
    """Accept experience sources without repository or run grouping restrictions."""

    def __call__(self, source: SourceRef) -> tuple[bool, str]:
        return True, "allowed"


def router_channel(router: Any, role: str) -> Any:
    """ModelRouter 的一个角色 → rag 层的 LLMChannel(同一个 router、同一个密钥与客户端;按角色记账)。"""
    from ..rag.memory.llm_budget import LLMChannel, RoleMeter

    def complete(prompt: str, schema: dict, *, max_tokens: int | None = None, retries: int = 1) -> dict:
        # 附加调用关闭 thinking:输出上限(总结 1.5k / 审核 500)只给 JSON 用,避免推理 token 截断回复
        return router.complete_json(
            role, prompt, schema, max_tokens=max_tokens, retries=retries, no_thinking=True
        )

    def meter() -> RoleMeter:
        u = router.usage().get(role)
        if u is None:
            return RoleMeter()
        return RoleMeter(u.calls, u.tokens_in + u.cache_read + u.cache_write, u.tokens_out, u.cost)

    return LLMChannel(role, complete, meter, model=getattr(router.roles.get(role), "model", ""))


def run_cost(events: list[dict], pricing: Any) -> float | None:
    """run 的估算成本:执行回合(llm_call)+ router 账目里其他角色(planner 等)。没有价格表返回 None。"""
    if pricing is None:
        return None
    t_in = c_r = c_w = t_out = 0
    other = 0.0
    for ev in events:
        if ev["kind"] == "llm_call":
            p = ev["payload"]
            t_in += int(ev.get("tokens_in", 0) or 0)
            t_out += int(ev.get("tokens_out", 0) or 0)
            c_r += int(p.get("cache_read", 0) or 0)
            c_w += int(p.get("cache_write", 0) or 0)
        elif ev["kind"] == "router":
            for role, u in ev["payload"].items():
                if role != "executor" and isinstance(u, dict):
                    other += float(u.get("cost", 0) or 0)
    return pricing.cost(t_in, c_r, c_w, t_out) + other


def trace_events(path: Path | str) -> list[dict]:
    return [
        {
            "kind": e.kind,
            "step": e.step,
            "payload": e.payload,
            "tokens_in": e.tokens_in,
            "tokens_out": e.tokens_out,
        }
        for e in load(path)
    ]


def summarize_run(
    store: Any,
    *,
    events: list[dict],
    source: SourceRef,
    policy: Callable[[SourceRef], tuple[bool, str]],
    forbidden: Callable[[str], str],
    run_cost_usd: float | None,
    price_in: float,
    price_out: float,
    summarizer_channel: Any = None,
    critic_channel: Any = None,
    evidence: dict | None = None,
    facts: dict | None = None,
    is_protected: Callable[[str], bool] | None = None,
    online: bool = True,
    thresholds: dict[str, int] | None = None,
    limits: Any = None,
    windows: list | None = None,
) -> dict[str, Any]:
    """一个 run 的经验总结管线:规则定位窗口 → summarizer → critic → consolidate。返回写进 trace(memory_summary)的统计。

    `windows` 可直接给定；否则从 evidence(fix)与 trace 事件(其他类)定位并按优先级
    取前 max_summaries 个。花费上限、调用次数与 token 上限见 rag/memory/llm_budget.py;附加调用不进入主循环的预算。"""
    from ..rag.memory.consolidate import Consolidator
    from ..rag.memory.critic import Critic
    from ..rag.memory.llm_budget import AuxBudget, AuxLimits
    from ..rag.memory.summarizer import ExperienceSummarizer
    from ..rag.memory.windows import event_windows, fix_windows, select

    limits = limits or AuxLimits()
    allowed, why = policy(source)
    found = list(windows or [])
    if windows is None:
        if evidence is not None and facts is not None and is_protected is not None:
            cands = extract(
                evidence,
                facts,
                source,
                is_protected=is_protected,
                change_issues=partial(changes_issues, is_protected=is_protected),
            )
            found += fix_windows(cands)
        found += event_windows(events, source, source_allowed=allowed)
        windows = select(found, limits.max_summaries)
    budget = AuxBudget(limits, run_cost=run_cost_usd, price_in=price_in, price_out=price_out)
    summarizer = ExperienceSummarizer(summarizer_channel, budget, source_allowed=policy)
    critic = Critic(critic_channel, budget, forbidden=forbidden, source_allowed=policy, thresholds=thresholds)
    cons = Consolidator(
        store, critic_channel, budget, online=online, run_id=source.run_id, source_allowed=policy
    )
    by_ref = {w.ref: w for w in windows}
    rows = []
    for out in summarizer.summarize(windows):
        row = out.trace()
        if out.entry is not None:
            verdict = critic.review(out.entry, {out.ref: by_ref[out.ref]})
            exp_id, op, detail = cons.ingest(out.entry, verdict.decision)
            row.update(verdict=verdict.trace(), op=op, id=exp_id, detail=detail[:200], kind=out.entry.kind)
        rows.append(row)
    return {
        "run_id": source.run_id,
        "source_allowed": allowed,
        "source_policy": why,
        "windows_found": len(found),
        "windows_by_origin": dict(Counter(w.origin for w in found)),
        "windows_selected": [w.ref for w in windows],
        "restricted": summarizer.restricted,
        "results": rows,
        "ops": dict(Counter(r.get("op", r["status"]) for r in rows)),
        "budget": budget.summary(),
        "calls": budget.log,
        "thresholds": thresholds,
        "online": online,
    }
