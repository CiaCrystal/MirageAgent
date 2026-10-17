"""PlanState:计划从一段静态文本变成可跟踪的状态。参考 Claude Code TodoWrite、Codex update_plan。

    state = PlanState.from_static(plan, report, runtime_usages, is_protected=..., repo_files=...)
    changes = state.observe(report, changed_files, locked, repo_files)   # on_test_report 钩子调用
    applied, rejected = state.apply_model_update(ops, file_ok)          # update_plan 工具
    state.apply_replan(ops)                                              # replanner 输出(已校验)
    state.render()                                                       # before_llm 注入的计划块

骨架来源(origin):static(Planner 的文件计划)/ runtime_usage(运行时弃用警告补上的文件)/
report(初始报告里静态计划覆盖不到的失败根因)/ cascade(改了文件之后冒出的新签名)/ replan / model。

程序自动更新(observe,只看全量测试报告):
    - 项的关联签名全部消失 → done;done 项的签名重新出现 → 重新 active;只认在报告里真实出现过的签名,
      从未出现过的签名(模型编写的文本)当作没有关联签名处理;
    - 没有关联签名的文件项:文件被改过且当前失败的帧都不落在这些文件里 → done;
    - 改了某项的文件而其签名仍在 → attempts + 1、active;
    - 改了文件之后出现的新签名 → 牵连错误(cascade),挂到帧/改动文件对应的项下,找不到对应项时新建 cascade 项;
    - 项的文件全部被策略控制锁定 → abandoned。
模型通过 update_plan 只能新增、拆分、补充说明、改为 active / blocked / pending;标 done 时若关联签名仍在失败则拒绝;
不能删除项,也不能标 abandoned(由程序决定)。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from .strategy import source_frames

STATUSES = ("pending", "active", "done", "blocked", "abandoned")
MODEL_STATUSES = ("pending", "active", "blocked", "done")


@dataclass
class PlanItem:
    id: str
    goal: str
    files: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    approach: str = ""
    avoid: list[str] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    depends_on: list[str] = field(default_factory=list)
    origin: str = "static"
    category: str = ""
    notes: list[str] = field(default_factory=list)
    cascade: list[str] = field(default_factory=list)  # 归到本项的牵连签名(也在 signatures 里)


def _groups(report: Any) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for f in getattr(report, "failures", None) or []:
        out.setdefault(f.signature, []).append(f)
    return out


@dataclass
class PlanState:
    items: list[PlanItem] = field(default_factory=list)
    version: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    is_protected: Callable[[str], bool] = field(default=lambda _p: False, repr=False)
    seen: set[str] = field(default_factory=set)  # 出现过的全部签名
    failing: set[str] = field(default_factory=set)  # 最近一份全量报告中的签名
    edited: set[str] = field(default_factory=set)  # 本 run 改过的文件
    uncovered_initial: list[str] = field(default_factory=list)  # 静态计划覆盖不到的初始签名
    _n: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_static(
        cls,
        plan: Any,
        report: Any,
        runtime_usages: Iterable[Any] = (),
        *,
        is_protected: Callable[[str], bool] | None = None,
        repo_files: set[str] | None = None,
    ) -> PlanState:
        st = cls(is_protected=is_protected or (lambda _p: False))
        for fp in getattr(plan, "files", None) or []:
            sym = ", ".join(s.split(".")[-1] for s, _ in fp.symbols.most_common(4))
            st._add(f"Migrate deprecated API usages in {fp.path} ({sym})", [fp.path], origin="static")
        by_file: dict[str, list[Any]] = {}
        for u in runtime_usages:
            by_file.setdefault(u.file, []).append(u)
        for path, us in sorted(by_file.items()):
            item = st._item_for_file(path)
            names = sorted({u.symbol for u in us})
            note = "runtime deprecation warnings: " + ", ".join(f"L{u.line} {u.symbol}" for u in us[:6])
            if item is None:
                item = st._add(
                    f"Replace deprecated calls reported at runtime in {path} ({', '.join(names[:4])})",
                    [path],
                    origin="runtime_usage",
                )
            item.notes.append(note)
        for sig, fs in _groups(report).items():
            frames = st._frames(fs, repo_files)
            item = next((it for p in frames if (it := st._item_for_file(p)) is not None), None)
            if item is None:
                st.uncovered_initial.append(sig)
                item = st._add(
                    f"Fix failing root cause ({len(fs)} test(s)): {sig[:160]}", frames[:3], "report"
                )
            item.signatures.append(sig)
        st.failing = set(_groups(report))
        st.seen = set(st.failing)
        st.version = 1
        st.history.append({"version": 1, "source": "static", "items": len(st.items)})
        return st

    def _add(self, goal: str, files: list[str], origin: str, **kw: Any) -> PlanItem:
        self._n += 1
        item = PlanItem(id=f"p{self._n}", goal=goal, files=list(files), origin=origin, **kw)
        self.items.append(item)
        return item

    def _item_for_file(self, path: str) -> PlanItem | None:
        return next((it for it in self.items if path in it.files and it.status != "abandoned"), None)

    def _frames(self, failures: list[Any], repo_files: set[str] | None) -> list[str]:
        out: list[str] = []
        for f in failures[:5]:
            for p in source_frames(getattr(f, "traceback", "") or "", self.is_protected, repo_files):
                if p not in out:
                    out.append(p)
        return out

    def get(self, item_id: str) -> PlanItem | None:
        return next((it for it in self.items if it.id == item_id), None)

    # ------------------------------------------------------------------ program updates
    def observe(
        self,
        report: Any,
        changed_files: Iterable[str] = (),
        locked: Iterable[str] = (),
        repo_files: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """一份全量测试报告 → 状态变更列表(写 trace);调用方保证只传全量报告。"""
        if getattr(report, "parse_error", None):
            return []
        changed = set(changed_files)
        self.edited |= changed
        groups = _groups(report)
        failing = set(groups)
        failing_frames = {p for fs in groups.values() for p in self._frames(fs, repo_files)}
        changes: list[dict[str, Any]] = []

        def mark(item: PlanItem, status: str, why: str) -> None:
            if item.status != status:
                changes.append({"id": item.id, "from": item.status, "to": status, "why": why})
                item.status = status

        for sig in sorted(failing - self.seen):
            frames = self._frames(groups[sig], repo_files)
            if changed:
                hit = [p for p in frames if p in changed] or sorted(changed)
                item = next((it for p in hit if (it := self._item_for_file(p)) is not None), None)
                if item is None:
                    item = self._add(
                        f"Fix new failure after edits ({len(groups[sig])} test(s)): {sig[:160]}",
                        frames[:3] or hit[:1],
                        "cascade",
                    )
                    changes.append({"id": item.id, "added": "cascade", "signature": sig[:200]})
                item.cascade.append(sig)
                changes.append({"id": item.id, "cascade": sig[:200]})
            else:
                item = self._add(
                    f"Fix failing root cause ({len(groups[sig])} test(s)): {sig[:160]}", frames[:3], "report"
                )
                changes.append({"id": item.id, "added": "report", "signature": sig[:200]})
            item.signatures.append(sig)

        locked = set(locked)
        observed = self.seen | failing
        for it in self.items:
            if it.status == "abandoned":
                continue
            if it.files and set(it.files) <= locked:
                mark(it, "abandoned", "all files locked by the strategy controller")
                continue
            # 只有在某份报告里真实出现过的签名才算关联失败;从未出现过的(如模型编写的文本)不能因「消失」判 done
            real = [s for s in it.signatures if s in observed]
            live = [s for s in real if s in failing]
            touched = bool(set(it.files) & changed)
            if real:
                if not live:
                    mark(it, "done", "all linked failures resolved")
                elif touched:
                    it.attempts += 1
                    mark(it, "active", "edited; linked failures remain")
                elif it.status == "done":
                    mark(it, "active", "linked failure reappeared")
            elif set(it.files) & self.edited:
                if set(it.files) & failing_frames:
                    mark(it, "active", "edited; failures still point at these files")
                else:
                    mark(it, "done", "edited and no failure points at these files")
        self.seen |= failing
        self.failing = failing
        if changes:
            self.version += 1
            self.history.append({"version": self.version, "source": "tests", "changes": changes})
        return changes

    # ------------------------------------------------------------------ model updates
    def apply_model_update(
        self, ops: list[dict[str, Any]], file_ok: Callable[[str], str] | None = None
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        """update_plan:逐条执行,返回 (已执行, [(被拒操作, 原因)])。`file_ok(path)` 返回拒绝原因或空串。"""
        applied: list[dict[str, Any]] = []
        rejected: list[tuple[dict[str, Any], str]] = []
        for op in ops or []:
            why = self._model_op(op, file_ok)
            (rejected.append((op, why)) if why else applied.append(op))
        if applied:
            self.version += 1
            self.history.append({"version": self.version, "source": "model", "ops": applied})
        return applied, rejected

    def _check_files(self, files: Any, file_ok) -> tuple[list[str], str]:
        if files is None:
            return [], ""
        if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
            return [], "files must be a list of paths"
        for f in files:
            why = file_ok(f) if file_ok else ""
            if why:
                return [], f"{f}: {why}"
        return list(files), ""

    def _model_op(self, op: dict[str, Any], file_ok) -> str:
        kind = op.get("op")
        if kind == "add":
            if not str(op.get("goal") or "").strip():
                return "add needs a goal"
            files, why = self._check_files(op.get("files"), file_ok)
            if why:
                return why
            self._add(str(op["goal"])[:300], files, "model")
            return ""
        item = self.get(str(op.get("id") or ""))
        if item is None:
            return f"unknown plan item {op.get('id')!r}"
        if kind == "split":
            parts = op.get("parts") or []
            if not isinstance(parts, list) or not parts:
                return "split needs parts=[{goal, files}]"
            checked = []
            for part in parts:
                if not isinstance(part, dict) or not str(part.get("goal") or "").strip():
                    return "every part needs a goal"
                files, why = self._check_files(part.get("files"), file_ok)
                if why:
                    return why
                checked.append((str(part["goal"])[:300], files))
            for goal, files in checked:
                self._add(goal, files, "model", depends_on=list(item.depends_on))
            item.notes.append(f"split into {len(checked)} sub-items")
            return ""
        if kind == "set_status":
            status = op.get("status")
            if status not in MODEL_STATUSES:
                return f"status must be one of {MODEL_STATUSES}"
            if item.status == "abandoned":
                return "abandoned by the program; cannot be reopened"
            live = [s for s in item.signatures if s in self.failing]
            if status == "done" and live:
                return f"{len(live)} linked failure(s) still fail, e.g. `{live[0][:120]}`; the program marks it done when they pass"
            item.status = status
            return ""
        if kind == "note":
            text = str(op.get("note") or "").strip()
            if not text:
                return "note is empty"
            item.notes.append(text[:300])
            return ""
        return f"unknown op {kind!r}; use add / split / set_status / note"

    # ------------------------------------------------------------------ replanner
    def apply_replan(self, ops: list[dict[str, Any]], *, source: str = "replan") -> list[str]:
        """replanner 的输出(已由 Replanner 校验)。返回新增/修改的项 id。"""
        touched: list[str] = []
        for op in ops:
            kind = op.get("op")
            fields = {
                "approach": str(op.get("approach") or "")[:600],
                "avoid": [str(a)[:200] for a in op.get("avoid") or []][:6],
            }
            item = self.get(str(op.get("id") or "")) if op.get("id") else None
            if kind in ("add", "split") or item is None:
                new = self._add(
                    str(op.get("goal") or "")[:300],
                    list(op.get("files") or []),
                    source,
                    signatures=[s for s in op.get("signatures") or [] if isinstance(s, str)],
                    category=str(op.get("category") or ""),
                    **fields,
                )
                new.status = "active"
                if kind == "split" and item is not None:
                    new.depends_on = list(item.depends_on)
                    item.notes.append(f"split by replanner into {new.id}")
                touched.append(new.id)
                continue
            if kind == "update":
                item.goal = str(op.get("goal") or item.goal)[:300]
                if op.get("files"):
                    item.files = list(op["files"])
                item.approach = fields["approach"] or item.approach
                item.avoid = list(dict.fromkeys(item.avoid + fields["avoid"]))[:8]
                if item.status in ("pending", "blocked"):
                    item.status = "active"
            elif kind == "block":
                item.status = "blocked"
                item.notes.append(fields["approach"] or "blocked by replanner")
            elif kind == "abandon":
                item.status = "abandoned"
                item.notes.append(fields["approach"] or "abandoned by replanner")
            touched.append(item.id)
        if touched:
            self.version += 1
            self.history.append({"version": self.version, "source": source, "items": touched})
        return touched

    # ------------------------------------------------------------------ views
    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for it in self.items:
            out[it.status] += 1
        return out

    def completion_rate(self) -> float:
        return self.counts()["done"] / len(self.items) if self.items else 0.0

    def plan_files(self) -> set[str]:
        return {f for it in self.items for f in it.files}

    def render(self, max_chars: int = 2400) -> str:
        c = self.counts()
        lines = [
            f"[plan v{self.version}] {c['done']}/{len(self.items)} done, {c['active']} active, {c['pending']} pending"
            + (f", {c['blocked']} blocked" if c["blocked"] else "")
            + (f", {c['abandoned']} abandoned" if c["abandoned"] else "")
            + ". Statuses follow the test results; use `update_plan` to add, split or annotate items."
        ]
        order = {"active": 0, "pending": 1, "blocked": 2}
        for it in sorted(
            (i for i in self.items if i.status in order), key=lambda i: (order[i.status], int(i.id[1:]))
        ):
            files = f" [{', '.join(it.files[:3])}{' …' if len(it.files) > 3 else ''}]" if it.files else ""
            line = f"- {it.id} ({it.status}{', ' + it.category if it.category else ''}"
            line += f", {it.attempts} attempt(s)" if it.attempts else ""
            line += f"): {it.goal}{files}"
            if it.approach:
                line += f"\n    approach: {it.approach}"
            if it.avoid:
                line += "\n    avoid: " + "; ".join(it.avoid[:4])
            lines.append(line)
        done = [i.id for i in self.items if i.status == "done"]
        if done:
            lines.append("- done: " + ", ".join(done))
        gone = [i.id for i in self.items if i.status == "abandoned"]
        if gone:
            lines.append("- abandoned: " + ", ".join(gone))
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 2] + " …"

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "items": [asdict(i) for i in self.items]}
