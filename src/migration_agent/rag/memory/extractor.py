"""规则抽取经验候选:在「报错 → 修改 → 测试变化」的局部窗口里定位修复,输出 candidate,不自动发布。

    candidates = extract(evidence, run, source, is_protected=..., policy=...)

`change_issues(before, after)` 由调用方注入(通常是 `migration.quality.changes_issues` 绑定 is_protected),
使 rag 层不依赖上层的迁移领域代码。输入是 `migration.evidence.EvidenceRecorder` 的证据(相邻两次全量测试之间的真实文件改动 + 逐用例失败签名),
`run` 是该 run 的结果摘要(success / reason / compatibility_import_count / rollback_steps / tools_by_step),
`source` 是 SourceRef 模板(仓库规范身份、家族、commit、run、trace/证据路径)。不调用模型。

窗口:每两次相邻的全量测试 (R_{i-1}, R_i) 之间若有文件改动,R_{i-1} 中某签名在 R_i 中完全消失且其用例
至少一个在 R_i 不再失败,则该签名是窗口的候选修复对象。**通过数上升本身不算证据**。

归因:窗口内被改的源文件中,出现在该签名 traceback 帧里的文件作为最小 diff;没有帧命中且只改了一个文件时
按单一改动归因;否则保留整个窗口的组合 diff 并标记归因不清。文件内有多个改动块时,若报错带 API 标签
(如 `@validator`、`BaseSettings`),只保留改动行涉及同族 API 的块(schema.related_tags);报错没有 API 标签、
无法收窄时保留全部块并标记归因不清,留给人工审核。窗口内全部改动都要过质量检查,不能只看被归因的文件。

检查分两级(结果写入 review.auto_checks):
  - 硬性(任一不满足 → rejected):来源允许、run 最终严格合格(成功、无兼容层、最终产出无库补丁/改测试/降依赖/
    skip/xfail)、窗口改动干净、改动保留到最终产出(未被回滚/撤销)、签名到最终测试仍不存在;
  - 证据充分性(任一不满足 → 留在候选区并写明 uncertain,不算通过自动检查):至少一个用例由失败变通过、
    归因清楚、窗口内无回滚事件、改动不全来自 shell 命令(diff 仍按快照准确重建)、最小 diff 不超过上限、
    最小 diff 自包含(不引用之前窗口才新定义的名字)。收窄 API 块时,同窗口中定义了被保留块所用名字的块一并保留。
"""

from __future__ import annotations

import ast
import difflib
import re
from collections.abc import Callable
from typing import Any

from .schema import (
    Experience,
    Review,
    SourceRef,
    TestEvidence,
    api_names,
    api_tags_in,
    changed_lines,
    error_api_tags,
    experience_id,
    related_tags,
)

MAX_DIFF_LINES = 80
CONTEXT = 2
HARD_CHECKS = (
    "source_allowed",
    "run_strict_success",
    "run_no_compat_layer",
    "final_output_clean",
    "window_clean",
    "retained_in_final",
    "signature_absent_at_end",
)
SOFT_CHECKS = (
    "tests_flipped_to_pass",
    "attribution_clear",
    "no_rollback_in_window",
    "edited_by_edit_tools",
    "diff_within_limit",
    "self_contained",
)
EDIT_TOOLS = ("edit_file", "write_file", "multi_edit")


def _groups(report: dict) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in report.get("failures", []):
        g = out.setdefault(f["signature"], {"tests": [], "frames": [], "example": None})
        g["tests"].append(f["test_id"])
        for fr in f.get("frames", []):
            if fr not in g["frames"]:
                g["frames"].append(fr)
        if g["example"] is None and f.get("traceback") is not None:
            g["example"] = f
    for g in out.values():
        if g["example"] is None:
            g["example"] = next(f for f in report["failures"] if f["test_id"] == g["tests"][0])
    return out


def _unified(rel: str, before: str, after: str, n: int = CONTEXT) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), fromfile=f"a/{rel}", tofile=f"b/{rel}", n=n, lineterm=""
        )
    )
    return "\n".join(lines)


def _code_before(diff: str) -> str:
    out = []
    for ln in diff.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith("@@"):
            if out:
                out.append("# ...")
            continue
        if ln.startswith((" ", "-")):
            out.append(ln[1:])
    return "\n".join(out).strip("\n")


_PATHS = re.compile(r"(?:[A-Za-z]:\\|/)[^\s:'\"]+")


def _error_text(example: dict) -> str:
    tb = example.get("traceback") or ""
    e_lines = [ln[1:].strip() for ln in tb.splitlines() if ln.startswith("E ")]
    text = "\n".join(e_lines[:8]) if e_lines else (example.get("message") or tb[-600:])
    text = _PATHS.sub("<path>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0x?", text)
    return text[:800].strip()


def _hunks(file_diff: str) -> tuple[list[str], list[list[str]]]:
    lines = file_diff.splitlines()
    hunks: list[list[str]] = []
    for ln in lines[2:]:
        if ln.startswith("@@"):
            hunks.append([ln])
        elif hunks:
            hunks[-1].append(ln)
    return lines[:2], hunks


def _changed(hunk: list[str], *, removed: bool = True) -> list[str]:
    marks = ("+", "-") if removed else ("+",)
    return [ln[1:] for ln in hunk[1:] if ln.startswith(marks)]


def _relevant(hunk: list[str], related: set[str]) -> bool:
    return bool(related) and bool(related & set(api_tags_in("\n".join(_changed(hunk)))))


_DEF = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)|^([A-Za-z_]\w*)\s*(?::[^=\n]+)?=(?!=)", re.M
)
_NAME = re.compile(r"[A-Za-z_]\w*")


_IMPORT = re.compile(r"^\s*(?:from\s+[\w.]+\s+)?import\s+\(?([^#\n]*)", re.M)


def _defined(text: str) -> set[str]:
    """片段(diff 新增行)里定义或导入的名字:def/class/模块级赋值 + 单行 import 的绑定名(含 as 别名)。"""
    out = {a or b for a, b in _DEF.findall(text)}
    for names in _IMPORT.findall(text):
        for part in names.replace(")", "").split(","):
            bits = part.split()
            if bits:
                out.add((bits[-1] if len(bits) == 3 and bits[1] == "as" else bits[0]).split(".")[0])
    return out


def _bound(source: str) -> set[str]:
    """整个文件绑定的名字:def/class/模块级赋值 + import(含 from-import 的别名)。解析失败时退回正则。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _defined(source)
    out = _defined(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return out


def _referenced(lines: list[str]) -> set[str]:
    return {n for ln in lines for n in _NAME.findall(ln.split("#", 1)[0])}


def _with_helpers(kept: list[list[str]], hunks: list[list[str]]) -> list[list[str]]:
    """收窄后补回同一窗口中定义了被保留块所引用名字的块(如新增的 helper 函数),避免最小 diff 不完整。"""
    out = list(kept)
    changed = True
    while changed:
        changed = False
        refs = _referenced([ln for h in out for ln in _changed(h, removed=False)])
        for h in hunks:
            if h not in out and _defined("\n".join(_changed(h, removed=False))) & refs:
                out.append(h)
                changed = True
    return [h for h in hunks if h in out]


def _retained(added: list[str], final: str | None) -> bool:
    """改动保留到最终产出:被归因块新增的每一行(去空白)都还在最终文件里。"""
    if final is None:
        return False
    final_lines = {ln.strip() for ln in final.splitlines()}
    return all(ln.strip() in final_lines for ln in added if ln.strip())


def extract(
    evidence: dict,
    run: dict,
    source: SourceRef,
    *,
    is_protected: Callable[[str], bool],
    change_issues: Callable[[dict[str, str], dict[str, str]], list[str]],
    policy: Any = None,
    target: str | None = None,
) -> list[Experience]:
    target = target or _target().name
    reports = [r for r in evidence.get("reports", []) if not r.get("subset") and "record_error" not in r]
    if len(reports) < 2:
        return []
    initial = evidence.get("initial_changed", {})
    final = evidence.get("final_changed", {})
    final_report = reports[-1]
    final_sigs = {f["signature"] for f in final_report.get("failures", [])}
    final_failing = {f["test_id"] for f in final_report.get("failures", [])}
    rollback_steps = set(run.get("rollback_steps", []))
    tools_by_step = {int(k): v for k, v in (run.get("tools_by_step") or {}).items()}

    source_ok, source_why = (True, "") if policy is None else policy.allows_source(source)
    final_issues = change_issues(initial, final)
    run_checks = {
        "source_allowed": source_ok,
        "run_strict_success": bool(run.get("success")) and run.get("reason") == "tests_passed",
        "run_no_compat_layer": not run.get("compatibility_import_count"),
        "final_output_clean": not final_issues,
    }
    from ...migration.evidence import version_key

    observed = evidence.get(version_key()) or ""
    out: list[Experience] = []
    introduced: dict[str, int] = {}  # 之前窗口新定义的名字 → 步骤(检查最小 diff 是否依赖更早的改动)
    for prev, cur in zip(reports, reports[1:], strict=False):
        changes = cur.get("changes") or {}
        if not changes:
            continue
        earlier = dict(introduced)
        for rel, c in changes.items():
            if rel.endswith(".py"):
                for name in _bound(c["after"]) - _bound(c["before"]):
                    introduced.setdefault(name, cur["step"])
        before_all = {rel: c["before"] for rel, c in changes.items()}
        after_all = {rel: c["after"] for rel, c in changes.items()}
        window_issues = change_issues(before_all, after_all)
        touched = [rel for rel in changes if rel.endswith(".py") and not is_protected(rel)]
        g_before, g_after = _groups(prev), _groups(cur)
        failing_after = {f["test_id"] for f in cur.get("failures", [])}
        steps = list(range(prev["step"] + 1, cur["step"] + 1)) or [cur["step"]]
        window_tools = [t for s in steps for t in tools_by_step.get(s, [])]
        resolved_sigs = [s for s in g_before if s not in g_after]
        for sig in resolved_sigs:
            g = g_before[sig]
            flipped = [t for t in g["tests"] if t not in failing_after]
            attributed = [rel for rel in touched if rel in g["frames"]]
            if attributed:
                files, attribution = attributed, "traceback frame"
            elif len(touched) == 1:
                files, attribution = touched, "single changed file"
            else:
                files, attribution = touched or sorted(changes), "unclear"
            err = _error_text(g["example"])
            related = related_tags(error_api_tags(sig + "\n" + err))
            per_file, n_hunks, n_kept = {}, 0, 0
            for rel in files:
                header, hunks = _hunks(_unified(rel, changes[rel]["before"], changes[rel]["after"]))
                kept = [h for h in hunks if _relevant(h, related)]
                if kept:
                    kept = _with_helpers(kept, hunks)
                n_hunks += len(hunks)
                n_kept += len(kept)
                per_file[rel] = (header, kept or hunks, bool(kept))
            if n_kept and len(per_file) > 1:  # 收窄后,不含相关块的文件不再算进最小 diff
                per_file = {rel: v for rel, v in per_file.items() if v[2]}
            files = list(per_file)
            if n_kept:  # 每个保留块都按 API 同族或作为其 helper 选出
                attribution += " + API-tag hunks" if n_kept < n_hunks else " (all hunks API-related)"
            elif n_hunks > 1 and not attribution.startswith("unclear"):
                attribution = "unclear (several hunks, no API tag to narrow)"
            diff = "\n".join(
                "\n".join(head + [ln for h in hs for ln in h]) for head, hs, _ in per_file.values()
            )
            removed, added = changed_lines(diff)
            removed_txt, added_txt = "\n".join(removed), "\n".join(added)
            api_removed = sorted(
                set(api_tags_in(removed_txt, v2=False)) - set(api_tags_in(added_txt, v2=False))
            )
            api_added = sorted(
                set(api_tags_in(added_txt, v1=False)) - set(api_tags_in(removed_txt, v1=False))
            )
            checks = dict(run_checks)
            checks["window_clean"] = not window_issues
            checks["retained_in_final"] = all(
                _retained([ln for h in per_file[rel][1] for ln in _changed(h, removed=False)], final.get(rel))
                for rel in files
            )
            checks["signature_absent_at_end"] = sig not in final_sigs and not (set(flipped) & final_failing)
            checks["tests_flipped_to_pass"] = bool(flipped)
            checks["attribution_clear"] = not attribution.startswith("unclear")
            checks["no_rollback_in_window"] = not (rollback_steps & set(steps))
            checks["edited_by_edit_tools"] = any(t in EDIT_TOOLS for t in window_tools) or not window_tools
            n_diff = sum(1 for ln in diff.splitlines() if not ln.startswith(("+++", "---")))
            checks["diff_within_limit"] = 0 < n_diff <= MAX_DIFF_LINES
            deps = {
                n: earlier[n]
                for n in sorted(_referenced(added) - _defined("\n".join(added)) - api_names())
                if n in earlier
            }
            checks["self_contained"] = not deps
            hard_fail = [k for k in HARD_CHECKS if not checks[k]]
            soft_fail = [k for k in SOFT_CHECKS if not checks[k]]
            details = {
                "attribution": attribution,
                "window_files": sorted(changes),
                "window_tools": window_tools,
                "window_issues": window_issues,
                "final_output_issues": final_issues,
                "source_policy": source_why,
                "diff_lines": n_diff,
                "hunks_in_files": n_hunks,
                "hunks_kept": n_kept or n_hunks,
                "depends_on_earlier_definitions": deps,
            }
            uncertain = []
            if not checks["tests_flipped_to_pass"]:
                uncertain.append(
                    "signature disappeared but its tests still fail with another error (transformed)"
                )
            if not checks["attribution_clear"]:
                uncertain.append(f"attribution {attribution}; combined diff kept as evidence")
            if not checks["no_rollback_in_window"]:
                uncertain.append("a rollback happened inside the window")
            if not checks["edited_by_edit_tools"]:
                uncertain.append("change made by a shell command; diff rebuilt from file snapshots")
            if not checks["self_contained"]:
                uncertain.append(
                    "diff uses names defined in an earlier window (not shown): "
                    + ", ".join(f"{n}@step{s}" for n, s in deps.items())
                )
            if not checks["diff_within_limit"]:
                uncertain.append(f"minimal diff has {n_diff} lines (limit {MAX_DIFF_LINES})")
            if len(resolved_sigs) > 1:
                details["window_resolved_signatures"] = len(resolved_sigs)
            review = Review(
                status="rejected" if hard_fail else "candidate",
                reason=("auto-rejected: " + ", ".join(hard_fail)) if hard_fail else "",
                reviewer="rules" if hard_fail else "",
                auto_checks={**checks, "details": details},
                auto_passed=not hard_fail and not soft_fail,
                uncertain=uncertain,
            )
            src = source.model_copy(update={"steps": steps, "files": files})
            exp = Experience(
                id="pending",
                target=target,
                versions={**target_versions(), **({"observed": observed} if observed else {})},
                error_signature=sig,
                error_type=sig.split(":", 1)[0].strip() if ":" in sig else "",
                error_text=err,
                api_tags=error_api_tags(sig + "\n" + err),
                api_removed=api_removed,
                api_added=api_added,
                preconditions=_preconditions(api_removed, observed, attribution),
                not_applicable=_not_applicable(api_added),
                code_before=_code_before(diff),
                diff=diff,
                explanation=_explanation(api_removed, api_added, files, flipped, prev, cur),
                combined=len(files) > 1 or diff.count("\n@@") > 0,
                sources=[src],
                test_evidence=[
                    TestEvidence(
                        run_id=source.run_id,
                        step_before=prev["step"],
                        step_after=cur["step"],
                        passed_before=prev["passed"],
                        passed_after=cur["passed"],
                        total_after=cur["total"],
                        signature_tests_before=len(g["tests"]),
                        resolved_tests=flipped[:5],
                        new_signatures_after=len([s for s in g_after if s not in g_before]),
                        final_passed=final_report["passed"],
                        final_total=final_report["total"],
                        signature_absent_at_end=checks["signature_absent_at_end"],
                    )
                ],
                status="rejected" if hard_fail else "candidate",
                review=review,
            ).finalize()
            exp.id = experience_id(exp.dedup_key)
            out.append(exp)
    return out


NOT_APPLICABLE = "fixing it with `{compat_layer}` imports or by patching {library} at runtime (not an acceptable migration)"
KEEPS_OLD = "code that must keep running on {old_api} (the fix uses {new}-only API: "


def target_versions() -> dict[str, str]:
    """Experience.versions 的版本范围:{目标包 memory.version_key: memory.version_range}(未声明时为空)。"""
    m = _target().spec.memory
    return {m.version_key: m.version_range} if m.version_key and m.version_range else {}


def _target():
    from ...migration.targets import active

    return active()


def _preconditions(removed: list[str], observed: str, attribution: str) -> list[str]:
    t = _target()
    head = t.text("memory.precondition", "{new_api} is installed")
    out = [f"{head}{f' (source run used {observed})' if observed else ''}"]
    use = t.text("memory.old_api_use", "the code still uses the {old} API")
    out += [f"{use} `{x}`" for x in removed]
    if attribution == "traceback frame":
        out.append("the failing traceback points into the module that defines the affected model/function")
    return out


def _not_applicable(added: list[str]) -> list[str]:
    t = _target()
    out = [t.text("memory.not_applicable", NOT_APPLICABLE)]
    if added:
        out.append(t.text("memory.keeps_old", KEEPS_OLD) + ", ".join(added) + ")")
    return out


def _explanation(removed, added, files, flipped, prev, cur) -> str:
    what = (
        f"replaced {', '.join(f'`{t}`' for t in removed)} with {', '.join(f'`{t}`' for t in added)}"
        if removed and added
        else f"introduced {', '.join(f'`{t}`' for t in added)}"
        if added
        else f"removed {', '.join(f'`{t}`' for t in removed)}"
        if removed
        else "changed the code shown in the diff"
    )
    return (
        f"In the source run the edit {what} in {len(files)} file(s). Afterwards {len(flipped)} test(s) that failed "
        f"with this error no longer failed (suite {prev['passed']}/{prev['total']} -> {cur['passed']}/{cur['total']} "
        "passing). Rule-generated summary of the error -> edit -> test window; not written by a model."
    )
