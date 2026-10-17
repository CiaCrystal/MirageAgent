"""记录全量测试的逐用例结果与相邻测试之间的文件变化，为经验抽取提供证据。子集测试不作为证据窗口边界。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..agent.hooks import Hook, TestReportEvent
from .strategy import source_frames

TEXT_SUFFIXES = (".py", ".txt", ".toml", ".cfg", ".ini", ".lock", ".in")
TEXT_NAMES = ("setup.py", "Pipfile", "pytest.ini", "tox.ini", "conftest.py")


def _texts(tree: dict[str, dict], max_bytes: int) -> dict[str, str]:
    out = {}
    for rel, entry in tree.items():
        if not (rel.endswith(TEXT_SUFFIXES) or rel.rsplit("/", 1)[-1] in TEXT_NAMES):
            continue
        data = base64.b64decode(entry["data"])
        if len(data) > max_bytes:
            continue
        out[rel] = data.decode("utf-8", "replace")
    return out


class EvidenceRecorder:
    def __init__(self, ctx: Any, tracer: Any, *, max_file_bytes: int = 400_000, max_tb_chars: int = 2500):
        self.ctx, self.tracer = ctx, tracer
        self.max_file_bytes, self.max_tb_chars = max_file_bytes, max_tb_chars
        self.reports: list[dict[str, Any]] = []
        self.initial: dict[str, str] | None = None
        self._prev: dict[str, str] | None = None
        self.target_version: str | None = None

    def attach(self, test_tool: Any) -> EvidenceRecorder:
        """挂到 RunTestsTool.listeners(每份报告产生后立即调用,时机与替换 run_report 的写法相同)。"""
        test_tool.listeners.append(lambda rep, subset, ctx: self._safe_record(rep, subset))
        return self

    def as_hook(self) -> EvidenceHook:
        """钩子写法:挂到 HookBus,经 on_test_report 记录(与 attach 二选一)。"""
        return EvidenceHook(self)

    def _safe_record(self, rep: Any, subset: bool) -> None:
        try:
            self.record(rep, subset=subset)
        except Exception as e:  # noqa: BLE001 — 证据记录失败不能影响 run
            self.reports.append({"seq": len(self.reports), "record_error": f"{type(e).__name__}: {e}"[:300]})

    def record(self, rep: Any, *, subset: bool = False) -> None:
        tree = self.ctx.files.tree()
        texts = _texts(tree, self.max_file_bytes)
        known = set(tree)
        if self.target_version is None:
            self.target_version = _target_version(self.ctx)
        failures = []
        seen_sig: set[str] = set()
        for f in getattr(rep, "failures", []) or []:
            item = {
                "test_id": f.test_id,
                "kind": f.kind,
                "signature": f.signature,
                "message": (f.message or "")[:300],
                "frames": source_frames(f.traceback, self.ctx.is_protected, known),
            }
            if f.signature not in seen_sig:  # 每个签名只保留首个用例的 traceback 节选
                seen_sig.add(f.signature)
                item["traceback"] = (f.traceback or "")[-self.max_tb_chars :]
            failures.append(item)
        row: dict[str, Any] = {
            "seq": len(self.reports),
            "step": int(getattr(self.tracer, "step", 0) or 0),
            "subset": subset,
            "passed": rep.passed,
            "failed": rep.failed,
            "errors": rep.errors,
            "total": rep.total,
            "skipped": rep.skipped,
            "parse_error": rep.parse_error,
            "collection_blocked": rep.collection_blocked,
            "failures": failures,
        }
        if not subset:
            if self.initial is None:
                self.initial = texts
            elif self._prev is not None:
                row["changes"] = {
                    rel: {"before": self._prev.get(rel, ""), "after": texts.get(rel, "")}
                    for rel in sorted(set(self._prev) | set(texts))
                    if self._prev.get(rel, "") != texts.get(rel, "")
                }
            self._prev = texts
        self.reports.append(row)

    def final_texts(self) -> dict[str, str]:
        return _texts(self.ctx.files.tree(), self.max_file_bytes)

    def changed_paths(self) -> list[str]:
        return sorted({rel for r in self.reports for rel in r.get("changes", {})})

    def to_dict(self, *, run_id: str, meta: dict[str, Any] | None = None, final: dict | None = None) -> dict:
        changed = self.changed_paths()
        final = final if final is not None else self.final_texts()
        return {
            "schema_version": 1,
            "run_id": run_id,
            version_key(): self.target_version,
            "meta": meta or {},
            "reports": self.reports,
            "initial_changed": {rel: (self.initial or {}).get(rel, "") for rel in sorted(changed)},
            "final_changed": {rel: final.get(rel, "") for rel in sorted(changed)},
        }

    def save(self, path: Path, **kw: Any) -> dict:
        data = self.to_dict(**kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
        return data


class EvidenceHook(Hook):
    name = "evidence"

    def __init__(self, recorder: EvidenceRecorder):
        self.recorder = recorder

    def on_test_report(self, ev: TestReportEvent) -> None:
        self.recorder._safe_record(ev.report, ev.subset)


def read_texts(repo: Path, rels: list[str]) -> dict[str, str]:
    """宿主上发布后的最终仓库中指定文件的文本(沙箱停止后读取最终状态用);不存在的文件为空串。"""
    out = {}
    for rel in rels:
        p = Path(repo) / rel
        out[rel] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    return out


def version_key() -> str:
    """证据 JSON 里记录目标库版本的键:`<version_package>_version`(已有证据文件沿用同一个键)。"""
    from .targets import active

    return f"{active().version_package}_version"


def _target_version(ctx: Any) -> str | None:
    from .targets import active

    pkg = active().version_package
    try:
        res = ctx.executor.exec(
            ["python", "-I", "-c", f"import importlib.metadata as m; print(m.version({pkg!r}))"], timeout=60
        )
        return (res.stdout.strip() or None) if res.ok else None
    except Exception:  # noqa: BLE001
        return None
