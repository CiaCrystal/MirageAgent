"""ExperienceStore:文件化经验库;MemorySnapshot:冻结的只读快照。

目录(`data/memory/<target>/`):
    experiences/<id>.md        审核通过的经验(事实来源,YAML front matter + Markdown 正文)
    candidates/pending/<id>.md 待审候选(含自动检查结果);不进入目录与索引,不可检索
    candidates/rejected/<id>.md 拒绝条目(规则或人工),保留原因
    MEMORY.md                  由 experiences/ 生成的简短目录(人可读;不每轮注入模型)
    index.json                 由 experiences/ 生成的检索缓存,带 content_version(各文件 sha256 的哈希)

- 候选导入 `add_candidate` 按 dedup_key 合并、保留全部来源;同一来源 (run, steps) 重复导入是 no-op。
- 发布 `review(id, "approved", ...)` 要求候选通过全部自动检查,且(若设置了 SourcePolicy)所有来源都被允许;
  已有同 id 的已发布经验时只合并来源。人工审核结论原样记录,不由代码代填。
- 复用计数 `record_reuse(id, run_id, success)` 对同一 run 幂等。
- 所有写操作持有目录锁(filelock)并原子替换;`load_index` 校验 content_version,过期时重建(只读库则报错)。
- `freeze(dest, policy, config)` 只复制来源全部被 policy 允许的已发布经验(混合来源整条排除),生成独立的
  MEMORY.md / index.json / snapshot.json(逐文件 sha256、来源清单、配置、快照哈希),文件设为只读。
  `MemorySnapshot` 打开时与 `verify()` 时逐一校验哈希;没有任何写接口,也不回退到其他库。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel

from .schema import Experience, SourceRef, from_markdown, signature_key, to_markdown


def estimate_tokens(text: str) -> int:
    """与 agent/compaction.estimate_tokens 相同的保守估算(UTF-8 字节 / 3);rag 层不依赖 agent 层。"""
    return -(-len(json.dumps(text, ensure_ascii=False).encode("utf-8")) // 3)


class QualityGateError(ValueError):
    pass


class StaleIndexError(RuntimeError):
    pass


class ReadOnlyMemoryError(PermissionError):
    pass


class SourcePolicy(BaseModel):
    """Experience provenance policy without run or repository grouping restrictions."""

    name: str = "reviewed-experiences"

    def allows_source(self, source: SourceRef) -> tuple[bool, str]:
        return True, "allowed"

    def allows(self, exp: Experience) -> tuple[bool, list[str]]:
        return (True, []) if exp.sources else (False, ["no sources"])

    def fingerprint(self) -> str:
        return _sha(json.dumps(self.model_dump(), sort_keys=True).encode())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def render_entry(exp: Experience) -> str:
    """经验返回给模型时的紧凑文本(完整条目,不截断;超出预算时整条跳过)。"""
    runs = len({s.run_id for s in exp.sources})
    ev = exp.test_evidence[0] if exp.test_evidence else None
    if exp.kind != "fix" or exp.has_summary():
        return _render_summary(exp, runs, ev)
    lines = [f"[{exp.id}] {exp.error_signature[:300]}"]
    if exp.preconditions:
        lines.append("Applies when: " + "; ".join(exp.preconditions))
    if exp.not_applicable:
        lines.append("Not for: " + "; ".join(exp.not_applicable))
    if exp.api_tags:
        lines.append("API: " + ", ".join(exp.api_tags))
    lines.append(
        f"Fix seen in {runs} other repo run(s)"
        + (f"; tests {ev.passed_before}->{ev.passed_after}/{ev.total_after} after it" if ev else "")
        + ":"
    )
    lines += ["```diff", exp.diff, "```"]
    if exp.explanation:
        lines.append("Why: " + exp.explanation)
    return "\n".join(lines)


_KIND_LABEL = {"fix": "Fix", "anti_pattern": "Avoid", "strategy": "Strategy", "env": "Environment"}


def _render_summary(exp: Experience, runs: int, ev: Any) -> str:
    """LLM 总结条目(LLM 概括 + 规则证据)。diff 与测试证据仍来自规则;说明文字标明由模型概括。"""
    lines = [f"[{exp.id}] {_KIND_LABEL.get(exp.kind, exp.kind)}: {exp.title()}"]
    lines.append(f"Error: {exp.error_signature[:300]}")
    for key, value in exp.narrative.items():
        lines.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    if exp.symptoms:
        lines.append("Symptoms: " + "; ".join(exp.symptoms))
    if exp.recipe:
        lines.append("Steps: " + " ".join(f"{i}) {r}" for i, r in enumerate(exp.recipe, 1)))
    if exp.pitfalls:
        lines.append("Pitfalls: " + "; ".join(exp.pitfalls))
    if exp.not_applicable:
        lines.append("Not for: " + "; ".join(exp.not_applicable))
    if exp.api_tags:
        lines.append("API: " + ", ".join(exp.api_tags))
    if exp.diff:
        label = "Attempted change (undone)" if exp.kind == "anti_pattern" else "Change"
        lines.append(
            f"{label} seen in {runs} other repo run(s)"
            + (f"; tests {ev.passed_before}->{ev.passed_after}/{ev.total_after} after it" if ev else "")
            + ":"
        )
        lines += ["```diff", exp.diff, "```"]
    if exp.generated_by:
        lines.append("(Explanation summarised by a model from the rule-located evidence above.)")
    return "\n".join(lines)


def search_text(exp: Experience) -> str:
    parts = [
        exp.error_signature,
        exp.error_text,
        " ".join(exp.api_tags),
        " ".join(exp.preconditions),
        exp.diff,
    ]

    extra = [exp.summary_title, *exp.narrative.values(), *exp.symptoms, *exp.query_variants]
    return "\n".join(parts + [x for x in extra if x])


def build_index(experiences: list[tuple[Path, Experience]], root: Path) -> dict[str, Any]:
    entries = []
    for path, exp in experiences:
        rel = path.relative_to(root).as_posix()
        entries.append(
            {
                "id": exp.id,
                "file": rel,
                "sha256": _sha(path.read_bytes()),
                "content_hash": exp.content_hash,
                "dedup_key": exp.dedup_key,
                "status": exp.status,
                "target": exp.target,
                "versions": exp.versions,
                "error_signature": exp.error_signature,
                "sig_key": signature_key(exp.error_signature),
                "error_type": exp.error_type,
                "api_tags": exp.api_tags,
                "sources": [{"repo": s.repo, "family": s.family, "run_id": s.run_id} for s in exp.sources],
                "tokens": estimate_tokens(render_entry(exp)),
                "text": search_text(exp),
            }
        )
    entries.sort(key=lambda e: e["id"])
    return {
        "schema_version": 1,
        "content_version": content_version(entries),
        "built_at": _now(),
        "entries": entries,
    }


def content_version(entries: list[dict]) -> str:
    return _sha(json.dumps(sorted((e["file"], e["sha256"]) for e in entries)).encode())


def render_catalog(experiences: list[Experience], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Generated from experiences/*.md (reviewed and approved entries only). Do not edit by hand.",
        "",
    ]
    if not experiences:
        lines.append("(no approved experiences)")
    for exp in sorted(experiences, key=lambda e: e.id):
        tags = ", ".join(exp.api_tags) or "-"
        runs = len({s.run_id for s in exp.sources})
        kind = f"[{exp.kind}] " if exp.kind != "fix" else ""
        vk = _version_key(exp.target)
        lines.append(
            f"- [{exp.id}](experiences/{exp.id}.md) — {kind}{exp.error_signature[:110].replace(chr(96), '')} · API: {tags} · "
            f"{vk} {exp.versions.get(vk, '?')} · sources: {runs}"
        )
    return "\n".join(lines) + "\n"


def _merge(base: Experience, new: Experience) -> tuple[Experience, bool]:
    """把 new 的来源与测试证据并入 base(内容沿用 base);返回 (合并结果, 是否有变化)。"""
    keys = {s.key() for s in base.sources}
    added = [s for s in new.sources if s.key() not in keys]
    ev_keys = {(e.run_id, e.step_before, e.step_after) for e in base.test_evidence}
    ev_added = [e for e in new.test_evidence if (e.run_id, e.step_before, e.step_after) not in ev_keys]
    if not added and not ev_added:
        return base, False
    merged = base.model_copy(deep=True)
    merged.sources += added
    merged.test_evidence += ev_added
    merged.combined = merged.combined or new.combined
    merged.updated_at = _now()
    return merged, True


class ExperienceStore:
    def __init__(self, root: Path | str, *, policy: SourcePolicy | None = None, title: str = "MEMORY"):
        self.root = Path(root)
        self.policy = policy
        self.title = title
        self.read_only = False

    # ---- paths
    @property
    def experiences_dir(self) -> Path:
        return self.root / "experiences"

    @property
    def pending_dir(self) -> Path:
        return self.root / "candidates" / "pending"

    @property
    def rejected_dir(self) -> Path:
        return self.root / "candidates" / "rejected"

    def _lock(self) -> FileLock:
        self.root.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.root / ".lock"), timeout=60)

    # ---- reading
    @staticmethod
    def _read_dir(d: Path) -> list[tuple[Path, Experience]]:
        if not d.is_dir():
            return []
        return [(p, from_markdown(p.read_text(encoding="utf-8"))) for p in sorted(d.glob("*.md"))]

    def all(self, approved_only: bool = True) -> list[Experience]:
        rows = [e for _, e in self._read_dir(self.experiences_dir)]
        if approved_only:
            return [e for e in rows if e.status == "approved"]
        return rows + self.candidates("candidate") + self.candidates("rejected")

    def candidates(self, status: str = "candidate") -> list[Experience]:
        d = self.pending_dir if status == "candidate" else self.rejected_dir
        return [e for _, e in self._read_dir(d)]

    def get(self, exp_id: str) -> Experience | None:
        for d in (self.experiences_dir, self.pending_dir, self.rejected_dir):
            p = d / f"{exp_id}.md"
            if p.is_file():
                return from_markdown(p.read_text(encoding="utf-8"))
        return None

    # ---- writing
    def _check_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyMemoryError("memory store is read-only")

    def add_candidate(self, exp: Experience) -> tuple[str, str]:
        """导入一条候选;返回 (id, created | merged | unchanged)。approved 状态不能直接导入。"""
        self._check_writable()
        if exp.status == "approved":
            raise QualityGateError("candidates cannot be imported as approved; use review()")
        exp = exp.model_copy(deep=True).finalize()
        exp.id = f"exp-{exp.dedup_key[:12]}"
        d = self.rejected_dir if exp.status == "rejected" else self.pending_dir
        with self._lock():
            published = self.experiences_dir / f"{exp.id}.md"
            if exp.status == "candidate" and published.is_file():
                pub = from_markdown(published.read_text(encoding="utf-8"))
                _, changed = _merge(pub, exp)
                if not changed:
                    return exp.id, "unchanged"
            path = d / f"{exp.id}.md"
            if path.is_file():
                current = from_markdown(path.read_text(encoding="utf-8"))
                merged, changed = _merge(current, exp)
                if not changed:
                    return exp.id, "unchanged"
                if not exp.review.auto_passed:
                    merged.review.auto_passed = False
                    merged.review.uncertain = sorted(set(merged.review.uncertain) | set(exp.review.uncertain))
                atomic_write(path, to_markdown(merged))
                return exp.id, "merged"
            atomic_write(path, to_markdown(exp))
            return exp.id, "created"

    def review(self, exp_id: str, decision: str, reason: str, reviewer: str) -> Experience:
        """人工审核结论:approved 发布到 experiences/(需通过自动检查与来源策略),rejected 移入 rejected/。"""
        self._check_writable()
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        if not reviewer or not reason:
            raise ValueError("review requires a reviewer and a reason")
        with self._lock():
            path = self.pending_dir / f"{exp_id}.md"
            if not path.is_file():
                raise FileNotFoundError(f"no pending candidate {exp_id}")
            exp = from_markdown(path.read_text(encoding="utf-8"))
            exp.review.reviewer, exp.review.reason, exp.review.reviewed_at = reviewer, reason, _now()
            exp.updated_at = _now()
            if decision == "rejected":
                exp.status = exp.review.status = "rejected"
                atomic_write(self.rejected_dir / f"{exp_id}.md", to_markdown(exp))
                path.unlink()
                return exp
            if not exp.review.auto_passed:
                raise QualityGateError(f"{exp_id} did not pass all automatic checks; cannot publish")
            if self.policy is not None:
                ok, why = self.policy.allows(exp)
                if not ok:
                    raise QualityGateError(f"{exp_id} has sources not allowed by {self.policy.name}: {why}")
            exp.status = exp.review.status = "approved"
            target = self.experiences_dir / f"{exp_id}.md"
            if target.is_file():
                exp, _ = _merge(from_markdown(target.read_text(encoding="utf-8")), exp)
            atomic_write(target, to_markdown(exp))
            path.unlink()
            self._rebuild_locked()
            return exp

    # ---- 生命周期(consolidate.py 使用)
    def _dir(self, where: str) -> Path:
        return {
            "experiences": self.experiences_dir,
            "pending": self.pending_dir,
            "rejected": self.rejected_dir,
        }[where]

    def locate(self, exp_id: str) -> list[tuple[str, Experience]]:
        """该 id 所在的位置(experiences / pending / rejected,可能多处:已发布条目可以另有待审候选)。"""
        out = []
        for where in ("experiences", "pending", "rejected"):
            p = self._dir(where) / f"{exp_id}.md"
            if p.is_file():
                out.append((where, from_markdown(p.read_text(encoding="utf-8"))))
        return out

    def lifecycle_entries(self) -> list[tuple[str, Experience]]:
        """可参与合并判断的条目:已发布(含 retired)与待审候选;规则或人工拒绝的不参与。"""
        return [("experiences", e) for _, e in self._read_dir(self.experiences_dir)] + [
            ("pending", e) for _, e in self._read_dir(self.pending_dir)
        ]

    def write_entry(self, exp: Experience, where: str) -> None:
        """写入一条完整条目(consolidate 已决定位置与内容)。发布到 experiences/ 要求通过自动检查与来源策略。"""
        self._check_writable()
        if where == "experiences" and exp.status == "approved":
            if not exp.review.auto_passed:
                raise QualityGateError(f"{exp.id} did not pass all automatic checks; cannot publish")
            if self.policy is not None:
                ok, why = self.policy.allows(exp)
                if not ok:
                    raise QualityGateError(f"{exp.id} has sources not allowed by {self.policy.name}: {why}")
        with self._lock():
            atomic_write(self._dir(where) / f"{exp.id}.md", to_markdown(exp))
            if where == "experiences":
                self._rebuild_locked()

    def remove_entry(self, exp_id: str, where: str) -> None:
        self._check_writable()
        with self._lock():
            p = self._dir(where) / f"{exp_id}.md"
            if p.is_file():
                p.unlink()
            if where == "experiences":
                self._rebuild_locked()

    def record_reuse(self, exp_id: str, run_id: str, success: bool) -> bool:
        """一次 run 对经验的复用结果;同一 run 只计一次。返回是否计入。"""
        self._check_writable()
        with self._lock():
            path = self.experiences_dir / f"{exp_id}.md"
            exp = from_markdown(path.read_text(encoding="utf-8"))
            if run_id in exp.reuse_runs:
                return False
            exp.reuse_runs.append(run_id)
            if success:
                exp.reuse_success += 1
            else:
                exp.reuse_failure += 1
            exp.updated_at = _now()
            atomic_write(path, to_markdown(exp))
            self._rebuild_locked()
            return True

    def rebuild(self) -> dict[str, Any]:
        self._check_writable()
        with self._lock():
            return self._rebuild_locked()

    def _rebuild_locked(self) -> dict[str, Any]:
        rows = [(p, e) for p, e in self._read_dir(self.experiences_dir) if e.status == "approved"]
        index = build_index(rows, self.root)
        atomic_write(self.root / "index.json", json.dumps(index, indent=1, ensure_ascii=False) + "\n")
        atomic_write(self.root / "MEMORY.md", render_catalog([e for _, e in rows], self.title))
        return index

    def load_index(self) -> dict[str, Any]:
        """读取 index.json 并校验它仍对应 experiences/ 的当前文件;过期或缺失时重建。"""
        path = self.root / "index.json"
        index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        current = [
            {"file": p.relative_to(self.root).as_posix(), "sha256": _sha(p.read_bytes())}
            for p, e in self._read_dir(self.experiences_dir)
            if e.status == "approved"
        ]
        if index is not None and index.get("content_version") == content_version(current):
            return index
        if self.read_only:
            raise StaleIndexError("index.json does not match experiences/; rebuild required")
        return self.rebuild()

    def entries(self) -> list[tuple[dict, Experience]]:
        index = self.load_index()
        return [
            (e, from_markdown((self.root / e["file"]).read_text(encoding="utf-8"))) for e in index["entries"]
        ]

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "store",
            "root": str(self.root),
            "content_version": self.load_index()["content_version"],
        }

    # ---- freeze
    def freeze(self, dest: Path | str, policy: SourcePolicy, config: dict | None = None) -> dict[str, Any]:
        dest = Path(dest)
        if dest.exists():
            raise FileExistsError(f"snapshot directory already exists: {dest}")
        with self._lock():
            self._rebuild_locked() if not self.read_only else None
            included, excluded = [], []
            for p, exp in self._read_dir(self.experiences_dir):
                ok, why = policy.allows(exp)
                if exp.status != "approved":
                    ok, why = False, [f"status {exp.status}"]
                (included if ok else excluded).append((p, exp, why))
            (dest / "experiences").mkdir(parents=True)
            rows = []
            for p, exp, _ in included:
                target = dest / "experiences" / p.name
                shutil.copyfile(p, target)
                rows.append((target, exp))
        index = build_index(rows, dest)
        atomic_write(dest / "index.json", json.dumps(index, indent=1, ensure_ascii=False) + "\n")
        atomic_write(
            dest / "MEMORY.md", render_catalog([e for _, e in rows], f"{self.title} (frozen snapshot)")
        )
        files = {
            p.relative_to(dest).as_posix(): _sha(p.read_bytes())
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "source_store": "reviewed-experiences",
            "policy": policy.model_dump(),
            "policy_hash": policy.fingerprint(),
            "config": config or {},
            "experiences": [
                {
                    "id": e.id,
                    "content_hash": e.content_hash,
                    "sources": [
                        s.model_dump(include={"repo", "family", "commit", "run_id", "steps"})
                        for s in e.sources
                    ],
                }
                for _, e in rows
            ],
            "excluded": [{"id": e.id, "why": why} for _, e, why in excluded],
            "files": files,
        }
        manifest["snapshot_hash"] = _sha(
            json.dumps({k: manifest[k] for k in ("files", "policy_hash", "config")}, sort_keys=True).encode()
        )
        atomic_write(dest / "snapshot.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        for p in dest.rglob("*"):
            if p.is_file():
                p.chmod(stat.S_IREAD)
        return manifest


class MemorySnapshot:
    """冻结快照的只读视图。打开即校验全部文件哈希;不存在写接口,不回退读取任何其他库。"""

    read_only = True

    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / "snapshot.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"not a frozen memory snapshot (snapshot.json missing): {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.verify()
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self._entries = [
            (e, from_markdown((self.root / e["file"]).read_text(encoding="utf-8")))
            for e in self.index["entries"]
        ]

    @property
    def snapshot_hash(self) -> str:
        return self.manifest["snapshot_hash"]

    def verify(self) -> str:
        if self.policy().fingerprint() != self.manifest["policy_hash"]:
            raise StaleIndexError("snapshot policy hash mismatch")
        expected = self.manifest["files"]
        actual = {
            p.relative_to(self.root).as_posix(): _sha(p.read_bytes())
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name != "snapshot.json"
        }
        if actual != expected:
            changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
            raise StaleIndexError(f"frozen snapshot modified: {changed}")
        recomputed = _sha(
            json.dumps(
                {k: self.manifest[k] for k in ("files", "policy_hash", "config")}, sort_keys=True
            ).encode()
        )
        if recomputed != self.snapshot_hash:
            raise StaleIndexError("snapshot.json hash mismatch")
        return self.snapshot_hash

    def entries(self) -> list[tuple[dict, Experience]]:
        return list(self._entries)

    def identity(self) -> dict[str, Any]:
        return {"kind": "snapshot", "root": str(self.root), "snapshot_hash": self.snapshot_hash}

    def policy(self) -> SourcePolicy:
        return SourcePolicy.model_validate(self.manifest["policy"])


def open_memory(path: Path | str, *, require_snapshot: bool) -> MemorySnapshot | ExperienceStore:
    "Runtime support."
    path = Path(path)
    if (path / "snapshot.json").is_file():
        return MemorySnapshot(path)
    if require_snapshot:
        raise FileNotFoundError(f"a frozen memory snapshot is required;; {path} is not one")
    if not path.is_dir():
        raise FileNotFoundError(f"memory store not found: {path}")
    return ExperienceStore(path)


def _version_key(target_name: str) -> str:
    """条目所属目标的版本键(目标包 memory.version_key);未知目标退回当前目标。"""
    from ...migration.targets import active, get_target

    try:
        t = get_target(target_name)
    except KeyError:
        t = active()
    return t.spec.memory.version_key or t.version_package
