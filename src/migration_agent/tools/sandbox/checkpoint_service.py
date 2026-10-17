"""CheckpointService:统一的检查点服务。

取代 AgentLoop(逐工具事务回滚)与 SupervisedLLM(按测试报告回滚)各自持有的 RollbackManager:两者共用同一后端
(Docker 沙箱的 git 快照或宿主上的私有 GitCheckpoints),检查点带标签、所属方(owner)、测试分数与签名,
供 orchestrator 选择「通过数最高的检查点」。

    svc = CheckpointService(ctx)
    cp = svc.create("step 3: edit_file", owner="loop")
    svc.restore(cp.id, owner="loop")      # 返回被恢复的文件
    svc.best()                            # 带测试报告的检查点里通过数最高的,同分取最新

每次创建/恢复调用 `listeners`(事件日志、ACI 读取新鲜度记录在这里挂接)。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .checkpoints import GitCheckpoints
from .protocol import Sandbox


@dataclass
class Checkpoint:
    id: str
    sha: str
    step: int
    label: str
    owner: str
    passed: int | None = None
    total: int | None = None
    signatures: list[str] = field(default_factory=list)


class CheckpointService:
    def __init__(self, ctx: Any, *, step: Callable[[], int] | None = None):
        self.ctx = ctx
        self.backend = ctx.executor if isinstance(ctx.executor, Sandbox) else GitCheckpoints(ctx.files)
        self._step = step or (lambda: 0)
        self._items: list[Checkpoint] = []
        self._by_sha: dict[str, Checkpoint] = {}
        self._restores: Counter[str] = Counter()
        self.listeners: list[Callable[[str, Checkpoint, list[str]], None]] = []

    def create(self, label: str, report: Any = None, *, owner: str = "loop") -> Checkpoint:
        sha = self.backend.snapshot(label)
        cp = Checkpoint(
            id=f"cp{len(self._items):04d}",
            sha=sha,
            step=int(self._step() or 0),
            label=label,
            owner=owner,
            passed=None if report is None else int(report.passed or 0),
            total=None if report is None else int(report.total or 0),
            signatures=sorted({f.signature for f in getattr(report, "failures", []) or []}),
        )
        self._items.append(cp)
        self._by_sha[sha] = cp
        self._notify("create", cp, [])
        return cp

    def _resolve(self, ref: str | Checkpoint) -> Checkpoint:
        if isinstance(ref, Checkpoint):
            return ref
        cp = self._by_sha.get(ref) or next((c for c in self._items if c.id == ref), None)
        if cp is None:
            raise ValueError(f"unknown checkpoint {ref!r}")
        return cp

    def restore(self, ref: str | Checkpoint, *, owner: str = "loop") -> list[str]:
        cp = self._resolve(ref)
        before = self.ctx.files.tree()
        self.backend.restore(cp.sha)
        after = self.ctx.files.tree()
        changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
        self._restores[owner] += 1
        self._notify("restore", cp, changed)
        return changed

    def restores(self, owner: str | None = None) -> int:
        return sum(self._restores.values()) if owner is None else self._restores[owner]

    def best(self) -> Checkpoint | None:
        scored = [c for c in self._items if c.passed is not None]
        return max(scored, key=lambda c: (c.passed, self._items.index(c)), default=None)

    def first_seen(self, signature: str) -> Checkpoint | None:
        return next((c for c in self._items if signature in c.signatures), None)

    def list(self) -> list[Checkpoint]:
        return list(self._items)

    def _notify(self, action: str, cp: Checkpoint, changed: list[str]) -> None:
        for fn in self.listeners:
            fn(action, cp, changed)

    def close(self) -> None:
        if isinstance(self.backend, GitCheckpoints):
            self.backend.close()


class ServiceRollback:
    """RollbackManager 的接口适配(checkpoint → sha、rollback(sha)),让既有调用方共用 CheckpointService。"""

    def __init__(self, service: CheckpointService, owner: str):
        self.service, self.owner = service, owner
        self._history: list[str] = []

    @property
    def rollbacks(self) -> int:
        return self.service.restores(self.owner)

    def checkpoint(self, label: str, report: Any = None) -> str:
        sha = self.service.create(label, report, owner=self.owner).sha
        self._history.append(sha)
        return sha

    def rollback(self, to: str | None = None) -> None:
        self.service.restore(to or self._history[-1], owner=self.owner)

    def history(self) -> list[str]:
        return list(self._history)

    def close(self) -> None:  # 由服务的拥有者(AgentLoop)统一关闭
        return None
