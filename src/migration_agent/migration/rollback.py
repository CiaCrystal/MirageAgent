"""Per-tool Git transactions, isolated from the user's Git repository and index."""

from __future__ import annotations

from ..tools.base import ToolContext
from ..tools.sandbox.checkpoints import GitCheckpoints
from ..tools.sandbox.protocol import Sandbox


class RollbackManager:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.backend = ctx.executor if isinstance(ctx.executor, Sandbox) else GitCheckpoints(ctx.files)
        self._history: list[str] = []
        self.rollbacks = 0

    def checkpoint(self, label: str) -> str:
        sha = self.backend.snapshot(label)
        self._history.append(sha)
        return sha

    def rollback(self, to: str | None = None) -> None:
        target = to or self._history[-1]
        self.backend.restore(target)
        self.rollbacks += 1

    def history(self) -> list[str]:
        return list(self._history)

    def close(self) -> None:
        if isinstance(self.backend, GitCheckpoints):
            self.backend.close()
