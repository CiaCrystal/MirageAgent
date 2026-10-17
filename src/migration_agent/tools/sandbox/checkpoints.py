"""Private Git checkpoints; never reset the user's Git index or working tree."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .files import RepositoryFiles


class GitCheckpoints:
    def __init__(self, files: RepositoryFiles):
        self.files = files
        self._tmp = tempfile.TemporaryDirectory(prefix="migration-checkpoints-")
        self.root = Path(self._tmp.name)
        self._git("init", "-q")
        self._git("config", "user.email", "checkpoint@migration-agent.local")
        self._git("config", "user.name", "Migration Agent")
        self._known: set[str] = set()

    def _git(self, *args: str) -> str:
        res = subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return res.stdout.strip()

    def snapshot(self, label: str) -> str:
        # Git stores a binary-safe manifest, including untracked files and modes.
        (self.root / "tree.json").write_text(json.dumps(self.files.tree(), sort_keys=True), encoding="utf-8")
        self._git("add", "tree.json")
        self._git("commit", "-q", "--allow-empty", "-m", label)
        sha = self._git("rev-parse", "HEAD")
        self._known.add(sha)
        return sha

    def restore(self, snapshot_id: str) -> None:
        self.files.restore(self.tree(snapshot_id))

    def tree(self, snapshot_id: str) -> dict:
        if snapshot_id not in self._known:
            raise ValueError("unknown checkpoint")
        return json.loads(self._git("show", f"{snapshot_id}:tree.json"))

    def close(self) -> None:
        self._tmp.cleanup()
