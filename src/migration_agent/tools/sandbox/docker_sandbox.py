"""Docker SDK sandbox with a private workspace, read-only tests and process timeouts."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

from .checkpoints import GitCheckpoints
from .files import IGNORED, RepositoryFiles
from .protocol import ExecResult

_EXEC = r"""
import json, os, signal, subprocess, sys
req = json.loads(sys.argv[1])
try:
    p = subprocess.Popen(req['cmd'], cwd=req['cwd'], env={**os.environ, **req['env']},
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
except OSError as e:
    print(json.dumps(dict(returncode=127, stdout='', stderr=str(e), timed_out=False)))
    sys.exit(0)
timed_out = False
try:
    out, err = p.communicate(timeout=req['timeout'])
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(p.pid, signal.SIGKILL)
    out, err = p.communicate()
finally:
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
print(json.dumps(dict(returncode=p.returncode, stdout=out.decode('utf-8', 'replace'),
                     stderr=err.decode('utf-8', 'replace'), timed_out=timed_out)))
"""


class DockerSandbox:
    def __init__(
        self,
        repo_path: Path | str,
        image: str = "migration-agent-sandbox:latest",
        *,
        protected_globs: tuple[str, ...] | None = None,
        client=None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.workdir = Path("/workspace")
        self.image = image
        self.protected_globs = protected_globs
        self.client = client
        self._owns_client = client is None
        self.container = None
        self._tmp = None
        self._checkpoints = None
        self._original: dict = {}

    def start(self) -> None:
        if self.container is not None:
            return
        try:
            if self.client is None:
                import docker

                # Docker SDK does not read the CLI's selected Docker Desktop context.
                host = os.environ.get("DOCKER_HOST")
                if not host and os.name == "nt":
                    host = "npipe:////./pipe/dockerDesktopLinuxEngine"
                self.client = (
                    docker.DockerClient(base_url=host, timeout=660) if host else docker.from_env(timeout=660)
                )
            self.client.ping()
            self._tmp = tempfile.TemporaryDirectory(prefix="migration-sandbox-")
            staging = Path(self._tmp.name) / "workspace"
            shutil.copytree(self.repo_path, staging, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
            if any(p.is_symlink() for p in staging.rglob("*")):
                raise PermissionError("sandbox input contains symlinks; supply a regular-file checkout")
            mounts = {str(staging): {"bind": "/workspace", "mode": "rw"}}
            from ..base import ToolContext
            from .local import LocalExecutor

            guard = ToolContext(repo_path=staging, executor=LocalExecutor(staging))
            if self.protected_globs is not None:
                guard.protected_globs = self.protected_globs
            protected = Path(self._tmp.name) / "protected"
            protected.mkdir()
            candidates = sorted(staging.rglob("*"), key=lambda p: len(p.parts))
            mounted: list[Path] = []
            for p in candidates:
                rel = p.relative_to(staging)
                is_test_dir = p.is_dir() and p.name in ("tests", "test")
                if any(p.is_relative_to(parent) for parent in mounted):
                    continue
                if is_test_dir or p.is_file() and guard.is_protected(rel.as_posix()):
                    dst = protected / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if p.is_dir():
                        shutil.copytree(p, dst)
                    else:
                        shutil.copy2(p, dst)
                    mounts[str(dst)] = {"bind": "/workspace/" + rel.as_posix(), "mode": "ro"}
                    mounted.append(p)
            self.container = self.client.containers.create(
                self.image,
                command=["sleep", "infinity"],
                name="migration-" + uuid.uuid4().hex,
                working_dir="/workspace",
                volumes=mounts,
                user="1000:1000",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                mem_limit="2g",
                pids_limit=256,
                labels={"migration-agent": "sandbox"},
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": "/workspace",
                    "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
                },
            )
            self.container.start()
            # Chown encounters ro test mounts; writable files still become agent-owned.
            self.container.exec_run(["chown", "-R", "1000:1000", "/workspace", "/opt/venv"], user="0")
            files = RepositoryFiles(self)
            self._original = files.tree()
            self._checkpoints = GitCheckpoints(files)
        except Exception:
            self.stop()
            raise

    def exec(
        self, cmd: list[str], *, cwd: str | None = None, timeout: int = 300, env: dict[str, str] | None = None
    ) -> ExecResult:
        if self.container is None:
            raise RuntimeError("sandbox is not started")
        target = PurePosixPath("/workspace") / (cwd or ".")
        if ".." in target.parts or not target.is_relative_to("/workspace"):
            raise PermissionError("cwd escapes repository")
        if timeout <= 0 or timeout > 600:
            raise ValueError("timeout must be between 1 and 600 seconds")
        t0 = time.perf_counter()
        request = json.dumps({"cmd": cmd, "cwd": str(target), "timeout": timeout, "env": env or {}})
        code, output = self.container.exec_run(
            ["/opt/venv/bin/python", "-I", "-c", _EXEC, request], user="1000:1000"
        )
        if code != 0:
            raise RuntimeError(output.decode(errors="replace"))
        return ExecResult(cmd=cmd, duration_ms=(time.perf_counter() - t0) * 1000, **json.loads(output))

    def snapshot(self, label: str) -> str:
        if self._checkpoints is None:
            raise RuntimeError("sandbox is not started")
        return self._checkpoints.snapshot(label)

    def restore(self, snapshot_id: str) -> None:
        if self._checkpoints is None:
            raise RuntimeError("sandbox is not started")
        self._checkpoints.restore(snapshot_id)

    def publish(self) -> None:
        """Copy validated net changes back, refusing concurrent host changes."""
        current = RepositoryFiles(self).tree()
        changed = [p for p in set(current) | set(self._original) if current.get(p) != self._original.get(p)]
        for rel in changed:
            p = self.repo_path / rel
            if p.is_symlink() or not p.resolve().is_relative_to(self.repo_path):
                raise PermissionError("unsafe publication path: " + rel)
            old = self._original.get(rel)
            actual = base64.b64encode(p.read_bytes()).decode() if p.is_file() else None
            if actual != (old["data"] if old else None):
                raise RuntimeError("host file changed during run: " + rel)
        for rel in changed:
            p = self.repo_path / rel
            if rel not in current:
                p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(base64.b64decode(current[rel]["data"]))

    def stop(self) -> None:
        try:
            if self.container is not None:
                self.container.remove(force=True, v=True)
        finally:
            self.container = None
            if self._checkpoints is not None:
                self._checkpoints.close()
                self._checkpoints = None
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None
            if self.client is not None and self._owns_client:
                self.client.close()
                self.client = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
