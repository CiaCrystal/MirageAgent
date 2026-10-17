"""LocalExecutor:在宿主机上直接用 subprocess 执行命令(无隔离)。

在宿主环境执行命令；需要进程隔离时使用 DockerSandbox。
`python_executable` 允许指定运行被迁移仓库测试所用的解释器(默认当前解释器)。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .protocol import ExecResult


class LocalExecutor:
    def __init__(self, workdir: Path | str, python_executable: str | None = None):
        self.workdir = Path(workdir).resolve()
        self.python_executable = python_executable or sys.executable

    def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # 把裸 "python" 映射到指定解释器,保证测试跑在正确环境
        if cmd and cmd[0] in ("python", "python3"):
            cmd = [self.python_executable, *cmd[1:]]
        if os.name == "nt" and cmd and cmd[0] == "bash":
            git_bash = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
            if git_bash.is_file():
                cmd = [str(git_bash), *cmd[1:]]
        full_env = {**os.environ, **(env or {})}
        run_cwd = str(self.workdir / cwd) if cwd else str(self.workdir)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_cwd,
                env=full_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return ExecResult(
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                cmd=cmd,
                returncode=-1,
                stdout=(e.stdout or b"").decode(errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or ""),
                stderr=f"timed out after {timeout}s",
                duration_ms=(time.perf_counter() - t0) * 1000,
                timed_out=True,
            )
        except FileNotFoundError as e:
            return ExecResult(
                cmd=cmd,
                returncode=127,
                stdout="",
                stderr=str(e),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
