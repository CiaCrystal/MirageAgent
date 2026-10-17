"""Executor 协议:在「某个执行环境」中运行命令,是 Sandbox 的最小子集。

    ExecResult = exec(cmd: list[str], cwd: str | None, timeout: int, env: dict | None)

LocalExecutor 直接用 subprocess;DockerSandbox 实现同一协议并增加
`start / stop / snapshot / restore / copy_in / copy_out`。所有工具只依赖 Executor,
因此切换沙箱不需要改工具代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ExecResult(BaseModel):
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@runtime_checkable
class Executor(Protocol):
    """命令执行环境。`workdir` 是被迁移仓库在该环境中的根路径。"""

    workdir: Path

    def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> ExecResult: ...


@runtime_checkable
class Sandbox(Executor, Protocol):
    """完整沙箱:生命周期 + 快照。"""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self, label: str) -> str: ...
    def restore(self, snapshot_id: str) -> None: ...
    def publish(self) -> None: ...
