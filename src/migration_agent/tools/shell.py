"""`run_command`:在执行环境(LocalExecutor / 沙箱)中运行 shell 命令,带超时与输出截断。

命令以字符串传入:不含 shell 元字符时用 shlex 拆分直接执行;含 `; | & > < $ ( )` 或反引号等时
自动包成 `bash -c "..."`。真实跑 trace 显示模型会自然地写 `cmd1; cmd2 | head`,
拒绝它只会浪费 2–3 步。命令都在沙箱里跑,shell 逃逸面不是问题。
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from .base import Tool, ToolContext, ToolResult

_SHELL_META = re.compile(r"[;|&<>$`()*?\[\]{}~]|&&|\|\||\n")


class RunCommandTool(Tool):
    name = "run_command"
    mutates_repo = True  # Shell commands can write even when their intent is read-only.
    # 示例里的包名来自当前目标(version_package)
    DESCRIPTION = (
        "Run a shell command inside the repository root (e.g. 'python -c ...', 'pip show {package}', "
        "'grep -rn validator src'). Pipes, redirects and ';' are supported. Returns exit code, stdout "
        "and stderr. Do NOT use it to run the test suite — use run_tests instead, which returns "
        "structured results."
    )

    def __init__(self, target: Any = None):
        if target is None:
            from ..migration.targets import active

            target = active()
        self.description = self.DESCRIPTION.format(package=target.version_package)

    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "seconds, default 120", "minimum": 1},
        },
        "required": ["command"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = (args.get("command") or "").strip()
        if not raw:
            return ToolResult.error("empty command")
        if _SHELL_META.search(raw):
            cmd = ["bash", "-c", raw]
        else:
            try:
                cmd = shlex.split(raw)
            except ValueError as e:
                return ToolResult.error(f"cannot parse command: {e}")
        timeout = int(args.get("timeout") or 120)
        res = ctx.executor.exec(cmd, timeout=timeout)
        body = f"exit_code: {res.returncode}{' (timed out)' if res.timed_out else ''}\n"
        if res.stdout:
            body += f"--- stdout ---\n{res.stdout}\n"
        if res.stderr:
            body += f"--- stderr ---\n{res.stderr}\n"
        return ToolResult(
            content=ctx.truncate(body),
            is_error=not res.ok,
            metadata={
                "returncode": res.returncode,
                "duration_ms": res.duration_ms,
                "timed_out": res.timed_out,
            },
        )
