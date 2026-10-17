"""Tool 协议与上下文。

    class Tool:
        name: str
        description: str
        input_schema: dict          # JSON Schema,直接作为 Anthropic tool 定义
        mutates_repo: bool = False  # True 时循环会在调用后自动跑测试
        def run(self, args: dict, ctx: ToolContext) -> ToolResult

ToolResult.content 是回填给 LLM 的文本;metadata 只进 trace,不给 LLM。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from .sandbox.files import RepositoryFiles
from .sandbox.protocol import Executor


class ToolResult(BaseModel):
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def error(cls, msg: str, **meta: Any) -> ToolResult:
        return cls(content=f"ERROR: {msg}", is_error=True, metadata=meta)


PROTECTED_GLOBS: tuple[str, ...] = (
    "tests/**",
    "test/**",
    "**/tests/**",
    "**/test/**",
    "test_*.py",
    "**/test_*.py",
    "*_test.py",
    "**/*_test.py",
    "test.py",
    "**/test.py",
    "tests.py",
    "**/tests.py",
    "conftest.py",
    "**/conftest.py",
)

PYTEST_NAMES = ("pytest", "py.test", "pytest.exe")
# pytest 中带独立取值的选项:其后一个参数不是测试路径
_VALUE_OPTS = frozenset(
    [
        "-k",
        "-m",
        "-p",
        "-c",
        "-o",
        "-W",
        "-n",
        "-r",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--rootdir",
        "--tb",
        "--maxfail",
        "--confcutdir",
        "--basetemp",
        "--durations",
        "--junitxml",
        "--log-level",
        "--import-mode",
        "--asyncio-mode",
        "--cov",
        "--cov-report",
        "--timeout",
    ]
)


def globs_from_test_cmd(test_cmd: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """test_cmd 里点名的测试路径 → 只读 glob(文件本身;目录再加 `dir/**`)。

    只看 pytest 之后的位置参数(跳过选项及其取值);`path::node` 取 path;`.`、绝对路径与越界路径忽略。
    没有 pytest 时(如 `python test.py`)只取以 .py 结尾的参数。"""
    from glob import escape

    cmd = list(test_cmd)
    idx = next(
        (i for i, a in enumerate(cmd) if PurePosixPath(a.replace("\\", "/")).name in PYTEST_NAMES), None
    )
    args = cmd[idx + 1 :] if idx is not None else [a for a in cmd[1:] if a.endswith(".py")]
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a.startswith("-"):
            skip = a in _VALUE_OPTS
            continue
        rel = a.split("::", 1)[0].replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.rstrip("/")
        if not rel or rel == "." or rel.startswith("/") or ":" in rel or ".." in rel.split("/"):
            continue
        out.append(escape(rel))
        if not rel.endswith(".py"):
            out.append(escape(rel) + "/**")
    return tuple(dict.fromkeys(out))


# 按 pytest 实际收集结果确定只读测试文件(settings.protect_collected_tests;Evaluator(protect_collected=True))。
# 命令行解析看不到 pytest.ini / setup.cfg 的 testpaths、python_files 等配置,收集结果能覆盖这些情况;
# 加 --continue-on-collection-errors,收集报错的文件(迁移前后最常见的失败)也会以 `ERROR path` 列出。
COLLECT_FLAGS = ("--collect-only", "-q", "--continue-on-collection-errors", "-p", "no:cacheprovider")
# `path::node`(-q)、`path: N`(-qq,test_cmd 自带 -q 时)、`ERROR path` / `ERROR collecting path`
_COLLECTED = re.compile(r"^(?:ERROR(?: collecting)?\s+)?(\S+?\.py)(?:::|:\s*\d+\s*$|\s|$)")


def is_pytest_cmd(test_cmd: list[str] | tuple[str, ...]) -> bool:
    return any(PurePosixPath(a.replace("\\", "/")).name in PYTEST_NAMES for a in test_cmd)


def parse_collected(output: str, repo_path: Path) -> tuple[str, ...]:
    """`pytest --collect-only` 输出 → 仓库内存在的 .py 文件(相对路径,按出现顺序去重)。

    容器内绝对路径去掉 /workspace/ 前缀;仓库里不存在的路径(rootdir 不是仓库根时的相对路径、site-packages)丢弃。"""
    out: list[str] = []
    root = Path(repo_path)
    for line in output.splitlines():
        m = _COLLECTED.match(line.strip())
        if not m:
            continue
        rel = m.group(1).replace("\\", "/").removeprefix("/workspace/")
        while rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("/") or ".." in rel.split("/") or "site-packages" in rel:
            continue
        if (root / rel).is_file() and rel not in out:
            out.append(rel)
    return tuple(out)


def collect_test_files(
    executor: Any, test_cmd: list[str] | tuple[str, ...], repo_path: Path, timeout: int = 300
) -> tuple[tuple[str, ...], str]:
    """在 executor 里跑一次 `<test_cmd> --collect-only`,返回 (测试文件, 错误说明)。不是 pytest 命令时什么都不做。"""
    if not is_pytest_cmd(test_cmd):
        return (), "test_cmd is not a pytest command"
    try:
        res = executor.exec([*test_cmd, *COLLECT_FLAGS], timeout=max(1, min(timeout, 600)))
    except Exception as e:  # noqa: BLE001 — 收集失败只退回到 glob 规则,不中断 run
        return (), f"{type(e).__name__}: {e}"[:300]
    files = parse_collected(f"{res.stdout}\n{res.stderr}", repo_path)
    return files, "" if files else f"no test files collected (exit {res.returncode})"


def protected_globs_for(test_cmd: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """默认只读 glob 与 test_cmd 点名的路径，供沙箱挂载保护使用。"""
    return tuple(dict.fromkeys((*PROTECTED_GLOBS, *globs_from_test_cmd(test_cmd))))


@dataclass
class ToolContext:
    """每个 run 一份,在工具间共享。"""

    repo_path: Path
    executor: Executor
    tool_output_max_chars: int = 12_000
    # 测试目录/文件只读;由 fs 工具与 Validator 共同使用。具体 case 再用 protect_test_cmd 补上 test_cmd 点名的路径
    protected_globs: tuple[str, ...] = PROTECTED_GLOBS
    changed_files: set[str] = field(default_factory=set)
    # run_tests 工具每次执行后写入,循环据此判断终止
    last_report: Any = None
    # ACI(harness):超长输出落盘回调,返回附在截断提示里的说明;None = 不落盘
    spill: Any = None

    @property
    def files(self) -> RepositoryFiles:
        return RepositoryFiles(self.executor)

    def resolve(self, rel: str) -> Path:
        """把工具传入的相对路径解析到仓库内,拒绝越界。"""
        p = (self.repo_path / rel).resolve()
        if not p.is_relative_to(self.repo_path.resolve()):
            raise PermissionError(f"path escapes repository: {rel}")
        return p

    def protect_test_cmd(self, test_cmd: list[str] | tuple[str, ...]) -> None:
        """把 test_cmd 点名的测试文件/目录并入只读集合(fs 工具、Validator、Evaluator 共用同一判定)。"""
        self.protected_globs = tuple(dict.fromkeys((*self.protected_globs, *globs_from_test_cmd(test_cmd))))

    def protect_files(self, paths: list[str] | tuple[str, ...]) -> None:
        """把具体文件(如 pytest 收集到的测试文件)并入只读集合。"""
        from glob import escape

        self.protected_globs = tuple(dict.fromkeys((*self.protected_globs, *(escape(p) for p in paths))))

    def is_protected(self, rel: str) -> bool:
        from fnmatch import fnmatch

        rel = rel.replace("\\", "/").lstrip("./")
        return any(fnmatch(rel, g) for g in self.protected_globs)

    def canonical(self, rel: str) -> str:
        return self.resolve(rel).relative_to(self.repo_path.resolve()).as_posix()

    def truncate(self, text: str) -> str:
        if len(text) <= self.tool_output_max_chars:
            return text
        half = self.tool_output_max_chars // 2
        omitted = len(text) - 2 * half
        note = ""
        if self.spill is not None:  # ACI:完整输出落盘,模型可分页读取
            note = self.spill(text)
        return f"{text[:half]}\n\n... [{omitted} chars truncated{note}] ...\n\n{text[-half:]}"


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    mutates_repo: bool = False

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def schema(self) -> dict[str, Any]:
        """Anthropic Messages API 的 tool 定义。"""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}
