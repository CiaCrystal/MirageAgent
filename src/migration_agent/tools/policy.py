"""策略包装器:在不改工具实现与循环的前提下,经 `Tool` 协议给工具加约束(full 档)。

    CommandPolicyTool(RunCommandTool(), project_names(repo))  # 拒绝直接联网、下载包源码、装本项目发行版、全盘搜索
    FileLockTool(EditFileTool(), locks)     # 拒绝编辑已被策略放弃(abandon)的文件

包装后名称、schema、`mutates_repo` 与被包装工具完全相同,模型看到的工具集不变,只是部分调用被拒。

命令策略的依据(baseline 运行的 trace):约 14% 的步数花在 `pip download` 本项目/依赖的其他版本、
用 urllib 从 GitHub/PyPI 抓源码、`find /` 全盘搜索与翻 pip 缓存上。抓被迁移项目自身的上游(已迁移)版本既浪费预算,
也违反反泄漏原则(oracle 只用于计算 diff 相似度)。规则是通用的,不针对任何 case:
    - 直接联网:URL(含 pip 的 git+https)、curl/wget、urllib/requests/httpx 调用、socket、git clone/fetch/pull;
    - pip download / pip wheel(下载包源码查看);
    - pip install 被迁移项目自身的发行版(任何版本;项目名取自 pyproject/setup.cfg/setup.py 与顶层包目录),
      本地路径安装(`pip install -e .`)除外;
    - 越界搜索:对文件系统根目录做 find/grep -r/ls -R,翻 pip/uv 缓存。
**允许**用 pip install 升级第三方依赖:旧依赖(如 fastapi<0.100)不支持目标库新版本时,升级依赖本身就是迁移的
一部分(baseline 的两个成功 case 都依赖这一步);降级目标库由 Validator 拦截。
读取 site-packages 里已安装包的源码不受限。
"""

from __future__ import annotations

import configparser
import re
import shlex
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolResult

_NETWORK = re.compile(
    r"https?://|\bcurl\b|\bwget\b|\burllib\b|\brequests\.(?:get|post|head|Session)\b"
    r"|\bhttpx\.(?:get|post|stream|Client|AsyncClient)\b|\bhttp\.client\b"
    r"|\bsocket\.|\bgit\s+(?:clone|fetch|pull|ls-remote)\b|\baiohttp\.ClientSession\b",
    re.I,
)
_PIP = re.compile(r"\b(?:pip3?|uv\s+pip|python3?\s+-m\s+pip)\s+(download|install|wheel)\b(.*)", re.I)
_SCAN = re.compile(
    r"\bfind\s+/(?:\s|$)|\bfind\s+/(?:home|root|opt|usr|tmp|var)?\s+-|\bgrep\s+-\w*r\w*\b[^|;&]*\s/(?:\s|$)"
    r"|\bls\s+-\w*R\w*\s+/(?:\s|$)|\.cache/(?:pip|uv)\b",
)

REFUSAL = (
    "Refused by the run policy: {why}. Direct network access, downloading package sources, installing any "
    "released version of the project being migrated, and whole-filesystem searches are not available in this "
    "task (upgrading third-party dependencies with `pip install` is allowed). Migrate the code in this repository "
    "using the test output, installed package sources and the search tools."
)


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def project_names(repo: Path) -> set[str]:
    """被迁移项目自身的发行版名(归一化):pyproject [project]/[tool.poetry] name、setup.cfg、setup.py、顶层包目录。"""
    names: set[str] = set()
    try:
        data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
        for n in (data.get("project", {}).get("name"), data.get("tool", {}).get("poetry", {}).get("name")):
            if isinstance(n, str):
                names.add(n)
    except (OSError, tomllib.TOMLDecodeError):
        pass
    try:
        cfg = configparser.ConfigParser()
        cfg.read_string((repo / "setup.cfg").read_text(encoding="utf-8"))
        if cfg.has_option("metadata", "name"):
            names.add(cfg.get("metadata", "name"))
    except (OSError, configparser.Error):
        pass
    try:
        m = re.search(r"\bname\s*=\s*[\"']([\w.-]+)[\"']", (repo / "setup.py").read_text(encoding="utf-8"))
        if m:
            names.add(m.group(1))
    except OSError:
        pass
    for base in (repo, repo / "src"):
        if base.is_dir():
            for d in base.iterdir():
                if (
                    d.is_dir()
                    and (d / "__init__.py").exists()
                    and d.name not in ("tests", "test", "docs", "examples")
                ):
                    names.add(d.name)
    return {normalize_name(n) for n in names if n}


def command_violation(command: str, own_names: Iterable[str] = ()) -> str | None:
    """返回拒绝原因;允许时返回 None。`own_names`:被迁移项目自身的发行版名(归一化)。"""
    own = {normalize_name(n) for n in own_names}
    if _NETWORK.search(command):
        return "network access"
    for m in _PIP.finditer(command):
        verb, rest = m.group(1).lower(), m.group(2)
        if verb != "install":
            return f"pip {verb}"
        rest = re.split(r"[;&|]", rest, maxsplit=1)[0]
        try:
            args = shlex.split(rest)
        except ValueError:
            args = rest.split()
        targets = [a for a in args if not a.startswith("-") and not a.startswith((".", "/workspace"))]
        for t in targets:
            name = re.split(r"[\[<>=!~;@ ]", t, maxsplit=1)[0]
            if normalize_name(name) in own:
                return "installing a released version of the project being migrated"
    if _SCAN.search(command):
        return "searching outside the repository"
    return None


class _Wrapper(Tool):
    def __init__(self, inner: Tool):
        self.inner = inner
        self.name = inner.name
        self.description = inner.description
        self.input_schema = inner.input_schema
        self.mutates_repo = inner.mutates_repo
        self.refusals = 0


class CommandPolicyTool(_Wrapper):
    def __init__(self, inner: Tool, own_names: Iterable[str] = ()):
        super().__init__(inner)
        self.own_names = set(own_names)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        why = command_violation(str(args.get("command") or ""), self.own_names)
        if why:
            self.refusals += 1
            return ToolResult.error(REFUSAL.format(why=why), policy_refusal=why)
        return self.inner.run(args, ctx)


class FileLockTool(_Wrapper):
    """`locked()` 返回 {路径: 原因};编辑这些路径的调用被拒。路径按 ctx.canonical 归一化后比较。"""

    def __init__(self, inner: Tool, locked: Callable[[], dict[str, str]]):
        super().__init__(inner)
        self.locked = locked

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        locks = self.locked()
        if locks:
            try:
                path = ctx.canonical(str(args.get("path") or ""))
            except (PermissionError, ValueError):
                path = ""
            if path in locks:
                self.refusals += 1
                return ToolResult.error(
                    f"{path} was abandoned by the strategy controller ({locks[path]}); it is locked for this run. "
                    "Work on the remaining failures instead.",
                    locked_file=path,
                )
        return self.inner.run(args, ctx)
