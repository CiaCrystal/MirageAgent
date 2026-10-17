"""`run_tests`:运行 pytest 并解析为结构化结果。

实现方式:给测试命令追加 `--junitxml=<repo>/.migration_agent/junit.xml -o junit_family=xunit2`
(junitxml 是 pytest 内置功能,不要求目标仓库安装 pytest-json-report),然后解析 XML:
    TestReport(total, passed, failed, errors, skipped, failures=[FailureInfo...], returncode, stdout_tail)
FailureInfo 含 test_id、kind(failure|error)、message、traceback 节选。
回填给 LLM 时按「异常首行」分组,同一根因的多个用例只展示一份 traceback,避免 30 个相同报错撑爆上下文。

三种失败形态都覆盖:断言失败(failure)、运行时异常(failure 或 error,视发生在 setup/call)、
收集错误(error,testcase.name 为文件路径)。

注意:XML 文件通过 ctx.resolve 从仓库目录读取。Docker 沙箱以 bind mount 挂载仓库,因此同样适用。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..migration.error_parser import ErrorParser
from .base import Tool, ToolContext, ToolResult

JUNIT_REL = ".migration_agent/junit.xml"


class FailureInfo(BaseModel):
    test_id: str
    kind: str  # failure | error
    message: str
    traceback: str
    signature: str  # 异常首行归一化,用于分组(ErrorParser 会替换为更精细的签名)


class TestReport(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list[FailureInfo] = Field(default_factory=list)
    returncode: int = 0
    duration_ms: float = 0.0
    stdout_tail: str = ""
    parse_error: str | None = None
    collection_blocked: bool = False

    @property
    def all_passed(self) -> bool:
        return (
            self.passed > 0
            and self.failed == 0
            and self.errors == 0
            and self.parse_error is None
            and self.returncode == 0
        )

    def summary_line(self) -> str:
        return (
            f"{self.passed}/{self.total} passed, {self.failed} failed, {self.errors} errors, "
            f"{self.skipped} skipped (exit {self.returncode})"
        )

    def render(self, max_groups: int = 8, tb_chars: int = 1800) -> str:
        """回填给 LLM 的文本:摘要 + 按根因分组的失败。"""
        lines = [f"TEST RESULT: {self.summary_line()}"]
        if self.collection_blocked:
            lines.append(
                "Collection was blocked in conftest: the single error record is a startup diagnostic, not a collected test."
            )
        if self.parse_error:
            lines.append(f"(could not parse junit xml: {self.parse_error})")
            lines.append(self.stdout_tail)
            return "\n".join(lines)
        if not self.failures:
            if self.total == 0:
                lines.append(self.stdout_tail)
            return "\n".join(lines)
        groups: dict[str, list[FailureInfo]] = defaultdict(list)
        for f in self.failures:
            groups[f.signature].append(f)
        lines.append(f"\n{len(groups)} distinct failure root cause(s):")
        for i, (sig, items) in enumerate(groups.items(), 1):
            if i > max_groups:
                lines.append(f"\n... and {len(groups) - max_groups} more root causes")
                break
            ids = ", ".join(x.test_id for x in items[:6]) + (" ..." if len(items) > 6 else "")
            lines.append(f"\n[{i}] {len(items)} test(s): {ids}\n    {sig}")
            tb = items[0].traceback
            if len(tb) > tb_chars:
                tb = "...\n" + tb[-tb_chars:]
            lines.append("    --- traceback (first affected test) ---")
            lines.append("\n".join("    " + ln for ln in tb.splitlines()))
        return "\n".join(lines)


_E_LINE = re.compile(r"^E\s+(.*)$", re.MULTILINE)
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_PATH_LINE = re.compile(r"(/[^\s:]+|[A-Za-z]:\\[^\s:]+):\d+")


def _signature(kind: str, message: str, text: str) -> str:
    """异常首行归一化:取第一条有效 `E   ...` 行(跳过目标库的文档链接行),去掉地址、路径行号。"""
    sig = ErrorParser().parse(text, message, kind)
    return str(sig) if sig.exception_type == kind else f"{kind}: {sig}"


def parse_junit(xml_text: str) -> TestReport:
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    rep = TestReport()
    for suite in suites:
        for tc in suite.iter("testcase"):
            classname = tc.get("classname") or ""
            name = tc.get("name") or ""
            test_id = f"{classname}::{name}" if classname else name
            rep.total += 1
            node = None
            kind = None
            for k in ("error", "failure", "skipped"):
                node = tc.find(k)
                if node is not None:
                    kind = k
                    break
            if kind is None:
                rep.passed += 1
                continue
            if kind == "skipped":
                rep.skipped += 1
                continue
            if kind == "failure":
                rep.failed += 1
            else:
                rep.errors += 1
            msg = node.get("message") or ""  # type: ignore[union-attr]
            text = (node.text or "").strip()  # type: ignore[union-attr]
            rep.failures.append(
                FailureInfo(
                    test_id=test_id,
                    kind=kind,
                    message=msg[:500],
                    traceback=text,
                    signature=_signature(kind, msg, text),
                )
            )
    return rep


class RunTestsTool(Tool):
    name = "run_tests"
    description = (
        "Run the repository's test suite with pytest and return structured results: counts plus "
        "failures grouped by root cause with one traceback each. Optionally pass a path or node id "
        "to run a subset (e.g. 'tests/test_models.py::test_user')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "optional test path / node id"},
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "extra pytest args, e.g. ['-x']",
            },
        },
    }

    def __init__(self, test_cmd: list[str] | None = None, timeout: int = 600):
        self.test_cmd = test_cmd or ["python", "-m", "pytest", "-q"]
        self.timeout = timeout
        # 每份报告产生后依次调用 listener(report, subset, ctx)(AgentLoop 经它广播 on_test_report)
        self.listeners: list = []

    def run_report(
        self, ctx: ToolContext, path: str | None = None, extra_args: list[str] | None = None
    ) -> TestReport:
        rep = self._run_report(ctx, path, extra_args)
        ctx.last_report = rep
        for fn in list(self.listeners):
            fn(rep, bool(path or extra_args), ctx)
        return rep

    def _run_report(self, ctx: ToolContext, path: str | None, extra_args: list[str] | None) -> TestReport:
        ctx.files.remove(JUNIT_REL)
        cmd = [
            *self.test_cmd,
            f"--junitxml={JUNIT_REL}",
            "-o",
            "junit_family=xunit2",
            "-p",
            "no:cacheprovider",
            # 有收集错误时仍运行其它模块,让 agent 一次看到全貌(默认行为是直接中断)
            "--continue-on-collection-errors",
        ]
        if extra_args:
            cmd += list(extra_args)
        if path:
            cmd.append(path)
        res = ctx.executor.exec(cmd, timeout=self.timeout)
        tail = (res.stdout + ("\n" + res.stderr if res.stderr else ""))[-3000:]
        try:
            xml_text = ctx.files.read(JUNIT_REL)
        except RuntimeError:
            # Root conftest loads before JUnit starts. Preserve a visibly marked
            # startup error so the agent can fix imports instead of giving up.
            if (
                not res.timed_out
                and res.returncode in (1, 4)
                and ("ImportError while loading conftest" in tail or "ConftestImportFailure" in tail)
            ):
                return TestReport(
                    total=1,
                    errors=1,
                    returncode=res.returncode,
                    duration_ms=res.duration_ms,
                    stdout_tail=tail,
                    collection_blocked=True,
                    failures=[
                        FailureInfo(
                            test_id="__pytest_startup__",
                            kind="error",
                            message="conftest blocked collection (not a collected test)",
                            traceback=tail,
                            signature=_signature("error", "conftest blocked collection", tail),
                        )
                    ],
                )
            return TestReport(
                returncode=res.returncode,
                duration_ms=res.duration_ms,
                stdout_tail=tail,
                parse_error="junit xml not produced (pytest missing, crashed, or timed out?)",
            )
        try:
            rep = parse_junit(xml_text)
        except ET.ParseError as e:
            return TestReport(
                returncode=res.returncode, duration_ms=res.duration_ms, stdout_tail=tail, parse_error=str(e)
            )
        rep.returncode = res.returncode
        rep.duration_ms = res.duration_ms
        rep.stdout_tail = tail
        return rep

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        rep = self.run_report(ctx, args.get("path"), args.get("extra_args"))
        return ToolResult(
            content=ctx.truncate(rep.render()),
            is_error=not rep.all_passed,
            metadata={
                "report": rep.model_dump(exclude={"failures", "stdout_tail"}),
                "n_failures": len(rep.failures),
            },
        )


def cleanup_artifacts(repo_path: Path) -> None:
    """删除 run_tests 留下的 .migration_agent 目录。"""
    d = repo_path / Path(JUNIT_REL).parent
    if d.is_dir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
