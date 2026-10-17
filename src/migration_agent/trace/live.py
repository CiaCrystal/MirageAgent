"""实时运行视图:作为 Tracer 监听器,把 trace 事件渲染成类似 Claude Code 的滚动输出 + 底部状态栏。

    renderer = make_renderer("live" | "plain", console, verbose=False)
    with renderer:                       # LiveRenderer 启动 / 停止 rich.Live;PlainRenderer 什么都不做
        tracer.add_listener(renderer)    # 或 assemble(..., observers=[renderer])
        ...
    replay(events, renderer, speed=5.0)  # 回放已有 trace(不花 API 费用调 UI)

- LiveRenderer:rich.Live 底部状态栏(spinner、当前阶段、step/max_steps、耗时、累计 tokens、最近一次测试);
  滚动区经 console.print 输出,与 Live 不冲突。
- PlainRenderer:同样的事件 → 逐行纯文本,无 Live、无动画、无 ANSI(非 TTY / CI / 重定向到文件)。
- 终端不支持 UTF-8(legacy Windows 控制台 / 非 utf 编码)时符号换成 ASCII。
监听器只读事件,不改变运行;渲染异常由 Tracer 吞掉。
"""

from __future__ import annotations

import contextlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .schema import TraceEvent

UNICODE_SYMBOLS = {
    "bullet": "⏺",
    "elbow": "⎿",
    "star": "✻",
    "done": "☒",
    "active": "◐",
    "pending": "☐",
    "blocked": "⊘",
    "abandoned": "☐",
    "warn": "⚠",
    "ellipsis": "…",
    "sep": "·",
}
ASCII_SYMBOLS = {
    "bullet": "*",
    "elbow": "->",
    "star": "~",
    "done": "[x]",
    "active": "[>]",
    "pending": "[ ]",
    "blocked": "[!]",
    "abandoned": "[-]",
    "warn": "!",
    "ellipsis": "...",
    "sep": "|",
}
PLAN_STYLES = {
    "done": "green",
    "active": "bold",
    "pending": "",
    "blocked": "yellow",
    "abandoned": "dim strike",
}
SPAN_KINDS = ("llm_call", "tool_call", "test_run", "retrieval")
ARG_MAX = 80
RESULT_MAX = 110
VERBOSE_LINES = 6


def use_ascii(console: Console) -> bool:
    """legacy Windows 控制台或输出编码不是 utf 时用 ASCII 符号。"""
    encoding = (console.encoding or "").lower().replace("-", "")
    return bool(console.legacy_windows) or not encoding.startswith("utf")


def _clip(text: Any, n: int = ARG_MAX) -> str:
    s = " ".join(str(text).split())  # 折叠换行与连续空白
    return s if len(s) <= n else s[: n - 3] + "..."


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in str(text).splitlines() if ln.strip()), "")


def _num(n: float) -> str:
    n = float(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{int(n)}"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------- 工具参数 / 结果摘要
def _path_arg(a: dict) -> str:
    return str(a.get("path") or a.get("file_path") or "")


def _read_args(a: dict) -> str:
    s, e = a.get("start_line"), a.get("end_line")
    rng = f":{s or 1}-{e or ''}" if s or e else ""
    return _path_arg(a) + rng


def _multi_edit_args(a: dict) -> str:
    n = len(a.get("edits") or [])
    return f"{_path_arg(a)}, {n} edit{'s' if n != 1 else ''}"


def _insert_args(a: dict) -> str:
    return f"{_path_arg(a)}:{a.get('insert_line', '?')}"


def _grep_args(a: dict) -> str:
    out = repr(str(a.get("pattern", "")))
    if a.get("path") and a["path"] != ".":
        out += f" in {a['path']}"
    if a.get("glob"):
        out += f" ({a['glob']})"
    return out


def _run_tests_args(a: dict) -> str:
    return " ".join([str(a.get("path") or "all"), *map(str, a.get("extra_args") or [])])


def _query_args(a: dict) -> str:
    return repr(str(a.get("query", "")))


def _update_plan_args(a: dict) -> str:
    ops = a.get("ops") or []
    parts = []
    for op in ops[:4]:
        kind = op.get("op", "?")
        if kind == "set_status":
            parts.append(f"{op.get('id', '?')}->{op.get('status', '?')}")
        elif kind == "add":
            parts.append(f"+{_clip(op.get('goal', ''), 30)}")
        else:
            parts.append(f"{kind} {op.get('id', '')}".strip())
    more = f" +{len(ops) - 4}" if len(ops) > 4 else ""
    return ", ".join(parts) + more


TOOL_ARGS: dict[str, Callable[[dict], str]] = {
    "read_file": _read_args,
    "edit_file": _path_arg,
    "multi_edit": _multi_edit_args,
    "write_file": _path_arg,
    "insert_line": _insert_args,
    "grep": _grep_args,
    "glob": lambda a: str(a.get("pattern", "")),
    "run_command": lambda a: _first_line(a.get("command", "")),
    "run_tests": _run_tests_args,
    "search_migration_docs": _query_args,
    "search_codebase": _query_args,
    "search_experience": _query_args,
    "find_usages": lambda a: str(a.get("symbol", "")),
    "update_plan": _update_plan_args,
    "view_plan": lambda a: "",
    "list_files": lambda a: " ".join(x for x in (str(a.get("path") or "."), str(a.get("glob") or "")) if x),
    "load_skill": lambda a: str(a.get("name", "")),
}


def tool_args_summary(name: str, args: Any) -> str:
    """工具参数摘要(≤ 80 字符);未知工具取第一个字符串参数。"""
    if not isinstance(args, dict):
        return _clip(args) if args else ""
    fn = TOOL_ARGS.get(name)
    if fn is None and name.startswith("search_"):
        fn = _query_args
    try:
        text = fn(args) if fn else next((v for v in args.values() if isinstance(v, str)), "")
    except Exception:  # noqa: BLE001 — 参数形状不符时退回通用摘要
        text = next((v for v in args.values() if isinstance(v, str)), "")
    return _clip(text)


def tool_result_summary(name: str, payload: dict) -> str:
    """工具结果的一行摘要。"""
    result = str(payload.get("result") or "")
    lines = [ln for ln in result.splitlines() if ln.strip()]
    if not lines:
        return "(no output)" if not payload.get("is_error") else "error"
    if name == "read_file" and not payload.get("is_error"):
        m = re.search(r"\(lines (\d+)-(\d+) of (\d+)\)", lines[0])
        if m:
            a, b, total = map(int, m.groups())
            return f"Read {b - a + 1} lines" + (f" (of {total})" if b - a + 1 < total else "")
    if name == "run_command" and lines[0].startswith("exit_code"):
        body = [ln for ln in lines[1:] if not ln.startswith("---")]
        return _clip(lines[0] + (f" - {body[0].strip()}" if body else ""), RESULT_MAX)
    if name in ("edit_file", "multi_edit", "write_file", "insert_line"):
        return _clip(lines[0], RESULT_MAX)  # 首行是「edited x / applied n edit(s)」,其后是改动后的代码片段
    if name == "load_skill" and not payload.get("is_error"):
        return f"Loaded {len(lines)} lines"
    if name in ("grep", "glob", "find_usages") and len(lines) > 1:
        return f"{len(lines)} lines - {_clip(lines[0], RESULT_MAX - 20)}"
    more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
    return _clip(lines[0], RESULT_MAX) + more


def _kv(payload: dict, keys: Iterable[str] | None = None, n: int = 4) -> str:
    """标量字段的 key=value 摘要。"""
    items = [(k, payload[k]) for k in keys if k in payload] if keys else list(payload.items())
    out = [f"{k}={_clip(v, 40)}" for k, v in items if isinstance(v, (str, int, float, bool)) and v != ""]
    return " ".join(out[:n])


# ---------------------------------------------------------------- 渲染器
class _Renderer:
    """事件 → 输出的公共逻辑;子类决定输出方式(Live / 纯文本)。可直接作为 Tracer 监听器调用。"""

    plain = False

    def __init__(self, console: Console | None = None, *, verbose: bool = False, time_scale: float = 1.0):
        self.console = console or Console()
        self.verbose = verbose
        self.time_scale = time_scale
        self.ascii = use_ascii(self.console)
        self.sym = ASCII_SYMBOLS if self.ascii else UNICODE_SYMBOLS
        # 状态栏数据
        self.step = 0
        self.max_steps: int | None = None
        self.tokens_in = 0
        self.tokens_out = 0
        self._tokens_base = (0, 0)  # 本次尝试开始前的累计值(run_end 时按 RunResult 校正)
        self.tests: tuple[int, int] | None = None
        self.activity = "Starting"
        self.t0: float | None = None
        self.delta = ""  # on_text_delta 累积的流式文本
        self._plan_version: Any = None
        self._open_tools: list[str] = []

    # ---- 上下文管理器(PlainRenderer 无事可做)----
    def __enter__(self) -> _Renderer:
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    # ---- 监听器入口 ----
    def __call__(self, ev: TraceEvent, phase: str) -> None:
        self.handle(ev, phase)

    def handle(self, ev: TraceEvent, phase: str) -> None:
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.step = ev.step
        handler = getattr(self, f"_on_{ev.kind}", None)
        if handler is not None:
            handler(ev, ev.payload or {}, phase)

    def on_text_delta(self, text: str) -> None:
        """预留:接 messages.stream 后逐段喂入模型文本;完整文本仍由 assistant_text 事件输出。"""
        self.delta += text
        self.activity = "Writing"

    # ---- 状态 ----
    @property
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else (time.monotonic() - self.t0) * self.time_scale

    def status_text(self) -> Text:
        sep = f" {self.sym['sep']} "
        parts = [f"step {self.step}/{self.max_steps}" if self.max_steps else f"step {self.step}"]
        parts.append(_duration(self.elapsed))
        parts.append(f"{_num(self.tokens_in)} in / {_num(self.tokens_out)} out tokens")
        if self.tests is not None:
            parts.append(f"tests {self.tests[0]}/{self.tests[1]}")
        dots = "" if self.activity == "Done" else self.sym["ellipsis"]
        text = Text.assemble(
            (f"{self.activity}{dots} ", "bold"),
            (f"({sep.join(parts)})", "dim"),
        )
        if self.delta:
            text.append(f"  {_clip(self.delta[-60:], 60)}", style="dim italic")
        return text

    # ---- 输出原语 ----
    def _print(self, renderable: RenderableType) -> None:
        self.console.print(renderable, highlight=False, soft_wrap=self.plain)

    def _line(self, text: str, style: str = "") -> None:
        self._print(Text(text, style=style))

    def _bullet(self, head: str, rest: str = "", style: str = "", head_style: str = "bold") -> None:
        self._print(
            Text.assemble((f"{self.sym['bullet']} ", style or "white"), (head, head_style), (rest, ""))
        )

    def _elbow(self, text: str, style: str = "") -> None:
        self._print(Text.assemble((f"  {self.sym['elbow']}  ", "dim"), (text, style)))

    def _warn(self, text: str) -> None:
        self._print(Text(f"{self.sym['warn']} {_clip(text, 160)}", style="yellow"))

    def _dim(self, text: str) -> None:
        self._print(Text(f"  {_clip(text, 140)}", style="dim"))

    def _rule(self, title: str) -> None:
        self._print(Rule(title, characters="-" if self.ascii else "─"))

    def _markdown(self, text: str) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        body: RenderableType = Text(text) if self.ascii else Markdown(text)
        grid.add_row(Text(self.sym["bullet"], style="white"), body)
        self._print(grid)

    # ---- 事件处理 ----
    def _on_run_start(self, ev: TraceEvent, p: dict, phase: str) -> None:
        budget = p.get("budget") or {}
        self.max_steps = budget.get("max_steps") or self.max_steps
        self.activity = "Starting"
        self._tokens_base = (self.tokens_in, self.tokens_out)
        tools = p.get("tools") or []
        self._dim(
            f"run {ev.run_id} {self.sym['sep']} target {p.get('target', '?')} {self.sym['sep']} "
            f"model {p.get('model', '?')} {self.sym['sep']} {len(tools)} tools {self.sym['sep']} "
            f"max {self.max_steps} steps"
        )

    def _on_run_end(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self.activity = "Done"  # 汇总由 CLI 打印
        # RunResult 的 token 还含规划 / 分类等附加调用,以它为准
        if p.get("tokens_in") is not None:
            self.tokens_in = max(self.tokens_in, self._tokens_base[0] + int(p["tokens_in"]))
            self.tokens_out = max(self.tokens_out, self._tokens_base[1] + int(p.get("tokens_out") or 0))

    def _on_attempt(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if "of" in p:
            self._rule(f"Attempt {p.get('attempt')}/{p.get('of')}")
        elif "end" in p:
            restored = f", restored {p['restored']}" if p.get("restored") else ""
            self._dim(f"attempt {p.get('attempt')} ended: {p.get('end')}{restored}")
        elif "postmortem" in p:
            self._dim(f"attempt {p.get('attempt')} postmortem replan done")

    def _on_llm_call(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if phase == "start":
            self.activity = "Thinking"
            return
        self.tokens_in += int(ev.tokens_in or 0)
        self.tokens_out += int(ev.tokens_out or 0)
        self.activity = "Working"
        self.delta = ""
        if p.get("short_circuit"):
            self._dim(f"model call short-circuited by {p['short_circuit']}")
        if p.get("exception"):
            self._warn(f"LLM call failed: {p['exception']}")

    def _on_assistant_text(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self.delta = ""
        text = str(p.get("text") or "").strip()
        if text:
            self._markdown(text)

    def _on_tool_call(self, ev: TraceEvent, p: dict, phase: str) -> None:
        name = str(p.get("name", "?"))
        if phase == "start":
            self._open_tools.append(name)
            self.activity = name
            self._tool_header(name, p.get("args"))
            return
        if name in self._open_tools:
            self._open_tools.remove(name)
        else:  # 没收到 start(监听器后挂上)时补一行标题
            self._tool_header(name, p.get("args"))
        self.activity = "Working"
        style = "red" if p.get("is_error") else ""
        lines = [ln.rstrip() for ln in str(p.get("result") or "").splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        if not (self.verbose and lines):
            self._elbow(tool_result_summary(name, p), style)
            return
        # verbose:结果前 6 行代替一行摘要
        indent = " " * (len(self.sym["elbow"]) + 4)
        self._elbow(lines[0][:200], style)
        for ln in lines[1:VERBOSE_LINES]:
            self._print(Text(f"{indent}{ln[:200]}", style=style or "dim"))
        if len(lines) > VERBOSE_LINES:
            self._print(
                Text(f"{indent}{self.sym['ellipsis']} +{len(lines) - VERBOSE_LINES} lines", style="dim")
            )

    def _tool_header(self, name: str, args: Any) -> None:
        summary = tool_args_summary(name, args)
        self._bullet(name, f"({summary})" if summary else "", style="green")

    def _on_test_run(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if phase == "start":
            self.activity = "Running tests"
            self._bullet(
                "Running tests", f" ({p.get('reason', '')})" if p.get("reason") else "", style="cyan"
            )
            return
        self.activity = "Working"
        if p.get("parse_error"):
            self._elbow(f"parse error: {_clip(p['parse_error'], RESULT_MAX)}", "red")
            return
        passed, total = int(p.get("passed") or 0), int(p.get("total") or 0)
        failed = int(p.get("failed") or 0) + int(p.get("errors") or 0)
        self.tests = (passed, total)
        sep = f" {self.sym['sep']} "
        parts = [f"{passed}/{total} passed", f"{failed} failed"]
        if p.get("n_root_causes") is not None:
            parts.append(f"{p['n_root_causes']} root causes")
        if p.get("collection_blocked"):
            parts.append("collection blocked")
        green = total > 0 and failed == 0
        self._elbow(sep.join(parts), "green bold" if green else "")

    def _on_retrieval(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if phase == "start":
            self.activity = "Retrieving"
            return
        hits = p.get("hits", p.get("memory_hits"))
        what = p.get("purpose", "")
        extra = f": {_clip(p['signature'], 70)}" if p.get("signature") else ""
        self._dim(
            f"retrieval {what}{extra}" + (f" {self.sym['sep']} {hits} hits" if hits is not None else "")
        )

    def _on_plan(self, ev: TraceEvent, p: dict, phase: str) -> None:
        items = p.get("items")
        version = p.get("version")
        if isinstance(items, list):
            if version != self._plan_version:
                self._plan_version = version
                self._plan(items, version)
        elif p.get("source") == "budget_mark":
            stalled = " (stalled)" if p.get("stalled") else ""
            self._dim(
                f"budget {int(float(p.get('mark', 0)) * 100)}% {self.sym['sep']} best {p.get('best')} passed{stalled}"
            )
            return
        elif p.get("source") == "checkpoint_failed":
            self._warn(f"plan checkpoint failed: {p.get('error', '')}")
            return
        elif version is not None and version != self._plan_version and p.get("counts"):
            self._plan_version = version
            counts = " ".join(f"{k} {v}" for k, v in p["counts"].items() if v)
            self._dim(f"plan v{version} ({p.get('source', '')}): {counts}")
        if p.get("source") == "final" and "completion_rate" in p:
            self._dim(f"plan completion {float(p['completion_rate']) * 100:.0f}%")

    def _plan(self, items: list[dict], version: Any) -> None:
        done = sum(1 for it in items if it.get("status") == "done")
        self._bullet("Plan", f" v{version} {self.sym['sep']} {done}/{len(items)} done", style="magenta")
        for i, it in enumerate(items):
            status = str(it.get("status", "pending"))
            mark = self.sym.get(status, self.sym["pending"])
            prefix = f"  {self.sym['elbow']}  " if i == 0 else " " * (len(self.sym["elbow"]) + 4)
            goal = _clip(it.get("goal", ""), 100)
            suffix = " (abandoned)" if status == "abandoned" and self.plain else ""
            self._print(
                Text.assemble(
                    (prefix, "dim"),
                    (f"{mark} ", PLAN_STYLES.get(status, "")),
                    (f"{it.get('id', '')} ", "dim"),
                    (goal + suffix, PLAN_STYLES.get(status, "")),
                )
            )

    def _on_replan(self, ev: TraceEvent, p: dict, phase: str) -> None:
        rejected = p.get("rejected") or []
        n_rej = len(rejected) if isinstance(rejected, list) else rejected
        state = "ok" if p.get("ok") else (p.get("skipped") or p.get("error") or "failed")
        self._dim(
            f"replan ({p.get('trigger', '?')}): {_clip(state, 40)} {self.sym['sep']} "
            f"{p.get('accepted', 0)} accepted {self.sym['sep']} {n_rej} rejected"
        )

    # 警告(黄色)
    def _on_rollback(self, ev: TraceEvent, p: dict, phase: str) -> None:
        violations = "; ".join(map(str, p.get("violations") or []))
        self._warn(f"rolled back {p.get('checkpoint', '')}: {violations}")

    def _on_validation(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if p.get("ok") is False or p.get("violations"):
            self._warn("validation failed: " + "; ".join(map(str, p.get("violations") or [])))

    def _on_stop_gate(self, ev: TraceEvent, p: dict, phase: str) -> None:
        src = f" ({p['source']})" if p.get("source") else ""
        self._warn(f"stop gate {p.get('action', '')}{src}: {p.get('reason') or p.get('message') or ''}")

    def _on_permission(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self._warn(
            f"permission denied: {p.get('tool', '?')} (rule {p.get('rule', '?')}) {_clip(p.get('args', ''), 60)}"
        )

    def _on_hook_error(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self._warn(f"hook {p.get('hook', '?')}.{p.get('event', '?')} failed: {p.get('error', '')}")

    def _on_llm_resume(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self._warn(f"LLM error, retrying ({p.get('attempt')}): {p.get('error', '')}")

    # 次要信息(灰色)
    def _on_compaction(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self._dim(f"context compacted {_kv(p)}")

    def _on_strategy(self, ev: TraceEvent, p: dict, phase: str) -> None:
        rest = {k: v for k, v in p.items() if k != "action"}
        self._dim(f"strategy {p.get('action', '')} {_kv(rest)}")

    def _on_budget(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if "mark" in p:
            self._dim(
                f"budget {int(float(p['mark']) * 100)}% used {self.sym['sep']} {_kv(p, ('steps', 'tokens'))}"
            )
        else:
            self._dim(f"budget {_kv(p)}")

    def _on_classify(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if p.get("error"):
            self._dim(f"classify failed: {p['error']}")
            return
        cats = Counter((p.get("categories") or {}).values())
        self._dim("classified " + ", ".join(f"{n} {c}" for c, n in cats.items()))

    def _on_subagent(self, ev: TraceEvent, p: dict, phase: str) -> None:
        ok = "ok" if p.get("ok") else f"failed {p.get('error') or ''}"
        self._dim(f"subagent {p.get('kind', '')}: {p.get('steps', 0)} steps {self.sym['sep']} {ok}")

    def _on_skill(self, ev: TraceEvent, p: dict, phase: str) -> None:
        self._dim(f"skill {_kv(p)}")

    def _on_note(self, ev: TraceEvent, p: dict, phase: str) -> None:
        if "nudge" in p:
            self._dim(f"model stopped with failing tests; nudge {p['nudge']}")


class PlainRenderer(_Renderer):
    """逐行纯文本:无 Live、无动画、无 ANSI 控制字符。"""

    plain = True

    def __init__(self, console: Console | None = None, **kw: Any):
        console = console or Console(color_system=None, highlight=False, emoji=False, soft_wrap=True)
        super().__init__(console, **kw)

    def _rule(self, title: str) -> None:
        self._line(f"---- {title} ----")

    def _markdown(self, text: str) -> None:
        lines = text.splitlines() or [""]
        self._line(f"{self.sym['bullet']} {lines[0]}")
        for ln in lines[1:]:
            self._line(f"  {ln}")


class _StatusBar:
    """Live 每次刷新时重新生成状态栏(spinner 动画由同一个 Spinner 对象维持)。"""

    def __init__(self, renderer: LiveRenderer):
        self.renderer = renderer
        self.spinner = Spinner("line" if renderer.ascii else "dots", style="cyan")

    def __rich__(self) -> RenderableType:
        self.spinner.update(text=self.renderer.status_text())
        return Group(Text(""), self.spinner)


class LiveRenderer(_Renderer):
    """rich.Live:滚动输出 + 底部状态栏。作为上下文管理器使用;Ctrl+C 时 __exit__ 照样恢复终端。"""

    def __init__(self, console: Console | None = None, **kw: Any):
        super().__init__(console, **kw)
        self._live: Live | None = None

    def __enter__(self) -> LiveRenderer:
        super().__enter__()
        self._live = Live(
            _StatusBar(self),
            console=self.console,
            refresh_per_second=10,
            transient=True,
            redirect_stderr=False,
        )
        self._live.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        live, self._live = self._live, None
        if live is not None:
            with contextlib.suppress(Exception):  # 终端恢复失败也不能掩盖原异常
                live.stop()
        return False


def make_renderer(ui: str, console: Console | None = None, *, verbose: bool = False, time_scale: float = 1.0):
    """ui: live | plain | quiet。live 在非终端下自动降级为 plain;quiet 返回 None。"""
    if ui not in ("live", "plain", "quiet"):
        raise ValueError(f"unknown ui {ui!r}; choose live | plain | quiet")
    if ui == "quiet":
        return None
    console = console or Console()
    if ui == "live" and console.is_terminal:
        return LiveRenderer(console, verbose=verbose, time_scale=time_scale)
    plain = Console(file=console.file, color_system=None, highlight=False, emoji=False, soft_wrap=True)
    return PlainRenderer(plain, verbose=verbose, time_scale=time_scale)


# ---------------------------------------------------------------- 汇总与回放
def summary_panel(r: dict[str, Any], *, ascii: bool = False) -> Panel:
    """run 结束汇总(RunResult.model_dump() 或 run_end payload)。"""
    ok = bool(r.get("success"))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    tests_style = "green" if ok else "red"
    grid.add_row("reason", str(r.get("reason", "")))
    grid.add_row(
        "tests", Text(f"{r.get('tests_passed', 0)}/{r.get('tests_total', 0)} passed", style=tests_style)
    )
    grid.add_row("steps", str(r.get("steps", 0)))
    grid.add_row(
        "tokens",
        f"in {r.get('tokens_in', 0):,}  cached {r.get('tokens_cached', 0):,}  out {r.get('tokens_out', 0):,}",
    )
    grid.add_row("time", _duration(float(r.get("duration_s") or 0)))
    quality = (
        f"violations {r.get('violations', 0)}  rollbacks {r.get('rollbacks', 0)}  "
        f"deprecated API {r.get('deprecated_api_count', 0)}  compat imports {r.get('compatibility_import_count', 0)}"
    )
    grid.add_row("quality", quality)
    files = r.get("files_changed") or []
    grid.add_row("changed", ", ".join(files) if files else "-")
    if r.get("trace_path"):
        grid.add_row("trace", str(r["trace_path"]))
    color = "green" if ok else "red"
    return Panel(
        grid,
        title=Text("SUCCESS" if ok else "FAILED", style=f"bold {color}"),
        title_align="left",
        border_style=color,
        box=box.ASCII if ascii else box.ROUNDED,
        expand=False,
    )


def _is_span(ev: TraceEvent) -> bool:
    if ev.kind == "retrieval":
        return ev.duration_ms > 0  # 只有初始预检索是 span,其余 retrieval 是 log
    return ev.kind in SPAN_KINDS


def replay_timeline(events: list[TraceEvent]) -> list[tuple[float, TraceEvent, str]]:
    """按发生时间排列的 (t, event, phase):span 事件按 ts - duration 补一个 start;
    llm_call 结束后补一个 assistant_text(trace 里只有 text[:500])。"""
    rows: list[tuple[float, int, int, TraceEvent, str]] = []
    for i, ev in enumerate(events):
        t = ev.ts.timestamp()
        if _is_span(ev):
            rows.append((t - ev.duration_ms / 1000, i, 0, ev, "start"))
        rows.append((t, i, 1, ev, "end"))
        if ev.kind == "llm_call" and ev.payload.get("text"):
            text_ev = TraceEvent.model_construct(
                run_id=ev.run_id,
                step=ev.step,
                ts=ev.ts,
                kind="assistant_text",
                payload={"text": ev.payload["text"]},
            )
            rows.append((t, i, 2, text_ev, "ephemeral"))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return [(t, ev, phase) for t, _, _, ev, phase in rows]


def replay(
    events: list[TraceEvent],
    renderer: Any,
    *,
    speed: float = 5.0,
    max_gap: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """把已有 trace 按 ts 间隔 / speed 依次喂给渲染器(单个间隔最多等 max_gap 秒)。"""
    prev: float | None = None
    for t, ev, phase in replay_timeline(events):
        if prev is not None and speed > 0:
            gap = (t - prev) / speed
            if gap > 0:
                sleep(min(gap, max_gap))
        prev = t
        renderer(ev, phase)
