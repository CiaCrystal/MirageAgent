"""在沙箱中运行带警告收集的测试，将运行时废弃 API 用法合并到静态计划。需要旧版本环境时临时切换依赖，结束后恢复；文件副作用同样恢复。收集错误时尽可能继续执行可运行的测试，探测失败则保留静态计划。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .strategy import _to_repo_path

PROBE_ARGS = ("-p", "no:cacheprovider", "-p", "warnings", "--continue-on-collection-errors")
_WARN_LINE = re.compile(r"^\s*((?:[A-Za-z]:)?[^\s:]+?\.py):(\d+): (\w+): (.*)$")


@dataclass
class RuntimeUsage:
    file: str
    line: int
    symbol: str
    message: str
    count: int = 1
    source: str = "runtime"


@dataclass
class ProbeResult:
    usages: list[RuntimeUsage] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    skipped: str = ""
    raw_warnings: int = 0
    returncode: int | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "runtime_usages": len(self.usages),
            "runtime_files": len({u.file for u in self.usages}),
            "runtime_probe_seconds": round(self.seconds, 2),
            "runtime_probe_error": self.error[:200],
            "runtime_probe_skipped": self.skipped,
            "runtime_probe_raw_warnings": self.raw_warnings,
            "runtime_probe_returncode": self.returncode,
        }


def parse_warnings(
    output: str,
    warning: str,
    symbols: dict[str, str],
    is_protected,
    repo_files: set[str],
    message_symbols: dict[str, str] | None = None,
    categories: list[str] | None = None,
) -> tuple[list[RuntimeUsage], int]:
    """pytest warnings summary → 仓库可编辑文件中的用法(按 文件/行/符号 去重计数)。
    `categories`:`-W always::<warning>` 也会放出子类,summary 里显示的是子类名;只计这些类别。"""
    category = warning.rsplit(".", 1)[-1]
    accepted = {category, *(categories or [])}
    found: dict[tuple[str, int, str], RuntimeUsage] = {}
    raw = 0
    for line in output.splitlines():
        m = _WARN_LINE.match(line)
        if not m or m.group(3) not in accepted:
            continue
        raw += 1
        path = m.group(1).replace("\\", "/")
        if "site-packages" in path:
            continue
        rel = _to_repo_path(path.removeprefix("./"), repo_files)
        if rel is None or is_protected(rel):
            continue
        msg = m.group(4)
        names = re.findall(r"`([^`]+)`", msg)
        symbol = next((symbols[n] for n in names if n in symbols), None)
        if symbol is None and message_symbols:  # 警告文本没有反引号名字时按目标声明的正则(按顺序)取符号
            symbol = next((sym for rx, sym in message_symbols.items() if re.search(rx, msg)), None)
        if symbol is None:
            symbol = names[0] if names else category
        key = (rel, int(m.group(2)), symbol)
        if key in found:
            found[key].count += 1
        else:
            found[key] = RuntimeUsage(rel, int(m.group(2)), symbol, msg[:200])
    return sorted(found.values(), key=lambda u: (u.file, u.line, u.symbol)), raw


def probe(ctx: Any, target: Any, test_cmd: list[str], *, timeout: int = 600) -> ProbeResult:
    spec = getattr(target, "runtime_usage_probe", None)
    if not spec:
        return ProbeResult(skipped="target declares no runtime usage probe")
    environment = spec.get("environment", "target")
    if environment == "source" and not spec.get("source_packages"):
        return ProbeResult(skipped="probe needs the source-version environment (no source_packages declared)")
    t0 = time.monotonic()
    res = ProbeResult()
    before = None
    restore: str | None = None
    try:
        before = ctx.files.tree()
        if environment == "source":
            restore = _installed(ctx, target.version_package)
            if not restore:
                raise RuntimeError(f"cannot read the installed {target.version_package} version")
            inst = ctx.executor.exec(
                ["python", "-m", "pip", "install", "-q", *spec["source_packages"]], timeout=timeout
            )
            if not inst.ok:
                raise RuntimeError("source-version install failed: " + (inst.stderr or inst.stdout)[-300:])
        cmd = [*test_cmd, *PROBE_ARGS, "-W", f"always::{spec['warning']}"]
        out = ctx.executor.exec(cmd, timeout=timeout, env=dict(spec.get("env") or {}) or None)
        text = (out.stdout or "") + "\n" + (out.stderr or "")
        res.returncode = out.returncode
        if out.timed_out:
            res.error = "timed out"
        elif out.returncode in (2, 3, 4):  # 0/1 正常,5 = 没有收集到测试
            tail = " ".join(text.strip().splitlines()[-2:])
            res.error = f"pytest exited {out.returncode}: {tail}"[:300]
        res.usages, res.raw_warnings = parse_warnings(
            text,
            spec["warning"],
            spec.get("symbols", {}),
            ctx.is_protected,
            set(before),
            spec.get("message_symbols"),
            spec.get("categories"),
        )
    except Exception as e:  # noqa: BLE001 — 探测失败退回纯静态计划
        res.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        if restore:
            back = ctx.executor.exec(
                ["python", "-m", "pip", "install", "-q", f"{target.version_package}=={restore}"],
                timeout=timeout,
            )
            if not back.ok or _installed(ctx, target.version_package) != restore:
                res.error = (res.error + f"; could not restore {target.version_package}=={restore}")[:300]
        try:
            if before is not None and ctx.files.tree() != before:
                ctx.files.restore(before)  # 测试副作用不留在仓库里
        except Exception as e:  # noqa: BLE001
            res.error = (res.error + f"; restore failed: {e}")[:300]
        res.seconds = time.monotonic() - t0
    return res


def _installed(ctx: Any, package: str) -> str | None:
    res = ctx.executor.exec(
        ["python", "-I", "-c", f"import importlib.metadata as m; print(m.version({package!r}))"], timeout=60
    )
    return (res.stdout.strip() or None) if res.ok else None


def merge(static: list[Any], runtime: list[RuntimeUsage]) -> list[Any]:
    """静态用法(rag.codebase Usage:file/line/symbol)与运行时用法取并集;(文件, 行, 符号) 相同的只保留静态那条。"""
    seen = {(u.file, u.line, u.symbol) for u in static}
    return list(static) + [u for u in runtime if (u.file, u.line, u.symbol) not in seen]
