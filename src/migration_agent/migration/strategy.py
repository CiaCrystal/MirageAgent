"""重复报错检测与策略升级(循环检测),纯逻辑,不调用 LLM、不碰沙箱。

    tracker = RepeatErrorTracker(threshold=3, is_protected=ctx.is_protected)
    actions = tracker.observe(report, changed_files, progressed)   # 每看到一份新测试报告调用一次
    for a in actions: a.level ∈ {"requery", "rollback", "abandon"}

计数口径(「一次出现」= 一次未奏效的尝试):
    - 初始报告里出现的签名记 1 次(首次出现);
    - 之后每份新报告里该签名仍在,且自上一份报告以来仓库有改动时:
        a) traceback 中能定位到仓库源文件的签名:改动的文件与这些源文件有交集才记 1 次(针对它的修改没奏效);
        b) 定位不到源文件的签名(如只有测试文件里的断言帧):整体没有进展(通过数没有超过此前最好成绩)、且改动不全是
           其他签名 traceback 里的文件时才记 1 次。
      别处的修改不会让未被触及的签名计数,避免「排队等待修复」的根因被误判为重复;
      两份报告之间仓库没有任何改动(只是重跑测试)时不计数。
    - 第 threshold 次出现 → requery(换检索词);再出现 1 次 → rollback;再出现 1 次 → abandon(放弃该文件)。
      阈值可配,级别只升不降;已放弃的签名不再计数。
    - dynamic_plan:级别序列为 LEVELS_REPLAN = requery → replan(LLM 重规划)→ rollback → abandon。

签名直接用 TestReport 中的 `FailureInfo.signature`(即 ErrorParser 归一化后的 ErrorSignature 字符串)。
签名对应的文件取 traceback 里不受只读保护的仓库源文件帧(最深的在前);取不到时(如报错发生在已安装的依赖里)只放弃签名、不锁文件。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

LEVELS = ("requery", "rollback", "abandon")
LEVELS_REPLAN = ("requery", "replan", "rollback", "abandon")  # dynamic_plan

# pytest 长格式 `path/to/file.py:123: in f`(Windows 上是反斜杠)与 Python 原生 `File "/workspace/x.py", line 12`
_FRAME_PATTERNS = (
    re.compile(r'File "([^"]+?\.py)", line \d+'),
    re.compile(r"^((?:[A-Za-z]:)?[^\s:]+?\.py):\d+:", re.MULTILINE),
)


def source_frames(
    traceback: str, is_protected: Callable[[str], bool], known: set[str] | None = None
) -> list[str]:
    """traceback 中出现的仓库内可编辑 .py 文件,最深的帧在前(去重)。

    `known` 为仓库当前文件集合时,绝对路径(容器内 /workspace/...、Windows 宿主的盘符路径)按最长后缀匹配到仓库文件,
    不在仓库里的(site-packages 等)丢弃;为空时只接受相对路径。
    """
    found: list[tuple[int, str]] = []
    for pat in _FRAME_PATTERNS:
        for m in pat.finditer(traceback or ""):
            path = m.group(1).replace("\\", "/").removeprefix("./")
            if "site-packages" in path or path.startswith(("<", "..")):
                continue
            path = _to_repo_path(path, known)
            if not path or is_protected(path):
                continue
            found.append((m.start(), path))
    found.sort(reverse=True)
    out: list[str] = []
    for _pos, path in found:
        if path not in out:
            out.append(path)
    return out


def _to_repo_path(path: str, known: set[str] | None) -> str | None:
    absolute = path.startswith("/") or re.match(r"^[A-Za-z]:/", path) is not None
    if known is None:
        return None if absolute else path
    if path in known:
        return path
    if absolute or path.startswith("workspace/"):
        parts = path.split("/")
        for i in range(1, len(parts)):
            cand = "/".join(parts[i:])
            if cand in known:
                return cand
    return None


@dataclass
class SignatureState:
    signature: str
    count: int = 0
    level: int = 0  # 已触发的级别数(0 = 尚未升级)
    files: list[str] = field(default_factory=list)  # traceback 中的可编辑源文件
    attempted: list[str] = field(default_factory=list)  # 计数时改过的文件(回退归属用)
    tests: int = 0
    example: Any = None  # 最近一次的 FailureInfo
    abandoned: bool = False
    first_seen_report: int = 0


@dataclass
class Escalation:
    level: str
    signature: str
    count: int
    files: list[str]
    attempted: list[str]
    example: Any
    first_seen_report: int


class RepeatErrorTracker:
    def __init__(
        self,
        threshold: int = 3,
        is_protected: Callable[[str], bool] | None = None,
        levels: tuple[str, ...] = LEVELS,
    ):
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self.threshold = threshold
        self.levels = levels
        self.is_protected = is_protected or (lambda _p: False)
        self.states: dict[str, SignatureState] = {}
        self.reports_seen = 0
        self.best_passed = -1
        self.repeat_events = 0

    @staticmethod
    def groups(report: Any) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {}
        for f in getattr(report, "failures", None) or []:
            out.setdefault(f.signature, []).append(f)
        return out

    def observe(
        self, report: Any, changed_files: Iterable[str] = (), repo_files: set[str] | None = None
    ) -> list[Escalation]:
        """`repo_files`:仓库当前文件集合,用于把绝对路径帧映射回仓库相对路径。"""
        self.reports_seen += 1
        changed = set(changed_files)
        passed = int(getattr(report, "passed", 0) or 0)
        progressed = passed > self.best_passed
        actions: list[Escalation] = []
        groups = self.groups(report)
        frames_of: dict[str, list[str]] = {}
        for sig, items in groups.items():
            frames: list[str] = []
            for f in items[:5]:
                for p in source_frames(f.traceback, self.is_protected, repo_files):
                    if p not in frames:
                        frames.append(p)
            frames_of[sig] = frames
        claimed = {p for fr in frames_of.values() for p in fr}
        claimed |= {p for sig in groups if sig in self.states for p in self.states[sig].files}
        for sig, items in groups.items():
            st = self.states.get(sig)
            frames = frames_of[sig]
            if st is None:
                st = self.states[sig] = SignatureState(sig, first_seen_report=self.reports_seen)
                st.count = 1
            elif st.abandoned:
                continue
            else:
                files = set(st.files) | set(frames)
                # 无源文件帧:整体停滞、且改动不全是冲着其他签名的文件去的,才算对它的一次尝试
                attempted = bool(changed & files) if files else bool(changed - claimed) and not progressed
                if not attempted:
                    st.tests, st.example = len(items), items[0]
                    st.files = frames or st.files
                    continue
                st.count += 1
                self.repeat_events += 1
                st.attempted.extend(p for p in sorted(changed) if p not in st.attempted)
            st.files = frames or st.files
            st.tests, st.example = len(items), items[0]
            due = st.count - self.threshold + 1  # 第 threshold 次 → 1 级
            if due > st.level and st.level < len(self.levels):
                st.level += 1
                level = self.levels[st.level - 1]
                if level == "abandon":
                    st.abandoned = True
                actions.append(
                    Escalation(
                        level=level,
                        signature=sig,
                        count=st.count,
                        files=list(st.files),
                        attempted=list(st.attempted),
                        example=st.example,
                        first_seen_report=st.first_seen_report,
                    )
                )
        self.best_passed = max(self.best_passed, passed)
        return actions

    def remaining(self, report: Any) -> list[str]:
        """报告中尚未被放弃的失败签名。"""
        return [s for s in self.groups(report) if not (s in self.states and self.states[s].abandoned)]

    def abandoned(self) -> list[str]:
        return [s for s, st in self.states.items() if st.abandoned]
