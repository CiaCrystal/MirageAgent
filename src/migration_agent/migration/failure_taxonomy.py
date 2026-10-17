"""失败分类:把失败签名归类,决定重规划模板,也作为经验条目的标签。规则优先,判不出来时才调 LLM。

类别
    env_dependency    最深帧在第三方依赖里(不是目标库、测试框架或标准库),且报错文本没有指向本仓库的包
    collection        收集阶段 / 启动阶段报错(collection_blocked、模块级用例 id),根因在仓库源码
    deprecated_api    仓库代码(或报错文本指向的仓库包)用了目标版本废弃 / 删除的 API
    behaviour_change  无异常的断言失败:错误文案、序列化结果、默认值语义变化
    test_impossible   异常由只读测试代码直接触发,且来自目标库(测试本身用了已删除的写法)
    cascade           本 run 改过文件之后出现、帧落在改过的文件里
    unknown           以上都不符合;例如断言里只看到 HTTP 500,真正的异常不在报告里

规则顺序(先命中者生效):cascade → test_impossible → env_dependency → collection → deprecated_api →
behaviour_change(含 500 → unknown)→ unknown。unknown 的签名在 `llm` 可用时合并成一次 LLM 调用(每 run 上限由调用方控制)。

    classify(failure, *, target, repo_files, changed_files, is_protected, report=None) -> (类别, 规则名)
    classify_report(report, *, target, repo_files, changed_files, is_protected, llm=None) -> {签名: 类别}
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

CATEGORIES = (
    "env_dependency",
    "collection",
    "deprecated_api",
    "behaviour_change",
    "test_impossible",
    "cascade",
    "unknown",
)
INFRA_PACKAGES = ("_pytest", "pytest", "pluggy", "importlib", "unittest", "asyncio", "pytest_asyncio")
# 通用的「API 已删除 / 已移动」措辞;目标包 failure.removed_phrases 可补充库专用措辞
GENERIC_REMOVED_PHRASES = (
    "has been moved",
    "has been removed",
    "is deprecated",
    "no attribute",
    "cannot import name",
)


def _failure_spec():
    from .targets import active

    return active().spec.failure


def target_packages() -> tuple[str, ...]:
    return tuple(_failure_spec().target_packages)


def removed_phrases() -> tuple[str, ...]:
    return tuple(_failure_spec().removed_phrases) or GENERIC_REMOVED_PHRASES


_FRAME = re.compile(r'File "([^"]+?\.py)", line (\d+)|^\s*((?:[A-Za-z]:)?[^\s:]+?\.py):(\d+):', re.M)


def _frames(tb: str) -> list[str]:
    out = []
    for m in _FRAME.finditer(tb or ""):
        out.append((m.group(1) or m.group(3)).replace("\\", "/"))
    return out


def _site_package(path: str) -> str:
    """site-packages 下的顶层包名;不在 site-packages 返回空串;标准库返回 "<stdlib>"。"""
    if "site-packages/" in path:
        return path.split("site-packages/", 1)[1].split("/", 1)[0].removesuffix(".py")
    if re.search(r"/lib/python3[\d.]*/", path) or re.search(r"/Lib/(?!site-packages)", path):
        return "<stdlib>"
    return ""


def _repo_path(path: str, repo_files: set[str]) -> str:
    p = path.removeprefix("./")
    if p in repo_files:
        return p
    parts = p.split("/")
    for i in range(1, len(parts)):
        if "/".join(parts[i:]) in repo_files:
            return "/".join(parts[i:])
    return "" if p.startswith("/") or re.match(r"^[A-Za-z]:/", p) else p


def _exc_type(sig: str) -> str:
    for part in sig.split(":")[:2]:
        part = part.strip()
        if re.fullmatch(r"[\w.]+(?:Error|Exception|Warning)", part):
            return part.rsplit(".", 1)[-1]
    return ""


def _repo_packages(repo_files: set[str], is_protected) -> set[str]:
    names: set[str] = set()
    for f in repo_files:
        if not f.endswith(".py") or is_protected(f):
            continue
        parts = f.split("/")
        for p in parts[:-1]:
            if p not in ("src", "lib") and p.isidentifier():
                names.add(p)
        if len(parts) == 1:
            names.add(parts[0].removesuffix(".py"))
    return names - {"tests", "test", "docs", "examples", "scripts"}


def classify(
    failure: Any,
    *,
    target: Any = None,
    repo_files: set[str] | None = None,
    changed_files: set[str] | None = None,
    is_protected: Callable[[str], bool] | None = None,
    report: Any = None,
    defines: Callable[[str], list[str]] | None = None,
) -> tuple[str, str]:
    """`defines(类名) -> 定义该类的仓库文件`(可选):区分测试构造的模型是在只读测试里定义的(test_impossible)
    还是在源码里定义的(源码模型的 v2 语义变化,behaviour_change)。"""
    repo_files = repo_files or set()
    changed = changed_files or set()
    is_protected = is_protected or (lambda _p: False)
    sig = failure.signature or ""
    tb = failure.traceback or ""
    text = f"{sig}\n{failure.message or ''}\n{tb[-1500:]}"
    frames = _frames(tb)
    repo_frames = [r for f in frames if not _site_package(f) and (r := _repo_path(f, repo_files))]
    source = [f for f in repo_frames if not is_protected(f)]
    exc = _exc_type(sig)
    is_assert = exc == "AssertionError" or re.match(r"^(failure|error): assert\b", sig) is not None

    # 1. cascade:本 run 改过的文件出现在帧里(只在有改动时)
    if changed and set(source) & changed:
        return "cascade", "frame in a file changed in this run"
    # 2. test_impossible:异常在只读测试代码里触发,来自目标库
    raising = repo_frames[-1] if repo_frames else ""
    deepest = frames[-1] if frames else ""
    spec = _failure_spec()
    target_raised = _site_package(deepest) in target_packages() or exc in tuple(spec.target_raised_exceptions)
    # 测试调用的是目标库仍然存在的(已弃用)模型方法时,失败通常来自源码里模型的配置(如 orm_mode → from_attributes),

    methods = spec.deprecated_methods
    calls_deprecated = bool(methods) and re.search(rf"(?:\.|def )(?:{'|'.join(methods)})\(", tb) is not None
    if raising and is_protected(raising) and not is_assert and target_raised and not calls_deprecated:
        m = re.search(r"validation errors? for (\w+)", text)
        where = defines(m.group(1)) if (m and defines is not None) else []
        if where and not any(is_protected(f) for f in where):
            return "behaviour_change", f"test builds source model {m.group(1)} whose v2 semantics changed"
        return "test_impossible", "raised by read-only test code through the target library"
    # 3. env_dependency:最深帧在第三方依赖里,报错文本没有指向本仓库的包
    third = [
        p
        for f in frames
        if (p := _site_package(f)) and p not in target_packages() + INFRA_PACKAGES + ("<stdlib>",)
    ]
    deepest_pkg = _site_package(deepest)
    own = _repo_packages(repo_files, is_protected)
    mentions_own = any(re.search(rf"\b{re.escape(n)}\b", (failure.message or "") + sig) for n in own)
    if third and deepest_pkg == third[-1] and not mentions_own and not is_assert:
        last_repo = max((i for i, f in enumerate(frames) if f and not _site_package(f)), default=-1)
        last_third = max(i for i, f in enumerate(frames) if _site_package(f) == third[-1])
        if last_third > last_repo:
            return "env_dependency", f"deepest frame in third-party package {third[-1]}"
    # 4. collection:收集 / 启动阶段报错,根因在仓库源码
    collecting = (
        getattr(report, "collection_blocked", False)
        or failure.test_id == "__pytest_startup__"
        or (failure.kind == "error" and "::" not in failure.test_id)
    )
    if collecting and (source or mentions_own) and not is_assert:
        return "collection", "collection/startup error rooted in repository source"
    # 5. deprecated_api:异常来自仓库代码或报错文本提到删除 / 废弃的 API
    if not is_assert and (source or mentions_own) and exc:
        return "deprecated_api", "exception raised through repository source"
    if (
        not is_assert
        and exc
        and (any(p in text for p in removed_phrases()) or re.search(r"\[[\w.@]+\]$", sig))
    ):
        return "deprecated_api", "message names a removed or deprecated API"
    # 6. 断言失败:500 说明真正的异常不在报告里 → unknown;其余为行为差异
    if is_assert:
        if re.search(r"\b500\b", text) and "Internal Server Error" in text:
            return "unknown", "assertion on HTTP 500; the server-side error is not in the report"
        return "behaviour_change", "assertion failed without an exception"
    return "unknown", "no rule matched"


LLM_SCHEMA = {"type": "object", "required": ["categories"], "properties": {"categories": {"type": "object"}}}


def classify_report(
    report: Any,
    *,
    target: Any = None,
    repo_files: set[str] | None = None,
    changed_files: set[str] | None = None,
    is_protected: Callable[[str], bool] | None = None,
    llm: Callable[[str, dict], dict] | None = None,
    signatures: set[str] | None = None,
    defines: Callable[[str], list[str]] | None = None,
) -> dict[str, str]:
    """签名 → 类别(每个签名取第一个用例分类)。规则判为 unknown 的签名在 `llm` 可用时合并成一次调用。"""
    firsts: dict[str, Any] = {}
    for f in getattr(report, "failures", None) or []:
        if signatures is None or f.signature in signatures:
            firsts.setdefault(f.signature, f)
    out = {
        s: classify(
            f,
            target=target,
            repo_files=repo_files,
            changed_files=changed_files,
            is_protected=is_protected,
            report=report,
            defines=defines,
        )[0]
        for s, f in firsts.items()
    }
    unknown = [s for s, c in out.items() if c == "unknown"]
    if unknown and llm is not None:
        lines = [
            "Classify each failing root cause of an automated migration "
            f"({target.describe() if target else 'library upgrade'}) into exactly one category: "
            + ", ".join(CATEGORIES)
            + '. Reply {"categories": {"<number>": "<category>"}}.'
        ]
        for i, s in enumerate(unknown, 1):
            lines.append(f"[{i}] {s[:240]}\n{(firsts[s].traceback or '')[-1000:]}")
        try:
            data = llm("\n".join(lines), LLM_SCHEMA)
            for k, v in (data.get("categories") or {}).items():
                idx = int(str(k).strip("[] ")) - 1
                if 0 <= idx < len(unknown) and v in CATEGORIES:
                    out[unknown[idx]] = v
        except Exception:  # noqa: BLE001 — LLM 分类失败保持 unknown
            pass
    return out


def defines_from_texts(texts: dict[str, str]) -> Callable[[str], list[str]]:
    """{文件: 源码} → defines(类名)。"""

    def defines(name: str) -> list[str]:
        pat = re.compile(rf"^\s*class\s+{re.escape(name)}(?![\w])", re.M)
        return sorted(f for f, t in texts.items() if f.endswith(".py") and pat.search(t))

    return defines
