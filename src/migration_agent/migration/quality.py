"""经验入库的严格质量门槛(只用于记忆抽取与报告,不改变运行时 Validator)。

测试全绿不是充分条件。对一个 run 的最终产出(相对初始源码)与每个候选窗口的改动,分别检查:
  - 运行时修改目标库模块或对象(属性赋值、setattr、sys.modules;模块名单 = 目标的 quality.patch_roots);
  - 新增目标声明的兼容层 import(quality.compat_modules / compat_from_names);
  - 修改受保护测试文件或 pytest 配置(pytest.ini / tox.ini / conftest.py,及 setup.cfg/pyproject 中的 pytest 段);
  - 新增排除新版本的目标库约束(依赖降级;quality.pin);
  - 新增 skip / xfail。
库相关的名单全部来自目标包(`migration.targets.active()`,或显式传入 `rules`)。"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

from .validator import _pins, _skip_calls


def _rules(rules: Any = None) -> Any:
    if rules is not None:
        return rules
    from .targets import active

    return active().quality_rules


def _in_roots(name: str, roots: tuple[str, ...]) -> bool:
    return any(name == r or name.startswith(r + ".") for r in roots)


def patch_sites(source: str, rules: Any = None) -> list[tuple[int, str]]:
    """对目标库对象的运行时改写:给目标库模块/类的属性赋值、setattr(<lib>..., ...)、写 sys.modules['<lib>...']。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    roots = tuple(_rules(rules).patch_roots)
    bound: set[str] = set()  # 绑定到目标库(子)模块或其中对象的本地名
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _in_roots(a.name, roots):
                    bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in roots:
            for a in node.names:
                bound.add(a.asname or a.name)

    def root(expr) -> str | None:
        while isinstance(expr, ast.Attribute):
            expr = expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    sites = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Attribute) and root(t) in bound:
                sites.append((node.lineno, ast.unparse(t)))
            if (
                isinstance(t, ast.Subscript)
                and ast.unparse(t.value) == "sys.modules"
                and any(r in ast.unparse(t.slice) for r in roots)
            ):
                sites.append((node.lineno, ast.unparse(t)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and node.args
            and root(node.args[0]) in bound
        ):
            sites.append((node.lineno, ast.unparse(node)[:80]))
    return sites


def compat_imports(source: str, rules: Any = None) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    r = _rules(rules)
    mods, from_names = tuple(r.compat_modules), r.compat_from_names
    if not mods and not from_names:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if _in_roots(node.module, mods):
                n += 1
            elif node.module in from_names:
                n += sum(a.name in from_names[node.module] for a in node.names)
        elif isinstance(node, ast.Import):
            n += sum(_in_roots(a.name, mods) for a in node.names)
    return n


TEST_CONFIG_NAMES = ("pytest.ini", "tox.ini", "conftest.py")
DEPENDENCY_SUFFIXES = (".txt", ".toml", ".cfg", ".lock")


def file_change_issues(rel: str, old: str, new: str, is_protected: Callable[[str], bool]) -> list[str]:
    """单个文件 old → new 的不合格改动。old 为空表示新增文件,new 为空且文件被删除时传 ''。"""
    if old == new:
        return []
    issues = []
    name = PurePosixPath(rel).name
    if is_protected(rel):
        issues.append(f"test file changed: {rel}")
    if name in TEST_CONFIG_NAMES:
        issues.append(f"test configuration changed: {rel}")
    if name in ("setup.cfg", "pyproject.toml") and _pytest_section(old) != _pytest_section(new):
        issues.append(f"pytest configuration changed: {rel}")
    if (name.endswith(DEPENDENCY_SUFFIXES) or name in ("setup.py", "Pipfile")) and (_pins(new) - _pins(old)):
        issues.append(f"{_label('library')} pin excluding {_label('new')} added: {rel}")
    if rel.endswith(".py") and new:
        try:
            new_tree = ast.parse(new)
        except SyntaxError:
            return [*issues, f"syntax error: {rel}"]
        try:
            old_tree = ast.parse(old) if old else ast.parse("")
        except SyntaxError:
            old_tree = ast.parse("")
        if _skip_calls(new_tree) - _skip_calls(old_tree):
            issues.append(f"new skip/xfail: {rel}")
        before = {text for _, text in patch_sites(old)}
        added = [f"{rel}:{line}: {text}" for line, text in patch_sites(new) if text not in before]
        issues += [f"library patch: {a}" for a in added]
        if compat_imports(new) > compat_imports(old):
            issues.append(f"{_label('compat_layer')} compatibility import added: {rel}")
    return issues


def _label(key: str) -> str:
    from .targets import active

    return active().label(key)


def _pytest_section(text: str) -> str:
    """setup.cfg 的 [tool:pytest] / pyproject 的 [tool.pytest...] 段原文(粗略,足以发现改动)。"""
    out, keep = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("["):
            keep = s.startswith(("[tool:pytest", "[pytest", "[tool.pytest"))
        if keep:
            out.append(ln)
    return "\n".join(out)


def changes_issues(
    before: dict[str, str], after: dict[str, str], is_protected: Callable[[str], bool]
) -> list[str]:
    """两个文件快照({相对路径: 文本})之间所有改动的不合格项。"""
    issues = []
    for rel in sorted(set(before) | set(after)):
        old, new = before.get(rel, ""), after.get(rel, "")
        if old != new:
            issues += file_change_issues(rel, old, new, is_protected)
    return issues
