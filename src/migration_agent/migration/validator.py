"""Hard repository invariants and non-blocking migration-quality measurements."""

from __future__ import annotations

import ast
import base64
import configparser
import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ..tools.base import ToolContext


@dataclass
class ValidationResult:
    violations: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    deprecated_api_count: int = 0
    compatibility_import_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def _text(entry: dict | None) -> str:
    return base64.b64decode(entry["data"]).decode("utf-8", "replace") if entry else ""


def _skip_calls(tree: ast.AST) -> Counter:
    aliases = {"pytest": "pytest", "unittest": "unittest"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    out = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            root, _, suffix = name.partition(".")
            name = aliases.get(root, root) + ("." + suffix if suffix else "")
            if name in (
                "pytest.skip",
                "pytest.xfail",
                "pytest.mark.skip",
                "pytest.mark.skipif",
                "pytest.mark.xfail",
                "unittest.SkipTest",
                "unittest.skip",
                "unittest.skipIf",
                "unittest.skipUnless",
            ):
                out[ast.dump(node, include_attributes=False)] += 1
    return out


def _target():
    from .targets import active

    return active()


def compat_shim_message(target=None) -> str:
    """Model-visible end message when tests pass but compatibility-layer imports remain (AgentLoop._finish)."""
    return (target or _target()).text(
        "loop.compat_shim_found", "Tests pass, but {compat_layer} compatibility imports remain."
    )


def _pins(text: str, rule=None) -> Counter:
    rule = rule or _target().quality_rules.pin
    pins = Counter()
    pattern = r"(?i)\b" + rule.package_regex + r"(?:\[[^\]]+\])?\s*((?:[<>=!~]=?\s*\d[^\s\]\"\'\n;]*)?)"
    for match in re.finditer(pattern, text):
        raw = match.group(1).rstrip(",")
        if not raw:
            continue
        try:
            spec = SpecifierSet(raw)
        except InvalidSpecifier:
            continue
        # All finite boundaries plus representative new-major releases; avoid enumerating patch versions.
        versions = set(rule.probe_versions)
        versions.update(s.version.rstrip(".*") for s in spec if s.version.startswith(rule.major_prefix))
        if not any(spec.contains(v, prereleases=True) for v in versions):
            pins[raw] += 1
    # e.g. Poetry's ^1.* and Pipfile entries that explicitly restrict to the old major.
    for extra in rule.extra_patterns:
        for m in re.finditer(extra, text):
            pins[m.group(1)] += 1
    return pins


def _compat_count(node: ast.Import | ast.ImportFrom, rules) -> int:
    mods = tuple(rules.compat_modules)

    def hit(name: str) -> bool:
        return any(name == m or name.startswith(m + ".") for m in mods)

    if isinstance(node, ast.ImportFrom):
        if not node.module:
            return 0
        n = int(hit(node.module))
        names = rules.compat_from_names.get(node.module)
        if names:
            n += sum(a.name in names for a in node.names)
        return n
    return sum(hit(a.name) for a in node.names)


class Validator:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.target = _target()
        self.baseline = ctx.files.tree()
        self.runtime_version = self._runtime_version()

    def _runtime_version(self) -> str | None:
        res = self.ctx.executor.exec(
            [
                "python",
                "-I",
                "-c",
                "import importlib.metadata, json; print(json.dumps(importlib.metadata.version("
                f"{self.target.version_package!r})))",
            ]
        )
        return json.loads(res.stdout) if res.ok else None

    def restore_runtime(self) -> None:
        """Restore the target package if a shell command downgraded/removed it."""
        if self.runtime_version and self._runtime_version() != self.runtime_version:
            res = self.ctx.executor.exec(
                ["python", "-m", "pip", "install", f"{self.target.version_package}=={self.runtime_version}"],
                timeout=600,
            )
            if not res.ok:
                raise RuntimeError("could not restore target runtime: " + res.stderr)

    def validate(self, before: dict | None = None, after: dict | None = None) -> ValidationResult:
        before = self.baseline if before is None else before
        after = self.ctx.files.tree() if after is None else after
        result = ValidationResult(
            changed_files=sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
        )
        runtime = self._runtime_version()
        rules = self.target.quality_rules
        major = rules.pin.major_prefix
        lib = self.target.label("library")
        if (
            self.runtime_version
            and self.runtime_version.startswith(major)
            and (not runtime or not runtime.startswith(major))
        ):
            result.violations.append(f"{lib} runtime downgraded or removed")
        for rel in set(self.baseline) | set(after):
            if self.ctx.is_protected(rel) and self.baseline.get(rel) != after.get(rel):
                result.violations.append(f"test file is read-only: {rel}")
        for rel in result.changed_files:
            old, new = _text(before.get(rel)), _text(after.get(rel))
            name = PurePosixPath(rel).name
            if name in ("pytest.ini", "tox.ini") and old != new:
                result.violations.append(f"test configuration is read-only: {rel}")
            if name == "pyproject.toml":
                try:
                    a, b = tomllib.loads(old), tomllib.loads(new)
                    if a.get("tool", {}).get("pytest") != b.get("tool", {}).get("pytest"):
                        result.violations.append(f"pytest configuration changed: {rel}")
                except tomllib.TOMLDecodeError:
                    result.violations.append(f"invalid TOML: {rel}")
            if name == "setup.cfg":
                try:
                    a, b = configparser.ConfigParser(), configparser.ConfigParser()
                    a.read_string(old)
                    b.read_string(new)
                    old_pytest = dict(a["tool:pytest"]) if a.has_section("tool:pytest") else {}
                    new_pytest = dict(b["tool:pytest"]) if b.has_section("tool:pytest") else {}
                    if old_pytest != new_pytest:
                        result.violations.append(f"pytest configuration changed: {rel}")
                except configparser.Error:
                    result.violations.append(f"invalid setup.cfg: {rel}")
            if (
                name.endswith((".txt", ".toml", ".cfg", ".lock")) or name in ("setup.py", "Pipfile")
            ) and _pins(new) - _pins(old):
                result.violations.append(f"{lib} dependency excludes {self.target.label('new')}: {rel}")
            if rel.endswith(".py") and rel in after:
                try:
                    tree = ast.parse(new, filename=rel)
                except SyntaxError as e:
                    result.violations.append(f"syntax error: {rel}:{e.lineno}: {e.msg}")
                    continue
                try:
                    previous = _skip_calls(ast.parse(old))
                except SyntaxError:
                    previous = Counter()
                if _skip_calls(tree) - previous:
                    result.violations.append(f"new skip/xfail: {rel}")
        # Static candidates only: method names can also belong to objects outside the target library.
        legacy = rules.legacy_api
        attrs, nested, calls = set(legacy.attrs), set(legacy.nested_classes), set(legacy.calls)
        for rel, entry in after.items():
            if not rel.endswith(".py") or self.ctx.is_protected(rel):
                continue
            try:
                tree = ast.parse(_text(entry))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in attrs:
                    result.deprecated_api_count += 1
                if (
                    calls
                    and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in calls
                ):
                    result.deprecated_api_count += 1
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    result.compatibility_import_count += _compat_count(node, rules)
                if isinstance(node, ast.ImportFrom) and node.module in legacy.imported_names:
                    result.deprecated_api_count += sum(
                        a.name in legacy.imported_names[node.module] for a in node.names
                    )
                if isinstance(node, ast.ClassDef) and node.name in nested:
                    result.deprecated_api_count += 1
        return result
