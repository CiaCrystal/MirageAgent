"""声明式目标包:`migration/targets/<name>/target.yaml` + knowledge.yaml + skills/ + 可选 hooks.py。

    target = PackageTarget.load(Path("migration/targets/<dir>"))
    target.spec.quality.patch_roots / target.text("stop_gate.return", DEFAULT) / target.knowledge_spec

hooks.py 只能定义 HOOK_NAMES 里的函数(其它公开名字在加载时报错),用来放确实无法声明的逻辑;"""

from __future__ import annotations

import importlib.util
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .base import MigrationTarget
from .spec import TargetSpec

HOOK_NAMES = ("runtime_probe", "quality_issues", "classify")


class TargetPackageError(ValueError):
    pass


class PackageTarget(MigrationTarget):
    def __init__(self, root: Path, spec: TargetSpec):
        self.root = root
        self.spec = spec
        self.name = spec.name
        self.from_version = spec.from_version
        self.to_version = spec.to_version
        self.version_package = spec.version_package
        self.knowledge_sources = list(spec.knowledge_sources)
        self.deprecated_symbols = list(spec.deprecated_symbols)
        self.forbidden_patterns = list(spec.quality.forbidden_patterns)
        self.runtime_usage_probe = spec.runtime_usage_probe
        self._hooks = _load_hooks(root)

    @classmethod
    def load(cls, root: Path) -> PackageTarget:
        path = root / "target.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise TargetPackageError(f"{path}: cannot read target.yaml: {e}") from e
        try:
            spec = TargetSpec.model_validate(data)
        except ValidationError as e:
            raise TargetPackageError(f"{path}: invalid target.yaml:\n{e}") from e
        target = cls(root, spec)
        if target.knowledge_path.exists():
            target.knowledge_spec  # noqa: B018 — 加载时校验
        return target

    def prompt_hints(self) -> str:
        return self.spec.prompt_hints

    @property
    def knowledge_path(self) -> Path:
        return self.root / "knowledge.yaml"

    @cached_property
    def knowledge_spec(self) -> Any:
        if not self.knowledge_path.exists():
            return None
        from ...rag.knowledge.source_spec import KnowledgeSpec

        try:
            return KnowledgeSpec.load(self.knowledge_path, target=self.name)
        except (ValueError, ValidationError) as e:
            raise TargetPackageError(f"{self.knowledge_path}: invalid knowledge.yaml:\n{e}") from e

    @property
    def skills_dir(self) -> Path | None:
        d = self.root / "skills"
        return d if d.is_dir() else None

    def hook(self, name: str) -> Any:
        return self._hooks.get(name)

    @property
    def has_hooks(self) -> bool:
        return bool(self._hooks)


def _load_hooks(root: Path) -> dict[str, Any]:
    path = root / "hooks.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"migration_target_hooks_{root.name}", path)
    if spec is None or spec.loader is None:
        raise TargetPackageError(f"{path}: cannot import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {n for n in vars(module) if not n.startswith("_") and callable(getattr(module, n))}
    names -= {
        n
        for n, v in vars(module).items()
        if isinstance(v, type) or getattr(v, "__module__", "") != module.__name__
    }
    unknown = names - set(HOOK_NAMES)
    if unknown:
        raise TargetPackageError(f"{path}: unknown hook(s) {sorted(unknown)}; allowed: {HOOK_NAMES}")
    return {n: getattr(module, n) for n in HOOK_NAMES if hasattr(module, n)}
