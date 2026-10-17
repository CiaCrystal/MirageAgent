"""迁移目标注册表(按目录自动发现声明式目标包,不再手工注册)。

    get_target(name) -> MigrationTarget     available_targets() -> list[str]     default_target_name() -> str
    set_active(target) / active()           当前 run 的目标(通用层取库相关规则的入口;assemble 按 task.target 设置)

目录 `migration/targets/<dir>/target.yaml` 即一个目标包;恰好一个目标包声明 `default: true`(未指定目标时使用)。
额外的目标包目录可经环境变量 `MIGRATION_AGENT_TARGET_PATHS`(os.pathsep 分隔)加入。
"""

from __future__ import annotations

import contextvars
import os
from functools import lru_cache
from pathlib import Path

from .base import MigrationTarget
from .package import PackageTarget, TargetPackageError

ROOT = Path(__file__).resolve().parent


def _roots() -> list[Path]:
    extra = [Path(p) for p in os.environ.get("MIGRATION_AGENT_TARGET_PATHS", "").split(os.pathsep) if p]
    return [ROOT, *extra]


@lru_cache(maxsize=4)
def _discover(roots: tuple[Path, ...]) -> dict[str, PackageTarget]:
    found: dict[str, PackageTarget] = {}
    for base in roots:
        dirs = [base] if (base / "target.yaml").exists() else sorted(p for p in base.iterdir() if p.is_dir())
        for d in dirs:
            if not (d / "target.yaml").exists():
                continue
            t = PackageTarget.load(d)
            if t.name in found:
                raise TargetPackageError(f"duplicate target name {t.name!r}: {found[t.name].root} and {d}")
            found[t.name] = t
    defaults = [n for n, t in found.items() if t.spec.default]
    if len(defaults) != 1:
        raise TargetPackageError(f"exactly one target package must set default: true (found {defaults})")
    return found


def _targets() -> dict[str, PackageTarget]:
    return _discover(tuple(_roots()))


def get_target(name: str | None = None) -> MigrationTarget:
    targets = _targets()
    if name is None:
        name = default_target_name()
    try:
        return targets[name]
    except KeyError:
        raise KeyError(f"unknown migration target '{name}'. Available: {', '.join(targets)}") from None


def available_targets() -> list[str]:
    return list(_targets())


def default_target_name() -> str:
    return next(n for n, t in _targets().items() if t.spec.default)


_ACTIVE: contextvars.ContextVar[MigrationTarget | None] = contextvars.ContextVar(
    "migration_target", default=None
)
_GLOBAL: list[MigrationTarget] = []


def set_active(target: MigrationTarget | str | None) -> MigrationTarget:
    """设置当前目标(进程级 + 当前上下文);返回目标对象。"""
    t = get_target(target) if (target is None or isinstance(target, str)) else target
    _ACTIVE.set(t)
    _GLOBAL[:] = [t]
    return t


def active() -> MigrationTarget:
    t = _ACTIVE.get()
    if t is not None:
        return t
    return _GLOBAL[0] if _GLOBAL else get_target(None)


__all__ = [
    "MigrationTarget",
    "PackageTarget",
    "TargetPackageError",
    "active",
    "available_targets",
    "default_target_name",
    "get_target",
    "set_active",
]
