"""MigrationTarget 协议:一个「从 X 版本迁到 Y 版本」的领域知识包。

name, from_version, to_version
prompt_hints() -> str          # 注入系统提示的简明迁移速查(废弃 API → 新 API)
deprecated_symbols -> list[str]# Planner 用 usages_of 扫描的符号
forbidden_patterns -> list[str]# Validator 判定「不达标」的模式(如兼容层导入)
runtime_usage_probe -> dict    # 运行时用法探测:warning 类别、环境变量、在哪个版本环境跑、警告文本 → 符号
扩展(由声明式目标包 package.py 实现):
version_package -> str         # 运行时版本检查 / 恢复用的分发名
spec -> TargetSpec             # target.yaml 全文(质量规则、Planner 类型族、报错归一化、失败分类、经验库词表 ...)
quality_rules -> QualityRules  # 库补丁的模块名单、兼容层模式、降级约束、遗留 API 计量
knowledge_spec -> KnowledgeSpec | None   # knowledge.yaml
skills_dir -> Path | None      # skills/*.md
label(key) / text(key, default)          # 模型可见文本:目标专用版本优先,否则用通用模板 + labels
hook(name) -> callable | None  # 可选 hooks.py 中的函数"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MigrationTarget(ABC):
    name: str
    from_version: str
    to_version: str
    knowledge_sources: list[str] = []
    deprecated_symbols: list[str] = []
    forbidden_patterns: list[str] = []
    # 运行时用法探测(migration/runtime_usages.py):{"warning": 警告类全名, "env": {...}, "environment": "target"|"source",
    # "symbols": {警告文本里的反引号名字: deprecated_symbols 中的符号}};None = 不探测
    runtime_usage_probe: dict | None = None
    version_package: str = ""
    spec: Any = None

    @abstractmethod
    def prompt_hints(self) -> str: ...

    def describe(self) -> str:
        return f"{self.name}: {self.from_version} -> {self.to_version}"

    @property
    def quality_rules(self) -> Any:
        return self.spec.quality if self.spec is not None else None

    @property
    def knowledge_spec(self) -> Any:
        return None

    @property
    def skills_dir(self) -> Any:
        return None

    def label(self, key: str) -> str:
        labels = self.spec.labels if self.spec is not None else {}
        return labels.get(key, "")

    def text(self, key: str, default: str = "") -> str:
        """模型可见文本:target.yaml 的 texts[key] 原样优先;否则通用模板按 labels 填充。"""
        if self.spec is not None and key in self.spec.texts:
            return self.spec.texts[key]
        return default.format_map(_Keep(self.spec.labels if self.spec is not None else {}))

    def hook(self, name: str) -> Any:
        return None


class _Keep(dict):
    """format_map 用:labels 里没有的占位符原样保留(留给调用方再 format)。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
