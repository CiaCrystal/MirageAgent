"""目标包 `target.yaml` 的 schema(声明式目标定义)。加载时校验,错误在加载时报出(未知字段一律拒绝)。

目标包目录:`migration/targets/<name>/`
    target.yaml       本 schema(版本、废弃符号、Planner 类型族、质量规则、运行时探测、报错归一化、失败分类、
                      经验库词表、检索前缀、提示文本)
    knowledge.yaml    知识源(rag/knowledge/source_spec.py 读取)
    skills/*.md       速查表(agent/skills.py;目录常驻、正文按需)
    hooks.py          可选:只放确实无法声明的逻辑(见 package.py 的 HOOK_NAMES)

通用层只经 `MigrationTarget` / 本 schema 取库相关内容;库名、模块名、API 名都写在这里,不写在通用代码里。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class TypeFamily(_Strict):
    """Planner 的属性符号族:`<族名>.<attr>` 只在接收者被推断为该族类型时计数。"""

    roots: list[str]  # 根类型的完整名;继承(传递地)到它们的仓库类也属于该族
    constructors: list[str] = []  # 在类上调用后返回实例的类方法
    instance_returning: list[str] = []  # 在实例上调用后返回同族实例的方法
    nested_classes: list[str] = []  # `<族名>.<X>` 也包含族内类体里的 `class X:`
    factories: list[str] = []  # 调用后返回该族实例的模块级函数(完整名)
    method_returns: dict[str, str] = {}  # 在该族实例上调用 attr 返回另一族实例:{attr: 族名}
    call_returns: str = ""  # 直接调用该族实例返回另一族实例(如会话工厂实例被调用后返回会话)


class PlannerSpec(_Strict):
    canonical: dict[str, str] = {}  # 子模块别名 → 公开完整名(Planner 与 find_usages 归一化)
    families: dict[str, TypeFamily] = {}
    # 代码检索的文本模式:符号 → 正则(只对代码行计数,用于静态分析表达不了的写法,如 `select([...])`)
    text_patterns: dict[str, str] = {}
    # 跟踪从仓库其他模块导入的模块级变量的类型。

    imported_variable_types: bool = False


class PinRule(_Strict):
    package_regex: str  # 依赖文件中包名的正则(不含版本部分)
    probe_versions: list[str]  # 代表性的目标大版本号;约束一个都不允许 → 视为降级
    major_prefix: str  # 运行时版本必须保持的前缀(如 "2.")
    extra_patterns: list[str] = []  # 其它显式锁回旧版的写法(如 Poetry 的 ^1.*),捕获组 1 为约束原文


class LegacyApi(_Strict):
    """Validator 报告用的旧 API 静态计数(只计量,不判失败)。"""

    attrs: list[str] = []
    imported_names: dict[str, list[str]] = {}  # {模块: [从该模块导入即计数的名字]}
    nested_classes: list[str] = []
    calls: list[str] = []  # `<recv>.<name>(` 形式的调用名(仍受支持的遗留 API,只计量)


class QualityRules(_Strict):
    patch_roots: list[str]  # 运行时改写这些模块(或其中对象)= 库补丁
    compat_modules: list[str] = []  # 兼容层模块(如 `<lib>.v1`)
    compat_from_names: dict[str, list[str]] = {}  # `from <模块> import <名字>` 形式的兼容层
    pin: PinRule
    legacy_api: LegacyApi = LegacyApi()
    forbidden_patterns: list[str] = []  # MigrationTarget.forbidden_patterns(兼容层 / 锁回旧版的正则)
    # replanner / critic 的禁止做法:[(名称, 正则)];名称与顺序是模型可见文本的一部分
    forbidden_approaches: list[tuple[str, str]] = []


class ErrorRule(_Strict):
    """报错归一化:`findall(pattern)` 有结果时,签名 = (exception, 排序去重后用 join 连接, symbol)。"""

    pattern: str
    exception: str
    symbol: str = ""
    join: str = ","


class ErrorSpec(_Strict):
    signature_rules: list[ErrorRule] = []
    skip_line_patterns: list[str] = []  # 异常首行归一化时跳过的提示行(如文档链接行)
    doc_url_code_pattern: str = ""  # 报错里的文档链接 → 错误码(捕获组 1),知识检索用


class FailureSpec(_Strict):
    target_packages: list[str]  # 帧落在这些 site-packages 包里 = 目标库抛出
    library_exceptions: list[str] = []
    target_raised_exceptions: list[str] = []
    impossible_exceptions: list[str] = []
    deprecated_methods: list[str] = []
    removed_phrases: list[str] = []


class MemorySpec(_Strict):
    old_api: dict[str, str] = {}  # 旧 API 标签 → 正则
    new_api: dict[str, str] = {}
    extra_api_names: list[str] = []
    tag_families: list[str] = []  # 空格分隔的同族标签
    generic_names: list[str] = []  # 落地性检查的通用名白名单
    version_key: str = ""  # Experience.versions 的键
    version_range: str = ""
    not_applicable: str = ""  # 自动生成条目的「不适用」说明
    precondition: str = ""  # 「{observed}」处填观察到的版本说明


class TargetSpec(_Strict):
    name: str
    default: bool = False
    from_version: str
    to_version: str
    version_package: str  # importlib.metadata 的分发名
    labels: dict[str, str]  # 提示模板用的名字(library、migration、compat_layer ...)
    prompt_hints: str = ""  # skills 外置关闭时注入系统提示的速查(原样)
    texts: dict[str, str] = {}  # 模型可见文本的目标专用版本(覆盖通用模板)
    knowledge_sources: list[str] = []
    deprecated_symbols: list[str] = []
    planner: PlannerSpec = PlannerSpec()
    runtime_usage_probe: dict[str, Any] | None = None
    quality: QualityRules
    errors: ErrorSpec = ErrorSpec()
    failure: FailureSpec
    memory: MemorySpec = MemorySpec()
    query_prefix: str = ""  # supervisor 重查知识库时的查询前缀

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: dict[str, str]) -> dict[str, str]:
        missing = {"library", "migration"} - set(v)
        if missing:
            raise ValueError(f"labels missing {sorted(missing)}")
        return v

    @field_validator("runtime_usage_probe")
    @classmethod
    def _probe(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        allowed = {
            "warning",
            "categories",
            "env",
            "environment",
            "symbols",
            "source_packages",
            "message_symbols",
        }
        extra = set(v) - allowed
        if extra:
            raise ValueError(f"runtime_usage_probe: unknown keys {sorted(extra)}")
        if "warning" not in v:
            raise ValueError("runtime_usage_probe.warning is required")
        if v.get("environment", "target") not in ("target", "source"):
            raise ValueError("runtime_usage_probe.environment must be 'target' or 'source'")
        return v
