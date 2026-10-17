"""装配模型、工具、沙箱、检索器与运行钩子。baseline 提供文件、命令与测试工具；full 加入知识库、代码库和可选经验检索、动态规划。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from ..config import Settings
from ..migration.planner import Planner, plan_briefing
from ..migration.strategy import LEVELS_REPLAN
from ..migration.targets import set_active
from ..migration.task import MigrationTask
from ..tools.base import ToolContext
from ..tools.fs import fs_tools
from ..tools.policy import CommandPolicyTool, FileLockTool, project_names
from ..tools.pytest_tool import RunTestsTool
from ..tools.registry import ToolRegistry
from ..tools.sandbox.docker_sandbox import DockerSandbox
from ..tools.sandbox.local import LocalExecutor
from ..tools.sandbox.protocol import Executor
from ..tools.search_tools import (
    FindUsagesTool,
    RepoCodebase,
    SearchCodebaseTool,
    SearchExperienceTool,
    SearchMigrationDocsTool,
    experience_briefing,
    knowledge_briefing,
)
from ..tools.shell import RunCommandTool
from ..trace.tracer import Tracer
from .compaction import ContextCompactor
from .llm import AnthropicClient, LLMClient
from .loop import AgentLoop
from .state import Budget
from .supervisor import SupervisedLLM, SupervisorConfig, SupervisorHook

TOOL_PROFILES = ("baseline", "full")

# 通用模板({...} 由目标包 labels 填充);目标包 texts 里有同名键(assemble.*)时原样使用
KNOWLEDGE_HINT = (
    "A `search_migration_docs` tool searches the official {Library} {new} migration guide, docs, error-code "
    "reference, release notes and {old}/{new} GitHub issues. When a failure is not covered by the quick reference or "
    "you are unsure how {new} behaves, search with the exact error message before editing."
)
CODEBASE_HINT = (
    "`find_usages` lists every statically resolved use of a symbol in this repo (e.g. `{example_symbol}`, "
    "`{example_attr_symbol}`) and `search_codebase` finds the functions/classes responsible for a behaviour. Use "
    "them to locate all affected code instead of reading files one by one, and re-run `find_usages` after editing "
    "to check nothing was missed."
)
MEMORY_HINT = (
    "`search_experience` returns reviewed fixes that resolved the same error in other repositories' migrations "
    "(minimal diff + test evidence). Query it with the exact error text when a failure is unfamiliar or persists; "
    "treat hits as hints, adapt them to this code, and never fix errors with {compat_layer} imports or by patching "
    "{library} at runtime."
)


def assemble(
    task: MigrationTask,
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    executor: Executor | None = None,
    tracer: Tracer | None = None,
    tool_profile: str = "baseline",
    python_executable: str | None = None,
    knowledge=None,
    memory=None,
    pricing=None,
    observers: list[Callable] | None = None,
) -> tuple[AgentLoop, ToolContext]:
    """`knowledge`:full 档使用的检索器(需有 `retrieve`);为空时按 settings.knowledge_dir 加载共享实例。
    `memory`:经验库(MemorySnapshot / ExperienceStore);仅 full 档且 memory_enabled 时使用。
    `observers`:trace 监听器(如实时 UI),逐个 add_listener 到本次的 tracer(外部传入的 tracer 同样适用)。"""
    if tool_profile not in TOOL_PROFILES:
        raise ValueError(f"unknown tool_profile {tool_profile!r}; choose from {TOOL_PROFILES}")

    repo = Path(task.repo_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repo not found: {repo}")
    target = set_active(task.target)  # 通用层经 migration.targets.active() 取库相关规则

    if executor is None:
        if python_executable or settings.sandbox_backend == "local":
            executor = LocalExecutor(repo, python_executable=python_executable)
        elif settings.sandbox_backend == "docker":
            executor = DockerSandbox(repo, image=settings.sandbox_image)
        else:
            raise ValueError(f"unknown sandbox_backend: {settings.sandbox_backend}")
    ctx = ToolContext(repo_path=repo, executor=executor, tool_output_max_chars=settings.tool_output_max_chars)
    ctx.protect_test_cmd(task.test_cmd)  # test_cmd 点名的测试文件(如根目录 test.py)同样只读

    test_tool = RunTestsTool(test_cmd=task.test_cmd, timeout=settings.pytest_timeout_seconds)
    full = tool_profile == "full"
    locks: dict[str, str] = {}  # SupervisedLLM 写入,FileLockTool 读取
    tools = [*fs_tools(), RunCommandTool()]
    hooked = settings.extension_mode == "hooks"
    if settings.extension_mode not in ("hooks", "wrappers"):
        raise ValueError(f"unknown extension_mode {settings.extension_mode!r}")
    if full and settings.strategy_controller and not hooked:
        tools = [
            FileLockTool(t, lambda: locks) if t.mutates_repo and t.name != "run_command" else t for t in tools
        ]
    if full and settings.command_policy and not hooked:
        own = project_names(repo)
        tools = [CommandPolicyTool(t, own) if t.name == "run_command" else t for t in tools]
    registry = ToolRegistry([*tools, test_tool])
    hints: list[str] = []
    providers: list = []
    codebase = None
    dyn = None  # DynamicPlanHook
    memory_hook = None  # MemoryHook(新签名补查 + 复用反馈)
    if full:
        if knowledge is None:
            from ..rag.knowledge.retriever import shared_retriever

            knowledge = shared_retriever(settings.knowledge_dir)
        registry.register(
            SearchMigrationDocsTool(knowledge, default_k=settings.knowledge_top_k, target=target)
        )
        hints.append(target.text("assemble.knowledge_hint", KNOWLEDGE_HINT))
        if settings.knowledge_preretrieve:
            providers.append(partial(knowledge_briefing, knowledge))
        if settings.codebase_tools:
            codebase = RepoCodebase(settings.codebase_embedding_model)
            registry.register(SearchCodebaseTool(codebase))
            registry.register(FindUsagesTool(codebase, target=target))
            hints.append(target.text("assemble.codebase_hint", CODEBASE_HINT))
            if settings.planner_briefing:
                planner = Planner(target, ctx.is_protected)
                providers.append(plan_briefing(planner, codebase, ctx))
        if hooked and settings.dynamic_plan:
            from .dynamic_plan import PLAN_HINT, DynamicPlanHook

            dyn = DynamicPlanHook(
                settings,
                ctx,
                target,
                planner=Planner(target, ctx.is_protected) if codebase is not None else None,
                codebase=codebase,
                test_cmd=task.test_cmd,
            )
            hints.append(PLAN_HINT)
            providers.append(dyn.briefing)
        if settings.memory_enabled and (
            memory is not None or settings.resolve_memory_dir(target) is not None
        ):
            retriever, session = memory_components(task, settings, memory)
            registry.register(
                SearchExperienceTool(
                    retriever,
                    session,
                    default_k=settings.memory_top_k,
                    max_tokens=settings.memory_tool_max_tokens,
                    target=target,
                )
            )
            hints.append(target.text("assemble.memory_hint", MEMORY_HINT))
            if settings.memory_preretrieve:
                providers.append(
                    partial(
                        experience_briefing,
                        retriever,
                        session,
                        ctx,
                        k=settings.memory_top_k,
                        max_tokens=settings.memory_initial_max_tokens,
                    )
                )
            if hooked and (settings.memory_requery_new_signatures or settings.memory_feedback):
                memory_hook = _memory_hook(retriever, session, settings)
    extra_system_context = "\n".join(hints)
    initial_context_provider = _compose(providers) if providers else None

    tracer = tracer or Tracer(settings.traces_dir / f"{task.run_id or _ts()}.jsonl", run_id=task.run_id)
    for fn in observers or ():
        tracer.add_listener(fn)
    if (
        settings.protect_collected_tests
    ):  # 默认关闭;pytest 实际收集到的测试文件只读(覆盖 ini 里的 testpaths 等)
        from ..tools.base import collect_test_files

        collected, why = collect_test_files(executor, task.test_cmd, repo, settings.pytest_timeout_seconds)
        ctx.protect_files(collected)
        tracer.log(
            "test_protection", {"collected": len(collected), "files": list(collected[:200]), "error": why}
        )
    llm = llm or AnthropicClient(model=settings.model, max_tokens=settings.max_output_tokens)
    if settings.llm_cache_url:  # 默认关闭;关闭时这里什么都不做
        from .llm_cache import wrap_from_settings

        llm = wrap_from_settings(llm, settings, pricing=pricing, tracer=tracer)
    budget = Budget(
        max_steps=task.max_steps or settings.max_steps,
        max_total_tokens=task.max_total_tokens or settings.max_total_tokens,
        max_wall_seconds=task.max_wall_seconds or settings.max_wall_seconds,
        token_mode=settings.budget_token_mode,
        max_cost_usd=settings.max_cost_usd,
        pricing=pricing,
    )
    if (settings.budget_token_mode == "priced" or settings.max_cost_usd is not None) and pricing is None:
        budget.token_mode, budget.max_cost_usd = "raw", None
        budget.fallback = "no pricing supplied: priced token budget and max_cost_usd disabled"
    hooks = None
    if hooked:
        from ..tools.permissions import PolicyEngine, load_rules
        from .hooks import HookBus

        rule_sets = ["base"] + (["command"] if full and settings.command_policy else [])
        hooks = HookBus(max_continues=settings.stop_gate_max_returns)
        if settings.llm_cache_url:  # run 结束写缓存汇总
            from .llm_cache import LlmCacheSummaryHook, find_cache

            hooks.add(LlmCacheSummaryHook(find_cache(llm), tracer))
        hooks.add(
            PolicyEngine(
                load_rules(*rule_sets),
                dynamic={"locked": lambda: locks},
                own_names=project_names(repo) if full and settings.command_policy else set(),
                tracer=tracer,
            )
        )
    if full and settings.strategy_controller:
        sup = SupervisedLLM(
            llm,
            ctx,
            budget,
            tracer,
            knowledge=knowledge,
            codebase=codebase,
            locks=locks,
            config=SupervisorConfig(
                repeat_threshold=settings.repeat_error_threshold,
                budget_marks=(0.5, 0.75, 0.9) if settings.budget_notices else (),
                exploration_nudge_steps=settings.exploration_nudge_steps if settings.budget_notices else 0,
                **({"levels": LEVELS_REPLAN} if dyn is not None else {}),
            ),
        )
        if dyn is not None:
            sup.replan_handler = dyn.on_repeat_error
            dyn.supervisor = sup
        if hooked:
            hooks.add(SupervisorHook(sup))
        else:
            llm = sup
    if hooked:
        from .harness import add_harness

        llm, context_manager, router = add_harness(
            hooks, registry, settings, llm, ctx, budget, pricing, need_router=dyn is not None
        )
        if (give_up := hooks.get("give_up_check")) is not None:
            give_up.tracer = tracer
        if dyn is not None:
            dyn.tracer, dyn.budget = tracer, budget
            dyn.attach(router, registry)
            hooks.add(dyn)
        if memory_hook is not None:
            memory_hook.tracer = tracer
            hooks.add(memory_hook)
        loop_router = router
    else:
        context_manager = None
        loop_router = None
    loop_target = target
    if settings.skills_external:  # 默认开启;系统提示展示目录，正文按需加载
        from .skills import LoadSkillTool, SkilledTarget, SkillLibrary

        if target.skills_dir is None:
            raise ValueError(f"skills_external is on but target {target.name!r} has no skills/ directory")
        library = SkillLibrary(target.skills_dir, target)
        registry.register(LoadSkillTool(library))
        loop_target = SkilledTarget(target, library)
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        test_tool=test_tool,
        tracer=tracer,
        budget=budget,
        target=loop_target,
        allow_compatibility_shim=settings.allow_compatibility_shim,
        extra_system_context=extra_system_context,
        initial_context_provider=initial_context_provider,
        hooks=hooks,
        context_manager=context_manager,
        llm_error_retries=settings.llm_error_retries,
        compactor=ContextCompactor(
            enabled=settings.compaction_enabled,
            context_window=settings.context_window_tokens,
            threshold=settings.compaction_threshold,
            keep_recent_turns=settings.compaction_keep_recent_turns,
        ),
    )
    loop.router = loop_router  # run 结束后的经验总结复用同一个 router(同一密钥、按角色记账)
    return loop, ctx


def memory_components(task: MigrationTask, settings: Settings, memory=None):
    """(MemoryRetriever, MemorySession)。快照自带来源策略,检索时再校验一次;签名用 ErrorParser 归一化。"""
    from ..migration.error_parser import ErrorParser
    from ..rag.memory.retriever import MemoryRetriever, MemorySession
    from ..rag.memory.store import open_memory

    if memory is None:
        memory_dir = settings.resolve_memory_dir(set_active(task.target))
        if memory_dir is None:
            return None, None
        memory = open_memory(memory_dir, require_snapshot=settings.memory_require_snapshot)
    parser = ErrorParser()
    hybrid = settings.memory_retrieval == "hybrid"
    retriever = MemoryRetriever(
        memory,
        target=task.target,
        policy=memory.policy() if hasattr(memory, "policy") and callable(memory.policy) else None,
        min_bm25=settings.memory_min_bm25,
        signature_fn=lambda text: str(parser.parse(text)),
        mode=settings.memory_retrieval,
        embedder=_memory_embedder(settings.memory_embedding_model) if hybrid else None,
    )
    return retriever, MemorySession()


def _memory_embedder(spec: str):
    from ..rag.common.embeddings import make_embedder

    return make_embedder(spec)


def _memory_hook(retriever, session, settings: Settings):
    """新签名补查 + 复用反馈。开发库(非冻结快照)且在线模式时,run 结束把反馈写回效用分;快照只写 trace。"""
    from ..rag.memory.store import ExperienceStore
    from .memory_hooks import MemoryHook

    on_outcome = None
    if settings.memory_online and isinstance(retriever.memory, ExperienceStore):
        from ..rag.memory.consolidate import Consolidator

        on_outcome = Consolidator(retriever.memory, online=True).run_outcome
    return MemoryHook(
        retriever,
        session,
        requery=settings.memory_requery_new_signatures,
        feedback=settings.memory_feedback,
        k=settings.memory_top_k,
        max_tokens=settings.memory_tool_max_tokens,
        on_outcome=on_outcome,
    )


def _compose(providers: list):
    """多个初始上下文提供者 → 一个:文本按顺序拼接,trace 元数据合并(键不冲突)。"""

    def provider(report):
        texts, meta = [], {}
        for p in providers:
            text, m = p(report)
            if text:
                texts.append(text)
            meta.update(m)
        return "\n\n".join(texts), meta

    return provider


def _ts() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")
