"""按配置添加上下文管理、工具约束、预算和模型角色路由。"""

from __future__ import annotations

from typing import Any


def add_harness(
    hooks, registry, settings, llm, ctx, budget, pricing, *, need_router: bool = False
) -> tuple[Any, Any, Any]:
    """返回 (llm, context_manager, router);`need_router`:动态规划需要 planner 角色。"""
    context_manager = None
    router = None
    if (
        need_router
        or any((settings.model_planner, settings.model_summarizer, settings.model_critic))
        or settings.context_strategy == "summary"
    ):
        from .router import ModelRouter

        router = ModelRouter.from_settings(settings, executor=llm, pricing=pricing)
        llm = router.client("executor")
    if settings.aci_tools:
        from ..tools.aci import AciHook, aci_tools

        for tool in aci_tools():
            registry.register(tool)
        hooks.add(AciHook(max_chars=settings.tool_output_max_chars))
    if settings.context_strategy != "placeholder":
        from .context import build_context_manager

        context_manager = build_context_manager(settings, router)
    if settings.stop_gate:
        from ..migration.stop_gate import StopGate

        hooks.add(StopGate(max_returns=settings.stop_gate_max_returns))
    if settings.give_up_check:  # 默认关闭;agent 要放弃时核对剩余失败是否经过可编辑源码
        from ..migration.give_up_check import GiveUpCheck

        hooks.add(GiveUpCheck(max_returns=settings.give_up_check_max_returns))
    return llm, context_manager, router
