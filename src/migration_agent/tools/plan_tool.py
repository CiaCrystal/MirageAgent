"""update_plan / view_plan:模型与程序共同维护的计划工具。参考 Claude Code TodoWrite、Codex update_plan。

工具
    ViewPlanTool    name="view_plan"     无参数;返回 PlanState.render(),不修改仓库
    UpdatePlanTool  name="update_plan"   args: ops=[{"op": "add|split|set_status|note", "id"?, "goal"?, "files"?,
                                                     "parts"?, "status"?, "note"?}]

约束(由 PlanState.apply_model_update 执行)
    - 模型可以新增、拆分、补充说明、把项标为 active / blocked / pending;
    - 不能把程序判定未完成(关联签名仍在失败)的项标成 done;不能删除项,不能标 abandoned;
    - files 必须存在且不受只读保护、未被锁定;违反的操作逐条拒绝并说明原因。

接入
    只在 dynamic_plan 开启时注册(agent/dynamic_plan.py);mutates_repo=False,不触发自动测试;每次更新写 plan 事件。
    `host` 是 DynamicPlanHook:提供 state、file_ok(path) 与 record(...)(写 trace)。
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult


class ViewPlanTool(Tool):
    name = "view_plan"
    description = (
        "Show the current migration plan: items, their status (tracked from the test results) and approach."
    )
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, host: Any):
        self.host = host

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self.host.state is None:
            return ToolResult.error("no plan yet")
        return ToolResult(
            content=self.host.state.render(), metadata={"plan_version": self.host.state.version}
        )


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = (
        "Update the migration plan. ops is a list of operations: "
        '{"op":"add","goal":...,"files":[...]} adds an item; '
        '{"op":"split","id":"p3","parts":[{"goal":...,"files":[...]}]} splits an item; '
        '{"op":"set_status","id":"p3","status":"active|blocked|pending|done"} changes status '
        "(done is refused while the item's linked failures still fail — statuses follow the test results); "
        '{"op":"note","id":"p3","note":...} records a finding. Items cannot be deleted.'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "split", "set_status", "note"]},
                        "id": {"type": "string"},
                        "goal": {"type": "string"},
                        "files": {"type": "array", "items": {"type": "string"}},
                        "parts": {"type": "array", "items": {"type": "object"}},
                        "status": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["op"],
                },
            }
        },
        "required": ["ops"],
    }

    def __init__(self, host: Any):
        self.host = host

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        state = self.host.state
        if state is None:
            return ToolResult.error("no plan yet")
        ops = args.get("ops")
        if not isinstance(ops, list) or not ops:
            return ToolResult.error("ops must be a non-empty list")
        applied, rejected = state.apply_model_update(ops, self.host.file_ok)
        self.host.record("model", applied=len(applied), rejected=[[o, why] for o, why in rejected])
        lines = [f"applied {len(applied)} op(s), rejected {len(rejected)}"]
        lines += [f"  rejected {o.get('op')} {o.get('id', '')}: {why}" for o, why in rejected]
        lines.append(state.render())
        return ToolResult(
            content="\n".join(lines),
            is_error=bool(rejected) and not applied,
            metadata={"plan_version": state.version, "rejected": len(rejected)},
        )
