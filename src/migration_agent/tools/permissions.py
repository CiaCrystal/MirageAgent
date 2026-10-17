"""PolicyEngine:声明式权限规则。参考 Claude Code permissions(allow / deny 规则)。

取代 tools/policy.py 的 CommandPolicyTool / FileLockTool 包装器与 fs 工具里分散的只读检查:规则写在 YAML 里,
Validator 仍是事后硬闸门,不并入这里。

规则字段:
    id            规则名(进拒绝日志)
    tool          工具名或列表;"*" 表示全部
    path          路径条件:glob(fnmatch,相对仓库根)或动态集合 "{protected}" / "{locked}"
    command_regex 对 run_command 的 command 做 re.search
    predicate     具名谓词(Python 实现,返回拒绝原因或 None):目前有 `command_policy`(命令策略)
    decision      allow | deny;第一条命中的规则生效,全部不命中时 allow
    reason        拒绝原因;`message` 可覆盖返回给模型的完整文本({path} / {reason} / {why} 占位)

内置规则集(`tools/policy_rules/*.yaml`):
    base.yaml / command.yaml:tests-read-only、abandoned-file-lock、command-policy;返回文本与 metadata 与包装器写法逐字相同,"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from ..agent.hooks import Allow, BeforeTool, Deny, Hook
from .base import ToolContext, ToolResult
from .policy import REFUSAL, command_violation

RULES_DIR = Path(__file__).resolve().parent / "policy_rules"
EDIT_TOOLS = ("edit_file", "write_file", "insert_line", "multi_edit")


@dataclass
class Rule:
    id: str
    tool: list[str]
    decision: str = "deny"
    path: str | None = None
    command_regex: str | None = None
    predicate: str | None = None
    reason: str = ""
    message: str | None = None
    meta_key: str | None = None  # 拒绝时 ToolResult.metadata 的键(兼容既有键:policy_refusal / locked_file)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rule:
        tool = d.get("tool", "*")
        return cls(
            id=d["id"],
            tool=[tool] if isinstance(tool, str) else list(tool),
            decision=d.get("decision", "deny"),
            path=d.get("path"),
            command_regex=d.get("command_regex"),
            predicate=d.get("predicate"),
            reason=d.get("reason", ""),
            message=d.get("message"),
            meta_key=d.get("meta_key"),
        )


def load_rules(*names_or_paths: str | Path) -> list[Rule]:
    rules: list[Rule] = []
    for n in names_or_paths:
        p = Path(n)
        if not p.suffix:
            p = RULES_DIR / f"{n}.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rules += [Rule.from_dict(r) for r in data.get("rules", [])]
    return rules


@dataclass
class PolicyEngine(Hook):
    """`dynamic`:动态路径集合({locked} 等),返回 {路径: 原因};`own_names`:command_policy 谓词用的本项目发行版名。"""

    rules: list[Rule]
    dynamic: dict[str, Callable[[], dict[str, str]]] = field(default_factory=dict)
    own_names: set[str] = field(default_factory=set)
    tracer: Any = None
    name: str = "permissions"
    denials: list[dict[str, Any]] = field(default_factory=list)

    def _predicates(self) -> dict[str, Callable[[dict[str, Any]], str | None]]:
        return {
            "command_policy": lambda args: command_violation(str(args.get("command") or ""), self.own_names)
        }

    def decide(self, tool: str, args: dict[str, Any], ctx: ToolContext) -> Allow | Deny:
        for rule in self.rules:
            if "*" not in rule.tool and tool not in rule.tool:
                continue
            hit = self._match(rule, args, ctx)
            if hit is None:
                continue
            if rule.decision == "allow":
                return Allow()
            return self._deny(rule, tool, args, hit)
        return Allow()

    def _match(self, rule: Rule, args: dict[str, Any], ctx: ToolContext) -> dict[str, str] | None:
        """命中返回占位值(path / reason / why),否则 None。"""
        vals: dict[str, str] = {"reason": rule.reason}
        if rule.path is not None:
            if "path" not in args:
                return None
            try:
                path = ctx.canonical(str(args.get("path") or ""))
            except (PermissionError, ValueError, KeyError):
                return None  # 越界等错误交给工具自己报
            vals["path"] = path
            if rule.path == "{protected}":
                if not ctx.is_protected(path):
                    return None
            elif rule.path.startswith("{") and rule.path.endswith("}"):
                items = self.dynamic.get(rule.path[1:-1], dict)()
                if path not in items:
                    return None
                vals["reason"] = items[path]
            elif not fnmatch(path, rule.path):
                return None
        if rule.command_regex is not None and not re.search(
            rule.command_regex, str(args.get("command") or "")
        ):
            return None
        if rule.predicate is not None:
            why = self._predicates()[rule.predicate](args)
            if why is None:
                return None
            vals["why"] = why
        return vals

    def _deny(self, rule: Rule, tool: str, args: dict[str, Any], vals: dict[str, str]) -> Deny:
        template = rule.message or "{reason}"
        if rule.predicate == "command_policy" and rule.message is None:
            text = REFUSAL.format(why=vals["why"])
        else:
            text = template.format(**{"path": "", "why": "", **vals})
        meta = {}
        if rule.meta_key:
            meta[rule.meta_key] = vals.get("why") or vals.get("path") or rule.id
        entry = {"rule": rule.id, "tool": tool, "args": _summary(args)}
        self.denials.append(entry)
        if self.tracer is not None:
            self.tracer.log("permission", entry)
        return Deny(ToolResult.error(text, **meta), rule=rule.id)

    # Hook
    def before_tool(self, ev: BeforeTool) -> Allow | Deny:
        return self.decide(ev.name, ev.args, ev.ctx)


def _summary(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        s = v if isinstance(v, (int, float, bool)) or v is None else str(v)
        out[k] = s[:200] if isinstance(s, str) else s
    return out
