"""经验总结的附加 LLM 调用(总结 / 审核 / 合并判断)的预算与记账。rag 层不依赖 agent:模型调用经注入的 `LLMChannel`。

    channel = LLMChannel("summarizer", complete_json, meter)   # complete_json(prompt, schema, *, max_tokens, retries)
    budget = AuxBudget(AuxLimits(), run_cost=0.08, price_in=0.30, price_out=1.20)
    data = budget.call("summary", channel, prompt, schema)   # 超限时抛 SkipCall,不发请求

动态规划开发中的教训全部落在这里:
- token 上限**调用前**预估(UTF-8 字节 / 3 的保守估算):输入超出直接跳过;输出上限作为 max_tokens 传给客户端;
- 次数按**实际**请求计(router 角色账目的 calls 差值;带重试的调用按实际次数记,失败的重试也算);
  剩余次数只有 1 时不允许重试;
- 花费上限 = min(run 成本 × cost_share, cost_cap_usd);调用前按最坏情况(全部尝试 × (输入 + 输出上限))预估,
  超出就不调用;实际花费取 router 账目差值;
- 这是独立于主循环 Budget 的账目:附加调用的花费不进入主循环的预算刻度,不会触发 MAX_COST / stalled 之类的逻辑;
  它们照常记在 router 的角色账目里(同一个 router、同一个密钥与客户端),并由调用方写进同一 run 的 trace。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    return -(-len(text.encode("utf-8")) // 3)


@dataclass
class RoleMeter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0


@dataclass
class LLMChannel:
    role: str
    complete_json: Callable[..., dict[str, Any]]
    meter: Callable[[], RoleMeter]
    model: str = ""


class SkipCall(RuntimeError):
    """调用前预估超出上限:不发请求。"""


@dataclass
class AuxLimits:
    max_summaries: int = 5  # 每 run 总结的窗口数
    max_calls: dict[str, int] = field(default_factory=lambda: {"summary": 5, "critic": 5, "merge": 3})
    in_tokens: dict[str, int] = field(
        default_factory=lambda: {"summary": 8000, "critic": 4000, "merge": 2000}
    )
    out_tokens: dict[str, int] = field(default_factory=lambda: {"summary": 1500, "critic": 500, "merge": 200})
    cost_share: float = 0.15
    cost_cap_usd: float = 0.02
    # router.complete_json 的系统提示(JSON 约束 + schema)与消息包装的余量
    overhead_tokens: int = 150


@dataclass
class AuxBudget:
    limits: AuxLimits
    run_cost: float | None
    price_in: float  # USD / 百万 token
    price_out: float
    calls: dict[str, int] = field(default_factory=dict)  # 实际请求次数(含重试)
    spent: float = 0.0
    tokens: dict[str, list[int]] = field(default_factory=dict)  # kind -> [in, out]
    log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cap(self) -> float:
        if self.run_cost is None:
            return 0.0  # 不知道 run 成本就无法保证 15% 上限:不调用模型(只留规则条目)
        return min(self.run_cost * self.limits.cost_share, self.limits.cost_cap_usd)

    def remaining_calls(self, kind: str) -> int:
        return self.limits.max_calls[kind] - self.calls.get(kind, 0)

    def estimate_input(self, prompt: str, schema: dict[str, Any]) -> int:
        return estimate_tokens(prompt) + estimate_tokens(json.dumps(schema)) + self.limits.overhead_tokens

    def worst_cost(self, kind: str, in_tokens: int, attempts: int) -> float:
        out = self.limits.out_tokens[kind]
        return attempts * (in_tokens * self.price_in + out * self.price_out) / 1_000_000

    def plan_cost(self, kind: str, in_tokens: int) -> float:
        """规划用:一次调用(不含重试)的最坏花费。"""
        return self.worst_cost(kind, in_tokens, 1)

    def check(self, kind: str, prompt: str, schema: dict[str, Any]) -> tuple[int, int]:
        """返回 (预估输入 token, 允许的尝试次数);超出任何上限抛 SkipCall。"""
        left = self.remaining_calls(kind)
        if left <= 0:
            raise SkipCall(f"{kind}: call limit reached ({self.limits.max_calls[kind]})")
        n_in = self.estimate_input(prompt, schema)
        if n_in > self.limits.in_tokens[kind]:
            raise SkipCall(f"{kind}: estimated input {n_in} tokens > limit {self.limits.in_tokens[kind]}")
        attempts = 2 if left >= 2 else 1
        while attempts >= 1 and self.spent + self.worst_cost(kind, n_in, attempts) > self.cap + 1e-12:
            attempts -= 1
        if attempts == 0:
            raise SkipCall(
                f"{kind}: worst-case cost {self.worst_cost(kind, n_in, 1):.6f} would exceed the cap "
                f"{self.cap:.6f} (spent {self.spent:.6f})"
            )
        return n_in, attempts

    def call(self, kind: str, channel: LLMChannel, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        n_in, attempts = self.check(kind, prompt, schema)
        before = channel.meter()
        error = ""
        try:
            return channel.complete_json(
                prompt, schema, max_tokens=self.limits.out_tokens[kind], retries=attempts - 1
            )
        except Exception as e:  # noqa: BLE001 — 失败照样记账,由调用方决定降级
            error = f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            after = channel.meter()
            d_calls = after.calls - before.calls
            d_cost = after.cost - before.cost
            d_in, d_out = after.tokens_in - before.tokens_in, after.tokens_out - before.tokens_out
            self.calls[kind] = self.calls.get(kind, 0) + d_calls
            self.spent += d_cost
            t = self.tokens.setdefault(kind, [0, 0])
            t[0] += d_in
            t[1] += d_out
            self.log.append(
                {
                    "kind": kind,
                    "role": channel.role,
                    "estimated_in": n_in,
                    "max_attempts": attempts,
                    "calls": d_calls,
                    "tokens_in": d_in,
                    "tokens_out": d_out,
                    "cost": round(d_cost, 8),
                    **({"error": error} if error else {}),
                }
            )

    def summary(self) -> dict[str, Any]:
        return {
            "cap_usd": round(self.cap, 6),
            "run_cost_usd": None if self.run_cost is None else round(self.run_cost, 6),
            "spent_usd": round(self.spent, 6),
            "share_of_run": None if not self.run_cost else round(self.spent / self.run_cost, 4),
            "calls": dict(self.calls),
            "tokens": {k: {"in": v[0], "out": v[1]} for k, v in self.tokens.items()},
        }
