"""模型角色路由：共用客户端，按 executor、planner、summarizer、critic 角色记录用量。"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, LLMError, LLMResponse

ROLES = ("executor", "planner", "summarizer", "critic")
_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


@dataclass
class RoleUsage:
    calls: int = 0
    tokens_in: int = 0
    cache_read: int = 0
    cache_write: int = 0
    tokens_out: int = 0
    cost: float = 0.0


@dataclass
class _Counting:
    role: str
    inner: LLMClient
    router: ModelRouter

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        resp = self.inner.complete(system, messages, tools)
        self.router._account(self.role, resp)
        return resp


@dataclass
class ModelRouter:
    roles: dict[str, LLMClient]
    pricing: Any = None
    _usage: dict[str, RoleUsage] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings, *, executor: LLMClient, pricing=None, factory=None) -> ModelRouter:
        """未单独配置的角色直接复用 executor 的客户端;单独配置了模型的角色由 `factory(model)` 构造。
        默认 factory 复用 executor 底层的 SDK 客户端(同一个密钥、同一个端点),只换模型名——所有 LLM 调用共用一个密钥。"""
        if factory is None:
            from .llm import AnthropicClient

            shared = getattr(executor, "_client", None)

            def factory(model: str) -> LLMClient:
                client = AnthropicClient(model=model, max_tokens=settings.max_output_tokens, _client=shared)
                # executor 带响应缓存时,其它角色共用同一个缓存
                return executor.rewrap(client) if hasattr(executor, "rewrap") else client

        roles: dict[str, LLMClient] = {"executor": executor}
        for role in ROLES[1:]:
            model = getattr(settings, f"model_{role}", None)
            roles[role] = factory(model) if model and model != executor.model else executor
        return cls(roles, pricing)

    def client(self, role: str) -> LLMClient:
        if role not in self.roles:
            raise KeyError(f"unknown role {role!r}; choose from {sorted(self.roles)}")
        return _Counting(role, self.roles[role], self)

    def _limited(self, role: str, max_tokens: int | None, no_thinking: bool = False) -> LLMClient:
        import dataclasses

        if role not in self.roles:
            raise KeyError(f"unknown role {role!r}; choose from {sorted(self.roles)}")
        inner = self.roles[role]
        while isinstance(inner, _Counting):
            inner = inner.inner
        if hasattr(inner, "with_limits") and getattr(inner, "max_tokens", None) is not None:
            # CachedLLM:换内层客户端的上限 / thinking,保留缓存层
            changes = {}
            if max_tokens is not None:
                changes["max_tokens"] = min(max_tokens, inner.max_tokens)
            if no_thinking and hasattr(inner, "thinking"):
                changes["thinking"] = {"type": "disabled"}
            return _Counting(role, inner.with_limits(**changes), self)
        if dataclasses.is_dataclass(inner) and hasattr(inner, "max_tokens"):
            changes: dict[str, Any] = {}
            if max_tokens is not None:
                changes["max_tokens"] = min(max_tokens, inner.max_tokens)
            if no_thinking and hasattr(inner, "thinking"):
                changes["thinking"] = {"type": "disabled"}
            # AnthropicClient:同一个 SDK 客户端(_client 原样带过去),只换输出上限 / thinking;
            # 不支持按字段重建的客户端保持原样(输出上限只由提示约束)
            with contextlib.suppress(TypeError):
                inner = dataclasses.replace(inner, **changes)
        return _Counting(role, inner, self)

    def _account(self, role: str, resp: LLMResponse) -> None:
        u = self._usage.setdefault(role, RoleUsage())
        u.calls += 1
        u.tokens_in += resp.tokens_in
        u.cache_read += resp.cache_read_tokens
        u.cache_write += resp.cache_write_tokens
        u.tokens_out += resp.tokens_out
        if self.pricing is not None:
            u.cost += self.pricing.cost(
                resp.tokens_in, resp.cache_read_tokens, resp.cache_write_tokens, resp.tokens_out
            )

    def usage(self) -> dict[str, RoleUsage]:
        return dict(self._usage)

    def complete_json(
        self,
        role: str,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
        retries: int = 1,
        no_thinking: bool = False,
    ) -> dict[str, Any]:
        """`max_tokens`:本次调用的输出上限(客户端有 max_tokens 字段时生效,共用同一个 SDK 客户端);
        `retries`:非法输出后的重试次数(经验总结的附加调用在剩余次数不足时传 0);`no_thinking`:客户端支持时关闭
        thinking,使输出上限只用于 JSON。缺省行为不变。"""
        system = (
            "Reply with a single JSON object and nothing else. It must match this JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        last = ""
        limited = max_tokens is not None or no_thinking
        client = self._limited(role, max_tokens, no_thinking) if limited else self.client(role)
        for _ in range(1 + max(0, retries)):
            resp = client.complete(system, messages, [])
            last = resp.text
            data, why = _parse(last, schema)
            if data is not None:
                return data
            _discard_cached(client)  # 未通过 schema 的回复不留在响应缓存里
            messages = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": last or "(empty)"}]},
                {"role": "user", "content": f"That was not valid: {why}. Reply with the JSON object only."},
            ]
        raise LLMError(f"{role}: output did not match schema after retry: {last[:300]}")


def _discard_cached(client: LLMClient) -> None:
    from .llm_cache import find_cache

    cache = find_cache(client)
    if cache is not None:
        cache.discard_last()


def _parse(text: str, schema: dict[str, Any]) -> tuple[dict | None, str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, "no JSON object found"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e})"
    if not isinstance(data, dict):
        return None, "not an object"
    for key in schema.get("required", []):
        if key not in data:
            return None, f"missing key {key!r}"
    for key, spec in schema.get("properties", {}).items():
        t = _JSON_TYPES.get(spec.get("type", ""))
        if key in data and t and not isinstance(data[key], t):
            return None, f"{key!r} should be {spec['type']}"
    return data, ""
