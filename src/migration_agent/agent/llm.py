"""Anthropic SDK 薄封装。

    class LLMClient(Protocol):
        def complete(system: str, messages: list[dict], tools: list[dict]) -> LLMResponse

LLMResponse 把 SDK 的 content blocks 转成纯 dict(可直接 append 回 messages),并抽出
tool_uses / text / stop_reason / token 用量。系统提示加 `cache_control` 以启用 prompt caching
(系统提示 + 工具定义在整个 run 中不变,命中率高)。重试交给 SDK 的 max_retries。

客户端实现遵循同一协议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


class LLMResponse(BaseModel):
    content: list[dict[str, Any]]  # 原样可回填的 assistant content blocks
    stop_reason: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[ToolUse]:
        return [
            ToolUse(id=b["id"], name=b["name"], input=b.get("input") or {})
            for b in self.content
            if b.get("type") == "tool_use"
        ]


class LLMClient(Protocol):
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


@dataclass
class AnthropicClient:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192
    max_retries: int = 4
    timeout: float = 300.0
    # 附加调用(总结 / 审核)可关闭 thinking,使输出上限只约束 JSON 本身(DeepSeek 兼容端点接受,已验证);
    # None = 不传该参数(executor 行为不变)
    thinking: dict[str, Any] | None = None
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=self.max_retries, timeout=self.timeout)

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e

        content = [_block_to_dict(b) for b in resp.content]
        u = resp.usage
        return LLMResponse(
            content=content,
            stop_reason=resp.stop_reason,
            tokens_in=u.input_tokens,
            tokens_out=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            model=resp.model,
        )


def _block_to_dict(block: Any) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # 其它 block(thinking 等)按 SDK 的 model_dump 回填
    return block.model_dump(exclude_none=True)
