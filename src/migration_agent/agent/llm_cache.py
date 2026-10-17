"""backend = RedisBackend("redis://localhost:6379/0")   # 或 MemoryBackend()(内存后端)
    llm = CachedLLM(inner, backend, namespace="dev", ttl_seconds=86400, pricing=pricing, tracer=tracer)
    llm.complete(system, messages, tools)                # 命中:不调用 inner,token 计 0

经 `LLMClient` 协议接入(包在 AnthropicClient 外面),由 assemble 按 `settings.llm_cache_url` 选用;不改 AgentLoop、
不包装工具。所有角色(executor / planner / summarizer / critic)共用同一个缓存:ModelRouter 按角色换模型或输出上限时
经 `rewrap` / `with_limits` 保留缓存层。

规则
    - 缓存键 = sha256(全部请求参数:模型、system、messages、tools、max_tokens、thinking + namespace);任一参数变化都不命中。
    - 值 = `LLMResponse` 的 JSON(原始 token 用量,用于估算节省)。
    - 命中时返回的回复 token 全部为 0:不进 router 的按角色成本、不进 run 预算;另记 hits / misses / 节省的 token 与估算费用,
      每次调用写 trace(`llm_cache`,含 cached 标记)。事件日志由 AgentLoop 照常记录回复内容(不改循环)。
    - 失败的调用(LLMError)不写入;`ModelRouter.complete_json` 发现回复未通过 schema 时调用 `discard_last()` 删除该条。
    - 后端连不上时直接抛 `CacheUnavailable`,不静默退回无缓存(避免误以为命中而实际花钱)。"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from .hooks import Hook
from .llm import LLMClient, LLMResponse

KEY_PREFIX = "migration_agent:llm:"


class CacheUnavailable(RuntimeError):
    pass


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int | None) -> None: ...

    def delete(self, key: str) -> None: ...


@dataclass
class MemoryBackend:
    """进程内后端，不需要 Redis。TTL 只记录不过期。"""

    data: dict[str, str] = field(default_factory=dict)
    ttls: dict[str, int | None] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None) -> None:
        self.data[key] = value
        self.ttls[key] = ttl_seconds

    def delete(self, key: str) -> None:
        self.data.pop(key, None)
        self.ttls.pop(key, None)

    def keys(self, pattern_prefix: str = "") -> list[str]:
        return [k for k in self.data if k.startswith(pattern_prefix)]


class RedisBackend:
    """Redis 后端(可选依赖 `redis`,`uv sync --extra cache`)。构造时 PING,连不上直接报错。"""

    def __init__(self, url: str, *, socket_timeout: float = 5.0) -> None:
        try:
            import redis
        except ImportError as e:  # pragma: no cover - 依赖缺失
            raise CacheUnavailable(
                "llm_cache_url is set but the 'redis' package is missing: uv sync --extra cache"
            ) from e
        self.url = url
        self._redis = redis.Redis.from_url(
            url, socket_timeout=socket_timeout, socket_connect_timeout=socket_timeout
        )
        self._errors = (redis.RedisError, OSError)
        try:
            self._redis.ping()
        except self._errors as e:
            raise CacheUnavailable(f"LLM cache backend unreachable at {url}: {e}") from e

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except self._errors as e:
            raise CacheUnavailable(f"LLM cache backend error ({self.url}): {e}") from e

    def get(self, key: str) -> str | None:
        v = self._call(self._redis.get, key)
        return v.decode("utf-8") if isinstance(v, bytes) else v

    def set(self, key: str, value: str, ttl_seconds: int | None) -> None:
        self._call(self._redis.set, key, value, ex=ttl_seconds or None)

    def delete(self, key: str) -> None:
        self._call(self._redis.delete, key)

    def clear_namespace(self, namespace: str) -> int:
        """删除一个命名空间下的全部条目(scripts/llm_cache_clear.py 用)。"""
        n = 0
        for k in self._call(self._redis.scan_iter, match=f"{KEY_PREFIX}{namespace}:*"):
            n += int(self._call(self._redis.delete, k))
        return n


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    discarded: int = 0
    saved_tokens_in: int = 0
    saved_cache_read: int = 0
    saved_cache_write: int = 0
    saved_tokens_out: int = 0
    saved_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        d["saved_usd"] = round(self.saved_usd, 6)
        return d


def request_key(namespace: str, params: dict[str, Any]) -> str:
    """键覆盖全部请求参数;JSON 规范化(sort_keys)后哈希,命名空间写进键前缀与哈希输入两处。"""
    body = json.dumps({"namespace": namespace, **params}, sort_keys=True, ensure_ascii=False, default=str)
    return f"{KEY_PREFIX}{namespace}:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"


@dataclass
class CachedLLM:
    inner: LLMClient
    backend: CacheBackend
    namespace: str = "default"
    ttl_seconds: int | None = None
    pricing: Any = None
    tracer: Any = None
    stats: CacheStats = field(default_factory=CacheStats)
    _last_key: str | None = field(default=None, repr=False)

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def _client(self) -> Any:  # ModelRouter.from_settings 复用底层 SDK 客户端
        return getattr(self.inner, "_client", None)

    @property
    def max_tokens(self) -> Any:
        return getattr(self.inner, "max_tokens", None)

    @property
    def thinking(self) -> Any:
        return getattr(self.inner, "thinking", None)

    def rewrap(self, inner: LLMClient) -> CachedLLM:
        """同一个缓存(后端、命名空间、统计)包住另一个客户端(router 为其它角色换模型时用)。"""
        return dataclasses.replace(self, inner=inner, _last_key=None)

    def with_limits(self, **changes: Any) -> CachedLLM:
        """router._limited:换输出上限 / thinking,缓存层保留(键随之变化)。"""
        inner = self.inner
        if dataclasses.is_dataclass(inner):
            inner = dataclasses.replace(inner, **{k: v for k, v in changes.items() if hasattr(inner, k)})
        return self.rewrap(inner)

    def params(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "client": type(self.inner).__name__,
            "model": self.inner.model,
            "max_tokens": getattr(self.inner, "max_tokens", None),
            "thinking": getattr(self.inner, "thinking", None),
            "system": system,
            "messages": messages,
            "tools": tools,
        }

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        key = request_key(self.namespace, self.params(system, messages, tools))
        raw = self.backend.get(key)
        if raw is not None:
            stored = LLMResponse.model_validate_json(raw)
            self._last_key = key
            self._saved(stored)
            self._log(key, True, stored)
            return stored.model_copy(
                update={"tokens_in": 0, "tokens_out": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
            )
        resp = self.inner.complete(system, messages, tools)  # LLMError 原样抛出,不写入
        self.backend.set(key, resp.model_dump_json(), self.ttl_seconds)
        self._last_key = key
        with self.stats._lock:
            self.stats.misses += 1
        self._log(key, False, resp)
        return resp

    def discard_last(self) -> None:
        """上一条回复不可用(未通过 schema):从缓存删除,下次重新请求。"""
        if self._last_key is not None:
            self.backend.delete(self._last_key)
            self._last_key = None
            with self.stats._lock:
                self.stats.discarded += 1

    def _saved(self, r: LLMResponse) -> None:
        with self.stats._lock:
            s = self.stats
            s.hits += 1
            s.saved_tokens_in += r.tokens_in
            s.saved_cache_read += r.cache_read_tokens
            s.saved_cache_write += r.cache_write_tokens
            s.saved_tokens_out += r.tokens_out
            if self.pricing is not None:
                s.saved_usd += self.pricing.cost(
                    r.tokens_in, r.cache_read_tokens, r.cache_write_tokens, r.tokens_out
                )

    def _log(self, key: str, cached: bool, r: LLMResponse) -> None:
        if self.tracer is None:
            return
        self.tracer.log(
            "llm_cache",
            {
                "cached": cached,
                "key": key.rsplit(":", 1)[-1][:16],
                "namespace": self.namespace,
                "model": self.inner.model,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
            },
        )


def find_cache(client: Any) -> CachedLLM | None:
    """沿 `.inner` 链找缓存层(router 的计数包装、包装器写法的 SupervisedLLM 都把真实客户端放在 inner)。"""
    seen = 0
    while client is not None and seen < 8:
        if isinstance(client, CachedLLM):
            return client
        client, seen = getattr(client, "inner", None), seen + 1
    return None


def build_backend(url: str) -> CacheBackend:
    if url.startswith("memory://"):
        return MemoryBackend()
    if url.startswith(("redis://", "rediss://", "unix://")):
        return RedisBackend(url)
    raise ValueError(f"unsupported llm_cache_url {url!r} (use redis://host:port/db)")


def wrap_from_settings(
    llm: LLMClient, settings: Any, *, pricing: Any = None, tracer: Any = None
) -> LLMClient:
    """`settings.llm_cache_url` 为空 → 原样返回(默认关闭,行为不变)。"""
    url = getattr(settings, "llm_cache_url", None)
    if not url:
        return llm
    return CachedLLM(
        llm,
        build_backend(url),
        namespace=settings.llm_cache_namespace,
        ttl_seconds=settings.llm_cache_ttl_seconds,
        pricing=pricing,
        tracer=tracer,
    )


class LlmCacheSummaryHook(Hook):
    """run 结束时把缓存汇总写进 trace(`llm_cache`,summary=true);只在缓存开启时挂上。"""

    name = "llm_cache"

    def __init__(self, cache: CachedLLM, tracer: Any) -> None:
        self.cache, self.tracer = cache, tracer

    def on_run_end(self, ev: Any) -> None:
        self.tracer.log(
            "llm_cache", {"summary": True, "namespace": self.cache.namespace, **self.cache.stats.as_dict()}
        )
