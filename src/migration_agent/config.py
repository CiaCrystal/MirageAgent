"""全局配置:模型名、预算上限(步数/token/时间)、路径、沙箱镜像名。

接口:`Settings`(pydantic-settings,环境变量前缀 `MIGRATION_AGENT_`),`load_settings() -> Settings`。

`.env` 处理:`load_dotenv()` 在导入时把 `.env` 加载进 `os.environ`(不覆盖已有变量)。
这样 `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` 也能被 Anthropic SDK 读到——
pydantic-settings 自己只读带前缀的变量,不会导出到环境。

兼容 Anthropic 协议的第三方端点(如 DeepSeek)只需设 `ANTHROPIC_BASE_URL` 与对应模型名,见 README。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MIGRATION_AGENT_", extra="ignore")

    model: str = "claude-sonnet-4-6"
    max_output_tokens: int = 8192

    max_steps: int = 40
    max_total_tokens: int = 2_000_000
    max_wall_seconds: int = 1800

    compaction_enabled: bool = True
    context_window_tokens: int = 128_000
    compaction_threshold: float = 0.7
    compaction_keep_recent_turns: int = 6

    tool_output_max_chars: int = 12_000
    command_timeout_seconds: int = 300
    pytest_timeout_seconds: int = 600

    data_dir: Path = Path("./data")
    traces_dir: Path = Path("./traces")

    knowledge_index_dir: Path | None = None
    embedding_model: str = "st:BAAI/bge-small-en-v1.5"
    knowledge_top_k: int = 5

    knowledge_preretrieve: bool = True

    codebase_tools: bool = True
    planner_briefing: bool = True
    codebase_embedding_model: str = "st:BAAI/bge-small-en-v1.5"

    strategy_controller: bool = True
    repeat_error_threshold: int = 3
    budget_notices: bool = True
    exploration_nudge_steps: int = 8
    command_policy: bool = True

    memory_enabled: bool = False
    memory_dir: Path | None = None
    memory_require_snapshot: bool = True
    memory_preretrieve: bool = True
    memory_top_k: int = 3
    memory_initial_max_tokens: int = 1500
    memory_tool_max_tokens: int = 1500
    memory_min_bm25: float = 2.0

    memory_extract_candidates: bool = False
    memory_candidate_dir: Path | None = None

    memory_llm_summary: bool = True
    memory_online: bool = True
    memory_critic_thresholds: dict[str, int] | None = None

    memory_retrieval: str = "hybrid"
    memory_embedding_model: str = "st:BAAI/bge-small-en-v1.5"
    memory_requery_new_signatures: bool = True
    memory_feedback: bool = True

    extension_mode: str = "hooks"
    aci_tools: bool = True
    context_strategy: str = "summary"
    stop_gate: bool = True
    stop_gate_max_returns: int = 2

    budget_token_mode: str = "priced"
    max_cost_usd: float | None = 0.5

    model_planner: str | None = None
    model_summarizer: str | None = None
    model_critic: str | None = None
    llm_error_retries: int = 0

    skills_external: bool = False

    llm_cache_url: str | None = None
    llm_cache_namespace: str = "dev"
    llm_cache_ttl_seconds: int | None = 7 * 24 * 3600

    dynamic_plan: bool = True
    plan_runtime_usages: bool = True
    replan_max_calls: int = 3
    classify_llm_max_calls: int = 1
    diagnosis_subagent: bool = True
    diagnosis_max_steps: int = 8
    diagnosis_max_tokens: int = 60_000
    orchestrator_attempts: int = 2
    first_attempt_budget: float = 0.7

    replan_trigger_quota: dict[str, int] | None = None

    give_up_check: bool = False
    give_up_check_max_returns: int = 1

    protect_collected_tests: bool = False

    sandbox_image: str = "migration-agent-sandbox:latest"
    sandbox_backend: str = "docker"
    allow_compatibility_shim: bool = False

    def resolve_memory_dir(self, target) -> Path | None:
        """Use an explicit memory path or the current target's bundled memory, if present."""
        if self.memory_dir is not None:
            return self.memory_dir
        directory = target.root / "memory"
        return directory if directory.is_dir() else None

    @property
    def knowledge_dir(self) -> Path:
        """按当前目标分目录(data/knowledge/<target>/index);knowledge.yaml 的 index_subdir 可指定旧路径。"""
        if self.knowledge_index_dir:
            return self.knowledge_index_dir
        from .migration.targets import active

        target = active()
        sub = target.knowledge_spec.index_subdir if target.knowledge_spec is not None else ""
        return (
            self.data_dir / "knowledge" / sub if sub else self.data_dir / "knowledge" / target.name / "index"
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
