"""清空 LLM 响应缓存的一个命名空间(只删 `migration_agent:llm:<namespace>:*`):

uv run python scripts/llm_cache_clear.py --url redis://localhost:6379/0 --namespace dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from migration_agent.agent.llm_cache import RedisBackend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="redis://localhost:6379/0")
    ap.add_argument("--namespace", required=True)
    a = ap.parse_args()
    print(f"deleted {RedisBackend(a.url).clear_namespace(a.namespace)} key(s) in namespace {a.namespace!r}")


if __name__ == "__main__":
    main()
