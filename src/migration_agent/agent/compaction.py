"""Compact old observations in place while preserving tool call/result pairs.

Inspired by pi-mono's recent-turn retention, with the deterministic placeholder
baseline compaction policy. No harness types or extra LLM calls are needed.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Conservative byte heuristic; trace labels this as an estimate, not usage."""
    return math.ceil(len(json.dumps(value, ensure_ascii=False).encode("utf-8")) / 3)


def file_key(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/"))


@dataclass
class ContextCompactor:
    enabled: bool = True
    context_window: int = 128_000
    threshold: float = 0.7
    keep_recent_turns: int = 6
    compacted_ids: set[str] = field(default_factory=set, init=False)
    pending_reads: set[str] = field(default_factory=set, init=False)
    re_read_after_compaction: int = field(default=0, init=False)
    events: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.context_window <= 0 or not 0 < self.threshold < 1 or self.keep_recent_turns < 1:
            raise ValueError("invalid compaction window, threshold or recent-turn count")

    def reset(self) -> None:
        self.compacted_ids.clear()
        self.pending_reads.clear()
        self.re_read_after_compaction = self.events = 0

    def compact(self, messages: list[dict], previous_input_tokens: int) -> dict | None:
        if not self.enabled or previous_input_tokens <= self.context_window * self.threshold:
            return None
        turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(turns) <= self.keep_recent_turns:
            return None
        boundary = turns[-self.keep_recent_turns]
        calls = {
            b["id"]: b
            for m in messages
            if m["role"] == "assistant" and isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_use"
        }
        before = estimate_tokens(messages)
        count = 0
        paths = []
        for m in messages[:boundary]:
            if not isinstance(m["content"], list):
                continue
            for block in m["content"]:
                if block.get("type") != "tool_result":
                    continue
                uid = block["tool_use_id"]
                call = calls.get(uid)
                if not call or uid in self.compacted_ids:
                    continue
                name, args = call["name"], call.get("input", {})
                # Keep complete edit history, including errors and affected-line excerpts.
                if name in {"edit_file", "write_file", "insert_line", "apply_patch"}:
                    continue
                content = block.get("content", "")
                path = args.get("path", "")
                detail = f"{name} {path}".strip()
                if name == "read_file" and isinstance(content, str):
                    match = re.search(r"of (\d+)\)", content.split("\n", 1)[0])
                    detail += f", {match[1]} lines" if match else ""
                placeholder = f"[compacted: {detail}; call {uid}; re-run the tool if needed]"
                if len(str(content)) <= len(placeholder):
                    continue
                block["content"] = placeholder
                self.compacted_ids.add(uid)
                count += 1
                if name == "read_file" and path and not block.get("is_error"):
                    key = file_key(path)
                    self.pending_reads.add(key)
                    paths.append(key)
        if not count:
            return None
        self.events += 1
        after = estimate_tokens(messages)
        return {
            "previous_input_tokens": previous_input_tokens,
            "tokens_before": before,
            "tokens_after": after,
            "token_count_method": "estimated_utf8_bytes_div_3",
            "blocks_compacted": count,
            "read_paths": paths,
            "keep_recent_turns": self.keep_recent_turns,
        }

    def record_read(self, name: str, args: dict, is_error: bool) -> str | None:
        if name != "read_file" or is_error:
            return None
        path = file_key(args.get("path", ""))
        if path not in self.pending_reads:
            return None
        self.pending_reads.remove(path)
        self.re_read_after_compaction += 1
        return path
