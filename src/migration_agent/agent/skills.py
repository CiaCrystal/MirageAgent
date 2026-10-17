"""从目标包加载 skill 目录，并通过 load_skill 按需读取完整迁移指导。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..tools.base import Tool, ToolContext, ToolResult

CATALOG_MAX_CHARS = 2500
_FRONT = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


@dataclass
class Skill:
    name: str
    description: str
    body: str


class SkillLibrary:
    def __init__(self, root: Path, target: Any = None):
        self.root, self.target = Path(root), target
        self.skills: dict[str, Skill] = {}
        for p in sorted(self.root.glob("*.md")):
            if p.name.upper() == "INDEX.MD":
                continue
            text = p.read_text(encoding="utf-8")
            m = _FRONT.match(text)
            meta = yaml.safe_load(m.group(1)) if m else {}
            desc = str((meta or {}).get("description") or "").strip()
            if not desc:
                raise ValueError(f"{p}: skill needs a front-matter description")
            self.skills[p.stem] = Skill(p.stem, desc, text[m.end() :].strip() if m else text.strip())
        index = self.root / "INDEX.md"
        self.header = index.read_text(encoding="utf-8").strip().split("\n\n", 1)[0] if index.exists() else ""

    def catalog(self) -> str:
        desc = self.target.describe() if self.target is not None else ""
        lines = [
            self.header
            or f"Migration skills for {desc}: short notes per topic (old API -> new API, pitfalls, minimal example).",
            "Load the body of a skill with the `load_skill` tool when a failure touches its topic:",
            *(f"- {s.name} — {s.description}" for s in self.skills.values()),
        ]
        text = "\n".join(lines)
        return text if len(text) <= CATALOG_MAX_CHARS else text[: CATALOG_MAX_CHARS - 1] + "…"

    def load(self, name: str) -> str:
        return self.skills[name].body


class LoadSkillTool(Tool):
    name = "load_skill"
    description = (
        "Load the full text of one migration skill listed in the system prompt (old API -> new API notes, pitfalls "
        "and a minimal example). Use it when a failure touches that topic; each skill only needs to be loaded once."
    )
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name from the catalog."}},
        "required": ["name"],
    }

    def __init__(self, library: SkillLibrary):
        self.library = library
        self.loaded: list[str] = []

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = str(args.get("name") or "").strip()
        if name not in self.library.skills:
            return ToolResult.error(f"unknown skill {name!r}; available: {', '.join(self.library.skills)}")
        if name in self.loaded:
            return ToolResult(
                content=f"Skill {name!r} is already loaded above in this run.",
                metadata={"skill": name, "repeat": True},
            )
        self.loaded.append(name)
        return ToolResult(
            content=ctx.truncate(self.library.load(name)), metadata={"skill": name, "repeat": False}
        )


class SkilledTarget:
    """把目标的 prompt_hints 换成 skills 目录(其余属性原样委托),交给 AgentLoop;不改 prompts / loop。"""

    def __init__(self, target: Any, library: SkillLibrary):
        self._target, self._library = target, library

    def prompt_hints(self) -> str:
        return self._library.catalog()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._target, item)
