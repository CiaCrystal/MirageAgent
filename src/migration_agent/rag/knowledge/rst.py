"""reStructuredText → Markdown(只做切块需要的那部分)。knowledge.yaml 写 `raw_format: rst` 的文档仓库用。

    rst_to_markdown(text) -> str

- 标题:文字行 + 等长(或更长)的下划线(可有同字符上划线);层级按字符首次出现的顺序 → `#` … `####`;
  标题前一行的 `.. _label:` 写成 `{#label}` 锚点(与 Markdown 切块器的锚点约定一致)。
- 代码:段尾 `::` 后的缩进块、`.. sourcecode::` / `.. code-block::` 指令 → ``` 围栏(围栏内的 `#` 行不会被当成标题)。
- 角色 :meth:`_engine.Engine.execute` → `Engine.execute`(去掉以 `_` 开头的模块前缀与 `~`);`text <target>`_ 链接 → text。
- 其它指令(seealso / note / versionadded …)保留缩进内容,丢掉指令行与字段行;`..` 注释行丢弃。"""

from __future__ import annotations

import re

_UNDER = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_LABEL = re.compile(r"^\.\.\s+_([\w.-]+):\s*$")
_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")
_FIELD = re.compile(r"^\s*:[\w-]+:")
_ROLE = re.compile(r":[\w:]+:`([^`]+)`")
_LINK = re.compile(r"`([^`<]+?)\s*<[^>]+>`_+")
_REF = re.compile(r"`([^`]+)`_+")
_CODE_DIRECTIVES = {"sourcecode", "code-block", "code", "doctest"}


def _role(m: re.Match) -> str:
    inner = m.group(1).strip()
    if "<" in inner and inner.endswith(">"):
        inner = inner.split("<", 1)[0].strip() or inner
    inner = inner.lstrip("~!")
    parts = inner.split(".")
    while len(parts) > 1 and parts[0].startswith("_"):
        parts = parts[1:]
    return "`" + ".".join(parts) + "`"


def _inline(line: str) -> str:
    line = _ROLE.sub(_role, line)
    line = _LINK.sub(lambda m: m.group(1).strip(), line)
    line = _REF.sub(lambda m: m.group(1), line)
    return line.replace("``", "`")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def rst_to_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    out: list[str] = []
    levels: list[str] = []
    label = ""
    i, n = 0, len(lines)

    def code_block(start: int, base: int) -> int:
        """从 start 起收集缩进 > base 的块(空行可穿插),输出围栏;返回块后的下一行号。"""
        j, body = start, []
        while j < n and (not lines[j].strip() or _indent(lines[j]) > base):
            body.append(lines[j])
            j += 1
        while body and not body[-1].strip():
            body.pop()
        content = [b for b in body if b.strip()]
        cut = min((_indent(b) for b in content), default=0)
        if content:
            out.extend(["```", *[b[cut:] for b in body], "```", ""])
        return j

    while i < n:
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1] if i + 1 < n else ""
        # 上划线 + 标题 + 下划线
        if _UNDER.match(line) and i + 2 < n and _UNDER.match(lines[i + 2]) and lines[i + 1].strip():
            ch = "^" + line.strip()[0]
            title = lines[i + 1].strip()
            i += 3
            if ch not in levels:
                levels.append(ch)
            out.append(
                "#" * min(levels.index(ch) + 1, 4)
                + " "
                + _inline(title)
                + (f" {{#{label}}}" if label else "")
            )
            out.append("")
            label = ""
            continue
        # 标题 + 下划线
        if stripped and not line.startswith(" ") and _UNDER.match(nxt) and len(nxt.strip()) >= len(stripped):
            ch = nxt.strip()[0]
            if ch not in levels:
                levels.append(ch)
            out.append(
                "#" * min(levels.index(ch) + 1, 4)
                + " "
                + _inline(stripped)
                + (f" {{#{label}}}" if label else "")
            )
            out.append("")
            label = ""
            i += 2
            continue
        m = _LABEL.match(line)
        if m:
            label = m.group(1)
            i += 1
            continue
        d = _DIRECTIVE.match(line)
        if d:
            base, name, arg = len(d.group(1)), d.group(2), d.group(3).strip()
            i += 1
            if name == "changelog":
                version = released = ""
                while i < n and (_FIELD.match(lines[i]) or not lines[i].strip()):
                    f = lines[i].strip()
                    if f.startswith(":version:"):
                        version = f.split(":", 2)[2].strip()
                    elif f.startswith(":released:"):
                        released = f.split(":", 2)[2].strip()
                    i += 1
                if version:
                    out.extend([f"## v{version}" + (f" ({released})" if released else ""), ""])
                continue
            while i < n and _FIELD.match(lines[i]):  # 指令字段行
                i += 1
            if name in _CODE_DIRECTIVES:
                i = code_block(i, base)
                continue
            if name in (
                "change",
                "seealso",
                "note",
                "tip",
                "warning",
                "versionadded",
                "versionchanged",
                "deprecated",
            ):
                if arg and name not in ("change",):
                    out.append(_inline(arg))
                continue  # 内容按普通缩进文本继续处理
            # 其它指令(toctree、autoclass、image …):整块丢弃
            while i < n and (not lines[i].strip() or _indent(lines[i]) > base):
                i += 1
            continue
        if stripped.startswith(".."):  # 注释
            i += 1
            while i < n and lines[i].strip() and _indent(lines[i]) > _indent(line):
                i += 1
            continue
        if stripped.endswith("::") and not stripped.startswith(".."):
            text_part = stripped[:-2].rstrip()
            if text_part:
                out.append(_inline(text_part + (":" if not text_part.endswith(":") else "")))
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            i = code_block(i, _indent(line))
            continue
        out.append(_inline(stripped) if stripped else "")
        i += 1
    md = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
