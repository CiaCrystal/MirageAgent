"""文档切块:Markdown 按标题层级切分,保留标题路径作为上下文前缀;issue 以「标题 + 正文 + 被采纳的回答」为单元。

    chunk_markdown(src) -> list[Document]     guide / docs / errors
    chunk_issue(src) -> list[Document]        一条 issue 一块(正文截断)

每块文本以 `<文档标题> > <h2> > <h3>` 开头(称 heading path),检索时标题词参与 BM25 与向量,
这对「`int_parsing`」这类正文很短、信息集中在标题里的小节尤其重要。
标题里的 `{#anchor}` 去掉后写进元数据(并拼成带锚点的 URL),报错链接里的 slug 也能命中它。
代码围栏内以 `#` 开头的行(注释、`#> 输出`)不当作标题。"""

from __future__ import annotations

import re

from ..common.schema import Document, stable_id
from .sources import RawSource

MAX_CHARS = 1800
ISSUE_BODY_CHARS = 1500
ISSUE_ANSWER_CHARS = 900

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_ANCHOR = re.compile(r"\s*\{#([\w-]+)\}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def slugify(heading: str) -> str:
    """mkdocs(Python-Markdown toc)默认 slug:小写、去标点、空白转 '-'。"""
    s = re.sub(r"[^\w\s-]", "", heading.replace("`", "")).strip().lower()
    return re.sub(r"[\s]+", "-", s)


def _sections(text: str) -> list[tuple[list[tuple[int, str, str]], str]]:
    """把 Markdown 切成 [(标题栈[(level, title, anchor)], 正文)]。"""
    text = _HTML_COMMENT.sub("", text)
    out: list[tuple[list[tuple[int, str, str]], str]] = []
    stack: list[tuple[int, str, str]] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            out.append((list(stack), "\n".join(buf).strip()))
            buf = []
            level, title = len(m.group(1)), m.group(2)
            am = _ANCHOR.search(title)
            anchor = am.group(1) if am else ""
            title = re.sub(r"<[^>]+>", "", _ANCHOR.sub("", title)).strip()
            stack = [h for h in stack if h[0] < level] + [(level, title, anchor or slugify(title))]
            continue
        buf.append(line)
    out.append((list(stack), "\n".join(buf).strip()))
    return [(s, body) for s, body in out if body or s]


def _split(body: str, limit: int) -> list[str]:
    """按段落(不在代码围栏中间断开)装箱到 limit 字符;单段超长时硬切。"""
    if len(body) <= limit:
        return [body]
    paras: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        cur.append(line)
        if not line.strip() and not in_fence:
            paras.append("\n".join(cur).strip())
            cur = []
    if cur:
        paras.append("\n".join(cur).strip())
    parts: list[str] = []
    acc = ""
    for p in filter(None, paras):
        while len(p) > limit:
            if acc:
                parts.append(acc)
                acc = ""
            parts.append(p[:limit])
            p = p[limit:]
        if acc and len(acc) + len(p) + 2 > limit:
            parts.append(acc)
            acc = p
        else:
            acc = f"{acc}\n\n{p}" if acc else p
    if acc:
        parts.append(acc)
    return parts


def chunk_markdown(src: RawSource, max_chars: int = MAX_CHARS, *, min_level: int = 1) -> list[Document]:
    docs: list[Document] = []
    for stack, body in _sections(src.text):
        heads = [h for h in stack if h[0] >= min_level]
        # 文档 H1 与 src.name 相同,避免重复
        titles = [t for _, t, _ in heads if t != src.name]
        path = " > ".join([src.name, *titles])
        anchor = heads[-1][2] if heads else ""
        if not body.strip():
            continue
        pieces = _split(body, max_chars)
        for i, piece in enumerate(pieces):
            header = (
                path
                + (f" (#{anchor})" if anchor else "")
                + (f" [part {i + 1}/{len(pieces)}]" if len(pieces) > 1 else "")
            )
            docs.append(
                Document(
                    id=stable_id(src.kind, src.url, path, str(i)),
                    text=f"{header}\n\n{piece}",
                    metadata={
                        "source_kind": src.kind,
                        "title": src.name,
                        "heading": path,
                        "anchor": anchor,
                        "url": src.url + (f"#{anchor}" if anchor else ""),
                        "part": i,
                        "ref": str(src.meta.get("ref", "")),
                    },
                )
            )
    return docs


def chunk_release_notes(src: RawSource, max_chars: int = MAX_CHARS) -> list[Document]:
    "以 `## vX.Y.Z (date)` 为一节;节内的 ### / #### 不再切出独立标题,只按长度切。"
    docs: list[Document] = []
    custom = src.meta.get("release_heading")
    if custom:
        starts = [m.start() for m in re.finditer(rf"(?m)^{custom}", src.text)]
        releases = [src.text[a:b] for a, b in zip(starts, [*starts[1:], len(src.text)], strict=True)]
    else:
        releases = re.split(r"(?m)^(?=## v\d)", src.text)
    heading = custom or r"## (v[\w.]+)\s*(?:\(([\d-]+)\))?"
    for rel in releases:
        m = re.match(heading, rel)
        if not m:
            continue
        version, date = m.group(1), m.group(2) or ""
        body = rel.split("\n", 1)[1].strip() if "\n" in rel else ""
        body = re.sub(r"\[GitHub release\]\([^)]*\)\s*", "", body)
        pieces = _split(body, max_chars)
        for i, piece in enumerate(pieces):
            path = f"Release notes > {version}" + (f" ({date})" if date else "")
            header = path + (f" [part {i + 1}/{len(pieces)}]" if len(pieces) > 1 else "")
            docs.append(
                Document(
                    id=stable_id("release_notes", version, str(i)),
                    text=f"{header}\n\n{piece}",
                    metadata={
                        "source_kind": "release_notes",
                        "title": "Release notes",
                        "heading": path,
                        "anchor": version,
                        "url": str(src.meta.get("release_url") or src.url).format(
                            version=version, version_number=version.lstrip("v")
                        ),
                        "part": i,
                        "version": version,
                        "date": date,
                        "ref": str(src.meta.get("ref", "")),
                    },
                )
            )
    return docs


_ISSUE_HEAD = [re.compile(r"### Initial Checks.*?(?=###|\Z)", re.S)]
_ISSUE_TAIL = [re.compile(r"- \[[ xX]\] .*")]


def clean_issue_body(body: str, noise: list[str] | tuple[str, ...] = ()) -> str:
    """issue 模板噪声:通用段 + 目标 knowledge.yaml 的 issue_noise_patterns(经 RawSource.meta 传入)。"""
    body = _HTML_COMMENT.sub("", body.replace("\r\n", "\n"))
    for pat in [*_ISSUE_HEAD, *(re.compile(p, re.S) for p in noise), *_ISSUE_TAIL]:
        body = pat.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def chunk_issue(src: RawSource) -> list[Document]:
    meta = src.meta
    noise = meta.get("noise_patterns", ())
    body = clean_issue_body(src.text, noise)
    if len(body) > ISSUE_BODY_CHARS:
        body = body[:ISSUE_BODY_CHARS] + " …"
    answers = [clean_issue_body(a, noise) for a in meta.get("comments", [])]
    answer_text = "\n\n".join(
        "Answer: " + (a[:ISSUE_ANSWER_CHARS] + " …" if len(a) > ISSUE_ANSWER_CHARS else a)
        for a in answers
        if a
    )
    header = f"{src.name}\nLabels: {meta.get('labels') or '-'} | state: {meta.get('state')}"
    text = f"{header}\n\n{body}" + (f"\n\n{answer_text}" if answer_text else "")
    return [
        Document(
            id=stable_id("issue", str(meta["number"])),
            text=text,
            metadata={
                "source_kind": "issue",
                "title": meta["title"],
                "heading": src.name,
                "anchor": str(meta["number"]),
                "url": src.url,
                "part": 0,
                "labels": meta.get("labels", ""),
                "state": meta.get("state", ""),
                "created_at": meta.get("created_at", ""),
                "has_answer": bool(answer_text),
            },
        )
    ]


def chunk_source(src: RawSource) -> list[Document]:
    if src.kind == "release_notes":
        return chunk_release_notes(src)
    if src.kind == "issue":
        return chunk_issue(src)
    return chunk_markdown(src)
