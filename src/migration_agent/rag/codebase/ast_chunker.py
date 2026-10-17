"""ASTChunker:用 tree-sitter-python 解析文件,按函数/类/模块级语句切块。

    chunks = chunk_file("pkg/models.py", source)        -> list[CodeChunk]

切块规则:
    function / method  完整定义(含装饰器);超过 `max_lines` 时按行窗口拆成多块,后续块重复签名行作上下文
    class              类头:装饰器、`class` 行、docstring、类体中除方法体以外的语句(字段、嵌套 Config 等),
                       方法只保留签名行;方法另各自成块并记录 parent
    module             连续的模块级非定义语句(import、常量、`if __name__` 等)按 `max_lines` 分窗

每块文本以 `# <file> :: <qualname> (<kind>)` 开头,方法块再附父类签名行,便于 BM25 命中文件名/类名。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from ..common.schema import Document, stable_id

_LOCAL = threading.local()
_TOKEN = re.compile(r"\w+|[^\w\s]")


def parse_python(source: str | bytes):
    """解析为 tree-sitter Tree;Parser 非线程安全,按线程缓存。"""
    parser = getattr(_LOCAL, "parser", None)
    if parser is None:
        import tree_sitter
        import tree_sitter_python

        parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
        _LOCAL.parser = parser
    data = source.encode("utf-8") if isinstance(source, str) else source
    return parser.parse(data)


def node_text(node) -> str:
    return node.text.decode("utf-8", errors="replace")


@dataclass
class CodeChunk:
    file: str
    qualname: str
    kind: str  # module | class | function | method | window
    start_line: int
    end_line: int
    text: str
    def_line: int = 0
    signature: str = ""
    docstring: str = ""
    parent: str = ""
    decorators: list[str] = field(default_factory=list)
    part: int = 0

    @property
    def id(self) -> str:
        return stable_id(self.file, self.qualname, self.kind, str(self.part), str(self.start_line))

    def to_document(self) -> Document:
        return Document(
            id=self.id,
            text=self.text,
            metadata={
                "file": self.file,
                "qualname": self.qualname,
                "kind": self.kind,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "def_line": self.def_line,
                "signature": self.signature,
                "parent": self.parent,
            },
        )


def _docstring(block) -> str:
    if block is None:
        return ""
    for child in block.named_children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement" and child.named_children:
            first = child.named_children[0]
            if first.type == "string":
                text = node_text(first)
                return text.strip("rRbBuUfF").strip("\"'").strip()
        return ""
    return ""


def _signature(defn, lines: list[str]) -> str:
    """`def`/`class` 起到冒号的源码(可能跨行),压成一行。"""
    body = defn.child_by_field_name("body")
    start = defn.start_point.row
    end = body.start_point.row if body is not None else defn.end_point.row
    if body is not None and body.start_point.row == start:
        text = lines[start][: body.start_point.column]
    else:
        text = "\n".join(lines[start : max(end, start + 1)])
    return re.sub(r"\s+", " ", text).strip().rstrip(":").strip()


class _Chunker:
    def __init__(self, file: str, source: str, max_lines: int):
        self.file, self.max_lines = file, max_lines
        self.lines = source.splitlines()
        self.tree = parse_python(source)
        self.out: list[CodeChunk] = []

    def header(self, qualname: str, kind: str, extra: str = "") -> str:
        head = f"# {self.file} :: {qualname} ({kind})"
        return head + ("\n" + extra if extra else "")

    def span_text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])

    def run(self) -> list[CodeChunk]:
        loose: list[tuple[int, int]] = []
        for node in self.tree.root_node.named_children:
            defn, decorators = _unwrap(node)
            if defn is not None and defn.type in ("function_definition", "class_definition"):
                self.definition(node, defn, decorators, prefix="", parent_sig="")
            else:
                loose.append((node.start_point.row + 1, node.end_point.row + 1))
        self.module_windows(loose)
        return self.out

    def module_windows(self, spans: list[tuple[int, int]]) -> None:
        # 合并相邻的模块级语句(中间只隔空行/注释)后按 max_lines 分窗
        merged: list[list[int]] = []
        for start, end in spans:
            if merged and all(not self.lines[i - 1].strip() for i in range(merged[-1][1] + 1, start)):
                merged[-1][1] = end
            elif merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        part = 0
        for start, end in merged:
            for s in range(start, end + 1, self.max_lines):
                e = min(end, s + self.max_lines - 1)
                body = self.span_text(s, e)
                if not body.strip():
                    continue
                self.out.append(
                    CodeChunk(
                        self.file,
                        "<module>",
                        "module",
                        s,
                        e,
                        self.header("<module>", "module") + "\n" + body,
                        part=part,
                    )
                )
                part += 1

    def definition(self, outer, defn, decorators, prefix: str, parent_sig: str) -> None:
        name_node = defn.child_by_field_name("name")
        name = node_text(name_node) if name_node is not None else "?"
        qualname = f"{prefix}.{name}" if prefix else name
        start, end = outer.start_point.row + 1, outer.end_point.row + 1
        def_line = defn.start_point.row + 1
        sig = _signature(defn, self.lines)
        body = defn.child_by_field_name("body")
        doc = _docstring(body)
        decos = [node_text(d).strip() for d in decorators]
        if defn.type == "class_definition":
            self.class_chunk(outer, defn, body, qualname, start, end, def_line, sig, doc, decos, prefix)
            return
        kind = "method" if prefix and parent_sig else "function"
        extra = parent_sig if kind == "method" else ""
        if end - start + 1 <= self.max_lines:
            text = self.header(qualname, kind, extra) + "\n" + self.span_text(start, end)
            self.out.append(
                CodeChunk(
                    self.file, qualname, kind, start, end, text, def_line, sig, doc[:500], prefix, decos
                )
            )
            return
        for part, s in enumerate(range(start, end + 1, self.max_lines)):
            e = min(end, s + self.max_lines - 1)
            context = extra if part == 0 else "\n".join(x for x in (extra, sig + ":  # (continued)") if x)
            text = self.header(qualname, kind, context) + "\n" + self.span_text(s, e)
            self.out.append(
                CodeChunk(
                    self.file,
                    qualname,
                    kind,
                    s,
                    e,
                    text,
                    def_line if part == 0 else 0,
                    sig,
                    doc[:500] if part == 0 else "",
                    prefix,
                    decos,
                    part=part,
                )
            )

    def class_chunk(self, outer, defn, body, qualname, start, end, def_line, sig, doc, decos, prefix):
        shown: list[str] = []
        cursor = start
        children = body.named_children if body is not None else []
        nested = []
        for child in children:
            cdefn, cdecos = _unwrap(child)
            if cdefn is not None and cdefn.type == "function_definition":
                c_start = child.start_point.row + 1
                shown.append(self.span_text(cursor, c_start - 1))
                for d in cdecos:
                    shown.append(self.lines[d.start_point.row])
                indent = re.match(r"\s*", self.lines[cdefn.start_point.row]).group(0)
                shown.append(indent + _signature(cdefn, self.lines) + ": ...")
                cursor = child.end_point.row + 2
                nested.append((child, cdefn, cdecos))
            elif cdefn is not None and cdefn.type == "class_definition":
                # 嵌套类(如 Config)保留在类头里;它的方法也单独成块
                nested.append((child, cdefn, cdecos))
        shown.append(self.span_text(cursor, end))
        text = "\n".join(s for s in shown if s.strip())
        lines = text.splitlines()
        if len(lines) > self.max_lines * 2:
            text = "\n".join(lines[: self.max_lines * 2] + ["    ... [class header truncated]"])
        self.out.append(
            CodeChunk(
                self.file,
                qualname,
                "class",
                start,
                end,
                self.header(qualname, "class") + "\n" + text,
                def_line,
                sig,
                doc[:500],
                prefix,
                decos,
            )
        )
        for child, cdefn, cdecos in nested:
            if cdefn.type == "class_definition":
                self._nested_class_methods(child, cdefn, qualname, sig)
            else:
                self.definition(child, cdefn, cdecos, prefix=qualname, parent_sig=sig + ":")

    def _nested_class_methods(self, outer, defn, prefix, parent_sig):
        name = node_text(defn.child_by_field_name("name"))
        qualname = f"{prefix}.{name}"
        sig = _signature(defn, self.lines)
        body = defn.child_by_field_name("body")
        for child in body.named_children if body is not None else []:
            cdefn, cdecos = _unwrap(child)
            if cdefn is not None and cdefn.type == "function_definition":
                self.definition(child, cdefn, cdecos, prefix=qualname, parent_sig=sig + ":")
            elif cdefn is not None and cdefn.type == "class_definition":
                self._nested_class_methods(child, cdefn, qualname, sig)


def _unwrap(node):
    """decorated_definition → (定义节点, 装饰器列表);其它 → (node 或 None, [])。"""
    if node.type == "decorated_definition":
        decorators = [c for c in node.named_children if c.type == "decorator"]
        return node.child_by_field_name("definition"), decorators
    if node.type in ("function_definition", "class_definition"):
        return node, []
    return None, []


def chunk_file(file: str, source: str, max_lines: int = 120) -> list[CodeChunk]:
    return _Chunker(file, source, max_lines).run()
