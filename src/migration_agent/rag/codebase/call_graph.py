"""CallGraph:基于 tree-sitter AST 的轻量符号解析、调用图与符号引用。

    graph = CallGraph(files)                  # {rel_path: source};库相关规则取当前目标(或 spec=PlannerSpec)
    graph.usages_of("<lib>.<name>")           -> list[Usage]   # 模块级符号
    graph.usages_of("<Family>.<attr>")        -> list[Usage]   # 接收者被推断为该类型族(目标包 planner.families)
    graph.callers("pkg/a.py::Foo.bar") / graph.callees(...)
    graph.update_file(path, source | None)    # 增量:只重新解析该文件,全局缓存失效后惰性重算

只看语法树里的代码节点,注释与字符串天然不会命中(tree-sitter 把它们解析成 comment / string 节点)。

库相关的内容全部来自目标包 target.yaml 的 planner 段:
    canonical       子模块别名 → 公开完整名(如旧版子模块里的同一对象);
    families        类型族:roots(根类型,继承到它们的仓库类也属于该族)、constructors(类上调用返回实例的类方法)、
                    instance_returning、nested_classes(`<族>.<X>` 也包含族内类体里的 `class X:`)、
                    factories(调用后返回该族实例的模块级函数)、method_returns(该族实例上调用 attr 返回另一族实例);
    text_patterns   静态解析表达不了的写法(如列表参数形式),对代码节点做正则匹配(注释与字符串除外)。

符号两类:
    模块级符号 `<lib>.X`:import 语句本身、经 import(含别名、函数内 import、仓库内再导出)绑定的名字、
        `<lib>.X` / `alias.X` / `<lib>.<子模块>.X` 属性链;同名局部变量/参数遮蔽时不算。
    方法/属性符号 `<族>.<attr>`:`<recv>.<attr>`,且 recv 的类型被推断为该族(实例或类)。

类型推断是有意保持简单的一层静态推断(动态特性退化为名称匹配 + import 追踪):
    族内类 = 基类(传递地)解析到族 roots 的类;只被族内类继承的 mixin 中的 self 也视为族实例;
    表达式:self/cls、`Model(...)`、构造类方法、工厂函数、`super()`、带注解的参数与返回值
        (Optional/List/Type/字符串注解)、无注解函数的 return 语句、局部/模块级赋值(取使用点之前最近一次)、
        for/推导式变量、字段注解、await、下标取元素、pytest fixture 参数(同文件或上级 conftest.py)。
    推断不到的接收者(如 mock、装饰器动态注入的属性)不计入,宁漏不误。
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from functools import cached_property

from .ast_chunker import node_text, parse_python

_OPTIONAL = frozenset({"Optional", "Union", "Annotated", "Final", "ClassVar", "Required", "NotRequired"})
_SEQUENCE = frozenset(
    {
        "List",
        "list",
        "Sequence",
        "Iterable",
        "Iterator",
        "Set",
        "set",
        "FrozenSet",
        "frozenset",
        "Tuple",
        "tuple",
        "Collection",
        "AsyncIterator",
        "AsyncIterable",
        "Generator",
        "MutableSequence",
    }
)
_TYPE = frozenset({"Type", "type"})
_FUNC_TYPES = ("function_definition",)
_SCOPE_TYPES = ("function_definition", "lambda")


@dataclass(frozen=True)
class Usage:
    symbol: str
    file: str
    line: int
    kind: str  # import | reference | attribute | definition
    text: str
    scope: str = ""


@dataclass
class FuncInfo:
    key: str  # file::qualname
    node: object
    cls: str = ""  # 所属类 key(方法)
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_fixture: bool = False


@dataclass
class ClassInfo:
    key: str
    node: object
    base_nodes: list = field(default_factory=list)
    methods: dict[str, str] = field(default_factory=dict)  # name -> func key
    fields: dict[str, object] = field(default_factory=dict)  # name -> annotation node


def module_names(path: str) -> list[str]:
    """仓库相对路径 → 可能的模块名(src 布局同时登记去掉 `src.` 的名字)。"""
    if not path.endswith(".py"):
        return []
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    names = [".".join(parts)]
    if parts[0] == "src" and len(parts) > 1:
        names.append(".".join(parts[1:]))
    return names


class FileInfo:
    """单文件解析结果:与其它文件无关,可单独重建。"""

    def __init__(self, path: str, source: str):
        self.path = path
        self.source = source
        self.lines = source.splitlines()
        self.tree = parse_python(source)
        self.modules = module_names(path)
        self.package = self._package()
        self.functions: dict[str, FuncInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.func_by_node: dict[int, FuncInfo] = {}
        self.class_by_node: dict[int, ClassInfo] = {}
        self.imports: list[tuple[int, str, str]] = []  # (line, fqn, bound name)
        self.name_refs: dict[str, list] = {}
        self.attr_refs: dict[str, list] = {}
        self.calls: list = []
        self._scope_cache: dict[int, dict[str, list]] = {}
        self._collect(self.tree.root_node, prefix="", cls_key="")

    def _package(self) -> str:
        if not self.modules:
            return ""
        mod = self.modules[0]
        return mod if self.path.endswith("__init__.py") else mod.rpartition(".")[0]

    def key(self, qualname: str) -> str:
        return f"{self.path}::{qualname}"

    def line_text(self, line: int) -> str:
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    # ---- 收集定义与引用 ----
    def _collect(self, node, prefix: str, cls_key: str) -> None:
        stack = [(node, prefix, cls_key)]
        while stack:
            n, pre, ck = stack.pop()
            t = n.type
            if t == "class_definition":
                name = node_text(n.child_by_field_name("name"))
                qual = f"{pre}.{name}" if pre else name
                info = ClassInfo(self.key(qual), n)
                sup = n.child_by_field_name("superclasses")
                if sup is not None:
                    info.base_nodes = [c for c in sup.named_children if c.type != "keyword_argument"]
                self.classes[info.key] = info
                self.class_by_node[n.id] = info
                body = n.child_by_field_name("body")
                for stmt in body.named_children if body is not None else []:
                    target = stmt.named_children[0] if stmt.type == "expression_statement" else None
                    if target is not None and target.type == "assignment":
                        left, ann = target.child_by_field_name("left"), target.child_by_field_name("type")
                        if left is not None and left.type == "identifier" and ann is not None:
                            info.fields[node_text(left)] = ann
                for c in reversed(n.children):
                    stack.append((c, qual, info.key))
                continue
            if t == "function_definition":
                name = node_text(n.child_by_field_name("name"))
                qual = f"{pre}.{name}" if pre else name
                decos = _decorator_names(n)
                owner = self.class_by_node.get(_class_of_body(n))
                info = FuncInfo(
                    self.key(qual),
                    n,
                    cls=owner.key if owner else "",
                    is_classmethod="classmethod" in decos,
                    is_staticmethod="staticmethod" in decos,
                    is_fixture=any(d.endswith("fixture") for d in decos),
                )
                if info.cls and name not in self.classes[info.cls].methods:
                    self.classes[info.cls].methods[name] = info.key
                self.functions[info.key] = info
                self.func_by_node[n.id] = info
                for c in reversed(n.children):
                    stack.append((c, qual, ""))
                continue
            if t in ("import_statement", "import_from_statement"):
                self._record_import(n)
                continue
            if t == "identifier":
                self.name_refs.setdefault(node_text(n), []).append(n)
            elif t == "attribute":
                attr = n.child_by_field_name("attribute")
                if attr is not None:
                    self.attr_refs.setdefault(node_text(attr), []).append(n)
            elif t == "call":
                self.calls.append(n)
            for c in reversed(n.children):
                stack.append((c, pre, ck))

    def _record_import(self, n) -> None:
        for name, fqn, line in self.import_bindings(n):
            self.imports.append((line, fqn, name))

    def import_bindings(self, n) -> list[tuple[str, str, int]]:
        """import 语句 → [(绑定名, 完整名, 行号)]。`import a.b` 绑定 `a`;星号导入忽略。"""
        out = []
        if n.type == "import_statement":
            for item in n.named_children:
                if item.type == "dotted_name":
                    full = node_text(item)
                    out.append((full.split(".")[0], full.split(".")[0], item.start_point.row + 1))
                elif item.type == "aliased_import":
                    full = node_text(item.child_by_field_name("name"))
                    alias = node_text(item.child_by_field_name("alias"))
                    out.append((alias, full, item.start_point.row + 1))
            return out
        mod_node = n.child_by_field_name("module_name")
        module = self._absolute(node_text(mod_node)) if mod_node is not None else ""
        for item in n.named_children:
            if item == mod_node:
                continue
            if item.type == "dotted_name":
                name = node_text(item)
                out.append((name.split(".")[-1], f"{module}.{name}", item.start_point.row + 1))
            elif item.type == "aliased_import":
                name = node_text(item.child_by_field_name("name"))
                alias = node_text(item.child_by_field_name("alias"))
                out.append((alias, f"{module}.{name}", item.start_point.row + 1))
        return out

    def _absolute(self, module: str) -> str:
        if not module.startswith("."):
            return module
        dots = len(module) - len(module.lstrip("."))
        rest = module[dots:]
        base = self.package.split(".") if self.package else []
        if dots > 1:
            base = base[: len(base) - (dots - 1)] if dots - 1 <= len(base) else []
        return ".".join([*base, rest] if rest else base)

    # ---- 作用域 ----
    def scope_bindings(self, scope) -> dict[str, list]:
        """作用域节点(函数/lambda/模块)内绑定的名字 → [(kind, node, extra)],按出现顺序。

        kind: param | assign | iter | import | def | class | other
        """
        cached = self._scope_cache.get(scope.id)
        if cached is not None:
            return cached
        out: dict[str, list] = {}

        def add(name_node, kind, value=None):
            out.setdefault(node_text(name_node), []).append((kind, name_node, value))

        if scope.type in _SCOPE_TYPES:
            params = scope.child_by_field_name("parameters")
            for i, p in enumerate(params.named_children if params is not None else []):
                ann = p.child_by_field_name("type")
                if p.type == "identifier":
                    add(p, "param", (None, i))
                elif p.type in ("typed_parameter",):
                    ident = next((c for c in p.named_children if c.type == "identifier"), None)
                    if ident is not None:
                        add(ident, "param", (ann, i))
                elif p.type in ("default_parameter", "typed_default_parameter"):
                    ident = p.child_by_field_name("name")
                    if ident is not None:
                        add(ident, "param", (ann, i))
                elif p.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                    ident = next((c for c in p.named_children if c.type == "identifier"), None)
                    if ident is not None:
                        add(ident, "other")
            body = scope.child_by_field_name("body")
        else:
            body = scope
        if body is None:
            self._scope_cache[scope.id] = out
            return out
        stack = list(reversed(body.children))
        while stack:
            n = stack.pop()
            t = n.type
            if t in ("function_definition", "class_definition"):
                name = n.child_by_field_name("name")
                if name is not None:
                    add(name, "def" if t == "function_definition" else "class", n)
                continue  # 不进入嵌套作用域
            if t == "lambda":
                continue
            if t == "decorated_definition":
                stack.extend(reversed(n.children))
                continue
            if t in ("import_statement", "import_from_statement"):
                for name, fqn, _line in self.import_bindings(n):
                    out.setdefault(name, []).append(("import", n, fqn))
                continue
            if t == "assignment":
                left, right = n.child_by_field_name("left"), n.child_by_field_name("right")
                ann = n.child_by_field_name("type")
                if left is not None and left.type == "identifier":
                    if ann is not None:
                        add(left, "assign", ("ann", ann, right))
                    else:
                        add(left, "assign", right)
                elif left is not None:
                    for ident in _pattern_names(left):
                        add(ident, "other")
            elif t == "named_expression":
                name = n.child_by_field_name("name")
                if name is not None:
                    add(name, "assign", n.child_by_field_name("value"))
            elif t in ("for_statement",):
                left, right = n.child_by_field_name("left"), n.child_by_field_name("right")
                if left is not None and left.type == "identifier":
                    add(left, "iter", right)
                elif left is not None:
                    for ident in _pattern_names(left):
                        add(ident, "other")
            elif t in ("with_item", "except_clause"):
                for c in n.named_children:
                    if c.type == "as_pattern":
                        alias = c.child_by_field_name("alias")
                        for ident in _pattern_names(alias) if alias is not None else []:
                            add(ident, "other")
            elif t in ("global_statement", "nonlocal_statement"):
                for c in n.named_children:
                    out.setdefault(node_text(c), []).append(("global", c, None))
            if t in (
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
            ):
                continue  # 推导式变量是推导式自己的作用域
            stack.extend(reversed(n.children))
        self._scope_cache[scope.id] = out
        return out


def _class_of_body(func_node) -> int:
    """方法定义 → 所属 class_definition 的 node.id(经过 block / decorated_definition)。"""
    p = func_node.parent
    if p is not None and p.type == "decorated_definition":
        p = p.parent
    if p is not None and p.type == "block":
        p = p.parent
    return p.id if p is not None and p.type == "class_definition" else -1


def _decorator_names(func_node) -> list[str]:
    p = func_node.parent
    if p is None or p.type != "decorated_definition":
        return []
    out = []
    for d in p.named_children:
        if d.type != "decorator":
            continue
        expr = d.named_children[0] if d.named_children else None
        if expr is not None and expr.type == "call":
            expr = expr.child_by_field_name("function")
        if expr is not None:
            out.append(node_text(expr))
    return out


def _pattern_names(node) -> list:
    if node is None:
        return []
    if node.type == "identifier":
        return [node]
    out = []
    for c in node.named_children:
        out.extend(_pattern_names(c))
    return out


def _dotted(node) -> str | None:
    """纯标识符属性链 → 'a.b.c';其它表达式 → None。"""
    parts = []
    while node is not None and node.type == "attribute":
        attr = node.child_by_field_name("attribute")
        parts.append(node_text(attr))
        node = node.child_by_field_name("object")
    if node is None or node.type != "identifier":
        return None
    parts.append(node_text(node))
    return ".".join(reversed(parts))


def _is_binding_position(ident) -> bool:
    """标识符是否处在「不是对名字的读取」的位置。"""
    p = ident.parent
    if p is None:
        return False
    t = p.type
    if t == "attribute" and p.child_by_field_name("attribute") == ident:
        return True
    if t == "keyword_argument" and p.child_by_field_name("name") == ident:
        return True
    if t in ("function_definition", "class_definition") and p.child_by_field_name("name") == ident:
        return True
    if t in (
        "parameters",
        "typed_parameter",
        "default_parameter",
        "typed_default_parameter",
        "lambda_parameters",
    ):
        if t in ("default_parameter", "typed_default_parameter"):
            return p.child_by_field_name("name") == ident
        return t != "typed_parameter" or p.named_children[0] == ident
    if t in ("list_splat_pattern", "dictionary_splat_pattern") and p.parent is not None:
        return p.parent.type in ("parameters", "lambda_parameters")
    if t in ("dotted_name", "aliased_import", "import_statement", "import_from_statement"):
        return True
    if t == "assignment" and p.child_by_field_name("left") == ident:
        return True
    return t in ("global_statement", "nonlocal_statement")


def _planner_spec():
    from ...migration.targets import active

    return active().spec.planner


class CallGraph:
    def __init__(self, files: dict[str, str] | None = None, spec=None):
        spec = spec if spec is not None else _planner_spec()
        self._canon: dict[str, str] = dict(spec.canonical)
        self._families = dict(spec.families)
        self._text_patterns = {k: re.compile(v) for k, v in spec.text_patterns.items()}
        self._imported_vars = bool(getattr(spec, "imported_variable_types", False))
        self._roots: dict[str, str] = {r: name for name, fam in self._families.items() for r in fam.roots}
        self._factories: dict[str, str] = {
            f: name for name, fam in self._families.items() for f in fam.factories
        }
        self.files: dict[str, FileInfo] = {}
        for path, source in sorted((files or {}).items()):
            self._add(path, source)
        self._invalidate()

    # ---- 文件管理 ----
    def _add(self, path: str, source: str) -> None:
        if path.endswith(".py"):
            self.files[path] = FileInfo(path, source)

    def update_file(self, path: str, source: str | None) -> None:
        self.files.pop(path, None)
        if source is not None:
            self._add(path, source)
        self._invalidate()

    def _invalidate(self) -> None:
        for name in (
            "_modules",
            "_defs",
            "_family_keys",
            "_model_keys",
            "_mixin_keys",
            "_subclasses",
            "_edges",
        ):
            self.__dict__.pop(name, None)
        self._type_memo: dict = {}
        self._ret_memo: dict = {}
        self._in_progress: set = set()

    @cached_property
    def _modules(self) -> dict[str, FileInfo]:
        return {m: f for f in self.files.values() for m in f.modules}

    @cached_property
    def _defs(self) -> dict[str, str]:
        """完整名(模块.qualname)→ key,仅模块顶层与类内定义。"""
        out = {}
        for f in self.files.values():
            for mod in f.modules:
                for key in [*f.classes, *f.functions]:
                    out[f"{mod}.{key.split('::', 1)[1]}"] = key
        return out

    def info_of(self, key: str):
        path = key.split("::", 1)[0]
        f = self.files.get(path)
        if f is None:
            return None
        return f.classes.get(key) or f.functions.get(key)

    # ---- 名字解析 ----
    def canonical(self, fqn: str) -> str:
        """沿仓库内再导出与目标包声明的子模块别名(planner.canonical)归一化完整名。"""
        for _ in range(12):
            if fqn in self._canon:
                fqn = self._canon[fqn]
                continue
            if fqn in self._defs:
                return self._defs[fqn]
            parts = fqn.split(".")
            moved = False
            for i in range(len(parts) - 1, 0, -1):
                mod = ".".join(parts[:i])
                f = self._modules.get(mod)
                if f is None:
                    continue
                name, rest = parts[i], parts[i + 1 :]
                bound = self._module_binding(f, name)
                if bound and bound[0] == "import":
                    fqn = ".".join([bound[2], *rest])
                    moved = True
                break
            if not moved:
                return fqn
        return fqn

    def _module_binding(self, f: FileInfo, name: str):
        rows = f.scope_bindings(f.tree.root_node).get(name)
        return rows[-1] if rows else None

    def lookup(self, f: FileInfo, ident) -> tuple | None:
        """标识符 → 绑定 (kind, node, extra, scope_node);逐层函数作用域 → 推导式 → 模块。"""
        name = node_text(ident)
        n = ident.parent
        while n is not None:
            if n.type in (
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
            ):
                for clause in n.named_children:
                    if clause.type == "for_in_clause":
                        left = clause.child_by_field_name("left")
                        if left is not None and left.type == "identifier" and node_text(left) == name:
                            return ("iter", left, clause.child_by_field_name("right"), n)
                        if left is not None and any(node_text(i) == name for i in _pattern_names(left)):
                            return ("other", left, None, n)
            if n.type in _SCOPE_TYPES:
                rows = f.scope_bindings(n).get(name)
                if rows:
                    if any(r[0] == "global" for r in rows):
                        break
                    return (*_latest(rows, ident), n)
            n = n.parent
        rows = f.scope_bindings(f.tree.root_node).get(name)
        if rows:
            return (*_latest(rows, ident), f.tree.root_node)
        return None

    def resolve_ident(self, f: FileInfo, ident) -> str | None:
        """标识符 → 完整名或仓库 key;局部变量/参数 → None。"""
        b = self.lookup(f, ident)
        if b is None:
            return None
        kind, node, extra, _scope = b
        if kind == "import":
            return self.canonical(extra)
        if kind in ("def", "class"):
            info = f.func_by_node.get(extra.id) or f.class_by_node.get(extra.id)
            return info.key if info else None
        return None

    def resolve_expr(self, f: FileInfo, node) -> str | None:
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        ident = node
        while ident.type == "attribute":
            ident = ident.child_by_field_name("object")
        base = self.resolve_ident(f, ident)
        if base is None:
            return None
        if not rest:
            return base
        if "::" in base:  # 仓库定义上的属性,如 Class.method
            info = self.info_of(base)
            if isinstance(info, ClassInfo) and "." not in rest and rest in info.methods:
                return info.methods[rest]
            return f"{base}.{rest}"
        return self.canonical(f"{base}.{rest}")

    # ---- 模型类 ----
    def _class_bases(self, info: ClassInfo) -> list[str]:
        f = self.files[info.key.split("::", 1)[0]]
        out = []
        for b in info.base_nodes:
            if b.type == "subscript":
                b = b.child_by_field_name("value")
            r = self.resolve_expr(f, b) if b is not None else None
            if r:
                out.append(r)
        return out

    @cached_property
    def _subclasses(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for f in self.files.values():
            for key, info in f.classes.items():
                for base in self._class_bases(info):
                    out.setdefault(base, set()).add(key)
        return out

    @cached_property
    def _family_keys(self) -> dict[str, set[str]]:
        """族名 → 继承(传递地)到该族 roots 的仓库类。"""
        out: dict[str, set[str]] = {}
        for name, fam in self._families.items():
            models: set[str] = set()
            frontier = list(fam.roots)
            while frontier:
                cur = frontier.pop()
                for sub in self._subclasses.get(cur, ()):
                    if sub not in models:
                        models.add(sub)
                        frontier.append(sub)
            out[name] = models
        return out

    @cached_property
    def _model_keys(self) -> set[str]:
        return set().union(*self._family_keys.values()) if self._family_keys else set()

    @cached_property
    def _mixin_keys(self) -> set[str]:
        """只被模型类继承的仓库类:其方法中的 self 运行时就是模型实例。"""
        out = set()
        for f in self.files.values():
            for key in f.classes:
                if key in self._model_keys:
                    continue
                subs = self._subclasses.get(key, set())
                if subs and subs <= self._model_keys:
                    out.add(key)
        return out

    def is_model(self, key: str | None) -> bool:
        return bool(key) and (key in self._roots or key in self._model_keys or key in self._mixin_keys)

    def family_of(self, key: str | None) -> str | None:
        """类 key(或根类型完整名)所属的类型族;mixin 归到继承它的族里第一个。"""
        if not key:
            return None
        if key in self._roots:
            return self._roots[key]
        for name, keys in self._family_keys.items():
            if key in keys:
                return name
        if key in self._mixin_keys:
            subs = self._subclasses.get(key, set())
            for name, keys in self._family_keys.items():
                if subs & keys:
                    return name
        return None

    def in_family(self, key: str | None, family: str) -> bool:
        if not key:
            return False
        if key in self._roots:
            return self._roots[key] == family
        keys = self._family_keys.get(family, set())
        if key in keys:
            return True
        subs = self._subclasses.get(key, set())
        return key in self._mixin_keys and bool(subs) and subs <= keys

    def model_classes(self) -> set[str]:
        return set(self._model_keys)

    def _mro(self, key: str) -> list[str]:
        seen, order, stack = set(), [], [key]
        while stack:
            k = stack.pop(0)
            if k in seen:
                continue
            seen.add(k)
            order.append(k)
            info = self.info_of(k)
            if isinstance(info, ClassInfo):
                stack.extend(self._class_bases(info))
        return order

    def find_method(self, cls_key: str, name: str) -> str | None:
        for k in self._mro(cls_key):
            info = self.info_of(k)
            if isinstance(info, ClassInfo) and name in info.methods:
                return info.methods[name]
        return None

    def find_field(self, cls_key: str, name: str):
        for k in self._mro(cls_key):
            info = self.info_of(k)
            if isinstance(info, ClassInfo) and name in info.fields:
                return k, info.fields[name]
        return None

    # ---- 类型推断 ----
    def type_of(self, f: FileInfo, node, depth: int = 0) -> tuple[str, str] | None:
        """表达式 → ("inst"|"cls"|"seq"|"func", key) 或 None。"""
        if node is None or depth > 12:
            return None
        memo_key = (f.path, node.id)
        if memo_key in self._type_memo:
            return self._type_memo[memo_key]
        self._type_memo[memo_key] = None  # 递归保护
        result = self._type_of(f, node, depth)
        self._type_memo[memo_key] = result
        return result

    def _type_of(self, f: FileInfo, node, depth: int):
        t = node.type
        if t in ("parenthesized_expression", "await"):
            inner = node.named_children[-1] if node.named_children else None
            return self.type_of(f, inner, depth + 1)
        if t == "identifier":
            return self._ident_type(f, node, depth)
        if t == "attribute":
            resolved = self.resolve_expr(f, node)
            entity = self._entity_type(resolved) if resolved else None
            if entity:
                return entity
            obj = self.type_of(f, node.child_by_field_name("object"), depth + 1)
            attr = node_text(node.child_by_field_name("attribute"))
            if obj and obj[0] in ("inst", "cls") and "::" in obj[1]:
                fld = self.find_field(obj[1], attr)
                if fld is not None and obj[0] == "inst":
                    owner_file = self.files[fld[0].split("::", 1)[0]]
                    return self.annotation_type(owner_file, fld[1])
                method = self.find_method(obj[1], attr)
                if method:
                    info = self.info_of(method)
                    if info is not None and "property" in _decorator_names(info.node):
                        return self.return_type(method)
                    return ("func", method)
            return None
        if t == "call":
            return self._call_type(f, node, depth)
        if t == "subscript":
            val = self.type_of(f, node.child_by_field_name("value"), depth + 1)
            if val and val[0] == "seq":
                return ("inst", val[1])
            return None
        if t in ("list", "tuple", "set"):
            for c in node.named_children:
                ct = self.type_of(f, c, depth + 1)
                if ct and ct[0] == "inst":
                    return ("seq", ct[1])
            return None
        if t in ("list_comprehension", "set_comprehension", "generator_expression"):
            body = node.child_by_field_name("body")
            ct = self.type_of(f, body, depth + 1)
            return ("seq", ct[1]) if ct and ct[0] == "inst" else None
        if t == "string":
            return None
        return None

    def _entity_type(self, resolved: str):
        if resolved in self._roots:
            return ("cls", resolved)
        if resolved in self._factories:  # 工厂函数:调用后返回该族根类型的实例
            return ("factory", self._families[self._factories[resolved]].roots[0])
        info = self.info_of(resolved) if "::" in resolved else None
        if isinstance(info, ClassInfo):
            return ("cls", resolved)
        if isinstance(info, FuncInfo):
            return ("func", resolved)
        return None

    def _ident_type(self, f: FileInfo, ident, depth: int):
        name = node_text(ident)
        b = self.lookup(f, ident)
        if b is None:
            return None
        kind, node, extra, scope = b
        if kind in ("import", "def", "class"):
            resolved = self.resolve_ident(f, ident)
            entity = self._entity_type(resolved) if resolved else None
            if entity is None and kind == "import" and resolved and self._imported_vars:
                return self._module_var_type(resolved, depth)
            return entity
        if kind == "param":
            ann, index = extra
            func = f.func_by_node.get(scope.id) if scope is not None else None
            if ann is not None:
                return self.annotation_type(f, ann)
            if func is not None and func.cls and index == 0 and not func.is_staticmethod:
                return ("cls" if func.is_classmethod else "inst", func.cls)
            if func is not None and name not in ("self", "cls"):
                return self._fixture_type(f, func, name)
            return None
        if kind == "assign":
            if isinstance(extra, tuple):  # 带注解的赋值:注解优先,其次右值
                return self.annotation_type(f, extra[1]) or self.type_of(f, extra[2], depth + 1)
            return self.type_of(f, extra, depth + 1)
        if kind == "iter":
            it = self.type_of(f, extra, depth + 1)
            return ("inst", it[1]) if it and it[0] == "seq" else None
        return None

    def _module_var_type(self, fqn: str, depth: int):
        """`pkg.mod.name` 是仓库模块的模块级变量时,按该模块里最后一次赋值推断类型(planner.imported_variable_types)。"""
        mod, _, name = fqn.rpartition(".")
        g = self._modules.get(mod)
        if g is None or not name:
            return None
        rows = [r for r in g.scope_bindings(g.tree.root_node).get(name, []) if r[0] == "assign"]
        if not rows:
            return None
        extra = rows[-1][2]
        if isinstance(extra, tuple):
            return self.annotation_type(g, extra[1]) or self.type_of(g, extra[2], depth + 1)
        return self.type_of(g, extra, depth + 1)

    def _fixture_type(self, f: FileInfo, func: FuncInfo, name: str):
        """pytest:测试函数参数名 = 同文件或上级目录 conftest.py 中的 fixture 函数名。"""
        if not (func.key.split("::")[1].split(".")[-1].startswith("test") or func.is_fixture):
            return None
        candidates = [f]
        d = posixpath.dirname(f.path)
        while True:
            cf = self.files.get(posixpath.join(d, "conftest.py") if d else "conftest.py")
            if cf is not None and cf is not f:
                candidates.append(cf)
            if not d:
                break
            d = posixpath.dirname(d)
        for cf in candidates:
            fx = cf.functions.get(cf.key(name))
            if fx is not None and fx.is_fixture:
                return self.return_type(fx.key)
        return None

    def _call_type(self, f: FileInfo, node, depth: int):
        fn = node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier" and node_text(fn) == "super":
            cls = self._enclosing_class(f, node)
            func = f.func_by_node.get(self._enclosing_func_node(node))
            # 类方法里的 super() 代理的是类,实例方法里代理的是实例
            return ("cls" if func is not None and func.is_classmethod else "inst", cls) if cls else None
        if fn.type == "attribute":
            attr = node_text(fn.child_by_field_name("attribute"))
            recv = self.type_of(f, fn.child_by_field_name("object"), depth + 1)
            fam = self._families.get(self.family_of(recv[1]) or "") if recv else None
            if recv and fam is not None and recv[0] == "cls" and attr in fam.constructors:
                return ("inst", recv[1])
            if recv and fam is not None and recv[0] == "inst" and attr in fam.instance_returning:
                return recv
            if recv and fam is not None and recv[0] == "inst" and attr in fam.method_returns:
                other = self._families.get(fam.method_returns[attr])
                if other is not None and other.roots:
                    return ("inst", other.roots[0])
            if recv and recv[0] in ("inst", "cls") and "::" in recv[1]:
                method = self.find_method(recv[1], attr)
                if method:
                    return self.return_type(method, owner=recv[1])
        ft = self.type_of(f, fn, depth + 1)
        if ft is None:
            return None
        if ft[0] in ("cls", "factory"):
            return ("inst", ft[1])
        if ft[0] == "inst":  # 调用一个实例:目标包声明了 call_returns 的族(如会话工厂)
            fam = self._families.get(self.family_of(ft[1]) or "")
            other = self._families.get(fam.call_returns) if fam is not None and fam.call_returns else None
            if other is not None and other.roots:
                return ("inst", other.roots[0])
        if ft[0] == "func":
            return self.return_type(ft[1])
        return None

    @staticmethod
    def _enclosing_func_node(node) -> int:
        n = node.parent
        while n is not None and n.type != "function_definition":
            n = n.parent
        return n.id if n is not None else -1

    def _enclosing_class(self, f: FileInfo, node) -> str:
        n = node.parent
        while n is not None:
            if n.type == "function_definition":
                info = f.func_by_node.get(n.id)
                if info and info.cls:
                    return info.cls
            if n.type == "class_definition":
                info = f.class_by_node.get(n.id)
                return info.key if info else ""
            n = n.parent
        return ""

    def return_type(self, func_key: str, owner: str = ""):
        memo = (func_key, owner)
        if memo in self._ret_memo:
            return self._ret_memo[memo]
        self._ret_memo[memo] = None
        info = self.info_of(func_key)
        result = None
        if isinstance(info, FuncInfo):
            f = self.files[func_key.split("::", 1)[0]]
            ann = info.node.child_by_field_name("return_type")
            if ann is not None:
                result = self.annotation_type(f, ann, self_cls=owner or info.cls)
            else:
                for ret in _returns(info.node):
                    value = ret.named_children[0] if ret.named_children else None
                    rt = self.type_of(f, value) if value is not None else None
                    if rt and owner and rt[1] == info.cls and owner != info.cls:
                        rt = (rt[0], owner)  # cls(...) 在子类上调用时返回子类实例
                    if rt and rt[0] in ("inst", "seq", "cls"):
                        result = rt
                        break
        self._ret_memo[memo] = result
        return result

    def annotation_type(self, f: FileInfo, ann, self_cls: str = ""):
        """类型注解节点 → ("inst"|"cls"|"seq", key)。"""
        if ann is None:
            return None
        node = ann
        while node.type == "type" and node.named_children:
            node = node.named_children[0]
        if node.type == "string":
            text = node_text(node).strip("rRbBuU").strip("\"'").strip()
            return self._annotation_text(f, text, self_cls)
        if node.type in ("identifier", "attribute"):
            text = node_text(node)
            if text in ("Self", "typing.Self", "typing_extensions.Self") and self_cls:
                return ("inst", self_cls)
            r = self.resolve_expr(f, node)
            et = self._entity_type(r) if r else None
            return ("inst", et[1]) if et and et[0] == "cls" else None
        if node.type == "binary_operator":  # X | None
            for side in (node.child_by_field_name("left"), node.child_by_field_name("right")):
                st = self.annotation_type(f, side, self_cls)
                if st:
                    return st
            return None
        if node.type in ("generic_type", "subscript"):
            head = (
                node.named_children[0] if node.type == "generic_type" else node.child_by_field_name("value")
            )
            head_name = node_text(head).split(".")[-1]
            args_node = (
                node.named_children[1]
                if node.type == "generic_type" and len(node.named_children) > 1
                else node.child_by_field_name("subscript")
            )
            args = (
                [a for a in args_node.named_children]
                if args_node is not None and args_node.type in ("type_parameter",)
                else ([args_node] if args_node is not None else [])
            )
            if node.type == "subscript":
                args = [c for c in node.named_children[1:]]
            if head_name in _OPTIONAL:
                for a in args:
                    st = self.annotation_type(f, a, self_cls)
                    if st:
                        return st
                return None
            inner = self.annotation_type(f, args[0], self_cls) if args else None
            if head_name in _SEQUENCE and inner and inner[0] == "inst":
                return ("seq", inner[1])
            if head_name in _TYPE and inner and inner[0] == "inst":
                return ("cls", inner[1])
            return None
        return None

    def _annotation_text(self, f: FileInfo, text: str, self_cls: str):
        if not text:
            return None
        tree = parse_python(text)
        root = tree.root_node
        expr = (
            root.named_children[0].named_children[0]
            if root.named_children and root.named_children[0].named_children
            else None
        )
        if expr is None:
            return None
        # 字符串注解里的名字在定义文件的模块作用域解析
        dotted = _dotted(expr)
        if dotted:
            head, _, rest = dotted.partition(".")
            rows = f.scope_bindings(f.tree.root_node).get(head)
            if rows:
                kind, node, extra = rows[-1]
                if kind == "import":
                    r = self.canonical(".".join(x for x in (extra, rest) if x))
                elif kind == "class":
                    info = f.class_by_node.get(extra.id)
                    r = info.key if info else None
                else:
                    r = None
                et = self._entity_type(r) if r else None
                return ("inst", et[1]) if et and et[0] == "cls" else None
            if head == "Self" and self_cls:
                return ("inst", self_cls)
            # 嵌套类等:按 qualname 在本文件查找
            key = f.key(dotted)
            if key in f.classes:
                return ("inst", key)
            return None
        sub = expr.child_by_field_name("value") if expr.type == "subscript" else None
        if sub is not None:
            head = node_text(sub).split(".")[-1]
            inner_nodes = [c for c in expr.named_children[1:]]
            inner = None
            for c in inner_nodes:
                inner = self._annotation_text(f, node_text(c), self_cls)
                if inner:
                    break
            if head in _OPTIONAL:
                return inner
            if head in _SEQUENCE and inner and inner[0] == "inst":
                return ("seq", inner[1])
            if head in _TYPE and inner and inner[0] == "inst":
                return ("cls", inner[1])
        return None

    # ---- 符号引用 ----
    def usages_of(self, symbol: str) -> list[Usage]:
        symbol = self._canon.get(symbol, symbol)
        head, _, attr = symbol.partition(".")
        if symbol in self._text_patterns:
            rows = self._text_pattern_usages(symbol)
        elif head in self._families and attr:
            rows = self._model_attr_usages(symbol, attr, head)
        else:
            rows = self._module_symbol_usages(symbol)
        seen: dict[tuple[str, int], Usage] = {}
        for u in sorted(rows, key=lambda u: (u.file, u.line, u.kind != "import")):
            seen.setdefault((u.file, u.line), u)
        return sorted(seen.values(), key=lambda u: (u.file, u.line))

    def _scope_name(self, f: FileInfo, node) -> str:
        n = node.parent
        names = []
        while n is not None:
            if n.type in ("function_definition", "class_definition"):
                names.append(node_text(n.child_by_field_name("name")))
            n = n.parent
        return ".".join(reversed(names))

    def _module_symbol_usages(self, symbol: str) -> list[Usage]:
        # 仓库内定义(如 `pkg.models.User`)归一化为 `file::qualname` key,与解析结果同口径比较
        target = self.canonical(symbol)
        leaf = target.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        out = []
        for f in self.files.values():
            for line, fqn, _bound in f.imports:
                if self.canonical(fqn) == target:
                    out.append(Usage(symbol, f.path, line, "import", f.line_text(line)))
            names = {leaf} | {bound for _l, fqn, bound in f.imports if self.canonical(fqn) == target}
            # 函数内 import 也会被 f.imports 收录;别名名字同样检查
            for name in names:
                for ident in f.name_refs.get(name, []):
                    if _is_binding_position(ident):
                        continue
                    if self.resolve_ident(f, ident) == target:
                        line = ident.start_point.row + 1
                        out.append(
                            Usage(
                                symbol,
                                f.path,
                                line,
                                "reference",
                                f.line_text(line),
                                self._scope_name(f, ident),
                            )
                        )
            for node in f.attr_refs.get(leaf, []):
                if self.resolve_expr(f, node) == target:
                    line = node.start_point.row + 1
                    out.append(
                        Usage(symbol, f.path, line, "attribute", f.line_text(line), self._scope_name(f, node))
                    )
        return out

    def _text_pattern_usages(self, symbol: str) -> list[Usage]:
        """目标包 planner.text_patterns:对代码行做正则匹配,命中位置落在注释 / 字符串节点里的不算。"""
        pat = self._text_patterns[symbol]
        out = []
        for f in self.files.values():
            lines = f.source.splitlines()
            for i, line in enumerate(lines):
                for m in pat.finditer(line):
                    col = len(line[: m.start()].encode("utf-8"))  # tree-sitter 的列是字节偏移
                    node = f.tree.root_node.descendant_for_point_range((i, col), (i, col))
                    if node is not None and _in_comment_or_string(node):
                        continue
                    out.append(
                        Usage(symbol, f.path, i + 1, "pattern", line.strip(), self._scope_name(f, node))
                    )
                    break
        return out

    def _model_attr_usages(self, symbol: str, attr: str, family: str) -> list[Usage]:
        nested = attr in self._families[family].nested_classes
        out = []
        for f in self.files.values():
            for node in f.attr_refs.get(attr, []):
                recv = self.type_of(f, node.child_by_field_name("object"))
                if recv and recv[0] in ("inst", "cls") and self.in_family(recv[1], family):
                    line = node.start_point.row + 1
                    out.append(
                        Usage(symbol, f.path, line, "attribute", f.line_text(line), self._scope_name(f, node))
                    )
            if nested:
                for key, info in f.classes.items():
                    parent = key.rsplit(".", 1)[0] if "." in key.split("::", 1)[1] else ""
                    if key.endswith("." + attr) and parent and self.in_family(parent, family):
                        line = info.node.start_point.row + 1
                        out.append(
                            Usage(
                                symbol, f.path, line, "definition", f.line_text(line), parent.split("::")[1]
                            )
                        )
        return out

    def imported_files(self, path: str) -> set[str]:
        """该文件 import 的仓库内文件(经再导出归一化)。"""
        f = self.files.get(path)
        out: set[str] = set()
        if f is None:
            return out
        for _line, fqn, _bound in f.imports:
            # 直接被 import 的仓库模块(`from M import N` 的 M,或 N 本身是子模块)
            for mod in (fqn, fqn.rpartition(".")[0]):
                if mod in self._modules:
                    out.add(self._modules[mod].path)
                    break
            r = self.canonical(fqn)
            if "::" in r:
                out.add(r.split("::", 1)[0])
        out.discard(path)
        return out

    # ---- 调用图 ----
    @cached_property
    def _edges(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        callees: dict[str, set[str]] = {}
        callers: dict[str, set[str]] = {}
        for f in self.files.values():
            for call in f.calls:
                src = self._enclosing_def(f, call)
                if not src:
                    continue
                target = self._call_target(f, call)
                if target and target != src:
                    callees.setdefault(src, set()).add(target)
                    callers.setdefault(target, set()).add(src)
        return callees, callers

    def _enclosing_def(self, f: FileInfo, node) -> str:
        n = node.parent
        while n is not None:
            if n.type == "function_definition":
                info = f.func_by_node.get(n.id)
                return info.key if info else ""
            n = n.parent
        return ""

    def _call_target(self, f: FileInfo, call) -> str | None:
        fn = call.child_by_field_name("function")
        if fn is None:
            return None
        r = self.resolve_expr(f, fn) if fn.type in ("identifier", "attribute") else None
        if r and "::" in r and self.info_of(r) is not None:
            return r
        if fn.type == "attribute":
            recv = self.type_of(f, fn.child_by_field_name("object"))
            if recv and recv[0] in ("inst", "cls") and "::" in recv[1]:
                return self.find_method(recv[1], node_text(fn.child_by_field_name("attribute")))
        return None

    def callees(self, key: str) -> set[str]:
        return set(self._edges[0].get(key, ()))

    def callers(self, key: str) -> set[str]:
        return set(self._edges[1].get(key, ()))


def _in_comment_or_string(node) -> bool:
    n = node
    while n is not None:
        if n.type in ("comment", "string", "string_content"):
            return True
        n = n.parent
    return False


def _latest(rows: list, ident) -> tuple:
    """取使用点之前最后一次绑定;都在之后(循环里先用后赋)时取第一次。"""
    before = [r for r in rows if r[1].start_byte <= ident.start_byte]
    return before[-1] if before else rows[0]


def _returns(func_node) -> list:
    body = func_node.child_by_field_name("body")
    out, stack = [], [body] if body is not None else []
    while stack:
        n = stack.pop()
        if n.type in ("function_definition", "class_definition", "lambda"):
            continue
        if n.type == "return_statement":
            out.append(n)
            continue
        stack.extend(reversed(n.children))
    return out
