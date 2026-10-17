"""Repository file operations over Executor; host and container paths never mix.

The helper runs in the selected environment. Checkpoints include binary/untracked
files and modes, but exclude Git metadata, installed environments and test outputs.
Symlinks are rejected rather than followed during validation or publication.
"""

from __future__ import annotations

import base64
import json

from .protocol import Executor

IGNORED = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".migration_agent",
    ".tox",
    "dist",
    "build",
    "*.egg-info",
)

_HELPER = r"""
import base64, fnmatch, json, os, pathlib, sys
root = pathlib.Path.cwd().resolve()
req = json.loads(sys.argv[1])
ignored = req['ignored']
def path(rel):
    p = root / rel
    if p.is_symlink() or any(q.is_symlink() for q in p.parents if q != root and root in q.parents):
        raise PermissionError('symlinks are not supported: ' + rel)
    p = p.resolve()
    if not p.is_relative_to(root):
        raise PermissionError('path escapes repository: ' + rel)
    return p
def files():
    out = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not any(fnmatch.fnmatch(d, g) for g in ignored))
        for d in dirs:
            path((pathlib.Path(directory) / d).relative_to(root).as_posix())
        for name in sorted(names):
            rel = (pathlib.Path(directory) / name).relative_to(root).as_posix()
            p = path(rel)
            if p.is_file(): out.append((rel, p))
    return out
op = req['op']
if op == 'tree':
    result = {r: {'data': base64.b64encode(p.read_bytes()).decode(), 'mode': p.stat().st_mode & 511}
              for r, p in files()}
elif op == 'list': result = [r for r, p in files()]
elif op == 'read': result = path(req['path']).read_text(encoding='utf-8', errors='replace')
elif op == 'write':
    p = path(req['path']); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(req['data'])); result = None
elif op == 'chunk':
    p = path(req['path']); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('ab' if req['append'] else 'wb') as f: f.write(base64.b64decode(req['data']))
    result = None
elif op == 'mode':
    path(req['path']).chmod(req['mode']); result = None
elif op == 'remove':
    p = path(req['path'])
    if p.exists(): p.unlink()
    result = None
elif op == 'restore':
    desired = req['tree']; current = dict(files())
    for rel, p in current.items():
        if rel not in desired: p.unlink()
    for rel, entry in desired.items():
        p = path(rel); data = base64.b64decode(entry['data'])
        if not p.exists() or p.read_bytes() != data:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        if (p.stat().st_mode & 511) != entry['mode']: p.chmod(entry['mode'])
    result = None
else: raise ValueError(op)
print(json.dumps(result))
"""


class RepositoryFiles:
    """Small JSON RPC adapter using only the existing Executor protocol."""

    def __init__(self, executor: Executor):
        self.executor = executor

    def _call(self, op: str, **kwargs):
        request = json.dumps({"op": op, "ignored": IGNORED, **kwargs})
        res = self.executor.exec(["python", "-I", "-c", _HELPER, request])
        if not res.ok:
            raise RuntimeError(res.stderr or res.stdout or f"file operation failed: {op}")
        return json.loads(res.stdout)

    def read(self, path: str) -> str:
        return self._call("read", path=path)

    def write(self, path: str, text: str) -> None:
        data = base64.b64encode(text.encode()).decode()
        for at in range(0, max(1, len(data)), 8000):
            self._call("chunk", path=path, data=data[at : at + 8000], append=at > 0)

    def remove(self, path: str) -> None:
        self._call("remove", path=path)

    def list(self) -> list[str]:
        return self._call("list")

    def tree(self) -> dict:
        return self._call("tree")

    def restore(self, tree: dict) -> None:
        current = self.tree()
        for rel in current.keys() - tree.keys():
            self.remove(rel)
        for rel, entry in tree.items():
            old = current.get(rel)
            if old == entry:
                continue
            if old is None or old["data"] != entry["data"]:
                data = entry["data"]
                for at in range(0, max(1, len(data)), 8000):
                    self._call("chunk", path=rel, data=data[at : at + 8000], append=at > 0)
            if old is None or old["mode"] != entry["mode"]:
                self._call("mode", path=rel, mode=entry["mode"])
