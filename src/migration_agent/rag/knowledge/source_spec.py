"""KnowledgeSpec:由迁移目标声明知识源,sources.py 不再写死任何仓库或页面。

目标包的 `knowledge.yaml`:

    repo: owner/name                 文档仓库
    ref: <commit>                    固定 ref,保证索引可复现
    site: https://.../               文档站根 URL(片段的可点击链接)
    raw_format: markdown | rst       rst 先转成 Markdown 标题层级再切块
    guide: {path, site, title}       官方迁移指南
    doc_pages: [[kind, path, site, title], ...]
    issue_queries: [GitHub Search 查询, ...]   (并集)
    issue_title_regex: 标题命中时 matched_by 记为 title
    issue_noise_patterns: [正则, ...]          issue 模板里的噪声段
    cache_dir: 原始响应缓存的子目录名

`KnowledgeSpec.load(path)` 校验未知字段;索引按目标分目录(`data/knowledge/<target>/index`;index_subdir 可沿用旧路径)。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuideSpec(_Strict):
    path: str
    site: str
    title: str = "Migration Guide"


class ReleaseNotesSpec(_Strict):
    path: str
    url: str  # 可含 {repo} {ref} {path}
    release_url: str = ""  # 可含 {version};空 = 用 url
    keep_heading: str  # 只保留匹配该正则的发布小节(捕获组 1 与 keep_major 比较)
    keep_major: str
    release_heading: str = r"## (v[\w.]+)\s*(?:\(([\d-]+)\))?"  # 版本标题:组 1 版本、组 2 日期


class KnowledgeSpec(_Strict):
    target: str = ""
    repo: str
    ref: str
    site: str
    raw_format: str = "markdown"
    guide: GuideSpec
    release_notes: ReleaseNotesSpec | None = None
    doc_pages: list[tuple[str, str, str, str]] = []
    issue_queries: list[str] = []
    issue_title_regex: str = r"\bv\s?[12]\b|v[12]\.\d|migrat|upgrad"
    issue_noise_patterns: list[str] = []
    exclude_repos: list[str] = []
    cache_dir: str = ""
    issues_cache: str = ""  # issue 响应缓存(相对 raw_dir);空 = <cache_dir>/issues.json
    index_subdir: str = ""  # 空 = data/knowledge/<target>/index
    manifest_ref_key: str = "ref"

    @classmethod
    def load(cls, path: Path, *, target: str = "") -> KnowledgeSpec:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("raw_format", "markdown") not in ("markdown", "rst"):
            raise ValueError("raw_format must be markdown or rst")
        return cls.model_validate({"target": target, **data})

    def raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.ref}/{path}"
