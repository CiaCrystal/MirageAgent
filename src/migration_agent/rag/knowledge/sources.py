"""仓库、固定 ref、页面列表、issue 查询、反泄漏名单全部来自目标包的 knowledge.yaml(`KnowledgeSpec`,
rag/knowledge/source_spec.py);本文件不含任何库相关常量。缺省取当前目标(`migration.targets.active()`)。

文档类来源一律读文档仓库在**固定 commit** 下的原文(raw.githubusercontent.com),而不是渲染后的网页:
标题层级、锚点与代码块都完整保留,且可复现。rst 原文(raw_format: rst)先转成 Markdown 标题层级(rst.py)。

GitHub issues 走 Search API(响应自带正文):取 spec.issue_queries 的并集。
有 GITHUB_TOKEN 时为每条 issue 拉取评论;匿名时(60 次/小时)只为评论最多的前若干条拉取,数量记入来源清单。

所有原始响应缓存到 `raw_dir`,重复构建不重复下载;`refresh=True` 强制重抓。"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_MAINTAINER = {"MEMBER", "OWNER", "COLLABORATOR"}


@dataclass
class RawSource:
    kind: str
    name: str  # 文档标题 / "Issue #123"
    url: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def _default_spec() -> Any:
    from ...migration.targets import active

    spec = active().knowledge_spec
    if spec is None:
        raise ValueError(f"migration target {active().name!r} declares no knowledge.yaml")
    return spec


class Fetcher:
    def __init__(
        self,
        raw_dir: Path,
        *,
        ref: str | None = None,
        refresh: bool = False,
        token: str | None = None,
        spec: Any = None,
    ):
        self.spec = spec or _default_spec()
        self.raw_dir, self.ref, self.refresh = raw_dir, ref or self.spec.ref, refresh
        self.token = token if token is not None else (os.environ.get("GITHUB_TOKEN") or "").strip()
        self.http = httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "migration-agent"})
        self.api_calls = 0

    # ---- 文档 ----
    def raw_file(self, path: str) -> str:
        cache = self.raw_dir / (self.spec.cache_dir or self.spec.target) / self.ref / path
        if cache.exists() and not self.refresh:
            text = cache.read_text(encoding="utf-8")
        else:
            resp = self.http.get(RAW.format(repo=self.spec.repo, ref=self.ref, path=path))
            resp.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(resp.text, encoding="utf-8")
            text = resp.text
        if self.spec.raw_format == "rst" and path.endswith((".rst", ".txt")):
            from .rst import rst_to_markdown

            return rst_to_markdown(text)
        return text

    def fetch_migration_guide(self) -> RawSource:
        g = self.spec.guide
        return RawSource("guide", g.title, self.spec.site + g.site, self.raw_file(g.path), {"ref": self.ref})

    def fetch_release_notes(self) -> RawSource | None:
        "Runtime support."
        c = self.spec.release_notes
        if c is None:
            return None
        text = self.raw_file(c.path)
        keep, on = [], False
        for line in text.splitlines():
            m = re.match(c.keep_heading, line)
            if m:
                on = m.group(1) == c.keep_major
            if on:
                keep.append(line)
        url = c.url.format(repo=self.spec.repo, ref=self.ref, path=c.path)
        meta = {"ref": self.ref}
        if c.release_url:
            meta["release_url"] = c.release_url
        if c.release_heading != type(c).model_fields["release_heading"].default:
            meta["release_heading"] = c.release_heading
        return RawSource("release_notes", "Release notes", url, "\n".join(keep), meta)

    def fetch_docs(self, pages: Iterable[tuple[str, str, str, str]] | None = None) -> list[RawSource]:
        out = []
        for kind, path, site, title in pages if pages is not None else self.spec.doc_pages:
            text = self.raw_file(path)
            out.append(RawSource(kind, title, self.spec.site + site, text, {"ref": self.ref, "path": path}))
        return out

    # ---- issues ----
    def _api(self, url: str, params: dict | None = None) -> Any:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for _ in range(6):
            resp = self.http.get(url, params=params, headers=headers)
            self.api_calls += 1
            if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
                wait = int(resp.headers.get("x-ratelimit-reset", time.time() + 60)) - time.time()
                if wait > 900:
                    raise RuntimeError(f"GitHub rate limit; resets in {wait:.0f}s")
                time.sleep(max(wait, 1) + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"GitHub API kept rate-limiting: {url}")

    def fetch_issues(
        self, exclude_repos: Iterable[str] = (), *, comment_budget: int | None = None
    ) -> list[RawSource]:
        cache = self.raw_dir / (
            self.spec.issues_cache or f"{self.spec.cache_dir or self.spec.target}/issues.json"
        )
        if cache.exists() and not self.refresh:
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = self._download_issues(comment_budget)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        excluded = [r.lower() for r in [*exclude_repos, *self.spec.exclude_repos]]
        out = []
        noise = self.spec.issue_noise_patterns
        for issue in data["issues"]:
            blob = f"{issue['title']}\n{issue['body']}\n" + "\n".join(issue["comments"])
            if any(r in blob.lower() for r in excluded):
                continue
            out.append(
                RawSource(
                    "issue",
                    f"Issue #{issue['number']}: {issue['title']}",
                    issue["url"],
                    issue["body"],
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "labels": ",".join(issue["labels"]),
                        "state": issue["state"],
                        "created_at": issue["created_at"],
                        "comments": issue["comments"],
                        "matched_by": issue["matched_by"],
                        **({"noise_patterns": list(noise)} if noise else {}),
                    },
                )
            )
        return out

    def _download_issues(self, comment_budget: int | None) -> dict:
        found: dict[int, dict] = {}
        title_re = re.compile(self.spec.issue_title_regex, re.I)
        for q in self.spec.issue_queries:
            for page in range(1, 11):
                data = self._api(
                    "https://api.github.com/search/issues",
                    {"q": q, "per_page": 100, "page": page, "sort": "comments", "order": "desc"},
                )
                for item in data.get("items", []):
                    if "pull_request" in item:
                        continue
                    labels = [lb["name"] for lb in item.get("labels", [])]
                    title_hit = bool(title_re.search(item["title"]))
                    row = found.setdefault(
                        item["number"],
                        {
                            "number": item["number"],
                            "title": item["title"],
                            "body": item.get("body") or "",
                            "labels": labels,
                            "state": item["state"],
                            "created_at": item["created_at"],
                            "url": item["html_url"],
                            "n_comments": item.get("comments", 0),
                            "comments": [],
                            "matched_by": [],
                        },
                    )
                    tag = "title" if title_hit and "in:title" in q else q.split("is:issue ", 1)[1]
                    if tag not in row["matched_by"]:
                        row["matched_by"].append(tag)
                if len(data.get("items", [])) < 100:
                    break
                if not self.token:
                    time.sleep(6.5)  # 匿名搜索限流 10 次/分钟
        issues = sorted(found.values(), key=lambda r: -r["n_comments"])
        if comment_budget is None:
            if self.token:
                comment_budget = len(issues)
            else:
                rate = self._api("https://api.github.com/rate_limit")["resources"]["core"]["remaining"]
                comment_budget = max(0, rate - 3)
        fetched = 0
        for issue in issues:
            if fetched >= comment_budget or issue["n_comments"] == 0:
                continue
            owner_repo = issue["url"].split("github.com/")[1].split("/issues/")[0]
            comments = self._api(
                f"https://api.github.com/repos/{owner_repo}/issues/{issue['number']}/comments",
                {"per_page": 30},
            )
            fetched += 1
            issue["comments"] = [
                {"author_association": c.get("author_association", ""), "body": c.get("body") or ""}
                for c in comments
            ]
        for issue in issues:
            issue["comments"] = _pick_answers(issue["comments"])
            issue["matched_by"] = ",".join(issue["matched_by"])
        return {
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "queries": list(self.spec.issue_queries),
            "authenticated": bool(self.token),
            "issues_with_comments_fetched": fetched,
            "issues": issues,
        }

    def fetch_all(self, exclude_repos: Iterable[str] = ()) -> tuple[list[RawSource], dict]:
        release_notes = self.fetch_release_notes()
        sources = [
            self.fetch_migration_guide(),
            *([release_notes] if release_notes else []),
            *self.fetch_docs(),
        ]
        issues = self.fetch_issues(exclude_repos)
        issues_path = self.raw_dir / (
            self.spec.issues_cache or f"{self.spec.cache_dir or self.spec.target}/issues.json"
        )
        issue_meta = json.loads(issues_path.read_text(encoding="utf-8"))
        manifest = {
            self.spec.manifest_ref_key: self.ref,
            "documents": [asdict(s) | {"text": None, "chars": len(s.text)} for s in sources],
            "issues": {
                "fetched_at": issue_meta["fetched_at"],
                "queries": issue_meta["queries"],
                "authenticated": issue_meta["authenticated"],
                "downloaded": len(issue_meta["issues"]),
                "excluded_repo_mentions": len(issue_meta["issues"]) - len(issues),
                "indexed": len(issues),
                "with_comment_answers": sum(1 for i in issues if i.meta["comments"]),
                "issues_with_comments_fetched": issue_meta["issues_with_comments_fetched"],
            },
        }
        for doc in manifest["documents"]:
            doc.pop("text")
        return [*sources, *issues], manifest


def _pick_answers(comments: list[dict]) -> list[str]:
    """优先维护者回复(近似「被采纳的回答」),否则取第一条;最多 2 条。"""
    maint = [c["body"] for c in comments if c.get("author_association") in _MAINTAINER and c["body"].strip()]
    if maint:
        return maint[:2]
    return [c["body"] for c in comments[:1] if c["body"].strip()]
