"""系统提示与初始用户消息模板。

接口:
    build_system_prompt(target: MigrationTarget, extra_context: str = "") -> str
    build_initial_message(repo_tree: str, initial_test_report: str, extra_context: str = "") -> str

`extra_context` 是 Planner / RAG 预检索结果的注入点;baseline 为空。
硬规则写在系统提示里只是第一层约束,真正的闸门是 fs 工具的只读保护与 Validator。
"""

from __future__ import annotations

from ..migration.targets.base import MigrationTarget

SYSTEM_TEMPLATE = """\
You are an expert Python engineer performing a library migration on an existing repository: {target_desc}.

Success criterion: the repository's existing test suite passes unchanged under the new library version.

Working rules:
1. Test files are READ-ONLY. Never edit, delete, skip, xfail, or add tests. If a test fails, fix the source code
   so the test's expectation is met. If a test's expectation is genuinely impossible under the new version, say so
   explicitly and stop — do not work around it.
2. Do not downgrade or pin the library back to the old version, and do not use compatibility shims.
3. Work incrementally: read the failing traceback, locate the code, make a focused edit, then run the tests
   (the test suite also runs automatically after every edit). Prefer `edit_file` over rewriting whole files.
4. Keep behaviour identical except for what the migration requires. Do not refactor unrelated code.
5. When the same error persists after two attempts, re-read the relevant code and the traceback carefully
   instead of retrying the same change.
6. When all tests pass, reply with a short summary of what you changed and stop calling tools.

{target_hints}
{extra_context}"""

INITIAL_TEMPLATE = """\
Repository file tree (test files marked read-only):
{repo_tree}

Initial test run under the NEW library version:
{initial_test_report}
{extra_context}
Begin the migration. Start by reading the files implicated in the failures."""


def build_system_prompt(target: MigrationTarget, extra_context: str = "") -> str:
    return SYSTEM_TEMPLATE.format(
        target_desc=target.describe(),
        target_hints=target.prompt_hints(),
        extra_context=f"\n{extra_context}\n" if extra_context else "",
    )


def build_initial_message(repo_tree: str, initial_test_report: str, extra_context: str = "") -> str:
    return INITIAL_TEMPLATE.format(
        repo_tree=repo_tree,
        initial_test_report=initial_test_report,
        extra_context=f"\n{extra_context}\n" if extra_context else "",
    )


NUDGE_TESTS_STILL_FAILING = (
    "The test suite is still not passing:\n{report}\n\nContinue the migration. "
    "If you believe it cannot be completed, say exactly why."
)
