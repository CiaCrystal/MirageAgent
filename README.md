# MirageAgent

MirageAgent 是一个由 LLM 驱动的 Python 库迁移 Agent。它在受约束的执行环境里读取报错、定位旧 API、修改源码并运行项目自带的测试。

## 特性

- Docker 沙箱、测试只读保护、命令策略、检查点与回滚，约束改动范围。
- 三路 RAG：官方迁移知识、代码库 AST 与调用关系、审核通过的经验记忆。
- 目标包可携带只读经验快照；缺少快照时可继续运行，不注入经验。
- skills 按主题提供 API 对照，支持目录常驻、正文按需加载。
- 动态规划结合静态用法、运行时警告、失败分类与重规划，支持诊断子 Agent 和多次尝试。
- 声明式目标包将规则、提示、知识源和迁移技能放在一起。

## 安装

需要 Python 3.11+、uv，以及启用 Linux 容器的 Docker。

```sh
uv sync --extra dev --extra sandbox --extra rag
uv run migration-agent --help
```

基础安装可使用 `uv sync`。`sandbox` 提供 Docker 支持，`rag` 提供向量检索和 AST 解析，`dev` 提供代码检查工具；`cache` 提供可选 Redis 响应缓存。

## 配置

将 `.env.example` 复制为 `.env`，填写 `ANTHROPIC_API_KEY`。使用兼容 Anthropic 协议的端点时设置 `ANTHROPIC_BASE_URL`，并用 `MIGRATION_AGENT_MODEL` 指定模型。

CLI 默认使用 `full` 模式，经验记忆、按需 skills 和动态规划默认开启。运行前请安装 `sandbox`、`rag` 依赖并为当前目标建立知识索引。

设置使用 `MIGRATION_AGENT_` 前缀。以下为默认值，可在 `.env` 中覆盖：

```dotenv
MIGRATION_AGENT_MEMORY_ENABLED=true
MIGRATION_AGENT_SKILLS_EXTERNAL=true
MIGRATION_AGENT_DYNAMIC_PLAN=true
MIGRATION_AGENT_MAX_STEPS=40
```

将 `MIGRATION_AGENT_MEMORY_ENABLED=false`、`MIGRATION_AGENT_SKILLS_EXTERNAL=false` 或 `MIGRATION_AGENT_DYNAMIC_PLAN=false` 写入 `.env` 可分别关闭对应能力。动态规划在 `full` 模式和默认的 `hooks` 扩展方式下生效。

`memory_dir` 未指定时从当前目标包的 `memory/` 读取；没有该目录时不使用经验。可用 `--memory-dir` 指定其他快照。候选经验可用 `--extract-candidates` 写入独立目录，随包经验保持只读。

`migrate --config settings.yaml` 可载入 `settings` 覆盖项和 `pricing`。价格表需提供 `currency`、`input_per_million`、`cache_read_per_million`、`cache_write_per_million`、`output_per_million` 与 `source`，供成本预算和经验总结使用；未提供价格表时按 token 预算运行。

## 构建沙箱

```sh
docker build -f docker/sandbox.Dockerfile -t migration-agent-sandbox:latest .
```

沙箱预装 Pydantic v2 与 pytest。其他目标需要在派生镜像中安装对应目标依赖，并用 `--image` 指定该镜像。

## 建立知识库

```sh
uv run migration-agent index --target pydantic-v2
uv run migration-agent kb-search "validator migration"
```

索引从目标包配置的固定上游版本获取指南、API 文档、版本说明与相关 issues，写入 `data/`。首次运行会下载嵌入模型；可选 `--embedder hash` 使用确定性轻量嵌入。GitHub API 可使用 `.env` 中的 `GITHUB_TOKEN`。`scripts/build_knowledge_index.py` 提供同一索引入口。

## 快速上手

先复制示例，避免直接改动随项目提供的源码：

```sh
# macOS / Linux
mkdir -p data
cp -R examples/pydantic_toy data/pydantic_run
# PowerShell: New-Item -ItemType Directory -Force data; Copy-Item -Recurse examples/pydantic_toy data/pydantic_run
uv run migration-agent migrate data/pydantic_run --target pydantic-v2
```

复制前请先创建 `data/` 目录。默认的 `full` 模式包含检索、规划与策略控制，需要先建立知识库。可显式传入 `--profile baseline` 使用文件、命令和测试工具；该模式不启用三路 RAG 和动态规划，skills 仍由独立开关控制。迁移会改动传入仓库的源码，保持测试只读。

其他示例是 `examples/sqlalchemy_toy/` 与 `examples/marshmallow_toy/`，分别使用目标名 `sqlalchemy-v2` 与 `marshmallow-v4`。本地执行可用 `--sandbox local --python <目标环境的 Python>`，需自行准备依赖环境。

运行记录写入 `traces/`：

```sh
uv run migration-agent trace show traces/run.jsonl
uv run migration-agent trace summary traces/run.jsonl
```

## 新增迁移目标

1. 在 `src/migration_agent/migration/targets/` 新建目录，参考现有目标编写 `target.yaml`：名称、版本、迁移速查、旧 API 检测、保护规则、分类与记忆词表。
2. 添加 `knowledge.yaml` 描述固定上游版本和知识来源。
3. 在 `skills/` 添加带 `description` front matter 的 Markdown 技能和可选 `INDEX.md`；审核后的经验可放入 `memory/`。
4. 必要时提供 `hooks.py`，仅实现支持的 `runtime_probe`、`quality_issues`、`classify` 钩子。
5. 用新目标名建立索引并运行迁移。注册表自动发现目标，内置目标中恰好一个应声明 `default: true`。

也可以用 `MIGRATION_AGENT_TARGET_PATHS` 指定额外目标目录。

## 目录结构

```text
src/migration_agent/
  agent/       模型、循环、上下文、角色路由、skills 与编排
  tools/       文件、命令、测试、检索、策略与沙箱
  rag/         knowledge、codebase、memory 与公共检索组件
  migration/   任务、规划、验证、经验抽取与 targets/
  trace/       运行事件与分析
  cli.py       命令入口
  config.py    配置
examples/      三个待迁移的小型仓库，含各自的行为测试
docker/        沙箱镜像定义
scripts/       知识索引入口与缓存管理
skills/        技能组织说明
```

## 许可

本仓库暂未附带开源许可证，保留所有权利。如需使用或转载，请先联系作者。
