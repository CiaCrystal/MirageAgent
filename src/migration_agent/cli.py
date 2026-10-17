"""命令行入口:

    migration-agent migrate <repo> [--target <name>] [--max-steps N] [--profile baseline|full] [--python PATH]
    migration-agent trace show <trace.jsonl>
    migration-agent trace summary <trace.jsonl>
    migration-agent index [--refresh]  /  kb-search <query>   (知识库)

只做参数解析与装配,不含业务逻辑。
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import typer
from rich.console import Console

from .config import load_settings
from .migration.task import MigrationTask

app = typer.Typer(no_args_is_help=True, add_completion=False, help="LLM + RAG library migration agent")
trace_app = typer.Typer(no_args_is_help=True, help="Inspect trace files")
app.add_typer(trace_app, name="trace")
console = Console()


def _run_session(task, settings, profile, python, pricing, first_loop):
    """dynamic_plan 开启时经 orchestrator.Session 运行(最多 orchestrator_attempts 次尝试,共用一个沙箱)。"""
    from .agent.assemble import assemble
    from .agent.orchestrator import Session

    first_loop.tracer.close()  # 预先装配的 loop 只用于打印信息;每次尝试重新装配
    executor = None
    if not python and settings.sandbox_backend == "docker":
        from .tools.sandbox.docker_sandbox import DockerSandbox

        executor = DockerSandbox(task.repo_path, image=settings.sandbox_image)
        executor.start()

    def build(t, attempt=1):
        return assemble(
            t, settings, tool_profile=profile, python_executable=python, pricing=pricing, executor=executor
        )

    try:
        out = Session(task, settings, build=build, executor=executor).run()
    finally:
        if executor is not None:
            executor.stop()
    for a in out.attempts:
        console.print(f"attempt {a.index}: {a.result.reason}  steps {a.result.steps}  restored {a.restored}")
    return out.combined


@app.command()
def migrate(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to the repository to migrate"),
    target: str | None = typer.Option(None, help="Migration target (default: the default target package)"),
    max_steps: int | None = typer.Option(None, help="Override max agent steps"),
    profile: str = typer.Option("baseline", help="Tool profile: baseline | full"),
    python: str | None = typer.Option(
        None, "--python", help="Python executable used to run the repo's tests"
    ),
    test_cmd: str | None = typer.Option(None, help="Override test command, e.g. 'python -m pytest -q tests'"),
    run_id: str | None = typer.Option(None, help="Trace run id (default: timestamp)"),
    sandbox: str = typer.Option("docker", help="Execution backend: docker | local (development only)"),
    image: str | None = typer.Option(None, help="Docker sandbox image"),
    allow_compatibility_shim: bool = typer.Option(
        False, help="Conservative mode: allow compatibility-layer imports"
    ),
    memory_dir: Path | None = typer.Option(
        None, help="Enable experience memory (full profile) from this frozen snapshot or store"
    ),
    allow_memory_store: bool = typer.Option(
        False, help="Accept a non-frozen store for --memory-dir (development only)"
    ),
    extract_candidates: Path | None = typer.Option(
        None,
        help="After the run, extract experience candidates into this store (never published automatically)",
    ),
    config: Path | None = typer.Option(None, help="YAML with `settings` overrides and `pricing`"),
) -> None:
    """Migrate REPO to TARGET until its test suite passes."""
    from .agent.assemble import assemble
    from .tools.pytest_tool import cleanup_artifacts

    settings = load_settings()  # 导入 config 时已 load_dotenv()
    settings = settings.model_copy(
        update={
            "sandbox_backend": sandbox,
            "allow_compatibility_shim": allow_compatibility_shim,
            **({"sandbox_image": image} if image else {}),
            **(
                {
                    "memory_enabled": True,
                    "memory_dir": memory_dir,
                    "memory_require_snapshot": not allow_memory_store,
                }
                if memory_dir
                else {}
            ),
            **(
                {"memory_extract_candidates": True, "memory_candidate_dir": extract_candidates}
                if extract_candidates
                else {}
            ),
        }
    )
    pricing = None
    if config:
        import yaml

        from .agent.pricing import Pricing

        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        overrides = {k: v for k, v in data.get("settings", {}).items() if k != "sandbox_backend"}
        settings = settings.model_copy(update=overrides)
        pricing = Pricing.model_validate(data["pricing"]) if "pricing" in data else None
    if settings.memory_enabled and profile != "full":
        raise typer.BadParameter("--memory-dir requires --profile full")
    if sandbox not in ("docker", "local"):
        raise typer.BadParameter("sandbox must be docker or local")
    if python and sandbox == "docker":
        raise typer.BadParameter("--python is for --sandbox local; Docker uses its own Python")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set[/] (export it or put it in .env)")
        raise typer.Exit(code=2)
    task = MigrationTask(
        repo_path=repo, **({"target": target} if target else {}), max_steps=max_steps, run_id=run_id
    )
    if test_cmd:
        task.test_cmd = shlex.split(test_cmd)
    loop, ctx = assemble(task, settings, tool_profile=profile, python_executable=python, pricing=pricing)
    recorder = None
    if settings.memory_extract_candidates:
        from .migration.evidence import EvidenceRecorder

        recorder = EvidenceRecorder(ctx, loop.tracer).attach(loop.test_tool)
    console.print(
        f"[bold]run_id[/] {loop.tracer.run_id}  [bold]model[/] {loop.llm.model}  "
        f"[bold]endpoint[/] {os.environ.get('ANTHROPIC_BASE_URL', 'api.anthropic.com')}  "
        f"[bold]tools[/] {', '.join(loop.registry.names())}"
    )

    try:
        if (
            profile == "full"
            and settings.extension_mode == "hooks"
            and settings.dynamic_plan
            and settings.orchestrator_attempts > 1
        ) and recorder is None:
            result = _run_session(task, settings, profile, python, pricing, loop)
        else:
            result = loop.run(task, ctx)
    finally:
        loop.tracer.close()
        cleanup_artifacts(ctx.repo_path)
    if recorder is not None:
        _extract_after_run(settings, recorder, result, ctx, loop=loop, pricing=pricing)

    color = "green" if result.success else "red"
    console.print(f"\n[{color} bold]{'SUCCESS' if result.success else 'FAILED'}[/] ({result.reason})")
    console.print(
        f"tests {result.tests_passed}/{result.tests_total}  steps {result.steps}  "
        f"tokens in={result.tokens_in} cached={result.tokens_cached} out={result.tokens_out}  {result.duration_s:.1f}s"
    )
    console.print(
        f"violations {result.violations}  rollbacks {result.rollbacks}  "
        f"deprecated API candidates {result.deprecated_api_count}  compatibility imports {result.compatibility_import_count}"
    )
    if settings.llm_cache_url:  # 命中不计费;账本只记真实发出的请求
        from .agent.llm_cache import find_cache

        cache = find_cache(loop.llm)
        if cache is not None:
            s = cache.stats.as_dict()
            console.print(
                f"llm cache [{cache.namespace}] hits {s['hits']} misses {s['misses']} "
                f"saved in={s['saved_tokens_in']} out={s['saved_tokens_out']} ~USD {s['saved_usd']:.4f}"
            )
    if result.files_changed:
        console.print("changed: " + ", ".join(result.files_changed))
    if result.message:
        console.print(f"\n{result.message}")
    console.print(f"\ntrace: {result.trace_path}")
    raise typer.Exit(code=0 if result.success else 1)


def _extract_after_run(settings, recorder, result, ctx, *, loop=None, pricing=None) -> None:
    "Runtime support."
    from .migration.evidence import read_texts
    from .migration.learning import dev_source, extract_into, run_facts
    from .rag.memory.store import ExperienceStore

    store_dir = Path(settings.memory_candidate_dir)
    evidence_path = store_dir / "evidence" / f"{result.run_id}.json"
    evidence = recorder.save(
        evidence_path,
        run_id=result.run_id,
        meta={"repo": str(ctx.repo_path)},
        final=read_texts(ctx.repo_path, recorder.changed_paths()),
    )
    facts = run_facts(result.trace_path, result)
    source = dev_source(ctx.repo_path, result.run_id, str(result.trace_path), str(evidence_path))
    stats = extract_into(ExperienceStore(store_dir), evidence, facts, source, is_protected=ctx.is_protected)
    console.print(
        f"memory candidates: {stats['windows']} window(s), {stats['auto_passed']} passed automatic checks, "
        f"{stats['uncertain']} uncertain, {stats['rejected']} rejected -> {store_dir / 'candidates'} (pending review)"
    )
    if settings.memory_llm_summary and loop is not None:
        _summarize_after_run(settings, loop, result, ctx, evidence, facts, source, store_dir, pricing)


def _summarize_after_run(settings, loop, result, ctx, evidence, facts, source, store_dir, pricing) -> None:
    """规则窗口 → summarizer → critic → consolidate(router 的 summarizer / critic 角色,与 executor 同一密钥与客户端)。
    花费 ≤ min(run 成本 × 15%, USD 0.02);没有价格表时不知道 run 成本,不调用模型。结果写回同一 trace(memory_summary)。"""
    import json

    from .agent.router import ModelRouter
    from .migration.learning import (
        LearningSourcePolicy,
        router_channel,
        run_cost,
        summarize_run,
        trace_events,
    )
    from .migration.replanner import forbidden_reason
    from .rag.memory.store import ExperienceStore
    from .trace.tracer import Tracer

    router = getattr(loop, "router", None) or ModelRouter.from_settings(
        settings, executor=loop.llm, pricing=pricing
    )
    policy = LearningSourcePolicy()
    events = trace_events(result.trace_path)
    stats = summarize_run(
        ExperienceStore(store_dir),
        events=events,
        source=source,
        policy=policy,
        forbidden=forbidden_reason,
        summarizer_channel=router_channel(router, "summarizer"),
        critic_channel=router_channel(router, "critic"),
        run_cost_usd=run_cost(events, pricing),
        price_in=pricing.input_per_million if pricing else 0.0,
        price_out=pricing.output_per_million if pricing else 0.0,
        evidence=evidence,
        facts=facts,
        is_protected=ctx.is_protected,
        online=settings.memory_online,
        thresholds=settings.memory_critic_thresholds,
    )
    tracer = Tracer(result.trace_path, run_id=result.run_id)
    try:
        # 角色账目快照(累计值,含本 run 之前的 summarizer 摘要压缩等);附加调用的增量见 stats["calls"]
        tracer.log(
            "memory_summary", {**stats, "router_after": {r: vars(u) for r, u in router.usage().items()}}
        )
    finally:
        tracer.close()
    (store_dir / "evidence" / f"{result.run_id}.summary.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    b = stats["budget"]
    console.print(
        f"memory summary: {len(stats['results'])} window(s) {stats['ops']}; spent USD {b['spent_usd']:.4f} "
        f"(cap {b['cap_usd']:.4f}){' — ' + stats['restricted'] if stats['restricted'] else ''}"
    )


@trace_app.command("show")
def trace_show(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Print the step-by-step timeline of a trace."""
    from .trace.analyze import load, render_timeline

    console.print(render_timeline(load(path)), highlight=False)


@trace_app.command("summary")
def trace_summary(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Print aggregate statistics of a trace."""
    from .trace.analyze import load, summarize

    s = summarize(load(path))
    console.print_json(s.model_dump_json())


@app.command()
def index(
    refresh: bool = typer.Option(False, help="Re-download sources instead of using the raw cache"),
    embedder: str | None = typer.Option(
        None, help="Embedding provider spec, e.g. st:BAAI/bge-small-en-v1.5 or hash"
    ),
    target: str | None = typer.Option(None, help="Migration target (default: the default target package)"),
) -> None:
    """Build (or incrementally update) the migration knowledge index (per target)."""
    from .migration.targets import set_active
    from .rag.knowledge.indexer import build_knowledge_index

    set_active(target)
    settings = load_settings()
    stats = build_knowledge_index(
        settings.knowledge_dir,
        settings.data_dir / "knowledge" / "raw",
        embedder or settings.embedding_model,
        refresh=refresh,
    )
    console.print(
        f"index {settings.knowledge_dir}: {stats.docs} chunks {stats.by_kind}; embedded {stats.embedded}, "
        f"deleted {stats.deleted}; corpus {stats.corpus_hash}; {stats.seconds:.1f}s"
    )


@app.command("kb-search")
def kb_search(
    query: str = typer.Argument(...),
    k: int = typer.Option(5),
    mode: str = typer.Option("hybrid", help="hybrid | bm25 | vector"),
) -> None:
    """Query the knowledge index from the shell (debugging aid)."""
    from .rag.knowledge.retriever import KnowledgeRetriever, normalize_query

    retriever = KnowledgeRetriever.load(load_settings().knowledge_dir)
    console.print(f"[dim]normalized: {normalize_query(query)}[/]")
    for h in retriever.retrieve(query, k=k, mode=mode):
        console.print(f"[bold]{h.rank}[/] {h.score:.4f} {h.ranks} [cyan]{h.doc.metadata['heading']}[/]")


if __name__ == "__main__":
    app()
