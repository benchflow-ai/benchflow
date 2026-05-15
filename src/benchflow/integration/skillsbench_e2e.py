"""File-driven SkillsBench E2E matrix runner used by ``bench eval create -f``."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from benchflow.agents.registry import AGENTS
from benchflow.integration.artifact_audit import write_artifact_audit
from benchflow.integration.audit_agent import write_audit_outputs
from benchflow.integration.parity import write_parity_report
from benchflow.models import RunResult
from benchflow.sdk import SDK

logger = logging.getLogger(__name__)

KIND = "skillsbench-e2e"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_TASKS = [
    "jax-computing-basics",
    "python-scala-translation",
    "jpg-ocr-stat",
    "grid-dispatch-operator",
    "threejs-to-obj",
    "data-to-d3",
    "lake-warming-attribution",
    "weighted-gdp-calc",
    "shock-analysis-supply",
]


@dataclass(frozen=True)
class MatrixEntry:
    """One E2E matrix cell."""

    agent: str
    task_name: str
    model: str
    environment: str

    @property
    def id(self) -> str:
        return f"{self.agent}/{self.task_name}"


@dataclass(frozen=True)
class E2EConfig:
    """Parsed ``tasks/skillsbench-e2e/e2e.yaml``."""

    path: Path
    source_repo: str
    source_path: str
    source_ref: str | None
    tasks_manifest: Path
    jobs_dir: Path
    model: str
    environment: str
    concurrency: int
    agents: list[str]
    max_retries: int
    resume: bool
    skills_dir: str | None
    baseline_repo: str | None
    baseline_ref: str | None
    audit_prompt: Path | None
    audit_agent_enabled: bool
    audit_agent: str
    audit_model: str
    audit_environment: str

    @property
    def config_dir(self) -> Path:
        return self.path.parent


def is_skillsbench_e2e_config(path: str | Path) -> bool:
    """Return True when *path* is a SkillsBench E2E config file."""
    try:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    except Exception:
        return False
    return raw.get("kind") == KIND


def _resolve_config_path(config_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else config_path.parent / p


def load_config(path: str | Path) -> E2EConfig:
    """Parse an E2E YAML config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get("kind") != KIND:
        raise ValueError(f"Not a {KIND!r} config: {path}")

    source = raw.get("source") or {}
    if not source.get("repo"):
        raise ValueError("skillsbench-e2e config requires source.repo")
    if not source.get("path"):
        raise ValueError("skillsbench-e2e config requires source.path")

    agents_raw = raw.get("agents", "all")
    agents = registered_matrix_agents() if agents_raw == "all" else list(agents_raw)

    audit = raw.get("audit") or {}
    audit_agent = audit.get("audit_agent") or {}
    audit_prompt = _resolve_config_path(path, audit_agent.get("prompt"))

    baseline = raw.get("baseline") or {}
    manifest = _resolve_config_path(path, raw.get("tasks_manifest"))
    if manifest is None:
        raise ValueError("skillsbench-e2e config requires tasks_manifest")

    return E2EConfig(
        path=path,
        source_repo=source["repo"],
        source_path=source["path"],
        source_ref=source.get("ref"),
        tasks_manifest=manifest,
        jobs_dir=Path(raw.get("jobs_dir", "jobs/skillsbench-e2e")),
        model=raw.get("model") or DEFAULT_MODEL,
        environment=raw.get("environment", "daytona"),
        concurrency=int(raw.get("concurrency", 30)),
        agents=agents,
        max_retries=int(raw.get("max_retries", 0)),
        resume=bool(raw.get("resume", True)),
        skills_dir=raw.get("skills_dir"),
        baseline_repo=baseline.get("repo"),
        baseline_ref=baseline.get("ref"),
        audit_prompt=audit_prompt,
        audit_agent_enabled=bool(audit_agent.get("enabled", False)),
        audit_agent=audit_agent.get("agent", "gemini"),
        audit_model=audit_agent.get("model") or raw.get("model") or DEFAULT_MODEL,
        audit_environment=audit_agent.get("environment", raw.get("environment", "daytona")),
    )


def load_manifest(path: str | Path) -> list[str]:
    """Load task names from a manifest file."""
    tasks = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            tasks.append(stripped)
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"Duplicate task in manifest: {path}")
    return tasks


def registered_matrix_agents() -> list[str]:
    """Return all current built-in agents, preserving registry order."""
    return list(AGENTS)


def build_matrix(
    task_names: list[str],
    agents: list[str],
    model: str,
    environment: str = "daytona",
) -> list[MatrixEntry]:
    """Build the agent × task matrix."""
    unknown = [agent for agent in agents if agent not in AGENTS]
    if unknown:
        raise ValueError(f"Unknown E2E agents: {unknown}")
    return [
        MatrixEntry(agent=agent, task_name=task, model=model, environment=environment)
        for agent in agents
        for task in task_names
    ]


def materialize_subset(
    source_tasks_dir: str | Path,
    task_names: list[str],
    output_dir: str | Path,
) -> Path:
    """Symlink/copy the selected SkillsBench tasks into *output_dir*."""
    source_tasks_dir = Path(source_tasks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in task_names if not (source_tasks_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing SkillsBench E2E tasks in {source_tasks_dir}: {', '.join(missing)}"
        )

    for name in task_names:
        src = source_tasks_dir / name
        dst = output_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst, symlinks=True)
    return output_dir


def _run_id(*, dry_run: bool) -> str:
    prefix = "dry-run" if dry_run else "run"
    return f"{prefix}-{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"


def _matrix_config_dict(config: E2EConfig, tasks: list[str]) -> dict[str, Any]:
    return {
        "kind": KIND,
        "config_file": str(config.path),
        "source": {
            "repo": config.source_repo,
            "path": config.source_path,
            "ref": config.source_ref,
        },
        "tasks": tasks,
        "agents": config.agents,
        "model": config.model,
        "environment": config.environment,
        "concurrency": config.concurrency,
        "max_retries": config.max_retries,
        "resume": config.resume,
        "skills_dir": config.skills_dir,
        "audit_agent": {
            "enabled": config.audit_agent_enabled,
            "agent": config.audit_agent,
            "model": config.audit_model,
            "environment": config.audit_environment,
        },
    }


def _entry_to_summary(entry: MatrixEntry, status: str, **extra: Any) -> dict[str, Any]:
    row = {
        "id": entry.id,
        "agent": entry.agent,
        "task_name": entry.task_name,
        "model": entry.model,
        "environment": entry.environment,
        "status": status,
    }
    row.update(extra)
    return row


def _existing_result_path(run_dir: Path, entry: MatrixEntry) -> Path | None:
    pattern = f"trials/{entry.agent}/results/{entry.task_name}__*/result.json"
    matches = sorted(run_dir.glob(pattern))
    return matches[-1] if matches else None


async def _run_one_entry(
    run_dir: Path,
    tasks_dir: Path,
    entry: MatrixEntry,
    config: E2EConfig,
) -> dict[str, Any]:
    if config.resume and (existing := _existing_result_path(run_dir, entry)):
        return _entry_to_summary(
            entry,
            "resumed",
            result_path=str(existing),
            trial_dir=str(existing.parent),
        )

    sdk = SDK()
    jobs_dir = run_dir / "trials" / entry.agent
    last: RunResult | None = None
    start = time.time()

    for attempt in range(config.max_retries + 1):
        result = await sdk.run(
            task_path=tasks_dir / entry.task_name,
            agent=entry.agent,
            model=entry.model,
            jobs_dir=jobs_dir,
            job_name="results",
            environment=entry.environment,
            skills_dir=config.skills_dir,
        )
        last = result
        if result.rewards is not None or result.verifier_error or not result.error:
            break
        await asyncio.sleep(min(2**attempt, 30))

    assert last is not None
    result_path = jobs_dir / "results" / last.trial_name / "result.json"
    reward = last.rewards.get("reward") if last.rewards else None
    status = "completed"
    if last.error:
        status = "agent_error"
    elif last.verifier_error:
        status = "verifier_error"
    elif reward == 1.0:
        status = "passed"
    elif reward is not None:
        status = "failed"

    return _entry_to_summary(
        entry,
        status,
        reward=reward,
        error=last.error,
        verifier_error=last.verifier_error,
        n_tool_calls=last.n_tool_calls,
        trial_name=last.trial_name,
        trial_dir=str(result_path.parent),
        result_path=str(result_path),
        elapsed_sec=round(time.time() - start, 1),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _resolve_baseline(config: E2EConfig) -> tuple[Path | None, str | None]:
    """Resolve the optional historical baseline repo for parity comparison."""
    if not config.baseline_repo:
        return None, None
    try:
        from benchflow.task_download import resolve_source

        return resolve_source(config.baseline_repo, ref=config.baseline_ref), None
    except Exception as exc:
        logger.warning("Could not resolve E2E baseline repo: %s", exc)
        return None, str(exc)


async def run_from_config_file(
    config_path: str | Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Run or dry-run the SkillsBench E2E matrix from a YAML config file."""
    config = load_config(config_path)
    tasks = load_manifest(config.tasks_manifest)
    matrix = build_matrix(tasks, config.agents, config.model, config.environment)

    run_dir = config.jobs_dir / _run_id(dry_run=dry_run)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "matrix_config.json", _matrix_config_dict(config, tasks))

    if dry_run:
        entries = [_entry_to_summary(entry, "planned") for entry in matrix]
        _write_json(
            run_dir / "matrix_summary.json",
            {
                "run_dir": str(run_dir),
                "dry_run": True,
                "total": len(entries),
                "entries": entries,
            },
        )
        write_artifact_audit(run_dir)
        write_parity_report(run_dir)
        write_audit_outputs(run_dir, config.audit_prompt)
        return run_dir

    from benchflow.task_download import resolve_source

    source_tasks_dir = resolve_source(
        config.source_repo, config.source_path, config.source_ref
    )
    tasks_dir = materialize_subset(source_tasks_dir, tasks, run_dir / "_tasks")

    sem = asyncio.Semaphore(config.concurrency)

    async def bounded(entry: MatrixEntry) -> dict[str, Any]:
        async with sem:
            try:
                return await _run_one_entry(run_dir, tasks_dir, entry, config)
            except Exception as exc:
                logger.exception("SkillsBench E2E entry failed: %s", entry.id)
                return _entry_to_summary(entry, "unexpected_error", error=str(exc))

    entries = await asyncio.gather(*(bounded(entry) for entry in matrix))
    _write_json(
        run_dir / "matrix_summary.json",
        {
            "run_dir": str(run_dir),
            "dry_run": False,
            "total": len(entries),
            "entries": entries,
        },
    )
    write_artifact_audit(run_dir)
    baseline_dir, baseline_error = _resolve_baseline(config)
    write_parity_report(run_dir, baseline_dir, baseline_error)
    write_audit_outputs(run_dir, config.audit_prompt)
    if config.audit_agent_enabled:
        await _run_audit_agent(run_dir, config)
    return run_dir


async def _run_audit_agent(run_dir: Path, config: E2EConfig) -> None:
    """Run the optional post-processing audit agent over deterministic outputs."""
    from benchflow.integration.audit_agent import create_audit_task

    task_dir = create_audit_task(run_dir, config.audit_prompt)
    result = await SDK().run(
        task_path=task_dir,
        agent=config.audit_agent,
        model=config.audit_model,
        jobs_dir=run_dir / "audit_agent",
        job_name="review",
        environment=config.audit_environment,
        sandbox_user="agent",
    )
    _write_json(
        run_dir / "audit_agent_result.json",
        {
            "task_name": result.task_name,
            "trial_name": result.trial_name,
            "agent": result.agent,
            "agent_name": result.agent_name,
            "model": result.model,
            "rewards": result.rewards,
            "error": result.error,
            "verifier_error": result.verifier_error,
            "n_tool_calls": result.n_tool_calls,
        },
    )
