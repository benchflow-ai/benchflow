"""Worker-sharded evaluation orchestration.

This module keeps the high-concurrency control-plane policy out of
``evaluation.py``. Each worker subprocess runs a normal Evaluation over a
disjoint include set with a modest local concurrency; the parent owns only the
durable shard plan, worker retries, and aggregate summary.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.scoring import pass_rate, pass_rate_excl_errors
from benchflow.evaluation import (
    EmptyTaskSelectionError,
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
)
from benchflow.loop_strategies import LoopStrategySpec
from benchflow.task.verifier_sandbox import effective_verifier_sandbox


@dataclass(frozen=True)
class EvalShard:
    index: int
    task_names: tuple[str, ...]
    concurrency: int


@dataclass(frozen=True)
class EvalShardPlan:
    total_concurrency: int
    worker_concurrency: int
    shards: tuple[EvalShard, ...]

    @property
    def worker_count(self) -> int:
        return len(self.shards)


class ShardWorkerError(RuntimeError):
    """Raised when one or more worker subprocesses fail after retries."""


def plan_eval_shards(
    task_names: list[str],
    *,
    total_concurrency: int,
    worker_concurrency: int,
) -> EvalShardPlan:
    """Split tasks across isolated workers without exceeding target concurrency."""
    if total_concurrency < 1:
        raise ValueError("total_concurrency must be >= 1")
    if worker_concurrency < 1:
        raise ValueError("worker_concurrency must be >= 1")
    if not task_names:
        return EvalShardPlan(total_concurrency, worker_concurrency, ())

    worker_count = min(
        len(task_names),
        max(1, math.ceil(total_concurrency / worker_concurrency)),
    )
    remaining = total_concurrency
    concurrencies: list[int] = []
    for _ in range(worker_count):
        concurrency = min(worker_concurrency, remaining)
        concurrencies.append(max(1, concurrency))
        remaining -= concurrency

    buckets: list[list[str]] = [[] for _ in range(worker_count)]
    for offset, task_name in enumerate(task_names):
        buckets[offset % worker_count].append(task_name)

    shards = tuple(
        EvalShard(index=i, task_names=tuple(bucket), concurrency=concurrencies[i])
        for i, bucket in enumerate(buckets)
        if bucket
    )
    return EvalShardPlan(total_concurrency, worker_concurrency, shards)


def _retry_payload(config: EvaluationConfig) -> dict[str, Any]:
    retry = config.retry
    return {
        "max_retries": retry.max_retries,
        "retry_on_install": retry.retry_on_install,
        "retry_on_pipe": retry.retry_on_pipe,
        "retry_on_acp": retry.retry_on_acp,
        "retry_on_idle_timeout": retry.retry_on_idle_timeout,
        "retry_on_infra": retry.retry_on_infra,
        "retry_on_verifier_infra": retry.retry_on_verifier_infra,
        "wait_multiplier": retry.wait_multiplier,
        "min_wait_sec": retry.min_wait_sec,
        "max_wait_sec": retry.max_wait_sec,
        "exclude_categories": sorted(retry.exclude_categories),
    }


def _shard_warm_pool(config: EvaluationConfig, shard: EvalShard) -> int:
    """This shard's share of the job's warm pool: 1 for the first N shards.

    N is the job's (``--warm-pool``, or its default for the job's whole
    ``--concurrency``), so the job's live sandboxes stay at most
    ``--concurrency`` + N however the trials are sharded.
    """
    from benchflow._warm_pool import warm_pool_size

    job_size = warm_pool_size(config.concurrency, config.warm_pool)
    # A shard without an index (a caller's own stand-in) counts as the first.
    return 1 if getattr(shard, "index", 0) < job_size else 0


def _config_payload(
    config: EvaluationConfig,
    *,
    shard: EvalShard,
) -> dict[str, Any]:
    if config.loop_strategy is not None and not isinstance(
        config.loop_strategy, LoopStrategySpec
    ):
        # EvaluationConfig.__post_init__ parses spec strings; anything else
        # here means the config bypassed validation — fail loudly rather
        # than silently dropping the strategy from the worker payload.
        raise TypeError(
            "EvaluationConfig.loop_strategy must be a parsed LoopStrategySpec "
            f"by sharding time, got {type(config.loop_strategy).__name__}"
        )
    payload = {
        "agent": config.agent,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "harness": config.harness,
        "environment": config.environment,
        "concurrency": shard.concurrency,
        # The job's warm pool is job-wide: its N prepared sandboxes are
        # shared out one per shard, so all workers together prepare at most N
        # (PERF-C review R2-M1).
        "warm_pool": _shard_warm_pool(config, shard),
        "prompts": config.prompts,
        "agent_env": config.agent_env,
        "reviewer": config.reviewer.to_dict(),
        "retry": _retry_payload(config),
        "skills_dir": config.skills_dir,
        "codex_apps_policy": config.codex_apps_policy,
        "sandbox_user": config.sandbox_user,
        "sandbox_locked_paths": config.sandbox_locked_paths,
        "sandbox_setup_timeout": config.sandbox_setup_timeout,
        "skip_agent_install": config.skip_agent_install,
        "agent_idle_timeout": config.agent_idle_timeout,
        "timeout_multiplier": config.timeout_multiplier,
        "extra_instruction": config.extra_instruction,
        "checkpoints": config.checkpoints,
        "checkpoint_keep": config.checkpoint_keep,
        "freeze_workspace": config.freeze_workspace,
        "fresh_install": config.fresh_install,
        "allow_cloud_metadata": config.allow_cloud_metadata,
        "integrity": config.integrity,
        "verifier_sandbox": effective_verifier_sandbox(
            config.verifier_sandbox, config.separate_verifier
        ),
        "retry_from_checkpoint": config.retry_from_checkpoint,
        "retry_prompt": config.retry_prompt,
        "retry_resume_session": config.retry_resume_session,
        "context_root": config.context_root,
        "base_image_override": config.base_image_override,
        "separate_verifier": config.separate_verifier,
        "separate_verifier_include": list(config.separate_verifier_include),
        "exclude_tasks": sorted(config.exclude_tasks),
        "include_tasks": sorted(shard.task_names),
        "skill_mode": config.skill_mode,
        "skill_creator_dir": config.skill_creator_dir,
        "self_gen_no_internet": config.self_gen_no_internet,
        "job_mode": config.job_mode,
        "source_provenance": config.source_provenance,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "dataset_task_digests": dict(config.dataset_task_digests),
        # Serialize the already-resolved manifest OBJECT (the S axis), not a
        # path. The parent resolves --state / --environment-manifest /
        # name@version into config.environment_manifest before sharding; a
        # path-only payload dropped that binding entirely for --state runs
        # (request.environment_manifest is None) and lost any inline tool
        # subset from resolve_state. model_dump round-trips the filtered
        # services so the worker boots the exact same world.
        "environment_manifest": (
            config.environment_manifest.model_dump(mode="json")
            if config.environment_manifest is not None
            else None
        ),
        "config_override": config.config_override,
        "loop_strategy": (
            config.loop_strategy.to_mapping() if config.loop_strategy else None
        ),
    }
    payload.update(config.usage_tracking.to_mapping())
    return payload


def _redacted_config_payload(config_payload: dict[str, Any]) -> dict[str, Any]:
    artifact_payload = dict(config_payload)
    agent_env = artifact_payload.get("agent_env")
    if isinstance(agent_env, dict):
        artifact_payload["agent_env"] = {
            str(key): str(value)
            for key, value in agent_env.items()
            if _should_record_env_entry(str(key), str(value))
        }
        artifact_payload["agent_env_keys"] = sorted(str(key) for key in agent_env)
    reviewer = artifact_payload.get("reviewer")
    if isinstance(reviewer, dict):
        from benchflow.review.options import ReviewerConfig

        artifact_payload["reviewer"] = ReviewerConfig.coerce(
            reviewer
        ).to_config_artifact()
    return artifact_payload


def _worker_payload_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact_payload = dict(payload)
    config_payload = artifact_payload.get("config")
    if isinstance(config_payload, dict):
        artifact_payload["config"] = _redacted_config_payload(config_payload)
    return artifact_payload


def _write_private_worker_payload(payload: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="benchflow-worker-payload-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        return Path(handle.name)


def _plan_payload(plan: EvalShardPlan) -> dict[str, Any]:
    return {
        "total_concurrency": plan.total_concurrency,
        "worker_concurrency": plan.worker_concurrency,
        "shards": [
            {
                "index": shard.index,
                "concurrency": shard.concurrency,
                "task_names": list(shard.task_names),
            }
            for shard in plan.shards
        ],
    }


def _write_or_validate_plan(path: Path, plan: EvalShardPlan) -> None:
    payload = _plan_payload(plan)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(
                f"Existing shard plan at {path} differs from requested run. "
                "Use a fresh --jobs-dir or the original sharding flags."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


async def _stream_worker_log(
    process: asyncio.subprocess.Process, log_path: Path
) -> None:
    assert process.stdout is not None
    with log_path.open("ab") as log:
        async for chunk in process.stdout:
            log.write(chunk)
            log.flush()


# How long a stopped worker gets to remove its sandboxes before SIGKILL.
WORKER_STOP_TIMEOUT_SEC = 120.0


async def _run_worker_once(payload_path: Path, log_path: Path) -> int:
    from benchflow.task.env import ALLOW_TASK_ENV_VAR, allowed_task_env

    env = os.environ.copy()
    # Names allowed through benchflow.task.env.allow_task_env live in this
    # process's context; the worker gets them as its variable.
    allowed = allowed_task_env()
    if allowed:
        env[ALLOW_TASK_ENV_VAR] = ",".join(sorted(allowed))
    python_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "benchflow.eval_worker",
        str(payload_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        await _stream_worker_log(process, log_path)
        return await process.wait()
    except asyncio.CancelledError:
        # The run is stopping (Ctrl-C, SIGTERM, SIGHUP): stop the worker the
        # same way and wait for its cleanup, so its sandboxes go with it.
        await _stop_worker(process, log_path)
        raise


async def _stop_worker(process: asyncio.subprocess.Process, log_path: Path) -> None:
    """SIGTERM a running worker and wait for its teardown; SIGKILL at the cap.

    Its output is still read meanwhile: a worker blocked on a full pipe
    could not finish removing its sandboxes.
    """
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(
            asyncio.gather(_stream_worker_log(process, log_path), process.wait()),
            WORKER_STOP_TIMEOUT_SEC,
        )
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()


async def _run_worker_with_retries(
    payload_path: Path,
    log_path: Path,
    *,
    max_retries: int,
) -> dict[str, Any]:
    result_path = Path(json.loads(payload_path.read_text())["result_path"])
    attempts = 0
    last_returncode = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        with log_path.open("ab") as log:
            log.write(
                f"\n=== worker attempt {attempts} "
                f"started {datetime.now(UTC).isoformat()} ===\n".encode()
            )
        last_returncode = await _run_worker_once(payload_path, log_path)
        if last_returncode == 0 and result_path.exists():
            result = json.loads(result_path.read_text())
            result["attempts"] = attempts
            result["returncode"] = last_returncode
            result_path.write_text(json.dumps(result, indent=2))
            return result
        with log_path.open("ab") as log:
            log.write(
                f"=== worker attempt {attempts} failed rc={last_returncode} ===\n".encode()
            )
    raise ShardWorkerError(
        f"{payload_path.parent.name} failed after {attempts} attempt(s), "
        f"last return code {last_returncode}; see {log_path}"
    )


def _aggregate_result(
    *,
    jobs_dir: Path,
    config: EvaluationConfig,
    plan: EvalShardPlan,
    shard_results: list[dict[str, Any]],
    elapsed_sec: float,
) -> EvaluationResult:
    result = EvaluationResult(
        job_name="worker-sharded",
        config=config,
        total=sum(int(r.get("total", 0)) for r in shard_results),
        passed=sum(int(r.get("passed", 0)) for r in shard_results),
        failed=sum(int(r.get("failed", 0)) for r in shard_results),
        errored=sum(int(r.get("errored", 0)) for r in shard_results),
        verifier_errored=sum(int(r.get("verifier_errored", 0)) for r in shard_results),
        guard_refused=sum(int(r.get("guard_refused", 0)) for r in shard_results),
        elapsed_sec=elapsed_sec,
    )
    summary = {
        "job_name": result.job_name,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "errored": result.errored,
        "verifier_errored": result.verifier_errored,
        "score": result.score,
        "score_ratio": pass_rate(passed=result.passed, total=result.total),
        "score_excl_errors": result.score_excl_errors,
        "score_excl_errors_ratio": pass_rate_excl_errors(
            passed=result.passed,
            failed=result.failed,
            refused=result.guard_refused,
        ),
        "elapsed_sec": result.elapsed_sec,
        "concurrency": plan.total_concurrency,
        "worker_concurrency": plan.worker_concurrency,
        "worker_count": plan.worker_count,
        "shards": shard_results,
    }
    jobs_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2)
    from benchflow._utils.disk_space import write_text_atomic

    write_text_atomic(jobs_dir / "summary.json", summary_text)
    aggregate_job_dir = jobs_dir / result.job_name
    aggregate_job_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(aggregate_job_dir / "summary.json", summary_text)
    return result


async def run_sharded_evaluation(
    *,
    tasks_dir: Path,
    jobs_dir: Path,
    config: EvaluationConfig,
    worker_concurrency: int,
    worker_retries: int,
    worker_start_stagger_sec: float,
) -> EvaluationResult:
    """Run a batch as isolated worker subprocesses and aggregate their summaries."""
    if config.pre_agent_hooks:
        raise ValueError(
            "pre_agent_hooks are Python callables and cannot reach worker "
            "processes; run without worker_concurrency (--worker-concurrency)"
        )
    discovery = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config)
    task_dirs = discovery._get_task_dirs()
    task_names = [task_dir.name for task_dir in task_dirs]
    plan = plan_eval_shards(
        task_names,
        total_concurrency=config.concurrency,
        worker_concurrency=worker_concurrency,
    )
    if not plan.shards:
        raise EmptyTaskSelectionError(discovery._empty_selection_message())
    # The workers each get an exact shard list, so only this discovery sees
    # the user's patterns.
    discovery.warn_unmatched_selection()
    # Refuse a task reading an unnamed variable once, here, rather than in
    # every worker (which inherit BENCHFLOW_ALLOW_TASK_ENV from this process).
    from benchflow.evaluation import _refuse_unnamed_task_env

    _refuse_unnamed_task_env(
        task_dirs, manifest=config.environment_manifest, sandbox=config.environment
    )

    # Check the complete selection before launching any worker; each worker
    # then validates its own credential context before starting a solver.
    from benchflow.review.automatic import prepare_review

    for task_dir in task_dirs:
        prepare_review(task_dir, config.reviewer)
    # A run that moves every verifier into the separate verifier sandbox is
    # refused before any shard folder or plan exists when a task cannot use it.
    from benchflow.integrity.trial import check_separate_plan

    check_separate_plan(
        task_dirs,
        sandbox=config.environment,
        run_setting=effective_verifier_sandbox(
            config.verifier_sandbox, config.separate_verifier
        ),
        integrity=config.integrity,
        config_override=config.config_override,
        environment_manifest=config.environment_manifest,
    )

    root = jobs_dir / "worker-shards"
    _write_or_validate_plan(root / "plan.json", plan)

    # Read resume state in each worker's own job namespace. Completed tasks
    # will not create networks; only unfinished tasks across all shards pin
    # space. Surviving networks are still excluded by daemon inspection.
    unfinished = []
    if config.environment in ("docker", "remote-docker"):
        from benchflow.eval_worker import _evaluation_config

        for shard in plan.shards:
            worker = Evaluation(
                tasks_dir=tasks_dir,
                jobs_dir=root / f"shard-{shard.index:03d}" / "jobs",
                config=_evaluation_config(_config_payload(config, shard=shard)),
                preflight=False,
            )
            completed = worker._get_completed_tasks()
            unfinished.extend(
                d
                for d in task_dirs
                if d.name in shard.task_names and d.name not in completed
            )

    # Every worker gets all unfinished tasks' pins, including other shards.
    from benchflow.evaluation import _pin_evaluation_subnets
    from benchflow.sandbox._docker_subnets import (
        backend_daemon_key,
        daemon_identity,
    )

    pinned = (
        _pin_evaluation_subnets(unfinished, config)
        if config.environment in ("docker", "remote-docker")
        else []
    )

    worker_tasks = []
    for shard in plan.shards:
        shard_dir = root / f"shard-{shard.index:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        payload_path = shard_dir / "worker_payload.json"
        result_path = shard_dir / "worker_result.json"
        payload = {
            "tasks_dir": str(tasks_dir),
            "jobs_dir": str(shard_dir / "jobs"),
            "result_path": str(result_path),
            "config": _config_payload(config, shard=shard),
            "docker_pinned_subnets": [str(net) for net in pinned],
            "docker_subnet_daemon": (
                daemon_identity(backend_daemon_key(config.environment))
                if pinned
                else None
            ),
        }
        payload_path.write_text(
            json.dumps(_worker_payload_artifact(payload), indent=2) + "\n"
        )
        private_payload_path = _write_private_worker_payload(payload)
        log_path = shard_dir / "worker.log"
        worker_tasks.append((shard, private_payload_path, log_path))

    async def run_one(offset: int, payload_path: Path, log_path: Path):
        if worker_start_stagger_sec > 0:
            await asyncio.sleep(offset * worker_start_stagger_sec)
        try:
            return await _run_worker_with_retries(
                payload_path,
                log_path,
                max_retries=worker_retries,
            )
        finally:
            with suppress(FileNotFoundError):
                payload_path.unlink()

    started = datetime.now(UTC)
    results_or_errors = await asyncio.gather(
        *[
            run_one(offset, payload_path, log_path)
            for offset, (_shard, payload_path, log_path) in enumerate(worker_tasks)
        ],
        return_exceptions=True,
    )
    errors = [err for err in results_or_errors if isinstance(err, BaseException)]
    if errors:
        failure_report = {
            "failed_at": datetime.now(UTC).isoformat(),
            "errors": [str(err) for err in errors],
            "plan": _plan_payload(plan),
        }
        (root / "worker_failures.json").write_text(json.dumps(failure_report, indent=2))
        raise ShardWorkerError("; ".join(str(err) for err in errors))

    elapsed = (datetime.now(UTC) - started).total_seconds()
    shard_results = [result for result in results_or_errors if isinstance(result, dict)]
    aggregate = _aggregate_result(
        jobs_dir=jobs_dir,
        config=config,
        plan=plan,
        shard_results=shard_results,
        elapsed_sec=elapsed,
    )
    stopped = [r["usage_limit"] for r in shard_results if r.get("usage_limit")]
    if stopped:
        # Every shard runs on the same login: raise the first shard's limit,
        # as an unsharded Evaluation does.
        from benchflow.agents.errors import UsageLimitError

        first = stopped[0]
        resets = first.get("resets_at")
        error = UsageLimitError(
            str(first.get("detail") or "usage limit reached"),
            login=first.get("login"),
            window=first.get("window"),
            resets_at=datetime.fromisoformat(resets) if resets else None,
        )
        error.result = aggregate
        raise error
    return aggregate
