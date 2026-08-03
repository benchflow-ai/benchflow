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

from benchflow._utils.scoring import pass_rate, pass_rate_excl_errors
from benchflow.evaluation import Evaluation, EvaluationConfig, EvaluationResult
from benchflow.loop_strategies import LoopStrategySpec
from benchflow.rollout._results import _should_record_env_entry


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
        "environment": config.environment,
        "concurrency": shard.concurrency,
        "prompts": config.prompts,
        "agent_env": config.agent_env,
        "retry": _retry_payload(config),
        "skills_dir": config.skills_dir,
        "sandbox_user": config.sandbox_user,
        "sandbox_locked_paths": config.sandbox_locked_paths,
        "sandbox_setup_timeout": config.sandbox_setup_timeout,
        "agent_idle_timeout": config.agent_idle_timeout,
        "context_root": config.context_root,
        "base_image_override": config.base_image_override,
        "exclude_tasks": sorted(config.exclude_tasks),
        "include_tasks": sorted(shard.task_names),
        "skill_mode": config.skill_mode,
        "skill_creator_dir": config.skill_creator_dir,
        "self_gen_no_internet": config.self_gen_no_internet,
        "job_mode": config.job_mode,
        "source_provenance": config.source_provenance,
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


async def _run_worker_once(payload_path: Path, log_path: Path) -> int:
    env = os.environ.copy()
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
    await _stream_worker_log(process, log_path)
    return await process.wait()


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
        ),
        "elapsed_sec": result.elapsed_sec,
        "concurrency": plan.total_concurrency,
        "worker_concurrency": plan.worker_concurrency,
        "worker_count": plan.worker_count,
        "shards": shard_results,
    }
    jobs_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2)
    (jobs_dir / "summary.json").write_text(summary_text)
    aggregate_job_dir = jobs_dir / result.job_name
    aggregate_job_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_job_dir / "summary.json").write_text(summary_text)
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
    discovery = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config)
    task_dirs = discovery._get_task_dirs()
    task_names = [task_dir.name for task_dir in task_dirs]
    plan = plan_eval_shards(
        task_names,
        total_concurrency=config.concurrency,
        worker_concurrency=worker_concurrency,
    )
    if not plan.shards:
        from benchflow.evaluation import EmptyTaskSelectionError

        raise EmptyTaskSelectionError(f"No tasks selected after filtering: {tasks_dir}")

    root = jobs_dir / "worker-shards"
    _write_or_validate_plan(root / "plan.json", plan)

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
    return _aggregate_result(
        jobs_dir=jobs_dir,
        config=config,
        plan=plan,
        shard_results=shard_results,
        elapsed_sec=elapsed,
    )
