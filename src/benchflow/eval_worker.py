"""Subprocess worker for sharded evaluation runs.

The public ``bench eval run`` command uses this module when the operator asks
for worker-level isolation. A worker runs one normal :class:`Evaluation` over a
small include set and exits zero when BenchFlow completed the shard, even if the
benchmark tasks themselves failed. Non-zero exits are reserved for control-plane
failures so the parent can retry or report the shard cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.loop_strategies import LoopStrategySpec
from benchflow.skill_policy import SKILL_MODE_NO_SKILL
from benchflow.usage_tracking import UsageTrackingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _retry_config(raw: dict[str, Any]) -> RetryConfig:
    # Centralized parsing: omitted fields fall back to RetryConfig's own
    # defaults (which exclude provider_auth), not hard-coded literals, so a
    # partial worker payload can't silently revert to retrying auth failures
    # (#564 finding 2).
    return RetryConfig.from_mapping(raw.get("retry"))


def _environment_manifest(raw: dict[str, Any]):
    from benchflow.environment.manifest import EnvironmentManifest, load_manifest

    manifest = raw.get("environment_manifest")
    if manifest is not None:
        # The parent serialized the already-resolved manifest object (S axis),
        # so rebuild it directly. This preserves an inline --state tool subset
        # and needs no $BENCHFLOW_ENV_REGISTRY (or registry re-resolution) in
        # the worker subprocess.
        return EnvironmentManifest.model_validate(manifest)
    # Back-compat: a pre-fix shard payload (e.g. one mid-flight across an
    # upgrade, retried) carried only a path.
    manifest_path = raw.get("environment_manifest_path")
    if manifest_path:
        return load_manifest(Path(manifest_path))
    return None


def _evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        agent=raw.get("agent") or "claude-agent-acp",
        model=raw.get("model"),
        reasoning_effort=raw.get("reasoning_effort"),
        environment=raw.get("environment") or "docker",
        concurrency=int(raw.get("concurrency") or 1),
        prompts=raw.get("prompts"),
        agent_env=dict(raw.get("agent_env") or {}),
        retry=_retry_config(raw),
        skills_dir=raw.get("skills_dir"),
        sandbox_user=raw.get("sandbox_user", "agent"),
        sandbox_locked_paths=raw.get("sandbox_locked_paths"),
        sandbox_setup_timeout=int(raw.get("sandbox_setup_timeout") or 120),
        agent_idle_timeout=raw.get("agent_idle_timeout"),
        context_root=raw.get("context_root"),
        base_image_override=raw.get("base_image_override"),
        exclude_tasks=set(raw.get("exclude_tasks") or []),
        include_tasks=set(raw.get("include_tasks") or []),
        skill_mode=raw.get("skill_mode") or SKILL_MODE_NO_SKILL,
        skill_creator_dir=raw.get("skill_creator_dir"),
        self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
        job_mode=raw.get("job_mode") or "parallel-independent",
        source_provenance=raw.get("source_provenance"),
        usage_tracking=UsageTrackingConfig.from_mapping(raw),
        environment_manifest=_environment_manifest(raw),
        config_override=raw.get("config_override"),
        loop_strategy=(
            LoopStrategySpec.from_mapping(raw["loop_strategy"])
            if raw.get("loop_strategy")
            else None
        ),
    )


def _result_payload(result) -> dict[str, Any]:
    return {
        "job_name": result.job_name,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "errored": result.errored,
        "verifier_errored": result.verifier_errored,
        "elapsed_sec": result.elapsed_sec,
        "score": result.score,
        "score_excl_errors": result.score_excl_errors,
    }


async def run_worker(payload_path: Path) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text())
    config = _evaluation_config(payload["config"])
    evaluation = Evaluation(
        tasks_dir=payload["tasks_dir"],
        jobs_dir=payload["jobs_dir"],
        config=config,
    )
    result = await evaluation.run()
    result_payload = _result_payload(result)
    output_path = Path(payload["result_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, indent=2))
    return result_payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m benchflow.eval_worker <payload.json>", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(run_worker(Path(args[0])))
    except Exception as exc:
        print(f"Worker failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess runs
    raise SystemExit(main())
