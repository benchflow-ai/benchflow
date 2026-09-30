"""Where each ``bench eval run`` / ``branch`` / ``metrics`` / ``compare`` /
``regrade``, ``bench train convert`` and ``bench train stream`` flag lives in
Python.

Every CLI flag maps to one :class:`Equivalent`: a target in the public SDK
(``sdk``), a Python function or field outside the top-level namespace
(``module``), or ``cli`` with the reason there is no Python equivalent (console
output, or a flag that only records a value). ``tests/test_cli_python_parity.py``
fails when a flag is added or removed without updating these tables, or when a
target stops resolving; ``docs/reference/cli-python-parity.md`` is generated
from them (``python -m benchflow.cli_parity``).

Targets are dotted paths: ``module.Object`` (the object must exist),
``module.Class.field`` (a dataclass or pydantic field, or an attribute), or
``module.function(param)`` (the function must take ``param``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["sdk", "module", "cli"]

_EC = "benchflow.EvaluationConfig"
_EV = "benchflow.Evaluation"
_RC = "benchflow.ReviewerConfig"
_HOSTED = "benchflow.hosted_env.HostedEnvRunConfig"
_SHARD = "benchflow.eval_sharding.run_sharded_evaluation"
_ART = "benchflow.eval_artifacts"
_HF = "benchflow.publish.huggingface"
_BR = "benchflow.branch"


@dataclass(frozen=True)
class Equivalent:
    """The Python side of one CLI flag."""

    kind: Kind
    target: str = ""
    note: str = ""


def _sdk(target: str, note: str = "") -> Equivalent:
    return Equivalent("sdk", target, note)


def _mod(target: str, note: str = "") -> Equivalent:
    return Equivalent("module", target, note)


def _cli(note: str) -> Equivalent:
    return Equivalent("cli", "", note)


EVAL_RUN: dict[str, Equivalent] = {
    "--config": _sdk(
        f"{_EV}.from_yaml", "also Evaluation.from_dict(); to_yaml()/to_dict() write it"
    ),
    "--tasks-dir": _sdk(f"{_EV}.__init__(tasks_dir)"),
    "--source-repo": _sdk(
        "benchflow.resolve_source(repo)", "pass the returned path as tasks_dir"
    ),
    "--source-path": _sdk("benchflow.resolve_source(path)"),
    "--source-ref": _sdk("benchflow.resolve_source(ref)"),
    "--source-env": _mod(
        f"{_HOSTED}.source_env", "run with benchflow.hosted_env.run_hosted_env()"
    ),
    "--source-env-version": _mod("benchflow.hosted_env.HostedEnvRef.version"),
    "--source-env-arg": _mod(f"{_HOSTED}.env_args"),
    "--source-env-num-examples": _mod(f"{_HOSTED}.num_examples"),
    "--source-env-rollouts-per-example": _mod(f"{_HOSTED}.rollouts_per_example"),
    "--source-env-max-tokens": _mod(f"{_HOSTED}.max_tokens"),
    "--source-env-temperature": _mod(f"{_HOSTED}.temperature"),
    "--source-env-sampling-arg": _mod(f"{_HOSTED}.sampling_args"),
    "--source-env-verifiers-version": _mod(f"{_HOSTED}.verifiers_version"),
    "--source-env-base-url": _mod(f"{_HOSTED}.api_base_url"),
    "--source-env-api-key-var": _mod(f"{_HOSTED}.api_key_var"),
    "--agent": _sdk(f"{_EC}.agent"),
    "--model": _sdk(f"{_EC}.model"),
    "--reasoning-effort": _sdk(f"{_EC}.reasoning_effort"),
    "--sandbox": _sdk(f"{_EC}.environment"),
    "--usage-tracking": _sdk(f"{_EC}.usage_tracking"),
    "--environment-manifest": _sdk(
        f"{_EC}.environment_manifest", "bf.load_manifest(path or name@version)"
    ),
    "--state": _mod(
        "benchflow._utils.env_registry.resolve_state(value)",
        "its manifest goes in EvaluationConfig.environment_manifest",
    ),
    "--prompt": _sdk(f"{_EC}.prompts"),
    "--config-override": _sdk(f"{_EC}.config_override"),
    "--concurrency": _sdk(f"{_EC}.concurrency"),
    "--build-concurrency": _sdk(f"{_EC}.build_concurrency"),
    "--worker-concurrency": _mod(f"{_SHARD}(worker_concurrency)"),
    "--worker-retries": _mod(f"{_SHARD}(worker_retries)"),
    "--worker-start-stagger-sec": _mod(f"{_SHARD}(worker_start_stagger_sec)"),
    "--agent-idle-timeout": _sdk(f"{_EC}.agent_idle_timeout"),
    "--checkpoints": _sdk(f"{_EC}.checkpoints"),
    "--checkpoint-keep": _sdk(f"{_EC}.checkpoint_keep"),
    "--freeze-workspace": _sdk(f"{_EC}.freeze_workspace"),
    "--quiet": _cli("console output only; Python reports through the logging module"),
    "--jobs-dir": _sdk(f"{_EV}.__init__(jobs_dir)"),
    "--fresh": _sdk(f"{_EV}.__init__(job_name)", "pass a new job_name"),
    "--job-name": _sdk(f"{_EV}.__init__(job_name)"),
    "--fail-under": _sdk(
        "benchflow.EvaluationResult.score", "compare it to your threshold"
    ),
    "--fail-on": _sdk(
        "benchflow.EvaluationResult.results",
        "count results by error_category / errored / verifier_errored",
    ),
    "--summary-out": _mod(
        "benchflow.job_export.run_summary_export(result)",
        "the benchflow.run-summary document of an EvaluationResult",
    ),
    "--codex-apps-policy": _sdk(f"{_EC}.codex_apps_policy"),
    "--sandbox-user": _sdk(f"{_EC}.sandbox_user"),
    "--sandbox-setup-timeout": _sdk(f"{_EC}.sandbox_setup_timeout"),
    "--context-root": _sdk(f"{_EC}.context_root"),
    "--base-image-override": _sdk(f"{_EC}.base_image_override"),
    "--skills-dir": _sdk(f"{_EC}.skills_dir"),
    "--skill-mode": _sdk(f"{_EC}.skill_mode"),
    "--skill-creator-dir": _sdk(f"{_EC}.skill_creator_dir"),
    "--self-gen-no-internet": _sdk(f"{_EC}.self_gen_no_internet"),
    "--loop-strategy": _sdk(f"{_EC}.loop_strategy"),
    "--reviewer-agent": _sdk(f"{_RC}.agent", "EvaluationConfig.reviewer"),
    "--reviewer-model": _sdk(f"{_RC}.model"),
    "--reviewer-reasoning-effort": _sdk(f"{_RC}.reasoning_effort"),
    "--reviewer-sandbox": _sdk(f"{_RC}.environment"),
    "--reviewer-timeout-sec": _sdk(f"{_RC}.timeout_sec"),
    "--reviewer-idle-timeout": _sdk(f"{_RC}.idle_timeout_sec"),
    "--reviewer-concurrency": _sdk(f"{_RC}.concurrency"),
    "--reviewer-image": _sdk(f"{_RC}.image"),
    "--reviewer-agent-env": _sdk(f"{_RC}.agent_env"),
    "--reviewer-open-network": _sdk(f"{_RC}.open_network"),
    "--agent-env": _sdk(f"{_EC}.agent_env"),
    "--include": _sdk(f"{_EC}.include_tasks"),
    "--exclude": _sdk(f"{_EC}.exclude_tasks"),
    "--dataset": _mod(
        "benchflow._utils.dataset_registry.resolve_dataset(spec)",
        "pass its tasks dir as tasks_dir and its name/version/digests as EvaluationConfig.dataset_*",
    ),
    "--registry": _mod("benchflow._utils.dataset_registry.resolve_dataset(registry)"),
    "--ignore-bench-version": _cli(
        "the CLI's bench-version gate for --dataset; resolve_dataset() does not check the bench version"
    ),
    "--task-manifest-out": _mod(f"{_ART}.write_task_manifest(path)"),
    "--run-config-out": _sdk(
        f"{_EV}.to_dict", "the CLI's run-config JSON also records the retry flags"
    ),
    "--health-summary-out": _mod(f"{_ART}.write_health_summary(job_dir)"),
    "--expected-tasks": _mod(f"{_ART}.write_canonical_selection(expected_tasks)"),
    "--canonicalize": _mod(f"{_ART}.write_canonical_selection(policy)"),
    "--canonical-selection-out": _mod(f"{_ART}.write_canonical_selection(path)"),
    "--canonical-jobs-dir": _mod(f"{_ART}.materialize_canonical_job(output_dir)"),
    "--retry-from-checkpoint": _sdk(
        f"{_EC}.retry_from_checkpoint", "needs checkpoints"
    ),
    "--retry-prompt": _sdk(f"{_EC}.retry_prompt"),
    "--retry-resume-session": _sdk(f"{_EC}.retry_resume_session"),
    "--max-cost-usd": _sdk(
        "benchflow.Budget.max_cost_usd", "Evaluation(budget=Budget(...))"
    ),
    "--max-sandbox-seconds": _sdk("benchflow.Budget.max_sandbox_seconds"),
    "--max-tokens": _sdk("benchflow.Budget.max_tokens"),
    "--retry-policy": _cli("reserved: only recorded in the --run-config-out file"),
    "--retry-attempts": _sdk(
        "benchflow.RetryConfig.max_retries", "EvaluationConfig.retry"
    ),
    "--retry-concurrency": _cli("reserved: only recorded in the --run-config-out file"),
    "--publish-hf": _mod(f"{_HF}.publish_folder_to_hf(repo_id)"),
    "--hf-prefix": _mod(f"{_HF}.publish_folder_to_hf(path_in_repo)"),
    "--hf-public-read-check": _mod(f"{_HF}.publish_folder_to_hf(public_read_check)"),
    "--publish-bucket": _mod(f"{_HF}.publish_folder_to_bucket(bucket_id)"),
    "--eval-results-model": _mod(f"{_HF}.open_eval_results_pr(model_repo)"),
    "--eval-results-dataset": _mod(f"{_HF}.open_eval_results_pr(dataset_id)"),
    "--eval-results-task": _mod(f"{_HF}.open_eval_results_pr(task_id)"),
    "--matrix": _sdk(
        f"{_EV}.__init__(config)",
        "one Evaluation per model entry, each with its own jobs_dir",
    ),
    "--trials": _sdk(
        f"{_EV}.__init__(job_name)",
        "run the Evaluation once per trial with a distinct job_name",
    ),
    "--n-tasks": _sdk(f"{_EC}.n_tasks"),
    "--sample-seed": _sdk(f"{_EC}.sample_seed"),
    "--timeout-multiplier": _sdk(f"{_EC}.timeout_multiplier"),
    "--extra-instruction": _sdk(f"{_EC}.extra_instruction"),
    "--dry-run": _cli("prints the resolved plan and task selection; nothing runs"),
    "--seeds": _sdk(
        f"{_EC}.seeds",
        "a list of ints; benchflow.embodied.rollouts.parse_seeds reads the CLI form",
    ),
}

EVAL_BRANCH: dict[str, Equivalent] = {
    "--tasks-dir": _sdk(f"{_BR}(task_path)", "bf.branch runs one task per call"),
    "--agent": _sdk(f"{_BR}(agent)"),
    "--model": _sdk(f"{_BR}(model)"),
    "--reasoning-effort": _sdk(f"{_BR}(reasoning_effort)"),
    "--sandbox": _sdk(f"{_BR}(sandbox)"),
    "--prompt": _sdk(f"{_BR}(prompts)"),
    "--checkpoint-after-prompt": _sdk(f"{_BR}(checkpoint_after)"),
    "--child": _sdk(f"{_BR}(children)", "a label -> prompt mapping or ChildSpec items"),
    "--snapshot-layers": _sdk(f"{_BR}(snapshot_layers)"),
    "--parent": _sdk(f"{_BR}(parent)"),
    "--retain-snapshots": _sdk(f"{_BR}(retain_snapshots)"),
    "--from-checkpoint": _sdk(f"{_BR}(from_checkpoint)"),
    "--fork": _sdk(f"{_BR}(fork)"),
    "--checkpoints": _sdk(
        f"{_BR}(checkpoints)",
        "a CheckpointPolicy: benchflow.checkpoints.parse_checkpoint_policy()",
    ),
    "--checkpoint-keep": _mod("benchflow.checkpoints.parse_checkpoint_policy(keep)"),
    "--resume-session": _sdk(f"{_BR}(resume_session)"),
    "--child-retries": _sdk(f"{_BR}(child_retries)"),
    "--stop-on-child-failure": _sdk(f"{_BR}(continue_after_child_failure)"),
    "--concurrency": _sdk(f"{_BR}(concurrency)"),
    "--isolate-children": _sdk(f"{_BR}(isolate_children)"),
    "--agent-env": _sdk(f"{_BR}(agent_env)"),
    "--include": _sdk(f"{_BR}(task_path)", "call bf.branch once per selected task"),
    "--jobs-dir": _sdk(f"{_BR}(jobs_dir)"),
    "--job-name": _sdk(f"{_BR}(job_name)"),
}

_JOB = "benchflow.Job"

EVAL_METRICS: dict[str, Equivalent] = {
    "--benchmark": _mod("benchflow.metrics.collect_metrics(benchmark)"),
    "--agent": _mod("benchflow.metrics.collect_metrics(agent)"),
    "--model": _mod("benchflow.metrics.collect_metrics(model)"),
    "--json": _sdk(
        f"{_JOB}.solve_rates",
        "collect_metrics(...).summary() plus bf.load_job(dir).solve_rates().to_dict()",
    ),
    "--k": _sdk(f"{_JOB}.solve_rates(ks)"),
    "--solve-threshold": _sdk(f"{_JOB}.solve_rates(solve_threshold)"),
}

EVAL_COMPARE: dict[str, Equivalent] = {
    "--json": _sdk("benchflow.Comparison.to_json"),
    "--labels": _sdk("benchflow.compare(labels)"),
    "--vary": _sdk("benchflow.compare(vary)"),
    "--on-mismatch": _sdk("benchflow.compare(on_mismatch)"),
    "--include-controls": _sdk("benchflow.compare(include_controls)"),
    "--attempts": _sdk("benchflow.load_job(attempts)", "load each side, then compare"),
    "--out": _sdk("benchflow.Comparison.to_json(path)"),
    "--by": _sdk(
        "benchflow.compare(by)", "comma-separated on the CLI, a tuple in Python"
    ),
    "--k": _sdk("benchflow.compare(ks)"),
    "--solve-threshold": _sdk("benchflow.compare(solve_threshold)"),
}

EVAL_REGRADE: dict[str, Equivalent] = {
    "--tasks-dir": _sdk("benchflow.regrade(tasks_dir)"),
    "--sandbox": _sdk("benchflow.regrade(sandbox)"),
    "--concurrency": _sdk("benchflow.regrade(concurrency)"),
    "--reason": _sdk("benchflow.regrade(reason)"),
    "--json": _sdk(
        "benchflow.RegradeSummary.to_dict", "also written to regrade-summary.json"
    ),
}

_PRIME = "benchflow.trajectories.export_prime_sft.export_prime_sft_jsonl"
_TRL = "benchflow.trajectories.export_trl_sft.export_trl_sft_jsonl"
_BRANCH = "benchflow.trajectories.export_branch.export_branch_jsonl"

TRAIN_CONVERT: dict[str, Equivalent] = {
    "--out": _mod(
        f"{_PRIME}(out)", "same parameter on the TRL and branch-tree exporters"
    ),
    "--format": _mod(
        _PRIME,
        f"one function per format: prime-sft {_PRIME}, trl-sft {_TRL}, "
        f"branch-tree {_BRANCH}",
    ),
    "--pairs": _mod(f"{_BRANCH}(pairs_out)"),
    "--pairs-any-request": _mod(f"{_BRANCH}(any_request_pairs)"),
    "--include-oracle": _mod(f"{_BRANCH}(include_oracle)"),
    "--min-reward": _mod(f"{_PRIME}(min_reward)"),
    "--row-mode": _mod(f"{_PRIME}(row_mode)"),
    "--manifest": _mod(f"{_PRIME}(manifest)"),
    "--expected-rows": _mod(f"{_PRIME}(expected_rows)"),
    "--canonical-selection": _mod(f"{_PRIME}(canonical_selection)"),
    "--redact": _mod(f"{_PRIME}(redact)"),
    "--context-policy": _mod(f"{_TRL}(context_policy)"),
    "--tokenizer": _mod(f"{_TRL}(tokenizer_id)"),
    "--tokenizer-revision": _mod(f"{_TRL}(tokenizer_revision)"),
    "--max-length": _mod(f"{_TRL}(max_length)"),
    "--subagent-rows": _mod(f"{_PRIME}(subagent_rows)"),
    "--reward-vector": _mod(
        f"{_PRIME}(reward_vector)",
        "same on the TRL exporter; benchflow.trajectories.training_signal."
        "rollout_training_signals() reads the vectors without converting",
    ),
    "--group-advantage": _mod(f"{_PRIME}(group_advantage)", "same on the TRL exporter"),
    "--group-by": _mod(
        f"{_PRIME}(group_by)", "comma-separated on the CLI, a tuple in Python"
    ),
}

_STREAM = "benchflow.stream_rollouts"

TRAIN_STREAM: dict[str, Equivalent] = {
    "--format": _sdk(
        "benchflow.StreamedRollout.to_json",
        "jsonl is the only format; Python yields StreamedRollout objects",
    ),
    "--follow": _sdk(f"{_STREAM}(follow)", "--no-follow is follow=False"),
    "--poll-interval": _sdk(f"{_STREAM}(poll_interval)"),
    "--timeout": _sdk(f"{_STREAM}(timeout)", "raises StreamTimeout (CLI exit 3)"),
    "--group-size": _sdk(f"{_STREAM}(group_size)"),
    "--group-by": _sdk(
        f"{_STREAM}(group_by)", "comma-separated on the CLI, a tuple in Python"
    ),
}

TABLES: dict[str, dict[str, Equivalent]] = {
    "bench eval run": EVAL_RUN,
    "bench eval branch": EVAL_BRANCH,
    "bench eval metrics": EVAL_METRICS,
    "bench eval compare": EVAL_COMPARE,
    "bench eval regrade": EVAL_REGRADE,
    "bench train convert": TRAIN_CONVERT,
    "bench train stream": TRAIN_STREAM,
}


def resolve(target: str) -> object:
    """Resolve a target; raise ``LookupError`` naming what is missing."""
    import importlib
    import inspect

    param = None
    if target.endswith(")"):
        target, _, param = target[:-1].partition("(")
    parts = target.split(".")
    obj: object = importlib.import_module(parts[0])
    for name in parts[1:]:
        fields = getattr(obj, "__dataclass_fields__", None) or getattr(
            obj, "model_fields", None
        )
        if hasattr(obj, name):
            # Package attributes win over same-named submodules (bf.branch).
            obj = getattr(obj, name)
        elif isinstance(fields, dict) and name in fields:
            obj = fields[name]
        elif inspect.ismodule(obj):
            try:
                obj = importlib.import_module(f"{obj.__name__}.{name}")
            except ModuleNotFoundError:
                raise LookupError(f"{target!r}: {name!r} not found") from None
        else:
            raise LookupError(f"{target!r}: {name!r} not found")
    if param is not None and (
        not callable(obj) or param not in inspect.signature(obj).parameters
    ):
        raise LookupError(f"{target!r} takes no parameter {param!r}")
    return obj


def markdown() -> str:
    """The parity tables as markdown (docs/reference/cli-python-parity.md)."""
    lines = [
        "# CLI and Python equivalents",
        "",
        "Generated by `python -m benchflow.cli_parity` from `src/benchflow/cli_parity.py`; "
        "`tests/test_cli_python_parity.py` fails when a flag has no row or a target stops "
        "existing. `sdk` targets are in the public `benchflow` namespace, `module` targets "
        "are Python functions or fields in a submodule, and `cli` flags have no Python "
        "equivalent for the reason given.",
    ]
    for command, table in TABLES.items():
        lines += [
            "",
            f"## `{command}`",
            "",
            "| Flag | Kind | Python | Note |",
            "|---|---|---|---|",
        ]
        for flag, eq in table.items():
            target = f"`{eq.target}`" if eq.target else ""
            lines.append(f"| `{flag}` | {eq.kind} | {target} | {eq.note} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(markdown(), end="")
