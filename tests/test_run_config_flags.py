"""``bench run --config`` honours every flag (audit CLI-1).

A run config is the whole run, and "flags win over its keys" (``bench run
--help``). ``_config_file_evaluation`` applied the command line to the file's
run one flag at a time, from a list it kept, and seven flags were missing
from it: ``--token-budget``, ``--turn-budget``, ``--task-skills``,
``--skill-mode``, ``--skills-dir``, ``--loop-strategy`` and
``--context-root`` (``--seeds`` too). The run went ahead without them and
exited 0. The flags now reach a run config through the builder the
task-folder path uses (``cli.main._flag_settings``).

Every ``bench run`` flag is passed both ways here, a task folder and a run
config, and each setting it gives the task folder's run must be the run
config's. A flag is either a setting (``SETTINGS``, with a value to pass) or
named in ``NOT_SETTINGS`` with why; a flag added later that is in neither
fails ``test_every_bench_run_flag_is_classified``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

import benchflow.cli.main as main
from benchflow.cli.main import app

AGENT = ["-a", "claude-agent-acp", "-m", "claude-haiku-5-5"]

# Flag (the click parameter's name) -> the arguments that give it, in a
# folder holding `tasks/hello-craft`, `skills/`, `ctx/` and `environment.toml`.
SETTINGS: dict[str, list[str]] = {
    "model": ["-m", "claude-sonnet-5-5"],
    "effort": ["-e", "high"],
    "harness": ["--harness", "native"],
    "env": ["--env", "FOO=bar"],
    "task_skills": ["--task-skills"],
    "include": ["-i", "hello-craft"],
    "exclude": ["-x", "other-task"],
    "limit": ["-n", "1"],
    "sandbox": ["-s", "daytona"],
    "concurrency": ["-j", "3"],
    "warm_pool": ["--warm-pool", "2"],
    "timeout_multiplier": ["--timeout-multiplier", "2.0"],
    "idle_timeout": ["--idle-timeout", "90s"],
    "save_outputs": ["--save-outputs"],
    "extra_instruction": ["--extra-instruction", "Be brief."],
    "verifier_sandbox": ["--verifier-sandbox", "separate"],
    "usage_tracking": ["--usage-tracking", "off"],
    "prompt": ["--prompt", "Do it."],
    "config_override": ["--config-override", '{"agent": {"timeout_sec": 30}}'],
    "build_concurrency": ["--build-concurrency", "2"],
    "checkpoints": ["--checkpoints", "every-prompt"],
    "checkpoint_keep": ["--checkpoints", "every-prompt", "--checkpoint-keep", "5"],
    "keep_sandbox": ["--keep-sandbox", "on-failure"],
    "accept_deviations": ["--accept-deviations", "a test"],
    "fresh_install": ["--fresh-install"],
    "allow_cloud_metadata": ["--allow-cloud-metadata"],
    # audit is the default: strict is the flag that changes a setting.
    "integrity": ["--integrity", "strict"],
    "retry_from_checkpoint": [
        "--checkpoints",
        "every-prompt",
        "--retry-from-checkpoint",
        "on-failure",
    ],
    "retry_prompt": [
        "--checkpoints",
        "every-prompt",
        "--retry-from-checkpoint",
        "on-failure",
        "--retry-prompt",
        "Again.",
    ],
    "retry_resume_session": [
        "--checkpoints",
        "every-prompt",
        "--retry-from-checkpoint",
        "on-failure",
        "--retry-resume-session",
    ],
    "max_cost_usd": ["--max-cost-usd", "5"],
    "max_sandbox_seconds": ["--max-sandbox-seconds", "600"],
    "max_tokens": ["--max-tokens", "100000"],
    "max_rollouts": ["--max-rollouts", "7"],
    "token_budget": ["--token-budget", "100000"],
    "turn_budget": ["--turn-budget", "40"],
    "codex_apps_policy": ["--codex-apps-policy", "inherit"],
    "sandbox_user": ["--sandbox-user", "root"],
    "sandbox_setup_timeout": ["--sandbox-setup-timeout", "300"],
    "verifier_setup_timeout": ["--verifier-setup-timeout", "240"],
    "context_root": ["--context-root", "ctx"],
    "base_image_override": ["--base-image-override", "ubuntu:24.04=ubuntu:22.04"],
    "separate_verifier": ["--separate-verifier"],
    "separate_verifier_include": [
        "--separate-verifier",
        "--separate-verifier-include",
        "/data",
    ],
    "skills_dir": ["--skill-mode", "with-skill", "--skills-dir", "skills"],
    "skill_mode": ["--skill-mode", "with-skill"],
    "skill_creator_dir": [
        "--skill-mode",
        "self-gen",
        "--skill-creator-dir",
        "skills",
    ],
    "self_gen_no_internet": [
        "--skill-mode",
        "self-gen",
        "--skill-creator-dir",
        "skills",
        "--self-gen-no-internet",
    ],
    "loop_strategy": ["--loop-strategy", "self-review:k=2"],
    "reviewer_agent": ["--reviewer-agent", "codex-acp"],
    "reviewer_model": ["--reviewer-model", "gpt-6.1-sol"],
    "reviewer_reasoning_effort": ["--reviewer-reasoning-effort", "high"],
    "reviewer_sandbox": ["--reviewer-sandbox", "daytona"],
    "reviewer_timeout_sec": ["--reviewer-timeout-sec", "900"],
    "reviewer_idle_timeout_sec": ["--reviewer-idle-timeout", "300"],
    "reviewer_storage_mb": ["--reviewer-storage-mb", "4096"],
    "reviewer_memory_mb": ["--reviewer-memory-mb", "4096"],
    "reviewer_setup_timeout_sec": ["--reviewer-setup-timeout", "900"],
    "reviewer_evidence_timeout_sec": ["--reviewer-evidence-timeout", "900"],
    "reviewer_concurrency": ["--reviewer-concurrency", "2"],
    "reviewer_image": ["--reviewer-image", "python:3.12-slim"],
    "reviewer_agent_env": ["--reviewer-agent-env", "BAR=baz"],
    "reviewer_open_network": ["--reviewer-open-network"],
    "reviewer_pdf_tools": ["--reviewer-pdf-tools"],
    "reviewer_submitted_file_rule": ["--reviewer-submitted-file-rule"],
    "reviewer_criterion_evidence": ["--reviewer-criterion-evidence"],
    "sample_seed": ["-n", "1", "--sample-seed", "7"],
    "environment_manifest": ["--environment-manifest", "environment.toml"],
    "state": ["--state", "environment.toml"],
}

# Flags that are no run setting of the trials: what to run, where the job's
# folder and outputs go, CI gates, publishing, and how the CLI runs the job.
NOT_SETTINGS: dict[str, str] = {
    "targets": "the task folder or repository (a source)",
    "agent": "the agent: _config_file_evaluation applies it with the file's",
    "config_file": "the run config itself",
    "tasks_dir": "a source",
    "source_repo": "a source",
    "source_path": "a source",
    "source_ref": "a source",
    "dataset": "a source",
    "registry": "where --dataset comes from",
    "hub": "where --dataset comes from",
    "ignore_bench_version": "a --dataset check",
    "matrix": "a run of several jobs, refused beside --config",
    "trials": "repeats, refused beside --config",
    "seeds": "seeded variants, refused beside --config",
    "jobs_dir": "the job's folder",
    "job": "the job's folder",
    "fresh": "the job's folder",
    "restart_changed": "the job's folder (Evaluation(restart_changed=))",
    "dry_run": "prints the plan",
    "quiet": "console output",
    "help_all": "help",
    "fail_under": "a CI gate on the result",
    "fail_on": "a CI gate on the result",
    "summary_out": "an output file",
    "task_manifest_out": "an output file",
    "run_config_out": "an output file",
    "health_summary_out": "an output file",
    "expected_tasks": "a CI check of the selection",
    "canonicalize": "post-run selection of the trials",
    "canonical_selection_out": "an output file",
    "canonical_jobs_dir": "an output folder",
    "publish_hf": "publishing after the run",
    "hf_prefix": "publishing after the run",
    "hf_public_read_check": "publishing after the run",
    "publish_bucket": "publishing after the run",
    "public": "publishing after the run",
    "eval_results_model": "publishing after the run",
    "eval_results_dataset": "publishing after the run",
    "eval_results_task": "publishing after the run",
    "worker_concurrency": "how the CLI splits the run into worker processes",
    "worker_retries": "how the CLI splits the run into worker processes",
    "worker_start_stagger_sec": "how the CLI splits the run into worker processes",
    "allow_task_env": "the host variables a task may read (process environment)",
    "retry_policy": "no EvaluationConfig field (audit CLI-10, another lane)",
    "retry_concurrency": "no EvaluationConfig field (audit CLI-10, another lane)",
    "retry_attempts": "RetryConfig.max_retries, set after the config on both paths "
    "(test_retry_attempts_reach_a_run_config)",
    "source_env": "a hosted environment run (a source)",
    "source_env_version": "a hosted environment run",
    "source_env_arg": "a hosted environment run",
    "source_env_num_examples": "a hosted environment run",
    "source_env_rollouts_per_example": "a hosted environment run",
    "source_env_max_tokens": "a hosted environment run",
    "source_env_temperature": "a hosted environment run",
    "source_env_sampling_arg": "a hosted environment run",
    "source_env_verifiers_version": "a hosted environment run",
    "source_env_base_url": "a hosted environment run",
    "source_env_api_key_var": "a hosted environment run",
}

# Fields that name the run's source or are bookkeeping, which differ between a
# task folder and a run config whatever the flags.
_SOURCE_FIELDS = {"source_provenance", "agent_defaulted", "model_given"}


def _bench_run_params() -> list[Any]:
    return list(get_command(app).commands["run"].params)


def test_every_bench_run_flag_is_classified():
    names = {p.name for p in _bench_run_params()}
    assert not set(SETTINGS) & set(NOT_SETTINGS)
    unclassified = names - set(SETTINGS) - set(NOT_SETTINGS)
    assert not unclassified, (
        f"bench run flags with no case here: {sorted(unclassified)}; add each to "
        "SETTINGS with a value (and check that a run config gets it) or to "
        "NOT_SETTINGS with why it is no setting of the run"
    )
    assert not (set(SETTINGS) | set(NOT_SETTINGS)) - names


@pytest.fixture
def folder(tmp_path, monkeypatch) -> Path:
    from tests.test_dogfood_dx_b import _task

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARBOR_TELEMETRY", "off")
    # The proxy's own keys, so a per-trial budget's route check passes.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-not-a-key")
    _task(tmp_path / "tasks")
    (tmp_path / "skills").mkdir()
    (tmp_path / "ctx").mkdir()
    (tmp_path / "environment.toml").write_text(
        '[environment]\nname = "env0"\nimage = "example/env0:latest"\n'
    )
    (tmp_path / "run.yaml").write_text(
        "tasks_dir: tasks\nagent: claude-agent-acp\nmodel: claude-haiku-5-5\n"
    )
    return tmp_path


def _plan(monkeypatch, *argv: str) -> Any:
    """The plan ``bench run <argv> --dry-run`` makes, before it prints."""
    seen: list[Any] = []
    # The stand-in takes what the dry run is given (a repository target's ref
    # and path too).
    monkeypatch.setattr(
        main, "_print_dry_run", lambda plan, _, **_given: seen.append(plan)
    )
    result = CliRunner().invoke(app, [*argv, "--dry-run"], prog_name="bench")
    assert result.exit_code == 0 and seen, (argv, result.output)
    return seen[0]


def _fields(config: Any) -> dict[str, Any]:
    return {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name not in _SOURCE_FIELDS
    }


@pytest.mark.parametrize("flag", sorted(SETTINGS))
def test_a_run_config_gets_every_setting_a_flag_gives(flag, folder, monkeypatch):
    args = SETTINGS[flag]
    # -m is the flag under test: the agent alone around it.
    agent = ["-a", "claude-agent-acp"] if flag == "model" else AGENT
    plain = _fields(_plan(monkeypatch, "run", "tasks", *agent).make_eval_config())
    flagged = _fields(
        _plan(monkeypatch, "run", "tasks", *agent, *args).make_eval_config()
    )
    changed = {name for name in flagged if flagged[name] != plain[name]}
    assert changed, f"{flag}: {args} changes no setting of a task-folder run"

    configured = main._config_file_evaluation(
        _plan(monkeypatch, "run", "--config", "run.yaml", *args)
    )._config
    got = _fields(configured)
    dropped = {
        name: (got[name], flagged[name])
        for name in changed
        if got[name] != flagged[name]
    }
    assert not dropped, f"{flag}: a run config's (got, flag's): {dropped}"


def test_the_audits_dropped_flags_reach_a_run_config(folder, monkeypatch):
    """The audit's reproduction (CLI-1), in one command line."""
    configured = main._config_file_evaluation(
        _plan(
            monkeypatch,
            "run",
            "--config",
            "run.yaml",
            "--skill-mode",
            "with-skill",
            "--skills-dir",
            "skills",
            "--token-budget",
            "1000",
            "--turn-budget",
            "5",
            "--loop-strategy",
            "self-review:k=2",
            "--context-root",
            "ctx",
        )
    )._config
    assert configured.skill_mode == "with-skill"
    assert configured.skills_dir is not None and configured.skills_dir.endswith(
        "skills"
    )
    assert (configured.trial_budget.tokens, configured.trial_budget.turns) == (1000, 5)
    assert configured.loop_strategy is not None
    assert configured.loop_strategy.to_mapping()["strategy"] == "self-review"
    assert configured.context_root is not None and configured.context_root.endswith(
        "ctx"
    )


def test_a_flag_given_at_its_default_still_wins(folder, monkeypatch):
    """``-j 4`` (BenchFlow's default) over a file's ``concurrency: 8``."""
    (folder / "run.yaml").write_text(
        "tasks_dir: tasks\nagent: claude-agent-acp\nmodel: claude-haiku-5-5\n"
        "concurrency: 8\nintegrity: audit\n"
    )
    plain = main._config_file_evaluation(
        _plan(monkeypatch, "run", "--config", "run.yaml")
    )._config
    assert (plain.concurrency, plain.integrity) == (8, "audit")
    configured = main._config_file_evaluation(
        _plan(
            monkeypatch, "run", "--config", "run.yaml", "-j", "4", "--integrity", "off"
        )
    )._config
    assert (configured.concurrency, configured.integrity) == (4, "off")


def test_retry_attempts_reach_a_run_config(folder, monkeypatch):
    configured = main._config_file_evaluation(
        _plan(monkeypatch, "run", "--config", "run.yaml", "--retry-attempts", "5")
    )._config
    assert configured.retry.max_retries == 5
