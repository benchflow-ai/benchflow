"""The lead's ruling: a service the task declares for BenchFlow to start
in main (environment.toml's ``[[environment.services]]``) cannot run beside a
model agent under a restricted network policy. The policy holds only the
agent's uid; the service runs as root outside it, and the agent reaches it on
localhost. Such a trial is refused before any sandbox starts, unscored and
never retried, naming the service and the policy and saying what to do."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import UNSUPPORTED_TASK_FEATURE, classify_error
from benchflow.task import Task
from benchflow.task.runtime_capabilities import (
    ENVIRONMENT_SERVICES_ISSUE_PATH,
    UnsupportedTaskFeatureError,
    environment_services_issue,
)

MANIFEST = """\
[environment]
name           = "with-db"
base_image     = "ubuntu:24.04"
owns_lifecycle = false
isolation      = "per_task"

[[environment.services]]
name    = "db"
command = "python3 -m http.server 5432"
port    = 5432
"""

# Each policy's task.md lines: (agent's, sandbox's).
POLICIES = {
    "allowlist": (
        "  network_mode: allowlist\n  allowed_hosts:\n    - example.com\n",
        "",
    ),
    "denylist": (
        "",
        "  allow_internet: true\n  network_mode: denylist\n"
        "  blocked_hosts:\n    - example.org\n",
    ),
    "agent no-network": ("  network_mode: no-network\n", ""),
    "open": ("", ""),
    "sandbox no-network": ("", "  network_mode: no-network\n"),
}


def _task(root: Path, policy: str, *, services: bool = True) -> Path:
    task = root / policy.replace(" ", "-")
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    agent, sandbox = POLICIES[policy]
    (task / "task.md").write_text(
        '---\nschema_version: "1.3"\n'
        f"agent:\n  timeout_sec: 60\n{agent}"
        "verifier:\n  timeout_sec: 60\n"
        f"sandbox:\n  cpus: 1\n{sandbox}"
        "---\nSay hello.\n"
    )
    if services:
        (task / "environment.toml").write_text(MANIFEST)
    return task


def _issue(task: Path, *, agent_launches: bool = True):
    return environment_services_issue(
        Task(task).config,
        sandbox="docker",
        agent_launches=agent_launches,
        task_dir=task,
    )


@pytest.mark.parametrize(
    ("policy", "named"),
    [
        ("allowlist", "allowlist"),
        ("denylist", "denylist"),
        ("agent no-network", "no-network, with a model agent"),
    ],
)
def test_a_declared_service_beside_a_held_model_agent_is_refused(
    tmp_path, policy, named
):
    issue = _issue(_task(tmp_path, policy))
    assert issue is not None
    assert issue.path == ENVIRONMENT_SERVICES_ISSUE_PATH
    assert (
        "the service 'db' in environment.toml at the task's root ([[environment.services]])"
        in (issue.reason)
    )
    assert f"the agent's network policy is {named}" in issue.reason
    # What to do.
    assert "as a compose service on an internal network that main shares" in (
        issue.reason
    )


@pytest.mark.parametrize("policy", ["open", "sandbox no-network"])
def test_an_open_or_wholly_offline_sandbox_is_not_refused(tmp_path, policy):
    """An open policy holds nothing; a sandbox with no network at all gives
    the service none either (the launch's quiet-main check leaves it too)."""
    assert _issue(_task(tmp_path, policy)) is None


def test_the_oracle_or_a_script_is_not_refused(tmp_path):
    assert _issue(_task(tmp_path, "allowlist"), agent_launches=False) is None


def test_a_task_without_declared_services_is_not_refused(tmp_path):
    assert _issue(_task(tmp_path, "allowlist", services=False)) is None


def test_the_run_is_refused_before_any_sandbox_unscored_and_never_retried(tmp_path):
    """The rollout's preflight (setup and the dry run, before any sandbox)
    raises it; the trial is unsupported_task_feature, never retried."""
    from benchflow.evaluation import RetryConfig
    from benchflow.rollout import preflight_rollout_network_policy

    task = _task(tmp_path, "allowlist")
    cfg = SimpleNamespace(
        environment="docker",
        primary_agent="claude-agent-acp",
        task_path=task,
        environment_manifest=None,
    )
    with pytest.raises(UnsupportedTaskFeatureError) as raised:
        preflight_rollout_network_policy(Task(task), cfg)
    error = f"UnsupportedTaskFeatureError: {raised.value}"
    assert "the service 'db'" in error
    assert classify_error(error) == UNSUPPORTED_TASK_FEATURE
    assert RetryConfig().should_retry(error) is False
    # The oracle runs no model agent: not refused.
    cfg.primary_agent = "oracle"
    preflight_rollout_network_policy(Task(task), cfg)


def test_the_manifest_task_md_names_is_read(tmp_path):
    """task.md's benchflow.environment.manifest names the manifest (the
    deterministic task's shape); its services are refused alike."""
    task = _task(tmp_path, "allowlist", services=False)
    (task / "env-plane.toml").write_text(MANIFEST)
    text = (task / "task.md").read_text()
    (task / "task.md").write_text(
        text.replace(
            "---\nSay hello.",
            "benchflow:\n  environment:\n    manifest: env-plane.toml\n---\nSay hello.",
        )
    )
    issue = _issue(task)
    assert issue is not None and "the service 'db'" in issue.reason
