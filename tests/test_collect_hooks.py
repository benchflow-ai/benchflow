"""The task.toml format's ``[[verifier.collect]]`` hooks and sidecar artifacts.

Ten Terminal-Bench 4.1 tasks were refused at load: they snapshot runtime state
with collect hooks (shadow-relay writes its workspace patch to
/tmp/agent.patch) or read artifacts from a compose service other than main
(ctr-optimization's /shared/verify_snapshot.json from ``api``). Hooks for main
run before the freeze, the others' before their artifacts are read; a sidecar
artifact crosses into the separate verifier at its source path when its bytes
still match the list.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout import _collect_hooks as hooks
from benchflow.task import Task


def _task(root: Path, toml: str) -> Task:
    (root / "environment").mkdir(parents=True)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text("echo 1\n")
    (root / "instruction.md").write_text("Do it.\n")
    (root / "task.toml").write_text(textwrap.dedent(toml))
    return Task(root)


TOML = """
    artifacts = [{ source = "/shared/verify_snapshot.json", service = "api" }, "/tmp/agent.patch"]
    [verifier]
    environment_mode = "separate"
    [[verifier.collect]]
    command = "git diff > /tmp/agent.patch"
    [[verifier.collect]]
    service = "api"
    command = "curl -sf http://localhost:5000/healthz"
    timeout_sec = 10.0
    user = "app"
"""


def test_the_tasks_load():
    import tempfile

    task = _task(Path(tempfile.mkdtemp()) / "t", TOML)
    assert [
        (h.service, h.timeout_sec, h.user) for h in task.config.verifier.collect
    ] == [
        ("main", 60.0, None),
        ("api", 10.0, "app"),
    ]
    assert [a.source for a in hooks.sidecar_artifacts(task)] == [
        "/shared/verify_snapshot.json"
    ]


def test_a_sidecar_artifact_needs_an_absolute_source(tmp_path):
    with pytest.raises(Exception, match="must be an absolute path"):
        _task(
            tmp_path / "t", 'artifacts = [{ source = "rel.json", service = "api" }]\n'
        )


def _rollout(tmp_path: Path, exec_result) -> SimpleNamespace:
    env = MagicMock()
    env.exec = AsyncMock(side_effect=exec_result)
    rollout = SimpleNamespace(
        _env=env,
        _task=_task(tmp_path / "task", TOML),
        _require_rollout_dir=lambda: tmp_path / "trial",
    )
    (tmp_path / "trial").mkdir()
    return rollout


@pytest.mark.asyncio
async def test_main_hooks_run_once_before_the_others(tmp_path):
    calls = []

    async def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            return_code=0 if "git" in command else 7, stdout="", stderr="no api"
        )

    rollout = _rollout(tmp_path, run)
    await hooks.run_collect_hooks(rollout, main=True)
    await hooks.run_collect_hooks(rollout, main=True)  # once
    await hooks.run_collect_hooks(rollout, main=False)
    assert [c[0] for c in calls] == [
        "git diff > /tmp/agent.patch",
        "curl -sf http://localhost:5000/healthz",
    ]
    assert calls[0][1]["service"] == "main" and "user" not in calls[0][1]
    assert calls[1][1] == {"service": "api", "timeout_sec": 10, "user": "app"}
    record = json.loads((tmp_path / "trial" / hooks.HOOKS_RECORD).read_text())
    assert [r["return_code"] for r in record] == [
        0,
        7,
    ]  # a failed hook is recorded, not raised


@pytest.mark.asyncio
async def test_a_sidecar_artifact_is_read_from_its_service_and_listed(tmp_path):
    async def run(command, **kwargs):
        if "mktemp" in command:
            assert kwargs["service"] == "api" and kwargs["user"] == "root"
            return SimpleNamespace(return_code=0, stdout="/tmp/tmp.abc\n", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    rollout = _rollout(tmp_path, run)

    async def download_dir(source, target, service):
        assert (source, service) == ("/tmp/tmp.abc", "api")
        (Path(target) / "verify_snapshot.json").write_text('{"ctr": 0.4}')

    rollout._env.download_dir = AsyncMock(side_effect=download_dir)
    entries = await hooks.collect_sidecar_artifacts(rollout)
    (item,) = entries[0]["files"]
    assert entries[0]["status"] == "collected"
    assert item["path"] == "/shared/verify_snapshot.json" and item["size"] == 12
    local = tmp_path / "trial" / "artifacts" / "services" / item["local"]
    assert local.read_text() == '{"ctr": 0.4}'


@pytest.mark.asyncio
async def test_sidecar_files_cross_only_unchanged_and_never_into_verifier_folders(
    tmp_path, monkeypatch
):
    root = tmp_path / "trial" / "artifacts" / "services"
    (root / "api" / "shared").mkdir(parents=True)
    (root / "api" / "shared" / "a.json").write_text("{}")
    (root / "api" / "tests").mkdir()
    (root / "api" / "tests" / "x").write_text("x")
    good = hooks._sha256(tmp_path / "trial", root / "api" / "shared" / "a.json")
    (root / hooks.SIDECAR_MANIFEST).write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "files": [
                            {
                                "path": "/shared/a.json",
                                "local": "api/shared/a.json",
                                "sha256": good,
                                "mode": 0o644,
                            },
                            {
                                "path": "/tests/x",
                                "local": "api/tests/x",
                                "sha256": "0",
                                "mode": 0o644,
                            },
                            {
                                "path": "/shared/b.json",
                                "local": "api/shared/a.json",
                                "sha256": "f" * 64,
                                "mode": 0o644,
                            },
                        ]
                    }
                ]
            }
        )
    )
    env = MagicMock()
    env.exec = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
    )
    env.upload_file = AsyncMock()
    record: dict = {}
    import benchflow.trust.state as trust

    admitted = []
    monkeypatch.setattr(trust, "admit", lambda *a, **k: admitted.append(a[2]) or "ok")
    await hooks.install_sidecar_artifacts(env, tmp_path / "trial", record)
    # Each file crosses with the separate verifier transfer's admission.
    assert admitted == [trust.IngressKind.SEPARATE_VERIFIER_TRANSFER]
    assert env.upload_file.call_args.kwargs["admission"] == "ok"
    assert [c.args[1] for c in env.upload_file.call_args_list] == ["/shared/a.json"]
    assert record["sidecar_artifacts"]["refused"] == [
        {"path": "/tests/x", "reason": "verifier-owned"},
        {"path": "/shared/b.json", "reason": "changed since collection"},
    ]


def _with_agent(rollout: SimpleNamespace, agent: str, sandbox_user: str | None):
    rollout._config = SimpleNamespace(
        primary_agent=agent,
        sandbox_user=sandbox_user,
        purpose="task",
        freeze_workspace=False,
    )
    return rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "sandbox_user", "expected"),
    [
        ("claude-agent-acp", "agent", "agent"),
        ("nop", "agent", "agent"),
        ("script:attempt.sh", "agent", "agent"),
        # solve.sh runs as root (or the task's [agent].user), as before.
        ("oracle", "agent", None),
        # --sandbox-user none: the agent ran as root.
        ("claude-agent-acp", None, None),
    ],
)
async def test_a_main_hook_runs_as_the_account_the_agent_ran_as(
    tmp_path, agent, sandbox_user, expected
):
    """Guards this fix against the regression introduced by PR #84 (8e0630a9):
    a main hook ran as root with git trust over the agent's repositories, so
    the agent's repository config (core.fsmonitor, a filter driver) ran as
    root. It runs as the workspace's owner, its command as the task wrote it,
    and an explicit ``user`` (here the api hook's) is kept."""
    calls = []

    async def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    rollout = _with_agent(_rollout(tmp_path, run), agent, sandbox_user)
    rollout._agent_cwd = "/app"
    assert await hooks.run_collect_hooks(rollout, main=True) is True
    assert await hooks.run_collect_hooks(rollout, main=True) is False  # once
    await hooks.run_collect_hooks(rollout, main=False)
    (main_command, main_kwargs), (_, api_kwargs) = calls
    assert main_command == "git diff > /tmp/agent.patch"
    assert main_kwargs.get("user") == expected
    assert api_kwargs["user"] == "app"
    record = json.loads((tmp_path / "trial" / hooks.HOOKS_RECORD).read_text())
    assert [r.get("user") for r in record] == [expected, "app"]


@pytest.mark.asyncio
async def test_a_main_hook_that_names_root_gets_no_git_trust(tmp_path):
    """Guards this fix against PR #84 (8e0630a9): no hook gets trust written
    for it; one that names root runs exactly its command as root."""
    calls = []

    async def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    rollout = _rollout(tmp_path, run)
    rollout._task.config.verifier.collect[0].user = "root"
    _with_agent(rollout, "claude-agent-acp", "agent")._agent_cwd = "/app"
    await hooks.run_collect_hooks(rollout, main=True)
    ((command, kwargs),) = calls
    assert (command, kwargs["user"]) == ("git diff > /tmp/agent.patch", "root")
    assert "safe.directory" not in command and "GIT_CONFIG" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize("with_hooks", [True, False])
async def test_the_agents_processes_are_stopped_again_after_its_hooks(
    tmp_path, monkeypatch, with_hooks
):
    """Guards this fix against PR #84 (8e0630a9): what a hook run as the
    agent's account leaves running there is stopped before the capture, and
    the kill adds to the turn-end record."""
    from benchflow._utils import kill_record
    from benchflow.rollout import _review

    order = []

    async def run(command, **kwargs):
        order.append(("hook", kwargs.get("user")))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def quiesce(env, user, **kwargs):
        order.append(("kill", user))
        kill_record.record_kill_before_grading(user, 1)

    rollout = _with_agent(_rollout(tmp_path, run), "claude-agent-acp", "agent")
    if not with_hooks:
        rollout._task.config.verifier.collect = []
    rollout._agent_cwd = "/app"
    rollout._branch_child_active = False
    rollout._review_plan = None
    rollout._kill_record = None
    rollout._planes = SimpleNamespace(quiesce_agent=quiesce)
    rollout.disconnect = AsyncMock()
    monkeypatch.setattr(_review, "_freeze_requested", lambda _: False)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.recovery_ineligible_reason",
        lambda _: "no recovery contract",
    )
    monkeypatch.setattr("benchflow.task.keep_services.trial_keep", lambda *a: None)
    await _review.capture_terminal_workspace(rollout)
    kills = 2 if with_hooks else 1
    assert order == (
        [("kill", "agent"), ("hook", "agent"), ("kill", "agent")]
        if with_hooks
        else [("kill", "agent")]
    )
    record = json.loads((tmp_path / "trial" / "kill-before-grading.json").read_text())
    assert record["killed_total"] == record["quiesce_killed"] == kills
