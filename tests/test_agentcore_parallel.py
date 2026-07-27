"""Parallel-safety of AgentCore image and runtime provisioning.

A rollout is a *session*; the image and the runtime behind it are shared. That
is not an optimization — the account quotas are 5000 concurrent sessions
against 100 total runtimes with ``CreateAgentRuntime`` at 5/s, so anything
that scales with rollouts instead of with distinct images cannot run a matrix.

These tests pin the three properties that make a fan-out safe:

* concurrent rollouts of one task build, push, and register **once**;
* runtime identity follows the image's content, not the task name, so repeated
  trials share a runtime instead of racing to create and delete one;
* ending a rollout stops its session and leaves the shared runtime alone.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.task.config import SandboxConfig


@pytest.fixture(autouse=True)
def _clean_provisioning_cache():
    provisioning.reset_cache()
    yield
    provisioning.reset_cache()


def _make_task(tmp_path, name="demo-task", dockerfile="FROM python:3.12-slim\n"):
    task_dir = tmp_path / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "Dockerfile").write_text(dockerfile)
    return task_dir


def _sandbox(task_dir, *, name="demo-task", session="run-1"):
    env = AgentCoreSandbox(
        environment_dir=task_dir,
        environment_name=name,
        session_id=session,
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    env._account_id = "123456789012"
    return env


class TestImageIdentity:
    def test_identity_is_stable_across_copies_of_the_same_task(self, tmp_path):
        """BenchFlow copies tasks to temp dirs; identity must survive that.

        A digest that folded in paths or mtimes would change every run and
        defeat image reuse entirely.
        """
        a = _make_task(tmp_path / "first")
        b = _make_task(tmp_path / "second")

        assert _sandbox(a)._image_identity() == _sandbox(b)._image_identity()

    def test_identity_changes_when_the_environment_changes(self, tmp_path):
        a = _make_task(tmp_path / "a")
        b = _make_task(tmp_path / "b", dockerfile="FROM python:3.13-slim\n")

        assert _sandbox(a)._image_identity() != _sandbox(b)._image_identity()

    def test_identity_changes_when_a_context_file_changes(self, tmp_path):
        """Skills baked into the image must produce a distinct image."""
        a = _make_task(tmp_path / "a")
        b = _make_task(tmp_path / "b")
        (b / "skills").mkdir()
        (b / "skills" / "SKILL.md").write_text("# a skill")

        assert _sandbox(a)._image_identity() != _sandbox(b)._image_identity()

    def test_runtime_name_follows_the_image_not_the_task_name(self, tmp_path):
        """Guards the delete-out-from-under-you bug in the first AgentCore cut.

        Naming the runtime after the task meant concurrent trials raced to
        create one runtime and the first to finish deleted it mid-run.
        """
        digest_a = "a" * 64
        digest_b = "b" * 64

        same = provisioning.runtime_name("demo", digest_a)
        again = provisioning.runtime_name("demo", digest_a)
        different = provisioning.runtime_name("demo", digest_b)

        assert same == again
        assert same != different
        assert same[0].isalpha()
        assert len(same) <= 48


class TestSingleFlight:
    @pytest.mark.asyncio
    async def test_concurrent_rollouts_build_and_push_once(self, tmp_path):
        """20 concurrent rollouts of one task must not run 20 builds."""
        task_dir = _make_task(tmp_path)
        builds = {"n": 0}

        async def _build(self, image_uri, *, force_build):
            builds["n"] += 1
            await asyncio.sleep(0.01)

        with (
            patch.object(AgentCoreSandbox, "_ensure_ecr_repository"),
            patch.object(AgentCoreSandbox, "_ecr_image_exists", return_value=False),
            patch.object(AgentCoreSandbox, "_build_and_push", new=_build),
        ):
            sandboxes = [_sandbox(task_dir, session=f"run-{i}") for i in range(20)]
            uris = await asyncio.gather(
                *(s._publish_image(force_build=False) for s in sandboxes)
            )

        assert builds["n"] == 1
        assert len(set(uris)) == 1

    @pytest.mark.asyncio
    async def test_concurrent_rollouts_register_one_runtime(self, tmp_path):
        """CreateAgentRuntime is a 5/s quota — it cannot run per rollout."""
        task_dir = _make_task(tmp_path)
        creates = {"n": 0}

        async def _create(self, name, image_uri):
            creates["n"] += 1
            await asyncio.sleep(0.01)
            return f"arn:aws:bedrock-agentcore:us-west-2:1:runtime/{name}", "rt-1"

        with patch.object(AgentCoreSandbox, "_create_or_adopt_runtime", new=_create):
            sandboxes = [_sandbox(task_dir, session=f"run-{i}") for i in range(20)]
            arns = await asyncio.gather(
                *(s._ensure_runtime("img:tag") for s in sandboxes)
            )

        assert creates["n"] == 1
        assert len(set(arns)) == 1

    @pytest.mark.asyncio
    async def test_a_failed_build_is_not_cached(self, tmp_path):
        """A transient throttle must not poison every later rollout."""
        task_dir = _make_task(tmp_path)
        calls = {"n": 0}

        async def _flaky(self, image_uri, *, force_build):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient throttle")

        with (
            patch.object(AgentCoreSandbox, "_ensure_ecr_repository"),
            patch.object(AgentCoreSandbox, "_ecr_image_exists", return_value=False),
            patch.object(AgentCoreSandbox, "_build_and_push", new=_flaky),
        ):
            env = _sandbox(task_dir)
            with pytest.raises(RuntimeError, match="transient throttle"):
                await env._publish_image(force_build=False)
            await env._publish_image(force_build=False)

        assert calls["n"] == 2


class TestSessionTeardown:
    @pytest.mark.asyncio
    async def test_stop_ends_the_session_and_keeps_the_runtime(self, tmp_path):
        """Deleting the runtime would tear down sibling trials still running."""
        env = _sandbox(_make_task(tmp_path))
        env.runtime_arn = "arn:aws:bedrock-agentcore:us-west-2:1:runtime/shared"
        env.runtime_session_id = "s" * 40
        env._runtime_id = "shared-1"

        data = MagicMock()
        control = MagicMock()

        def _client(service):
            return control if service.endswith("-control") else data

        with patch.object(env, "_client", side_effect=_client):
            await env.stop(delete=True)

        data.stop_runtime_session.assert_called_once()
        control.delete_agent_runtime.assert_not_called()
        assert env.runtime_session_id is None


class TestImageSizeGate:
    def test_image_within_the_cap_is_accepted(self):
        assert provisioning.image_size_error(1500 * 1024 * 1024, "img") is None

    def test_oversized_image_names_the_hard_quota(self):
        """A 2 GB cap that is not adjustable deserves a message that says so."""
        message = provisioning.image_size_error(3000 * 1024 * 1024, "img:tag")

        assert message is not None
        assert "2048" in message
        assert "not" in message and "adjustable" in message
        assert "daytona" in message


class TestReaper:
    def _control(self, runtimes, tags):
        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"agentRuntimes": runtimes}]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.side_effect = lambda resourceArn: {
            "tags": tags.get(resourceArn, {})
        }
        return control

    def test_only_benchflow_managed_runtimes_are_reaped(self):
        """Never delete something another tool created in the same account."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        old = datetime.now(UTC) - timedelta(days=3)
        runtimes = [
            {"agentRuntimeArn": "arn:mine", "agentRuntimeId": "mine", "createdAt": old},
            {
                "agentRuntimeArn": "arn:other",
                "agentRuntimeId": "other",
                "createdAt": old,
            },
        ]
        control = self._control(
            runtimes,
            {"arn:mine": {provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE}},
        )

        report = reap_stale_runtimes(control, max_age_minutes=60)

        assert report.deleted == ["mine"]
        assert report.skipped_unmanaged == 1

    def test_recent_runtimes_are_kept(self):
        """A runtime from a run still in flight must survive cleanup."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "createdAt": datetime.now(UTC) - timedelta(minutes=5),
            }
        ]
        control = self._control(
            runtimes,
            {"arn:mine": {provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE}},
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == []
        assert report.skipped_recent == 1
        control.delete_agent_runtime.assert_not_called()

    def test_dry_run_reports_without_deleting(self):
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "createdAt": datetime.now(UTC) - timedelta(days=3),
            }
        ]
        control = self._control(
            runtimes,
            {"arn:mine": {provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE}},
        )

        report = reap_stale_runtimes(control, max_age_minutes=60, dry_run=True)

        assert report.deleted == ["mine"]
        control.delete_agent_runtime.assert_not_called()

    def test_unreadable_tags_fail_closed(self):
        """If tags can't be read, assume it isn't ours."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [
                {
                    "agentRuntimeArn": "arn:x",
                    "agentRuntimeId": "x",
                    "createdAt": datetime.now(UTC) - timedelta(days=3),
                }
            ],
            {},
        )
        control.list_tags_for_resource.side_effect = RuntimeError("denied")

        report = reap_stale_runtimes(control, max_age_minutes=60)

        assert report.deleted == []
        assert report.skipped_unmanaged == 1


class TestCleanupCommandGating:
    """``bench sandbox cleanup`` must not phone AWS unless AgentCore is set up."""

    def test_cleanup_is_inert_without_agentcore_configuration(self, monkeypatch):
        """A dev machine with AWS credentials must not trigger a live call.

        ``boto3`` ships with several unrelated extras, so importability is not
        evidence that this account is being used for AgentCore runs.
        """
        from benchflow.cli.sandbox import _cleanup_agentcore_runtimes

        monkeypatch.delenv("BENCHFLOW_AGENTCORE_ROLE_ARN", raising=False)
        with patch("boto3.Session") as session:
            assert (
                _cleanup_agentcore_runtimes(dry_run=True, max_age_minutes=60) is False
            )
        session.assert_not_called()

    def test_credential_failure_degrades_instead_of_crashing(self, monkeypatch):
        """Cleanup may still have Daytona work to do; don't abort the command."""
        from benchflow.cli.sandbox import _cleanup_agentcore_runtimes

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/x")
        with patch("boto3.Session", side_effect=RuntimeError("no credentials")):
            assert (
                _cleanup_agentcore_runtimes(dry_run=True, max_age_minutes=60) is False
            )

    def test_daytona_failure_does_not_block_agentcore_cleanup(self, monkeypatch):
        """One backend's broken credentials must not strand the other's resources.

        The Daytona path exits when DAYTONA_API_KEY is missing; before this was
        isolated it aborted the whole command, silently leaving AgentCore
        runtimes to accumulate against a 100-per-account quota.
        """
        import typer

        from benchflow.cli import sandbox as sandbox_cli

        calls: list[str] = []
        monkeypatch.setattr(sandbox_cli, "_daytona_sdk_available", lambda: True)
        monkeypatch.setattr(
            sandbox_cli,
            "_cleanup_agentcore_runtimes",
            lambda **kw: calls.append("agentcore") or True,
        )
        fake_main = MagicMock()
        fake_main._cleanup_daytona_sandboxes.side_effect = typer.Exit(1)
        monkeypatch.setitem(__import__("sys").modules, "benchflow.cli.main", fake_main)

        sandbox_cli.sandbox_cleanup(dry_run=True, max_age_minutes=60)

        assert calls == ["agentcore"]
