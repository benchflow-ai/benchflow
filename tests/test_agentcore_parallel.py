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
            patch.object(
                AgentCoreSandbox,
                "_resolve_image_digest",
                lambda self, tag: f"reg/repo@sha256:{tag}",
            ),
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
            patch.object(
                AgentCoreSandbox,
                "_resolve_image_digest",
                lambda self, tag: f"reg/repo@sha256:{tag}",
            ),
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
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
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
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
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
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
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


class TestAwsResponseShapeConformance:
    """Pin the mocked shapes to the SDK's real ones.

    Every fixture in this file invents AWS responses. When an invented shape
    drifts from the service model the tests keep passing while production
    breaks — which is exactly how the reaper came to read a ``createdAt`` that
    ``ListAgentRuntimes`` never returns, and so deleted fresh runtimes.
    """

    def _shape(self, operation, member=None):
        boto3 = pytest.importorskip("boto3")
        client = boto3.Session(region_name="us-west-2").client(
            "bedrock-agentcore-control",
            aws_access_key_id="x",
            aws_secret_access_key="y",
        )
        shape = client.meta.service_model.operation_model(operation).output_shape
        return shape.members[member].member if member else shape

    def test_list_agent_runtimes_has_no_created_at(self):
        """The field the reaper originally keyed on does not exist here."""
        members = self._shape("ListAgentRuntimes", "agentRuntimes").members

        assert "lastUpdatedAt" in members
        assert "createdAt" not in members

    def test_get_agent_runtime_carries_the_bound_image(self):
        """The adoption check reads this to refuse a mismatched runtime."""
        members = self._shape("GetAgentRuntime").members

        assert "agentRuntimeArtifact" in members
        container = members["agentRuntimeArtifact"].members["containerConfiguration"]
        assert "containerUri" in container.members


class TestReaperAgeHandling:
    def _control(self, runtimes, tags):
        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"agentRuntimes": runtimes}]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.side_effect = lambda resourceArn: {
            "tags": tags.get(resourceArn, {})
        }
        return control

    def _managed(self, arn, *, leased_until=None):
        """Managed tags. Default lease is expired, i.e. reapable."""
        expiry = leased_until or (datetime.now(UTC) - timedelta(hours=1))
        return {
            arn: {
                provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                provisioning.LEASE_TAG: expiry.isoformat(),
            }
        }

    def test_fresh_runtime_in_the_real_list_shape_is_kept(self):
        """Reproduces the reported P1 with the shape AWS actually returns.

        ``ListAgentRuntimes`` has only ``lastUpdatedAt``; keying on
        ``createdAt`` yielded None, skipped the age comparison, and selected a
        minutes-old runtime for deletion under a one-day policy.
        """
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "agentRuntimeName": "bf_x",
                "lastUpdatedAt": datetime.now(UTC) - timedelta(minutes=3),
                "status": "READY",
            }
        ]
        control = self._control(runtimes, self._managed("arn:mine"))

        report = reap_stale_runtimes(control, max_age_minutes=1440, dry_run=True)

        assert report.deleted == []
        assert report.skipped_recent == 1

    def test_runtime_with_no_timestamp_is_kept(self):
        """Unknown age must never be read as 'old enough to delete'."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [{"agentRuntimeArn": "arn:mine", "agentRuntimeId": "mine"}],
            self._managed("arn:mine"),
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == []
        assert report.skipped_recent == 1
        control.delete_agent_runtime.assert_not_called()

    def test_genuinely_old_runtime_in_the_real_shape_is_reaped(self):
        """The positive control: cleanup must still do its job."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [
                {
                    "agentRuntimeArn": "arn:mine",
                    "agentRuntimeId": "mine",
                    "lastUpdatedAt": datetime.now(UTC) - timedelta(days=3),
                }
            ],
            self._managed("arn:mine"),
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == ["mine"]


class TestRuntimeImageBinding:
    @pytest.mark.asyncio
    async def test_a_runtime_bound_to_another_image_is_updated(
        self, tmp_path, monkeypatch
    ):
        """Adopting a stale runtime would run the agent in the wrong environment."""
        from botocore.exceptions import ClientError

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ConflictException", "Message": "exists"}},
            "CreateAgentRuntime",
        )
        control.get_agent_runtime.return_value = {
            "status": "READY",
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "new.example/repo@sha256:b"}
            },
            "lifecycleConfiguration": env._lifecycle_configuration(),
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }

        with (
            patch.object(env, "_client", return_value=control),
            patch.object(
                provisioning,
                "find_runtime_by_name",
                return_value=("arn:rt", "rt-1", "old.example/repo@sha256:a"),
            ),
        ):
            arn, _rid = await env._create_or_adopt_runtime(
                "bf_x", "new.example/repo@sha256:b"
            )

        assert arn == "arn:rt"
        control.update_agent_runtime.assert_called_once()
        sent = control.update_agent_runtime.call_args.kwargs
        assert (
            sent["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"]
            == "new.example/repo@sha256:b"
        )

    def test_adoption_fails_closed_when_the_image_still_mismatches(self):
        """If the update did not take, refuse rather than run the wrong image."""
        from benchflow.sandbox.agentcore import AgentCoreSandbox
        from benchflow.sandbox.protocol import SandboxStartupError

        control = MagicMock()
        control.get_agent_runtime.return_value = {
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "old.example/repo@sha256:a"}
            },
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 900,
                "maxLifetime": 28800,
            },
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }

        with pytest.raises(SandboxStartupError, match="wrong image"):
            AgentCoreSandbox._verify_adopted_runtime(
                control,
                "rt-1",
                "new.example/repo@sha256:b",
                {"idleRuntimeSessionTimeout": 900, "maxLifetime": 28800},
                "arn:aws:iam::1:role/rt",
                {"networkMode": "PUBLIC"},
                {"serverProtocol": "HTTP"},
            )

    def test_adoption_fails_closed_when_lifecycle_drifted(self):
        """An adopted runtime on service defaults reclaims sessions early.

        Live-observed: an update that omits lifecycleConfiguration silently
        resets a configured 600/7200 window to the 900/28800 defaults.
        """
        from benchflow.sandbox.agentcore import AgentCoreSandbox
        from benchflow.sandbox.protocol import SandboxStartupError

        control = MagicMock()
        control.get_agent_runtime.return_value = {
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "repo@sha256:a"}
            },
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 900,
                "maxLifetime": 28800,
            },
        }

        with pytest.raises(SandboxStartupError, match="does not match"):
            AgentCoreSandbox._verify_adopted_runtime(
                control,
                "rt-1",
                "repo@sha256:a",
                {"idleRuntimeSessionTimeout": 600, "maxLifetime": 7200},
                "arn:aws:iam::1:role/rt",
                {"networkMode": "PUBLIC"},
                {"serverProtocol": "HTTP"},
            )

    @pytest.mark.asyncio
    async def test_update_preserves_the_configured_lifecycle(
        self, tmp_path, monkeypatch
    ):
        """The rebind must carry lifecycle, or AWS resets it to defaults."""
        from botocore.exceptions import ClientError

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_IDLE_TIMEOUT_SEC", "600")
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_MAX_LIFETIME_SEC", "7200")
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ConflictException", "Message": "exists"}},
            "CreateAgentRuntime",
        )
        control.get_agent_runtime.return_value = {
            "status": "READY",
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "repo@sha256:new"}
            },
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 600,
                "maxLifetime": 7200,
            },
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }

        with (
            patch.object(env, "_client", return_value=control),
            patch.object(
                provisioning,
                "find_runtime_by_name",
                return_value=("arn:rt", "rt-1", "repo@sha256:old"),
            ),
        ):
            await env._create_or_adopt_runtime("bf_x", "repo@sha256:new")

        sent = control.update_agent_runtime.call_args.kwargs
        assert sent["lifecycleConfiguration"] == {
            "idleRuntimeSessionTimeout": 600,
            "maxLifetime": 7200,
        }


class TestComposeRejection:
    def test_compose_task_is_refused_at_construction(self, tmp_path):
        """One container cannot host a task's side services."""
        task_dir = _make_task(tmp_path)
        (task_dir / "docker-compose.yaml").write_text("services:\n  target: {}\n")

        with pytest.raises(ValueError, match="single-container"):
            _sandbox(task_dir)

    def test_compose_task_is_refused_by_the_capability_gate(self, tmp_path):
        """Fail during planning, before any image is built."""
        from benchflow.task.config import TaskConfig
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM scratch\n")
        (tmp_path / "environment" / "compose.yaml").write_text("services: {}\n")

        issues = validate_task_runtime_support(
            TaskConfig.model_validate({}), sandbox="agentcore", task_dir=tmp_path
        )

        assert any("compose" in issue.reason for issue in issues)

    def test_docker_still_accepts_compose_tasks(self, tmp_path):
        """The gate must not regress the backends that do support compose."""
        from benchflow.task.config import TaskConfig
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "docker-compose.yaml").write_text("services: {}\n")

        issues = validate_task_runtime_support(
            TaskConfig.model_validate({}), sandbox="docker", task_dir=tmp_path
        )

        assert not any("compose" in issue.reason for issue in issues)


class TestLeaseProtectsActiveRuntimes:
    """Runtime age is not a session-activity signal.

    Session traffic does not move a runtime's ``lastUpdatedAt``, and there is
    no API that enumerates a runtime's active sessions (``ListSessions`` is
    Memory-scoped). An old runtime serving a matrix right now is therefore
    indistinguishable from an idle one by age alone — which is how cleanup
    selected a runtime whose session was mid-command. The lease is the explicit
    contract that closes that gap.
    """

    def _control(self, tags):
        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "agentRuntimes": [
                    {
                        "agentRuntimeArn": "arn:mine",
                        "agentRuntimeId": "mine",
                        # Deliberately ancient: age alone would select it.
                        "lastUpdatedAt": datetime.now(UTC) - timedelta(days=30),
                    }
                ]
            }
        ]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.return_value = {"tags": tags}
        return control

    def _managed(self, **extra):
        return {provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE, **extra}

    def test_an_old_but_leased_runtime_is_not_deleted(self):
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        future = (datetime.now(UTC) + timedelta(hours=4)).isoformat()
        control = self._control(self._managed(**{provisioning.LEASE_TAG: future}))

        report = reap_stale_runtimes(control, max_age_minutes=0)

        assert report.deleted == []
        assert report.skipped_active == 1
        control.delete_agent_runtime.assert_not_called()

    def test_an_expired_lease_allows_reaping(self):
        """Positive control: the lease must not block cleanup forever."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        past = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        control = self._control(self._managed(**{provisioning.LEASE_TAG: past}))

        report = reap_stale_runtimes(control, max_age_minutes=0)

        assert report.deleted == ["mine"]

    def test_an_unparseable_lease_is_treated_as_active(self):
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(self._managed(**{provisioning.LEASE_TAG: "soon"}))

        report = reap_stale_runtimes(control, max_age_minutes=0)

        assert report.deleted == []
        assert report.skipped_active == 1

    def test_unreadable_tags_are_never_deleted(self):
        """No tags means no proof of anything, including that it is ours."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control({})
        control.list_tags_for_resource.side_effect = RuntimeError("denied")

        report = reap_stale_runtimes(control, max_age_minutes=0)

        assert report.deleted == []
        assert report.skipped_unmanaged == 1

    def test_negative_age_is_rejected(self):
        """A sign mistake must not reach delete_agent_runtime."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(self._managed())

        with pytest.raises(ValueError, match="must be >= 0"):
            reap_stale_runtimes(control, max_age_minutes=-1)

        control.delete_agent_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_provisioning_writes_a_lease(self, tmp_path, monkeypatch):
        """Without this the reaper has no activity signal at all."""
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.return_value = {
            "agentRuntimeId": "rt-1",
            "agentRuntimeArn": "arn:rt",
        }
        control.get_agent_runtime.return_value = {"status": "READY"}

        with patch.object(env, "_client", return_value=control):
            await env._create_or_adopt_runtime("bf_x", "repo@sha256:a")

        tags = control.tag_resource.call_args.kwargs["tags"]
        assert provisioning.LEASE_TAG in tags
        assert datetime.fromisoformat(tags[provisioning.LEASE_TAG]) > datetime.now(UTC)


class TestLeaseIntegrity:
    """A lease is only protection if it is written, kept, and refreshed."""

    @pytest.mark.asyncio
    async def test_a_failed_lease_write_aborts_the_launch(self, tmp_path, monkeypatch):
        """Swallowing it starts a session on a runtime cleanup may delete."""
        from benchflow.sandbox.protocol import SandboxStartupError

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.return_value = {
            "agentRuntimeId": "rt-1",
            "agentRuntimeArn": "arn:rt",
        }
        control.tag_resource.side_effect = RuntimeError("AccessDenied")

        with (
            patch.object(env, "_client", return_value=control),
            pytest.raises(SandboxStartupError, match="lease"),
        ):
            await env._create_or_adopt_runtime("bf_x", "repo@sha256:a")

    def test_a_managed_runtime_without_a_lease_is_not_deleted(self):
        """Every provisioned runtime is leased, so an unleased one is unexplained."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "agentRuntimes": [
                    {
                        "agentRuntimeArn": "arn:mine",
                        "agentRuntimeId": "mine",
                        "lastUpdatedAt": datetime.now(UTC) - timedelta(days=30),
                    }
                ]
            }
        ]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.return_value = {
            "tags": {provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE}
        }

        report = reap_stale_runtimes(control, max_age_minutes=0)

        assert report.deleted == []
        assert report.skipped_active == 1

    def test_renewal_is_due_again_after_the_throttle_window(self):
        """Cache-hit rollouts must refresh, or a long matrix outlives its lease."""
        window = 7200.0
        arn = "arn:renew"

        assert provisioning.lease_needs_renewal(arn, window, 0.0) is True
        assert provisioning.lease_needs_renewal(arn, window, 10.0) is False
        assert provisioning.lease_needs_renewal(arn, window, window) is True

    @pytest.mark.asyncio
    async def test_every_rollout_renews_when_due(self, tmp_path, monkeypatch):
        """Provisioning is memoized, so renewal cannot live only in creation."""
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        env.runtime_arn = "arn:rt-renew-probe"
        control = MagicMock()

        with patch.object(env, "_client", return_value=control):
            await env._renew_lease()

        control.tag_resource.assert_called_once()
        assert provisioning.LEASE_TAG in control.tag_resource.call_args.kwargs["tags"]


class TestAdoptionContract:
    def _detail(self, **overrides):
        detail = {
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "repo@sha256:a"}
            },
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 600,
                "maxLifetime": 7200,
            },
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }
        detail.update(overrides)
        return detail

    def _verify(self, detail):
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        control = MagicMock()
        control.get_agent_runtime.return_value = detail
        AgentCoreSandbox._verify_adopted_runtime(
            control,
            "rt-1",
            "repo@sha256:a",
            {"idleRuntimeSessionTimeout": 600, "maxLifetime": 7200},
            "arn:aws:iam::1:role/rt",
            {"networkMode": "PUBLIC"},
            {"serverProtocol": "HTTP"},
        )

    def test_a_fully_matching_runtime_is_accepted(self):
        self._verify(self._detail())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("roleArn", "arn:aws:iam::1:role/someone-else"),
            ("networkConfiguration", {"networkMode": "VPC"}),
            ("protocolConfiguration", {"serverProtocol": "MCP"}),
        ],
    )
    def test_a_differing_contract_is_refused(self, field, value):
        """Right image, wrong contract: wrong permissions, egress, or shell."""
        from benchflow.sandbox.protocol import SandboxStartupError

        with pytest.raises(SandboxStartupError, match="does not match"):
            self._verify(self._detail(**{field: value}))


class TestDeprecatedCleanupAliasIsSafe:
    """`bench environment cleanup` reaches the same destructive code.

    Guarding only the new command left the deprecated alias able to select
    STARTED sandboxes for deletion via a negative age.
    """

    def test_negative_age_is_rejected_by_the_daytona_reaper(self):
        from benchflow.sandbox.daytona_reaper import reap_stale_sandboxes

        client = MagicMock()

        with pytest.raises(ValueError, match="must be >= 0"):
            reap_stale_sandboxes(client, max_age_minutes=-1, dry_run=True)

        client.delete.assert_not_called()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_age_minutes": -1},
            {"failed_max_age_minutes": -1},
            {"min_idle_minutes": -1},
        ],
    )
    def test_every_age_knob_rejects_negatives(self, kwargs):
        from benchflow.sandbox.daytona_reaper import reap_stale_sandboxes

        with pytest.raises(ValueError, match="must be >= 0"):
            reap_stale_sandboxes(MagicMock(), dry_run=True, **kwargs)

    @pytest.mark.parametrize("command", ["sandbox", "environment"])
    def test_cli_rejects_negative_max_age(self, command):
        """Both the current command and the deprecated alias must refuse it."""
        from typer.testing import CliRunner

        from benchflow.cli.main import app

        result = CliRunner().invoke(app, [command, "cleanup", "--max-age", "-1"])

        assert result.exit_code != 0
