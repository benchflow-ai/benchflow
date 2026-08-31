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
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.sandbox.agentcore_image import AgentCoreImagePublisher
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
    env._images._account_id = "123456789012"
    return env


class TestImageIdentity:
    def test_identity_is_stable_across_copies_of_the_same_task(self, tmp_path):
        """BenchFlow copies tasks to temp dirs; identity must survive that.

        A digest that folded in paths or mtimes would change every run and
        defeat image reuse entirely.
        """
        a = _make_task(tmp_path / "first")
        b = _make_task(tmp_path / "second")

        assert _sandbox(a)._images.identity() == _sandbox(b)._images.identity()

    def test_identity_changes_when_the_environment_changes(self, tmp_path):
        a = _make_task(tmp_path / "a")
        b = _make_task(tmp_path / "b", dockerfile="FROM python:3.13-slim\n")

        assert _sandbox(a)._images.identity() != _sandbox(b)._images.identity()

    def test_identity_changes_when_a_context_file_changes(self, tmp_path):
        """Skills baked into the image must produce a distinct image."""
        a = _make_task(tmp_path / "a")
        b = _make_task(tmp_path / "b")
        (b / "skills").mkdir()
        (b / "skills" / "SKILL.md").write_text("# a skill")

        assert _sandbox(a)._images.identity() != _sandbox(b)._images.identity()

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
            patch.object(AgentCoreImagePublisher, "_ensure_ecr_repository"),
            patch.object(AgentCoreImagePublisher, "_image_exists", return_value=False),
            patch.object(AgentCoreImagePublisher, "_build_and_push", new=_build),
            patch.object(
                AgentCoreImagePublisher,
                "_resolve_image_digest",
                lambda self, tag: f"reg/repo@sha256:{tag}",
            ),
        ):
            sandboxes = [_sandbox(task_dir, session=f"run-{i}") for i in range(20)]
            uris = await asyncio.gather(
                *(s._images.publish(force_build=False) for s in sandboxes)
            )

        assert builds["n"] == 1
        assert len(set(uris)) == 1

    @pytest.mark.asyncio
    async def test_concurrent_rollouts_register_one_runtime(
        self, tmp_path, monkeypatch
    ):
        """CreateAgentRuntime is a 5/s quota — it cannot run per rollout."""
        monkeypatch.setenv(
            "BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/runtime"
        )
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

    def test_runtime_identity_changes_with_execution_contract(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #942: incompatible lifecycle contracts cannot collide."""
        monkeypatch.setenv(
            "BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/runtime"
        )
        short = _sandbox(_make_task(tmp_path / "short"))
        long = _sandbox(_make_task(tmp_path / "long"))
        short.configure_agent_timeout(900)
        long.configure_agent_timeout(1800)

        image_digest = "a" * 64
        assert short._runtime_contract_digest(
            image_digest
        ) != long._runtime_contract_digest(image_digest)

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
            patch.object(AgentCoreImagePublisher, "_ensure_ecr_repository"),
            patch.object(AgentCoreImagePublisher, "_image_exists", return_value=False),
            patch.object(AgentCoreImagePublisher, "_build_and_push", new=_flaky),
            patch.object(
                AgentCoreImagePublisher,
                "_resolve_image_digest",
                lambda self, tag: f"reg/repo@sha256:{tag}",
            ),
        ):
            env = _sandbox(task_dir)
            with pytest.raises(RuntimeError, match="transient throttle"):
                await env._images.publish(force_build=False)
            await env._images.publish(force_build=False)

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


class TestLifecycleConfiguration:
    def test_effective_agent_timeout_expands_default_idle_window(self, tmp_path):
        """Guards PR #937: long rollouts must not inherit the 15-minute default."""
        env = _sandbox(_make_task(tmp_path))
        env.configure_agent_timeout(1200)

        assert env._lifecycle_configuration()["idleRuntimeSessionTimeout"] == 1200

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("BENCHFLOW_AGENTCORE_IDLE_TIMEOUT_SEC", "59"),
            ("BENCHFLOW_AGENTCORE_IDLE_TIMEOUT_SEC", "28801"),
            ("BENCHFLOW_AGENTCORE_MAX_LIFETIME_SEC", "-1"),
            ("BENCHFLOW_AGENTCORE_MAX_LIFETIME_SEC", "forever"),
        ],
    )
    def test_invalid_service_bounds_fail_before_an_aws_call(
        self, tmp_path, monkeypatch, name, value
    ):
        """Guards PR #937: reject invalid AgentCore lifecycle values actionably."""
        monkeypatch.setenv(name, value)
        env = _sandbox(_make_task(tmp_path))

        with pytest.raises(ValueError, match=name):
            env._lifecycle_configuration()

    def test_task_timeout_above_agentcore_limit_is_not_silently_clamped(self, tmp_path):
        """Guards PR #937: a backend that cannot honor the task must say so."""
        env = _sandbox(_make_task(tmp_path))
        env.configure_agent_timeout(28801)

        with pytest.raises(ValueError, match="28800"):
            env._lifecycle_configuration()


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
        """Guards PR #937: initial lease covers the throttle boundary."""
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
        remaining = (
            datetime.fromisoformat(tags[provisioning.LEASE_TAG]) - datetime.now(UTC)
        ).total_seconds()
        window = 28_800.0
        assert remaining == pytest.approx(
            window + provisioning.lease_renewal_interval(window), abs=2
        )
        # Successful create/adopt writes mark the throttle; the first fan-out
        # must not immediately send a redundant TagResource call.
        assert (
            provisioning.lease_needs_renewal("arn:rt", window, time.monotonic())
            is False
        )


class TestLeaseIntegrity:
    """A lease is only protection if it is written, kept, and refreshed."""

    @pytest.mark.asyncio
    async def test_a_failed_lease_write_aborts_the_launch(self, tmp_path, monkeypatch):
        """Swallowing it starts a session on a runtime cleanup may delete.

        Guards PR #937.
        """
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
        """Every provisioned runtime is leased, so an unleased one is unexplained.

        Guards PR #937.
        """
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
        """Cache-hit rollouts must refresh, or a long matrix outlives its lease.

        Guards PR #937.
        """
        window = 7200.0
        arn = "arn:renew"

        assert provisioning.lease_needs_renewal(arn, window, 0.0) is True
        provisioning.mark_lease_renewed(arn, 0.0)
        assert provisioning.lease_needs_renewal(arn, window, 10.0) is False
        assert provisioning.lease_needs_renewal(arn, window, window) is True

    @pytest.mark.asyncio
    async def test_every_rollout_renews_when_due(self, tmp_path, monkeypatch):
        """Provisioning is memoized, so renewal cannot live only in creation.

        Guards PR #937.
        """
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        env.runtime_arn = "arn:rt-renew-probe"
        control = MagicMock()

        with patch.object(env, "_client", return_value=control):
            await env._renew_lease()

        control.tag_resource.assert_called_once()
        assert provisioning.LEASE_TAG in control.tag_resource.call_args.kwargs["tags"]

    @pytest.mark.asyncio
    async def test_concurrent_due_rollouts_single_flight_the_lease(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #937: due-check/write/mark must be atomic per runtime."""
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        env.runtime_arn = "arn:rt-concurrent-renewal"
        control = MagicMock()

        with patch.object(env, "_client", return_value=control):
            await asyncio.gather(*(env._renew_lease() for _ in range(20)))

        control.tag_resource.assert_called_once()


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
        """Right image, wrong contract: wrong permissions, egress, or shell.

        Guards PR #937.
        """
        from benchflow.sandbox.protocol import SandboxStartupError

        with pytest.raises(SandboxStartupError, match="does not match"):
            self._verify(self._detail(**{field: value}))


_VPC_NETWORK = {
    "networkMode": "VPC",
    "networkModeConfig": {
        "subnets": ["subnet-0a1b2c3d4e5f67890"],
        "securityGroups": ["sg-0a1b2c3d4e5f67890"],
    },
}


class TestNetworkConfigurationReachesTheRuntime:
    """The configured network placement has to survive to the control plane.

    ``_network_configuration`` deciding VPC is worth nothing if registration
    still sends the old constant, and the mistake is invisible: the runtime
    reaches READY either way, just reachable from the internet.
    """

    def _vpc_env(self, monkeypatch):
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_NETWORK_MODE", "VPC")
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_SUBNETS", "subnet-0a1b2c3d4e5f67890")
        monkeypatch.setenv(
            "BENCHFLOW_AGENTCORE_SECURITY_GROUPS", "sg-0a1b2c3d4e5f67890"
        )

    def _adopt(self, found):
        control = MagicMock()
        control.get_agent_runtime.return_value = {
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "repo@sha256:a"}
            },
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 600,
                "maxLifetime": 7200,
            },
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": found,
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }
        AgentCoreSandbox._verify_adopted_runtime(
            control,
            "rt-1",
            "repo@sha256:a",
            {"idleRuntimeSessionTimeout": 600, "maxLifetime": 7200},
            "arn:aws:iam::1:role/rt",
            _VPC_NETWORK,
            {"serverProtocol": "HTTP"},
        )

    def test_a_service_managed_member_does_not_block_adoption(self):
        """``VpcConfig`` carries members BenchFlow never sends.

        ``requireServiceS3Endpoint`` cannot even be set on create. Adoption is
        the path every rollout after the first takes, so comparing the whole
        document would fail a VPC matrix on everything after its first create.
        """
        self._adopt(
            {
                "networkMode": "VPC",
                "networkModeConfig": {
                    **_VPC_NETWORK["networkModeConfig"],
                    "requireServiceS3Endpoint": True,
                },
            }
        )

    def test_a_different_subnet_is_still_refused(self):
        """Tolerating extra members must not tolerate the wrong network."""
        from benchflow.sandbox.protocol import SandboxStartupError

        with pytest.raises(SandboxStartupError, match="does not match"):
            self._adopt(
                {
                    "networkMode": "VPC",
                    "networkModeConfig": {
                        "subnets": ["subnet-0987654321fedcba"],
                        "securityGroups": ["sg-0a1b2c3d4e5f67890"],
                    },
                }
            )

    @pytest.mark.asyncio
    async def test_registration_sends_the_configured_vpc(self, tmp_path, monkeypatch):
        self._vpc_env(monkeypatch)
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.return_value = {
            "agentRuntimeId": "rt-1",
            "agentRuntimeArn": "arn:rt",
        }
        control.get_agent_runtime.return_value = {"status": "READY"}

        with patch.object(env, "_client", return_value=control):
            await env._create_or_adopt_runtime("bf_x", "repo@sha256:a")

        sent = control.create_agent_runtime.call_args.kwargs
        assert sent["networkConfiguration"] == _VPC_NETWORK

    @pytest.mark.asyncio
    async def test_repointing_a_stale_runtime_keeps_the_vpc(
        self, tmp_path, monkeypatch
    ):
        """``update_agent_runtime`` replaces the whole config it is given.

        Omitting the network here is how a VPC runtime silently reverts to
        PUBLIC on the next image change.
        """
        from botocore.exceptions import ClientError

        self._vpc_env(monkeypatch)
        env = _sandbox(_make_task(tmp_path))
        control = MagicMock()
        control.create_agent_runtime.side_effect = ClientError(
            {"Error": {"Code": "ConflictException", "Message": "exists"}},
            "CreateAgentRuntime",
        )
        control.get_agent_runtime.return_value = {
            "status": "READY",
            "agentRuntimeArtifact": {
                "containerConfiguration": {"containerUri": "repo@sha256:b"}
            },
            "lifecycleConfiguration": env._lifecycle_configuration(),
            "roleArn": "arn:aws:iam::1:role/rt",
            "networkConfiguration": _VPC_NETWORK,
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        }

        with (
            patch.object(env, "_client", return_value=control),
            patch.object(
                provisioning,
                "find_runtime_by_name",
                return_value=("arn:rt", "rt-1", "repo@sha256:a"),
            ),
        ):
            await env._create_or_adopt_runtime("bf_x", "repo@sha256:b")

        sent = control.update_agent_runtime.call_args.kwargs
        assert sent["networkConfiguration"] == _VPC_NETWORK

    def test_a_vpc_runtime_does_not_share_an_identity_with_a_public_one(
        self, tmp_path, monkeypatch
    ):
        """Same name means the VPC run adopts the existing PUBLIC runtime."""
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        public = env._runtime_contract_digest("sha256:a")

        self._vpc_env(monkeypatch)

        assert env._runtime_contract_digest("sha256:a") != public


class TestDeprecatedCleanupAliasIsSafe:
    """`bench environment cleanup` reaches the same destructive code.

    Guards PR #937. Guarding only the new command left the deprecated alias
    able to select STARTED sandboxes for deletion via a negative age.
    """

    def test_negative_age_is_rejected_by_the_daytona_reaper(self):
        """Guards PR #937 through the deprecated cleanup alias."""
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
        """Guards PR #937: no library age parameter may accept negatives."""
        from benchflow.sandbox.daytona_reaper import reap_stale_sandboxes

        with pytest.raises(ValueError, match="must be >= 0"):
            reap_stale_sandboxes(MagicMock(), dry_run=True, **kwargs)

    @pytest.mark.parametrize("command", ["sandbox", "environment"])
    def test_cli_rejects_negative_max_age(self, command):
        """Guards PR #937: both current and deprecated aliases must refuse it."""
        from typer.testing import CliRunner

        from benchflow.cli.main import app

        result = CliRunner().invoke(app, [command, "cleanup", "--max-age", "-1"])

        assert result.exit_code != 0


class TestLeaseRenewalRetry:
    """Guards PR #937: a failed renewal must not be throttled as a success."""

    def _sandbox_with_control(self, tmp_path, control):
        env = _sandbox(_make_task(tmp_path))
        env.runtime_arn = "arn:retry-probe"
        return env

    @pytest.mark.asyncio
    async def test_a_failed_renewal_retries_immediately(self, tmp_path, monkeypatch):
        """Guards PR #937: the throttle must record the write, not the attempt.

        A first renewal that raised still advanced the window, so the retry
        that would have fixed it made zero tag_resource calls and the rollout
        continued against the un-extended lease.
        """
        from benchflow.sandbox.protocol import SandboxStartupError

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = self._sandbox_with_control(tmp_path, None)
        control = MagicMock()
        control.tag_resource.side_effect = RuntimeError("AccessDenied")

        with (
            patch.object(env, "_client", return_value=control),
            pytest.raises(SandboxStartupError),
        ):
            await env._renew_lease()

        # AWS recovers; the very next attempt must actually call AWS.
        control.tag_resource.side_effect = None
        with patch.object(env, "_client", return_value=control):
            await env._renew_lease()

        assert control.tag_resource.call_count == 2

    def test_the_throttle_only_advances_on_success(self):
        """Guards PR #937: the predicate must record nothing by itself."""
        arn = "arn:pure-predicate"
        window = 7200.0

        assert provisioning.lease_needs_renewal(arn, window, 0.0) is True
        # Still due — checking is not renewing.
        assert provisioning.lease_needs_renewal(arn, window, 1.0) is True

        provisioning.mark_lease_renewed(arn, 1.0)

        assert provisioning.lease_needs_renewal(arn, window, 2.0) is False

    @pytest.mark.asyncio
    async def test_start_renews_before_warming_the_session(self, tmp_path, monkeypatch):
        """Guards PR #937: renewal must be wired into start(), not just callable.

        Provisioning is memoized, so a cache-hit rollout reaches neither the
        create nor the adopt path; if start() did not renew, the lease would
        only ever be written by the first rollout of an image.
        """
        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        order: list[str] = []

        async def _publish(*, force_build):
            return "repo@sha256:a"

        async def _ensure(image_uri):
            env.runtime_arn = "arn:started"
            return "arn:started"

        async def _renew():
            order.append("renew")

        async def _warm():
            order.append("warm")

        with (
            patch.object(env._images, "publish", side_effect=_publish),
            patch.object(env, "_ensure_runtime", side_effect=_ensure),
            patch.object(env, "_renew_lease", side_effect=_renew),
            patch.object(env, "_warm_session", side_effect=_warm),
        ):
            await env.start(force_build=False)

        assert order == ["renew", "warm"]

    @pytest.mark.asyncio
    async def test_start_resets_sealed_channel_for_new_session(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #942: a new microVM cannot reuse the prior sealing key."""

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
        env = _sandbox(_make_task(tmp_path))
        prior_channel = env._sealed
        prior_channel._public_key = "old-session-key"

        async def _publish(*, force_build):
            return "repo@sha256:a"

        async def _ensure(image_uri):
            return "arn:started"

        with (
            patch.object(env._images, "publish", side_effect=_publish),
            patch.object(env, "_ensure_runtime", side_effect=_ensure),
            patch.object(env, "_renew_lease"),
            patch.object(env, "_warm_session"),
        ):
            await env.start(force_build=False)

        assert env._sealed is not prior_channel
        assert env._sealed._public_key is None


class TestReservedPathsArePreserved:
    """Guards PR #937: never clobber a task file at a generated path."""

    @pytest.mark.parametrize(
        "reserved",
        ["Dockerfile.benchflow-agentcore", ".benchflow_agentcore_shim.py"],
    )
    def test_a_colliding_task_file_is_refused(self, tmp_path, reserved):
        """Guards PR #937: generated names are an explicit backend contract.

        Staging no longer mutates the caller's tree, but a task shipping either
        reserved name would still be excluded from the canonical context and
        silently replaced by backend scaffolding without this refusal.
        """
        task_dir = _make_task(tmp_path)
        original = "important task content\n"
        (task_dir / reserved).write_text(original)

        with pytest.raises(ValueError, match="reserved"):
            _sandbox(task_dir)

        # The refusal must happen before anything touches the file.
        assert (task_dir / reserved).read_text() == original

    @pytest.mark.parametrize(
        "reserved",
        ["Dockerfile.benchflow-agentcore", ".benchflow_agentcore_shim.py"],
    )
    def test_a_dangling_reserved_symlink_is_refused(self, tmp_path, reserved):
        """Guards PR #937: ``Path.exists`` misses dangling symlinks."""
        task_dir = _make_task(tmp_path)
        external = tmp_path / "outside" / "missing"
        (task_dir / reserved).symlink_to(external)

        with pytest.raises(ValueError, match="reserved"):
            _sandbox(task_dir)

        assert not external.exists()

    def test_a_symlinked_dockerfile_cannot_escape_the_context(self, tmp_path):
        """Guards PR #937: external Dockerfile bytes must never reach AWS."""
        task_dir = _make_task(tmp_path)
        external = tmp_path / "outside.Dockerfile"
        external.write_text("FROM scratch\nRUN echo EXTERNAL_SECRET\n")
        dockerfile = task_dir / "Dockerfile"
        dockerfile.unlink()
        dockerfile.symlink_to(external)

        with pytest.raises(ValueError, match="regular, non-symlink"):
            _sandbox(task_dir)

        with pytest.raises(ValueError, match="regular, non-symlink"):
            provisioning.read_regular_text(dockerfile)
