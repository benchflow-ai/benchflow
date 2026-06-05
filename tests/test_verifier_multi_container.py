"""Tests for the deferred multi-container verification items of #248.

PR #310 shipped first-class compose *service selection* on the ``exec`` path
but deferred the verification-side wiring. This file covers the follow-up:

- **Item 4** — target-side ``test.sh`` verification: the test-script verifier
  can run inside a non-``main`` compose service so it can inspect target-side
  state (RCE markers, DB modifications) instead of only the agent workspace.
- **Item 5** — cross-container flag plumbing / task-schema convention: the
  ``[verifier].service`` knob in ``task.toml`` is the declarative convention
  task authors use to point verification at a target container.
- **Item 3** — cross-container hardening policy: ``harden_before_verify``
  intentionally hardens only ``main``; deliberately vulnerable target
  containers are never hardened.

All tests are unit tests with mocked sandboxes — no Docker/Daytona infra.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig

# ---------------------------------------------------------------------------
# Item 5 — [verifier].service task-schema convention
# ---------------------------------------------------------------------------


class TestVerifierServiceConfig:
    """#248 item 5: [verifier].service is the cross-container schema knob."""

    def test_service_defaults_to_main(self) -> None:
        """#248: omitting [verifier].service keeps the agent container."""
        cfg = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        assert cfg.verifier.service == "main"

    def test_service_can_target_a_named_container(self) -> None:
        """#248: task.toml can point the verifier at a target service."""
        toml = """\
version = "1.0"

[verifier]
service = "target"
"""
        cfg = TaskConfig.model_validate_toml(toml)
        assert cfg.verifier.service == "target"

    def test_service_is_keyword_compatible_with_existing_tasks(self) -> None:
        """#248: existing task.toml files (no service key) are unaffected."""
        toml = """\
version = "1.0"

[verifier]
timeout_sec = 120
user = "root"
"""
        cfg = TaskConfig.model_validate_toml(toml)
        assert cfg.verifier.service == "main"
        assert cfg.verifier.user == "root"


# ---------------------------------------------------------------------------
# Item 4 — target-side test.sh verification
# ---------------------------------------------------------------------------


def _make_task(tmp_path: Path, toml: str) -> MagicMock:
    """Build a task stub backed by a real ``tests/`` directory."""
    task_dir = tmp_path / "task"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    task = MagicMock()
    task.task_dir = task_dir
    task.paths.task_dir = task_dir
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.config = TaskConfig.model_validate_toml(toml)
    task.instruction = "Exploit the target."
    return task


class _RecordingSandbox:
    """Sandbox stub that records the ``service`` used for each operation."""

    def __init__(
        self,
        rollout_paths: RolloutPaths,
        reward: str = "1.0",
        is_mounted: bool = False,
    ) -> None:
        self.is_mounted = is_mounted
        self._rollout_paths = rollout_paths
        self._reward = reward
        self.upload_calls: list[dict] = []
        self.download_calls: list[dict] = []
        self.exec_calls: list[dict] = []

    async def upload_dir(self, source_dir, target_dir, service: str = "main") -> None:
        self.upload_calls.append(
            {"source": source_dir, "target": target_dir, "service": service}
        )

    async def download_dir(self, source_dir, target_dir, service: str = "main") -> None:
        self.download_calls.append(
            {"source": source_dir, "target": target_dir, "service": service}
        )
        # Mimic a target-side test.sh that wrote reward artifacts.
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "reward.txt").write_text(self._reward)
        (dest / "reward-details.json").write_text(
            '{"criteria": {"correctness": {"score": 1.0}}}'
        )

    async def exec(self, command, service: str = "main", **kwargs) -> ExecResult:
        self.exec_calls.append({"command": command, "service": service, **kwargs})
        if self.is_mounted and service == "main" and "test-stdout.txt" in command:
            self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            self._rollout_paths.reward_text_path.write_text(self._reward)
        return ExecResult(stdout="", stderr="", return_code=0)


class TestTargetSideTestScriptVerification:
    """#248 item 4: the test-script verifier can run inside a target service."""

    @pytest.mark.asyncio
    async def test_default_verifier_runs_test_script_in_main(
        self, tmp_path: Path
    ) -> None:
        """#248: with no [verifier].service the test.sh still runs in main."""
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths)

        await Verifier(task, rollout_paths, sandbox).verify()

        assert {c["service"] for c in sandbox.upload_calls} == {"main"}
        assert {c["service"] for c in sandbox.exec_calls} == {"main"}

    @pytest.mark.asyncio
    async def test_verifier_runs_test_script_in_target_service(
        self, tmp_path: Path
    ) -> None:
        """#248: [verifier].service routes test.sh into the target container."""
        toml = 'version = "1.0"\n[verifier]\nservice = "target"\n'
        task = _make_task(tmp_path, toml)
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        # tests/ uploaded into the target container, test.sh exec'd there.
        assert {c["service"] for c in sandbox.upload_calls} == {"target"}
        assert {c["service"] for c in sandbox.exec_calls} == {"target"}
        assert result.rewards == {"reward": 1.0}

    @pytest.mark.asyncio
    async def test_target_side_reward_downloaded_from_target_service(
        self, tmp_path: Path
    ) -> None:
        """#248: the reward file is fetched from the target, not main."""
        toml = 'version = "1.0"\n[verifier]\nservice = "target"\n'
        task = _make_task(tmp_path, toml)
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths, reward="1.0")

        await Verifier(task, rollout_paths, sandbox).verify()

        assert sandbox.download_calls, "reward must be downloaded from the target"
        assert {c["service"] for c in sandbox.download_calls} == {"target"}

    @pytest.mark.asyncio
    async def test_reward_details_json_preserved_on_verifier_download(
        self, tmp_path: Path
    ) -> None:
        """Guards task-standard F8 reward-details.json copy-through on download."""
        toml = 'version = "1.0"\n[verifier]\nservice = "target"\n'
        task = _make_task(tmp_path, toml)
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        rollout_paths.reward_details_path.write_text('{"stale": true}')
        sandbox = _RecordingSandbox(rollout_paths, reward="1.0")

        await Verifier(task, rollout_paths, sandbox).verify()

        details = rollout_paths.reward_details_path.read_text()
        assert '"criteria"' in details
        assert '"stale"' not in details

    @pytest.mark.asyncio
    async def test_target_service_logs_verifier_dir_created_before_test_sh(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #321: create /logs/verifier in a non-main target service.

        ``/logs/verifier`` is bind-mounted (and re-created by
        ``harden_before_verify``) only in the ``main`` container. When the
        verifier runs ``test.sh`` in a target service its stdout is redirected
        to ``/logs/verifier/test-stdout.txt`` — a path that does not exist on a
        typical target image. Without an explicit ``mkdir -p`` the redirect
        fails before ``test.sh`` starts and target-side verification silently
        produces no reward file. The verifier must create the directory in the
        target container *before* the ``test.sh`` exec.
        """
        toml = 'version = "1.0"\n[verifier]\nservice = "target"\n'
        task = _make_task(tmp_path, toml)
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths)

        await Verifier(task, rollout_paths, sandbox).verify()

        commands = [c["command"] for c in sandbox.exec_calls]
        mkdir_indices = [
            i
            for i, cmd in enumerate(commands)
            if "mkdir" in cmd and "/logs/verifier" in cmd
        ]
        assert mkdir_indices, "verifier must mkdir /logs/verifier in the target service"
        # The directory is created in the target service, as root.
        mkdir_call = sandbox.exec_calls[mkdir_indices[0]]
        assert mkdir_call["service"] == "target"
        assert mkdir_call.get("user") == "root"
        # It must run before test.sh redirects its stdout into that directory.
        test_sh_indices = [
            i for i, cmd in enumerate(commands) if "test-stdout.txt" in cmd
        ]
        assert test_sh_indices, "test.sh redirect exec not found"
        assert mkdir_indices[0] < test_sh_indices[0], (
            "/logs/verifier must be created before the test.sh stdout redirect"
        )

    @pytest.mark.asyncio
    async def test_main_service_skips_redundant_logs_verifier_mkdir(
        self, tmp_path: Path
    ) -> None:
        """Guards PR #321: main-service runs are unaffected by the target fix.

        The ``main`` container already has ``/logs/verifier`` bind-mounted and
        re-created by ``harden_before_verify``, so the verifier must not issue
        an extra ``mkdir`` for it — the fix is scoped to non-``main`` services.
        """
        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths, is_mounted=True)

        await Verifier(task, rollout_paths, sandbox).verify()

        mkdir_calls = [
            c
            for c in sandbox.exec_calls
            if "mkdir" in c["command"] and "/logs/verifier" in c["command"]
        ]
        assert not mkdir_calls, (
            "main-service verifier must not mkdir /logs/verifier (already mounted)"
        )

    @pytest.mark.asyncio
    async def test_mounted_sandbox_still_downloads_from_target_service(
        self, tmp_path: Path
    ) -> None:
        """#248: a host-mounted sandbox must still download target rewards.

        The ``is_mounted`` fast path that skips ``download_dir`` is gated on
        ``service == "main"`` — only the agent container has the rollout dir
        bind-mounted. A non-``main`` target service is never mounted, so its
        ``reward.txt`` must still be downloaded even when the sandbox is
        otherwise mounted. This fails (RewardFileNotFoundError) if the
        ``service == "main" and`` guard on the fast path is dropped.
        """
        toml = 'version = "1.0"\n[verifier]\nservice = "target"\n'
        task = _make_task(tmp_path, toml)
        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        rollout_paths.mkdir()
        sandbox = _RecordingSandbox(rollout_paths, reward="1.0", is_mounted=True)

        result = await Verifier(task, rollout_paths, sandbox).verify()

        assert sandbox.download_calls, (
            "target reward must be downloaded even when the sandbox is mounted"
        )
        assert {c["service"] for c in sandbox.download_calls} == {"target"}
        assert result.rewards == {"reward": 1.0}


# ---------------------------------------------------------------------------
# Item 3 — cross-container hardening policy
# ---------------------------------------------------------------------------


class TestCrossContainerHardeningPolicy:
    """#248 item 3: anti-tamper hardening applies to ``main`` only.

    A vulhub-style target is deliberately vulnerable. Running the agent
    anti-tamper hardening (kill user processes, scrub PATH, restore build
    config) inside it would be wrong — and pointless, since the agent never
    has a shell there. The verifier hardening therefore only touches
    ``main``; ``[verifier].service`` selects where ``test.sh`` *runs*, not
    where hardening happens.
    """

    @pytest.mark.asyncio
    async def test_harden_before_verify_only_touches_main(self, tmp_path: Path) -> None:
        """#248: harden_before_verify never execs into a non-main service."""
        from benchflow.sandbox.lockdown import harden_before_verify

        task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\nservice = "target"\n')
        env = MagicMock()
        services_seen: list[str] = []

        async def fake_exec(command, service: str = "main", **kwargs) -> ExecResult:
            services_seen.append(service)
            return ExecResult(stdout="[]", stderr="", return_code=0)

        env.exec = AsyncMock(side_effect=fake_exec)

        await harden_before_verify(env, task, sandbox_user=None, workspace="/app")

        # Every hardening command stayed in the agent container.
        assert services_seen, "hardening should issue at least one exec"
        assert set(services_seen) == {"main"}
