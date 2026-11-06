"""Behavioral guards for PY8 AG-C1..C4 at bf2d8c83; fake credentials only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents import antigravity_config, credentials, muse_acp_shim, muse_config
from benchflow.agents import env as agent_env
from benchflow.agents.registry import AGENTS
from benchflow.review.evidence import capture_task_evidence
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
from benchflow.rollout._artifacts import MANIFEST_NAME
from benchflow.rollout._separate_verifier import build_transfer_payload
from benchflow.runtime import check_credentials
from benchflow.sandbox.protocol import ExecResult
from tests._receipt_fakes import receipt_aware

LOGIN = '{"providers":{"meta":{"refresh_token":"fake-ag-review-login"}}}'


@pytest.mark.parametrize("agent", ["muse", "agy"])
@pytest.mark.parametrize("failure", ["digest", "shasum", "download", "none"])
def test_bad_download_installs_nothing(tmp_path, monkeypatch, agent, failure):
    """Guards AG-C1 at bf2d8c83: conditional errexit installed bad bytes."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for command in (
        "sh",
        "grep",
        "head",
        "tr",
        "uname",
        "mktemp",
        "mkdir",
        "mv",
        "chmod",
        "ln",
        "rm",
        "tar",
        "gzip",
        "cp",
        "cut",
        "cat",
    ):
        source = shutil.which(command)
        assert source, command
        (tools / command).symlink_to(source)
    checksum = "sha256sum" if agent == "muse" else "sha512sum"
    fallback = tools / "shasum"
    fallback.write_text(
        f'#!/bin/sh\nshift 2\nexec {shlex.quote(shutil.which(checksum))} "$@"\n'
    )
    fallback.chmod(0o755)
    if failure != "shasum":
        (tools / checksum).symlink_to(shutil.which(checksum))
    version = (
        muse_config.MUSE_VERSION if agent == "muse" else antigravity_config.AGY_VERSION
    )
    binary = f"#!/bin/sh\necho '{version}'\n".encode()
    payload = tmp_path / "download"
    if agent == "muse":
        payload.write_bytes(binary)
        digest = hashlib.sha256(binary).hexdigest()
    else:
        with tarfile.open(payload, "w:gz") as archive:
            entry = tarfile.TarInfo("antigravity")
            entry.size = len(binary)
            entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(binary))
        digest = hashlib.sha512(payload.read_bytes()).hexdigest()
    if failure not in ("download", "none"):
        digest = "0" * len(digest)
    curl = tools / "curl"
    curl.write_text(
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do\n"
        'case "$1" in -o|-fsSLo) shift; out=$1;; esac\nshift\ndone\n'
        f'cp {shlex.quote(str(payload))} "$out"\n'
        + (
            "echo 'fake download failed' >&2; exit 22\n"
            if failure == "download"
            else ""
        )
    )
    curl.chmod(0o755)
    target = tmp_path / "installed"
    if agent == "muse":
        monkeypatch.setattr(muse_config, "MUSE_DIR", str(target))
        monkeypatch.setattr(muse_config, "MUSE_BIN", str(target / "muse"))
        monkeypatch.setattr(
            muse_config,
            "MUSE_RELEASES",
            {
                arch: ("https://invalid.example/fake", digest, len(binary))
                for arch in ("x86_64", "aarch64")
            },
        )
        command = muse_config.muse_install_cmd("exit 99")
    else:
        monkeypatch.setattr(antigravity_config, "AGY_BIN", str(target / "agy"))
        monkeypatch.setattr(
            antigravity_config,
            "AGY_RELEASES",
            {
                arch: ("https://invalid.example/fake", digest)
                for arch in ("x86_64", "aarch64")
            },
        )
        command = antigravity_config.agy_install_cmd("exit 99")
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={**os.environ, "PATH": str(tools), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert target.exists()
        assert version in result.stdout
        return
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert "failed" in result.stderr.lower() or "did not match" in result.stderr.lower()
    assert not target.exists(), "unverified binary was installed"


class LocalTransport:
    async def exec(self, command, **kwargs):
        result = subprocess.run(shlex.split(command), capture_output=True, text=True)
        return ExecResult(result.returncode, result.stdout, result.stderr)

    async def download_file(self, source, destination):
        shutil.copyfile(source, destination)


async def test_xdg_login_never_enters_evidence_or_transfer(tmp_path, monkeypatch):
    """Guards AG-C2 at bf2d8c83 through real evidence and transfer writers."""
    home = tmp_path / "managed"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    override = workspace / "runtime-config"
    monkeypatch.setenv("BENCHFLOW_AGENT_HOME", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(override))
    monkeypatch.setenv("MUSE_AUTH_JSON", LOGIN)
    assert muse_acp_shim.prepare_credentials() == "subscription"
    trial = tmp_path / "trial"
    await capture_task_evidence(LocalTransport(), str(workspace), trial / "evidence")
    (trial / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "collections": [{"kind": "logs", "status": "empty"}],
                "files": [],
            }
        )
    )
    transfer = trial / "transfer.tar"
    build_transfer_payload(trial, transfer)
    assert not (trial / "evidence/workspace/runtime-config/muse/auth.json").exists()
    with tarfile.open(transfer) as archive:
        assert not any(
            "runtime-config/muse/auth.json" in entry.name for entry in archive
        )
        for entry in archive:
            if entry.isfile():
                handle = archive.extractfile(entry)
                assert handle is not None
                assert b"fake-ag-review-login" not in handle.read()
    assert not (override / "muse/auth.json").exists()
    assert (home / ".config/muse/auth.json").read_text() == LOGIN
    assert muse_acp_shim.muse_env("subscription")["XDG_CONFIG_HOME"] == str(
        home / ".config"
    )


@pytest.mark.parametrize(
    "model",
    [
        "google-vertex/gemini-3.8-flash",
        "openai/gpt-5.5",
        "meta/other",
        "muse-spark-1.3-contributor",
    ],
)
def test_invalid_muse_model_precedes_credential_discovery(monkeypatch, model):
    """Guards AG-C4 at bf2d8c83: no dotenv/provider/ADC discovery on refusal."""
    discover = MagicMock(return_value={})
    adc = MagicMock()
    monkeypatch.setattr(agent_env, "load_dotenv_env", discover)
    monkeypatch.setattr(agent_env, "inject_vertex_credentials", adc)
    with pytest.raises(ValueError, match=r"Muse|muse|contributor"):
        agent_env.resolve_agent_env("muse-code", model, {"MUSE_AUTH_JSON": LOGIN})
    discover.assert_not_called()
    adc.assert_not_called()


def test_preflight_refuses_bad_muse_role_before_any_adc_read(tmp_path, monkeypatch):
    """Guards AG-C4 at bf2d8c83 even when a valid Vertex role comes first."""
    adc_file = tmp_path / ".config/gcloud/application_default_credentials.json"
    adc_file.parent.mkdir(parents=True)
    adc_file.write_text('{"type":"authorized_user","refresh_token":"fake-adc"}')
    reads = []
    read_text = Path.read_text

    def tracked_read(path, *args, **kwargs):
        if path == adc_file:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_env, "load_dotenv_env", lambda: {})
    env = {
        "MUSE_AUTH_JSON": LOGIN,
        "GOOGLE_APPLICATION_CREDENTIALS": str(adc_file),
        "GOOGLE_CLOUD_PROJECT": "fake-project",
    }
    roles = [
        Role(name="first", agent="opencode", model="google-vertex/gemini-3.8-flash"),
        Role(name="bad", agent="muse-code", model="google-vertex/gemini-3.8-flash"),
    ]
    config = SimpleNamespace(
        agent="opencode",
        model=None,
        agent_env=env,
        scenes=[SimpleNamespace(roles=roles)],
    )
    with pytest.raises(ValueError, match=r"Muse|muse"):
        check_credentials([config])
    assert reads == []


async def test_muse_writer_never_uploads_provider_credentials(monkeypatch):
    """Guards AG-C4 at bf2d8c83 at the independent writer boundary."""
    upload = AsyncMock()
    monkeypatch.setattr(credentials, "upload_credential", upload)
    with pytest.raises(ValueError, match=r"Muse|muse"):
        await credentials.write_credential_files(
            object(),
            "muse-code",
            {
                "MUSE_AUTH_JSON": LOGIN,
                "GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"fake":"adc"}',
            },
            AGENTS["muse-code"],
            "google-vertex/gemini-3.8-flash",
            "/tmp/fake-agent-home",
        )
    upload.assert_not_awaited()


def test_supported_muse_models_skip_provider_discovery(monkeypatch):
    """Guards AG-C4 at bf2d8c83 for a supported subscription selection."""
    monkeypatch.setattr(agent_env, "load_dotenv_env", lambda: {})
    adc = MagicMock()
    provider = MagicMock()
    monkeypatch.setattr(agent_env, "inject_vertex_credentials", adc)
    monkeypatch.setattr(agent_env, "resolve_provider_env", provider)
    resolved = agent_env.resolve_agent_env(
        "muse-code", "meta/muse-spark-1.3", {"MUSE_AUTH_JSON": LOGIN}
    )
    assert resolved["MUSE_AUTH_JSON"] == LOGIN
    adc.assert_not_called()
    provider.assert_not_called()


def test_mixed_muse_scene_is_refused_before_credentials(tmp_path):
    """Guards AG-C3 at bf2d8c83: shared-UID mixed scenes cannot isolate logins."""
    with pytest.raises(ValueError, match=r"muse-code.*mixed|mixed.*muse-code"):
        RolloutConfig(
            task_path=tmp_path,
            scenes=[
                Scene(
                    roles=[
                        Role(name="muse", agent="muse-code"),
                        Role(name="other", agent="opencode"),
                    ]
                )
            ],
        )


def test_root_muse_run_is_refused(tmp_path):
    """Guards AG-C3 at bf2d8c83: root agents cannot be safely UID-quiesced."""
    with pytest.raises(ValueError, match=r"muse-code.*sandbox_user"):
        RolloutConfig(task_path=tmp_path, agent="muse-code", sandbox_user=None)


def test_muse_plan_validation_preserves_self_gen_config(tmp_path):
    """Guards AG-C4's 2dcd5be0: self-gen uses its orchestrator, not scenes."""
    config = RolloutConfig(
        task_path=tmp_path, agent="claude-agent-acp", skill_mode="self-gen"
    )
    assert config.skill_mode == "self-gen"


def _rollout(tmp_path):
    trial = Rollout(
        RolloutConfig(task_path=tmp_path, agent="muse-code", model="muse-spark-1.3")
    )
    trial._rollout_dir = tmp_path
    trial._env = MagicMock()
    trial._env.exec = AsyncMock(return_value=ExecResult(0, "", ""))
    trial._agent_cfg = AGENTS["muse-code"]
    trial._agent_launch = "/opt/benchflow/bin/muse-acp-shim"
    trial._agent_env = {"MUSE_AUTH_JSON": LOGIN}
    trial._agent_cwd = "/tmp/fake-workspace"
    trial._phase = "installed"
    trial._timeout = 600
    trial._task = SimpleNamespace(
        config=SimpleNamespace(sandbox=SimpleNamespace(mcp_servers=[]))
    )
    planes = MagicMock()
    planes.ensure_litellm_runtime = AsyncMock(
        side_effect=lambda **kw: (kw["agent_env"], None)
    )
    planes.connect_acp = AsyncMock(
        return_value=(
            MagicMock(close=AsyncMock()),
            MagicMock(),
            MagicMock(),
            "muse-code",
        )
    )
    planes.write_credential_files = AsyncMock()
    planes.quiesce_agent = AsyncMock()
    planes.agent_launch.return_value = trial._agent_launch
    planes.resolve_agent_env.side_effect = lambda _agent, _model, env: {
        "MUSE_AUTH_JSON": LOGIN,
        **env,
    }
    planes.apply_web_tool_policy = AsyncMock()
    planes.install_agent = AsyncMock(return_value=AGENTS["opencode"])
    planes.upload_subscription_auth = AsyncMock()
    trial._planes = planes
    return trial


async def test_disconnect_quiesces_then_removes_and_reconnect_restages(tmp_path):
    """Guards AG-C3 at bf2d8c83 across two identical Muse role connections."""
    trial = _rollout(tmp_path)
    events = []
    home = tmp_path / "fake-home"
    login = home / ".config/muse/auth.json"

    async def stage(*args, **kwargs):
        login.parent.mkdir(parents=True, exist_ok=True)
        login.write_text(LOGIN)

    # bf2d8c83 staged this at install time; model that existing login too.
    await stage()
    trial._planes.write_credential_files.side_effect = stage
    trial._planes.quiesce_agent.side_effect = lambda *args: events.append("quiesced")

    async def execute(command, **kwargs):
        if "/proc/self/mountinfo" in command:
            from tests.sandbox_mounts import ROOT_MOUNTINFO

            return ExecResult(0, ROOT_MOUNTINFO, "")
        if "rm -f" in command and ".config/muse/auth.json" in command:
            if events == ["closed", "quiesced"]:
                assert events == ["closed", "quiesced"]
                events.append("removed")
            assert kwargs["user"] == "agent"
            result = subprocess.run(
                ["sh", "-c", command.replace("/home/agent", str(home))],
                capture_output=True,
                text=True,
            )
            return ExecResult(result.returncode, result.stdout, result.stderr)
        return ExecResult(0, "", "")

    trial._env.exec.side_effect = execute
    role = Role(name="solver", agent="muse-code", model="muse-spark-1.3")
    await trial.connect_as(role)
    assert login.read_text() == LOGIN
    trial._acp_client.close.side_effect = lambda: events.append("closed")
    await trial.disconnect()
    assert events == ["closed", "quiesced", "removed"]
    assert not login.exists()
    await trial.connect_as(role)
    assert login.read_text() == LOGIN
    assert trial._planes.write_credential_files.await_count == 2
    trial._planes.connect_acp.assert_awaited()


async def test_role_switch_after_muse_disconnect_is_refused(tmp_path):
    """Guards AG-C3 at bf2d8c83 for a dynamic Muse-to-OpenCode switch."""
    trial = _rollout(tmp_path)
    await trial.connect()
    await trial.disconnect()
    trial._planes.connect_acp.reset_mock()
    with pytest.raises(ValueError, match=r"mixed.*muse-code|muse-code.*mixed"):
        await trial.connect_as(
            Role(name="other", agent="opencode", model="openai/gpt-5.5")
        )
    trial._planes.connect_acp.assert_not_awaited()


async def test_disconnect_does_not_ignore_quiescence_failure(tmp_path):
    """Guards AG-C3 at bf2d8c83: no cleanup success while a writer survives."""
    trial = _rollout(tmp_path)
    await trial.connect()
    trial._planes.quiesce_agent.side_effect = RuntimeError("fake surviving process")
    with pytest.raises(RuntimeError, match="surviving process"):
        await trial.disconnect()


async def test_shared_verifier_removes_muse_login_after_quiescence(
    tmp_path, monkeypatch
):
    """Guards AG-C3 at bf2d8c83 on the shared-verifier hardening path."""
    from benchflow.sandbox import lockdown

    events = []
    env = MagicMock()
    env._benchflow_path_inspectors = {"main": "/etc/fake-inspector"}
    home = tmp_path / "fake-home"
    login = home / ".config/muse/auth.json"
    login.parent.mkdir(parents=True)
    login.write_text(LOGIN)

    async def execute(command, **kwargs):
        if "/proc/self/mountinfo" in command:
            from tests.sandbox_mounts import ROOT_MOUNTINFO

            return ExecResult(0, ROOT_MOUNTINFO, "")
        if "printf 'complete\\n'" in command:
            return ExecResult(0, "complete\n", "")
        if "rm -f" in command and ".config/muse/auth.json" in command:
            assert events == ["quiesced"]
            assert kwargs["user"] == "agent"
            events.append("removed")
            result = subprocess.run(
                ["sh", "-c", command.replace("/home/agent", str(home))],
                capture_output=True,
                text=True,
            )
            return ExecResult(result.returncode, result.stdout, result.stderr)
        return ExecResult(0, "", "")

    env.exec = AsyncMock(side_effect=receipt_aware(execute))
    monkeypatch.setattr(
        lockdown,
        "_kill_sandbox_user_procs",
        AsyncMock(side_effect=lambda *a: events.append("quiesced")),
    )
    for name in (
        "ensure_legacy_app_dir",
        "_reclaim_disk",
        "cleanup_verifier_python_hooks",
    ):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(
        lockdown,
        "clear_verifier_output_dir",
        AsyncMock(side_effect=lambda *a, **kw: events.append("verifier")),
    )
    monkeypatch.setattr(lockdown, "_read_hardening_config", lambda *a: {})
    monkeypatch.setattr(lockdown, "_build_verifier_env", AsyncMock(return_value={}))
    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(env={}, service="main")),
    )
    await lockdown.harden_before_verify(env, task, "agent")
    assert events == ["quiesced", "removed", "verifier"]
    assert not login.exists()


async def test_cleanup_failure_blocks_disconnect(tmp_path):
    """Guards AG-C3 at bf2d8c83: failed unlink is not a successful role end."""
    trial = _rollout(tmp_path)
    await trial.connect()
    trial._env.exec.return_value = ExecResult(1, "", "fake failure")
    with pytest.raises(RuntimeError, match="credential cleanup failed"):
        await trial.disconnect()


@pytest.mark.parametrize(
    "receipt",
    [None, SimpleNamespace(return_code=False), SimpleNamespace(return_code="0")],
)
async def test_disconnect_requires_an_integer_cleanup_receipt(tmp_path, receipt):
    """Guards AG-C3 at bf2d8c83: missing or untyped success is not removal."""
    trial = _rollout(tmp_path)
    await trial.connect()
    trial._env.exec.return_value = receipt
    with pytest.raises(RuntimeError, match="credential cleanup failed"):
        await trial.disconnect()


async def test_disconnect_accepts_the_legacy_integer_status_alias(tmp_path):
    """Guards AG-C3 at bf2d8c83 without breaking the sandbox status alias."""
    trial = _rollout(tmp_path)
    await trial.connect()
    trial._env.exec.reset_mock()
    trial._env.exec.return_value = SimpleNamespace(exit_code=0)
    await trial.disconnect()
    assert any(
        ".config/muse/auth.json" in call.args[0] and call.kwargs.get("user") == "agent"
        for call in trial._env.exec.await_args_list
    )


async def test_muse_connection_pins_both_home_overrides(tmp_path):
    """Guards AG-C2 at bf2d8c83 against a second managed-home override."""
    trial = _rollout(tmp_path)
    trial._agent_env.update(
        {
            "BENCHFLOW_AGENT_HOME": "/tmp/fake-workspace",
            "XDG_CONFIG_HOME": "/tmp/fake-workspace/runtime-config",
        }
    )
    await trial.connect()
    env = trial._planes.connect_acp.await_args.kwargs["agent_env"]
    assert env["BENCHFLOW_AGENT_HOME"] == "/home/agent"
    assert env["XDG_CONFIG_HOME"] == "/home/agent/.config"


def test_a_rollout_built_without_init_has_no_muse_login_staged(tmp_path):
    """A bare Rollout (Rollout.__new__, an SDK subclass skipping __init__)
    reads as having no Muse login, so verify() and disconnect() do not fail
    on the flag before reaching their own work (integrator batch 4)."""
    bare = Rollout.__new__(Rollout)
    assert bare._muse_credentials_staged is False
    # A constructed Rollout keeps its own flag; setting it leaves the default.
    trial = Rollout(RolloutConfig(task_path=tmp_path, agent="opencode"))
    assert vars(trial)["_muse_credentials_staged"] is False
    trial._muse_credentials_staged = True
    assert Rollout.__new__(Rollout)._muse_credentials_staged is False
