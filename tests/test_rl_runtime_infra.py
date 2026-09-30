"""Sandbox and gateway rules the RL runtime relies on.

Each test pins one gap PostTrain Arena hit running RL on BenchFlow:

- a loopback policy URL was unreachable from a gateway running inside the
  sandbox (no-web tasks on Docker, every Daytona run);
- the ACP handshake timeout was fixed per process and, when it fired, looked
  like the agent running out of time (a scored timeout);
- ``/oracle`` was uploaded for every agent, so a root agent could read it;
- a killed job left its sandboxes running (84 on Daytona once).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import ACP_ERROR, TIMED_OUT, classify_error
from benchflow._utils.startup_timeouts import startup_timeout_overrides
from benchflow.providers.litellm_config import LiteLLMRoute
from benchflow.providers.litellm_runtime import (
    SANDBOX_BASE_URL_ENV,
    _effective_health_deadline,
    route_for_sandbox_gateway,
)
from benchflow.sandbox import leases

REPO = Path(__file__).resolve().parents[1]


def _route(api_base: str) -> LiteLLMRoute:
    return LiteLLMRoute(
        requested_model="vllm/p",
        model_alias="benchflow-vllm-p",
        upstream_model="openai/p",
        provider_name="vllm",
        litellm_params={"model": "openai/p", "api_base": api_base},
    )


# --- the gateway inside a sandbox reaches a loopback policy URL -------------------


def test_loopback_url_is_rewritten_for_a_gateway_in_a_docker_sandbox(monkeypatch):
    import benchflow.providers.litellm_runtime as rt

    monkeypatch.setattr(rt, "_docker_host_address", lambda: "172.17.0.1")
    route = route_for_sandbox_gateway(_route("http://127.0.0.1:8000/v1"), {}, "docker")
    assert route.litellm_params["api_base"] == "http://172.17.0.1:8000/v1"


def test_the_relay_url_for_sandboxes_wins():
    route = route_for_sandbox_gateway(
        _route("http://127.0.0.1:40000/v1"),
        {SANDBOX_BASE_URL_ENV: "https://relay.example.test/v1"},
        "daytona",
    )
    assert route.litellm_params["api_base"] == "https://relay.example.test/v1"


def test_a_loopback_url_on_a_remote_provider_is_refused_with_the_fix():
    with pytest.raises(ValueError, match="relay_public_url"):
        route_for_sandbox_gateway(_route("http://localhost:8000/v1"), {}, "daytona")


def test_a_reachable_url_is_left_alone():
    route = _route("https://policy.example.test/v1")
    assert route_for_sandbox_gateway(route, {}, "daytona") is route


def test_the_sandbox_url_never_reaches_the_agent():
    from benchflow.providers.litellm_runtime import _PROVIDER_ENDPOINT_ENV_NAMES

    assert SANDBOX_BASE_URL_ENV in _PROVIDER_ENDPOINT_ENV_NAMES


# --- startup timeouts: per rollout, and reported for what they are -----------------


def test_startup_timeout_overrides_apply_inside_the_block(monkeypatch):
    from benchflow.acp.runtime import _acp_handshake_timeout_sec

    monkeypatch.delenv("BENCHFLOW_ACP_HANDSHAKE_TIMEOUT", raising=False)
    assert _acp_handshake_timeout_sec() == 60.0
    with startup_timeout_overrides(acp_handshake_sec=240, gateway_sec=30):
        assert _acp_handshake_timeout_sec() == 240
        assert _effective_health_deadline(None) == 30
        assert _effective_health_deadline(5.0) == 5.0
    assert _acp_handshake_timeout_sec() == 60.0
    with (
        pytest.raises(ValueError, match="positive"),
        startup_timeout_overrides(acp_handshake_sec=0),
    ):
        pass


def test_acp_handshake_timeout_is_an_acp_error_not_a_timeout():
    text = "TransportClosedError: ACP initialize timed out after 60s before the first prompt"
    assert classify_error(text) == ACP_ERROR
    assert classify_error("Agent timed out after 900s") == TIMED_OUT


# --- the oracle is never in a non-oracle agent's sandbox ---------------------------


def test_only_the_oracle_and_oracle_access_users_get_oracle_files(tmp_path):
    from benchflow.rollout import RolloutConfig, _uses_oracle_files

    assert _uses_oracle_files(RolloutConfig(task_path=tmp_path, agent="oracle"))
    assert not _uses_oracle_files(RolloutConfig(task_path=tmp_path, agent="nop"))
    assert not _uses_oracle_files(
        RolloutConfig(task_path=tmp_path, agent="claude-agent-acp", sandbox_user=None)
    )


async def test_upload_skips_the_oracle_when_told(tmp_path):
    from benchflow.rollout._setup import _start_env_and_upload

    task = tmp_path / "task"
    (task / "oracle").mkdir(parents=True)
    (task / "oracle" / "solve.sh").write_text("echo solved\n")
    (task / "instruction.md").write_text("do it\n")
    uploads: list[str] = []

    class Env:
        async def start(self, force_build: bool) -> None:
            return None

        async def upload_file(self, source, target) -> None:
            uploads.append(str(target))

        async def upload_dir(self, source, target) -> None:
            uploads.append(str(target))

    await _start_env_and_upload(Env(), task, {}, upload_oracle=False)
    assert uploads == ["/instruction.md"]
    uploads.clear()
    await _start_env_and_upload(Env(), task, {}, upload_oracle=True)
    assert uploads == ["/instruction.md", "/oracle"]


# --- leases: kill-safe cleanup --------------------------------------------------------


@pytest.fixture
def lease_home(tmp_path, monkeypatch):
    directory = tmp_path / "leases"
    monkeypatch.setenv(leases.LEASE_DIR_ENV, str(directory))
    # The registry is process-wide: tests elsewhere that start a mocked
    # sandbox without stopping it leave entries a lease file would list.
    monkeypatch.setattr(leases, "_live", {})
    yield directory
    for entry in leases.live():
        leases.release(entry["provider"], entry["id"])


def test_record_and_release_keep_the_lease_file_current(lease_home):
    leases.record("docker", "proj-a")
    leases.record("daytona", "sb-1")
    [path] = list(lease_home.glob("*.json"))
    data = json.loads(path.read_text())
    assert data["process"] == leases.lease_token()
    assert {(r["provider"], r["id"]) for r in data["resources"]} == {
        ("docker", "proj-a"),
        ("daytona", "sb-1"),
    }
    leases.release("docker", "proj-a")
    leases.release("daytona", "sb-1")
    assert not path.exists()


def test_lease_state_of_this_a_dead_and_a_foreign_process():
    assert leases.lease_state(leases.lease_token()) == "this"
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    me = leases.lease_token().split(":")
    dead = ":".join([me[0], child.stdout.strip(), me[2], "1"])
    assert leases.lease_state(dead) == "gone"
    assert leases.lease_state("another-host:1:abc:1") == "unknown"
    assert leases.lease_state("garbage") == "unknown"


def _child_script(body: str) -> str:
    return textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {str(REPO / "src")!r})
        from benchflow.sandbox import leases
        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
        """
    )


def test_reap_dead_leases_deletes_what_a_killed_process_left(lease_home, monkeypatch):
    # A process records two sandboxes and is SIGKILLed before teardown.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _child_script(
                """
                leases.record("docker", "killed-proj")
                leases.record("daytona", "killed-sb")
                print("ready", flush=True)
                time.sleep(60)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, leases.LEASE_DIR_ENV: str(lease_home)},
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "ready"
    child.kill()
    child.wait()
    deleted: list[str] = []

    def fake_delete(resources, *, timeout_sec=120.0):
        keys = [f"{r['provider']}:{r['id']}" for r in resources]
        deleted.extend(keys)
        return {"deleted": keys, "failed": []}

    monkeypatch.setattr(leases, "delete_resources", fake_delete)
    report = leases.reap_dead_leases()
    assert sorted(deleted) == ["daytona:killed-sb", "docker:killed-proj"]
    assert report["leases"] == 1
    assert list(lease_home.glob("*.json")) == []


def test_a_live_process_lease_is_left_alone(lease_home, monkeypatch):
    leases.record("docker", "mine")
    monkeypatch.setattr(
        leases, "delete_resources", lambda *a, **k: pytest.fail("deleted a live lease")
    )
    assert leases.reap_dead_leases()["leases"] == 0


def test_sigterm_deletes_live_sandboxes_before_the_process_ends(lease_home, tmp_path):
    marker = tmp_path / "deleted.txt"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _child_script(
                f"""
                def fake(resources, *, timeout_sec=120.0):
                    keys = [r["provider"] + ":" + r["id"] for r in resources]
                    open({str(marker)!r}, "w").write("\\n".join(keys))
                    return {{"deleted": keys, "failed": []}}
                leases.delete_resources = fake
                leases.install_signal_cleanup()
                leases.record("daytona", "live-sb")
                leases.record("docker", "live-proj")
                print("ready", flush=True)
                time.sleep(60)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, leases.LEASE_DIR_ENV: str(lease_home)},
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "ready"
    child.send_signal(signal.SIGTERM)
    assert child.wait(timeout=30) == -signal.SIGTERM
    assert sorted(marker.read_text().split()) == ["daytona:live-sb", "docker:live-proj"]
    # Deleted resources leave the lease file, so nothing is reaped twice.
    assert list(lease_home.glob("*.json")) == []


def test_an_ignored_signal_stays_ignored():
    """A trainer under nohup ignores SIGHUP; closing its terminal must not
    delete the sandboxes its rollouts are still using."""
    import signal

    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        uninstall = leases.install_signal_cleanup()
        try:
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is leases._on_signal
        finally:
            uninstall()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, before)


def test_daytona_lifetime_from_the_rollout_then_the_environment(monkeypatch):
    monkeypatch.delenv(leases.DAYTONA_AUTO_STOP_ENV, raising=False)
    monkeypatch.delenv(leases.DAYTONA_AUTO_DELETE_ENV, raising=False)
    assert leases.daytona_lifetime() == (1440, 1440)
    monkeypatch.setenv(leases.DAYTONA_AUTO_STOP_ENV, "45")
    assert leases.daytona_lifetime() == (45, 1440)
    lifetime = leases.SandboxLifetime(
        lease_sec=600, auto_stop_min=30, auto_delete_min=0
    )
    with leases.sandbox_lifetime(lifetime):
        assert leases.daytona_lifetime() == (30, 0)
        labels = leases.lease_labels()
        assert labels[leases.LEASE_LABEL] == leases.lease_token()
        assert abs(int(labels[leases.EXPIRES_LABEL]) - (time.time() + 600)) < 5
    assert leases.EXPIRES_LABEL not in leases.lease_labels()


def test_daytona_reaper_deletes_expired_and_orphaned_leases(monkeypatch):
    from datetime import UTC, datetime

    from benchflow.sandbox.daytona_reaper import (
        _benchflow_managed_value,
        reap_stale_sandboxes,
    )

    monkeypatch.delenv("BENCHFLOW_DAYTONA_OWNER", raising=False)
    managed = {"benchflow.managed": _benchflow_managed_value()}
    now = datetime.now(UTC).isoformat()
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    me = leases.lease_token().split(":")
    dead = ":".join([me[0], child.stdout.strip(), me[2], "1"])
    sandboxes = [
        SimpleNamespace(
            id="expired",
            labels={**managed, leases.EXPIRES_LABEL: "1"},
            created_at=now,
            state="started",
        ),
        SimpleNamespace(
            id="orphan",
            labels={**managed, leases.LEASE_LABEL: dead},
            created_at=now,
            state="started",
        ),
        SimpleNamespace(
            id="live",
            labels={**managed, leases.LEASE_LABEL: leases.lease_token()},
            created_at=now,
            state="started",
        ),
        SimpleNamespace(
            # Same host name, another machine (or boot): not provably dead.
            id="other-boot",
            labels={**managed, leases.LEASE_LABEL: f"{me[0]}:99999999:0000beef:1"},
            created_at=now,
            state="started",
        ),
        SimpleNamespace(
            id="foreign",
            labels={leases.EXPIRES_LABEL: "1"},
            created_at=now,
            state="started",
        ),
    ]
    deleted: list[str] = []
    client = SimpleNamespace(
        list=lambda: sandboxes, delete=lambda sb: deleted.append(sb.id)
    )
    counts = reap_stale_sandboxes(client)
    assert sorted(deleted) == ["expired", "orphan"]
    assert counts["deleted"] == 2
