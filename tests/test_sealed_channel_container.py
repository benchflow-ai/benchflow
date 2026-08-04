"""Receiver-side sealed-channel tests against a real container.

The unit suite proves sender-side construction; these tests drive the
ACTUAL receiver commands (openssl verify + decrypt, env sourcing, cleanup
traps) inside ``python:3.12-slim``, covering the round-4 PR #942 findings:
plaintext-at-creation mode, split-IV tampering at the receiver boundary,
staged-env ownership, and cleanup across cd-failure and ``exec``
replacement. Skipped when no Docker daemon is available.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import shutil
import subprocess
import uuid

import pytest

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture(scope="module")
def container():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "bf-sealed-test-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, IMAGE, "sleep", "600"],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start container: {run.stderr[:200]}")
    subprocess.run(
        ["docker", "exec", name, "sh", "-c", "useradd -m agent || true"],
        capture_output=True,
    )
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _dx(container: str, command: str, user: str | None = None):
    args = ["docker", "exec"]
    if user:
        args += ["-u", user]
    return subprocess.run(
        [*args, container, "sh", "-c", command], capture_output=True, text=True
    )


@pytest.fixture()
def channel(container):
    from benchflow.sandbox.agentcore_sealed import SealedChannel

    class _R:
        pass

    async def exec_raw(command, *, timeout_sec=None, user=None):
        result = _dx(
            container, command, user=None if user in (None, "root") else str(user)
        )
        r = _R()
        r.return_code = result.returncode
        r.stdout = result.stdout
        r.stderr = result.stderr
        return r

    return SealedChannel(exec_raw, logging.getLogger("test"))


class TestReceiverBoundary:
    def test_secret_upload_lands_0600_with_spaced_parent(self, container, channel):
        """Mode is applied atomically under umask 077 — no 0644 window —
        and parents with spaces are quoted (host-computed, not $(dirname))."""
        target = "/tmp/spaced dir/secret.bin"
        asyncio.run(channel.upload(b"TOKEN=abc\n" * 20, target=target, mode="600"))
        result = _dx(container, "stat -c '%a' '/tmp/spaced dir/secret.bin'")
        assert result.stdout.strip() == "600"

    def test_receiver_rejects_blob_iv_tampering(self, container, channel):
        """Round 4: the receiver's IV comes from the authenticated blob, so
        flipping IV bits in the staged blob fails the tag — the previous
        split-use (HMAC input vs -iv argument) is structurally gone."""
        import benchflow.sandbox.agentcore_sealed as mod

        pem = asyncio.run(channel.public_key())
        sealed = mod.seal(pem, b"attack-me" * 10)
        blob = bytearray(base64.b64decode(sealed.blob_b64))
        blob[0] ^= 1  # IV bit-flip inside the authenticated blob
        bad = dataclasses.replace(
            sealed, blob_b64=base64.b64encode(bytes(blob)).decode()
        )
        original = mod.seal
        mod.seal = lambda *_a, **_k: bad
        try:
            with pytest.raises(RuntimeError):
                asyncio.run(channel.upload(b"ignored", target="/tmp/attack.bin"))
        finally:
            mod.seal = original
        assert _dx(container, "test ! -e /tmp/attack.bin").returncode == 0

    def test_receiver_command_carries_no_iv_argument(self, channel):
        """No second IV copy exists for an attacker to alter independently."""
        commands: list[str] = []
        real = channel._exec_raw

        async def spy(command, **kwargs):
            commands.append(command)
            return await real(command, **kwargs)

        channel._exec_raw = spy
        asyncio.run(channel.upload(b"x", target="/tmp/iv-probe.bin"))
        final = commands[-1]
        assert '-iv "$IVHEX"' in final  # derived from the MAC'd blob
        assert "IVHEX=$(od" in final
        # and no literal hex IV appears as an -iv argument
        import re

        assert not re.search(r"-iv [0-9a-f]{32}", final)


class TestStagedEnvBehavior:
    def test_agent_user_sources_env_and_file_is_removed(self, container, channel):
        env_path = asyncio.run(channel.stage_env({"TOK": "sekrit-1"}, owner="agent"))
        # ownership: the exec user, not root, must be able to read it
        result = _dx(
            container,
            f"trap 'rm -f {env_path}' EXIT; set -a; . {env_path} || exit 97; "
            f'set +a; rm -f {env_path}; printf %s "$TOK"',
            user="agent",
        )
        assert result.stdout == "sekrit-1"
        assert _dx(container, f"test ! -e {env_path}").returncode == 0

    def test_cd_failure_does_not_leak_env_file(self, container, channel):
        """Trap installs before cd, so a failed cd still cleans up."""
        env_path = asyncio.run(channel.stage_env({"A": "1"}, owner="agent"))
        result = _dx(
            container,
            f"trap 'rm -f {env_path}' EXIT; set -a; . {env_path} || exit 97; "
            f"set +a; rm -f {env_path}; cd /nonexistent && echo RAN",
            user="agent",
        )
        assert result.returncode != 0
        assert "RAN" not in result.stdout
        assert _dx(container, f"test ! -e {env_path}").returncode == 0

    def test_exec_replacement_does_not_leak_env_file(self, container, channel):
        """The file is removed inline before the user command, so a command
        that ``exec``s (EXIT trap never fires) cannot leave it behind."""
        env_path = asyncio.run(channel.stage_env({"B": "2"}, owner="agent"))
        result = _dx(
            container,
            f"trap 'rm -f {env_path}' EXIT; set -a; . {env_path} || exit 97; "
            f'set +a; rm -f {env_path}; exec printf %s "$B"',
            user="agent",
        )
        assert result.stdout == "2"
        assert _dx(container, f"test ! -e {env_path}").returncode == 0

    def test_unreadable_env_aborts_instead_of_running_without_env(self, container):
        _dx(
            container,
            "printf 'X=1\\n' > /tmp/rootonly.sh && chmod 600 /tmp/rootonly.sh",
        )
        result = _dx(
            container,
            "trap 'rm -f /tmp/rootonly.sh' EXIT; set -a; "
            ". /tmp/rootonly.sh || exit 97; set +a; echo SHOULD_NOT_PRINT",
            user="agent",
        )
        assert result.returncode != 0
        assert "SHOULD_NOT_PRINT" not in result.stdout
