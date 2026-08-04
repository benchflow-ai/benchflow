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
import shlex
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


def _dx(container: str, command: str, user: str | None = None, *, shell: str = "sh"):
    args = ["docker", "exec"]
    if user:
        args += ["-u", user]
    return subprocess.run(
        [*args, container, shell, "-c", command], capture_output=True, text=True
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
        """Guards PR #942: spaced parents retain the requested private mode."""
        target = "/tmp/spaced dir's/secret.bin"
        asyncio.run(channel.upload(b"TOKEN=abc\n" * 20, target=target, mode="600"))
        result = _dx(container, f"stat -c %a {shlex.quote(target)}")
        assert result.stdout.strip() == "600"

    def test_existing_target_is_replaced_from_private_temporary(
        self, container, channel
    ):
        """Guards PR #942: overwrite cannot expose plaintext through mode 0644."""

        target = "/tmp/existing-secret.bin"
        _dx(container, f"printf old > {target} && chmod 644 {target}")
        before = _dx(container, f"stat -c '%i:%a' {target}").stdout.strip()
        asyncio.run(channel.upload(b"replacement", target=target, mode="600"))
        after = _dx(container, f"stat -c '%i:%a' {target}").stdout.strip()
        assert before.split(":", 1)[0] != after.split(":", 1)[0]
        assert after.endswith(":600")
        assert _dx(container, f"cat {target}").stdout == "replacement"

    def test_existing_target_symlink_is_replaced_not_followed(self, container, channel):
        """Guards PR #942: root decryption must not overwrite a symlink victim."""

        _dx(
            container,
            "printf victim > /tmp/sealed-victim && "
            "ln -sf /tmp/sealed-victim /tmp/sealed-target",
        )
        asyncio.run(
            channel.upload(b"replacement", target="/tmp/sealed-target", mode="600")
        )
        assert _dx(container, "test ! -L /tmp/sealed-target").returncode == 0
        assert _dx(container, "cat /tmp/sealed-target").stdout == "replacement"
        assert _dx(container, "cat /tmp/sealed-victim").stdout == "victim"

    def test_invalid_mode_is_rejected_before_receiver_execution(
        self, container, channel
    ):
        """Guards PR #942: upload mode cannot append a root shell command."""

        with pytest.raises(ValueError, match="octal"):
            asyncio.run(
                channel.upload(
                    b"x",
                    target="/tmp/invalid-mode-target",
                    mode="600; touch /tmp/mode-injected",
                )
            )
        assert _dx(container, "test ! -e /tmp/mode-injected").returncode == 0

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

    def test_agentcore_exec_uses_production_env_wrapper(self, container, channel):
        """Guards PR #942 across owner, cwd, exec, and source-failure boundaries."""

        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.agentcore import AgentCoreSandbox

        sandbox = object.__new__(AgentCoreSandbox)
        sandbox.runtime_arn = "arn:test"
        sandbox.runtime_session_id = "s" * 40
        sandbox.default_user = "agent"
        sandbox._persistent_env = {}
        sandbox._sealed = channel
        staged: list[str] = []
        real_stage_env = channel.stage_env

        async def record_stage_env(env, *, owner=None):
            path = await real_stage_env(env, owner=owner)
            staged.append(path)
            return path

        async def dispatch(command, *, timeout_sec, resolved_user):
            result = _dx(
                container,
                command,
                user=None if resolved_user in (None, "root") else str(resolved_user),
                shell="bash",
            )
            return ExecResult(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        channel.stage_env = record_stage_env
        sandbox._dispatch_command = dispatch

        sourced = asyncio.run(
            sandbox.exec('printf %s "$OWNER_TEST"', env={"OWNER_TEST": "ok"})
        )
        assert sourced.stdout == "ok"
        assert _dx(container, f"test ! -e {staged[-1]}").returncode == 0

        failed_cd = asyncio.run(
            sandbox.exec("echo SHOULD_NOT_RUN", cwd="/nonexistent", env={"A": "1"})
        )
        assert failed_cd.return_code != 0
        assert "SHOULD_NOT_RUN" not in (failed_cd.stdout or "")
        assert _dx(container, f"test ! -e {staged[-1]}").returncode == 0

        replaced = asyncio.run(sandbox.exec('exec printf %s "$B"', env={"B": "2"}))
        assert replaced.stdout == "2"
        assert _dx(container, f"test ! -e {staged[-1]}").returncode == 0

        _dx(
            container,
            "printf 'X=1\\n' > /tmp/rootonly.sh && chmod 600 /tmp/rootonly.sh",
        )

        async def unreadable_env(_env, *, owner=None):
            return "/tmp/rootonly.sh"

        channel.stage_env = unreadable_env
        unreadable = asyncio.run(sandbox.exec("echo SHOULD_NOT_PRINT", env={"X": "1"}))
        assert unreadable.return_code != 0
        assert "SHOULD_NOT_PRINT" not in (unreadable.stdout or "")
        _dx(container, "rm -f /tmp/rootonly.sh")
