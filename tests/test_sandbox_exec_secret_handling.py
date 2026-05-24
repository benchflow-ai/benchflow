"""Regression tests for sandbox exec secret handling (issue #412).

Sandbox exec helpers must never serialize raw secret values into the remote
process argv or command string. ``ps``, shell history, and provider-side
command audit logs all see argv, so a leak there is equivalent to publishing
the secret.

DockerSandbox.exec already routes env through a base64-encoded file decoded
inside the container (covered by ``tests/test_sandbox.py``). This module
covers the Daytona path, where ``_sandbox_exec`` previously emitted
``env KEY=value`` argv.
"""

from __future__ import annotations

import base64

import pytest

from benchflow.sandbox.daytona import _wrap_daytona_command_with_env_file


class TestDaytonaExecEnvSecrecy:
    """Daytona ``_sandbox_exec`` must not inline secret values into argv."""

    def test_wrap_does_not_inline_secret_values(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-very-secret",
            "AUTHORIZATION": "Bearer hunter2",
            "SAFE_VAR": "ordinary-value",
        }
        wrapped = _wrap_daytona_command_with_env_file(env, "run-verifier")

        # No raw value appears in the wrapper text — ``ps aux`` on the
        # Daytona worker would otherwise reveal them. The values live only
        # inside a base64 blob.
        assert "sk-very-secret" not in wrapped
        assert "hunter2" not in wrapped
        # The ordinary value is *also* base64-encoded; this is a side effect
        # of putting the whole env dict into one file, and is the desired
        # behavior because it makes the wrapper indifferent to whether a
        # given key is sensitive.
        assert "ordinary-value" not in wrapped

        # The command sources a file and cleans up on EXIT.
        assert wrapped.startswith("trap 'rm -f ")
        assert "EXIT" in wrapped
        assert "base64 -d" in wrapped
        assert "(umask 077 &&" in wrapped
        assert wrapped.endswith("run-verifier")

    def test_wrap_decoded_env_contains_all_values(self) -> None:
        """The base64 payload sourced inside the sandbox must carry the env.

        We decode the blob the same way the remote shell does and assert the
        sourceable script exports each key/value pair.
        """
        env = {
            "OPENAI_API_KEY": "sk-very-secret",
            "AUTHORIZATION": "Bearer hunter2",
        }
        wrapped = _wrap_daytona_command_with_env_file(env, "tool")

        token = wrapped.split("printf %s ", 1)[1].split(" |", 1)[0]
        encoded = token.strip("'")
        body = base64.b64decode(encoded).decode()

        assert "export OPENAI_API_KEY=" in body
        assert "sk-very-secret" in body
        assert "export AUTHORIZATION=" in body
        assert "hunter2" in body

    def test_wrap_skips_non_identifier_env_keys(self) -> None:
        """Keys that aren't valid POSIX identifiers cannot be ``export``-ed.

        Emitting them would break ``. envfile`` and the user command would
        never run, so they are dropped with a warning (matches the docker
        wrapper's behavior).
        """
        env = {
            "VALID_KEY": "keep-me",
            "dotted.key": "drop-me",
            "dashed-key": "drop-me-too",
        }
        wrapped = _wrap_daytona_command_with_env_file(env, "tool")

        token = wrapped.split("printf %s ", 1)[1].split(" |", 1)[0]
        encoded = token.strip("'")
        body = base64.b64decode(encoded).decode()

        assert "export VALID_KEY=" in body
        assert "dotted.key" not in body
        assert "dashed-key" not in body
        # User command still reachable.
        assert wrapped.endswith("tool")

    @pytest.mark.asyncio
    async def test_sandbox_exec_passes_no_env_kv_argv(self, monkeypatch) -> None:
        """``_sandbox_exec`` must not emit ``env K=V ...`` argv anywhere.

        Stubs ``execute_session_command`` and inspects the command string
        that would be sent to the Daytona session API.
        """
        from benchflow.sandbox.daytona import DaytonaSandbox

        sandbox = DaytonaSandbox.__new__(DaytonaSandbox)

        captured: dict[str, str] = {}

        class _Resp:
            cmd_id = "fake-cmd-id"
            exit_code = 0
            result = ""

        class _ProcessAPI:
            async def create_session(self, session_id: str) -> None:
                return None

            async def execute_session_command(self, session_id, request, timeout=None):
                captured["command"] = request.command
                return _Resp()

        class _FakeSandbox:
            process = _ProcessAPI()

        sandbox._sandbox = _FakeSandbox()

        async def fake_poll(session_id, command_id, timeout_sec=None):  # type: ignore[no-untyped-def]
            from benchflow.sandbox._base import ExecResult

            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(sandbox, "_poll_response", fake_poll, raising=False)

        await sandbox._sandbox_exec(
            "verify-now",
            env={"OPENAI_API_KEY": "sk-leak", "AUTHORIZATION": "Bearer leak"},
        )

        cmd = captured["command"]
        # The argv must not contain raw secret values...
        assert "sk-leak" not in cmd
        assert "Bearer leak" not in cmd
        # ...nor the old ``env KEY=value KEY2=value2 ...`` argv prefix.
        # (We allow the *string* "env" appearing in ``OPENAI_API_KEY`` etc.;
        # the regression marker is ``env KEY=`` adjacency at a word boundary.)
        assert "env OPENAI_API_KEY=" not in cmd
        assert "env AUTHORIZATION=" not in cmd
