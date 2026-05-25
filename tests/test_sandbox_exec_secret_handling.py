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
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk

        # ``DaytonaSandbox.__init__`` is what normally materializes the SDK
        # handles ``_sandbox_exec`` consumes (e.g. ``SessionExecuteRequest``).
        # This test bypasses ``__init__`` to keep its setup tiny, so trigger
        # the same lazy-load explicitly.
        _load_daytona_sdk()
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

    @pytest.mark.asyncio
    async def test_daytona_dind_exec_no_secret_in_argv(self) -> None:
        """``_DaytonaDinD.exec`` must not emit ``docker compose exec -e K=V ...``.

        The DinD strategy is the multi-service compose path: ``exec`` runs
        ``docker compose exec -T -e KEY=VALUE ... <service> sh -c <command>``
        inside the DinD VM. Putting ``-e KEY=VALUE`` flags on the argv leaks
        every verifier env var (LLM judge API keys, agent tokens) into the
        DinD VM process list and any provider-side command audit log on every
        multi-service task. Regression for the #412 follow-up.
        """
        pytest.importorskip("daytona")  # sandbox-daytona optional dependency
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]

        await strategy.exec(
            "run-verifier",
            env={"OPENAI_API_KEY": "sk-dind-leak", "AUTHORIZATION": "Bearer dind"},
            service="main",
        )

        sub = captured[0]
        # No raw secret values anywhere in the compose argv.
        joined = " ".join(sub)
        assert "sk-dind-leak" not in joined
        assert "Bearer dind" not in joined
        # And no ``-e KEY=VALUE`` flags at all — those are exactly what
        # would land in ``ps`` and Daytona audit logs.
        assert "-e" not in sub
        assert not any(
            token.startswith("OPENAI_API_KEY=") or token.startswith("AUTHORIZATION=")
            for token in sub
        )
        # The wrapped command is still routed to the target container via
        # ``sh -c <wrapper>`` and the wrapper sources an env file inside it.
        assert sub[-3] == "sh"
        assert sub[-2] == "-c"
        wrapped = sub[-1]
        assert wrapped.startswith("trap 'rm -f ")
        assert "base64 -d" in wrapped
        assert wrapped.endswith("run-verifier")

    @pytest.mark.asyncio
    async def test_daytona_dind_exec_without_env_unchanged(self) -> None:
        """No env -> no wrapping; the user command is passed through verbatim.

        Guards against the wrapper being applied for empty/None env (which
        would change behavior unnecessarily and complicate debugging).
        """
        pytest.importorskip("daytona")
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.daytona import _DaytonaDinD

        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        captured: list[list[str]] = []

        async def fake_compose_exec(subcommand, timeout_sec=None):
            captured.append(subcommand)
            return ExecResult(stdout="", stderr="", return_code=0)

        strategy._compose_exec = fake_compose_exec  # type: ignore[method-assign]

        await strategy.exec("echo hi", service="main")

        sub = captured[0]
        assert sub[-1] == "echo hi"
        assert "-e" not in sub
