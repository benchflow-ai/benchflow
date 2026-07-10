"""Regression coverage for the BenchFlow LiteLLM retry patch."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from benchflow.providers import litellm_retry_preflight as preflight_mod
from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import resolve_litellm_route


def test_runtime_retry_patch_retries_only_transient_5xx(tmp_path):
    """Guards PR #882: LiteLLM Router retries upstream 500s, but not 400s."""
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    probe = textwrap.dedent(
        r"""
        import asyncio

        import litellm
        from litellm import Router
        from litellm.utils import ModelResponse


        async def healthy_deployments(*args, **kwargs):
            return ([], [{"litellm_params": {}}])


        def router():
            routed = Router(
                model_list=[
                    {
                        "model_name": "benchflow-test",
                        "litellm_params": {
                            "model": "openai/test",
                            "api_key": "sk-test",
                        },
                    }
                ],
                num_retries=0,
                retry_policy={"InternalServerErrorRetries": 2},
                disable_cooldowns=True,
                retry_after=0,
            )
            routed._async_get_healthy_deployments = healthy_deployments
            routed._time_to_sleep_before_retry = lambda **kwargs: 0
            return routed


        async def main():
            routed = router()
            calls = 0

            async def transient_then_success(original_function, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise litellm.InternalServerError(
                        "upstream 500",
                        llm_provider="openai",
                        model="openai/test",
                    )
                return ModelResponse(model="openai/test", choices=[])

            routed.make_call = transient_then_success
            response = await routed.async_function_with_retries(
                original_function=lambda: None,
                model="benchflow-test",
                messages=[],
                metadata={},
                num_retries=0,
            )
            assert calls == 3, calls
            headers = response._hidden_params["additional_headers"]
            assert headers["x-litellm-attempted-retries"] == 2, headers
            assert headers["x-litellm-max-retries"] == 2, headers

            async def assert_fails_fast(error_factory, expected_type):
                routed = router()
                calls = 0

                async def fail(original_function, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    raise error_factory()

                routed.make_call = fail
                try:
                    await routed.async_function_with_retries(
                        original_function=lambda: None,
                        model="benchflow-test",
                        messages=[],
                        metadata={},
                        num_retries=0,
                    )
                except expected_type:
                    pass
                else:
                    raise AssertionError(f"{expected_type.__name__} should fail fast")
                assert calls == 1, calls

            await assert_fails_fast(
                lambda: litellm.BadRequestError(
                    "bad request",
                    llm_provider="openai",
                    model="openai/test",
                ),
                litellm.BadRequestError,
            )
            await assert_fails_fast(
                lambda: litellm.ContextWindowExceededError(
                    "context window exceeded",
                    llm_provider="openai",
                    model="openai/test",
                ),
                litellm.ContextWindowExceededError,
            )


        asyncio.run(main())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_retry_patch_preflight_passes_when_runtime_files_on_pythonpath(tmp_path):
    """Guards PR #882: proxy startup must prove the retry monkeypatch loaded."""
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", preflight_mod.RETRY_PATCH_PREFLIGHT_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_retry_patch_preflight_fails_closed_when_patch_not_loaded(tmp_path):
    """Guards PR #882 against silently booting a proxy without 5xx retry behavior."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", preflight_mod.RETRY_PATCH_PREFLIGHT_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "unpatched" in result.stdout


def test_host_retry_preflight_uses_litellm_runtime_python(tmp_path):
    """Guards PR #882 host proxy startup: preflight runs in the LiteLLM runtime."""
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    litellm = tmp_path / "litellm"
    litellm.write_text(f"#!{sys.executable}\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)

    preflight_mod.preflight_host_retry_patch(
        env=env,
        litellm_executable=str(litellm),
    )


class _ExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeSandbox:
    def __init__(self, *, fail_retry_preflight: bool = False):
        self.uploaded: dict[str, str] = {}
        self.exec_calls: list[str] = []
        self.exec_timeouts: list[int | None] = []
        self.fail_retry_preflight = fail_retry_preflight
        self._started = False

    async def upload_file(self, local_path, remote_path) -> None:
        self.uploaded[str(remote_path)] = Path(local_path).read_text()

    async def exec(self, command: str, timeout_sec: int | None = None) -> _ExecResult:
        self.exec_calls.append(command)
        self.exec_timeouts.append(timeout_sec)
        if "retry_patch_preflight.py" in command:
            if self.fail_retry_preflight:
                return _ExecResult(1, stdout="transient-5xx retry helper is unpatched")
            return _ExecResult(0)
        if "urllib.request" in command:
            return _ExecResult(0)
        if "launcher.py" in command:
            self._started = True
            return _ExecResult(0)
        if command.strip().startswith("cat") and "state.json" in command:
            if self._started:
                return _ExecResult(0, stdout='{"pid": 4242, "port": 45999}')
            return _ExecResult(0, stdout="")
        if command.strip().startswith("rm -rf"):
            return _ExecResult(0)
        return _ExecResult(0)


@pytest.mark.asyncio
async def test_sandbox_litellm_runs_retry_preflight_before_returning_proxy():
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox()

    proc = await runtime_mod._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="sk-master",
        agent_env={
            "MINIMAX_API_KEY": "k",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
        },
        session_id="s",
        agent_name="openhands",
    )

    assert proc.base_url == "http://127.0.0.1:45999"
    retry_preflights = [
        c for c in sandbox.exec_calls if "retry_patch_preflight.py" in c
    ]
    assert retry_preflights, "sandbox LiteLLM must fail-close probe retry patch"


@pytest.mark.asyncio
async def test_sandbox_litellm_retry_preflight_failure_fails_closed():
    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox(fail_retry_preflight=True)

    with pytest.raises(RuntimeError, match="transient-5xx retry patch is NOT active"):
        await runtime_mod._start_sandbox_litellm(
            sandbox=sandbox,
            route=route,
            master_key="sk-master",
            agent_env={
                "MINIMAX_API_KEY": "k",
                "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
            },
            session_id="s",
            agent_name="openhands",
        )

    assert any(call.strip().startswith("rm -rf") for call in sandbox.exec_calls)
