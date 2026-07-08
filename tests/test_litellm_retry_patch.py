"""Regression coverage for the BenchFlow LiteLLM retry patch."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from benchflow.providers import litellm_runtime as runtime_mod


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
