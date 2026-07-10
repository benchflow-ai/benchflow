"""Fail-closed checks for BenchFlow's LiteLLM 5xx retry patch."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from benchflow.providers.litellm_bedrock_preflight import (
    _exec_details,
    _host_python_for_litellm,
)


class RetryPatchPreflightError(RuntimeError):
    """Raised when the transient-5xx retry patch cannot be proven active."""


RETRY_PATCH_PREFLIGHT_SOURCE = """\
import sys

failures = []
try:
    import litellm
    import litellm.router as router_mod

    helper = getattr(router_mod, "_get_num_retries_from_retry_policy", None)
    if not getattr(helper, "__benchflow_retry_patch__", False):
        failures.append("transient-5xx retry helper is unpatched")
    else:
        policy = {"InternalServerErrorRetries": 2}
        transient = litellm.InternalServerError(
            "upstream 500",
            llm_provider="openai",
            model="openai/test",
        )
        retries = helper(exception=transient, retry_policy=policy)
        if retries != 2:
            failures.append(
                f"InternalServerError retry probe returned {retries!r}, expected 2"
            )

        permanent = litellm.BadRequestError(
            "bad request",
            llm_provider="openai",
            model="openai/test",
        )
        permanent_retries = helper(exception=permanent, retry_policy=policy)
        if permanent_retries is not None:
            failures.append(
                "BadRequestError retry probe returned "
                f"{permanent_retries!r}, expected fail-fast None"
            )
except Exception as exc:  # noqa: BLE001 - report any LiteLLM shape drift
    failures.append(f"retry preflight probe failed: {exc}")

if failures:
    print("; ".join(failures))
    sys.exit(1)
"""


def _failure_message(runtime: str, detail: str) -> str:
    return (
        "BenchFlow LiteLLM transient-5xx retry patch is NOT active in the "
        f"{runtime} LiteLLM runtime: {detail[:2000]}. Failing closed before "
        "agent launch - a silent fallback would expose transient upstream "
        "500s to agents instead of retrying them at the proxy."
    )


def preflight_host_retry_patch(
    *,
    env: dict[str, str],
    litellm_executable: str,
) -> None:
    """Fail closed if the retry patch is not active for the host proxy."""
    try:
        result = subprocess.run(
            [
                _host_python_for_litellm(litellm_executable, env=env),
                "-c",
                RETRY_PATCH_PREFLIGHT_SOURCE,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        raise RetryPatchPreflightError(
            _failure_message("host", f"preflight execution failed: {exc}")
        ) from exc
    if result.returncode != 0:
        detail = (result.stdout or "").strip() or (result.stderr or "").strip()
        raise RetryPatchPreflightError(_failure_message("host", detail))


async def preflight_sandbox_retry_patch(
    sandbox: Any,
    *,
    python: str,
    runtime_dir: str,
    preflight_path: str,
) -> None:
    """Fail closed if the retry patch is not active in a sandbox proxy."""
    command = (
        f"PYTHONPATH={shlex.quote(runtime_dir)} "
        f"{shlex.quote(python)} {shlex.quote(preflight_path)}"
    )
    try:
        result = await sandbox.exec(command, timeout_sec=120)
    except Exception as exc:
        raise RetryPatchPreflightError(
            _failure_message("sandbox", f"preflight execution failed: {exc}")
        ) from exc
    if result.return_code != 0:
        raise RetryPatchPreflightError(
            _failure_message(
                "sandbox",
                _exec_details("retry patch preflight", result),
            )
        )
