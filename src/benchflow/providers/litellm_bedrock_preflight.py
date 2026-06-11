"""Fail-closed checks for BenchFlow's Bedrock Claude 4.8+ LiteLLM patch."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchflow.providers.litellm_config import LiteLLMRoute

# Behavioral preflight for the Bedrock Claude 4.8+ adaptive-thinking patch
# (#602). Runs in a fresh interpreter with the same env/PYTHONPATH as the proxy,
# so site loads the runtime dir's sitecustomize exactly like the proxy process
# does. It deliberately imports only litellm internals, never the patch module
# itself, which would apply the patches in-process and mask a load failure.
BEDROCK_PATCH_PREFLIGHT_SOURCE = """\
import sys

# Probe with a versioned Bedrock inference-profile ID: stock litellm 1.88.0rc1
# resolves the bare alias through its cost map, so only the versioned form
# discriminates patched from stock (#602).
PROBE = "us.anthropic.claude-opus-4-8-20251101-v1:0"

failures = []
try:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    if not AnthropicConfig._is_adaptive_thinking_model(PROBE):
        failures.append(
            "adaptive-thinking gate inactive: "
            f"_is_adaptive_thinking_model({PROBE!r}) is False"
        )
except Exception as exc:  # noqa: BLE001 - report any import/shape drift
    failures.append(f"anthropic transform unavailable: {exc}")
try:
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    handler = AmazonConverseConfig._handle_reasoning_effort_parameter
    if not getattr(handler, "__benchflow_bedrock_patch__", False):
        failures.append("reasoning-effort override inactive")
except Exception as exc:  # noqa: BLE001 - report any import/shape drift
    failures.append(f"bedrock converse transform unavailable: {exc}")
if failures:
    print("; ".join(failures))
    sys.exit(1)
"""


def route_requires_bedrock_patch(route: LiteLLMRoute) -> bool:
    """True when this resolved route depends on the Bedrock 4.8+ patch (#602)."""
    return (
        route.provider_name == "aws-bedrock"
        and "reasoning_effort" in route.litellm_params
    )


def _host_python_for_litellm(litellm_executable: str) -> str:
    sibling = Path(litellm_executable).with_name("python")
    if sibling.exists():
        return str(sibling)
    return sys.executable


def preflight_host_bedrock_patch(
    *,
    env: dict[str, str],
    litellm_executable: str,
) -> None:
    """Fail closed if the Bedrock 4.8+ patch is not active for the host proxy."""
    result = subprocess.run(
        [
            _host_python_for_litellm(litellm_executable),
            "-c",
            BEDROCK_PATCH_PREFLIGHT_SOURCE,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stdout or "").strip() or (result.stderr or "").strip()
        raise RuntimeError(
            "Bedrock Claude 4.8+ adaptive-thinking patch is NOT active in the "
            f"host LiteLLM runtime: {detail[:2000]}. Failing closed before agent "
            "launch (#602) - a silent fallback would send the legacy thinking "
            "shape Bedrock rejects."
        )


async def preflight_sandbox_bedrock_patch(
    sandbox: Any,
    *,
    python: str,
    runtime_dir: str,
    preflight_path: str,
) -> None:
    """Fail closed if the Bedrock 4.8+ patch is not active in a sandbox proxy."""
    command = (
        f"PYTHONPATH={shlex.quote(runtime_dir)} "
        f"{shlex.quote(python)} {shlex.quote(preflight_path)}"
    )
    result = await sandbox.exec(command, timeout_sec=120)
    if result.return_code != 0:
        raise RuntimeError(
            "Bedrock Claude 4.8+ adaptive-thinking patch is NOT active in the "
            "sandbox LiteLLM runtime: "
            f"{_exec_details('bedrock patch preflight', result)}. Failing closed "
            "before agent launch (#602) - a silent fallback would send the "
            "legacy thinking shape Bedrock rejects."
        )


def _exec_details(label: str, result: Any) -> str:
    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    details = [f"{label} failed with exit code {getattr(result, 'return_code', '?')}"]
    if stdout:
        details.append(f"stdout: {stdout[:2000]}")
    if stderr:
        details.append(f"stderr: {stderr[:2000]}")
    return "; ".join(details)
