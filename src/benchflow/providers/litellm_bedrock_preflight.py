"""Fail-closed checks for BenchFlow's Bedrock Claude 4.8+ LiteLLM patch."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchflow.providers.litellm_config import LiteLLMRoute


class BedrockPatchPreflightError(RuntimeError):
    """Raised when the Bedrock patch cannot be proven active."""


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


def _python_from_shebang(executable: Path, *, path_env: str | None) -> str | None:
    try:
        line = executable.open("rb").readline(4096).decode(errors="ignore").strip()
    except OSError:
        return None
    if not line.startswith("#!"):
        return None
    try:
        parts = shlex.split(line[2:].strip())
    except ValueError:
        return None
    if not parts:
        return None

    command = parts[0]
    if Path(command).name == "env":
        args = parts[1:]
        while args and args[0].startswith("-"):
            args = args[1:]
        if not args:
            return None
        command = args[0]

    if not Path(command).name.startswith("python"):
        return None
    if "/" in command:
        return command
    return shutil.which(command, path=path_env) or command


def _host_python_for_litellm(
    litellm_executable: str, *, env: dict[str, str] | None = None
) -> str:
    executable = Path(litellm_executable)
    shebang_python = _python_from_shebang(
        executable, path_env=env.get("PATH") if env else None
    )
    if shebang_python:
        return shebang_python
    sibling = executable.with_name("python")
    if sibling.exists():
        return str(sibling)
    return sys.executable


def _failure_message(runtime: str, detail: str) -> str:
    return (
        "Bedrock Claude 4.8+ adaptive-thinking patch is NOT active in the "
        f"{runtime} LiteLLM runtime: {detail[:2000]}. Failing closed before "
        "agent launch (#602) - a silent fallback would send the legacy thinking "
        "shape Bedrock rejects."
    )


def preflight_host_bedrock_patch(
    *,
    env: dict[str, str],
    litellm_executable: str,
) -> None:
    """Fail closed if the Bedrock 4.8+ patch is not active for the host proxy."""
    try:
        result = subprocess.run(
            [
                _host_python_for_litellm(litellm_executable, env=env),
                "-c",
                BEDROCK_PATCH_PREFLIGHT_SOURCE,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        raise BedrockPatchPreflightError(
            _failure_message("host", f"preflight execution failed: {exc}")
        ) from exc
    if result.returncode != 0:
        detail = (result.stdout or "").strip() or (result.stderr or "").strip()
        raise BedrockPatchPreflightError(_failure_message("host", detail))


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
    try:
        result = await sandbox.exec(command, timeout_sec=120)
    except Exception as exc:
        raise BedrockPatchPreflightError(
            _failure_message("sandbox", f"preflight execution failed: {exc}")
        ) from exc
    if result.return_code != 0:
        raise BedrockPatchPreflightError(
            _failure_message(
                "sandbox",
                _exec_details("bedrock patch preflight", result),
            )
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
