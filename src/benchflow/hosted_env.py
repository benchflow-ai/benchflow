"""Hosted environment adapters.

This module handles environments that live in external environment hubs, such
as PrimeIntellect's Verifiers hub. These are not BenchFlow task directories and
they do not use BenchFlow's Docker/Daytona sandbox runner directly. The adapter
keeps their hosted identity intact and runs them through their native Verifiers
execution surface.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRIME_SIMPLE_INDEX = "https://hub.primeintellect.ai/primeintellect/simple/"
PRIME_HUB_ENV_URL = "https://app.primeintellect.ai/dashboard/environments"


class HostedEnvError(RuntimeError):
    """A hosted environment command could not be prepared or executed."""


@dataclass(frozen=True)
class HostedEnvRef:
    """Canonical reference to a hosted environment."""

    provider: str
    owner: str | None
    name: str
    version: str | None = None

    @classmethod
    def parse(
        cls,
        raw: str,
        *,
        version: str | None = None,
        default_provider: str = "primeintellect",
    ) -> HostedEnvRef:
        """Parse a hosted environment reference.

        Supported forms:
        - ``primeintellect/general-agent``
        - ``primeintellect/general-agent@0.1.1``
        - ``primeintellect:primeintellect/general-agent@0.1.1``
        """
        provider = default_provider
        value = raw.strip()
        if not value:
            raise HostedEnvError("--source-env cannot be empty")

        if ":" in value:
            prefix, rest = value.split(":", 1)
            if "/" in rest:
                provider = prefix
                value = rest

        if "@" in value:
            value, embedded_version = value.rsplit("@", 1)
            version = version or embedded_version

        if "/" in value:
            owner, name = value.split("/", 1)
        else:
            owner, name = None, value

        provider = provider.lower()
        if provider != "primeintellect":
            raise HostedEnvError(
                f"Unsupported hosted environment provider: {provider}. "
                "Only primeintellect Verifiers environments are executable today."
            )
        if not name:
            raise HostedEnvError(f"Invalid hosted environment reference: {raw}")
        return cls(provider=provider, owner=owner or None, name=name, version=version)

    @property
    def env_id(self) -> str:
        """Provider-native owner/name reference."""
        return f"{self.owner}/{self.name}" if self.owner else self.name

    @property
    def versioned_env_id(self) -> str:
        """Provider-native owner/name@version reference."""
        return f"{self.env_id}@{self.version}" if self.version else self.env_id

    @property
    def env_uid(self) -> str:
        """BenchFlow identity for hosted environment hubs."""
        suffix = f"@{self.version}" if self.version else "@latest"
        return f"{self.provider}:{self.env_id}{suffix}"

    @property
    def hub_url(self) -> str:
        """Canonical hosted hub URL."""
        if self.provider == "primeintellect" and self.owner:
            return f"{PRIME_HUB_ENV_URL}/{self.owner}/{self.name}"
        return PRIME_HUB_ENV_URL

    @property
    def python_package(self) -> str:
        """Package name used by Prime's simple index."""
        return self.name.replace("-", "_")

    @property
    def verifiers_env_id(self) -> str:
        """Environment id passed to ``vf-eval`` / ``load_environment``."""
        return self.name


@dataclass
class HostedEnvRunConfig:
    """Configuration for running a hosted Verifiers environment."""

    source_env: HostedEnvRef
    model: str
    env_args: dict[str, Any] = field(default_factory=dict)
    agent: str = ""
    jobs_dir: Path = Path("jobs")
    concurrency: int = 1
    num_examples: int = 1
    rollouts_per_example: int = 1
    max_tokens: int = 1024
    temperature: float = 0.0
    sampling_args: dict[str, Any] = field(default_factory=dict)
    python: str = "3.12"
    runner: str = "verifiers"


@dataclass
class HostedEnvRunResult:
    """Result from a hosted environment run."""

    source_env: HostedEnvRef
    run_dir: Path
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    model: str
    normalized_model: str
    reward: float | None = None
    total_tool_calls: int | None = None
    verifiers_error: str | None = None

    @property
    def error(self) -> str | None:
        if self.verifiers_error:
            return self.verifiers_error
        if self.returncode == 0:
            return None
        detail = (self.stderr or self.stdout).strip().splitlines()
        return (
            detail[-1]
            if detail
            else f"hosted environment run failed ({self.returncode})"
        )


def parse_source_env_args(entries: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` source environment args."""
    parsed: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise HostedEnvError(
                f"Invalid --source-env-arg {entry!r}; expected KEY=VALUE"
            )
        key, value = entry.split("=", 1)
        if not key:
            raise HostedEnvError(f"Invalid --source-env-arg {entry!r}; empty key")
        parsed[key] = _parse_scalar(value)
    return parsed


def parse_sampling_args(entries: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` Verifiers sampling args."""
    parsed: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise HostedEnvError(
                f"Invalid --source-env-sampling-arg {entry!r}; expected KEY=VALUE"
            )
        key, value = entry.split("=", 1)
        if not key:
            raise HostedEnvError(
                f"Invalid --source-env-sampling-arg {entry!r}; empty key"
            )
        parsed[key] = _parse_scalar(value)
    return parsed


def normalize_verifiers_model(model: str) -> str:
    """Normalize BenchFlow-style model ids for Prime's Verifiers provider."""
    if "/" in model:
        return model
    if model.startswith("gemini-"):
        return f"google/{model}"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    return model


def run_hosted_env(config: HostedEnvRunConfig) -> HostedEnvRunResult:
    """Run a hosted environment using a controlled local Verifiers install."""
    if config.runner != "verifiers":
        raise HostedEnvError(
            "Only runner='verifiers' is implemented. Use Prime CLI directly for "
            "Prime-hosted runs."
        )
    if not config.source_env.version:
        raise HostedEnvError("--source-env-version is required for reproducible runs")
    if not config.model:
        raise HostedEnvError("--model is required for --source-env runs")

    uv = shutil.which("uv")
    if not uv:
        raise HostedEnvError("uv is required to run hosted Verifiers environments")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.source_env.env_id)
    jobs_dir = config.jobs_dir.expanduser().resolve()
    run_dir = jobs_dir / "hosted-env" / f"{safe_name}__{timestamp}"
    venv_dir = run_dir / ".venv"
    output_dir = run_dir / "vf-results"
    run_dir.mkdir(parents=True, exist_ok=True)

    install_cmd = [
        uv,
        "pip",
        "install",
        "--python",
        _venv_python(venv_dir),
        "--prerelease=allow",
        f"{config.source_env.python_package}=={config.source_env.version}",
        "--extra-index-url",
        PRIME_SIMPLE_INDEX,
    ]
    _run_checked([uv, "venv", "--python", config.python, str(venv_dir)], cwd=run_dir)
    _run_checked(install_cmd, cwd=run_dir)

    normalized_model = normalize_verifiers_model(config.model)
    sampling_args = {"reasoning_effort": "minimal", **config.sampling_args}
    command = [
        str(venv_dir / "bin" / "vf-eval"),
        config.source_env.verifiers_env_id,
        "--env-args",
        json.dumps(config.env_args, sort_keys=True),
        "--num-examples",
        str(config.num_examples),
        "--rollouts-per-example",
        str(config.rollouts_per_example),
        "--max-concurrent",
        str(config.concurrency),
        "--model",
        normalized_model,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
        "--sampling-args",
        json.dumps(sampling_args, sort_keys=True),
        "--output-dir",
        str(output_dir),
        "--save-results",
        "--disable-tui",
    ]
    proc = subprocess.run(
        command,
        cwd=run_dir,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    result = HostedEnvRunResult(
        source_env=config.source_env,
        run_dir=run_dir,
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        model=config.model,
        normalized_model=normalized_model,
        reward=_extract_metric(proc.stdout, "reward"),
        total_tool_calls=_extract_int_metric(proc.stdout, "total_tool_calls"),
        verifiers_error=_extract_verifiers_error(proc.stdout + "\n" + proc.stderr),
    )
    _write_run_artifacts(result, config, install_cmd)
    return result


def prime_env_list(
    *,
    owner: str | None = None,
    search: str | None = None,
    limit: int | None = None,
) -> str:
    """Return Prime environment list JSON."""
    cmd = ["prime", "--plain", "env", "list", "--output", "json"]
    if owner:
        cmd.extend(["--owner", owner])
    if search:
        cmd.extend(["--search", search])
    if limit:
        cmd.extend(["--num", str(limit)])
    return _run_prime(cmd)


def prime_env_info(ref: HostedEnvRef) -> str:
    """Return Prime environment info output."""
    cmd = ["prime", "--plain", "env", "info", ref.env_id]
    if ref.version:
        cmd.extend(["-v", ref.version])
    return _run_prime(cmd)


def prime_env_inspect(ref: HostedEnvRef, path: str = "README.md") -> str:
    """Return a file from a Prime environment package."""
    return _run_prime(
        ["prime", "--plain", "env", "inspect", ref.versioned_env_id, path]
    )


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _run_prime(cmd: list[str]) -> str:
    if not shutil.which("prime"):
        raise HostedEnvError("prime CLI is not installed")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise HostedEnvError(detail or f"prime command failed: {' '.join(cmd)}")
    return proc.stdout


def _run_checked(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise HostedEnvError(detail or f"command failed: {' '.join(cmd)}")


def _venv_python(venv_dir: Path) -> str:
    return str(venv_dir / "bin" / "python")


def _extract_metric(text: str, name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}:\s+avg\s+-\s+([-+]?\d*\.?\d+)", text, re.M)
    return float(match.group(1)) if match else None


def _extract_int_metric(text: str, name: str) -> int | None:
    value = _extract_metric(text, name)
    return int(value) if value is not None else None


def _extract_verifiers_error(text: str) -> str | None:
    aborted = re.search(r"Aborted rollout due to (.+)", text)
    if aborted:
        return aborted.group(1).strip()
    if re.search(r"stop_conditions:\s+has_error:", text):
        return "Verifiers reported has_error stop condition"
    return None


def _write_run_artifacts(
    result: HostedEnvRunResult,
    config: HostedEnvRunConfig,
    install_cmd: list[str],
) -> None:
    (result.run_dir / "stdout.log").write_text(result.stdout)
    (result.run_dir / "stderr.log").write_text(result.stderr)
    payload = {
        "source_env": result.source_env.env_id,
        "source_env_version": result.source_env.version,
        "env_uid": result.source_env.env_uid,
        "hub_url": result.source_env.hub_url,
        "runner": config.runner,
        "agent": config.agent or None,
        "model": result.model,
        "normalized_model": result.normalized_model,
        "env_args": config.env_args,
        "returncode": result.returncode,
        "rewards": {"reward": result.reward} if result.reward is not None else None,
        "total_tool_calls": result.total_tool_calls,
        "verifiers_error": result.verifiers_error,
        "error": result.error,
        "command": result.command,
        "install_command": install_cmd,
    }
    (result.run_dir / "result.json").write_text(json.dumps(payload, indent=2))
