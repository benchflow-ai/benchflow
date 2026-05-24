"""Hosted environment adapters.

This module handles environments that live in external environment hubs, such
as PrimeIntellect's Verifiers hub. These are not BenchFlow task directories and
they do not use BenchFlow's Docker/Daytona sandbox runner directly. The adapter
keeps their hosted identity intact and runs them through their native Verifiers
execution surface.

Hosted runs share the rollout artifact contract — ``result.json``,
``rewards.jsonl``, ``trajectory/acp_trajectory.jsonl``, ``config.json``,
``timing.json``, and ``prompts.json`` — so dashboards and release checks can
treat them as first-class evidence (with ``source.type="hosted_env"`` and
``trajectory_source="hosted_env"`` marking the lineage). Raw vf-eval evidence
is preserved under the ``hosted_env/`` subdir for forensics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

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
        - ``primeintellect:general-agent@0.1.1``
        - ``primeintellect:primeintellect/general-agent@0.1.1``
        """
        provider = default_provider
        value = raw.strip()
        if not value:
            raise HostedEnvError("--source-env cannot be empty")

        if ":" in value:
            prefix, rest = value.split(":", 1)
            if not prefix or not rest:
                raise HostedEnvError(f"Invalid hosted environment reference: {raw}")
            provider = prefix
            value = rest

        if "@" in value:
            value, embedded_version = value.rsplit("@", 1)
            version = version or embedded_version

        if ":" in value:
            raise HostedEnvError(
                f"Invalid hosted environment reference: {raw}. "
                "Use provider:owner/name or owner/name."
            )

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
    return _parse_key_value_entries(entries, "--source-env-arg")


def parse_sampling_args(entries: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` Verifiers sampling args."""
    return _parse_key_value_entries(entries, "--source-env-sampling-arg")


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

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S-%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.source_env.env_id)
    jobs_dir = config.jobs_dir.expanduser().resolve()
    run_id = f"{timestamp}__pid-{os.getpid()}__{uuid4().hex[:8]}"
    run_dir = jobs_dir / "hosted-env" / f"{safe_name}__{run_id}"
    venv_dir = run_dir / ".venv"
    output_dir = run_dir / "vf-results"
    run_dir.mkdir(parents=True, exist_ok=False)

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
    sampling_args = dict(config.sampling_args)
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
    started_at = datetime.now(UTC)
    proc = subprocess.run(
        command,
        cwd=run_dir,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    finished_at = datetime.now(UTC)
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
    _write_run_artifacts(
        result,
        config,
        install_cmd,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )
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


def _parse_key_value_entries(
    entries: list[str] | None,
    flag_name: str,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise HostedEnvError(f"Invalid {flag_name} {entry!r}; expected KEY=VALUE")
        key, value = entry.split("=", 1)
        if not key:
            raise HostedEnvError(f"Invalid {flag_name} {entry!r}; empty key")
        parsed[key] = _parse_scalar(value)
    return parsed


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
    *,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Write rollout-contract artifacts plus hosted-env-specific evidence.

    The contract files (``result.json``, ``rewards.jsonl``,
    ``trajectory/acp_trajectory.jsonl``, ``config.json``, ``timing.json``,
    ``prompts.json``) match the schema produced by
    ``benchflow.rollout._build_rollout_result`` so downstream tools treat
    hosted runs as equivalent to native rollouts. Raw vf-eval evidence
    (stdout, stderr, hosted-env metadata) stays under ``hosted_env/``.
    """
    (result.run_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    (result.run_dir / "agent").mkdir(parents=True, exist_ok=True)
    (result.run_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (result.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    hosted_dir = result.run_dir / "hosted_env"
    hosted_dir.mkdir(parents=True, exist_ok=True)

    # Hosted-env-specific evidence (forensics, debugging).
    (hosted_dir / "stdout.log").write_text(result.stdout)
    (hosted_dir / "stderr.log").write_text(result.stderr)
    hosted_payload = {
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
        "output_dir": str(output_dir),
    }
    (hosted_dir / "hosted_run.json").write_text(json.dumps(hosted_payload, indent=2))

    # Rollout-contract artifacts.
    trajectory = _reconstruct_trajectory(output_dir, started_at)
    rewards = _build_rewards_dict(result, output_dir)
    prompts = _collect_prompts(output_dir, config)
    timing = {"total": round((finished_at - started_at).total_seconds(), 1)}
    task_name = result.source_env.env_uid
    rollout_name = result.run_dir.name
    source_provenance = _hosted_source_provenance(
        result.source_env,
        runner=config.runner,
        env_args=config.env_args,
    )

    (result.run_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in trajectory)
        + ("\n" if trajectory else "")
    )

    result_payload: dict[str, Any] = {
        "task_name": task_name,
        "rollout_name": rollout_name,
        "rewards": rewards,
        "agent": config.agent or None,
        "agent_name": "verifiers",
        "model": result.normalized_model or result.model or None,
        "n_tool_calls": result.total_tool_calls or 0,
        "n_prompts": len(prompts),
        "agent_result": {
            "n_tool_calls": result.total_tool_calls or 0,
            "n_prompts": len(prompts),
            "n_input_tokens": None,
            "n_output_tokens": None,
            "n_cache_read_tokens": None,
            "n_cache_creation_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "usage_source": "unavailable",
            "price_source": None,
        },
        "error": result.error if result.returncode != 0 or not result.verifiers_error else None,
        "error_category": None,
        "verifier_error": result.verifiers_error,
        "verifier_error_category": None,
        "idle_timeout_info": None,
        "sandbox_startup_info": None,
        "transport_error_info": None,
        "verifier_timeout_info": None,
        "partial_trajectory": False,
        "trajectory_source": "hosted_env" if trajectory else None,
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "timing": timing,
        "scenes": [],
        "source": source_provenance,
    }
    (result.run_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
    (result.run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (result.run_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    config_payload: dict[str, Any] = {
        "task_path": None,
        "agent": config.agent or None,
        "model": result.normalized_model or result.model or None,
        "environment": "hosted_env",
        "skills_dir": None,
        "sandbox_user": None,
        "sandbox_locked_paths": None,
        "sandbox_setup_timeout": None,
        "context_root": None,
        "timeout_sec": None,
        "concurrency": config.concurrency,
        "agent_idle_timeout_sec": None,
        "started_at": str(started_at),
        "agent_env": {},
        "scenes": [],
        "source": source_provenance,
        "hosted_env": {
            "provider": result.source_env.provider,
            "env_uid": result.source_env.env_uid,
            "runner": config.runner,
            "num_examples": config.num_examples,
            "rollouts_per_example": config.rollouts_per_example,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "sampling_args": config.sampling_args,
            "env_args": config.env_args,
        },
    }
    (result.run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    if rewards:
        _write_hosted_rewards_jsonl(result.run_dir, rewards, finished_at)


def _hosted_source_provenance(
    ref: HostedEnvRef,
    *,
    runner: str,
    env_args: dict[str, Any],
) -> dict[str, Any]:
    """Provenance block stamped on hosted-env artifacts.

    Uses ``type="hosted_env"`` so audit tools can distinguish imported
    evidence from native git-rooted rollouts (which use ``type="github"``).
    """
    return {
        "type": "hosted_env",
        "provider": ref.provider,
        "env_id": ref.env_id,
        "env_uid": ref.env_uid,
        "version": ref.version,
        "hub_url": ref.hub_url,
        "runner": runner,
        "env_args": env_args,
    }


def _reconstruct_trajectory(
    output_dir: Path,
    started_at: datetime,
) -> list[dict[str, Any]]:
    """Build an ACP-shaped trajectory from vf-eval's saved results.

    vf-eval ``--save-results`` writes per-example completions to
    ``output_dir/results.jsonl``. Each line typically carries a list of
    chat messages and a reward. We map each row to a ``user_message`` +
    ``agent_message`` pair so the trajectory is non-empty and consumers
    can audit what the model saw. The reconstruction is best-effort:
    unknown shapes are skipped rather than raising.
    """
    results_path = _find_results_jsonl(output_dir)
    if results_path is None:
        return []

    trajectory: list[dict[str, Any]] = []
    ts = started_at.isoformat()
    try:
        with results_path.open() as f:
            for example_idx, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("skipping non-JSON vf-eval row: %r", line[:80])
                    continue
                if not isinstance(row, dict):
                    continue
                trajectory.extend(_row_to_acp_events(row, example_idx, ts))
    except OSError as e:
        logger.debug("could not read %s: %s", results_path, e)
        return []
    return trajectory


def _find_results_jsonl(output_dir: Path) -> Path | None:
    """Locate the vf-eval ``results.jsonl`` artifact, if it exists."""
    if not output_dir.exists():
        return None
    candidates = [output_dir / "results.jsonl", *output_dir.rglob("results.jsonl")]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _row_to_acp_events(
    row: dict[str, Any],
    example_idx: int,
    ts: str,
) -> list[dict[str, Any]]:
    """Convert one vf-eval results row to ACP-style session events."""
    events: list[dict[str, Any]] = []
    prompt = row.get("prompt") or row.get("question") or row.get("input")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    events.append(
                        {
                            "type": "user_message",
                            "ts": ts,
                            "example_index": example_idx,
                            "content": content,
                        }
                    )
    elif isinstance(prompt, str) and prompt:
        events.append(
            {
                "type": "user_message",
                "ts": ts,
                "example_index": example_idx,
                "content": prompt,
            }
        )

    completion = row.get("completion") or row.get("response") or row.get("output")
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    events.append(
                        {
                            "type": "agent_message",
                            "ts": ts,
                            "example_index": example_idx,
                            "content": content,
                        }
                    )
    elif isinstance(completion, str) and completion:
        events.append(
            {
                "type": "agent_message",
                "ts": ts,
                "example_index": example_idx,
                "content": completion,
            }
        )

    reward = row.get("reward")
    if isinstance(reward, int | float):
        events.append(
            {
                "type": "reward",
                "ts": ts,
                "example_index": example_idx,
                "value": float(reward),
                "source": "verifiers",
            }
        )
    return events


def _collect_prompts(output_dir: Path, config: HostedEnvRunConfig) -> list[str]:
    """Collect unique user prompts observed in vf-eval results.

    Falls back to a synthetic descriptor when results.jsonl is missing.
    """
    results_path = _find_results_jsonl(output_dir)
    if results_path is None:
        return [
            f"<hosted_env:{config.source_env.env_uid} num_examples="
            f"{config.num_examples}>"
        ]
    prompts: list[str] = []
    seen: set[str] = set()
    try:
        with results_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                raw_prompt = row.get("prompt") or row.get("question") or row.get("input")
                text = _stringify_prompt(raw_prompt)
                if text and text not in seen:
                    seen.add(text)
                    prompts.append(text)
    except OSError:
        pass
    if not prompts:
        prompts.append(
            f"<hosted_env:{config.source_env.env_uid} num_examples="
            f"{config.num_examples}>"
        )
    return prompts


def _stringify_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for msg in value:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return ""


def _build_rewards_dict(
    result: HostedEnvRunResult,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Construct a rewards dict matching the native rollout shape.

    Native rollouts store rewards as ``{"reward": float, "rubric": [...]}``.
    Hosted runs report a single average reward from vf-eval's stdout
    summary, optionally augmented by per-example rubric items reconstructed
    from ``results.jsonl``.
    """
    if result.reward is None:
        return None
    rubric: list[dict[str, Any]] = []
    results_path = _find_results_jsonl(output_dir)
    if results_path is not None:
        try:
            with results_path.open() as f:
                for example_idx, raw_line in enumerate(f):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    reward = row.get("reward")
                    if isinstance(reward, int | float):
                        rubric.append(
                            {
                                "name": f"example_{example_idx}",
                                "score": float(reward),
                            }
                        )
        except OSError:
            pass
    payload: dict[str, Any] = {"reward": float(result.reward)}
    if rubric:
        payload["rubric"] = rubric
    return payload


def _write_hosted_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict[str, Any],
    finished_at: datetime,
) -> None:
    """Mirror ``rollout._write_rewards_jsonl`` for hosted-env rewards.

    Kept in this module to avoid importing from ``benchflow.rollout`` (which
    pulls heavy ACP/sandbox deps at import time).
    """
    events: list[dict[str, Any]] = []
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            events.append(
                {
                    "ts": finished_at.isoformat(),
                    "type": "process",
                    "source": "verifier_rubric",
                    "value": item.get("score", 0.0),
                    "tag": item.get("name", f"rubric_{i}"),
                    "step_index": i,
                    "meta": {
                        k: v for k, v in item.items() if k not in ("score", "name")
                    },
                }
            )
    scalar = rewards.get("reward")
    if scalar is not None:
        non_event_keys = {"reward", "rubric"}
        events.append(
            {
                "ts": finished_at.isoformat(),
                "type": "terminal",
                "source": "verifier",
                "value": scalar,
                "tag": "reward",
                "step_index": None,
                "meta": {k: v for k, v in rewards.items() if k not in non_event_keys},
            }
        )
    if events:
        path = rollout_dir / "rewards.jsonl"
        path.write_text("\n".join(json.dumps(e, default=str) for e in events) + "\n")
