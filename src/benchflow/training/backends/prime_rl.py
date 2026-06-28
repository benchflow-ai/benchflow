"""Prime-RL SFT launch wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import TextIO

from benchflow.training.run_manifest import (
    CommandRecord,
    TrainComponent,
    TrainRunManifest,
    utc_now,
    write_manifest,
)

_ENV_KEYS_TO_RECORD = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "PRIME_API_KEY",
    "PRIMEINTELLECT_API_KEY",
    "WANDB_API_KEY",
)


@dataclass(frozen=True)
class PrimeRlSftSpec:
    config: Path
    work_dir: Path
    data: str | None = None
    output_dir: Path | None = None
    dry_run: bool = False
    follow: bool = False
    uv_no_sync: bool = False
    overrides: tuple[str, ...] = ()
    force: bool = False
    cwd: Path | None = None


@dataclass(frozen=True)
class PrimeRlSftResult:
    manifest_path: Path
    command_path: Path
    returncode: int


def _parse_overrides(overrides: Iterable[str]) -> list[str]:
    argv: list[str] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--override must have a non-empty key: {override!r}")
        argv.extend([f"--{key}", value])
    return argv


def _resolve_config_path(config: Path, cwd: Path | None) -> Path:
    if config.is_file():
        return config.resolve()
    if cwd is not None and not config.is_absolute():
        candidate = cwd / config
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"--config not found: {config}")


def build_prime_rl_sft_argv(spec: PrimeRlSftSpec) -> list[str]:
    config = _resolve_config_path(spec.config, spec.cwd)
    work_dir = spec.work_dir.resolve()
    output_dir = (
        spec.output_dir.resolve() if spec.output_dir else work_dir / "prime-rl-output"
    )
    argv = ["uv", "run"]
    if spec.uv_no_sync:
        argv.append("--no-sync")
    argv.extend(["sft", "@", str(config)])
    if spec.data:
        argv.extend(["--data.name", spec.data])
    argv.extend(["--output-dir", str(output_dir)])
    if spec.dry_run:
        argv.append("--dry-run")
    argv.extend(_parse_overrides(spec.overrides))
    return argv


def _recorded_env_keys() -> list[str]:
    return sorted(key for key in _ENV_KEYS_TO_RECORD if os.environ.get(key))


def _shell_quote(argv: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(arg) for arg in argv)


def _copy_stream(stream: TextIO, handle: TextIO, *, echo: bool) -> None:
    for line in stream:
        handle.write(line)
        handle.flush()
        if echo:
            print(line, end="", flush=True)


def _initial_manifest(
    spec: PrimeRlSftSpec, argv: list[str], logs: list[str]
) -> TrainRunManifest:
    work_dir = spec.work_dir.resolve()
    output_dir = (
        spec.output_dir.resolve() if spec.output_dir else work_dir / "prime-rl-output"
    )
    command = CommandRecord(
        id="prime-rl-sft",
        argv=argv,
        cwd=str((spec.cwd or Path.cwd()).resolve()),
        env_keys=_recorded_env_keys(),
    )
    component = TrainComponent(
        name="trainer",
        role="primary",
        command_id=command.id,
        status="pending",
        logs=logs,
    )
    now = utc_now()
    return TrainRunManifest(
        schema_version=1,
        run_type="sft",
        backend="prime-rl",
        config=str(_resolve_config_path(spec.config, spec.cwd)),
        work_dir=str(work_dir),
        output_dir=str(output_dir),
        dry_run=spec.dry_run,
        created_at=now,
        updated_at=now,
        overall_status="pending",
        commands=[command],
        components=[component],
    )


def run_prime_rl_sft(spec: PrimeRlSftSpec) -> PrimeRlSftResult:
    if spec.cwd is not None and not spec.cwd.is_dir():
        raise ValueError(f"--prime-rl-dir not found: {spec.cwd}")
    _resolve_config_path(spec.config, spec.cwd)
    work_dir = spec.work_dir.resolve()
    manifest_path = work_dir / "train-run.json"
    if manifest_path.exists() and not spec.force:
        raise ValueError(f"{manifest_path} already exists; pass --force to overwrite")

    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required to launch Prime-RL SFT")

    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "prime-rl"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    command_path = work_dir / "command.txt"

    argv = build_prime_rl_sft_argv(spec)
    command_path.write_text(_shell_quote(argv) + "\n", encoding="utf-8")
    manifest = _initial_manifest(
        spec,
        argv,
        [
            str(stdout_path.relative_to(work_dir)),
            str(stderr_path.relative_to(work_dir)),
        ],
    )
    manifest.overall_status = "running"
    manifest.components[0].status = "running"
    write_manifest(manifest_path, manifest)

    cwd = spec.cwd.resolve() if spec.cwd else Path.cwd()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = Thread(
            target=_copy_stream,
            args=(process.stdout, stdout_handle),
            kwargs={"echo": spec.follow},
            daemon=True,
        )
        stderr_thread = Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_handle),
            kwargs={"echo": False},
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    if returncode == 0:
        manifest.overall_status = "succeeded"
        manifest.components[0].status = "succeeded"
    else:
        manifest.overall_status = "failed"
        manifest.components[0].status = "failed"
        manifest.components[0].extra["returncode"] = returncode
    write_manifest(manifest_path, manifest)
    return PrimeRlSftResult(
        manifest_path=manifest_path,
        command_path=command_path,
        returncode=returncode,
    )
