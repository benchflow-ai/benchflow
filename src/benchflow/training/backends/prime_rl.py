"""Prime-RL SFT launch wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from threading import Thread
from typing import Any, TextIO

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
    target_examples: int | None = None
    sync_scheduler_to_max_steps: bool = True
    pack_function: str | None = None
    loss_mask: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None
    tool_defs_mode: str = "preserve"
    allow_unsafe_stack_flash_attn: bool = False
    force: bool = False
    cwd: Path | None = None
    publish_model: str | None = None
    model_tag: str | None = None
    model_card: str | None = None
    publish_artifacts: str | None = None
    hf_prefix: str | None = None
    hf_public_read_check: bool = False


@dataclass(frozen=True)
class PrimeRlSftResult:
    manifest_path: Path
    command_path: Path
    returncode: int


@dataclass(frozen=True)
class PrimeRlSftExposurePlan:
    target_examples: int | None = None
    data_batch_size: int | None = None
    derived_max_steps: int | None = None
    sync_scheduler_to_max_steps: bool = False
    pack_function: str | None = None
    loss_mask: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None
    generated_overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_examples": self.target_examples,
            "data_batch_size": self.data_batch_size,
            "derived_max_steps": self.derived_max_steps,
            "sync_scheduler_to_max_steps": self.sync_scheduler_to_max_steps,
            "pack_function": self.pack_function,
            "loss_mask": self.loss_mask,
            "model_attn": self.model_attn,
            "renderer_mode": self.renderer_mode,
            "generated_overrides": list(self.generated_overrides),
        }


@dataclass(frozen=True)
class PrimeRlSftDatasetPlan:
    source_data: str
    resolved_data: str
    kind: str
    dataset_dir: str | None = None
    train_jsonl: str | None = None
    tool_defs_mode: str = "preserve"
    tool_defs_removed_rows: int | None = None
    validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_data": self.source_data,
            "resolved_data": self.resolved_data,
            "kind": self.kind,
            "dataset_dir": self.dataset_dir,
            "train_jsonl": self.train_jsonl,
            "tool_defs_mode": self.tool_defs_mode,
            "tool_defs_removed_rows": self.tool_defs_removed_rows,
            "validation": self.validation,
        }


@dataclass(frozen=True)
class PrimeRlSftLaunch:
    argv: list[str]
    exposure_plan: PrimeRlSftExposurePlan | None = None


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


def _override_map(overrides: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--override must have a non-empty key: {override!r}")
        values[key] = value
    return values


def _resolve_config_path(config: Path, cwd: Path | None) -> Path:
    if config.is_file():
        return config.resolve()
    if cwd is not None and not config.is_absolute():
        candidate = cwd / config
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"--config not found: {config}")


def _load_toml(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, Mapping) else {}


def _nested_config_value(data: Mapping[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _parse_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    if parsed <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return parsed


def _resolve_data_batch_size(
    config: Mapping[str, Any], overrides: Mapping[str, str]
) -> int:
    raw = overrides.get("data.batch_size")
    source = (
        "--override data.batch_size" if raw is not None else "config data.batch_size"
    )
    if raw is None:
        raw = _nested_config_value(config, "data.batch_size")
    if raw is None:
        raise ValueError(
            "--target-examples requires data.batch_size in the Prime-RL config "
            "or an explicit --override data.batch_size=..."
        )
    return _parse_positive_int(raw, key=source)


def _loss_mask_overrides(raw: str) -> tuple[str, tuple[str, ...]]:
    value = raw.strip().lower().replace("_", "-")
    role_names = ("system", "user", "assistant", "tool")
    role_set = set(role_names)
    if value == "all":
        enabled = role_set
    elif value == "assistant":
        enabled = {"assistant"}
    else:
        enabled = {part.strip().replace("_", "-") for part in value.split(",")}
        if not enabled or any(not part for part in enabled):
            raise ValueError(
                "--loss-mask must be 'all', 'assistant', or comma-separated roles"
            )
        unknown = sorted(enabled - role_set)
        if unknown:
            raise ValueError(
                "--loss-mask roles must be drawn from system,user,assistant,tool; "
                f"got {','.join(unknown)}"
            )
    normalized = (
        "all"
        if enabled == role_set
        else ",".join(role for role in role_names if role in enabled)
    )
    return normalized, tuple(
        f"data.loss_mask.{role}={'true' if role in enabled else 'false'}"
        for role in role_names
    )


def _resolve_effective_value(
    config: Mapping[str, Any], overrides: Mapping[str, str], key: str
) -> Any:
    if key in overrides:
        return overrides[key]
    return _nested_config_value(config, key)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_tool_defs_mode(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    aliases = {"keep": "preserve", "strip": "omit", "drop": "omit"}
    value = aliases.get(value, value)
    if value not in {"preserve", "omit"}:
        raise ValueError("--tool-defs-mode must be either 'preserve' or 'omit'")
    return value


def _is_qwen35_model(model_name: str | None) -> bool:
    if model_name is None:
        return False
    normalized = model_name.lower().replace("_", "-")
    return normalized.startswith("qwen/qwen3.5-")


def _validate_prime_rl_mode(
    spec: PrimeRlSftSpec,
    config: Mapping[str, Any],
    effective_overrides: Mapping[str, str],
) -> None:
    pack_function = _string_or_none(
        _resolve_effective_value(config, effective_overrides, "data.pack_function")
    )
    model_name = _string_or_none(
        _resolve_effective_value(config, effective_overrides, "model.name")
    )
    model_attn = _string_or_none(
        _resolve_effective_value(config, effective_overrides, "model.attn")
    )
    if (
        not spec.allow_unsafe_stack_flash_attn
        and pack_function == "stack"
        and _is_qwen35_model(model_name)
        and model_attn in {"flash_attention_2", "flash_attention_3", "fa4"}
    ):
        raise ValueError(
            "Prime-RL stack packing with Qwen/Qwen3.5-* and flash attention is "
            "blocked because it can misinterpret padded position_ids as packed "
            "sequence starts and fail inside Qwen3.5 varlen kernels. Use "
            "--model-attn sdpa or --override model.attn=sdpa for the "
            "custom-trainer-compatible stack path, or pass "
            "--allow-unsafe-stack-flash-attn to run the native Prime-RL mode anyway."
        )


def _build_generated_overrides(
    spec: PrimeRlSftSpec, config: Mapping[str, Any]
) -> PrimeRlSftExposurePlan | None:
    overrides = _override_map(spec.overrides)
    generated: list[str] = []
    target_examples: int | None = None
    data_batch_size: int | None = None
    derived_max_steps: int | None = None
    pack_function: str | None = None
    loss_mask: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None

    if spec.target_examples is not None:
        if "max_steps" in overrides:
            raise ValueError(
                "--target-examples cannot be combined with --override max_steps=..."
            )
        target_examples = _parse_positive_int(
            spec.target_examples, key="--target-examples"
        )
        data_batch_size = _resolve_data_batch_size(config, overrides)
        derived_max_steps = ceil(target_examples / data_batch_size)
        generated.append(f"max_steps={derived_max_steps}")
        if spec.sync_scheduler_to_max_steps:
            if "scheduler.decay_steps" in overrides:
                raise ValueError(
                    "--sync-scheduler-to-max-steps cannot be combined with "
                    "--override scheduler.decay_steps=..."
                )
            generated.append(f"scheduler.decay_steps={derived_max_steps}")

    if spec.pack_function is not None:
        if spec.pack_function not in {"cat", "stack"}:
            raise ValueError("--pack-function must be either 'cat' or 'stack'")
        if "data.pack_function" in overrides:
            raise ValueError(
                "--pack-function cannot be combined with "
                "--override data.pack_function=..."
            )
        pack_function = spec.pack_function
        generated.append(f"data.pack_function={pack_function}")

    if spec.loss_mask is not None:
        loss_keys = [
            f"data.loss_mask.{role}" for role in ("system", "user", "assistant", "tool")
        ]
        conflicting = sorted(key for key in loss_keys if key in overrides)
        if conflicting:
            raise ValueError(
                "--loss-mask cannot be combined with --override "
                + ", ".join(f"{key}=..." for key in conflicting)
            )
        loss_mask, loss_overrides = _loss_mask_overrides(spec.loss_mask)
        generated.extend(loss_overrides)

    if spec.model_attn is not None:
        model_attn = spec.model_attn.strip()
        if not model_attn:
            raise ValueError("--model-attn must be non-empty")
        if "model.attn" in overrides:
            raise ValueError(
                "--model-attn cannot be combined with --override model.attn=..."
            )
        generated.append(f"model.attn={model_attn}")

    if spec.renderer_mode is not None:
        renderer_mode = spec.renderer_mode.strip().lower().replace("_", "-")
        if renderer_mode != "none":
            raise ValueError("--renderer-mode currently supports only 'none'")
        conflicting = sorted(
            key for key in overrides if key == "renderer" or key.startswith("renderer.")
        )
        if conflicting:
            raise ValueError(
                "--renderer-mode cannot be combined with --override "
                + ", ".join(f"{key}=..." for key in conflicting)
            )
        generated.append("renderer=None")

    effective_overrides = _override_map((*spec.overrides, *generated))
    _validate_prime_rl_mode(spec, config, effective_overrides)

    if not generated:
        return None
    return PrimeRlSftExposurePlan(
        target_examples=target_examples,
        data_batch_size=data_batch_size,
        derived_max_steps=derived_max_steps,
        sync_scheduler_to_max_steps=(
            bool(spec.sync_scheduler_to_max_steps)
            if target_examples is not None
            else False
        ),
        pack_function=pack_function,
        loss_mask=loss_mask,
        model_attn=model_attn,
        renderer_mode=renderer_mode,
        generated_overrides=tuple(generated),
    )


def build_prime_rl_sft_launch(spec: PrimeRlSftSpec) -> PrimeRlSftLaunch:
    config = _resolve_config_path(spec.config, spec.cwd)
    config_data = _load_toml(config)
    work_dir = spec.work_dir.resolve()
    output_dir = (
        spec.output_dir.resolve() if spec.output_dir else work_dir / "prime-rl-output"
    )
    exposure_plan = _build_generated_overrides(spec, config_data)
    effective_overrides = spec.overrides + (
        exposure_plan.generated_overrides if exposure_plan is not None else ()
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
    argv.extend(_parse_overrides(effective_overrides))
    return PrimeRlSftLaunch(argv=argv, exposure_plan=exposure_plan)


def build_prime_rl_sft_argv(spec: PrimeRlSftSpec) -> list[str]:
    return build_prime_rl_sft_launch(spec).argv


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


def _local_data_path(data: str) -> Path | None:
    path = Path(data).expanduser()
    if path.exists():
        return path.resolve()
    if path.suffix == ".jsonl":
        raise ValueError(f"--data JSONL file not found: {data}")
    return None


def _copy_jsonl_omitting_tool_defs(source: Path, destination: Path) -> int:
    """Copy JSONL while dropping tool schema columns from the training copy."""
    removed_rows = 0
    with (
        source.open("r", encoding="utf-8") as src,
        destination.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                dst.write(line)
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}: every JSONL row must be an object")
            if "tool_defs" in row or "tools" in row:
                removed_rows += 1
            row.pop("tool_defs", None)
            row.pop("tools", None)
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return removed_rows


def _prepare_prime_rl_data(
    spec: PrimeRlSftSpec, work_dir: Path
) -> tuple[PrimeRlSftSpec, PrimeRlSftDatasetPlan | None]:
    """Make local BenchFlow JSONL usable by Prime-RL's ``load_dataset`` path."""
    if not spec.data:
        return spec, None

    tool_defs_mode = _normalize_tool_defs_mode(spec.tool_defs_mode)
    source_path = _local_data_path(spec.data)
    if source_path is None:
        if tool_defs_mode != "preserve":
            raise ValueError(
                "--tool-defs-mode omit requires --data to be a local JSONL file "
                "or a local dataset directory"
            )
        return spec, None

    if source_path.is_dir():
        train_jsonl = source_path / "train.jsonl"
        validation = None
        if train_jsonl.is_file():
            from benchflow.trajectories.export_prime_sft import (
                validate_prime_sft_jsonl,
            )

            validation = validate_prime_sft_jsonl(train_jsonl)
        if tool_defs_mode == "omit":
            if not train_jsonl.is_file():
                raise ValueError(
                    f"--tool-defs-mode omit requires {source_path} to contain train.jsonl"
                )
            dataset_dir = work_dir / "prime-rl-dataset"
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            shutil.copytree(source_path, dataset_dir)
            transformed_train_jsonl = dataset_dir / "train.jsonl"
            removed_rows = _copy_jsonl_omitting_tool_defs(
                train_jsonl, transformed_train_jsonl
            )
            resolved_spec = replace(spec, data=str(dataset_dir))
            return resolved_spec, PrimeRlSftDatasetPlan(
                source_data=spec.data,
                resolved_data=str(dataset_dir),
                kind="local_dataset_dir_transformed",
                dataset_dir=str(dataset_dir),
                train_jsonl=str(transformed_train_jsonl),
                tool_defs_mode=tool_defs_mode,
                tool_defs_removed_rows=removed_rows,
                validation=validation,
            )
        resolved_spec = replace(spec, data=str(source_path))
        return resolved_spec, PrimeRlSftDatasetPlan(
            source_data=spec.data,
            resolved_data=str(source_path),
            kind="local_dataset_dir",
            dataset_dir=str(source_path),
            train_jsonl=str(train_jsonl) if train_jsonl.is_file() else None,
            tool_defs_mode=tool_defs_mode,
            validation=validation,
        )

    if source_path.suffix != ".jsonl":
        raise ValueError(
            f"--data local files must be Prime-SFT JSONL files, got {source_path}"
        )

    from benchflow.trajectories.export_prime_sft import validate_prime_sft_jsonl

    validation = validate_prime_sft_jsonl(source_path)
    dataset_dir = work_dir / "prime-rl-dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    train_jsonl = dataset_dir / "train.jsonl"
    removed_rows = None
    if tool_defs_mode == "omit":
        removed_rows = _copy_jsonl_omitting_tool_defs(source_path, train_jsonl)
    else:
        shutil.copy2(source_path, train_jsonl)
    resolved_spec = replace(spec, data=str(dataset_dir))
    return resolved_spec, PrimeRlSftDatasetPlan(
        source_data=spec.data,
        resolved_data=str(dataset_dir),
        kind="local_jsonl_packaged",
        dataset_dir=str(dataset_dir),
        train_jsonl=str(train_jsonl),
        tool_defs_mode=tool_defs_mode,
        tool_defs_removed_rows=removed_rows,
        validation=validation,
    )


def _initial_manifest(
    spec: PrimeRlSftSpec,
    argv: list[str],
    logs: list[str],
    exposure_plan: PrimeRlSftExposurePlan | None = None,
    dataset_plan: PrimeRlSftDatasetPlan | None = None,
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
    manifest = TrainRunManifest(
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
    if exposure_plan is not None:
        manifest.extra["prime_rl_sft_exposure_plan"] = exposure_plan.to_dict()
    if dataset_plan is not None:
        manifest.extra["prime_rl_sft_dataset"] = dataset_plan.to_dict()
    return manifest


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

    launch_spec, dataset_plan = _prepare_prime_rl_data(spec, work_dir)
    launch = build_prime_rl_sft_launch(launch_spec)
    argv = launch.argv
    command_path.write_text(_shell_quote(argv) + "\n", encoding="utf-8")
    manifest = _initial_manifest(
        launch_spec,
        argv,
        [
            str(stdout_path.relative_to(work_dir)),
            str(stderr_path.relative_to(work_dir)),
        ],
        launch.exposure_plan,
        dataset_plan,
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
        write_manifest(manifest_path, manifest)
        try:
            _publish_outputs(spec, manifest, manifest_path)
        except ValueError as exc:
            manifest.overall_status = "failed"
            manifest.extra["publish_error"] = str(exc)
            write_manifest(manifest_path, manifest)
            raise
        write_manifest(manifest_path, manifest)
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


def _publish_outputs(
    spec: PrimeRlSftSpec, manifest: TrainRunManifest, manifest_path: Path
) -> None:
    if spec.model_card not in {None, "auto"}:
        raise ValueError("--model-card currently supports only 'auto'")
    if not spec.publish_model and not spec.publish_artifacts:
        return
    from benchflow.publish.huggingface import publish_folder_to_hf

    output_dir = (
        spec.output_dir.resolve()
        if spec.output_dir is not None
        else spec.work_dir.resolve() / "prime-rl-output"
    )
    publishes: list[dict[str, str | None]] = []
    if spec.publish_model:
        model_prefix = spec.model_tag or ""
        result = publish_folder_to_hf(
            output_dir,
            repo_id=spec.publish_model,
            repo_type="model",
            path_in_repo=model_prefix,
            public_read_check=spec.hf_public_read_check,
            commit_message="Upload BenchFlow SFT model artifacts",
        )
        manifest.artifacts["exported_models"].append(result.url)
        publishes.append(
            {
                "type": "model",
                "repo": spec.publish_model,
                "path": model_prefix,
                "url": result.url,
                "commit_url": result.commit_url,
            }
        )
        manifest.extra["published"] = publishes
        write_manifest(manifest_path, manifest)
    if spec.publish_artifacts:
        artifact_prefix = spec.hf_prefix or Path(spec.work_dir).name
        result = publish_folder_to_hf(
            spec.work_dir.resolve(),
            repo_id=spec.publish_artifacts,
            repo_type="dataset",
            path_in_repo=artifact_prefix,
            public_read_check=spec.hf_public_read_check,
            commit_message="Upload BenchFlow SFT training artifacts",
        )
        publishes.append(
            {
                "type": "dataset",
                "repo": spec.publish_artifacts,
                "path": artifact_prefix,
                "url": result.url,
                "commit_url": result.commit_url,
            }
        )
    manifest.extra["published"] = publishes
