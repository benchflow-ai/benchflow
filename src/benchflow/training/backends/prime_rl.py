"""Prime-RL SFT launch wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from hashlib import sha256
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
    compat_profile: str | None = None
    dry_run: bool = False
    follow: bool = False
    uv_no_sync: bool = False
    overrides: tuple[str, ...] = ()
    target_examples: int | None = None
    target_micro_steps: int | None = None
    sync_scheduler_to_max_steps: bool = True
    sync_ckpt_to_max_steps: bool = False
    pack_function: str | None = None
    loss_mask: str | None = None
    loss_normalization: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None
    tool_defs_mode: str = "preserve"
    chat_template_kwargs: tuple[str, ...] = ()
    message_tail_truncation: str = "off"
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
    target_micro_steps: int | None = None
    data_batch_size: int | None = None
    derived_max_steps: int | None = None
    effective_train_examples: int | None = None
    unapplied_micro_steps: int | None = None
    sync_scheduler_to_max_steps: bool = False
    sync_ckpt_to_max_steps: bool = False
    pack_function: str | None = None
    loss_mask: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None
    generated_overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_examples": self.target_examples,
            "target_micro_steps": self.target_micro_steps,
            "data_batch_size": self.data_batch_size,
            "derived_max_steps": self.derived_max_steps,
            "effective_train_examples": self.effective_train_examples,
            "unapplied_micro_steps": self.unapplied_micro_steps,
            "sync_scheduler_to_max_steps": self.sync_scheduler_to_max_steps,
            "sync_ckpt_to_max_steps": self.sync_ckpt_to_max_steps,
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
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template_kwargs_rows: int | None = None
    message_tail_truncation: str = "off"
    message_tail_truncated_rows: int | None = None
    message_tail_max_area: int | None = None
    message_tail_max_tokens_before: int | None = None
    message_tail_max_tokens_after: int | None = None
    custom_trainer_pretokenized_rows: int | None = None
    custom_trainer_pretokenized_trainable_tokens: int | None = None
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
            "chat_template_kwargs": self.chat_template_kwargs,
            "chat_template_kwargs_rows": self.chat_template_kwargs_rows,
            "message_tail_truncation": self.message_tail_truncation,
            "message_tail_truncated_rows": self.message_tail_truncated_rows,
            "message_tail_max_area": self.message_tail_max_area,
            "message_tail_max_tokens_before": self.message_tail_max_tokens_before,
            "message_tail_max_tokens_after": self.message_tail_max_tokens_after,
            "custom_trainer_pretokenized_rows": (self.custom_trainer_pretokenized_rows),
            "custom_trainer_pretokenized_trainable_tokens": (
                self.custom_trainer_pretokenized_trainable_tokens
            ),
            "validation": self.validation,
        }


@dataclass(frozen=True)
class PrimeRlSftShimPlan:
    name: str
    description: str
    shim_dir: str
    sitecustomize: str
    env: dict[str, str]
    guards: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "shim_dir": self.shim_dir,
            "sitecustomize": self.sitecustomize,
            "env": dict(self.env),
            "guards": list(self.guards),
        }


@dataclass(frozen=True)
class PrimeRlSftLaunch:
    argv: list[str]
    exposure_plan: PrimeRlSftExposurePlan | None = None


_MOBILE300_PROFILE = "env0-mobile300-pr828"
_CUSTOM_TRAINER_TOKEN_SUFFIX_MODE = "custom-trainer-token-suffix"
_CUSTOM_TRAINER_PRETOKENIZED_MODE = "custom-trainer-pretokenized"
_TOKEN_MEAN_LOSS_NORMALIZATION = "token_mean"
_SAMPLE_MEAN_LOSS_NORMALIZATION = "sample_mean"
_SAMPLE_MEAN_SHIM_ENV = "BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS"
_PRETOKENIZED_DATA_SHIM_ENV = "BENCHFLOW_PRIME_RL_PRETOKENIZED_SFT_DATA"
_COMPAT_PROFILE_ALIASES = {
    _MOBILE300_PROFILE: _MOBILE300_PROFILE,
    "env-0-mobile300-pr828": _MOBILE300_PROFILE,
    "env0-mobile300-custom-sft": _MOBILE300_PROFILE,
    "env-0-mobile300-custom-sft": _MOBILE300_PROFILE,
}


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


def _normalize_loss_normalization(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().lower().replace("-", "_")
    aliases = {
        "token": _TOKEN_MEAN_LOSS_NORMALIZATION,
        "token_mean": _TOKEN_MEAN_LOSS_NORMALIZATION,
        "token_weighted": _TOKEN_MEAN_LOSS_NORMALIZATION,
        "sample": _SAMPLE_MEAN_LOSS_NORMALIZATION,
        "sample_mean": _SAMPLE_MEAN_LOSS_NORMALIZATION,
        "row": _SAMPLE_MEAN_LOSS_NORMALIZATION,
        "row_mean": _SAMPLE_MEAN_LOSS_NORMALIZATION,
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError("--loss-normalization must be 'token_mean' or 'sample_mean'")
    return normalized


def _resolve_effective_value(
    config: Mapping[str, Any], overrides: Mapping[str, str], key: str
) -> Any:
    if key in overrides:
        return overrides[key]
    return _nested_config_value(config, key)


def _resolve_effective_positive_int(
    config: Mapping[str, Any],
    overrides: Mapping[str, str],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = _resolve_effective_value(config, overrides, key)
    if value is None:
        if default is None:
            raise ValueError(f"{key} must be configured")
        value = default
    return _parse_positive_int(value, key=key)


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


def _normalize_message_tail_truncation(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    aliases = {
        "none": "off",
        "false": "off",
        "disabled": "off",
        "keep-user": "keep-first-user",
        "first-user": "keep-first-user",
        "keep-user-suffix": "keep-first-user",
        "token-suffix": _CUSTOM_TRAINER_PRETOKENIZED_MODE,
        "rendered-token-suffix": _CUSTOM_TRAINER_PRETOKENIZED_MODE,
        "custom-token-suffix": _CUSTOM_TRAINER_PRETOKENIZED_MODE,
        "custom-trainer-suffix": _CUSTOM_TRAINER_PRETOKENIZED_MODE,
        _CUSTOM_TRAINER_TOKEN_SUFFIX_MODE: _CUSTOM_TRAINER_PRETOKENIZED_MODE,
    }
    value = aliases.get(value, value)
    if value not in {"off", "keep-first-user", _CUSTOM_TRAINER_PRETOKENIZED_MODE}:
        raise ValueError(
            "--message-tail-truncation must be 'off', 'keep-first-user', or "
            f"'{_CUSTOM_TRAINER_PRETOKENIZED_MODE}'"
        )
    return value


def _parse_chat_template_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "false", "null"}:
        value = value.lower()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return raw


def _parse_chat_template_kwargs(raw: Iterable[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--chat-template-kwarg must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"--chat-template-kwarg must have a non-empty key: {item!r}"
            )
        values[key] = _parse_chat_template_value(value)
    return values


def _profile_chat_template_kwargs(
    raw: tuple[str, ...],
    defaults: Mapping[str, Any],
    *,
    profile: str,
) -> tuple[str, ...]:
    values = _parse_chat_template_kwargs(raw)
    generated: list[str] = []
    for key, required in defaults.items():
        if key in values:
            if values[key] != required:
                rendered = json.dumps(required, sort_keys=True)
                raise ValueError(
                    f"--compat-profile {profile} requires "
                    f"--chat-template-kwarg {key}={rendered}; got {values[key]!r}"
                )
            continue
        generated.append(f"{key}={json.dumps(required, sort_keys=True)}")
    return (*raw, *generated)


def _normalize_compat_profile(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().lower().replace("_", "-")
    if not value:
        raise ValueError("--compat-profile must be non-empty")
    normalized = _COMPAT_PROFILE_ALIASES.get(value)
    if normalized is None:
        supported = ", ".join(sorted(_COMPAT_PROFILE_ALIASES))
        raise ValueError(
            f"Unsupported --compat-profile {raw!r}; supported profiles: {supported}"
        )
    return normalized


def _profile_field(
    value: str | int | None,
    required: str | int,
    *,
    field: str,
    profile: str,
) -> str | int:
    if value is None:
        return required
    if str(value).lower() != str(required).lower():
        raise ValueError(
            f"--compat-profile {profile} requires {field}={required!r}; got {value!r}"
        )
    return required


def _apply_compat_profile(spec: PrimeRlSftSpec) -> PrimeRlSftSpec:
    profile = _normalize_compat_profile(spec.compat_profile)
    if profile is None:
        return spec
    if profile != _MOBILE300_PROFILE:
        raise AssertionError(f"unhandled compat profile: {profile}")
    if not spec.sync_scheduler_to_max_steps:
        raise ValueError(
            f"--compat-profile {profile} requires --sync-scheduler-to-max-steps"
        )
    if spec.target_examples is not None:
        _profile_field(
            spec.target_examples, 300, field="--target-examples", profile=profile
        )
    if spec.chat_template_kwargs:
        raise ValueError(
            f"--compat-profile {profile} stages pre-tokenized custom-trainer "
            "samples and does not support --chat-template-kwarg"
        )
    loss_normalization = _normalize_loss_normalization(spec.loss_normalization)
    if loss_normalization not in {None, _SAMPLE_MEAN_LOSS_NORMALIZATION}:
        raise ValueError(
            f"--compat-profile {profile} requires "
            f"--loss-normalization {_SAMPLE_MEAN_LOSS_NORMALIZATION}; "
            f"got {loss_normalization!r}"
        )
    return replace(
        spec,
        compat_profile=profile,
        target_examples=None,
        target_micro_steps=int(
            _profile_field(
                spec.target_micro_steps,
                300,
                field="--target-micro-steps",
                profile=profile,
            )
        ),
        sync_ckpt_to_max_steps=True,
        pack_function=str(
            _profile_field(
                spec.pack_function, "stack", field="--pack-function", profile=profile
            )
        ),
        loss_mask=str(
            _profile_field(spec.loss_mask, "all", field="--loss-mask", profile=profile)
        ),
        loss_normalization=_SAMPLE_MEAN_LOSS_NORMALIZATION,
        model_attn=str(
            _profile_field(
                spec.model_attn, "sdpa", field="--model-attn", profile=profile
            )
        ),
        renderer_mode=str(
            _profile_field(
                spec.renderer_mode, "none", field="--renderer-mode", profile=profile
            )
        ),
        tool_defs_mode="omit",
        chat_template_kwargs=(),
        message_tail_truncation=_CUSTOM_TRAINER_PRETOKENIZED_MODE,
    )


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


def _validate_prime_rl_loss_normalization(
    spec: PrimeRlSftSpec,
    config: Mapping[str, Any],
    effective_overrides: Mapping[str, str],
) -> None:
    loss_normalization = _normalize_loss_normalization(spec.loss_normalization)
    if loss_normalization in {None, _TOKEN_MEAN_LOSS_NORMALIZATION}:
        return
    pack_function = _string_or_none(
        _resolve_effective_value(config, effective_overrides, "data.pack_function")
    )
    if pack_function != "stack":
        raise ValueError(
            "--loss-normalization sample_mean requires data.pack_function=stack "
            "so each batch row remains one original training example"
        )
    model_cp = _resolve_effective_positive_int(
        config, effective_overrides, "model.cp", default=1
    )
    if model_cp != 1:
        raise ValueError(
            "--loss-normalization sample_mean requires model.cp=1 because "
            "sequence-sharded rows cannot be reduced to per-sample means by "
            "the BenchFlow Prime-RL wrapper"
        )
    loss_impl = (
        _string_or_none(
            _resolve_effective_value(config, effective_overrides, "loss_impl")
        )
        or "torch"
    )
    if loss_impl in {"liger_fused", "quack_fused"}:
        raise ValueError(
            "--loss-normalization sample_mean does not support fused Prime-RL "
            f"loss_impl={loss_impl!r}; use loss_impl=torch or loss_impl=liger"
        )


def _resolve_sample_max_area(
    config: Mapping[str, Any], overrides: Mapping[str, str]
) -> int:
    seq_len = _resolve_effective_positive_int(
        config, overrides, "data.seq_len", default=128
    )
    micro_batch_size = _resolve_effective_positive_int(
        config, overrides, "data.micro_batch_size", default=1
    )
    return seq_len * micro_batch_size


def _build_generated_overrides(
    spec: PrimeRlSftSpec, config: Mapping[str, Any]
) -> PrimeRlSftExposurePlan | None:
    overrides = _override_map(spec.overrides)
    generated: list[str] = []
    target_examples: int | None = None
    target_micro_steps: int | None = None
    data_batch_size: int | None = None
    derived_max_steps: int | None = None
    effective_train_examples: int | None = None
    unapplied_micro_steps: int | None = None
    pack_function: str | None = None
    loss_mask: str | None = None
    model_attn: str | None = None
    renderer_mode: str | None = None

    if spec.target_examples is not None and spec.target_micro_steps is not None:
        raise ValueError(
            "--target-examples and --target-micro-steps cannot be combined"
        )

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
        effective_train_examples = derived_max_steps * data_batch_size
        generated.append(f"max_steps={derived_max_steps}")
        if spec.sync_scheduler_to_max_steps:
            if "scheduler.decay_steps" in overrides:
                raise ValueError(
                    "--sync-scheduler-to-max-steps cannot be combined with "
                    "--override scheduler.decay_steps=..."
                )
            generated.append(f"scheduler.decay_steps={derived_max_steps}")

    if spec.target_micro_steps is not None:
        if "max_steps" in overrides:
            raise ValueError(
                "--target-micro-steps cannot be combined with --override max_steps=..."
            )
        target_micro_steps = _parse_positive_int(
            spec.target_micro_steps, key="--target-micro-steps"
        )
        data_batch_size = _resolve_data_batch_size(config, overrides)
        derived_max_steps = target_micro_steps // data_batch_size
        if derived_max_steps <= 0:
            raise ValueError(
                "--target-micro-steps must cover at least one effective "
                f"Prime-RL batch ({target_micro_steps} < {data_batch_size})"
            )
        effective_train_examples = derived_max_steps * data_batch_size
        unapplied_micro_steps = target_micro_steps - effective_train_examples
        generated.append(f"max_steps={derived_max_steps}")
        if spec.sync_scheduler_to_max_steps:
            if "scheduler.decay_steps" in overrides:
                raise ValueError(
                    "--sync-scheduler-to-max-steps cannot be combined with "
                    "--override scheduler.decay_steps=..."
                )
            generated.append(f"scheduler.decay_steps={derived_max_steps}")

    if spec.sync_ckpt_to_max_steps:
        if derived_max_steps is None:
            raise ValueError(
                "--sync-ckpt-to-max-steps requires --target-examples or "
                "--target-micro-steps"
            )
        ckpt_keys = ("ckpt.interval", "ckpt.keep_interval")
        conflicting = sorted(key for key in ckpt_keys if key in overrides)
        if conflicting:
            raise ValueError(
                "--sync-ckpt-to-max-steps cannot be combined with --override "
                + ", ".join(f"{key}=..." for key in conflicting)
            )
        generated.extend(
            [
                f"ckpt.interval={derived_max_steps}",
                f"ckpt.keep_interval={derived_max_steps}",
            ]
        )

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
    _validate_prime_rl_loss_normalization(spec, config, effective_overrides)

    if not generated:
        return None
    return PrimeRlSftExposurePlan(
        target_examples=target_examples,
        target_micro_steps=target_micro_steps,
        data_batch_size=data_batch_size,
        derived_max_steps=derived_max_steps,
        effective_train_examples=effective_train_examples,
        unapplied_micro_steps=unapplied_micro_steps,
        sync_scheduler_to_max_steps=(
            bool(spec.sync_scheduler_to_max_steps)
            if target_examples is not None or target_micro_steps is not None
            else False
        ),
        sync_ckpt_to_max_steps=bool(spec.sync_ckpt_to_max_steps),
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


def _shell_quote_env(env: Mapping[str, str]) -> str:
    import shlex

    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


_PRIME_RL_SFT_COMPAT_SITE_CUSTOMIZE = r'''
"""BenchFlow Prime-RL SFT compatibility shim.

This file is generated by BenchFlow for one trainer subprocess. It does not
modify the installed Prime-RL package. It can patch loss reduction and make
BenchFlow pre-tokenized tensor rows bypass Prime-RL message rendering.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

_LOSS_ENV = "BENCHFLOW_PRIME_RL_SAMPLE_MEAN_LOSS"
_DATA_ENV = "BENCHFLOW_PRIME_RL_PRETOKENIZED_SFT_DATA"
_LOSS_TARGET = "prime_rl.trainer.sft.train"
_DATA_TARGET = "prime_rl.trainer.sft.data"

_EXPECTED = """\
        if config.model.lora is not None:
            set_lora_num_tokens(torch.full((1,), input_ids.numel(), dtype=torch.int32, device="cuda"))

        token_count = loss_mask.sum(dtype=torch.int64)

        with maybe_activation_offloading(config.model.ac_offloading):
            if config.loss_impl in ("liger_fused", "quack_fused"):
                masked_target_ids = target_ids.clone()
                masked_target_ids[~loss_mask] = FUSED_CE_IGNORE_INDEX
                out = forward(model, input_ids, position_ids, labels=masked_target_ids)
                loss_sum = out["loss"] * token_count
            else:
                out = forward(model, input_ids, position_ids)
                logits = out["logits"]
                B, L, V = logits.shape
                token_loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)
                loss_sum = token_loss[loss_mask].sum()
                del logits

        del out
        return loss_sum, token_count
"""

_REPLACEMENT = """\
        if cp_enabled:
            raise RuntimeError(
                "BenchFlow sample-mean loss requires model.cp=1; context parallel "
                "sequence shards cannot preserve original row means."
            )

        if config.model.lora is not None:
            set_lora_num_tokens(torch.full((1,), input_ids.numel(), dtype=torch.int32, device="cuda"))

        with maybe_activation_offloading(config.model.ac_offloading):
            if config.loss_impl in ("liger_fused", "quack_fused"):
                raise RuntimeError(
                    "BenchFlow sample-mean loss supports loss_impl=torch or "
                    "loss_impl=liger, not fused loss kernels."
                )
            out = forward(model, input_ids, position_ids)
            logits = out["logits"]
            B, L, V = logits.shape
            token_loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)
            per_sample_token_count = loss_mask.sum(dim=1)
            valid_sample_mask = per_sample_token_count > 0
            if not torch.any(valid_sample_mask):
                raise RuntimeError(
                    "BenchFlow sample-mean loss received a batch with no "
                    "trainable samples."
                )
            per_sample_loss_sum = (token_loss * loss_mask.to(token_loss.dtype)).sum(dim=1)
            loss_sum = (
                per_sample_loss_sum[valid_sample_mask]
                / per_sample_token_count[valid_sample_mask].to(token_loss.dtype)
            ).sum()
            sample_count = valid_sample_mask.sum(dtype=torch.int64)
            del logits

        del out
        return loss_sum, sample_count
"""

_COMMENT_EXPECTED = "        # All-reduce token counts and rescale gradients to get a global token-weighted mean."
_COMMENT_REPLACEMENT = "        # All-reduce sample counts and rescale gradients to get a global sample-weighted mean."


class _BenchFlowSampleMeanLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str):
        source_path = self.get_filename(fullname)
        source_bytes = self.get_data(source_path)
        return self.source_to_code(source_bytes, source_path)

    def source_to_code(self, data, path, *, _optimize=-1):
        source = data.decode("utf-8") if isinstance(data, bytes) else data
        if _EXPECTED not in source:
            raise RuntimeError(
                "BenchFlow Prime-RL sample-mean shim could not find the expected "
                f"loss block in {path}. Refusing to run against an unknown "
                "Prime-RL train loop."
            )
        patched = source.replace(_EXPECTED, _REPLACEMENT, 1)
        patched = patched.replace(_COMMENT_EXPECTED, _COMMENT_REPLACEMENT, 1)
        return super().source_to_code(patched, path, _optimize=_optimize)


class _BenchFlowSampleMeanFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _LOSS_TARGET or os.environ.get(_LOSS_ENV) != "1":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(
            spec.loader, importlib.machinery.SourceFileLoader
        ):
            raise ImportError(
                "BenchFlow Prime-RL sample-mean shim requires a source-backed "
                f"loader for {_LOSS_TARGET}."
            )
        spec.loader = _BenchFlowSampleMeanLoader(spec.loader.name, spec.loader.path)
        return spec


class _BenchFlowPretokenizedDataLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        if os.environ.get(_DATA_ENV) != "1":
            return
        dataset_cls = getattr(module, "SFTDataset", None)
        if dataset_cls is None:
            raise RuntimeError(
                "BenchFlow Prime-RL pretokenized data shim could not find "
                "SFTDataset."
            )
        original_process = dataset_cls._process

        def _as_int_list(example, key):
            value = example.get(key)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a list")
            return [int(item) for item in value]

        def _as_bool_list(example, key):
            value = example.get(key)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a list")
            return [bool(item) for item in value]

        def benchflow_process(self, example):
            if (
                isinstance(example, dict)
                and example.get("benchflow_custom_trainer_pretokenized")
            ):
                input_ids = _as_int_list(example, "benchflow_input_ids")
                target_ids = _as_int_list(example, "benchflow_target_ids")
                loss_mask = _as_bool_list(example, "benchflow_loss_mask")
                position_ids = _as_int_list(example, "benchflow_position_ids")
                lengths = {
                    len(input_ids),
                    len(target_ids),
                    len(loss_mask),
                    len(position_ids),
                }
                if len(lengths) != 1:
                    raise ValueError(
                        "BenchFlow pretokenized SFT row has inconsistent "
                        "input_ids/target_ids/loss_mask/position_ids lengths"
                    )
                if not input_ids:
                    raise ValueError("BenchFlow pretokenized SFT row is empty")
                if not any(loss_mask):
                    raise ValueError(
                        "BenchFlow pretokenized SFT row has no trainable tokens"
                    )
                return {
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "loss_mask": loss_mask,
                    "position_ids": position_ids,
                }
            return original_process(self, example)

        dataset_cls._process = benchflow_process
        sys.stderr.write("BenchFlow Prime-RL pretokenized SFT data shim enabled\\n")


class _BenchFlowPretokenizedDataFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _DATA_TARGET or os.environ.get(_DATA_ENV) != "1":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(
            spec.loader, importlib.machinery.SourceFileLoader
        ):
            raise ImportError(
                "BenchFlow Prime-RL pretokenized data shim requires a "
                f"source-backed loader for {_DATA_TARGET}."
            )
        spec.loader = _BenchFlowPretokenizedDataLoader(
            spec.loader.name, spec.loader.path
        )
        return spec


if os.environ.get(_LOSS_ENV) == "1":
    sys.meta_path.insert(0, _BenchFlowSampleMeanFinder())
    sys.stderr.write("BenchFlow Prime-RL sample-mean loss shim enabled\\n")
if os.environ.get(_DATA_ENV) == "1":
    sys.meta_path.insert(0, _BenchFlowPretokenizedDataFinder())
'''


def _write_prime_rl_sft_compat_shim(
    work_dir: Path, *, sample_mean: bool, pretokenized_data: bool
) -> PrimeRlSftShimPlan:
    shim_dir = work_dir / "prime-rl-sft-compat-shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        _PRIME_RL_SFT_COMPAT_SITE_CUSTOMIZE.lstrip(), encoding="utf-8"
    )
    env = {"PYTHONPATH": str(shim_dir)}
    guards: list[str] = []
    if sample_mean:
        env[_SAMPLE_MEAN_SHIM_ENV] = "1"
        guards.extend(
            [
                "Prime-RL train.py loss block must match the known token-mean source",
                "data.pack_function=stack",
                "model.cp=1",
                "loss_impl=torch or loss_impl=liger",
                "every effective batch must contain at least one trainable row",
            ]
        )
    if pretokenized_data:
        env[_PRETOKENIZED_DATA_SHIM_ENV] = "1"
        guards.extend(
            [
                "Prime-RL SFTDataset must be source-backed",
                "pretokenized rows must carry input_ids/target_ids/loss_mask/position_ids",
            ]
        )
    return PrimeRlSftShimPlan(
        name="prime_rl_sft_compatibility",
        description=(
            "Import-time Prime-RL SFT compatibility shim that can change loss "
            "reduction from token mean to per-sample mean and can feed "
            "BenchFlow pre-tokenized custom-trainer tensor rows while leaving "
            "the Prime-RL package files untouched."
        ),
        shim_dir=str(shim_dir),
        sitecustomize=str(sitecustomize),
        env=env,
        guards=tuple(guards),
    )


def _build_prime_rl_env(shim_plan: PrimeRlSftShimPlan | None) -> dict[str, str] | None:
    if shim_plan is None:
        return None
    env = os.environ.copy()
    shim_pythonpath = shim_plan.env["PYTHONPATH"]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        shim_pythonpath
        if not existing_pythonpath
        else os.pathsep.join((shim_pythonpath, existing_pythonpath))
    )
    for key, value in shim_plan.env.items():
        if key != "PYTHONPATH":
            env[key] = value
    return env


def _command_env(shim_plan: PrimeRlSftShimPlan | None) -> dict[str, str]:
    if shim_plan is None:
        return {}
    return dict(shim_plan.env)


def _prepare_prime_rl_shim(
    spec: PrimeRlSftSpec, work_dir: Path
) -> PrimeRlSftShimPlan | None:
    sample_mean = (
        _normalize_loss_normalization(spec.loss_normalization)
        == _SAMPLE_MEAN_LOSS_NORMALIZATION
    )
    pretokenized_data = (
        _normalize_message_tail_truncation(spec.message_tail_truncation)
        == _CUSTOM_TRAINER_PRETOKENIZED_MODE
    )
    if not sample_mean and not pretokenized_data:
        return None
    return _write_prime_rl_sft_compat_shim(
        work_dir, sample_mean=sample_mean, pretokenized_data=pretokenized_data
    )


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


@dataclass(frozen=True)
class _JsonlTransformStats:
    tool_defs_removed_rows: int | None = None
    chat_template_kwargs_rows: int | None = None
    message_tail_truncated_rows: int | None = None
    message_tail_max_area: int | None = None
    message_tail_max_tokens_before: int | None = None
    message_tail_max_tokens_after: int | None = None
    custom_trainer_pretokenized_rows: int | None = None
    custom_trainer_pretokenized_trainable_tokens: int | None = None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_dataset_dir(work_dir: Path) -> Path:
    dataset_dir = work_dir / "prime-rl-dataset-staging"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    return dataset_dir


def _finalize_staged_dataset_dir(work_dir: Path, dataset_dir: Path) -> Path:
    train_jsonl = dataset_dir / "train.jsonl"
    digest = _sha256_file(train_jsonl)[:12]
    final_dir = work_dir / f"prime-rl-dataset-{digest}"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    dataset_dir.rename(final_dir)
    return final_dir


def _load_tail_truncation_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "--message-tail-truncation requires transformers in the active "
            "environment so BenchFlow can match Prime-RL tokenizer lengths"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _normalize_tool_call_for_custom_trainer(call: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(call))
    function = normalized.get("function")
    if not isinstance(function, dict):
        function = {
            "name": normalized.get("name") or "tool",
            "arguments": normalized.get("arguments") or {},
        }
        normalized = {"type": "function", "function": function}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    function["arguments"] = arguments or {}
    return normalized


def _normalize_messages_for_custom_trainer(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            tool_calls = item.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                tool_calls = []
            item["tool_calls"] = [
                _normalize_tool_call_for_custom_trainer(call)
                for call in tool_calls
                if isinstance(call, Mapping)
            ]
        out.append(item)
    return out


def _render_custom_trainer_full_token_ids(
    tokenizer: Any, row: Mapping[str, Any]
) -> list[int]:
    """Return the untruncated custom-trainer token stream for one source row."""
    messages = _normalize_messages_for_custom_trainer(row.get("messages") or [])
    if not messages:
        return []
    tools = row.get("tools") or None
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return list(token_ids)


def _custom_trainer_prefix_token_count(tokenizer: Any, row: Mapping[str, Any]) -> int:
    messages = _normalize_messages_for_custom_trainer(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        return 0
    prefix_messages = messages[:-1]
    tools = row.get("tools") or None
    try:
        rendered = tokenizer.apply_chat_template(
            prefix_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _custom_trainer_pretokenized_row(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
) -> tuple[dict[str, Any], int, int, int]:
    """Stage one row as the exact shifted tensor sample used by Prime-RL.

    The historical custom trainer passed ``input_ids`` and ``labels`` directly
    to Hugging Face, whose CausalLM loss shifts labels internally. Prime-RL's
    SFT loop shifts before the model call, so BenchFlow materializes the same
    target positions as ``target_ids`` plus a boolean ``loss_mask``.
    """
    full_ids = _render_custom_trainer_full_token_ids(tokenizer, row)
    before = len(full_ids)
    if before < 2:
        raise ValueError("cannot stage custom-trainer row with fewer than 2 tokens")

    labels = list(full_ids)
    if row.get("label_last_assistant_only"):
        messages = _normalize_messages_for_custom_trainer(row.get("messages") or [])
        if messages and messages[-1].get("role") == "assistant":
            prefix_len = min(
                _custom_trainer_prefix_token_count(tokenizer, row), len(labels)
            )
            labels = [-100] * prefix_len + labels[prefix_len:]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    if len(full_ids) < 2:
        raise ValueError("custom-trainer tail kept fewer than 2 tokens")

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    loss_mask = [label != -100 for label in labels[1:]]
    if not any(loss_mask):
        raise ValueError("custom-trainer row has no trainable shifted labels")

    staged = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "messages",
            "tools",
            "tool_defs",
            "chat_template_kwargs",
            "label_last_assistant_only",
        }
    }
    staged["benchflow_custom_trainer_pretokenized"] = {
        "original_token_count": before,
        "staged_token_count": len(full_ids),
        "trainable_token_count": int(sum(loss_mask)),
    }
    staged["benchflow_input_ids"] = input_ids
    staged["benchflow_target_ids"] = target_ids
    staged["benchflow_loss_mask"] = loss_mask
    staged["benchflow_position_ids"] = list(range(len(input_ids)))
    return staged, before, len(full_ids), int(sum(loss_mask))


def _normalize_messages_for_prime_rl_render(
    messages: Any, *, default_role: str = "assistant"
) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, str):
        normalized = [{"role": default_role, "content": messages}]
    elif isinstance(messages, Mapping):
        normalized = [dict(messages)]
    elif isinstance(messages, list):
        normalized = []
        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": default_role, "content": message})
            elif isinstance(message, Mapping):
                normalized.append(dict(message))
            else:
                raise ValueError(
                    f"Unsupported message type in Prime-RL row: {type(message)}"
                )
    else:
        raise ValueError(
            f"Unsupported messages container in Prime-RL row: {type(messages)}"
        )

    out: list[dict[str, Any]] = []
    for message in normalized:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = content.strip()
        if "tool_calls" in item:
            tool_calls = []
            for tool_call in item.get("tool_calls") or []:
                if not isinstance(tool_call, Mapping):
                    raise ValueError("tool_calls entries must be objects")
                call = dict(tool_call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    with suppress(json.JSONDecodeError):
                        arguments = json.loads(arguments)
                function["arguments"] = arguments
                call["function"] = function
                tool_calls.append(call)
            item["tool_calls"] = tool_calls
        out.append(item)
    return out


def _prime_rl_tools_from_row(row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw_tools = row.get("tools", row.get("tool_defs"))
    if not raw_tools:
        return None
    if isinstance(raw_tools, str):
        raw_tools = json.loads(raw_tools)
    if not isinstance(raw_tools, list):
        raise ValueError("tools/tool_defs must be a list or JSON-encoded list")

    tools: list[dict[str, Any]] = []
    for item in raw_tools:
        if not isinstance(item, Mapping):
            raise ValueError("tools/tool_defs entries must be objects")
        tool = dict(item)
        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            tools.append(tool)
            continue
        function = {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
        }
        if tool.get("strict") is not None:
            function["strict"] = tool["strict"]
        tools.append({"type": "function", "function": function})
    return tools


def _render_prime_rl_row_token_ids(
    tokenizer: Any,
    row: Mapping[str, Any],
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: Mapping[str, Any],
) -> list[int]:
    kwargs = dict(chat_template_kwargs)
    kwargs["add_generation_prompt"] = False
    kwargs["return_dict"] = False
    if tools is not None:
        kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(
        _normalize_messages_for_prime_rl_render(messages),
        **kwargs,
    )
    return list(rendered)


def _prime_rl_effective_sample_len(tokenizer: Any, token_ids: list[int]) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id in token_ids:
        return max(0, len(token_ids) - 1)
    return len(token_ids)


def _row_effective_sample_len(
    tokenizer: Any,
    row: Mapping[str, Any],
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: Mapping[str, Any],
) -> int:
    token_ids = _render_prime_rl_row_token_ids(
        tokenizer,
        row,
        messages,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    return _prime_rl_effective_sample_len(tokenizer, token_ids)


def _tail_truncate_messages_keep_first_user(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_area: int,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int, bool]:
    raw_messages = row.get("messages")
    messages = (
        [dict(message) for message in raw_messages]
        if isinstance(raw_messages, list)
        else _normalize_messages_for_prime_rl_render(raw_messages)
    )
    before = _row_effective_sample_len(
        tokenizer,
        row,
        messages,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if before <= max_area:
        return messages, before, before, False

    first_user_idx = next(
        (idx for idx, message in enumerate(messages) if message.get("role") == "user"),
        None,
    )
    if first_user_idx is None:
        raise ValueError(
            "cannot tail-truncate an overlength Prime-RL row without a user message"
        )

    first_user = [messages[first_user_idx]]
    low = first_user_idx + 1
    high = len(messages)
    best_start: int | None = None
    best_len: int | None = None
    length_cache: dict[int, int] = {}

    def length_for(start: int) -> int:
        cached = length_cache.get(start)
        if cached is not None:
            return cached
        length = _row_effective_sample_len(
            tokenizer,
            row,
            first_user + messages[start:],
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        length_cache[start] = length
        return length

    while low < high:
        mid = (low + high) // 2
        try:
            candidate_len = length_for(mid)
        except Exception:
            low = mid + 1
            continue
        if candidate_len <= max_area:
            best_start = mid
            best_len = candidate_len
            high = mid
        else:
            low = mid + 1

    if best_start is not None:
        assert best_len is not None
        start = best_start
        while start > first_user_idx + 1:
            try:
                previous_len = length_for(start - 1)
            except Exception:
                break
            if previous_len > max_area:
                break
            start -= 1
            best_len = previous_len
        return first_user + messages[start:], before, int(best_len), True

    only_user_len = _row_effective_sample_len(
        tokenizer,
        row,
        first_user,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if only_user_len > max_area:
        raise ValueError(
            "cannot tail-truncate Prime-RL row: first user message alone exceeds "
            f"the sample window ({only_user_len} > {max_area})"
        )
    return first_user, before, only_user_len, True


def _copy_prime_rl_jsonl(
    source: Path,
    destination: Path,
    *,
    omit_tool_defs: bool,
    chat_template_kwargs: Mapping[str, Any],
    message_tail_truncation: str,
    tokenizer: Any | None,
    message_tail_max_area: int | None,
) -> _JsonlTransformStats:
    """Copy JSONL while applying BenchFlow-owned Prime-RL data transforms."""
    removed_rows = 0
    chat_template_kwargs_rows = 0
    message_tail_truncated_rows = 0
    custom_trainer_pretokenized_rows = 0
    custom_trainer_pretokenized_trainable_tokens = 0
    max_tokens_before: int | None = None
    max_tokens_after: int | None = None
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
            if message_tail_truncation == _CUSTOM_TRAINER_PRETOKENIZED_MODE:
                if omit_tool_defs and ("tool_defs" in row or "tools" in row):
                    removed_rows += 1
                if tokenizer is None or message_tail_max_area is None:
                    raise AssertionError(
                        "custom-trainer pretokenized staging requires tokenizer "
                        "and max area"
                    )
                row, before, after, trainable_tokens = _custom_trainer_pretokenized_row(
                    tokenizer,
                    row,
                    max_length=message_tail_max_area,
                )
                custom_trainer_pretokenized_rows += 1
                custom_trainer_pretokenized_trainable_tokens += trainable_tokens
                message_tail_truncated_rows += int(before > after)
                max_tokens_before = (
                    before
                    if max_tokens_before is None
                    else max(max_tokens_before, before)
                )
                max_tokens_after = (
                    after if max_tokens_after is None else max(max_tokens_after, after)
                )
                dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                continue
            if omit_tool_defs and ("tool_defs" in row or "tools" in row):
                removed_rows += 1
            if omit_tool_defs:
                row.pop("tool_defs", None)
                row.pop("tools", None)
            if chat_template_kwargs:
                existing = row.get("chat_template_kwargs")
                if existing is None:
                    existing = {}
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"{source}: chat_template_kwargs must be an object when present"
                    )
                row["chat_template_kwargs"] = {
                    **existing,
                    **dict(chat_template_kwargs),
                }
                chat_template_kwargs_rows += 1
            if message_tail_truncation != "off":
                if tokenizer is None or message_tail_max_area is None:
                    raise AssertionError(
                        "tail truncation requires tokenizer and max area"
                    )
                tools = _prime_rl_tools_from_row(row)
                row_chat_template_kwargs = row.get("chat_template_kwargs") or {}
                if not isinstance(row_chat_template_kwargs, Mapping):
                    raise ValueError(
                        f"{source}: chat_template_kwargs must be an object when present"
                    )
                messages, before, after, changed = (
                    _tail_truncate_messages_keep_first_user(
                        tokenizer,
                        row,
                        max_area=message_tail_max_area,
                        tools=tools,
                        chat_template_kwargs=row_chat_template_kwargs,
                    )
                )
                max_tokens_before = (
                    before
                    if max_tokens_before is None
                    else max(max_tokens_before, before)
                )
                max_tokens_after = (
                    after if max_tokens_after is None else max(max_tokens_after, after)
                )
                if changed:
                    row["messages"] = messages
                    message_tail_truncated_rows += 1
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return _JsonlTransformStats(
        tool_defs_removed_rows=removed_rows if omit_tool_defs else None,
        chat_template_kwargs_rows=(
            chat_template_kwargs_rows if chat_template_kwargs else None
        ),
        message_tail_truncated_rows=(
            message_tail_truncated_rows if message_tail_truncation != "off" else None
        ),
        message_tail_max_area=(
            message_tail_max_area if message_tail_truncation != "off" else None
        ),
        message_tail_max_tokens_before=max_tokens_before,
        message_tail_max_tokens_after=max_tokens_after,
        custom_trainer_pretokenized_rows=(
            custom_trainer_pretokenized_rows
            if message_tail_truncation == _CUSTOM_TRAINER_PRETOKENIZED_MODE
            else None
        ),
        custom_trainer_pretokenized_trainable_tokens=(
            custom_trainer_pretokenized_trainable_tokens
            if message_tail_truncation == _CUSTOM_TRAINER_PRETOKENIZED_MODE
            else None
        ),
    )


def _prepare_prime_rl_data(
    spec: PrimeRlSftSpec, work_dir: Path, config: Mapping[str, Any]
) -> tuple[PrimeRlSftSpec, PrimeRlSftDatasetPlan | None]:
    """Make local BenchFlow JSONL usable by Prime-RL's ``load_dataset`` path."""
    message_tail_truncation = _normalize_message_tail_truncation(
        spec.message_tail_truncation
    )
    if not spec.data:
        if spec.chat_template_kwargs:
            raise ValueError("--chat-template-kwarg requires --data")
        if message_tail_truncation != "off":
            raise ValueError("--message-tail-truncation requires --data")
        return spec, None
    source_data = spec.data

    tool_defs_mode = _normalize_tool_defs_mode(spec.tool_defs_mode)
    chat_template_kwargs = _parse_chat_template_kwargs(spec.chat_template_kwargs)
    if message_tail_truncation == _CUSTOM_TRAINER_PRETOKENIZED_MODE:
        if tool_defs_mode != "omit":
            raise ValueError(
                f"--message-tail-truncation {_CUSTOM_TRAINER_PRETOKENIZED_MODE} "
                "requires --tool-defs-mode omit because it stages pre-tokenized "
                "custom-trainer tensors instead of chat/tool schema rows"
            )
        if chat_template_kwargs:
            raise ValueError(
                f"--message-tail-truncation {_CUSTOM_TRAINER_PRETOKENIZED_MODE} "
                "stages pre-tokenized tensors and cannot be combined with "
                "--chat-template-kwarg"
            )
    effective_overrides = _override_map(spec.overrides)
    message_tail_max_area: int | None = None
    tokenizer: Any | None = None
    if message_tail_truncation != "off":
        model_name = _string_or_none(
            _resolve_effective_value(config, effective_overrides, "model.name")
        )
        if not model_name:
            raise ValueError(
                "--message-tail-truncation requires model.name in the Prime-RL config "
                "or an explicit --override model.name=..."
            )
        message_tail_max_area = _resolve_sample_max_area(config, effective_overrides)
        tokenizer = _load_tail_truncation_tokenizer(model_name)

    source_path = _local_data_path(spec.data)
    if source_path is None:
        if tool_defs_mode != "preserve":
            raise ValueError(
                "--tool-defs-mode omit requires --data to be a local JSONL file "
                "or a local dataset directory"
            )
        if chat_template_kwargs:
            raise ValueError(
                "--chat-template-kwarg requires --data to be a local JSONL file "
                "or a local dataset directory"
            )
        if message_tail_truncation != "off":
            raise ValueError(
                "--message-tail-truncation requires --data to be a local JSONL "
                "file or a local dataset directory"
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
        should_transform = (
            tool_defs_mode == "omit"
            or bool(chat_template_kwargs)
            or message_tail_truncation != "off"
        )
        if should_transform:
            if not train_jsonl.is_file():
                raise ValueError(
                    f"local Prime-RL data transforms require {source_path} "
                    "to contain train.jsonl"
                )
            dataset_dir = _staged_dataset_dir(work_dir)
            shutil.copytree(source_path, dataset_dir)
            transformed_train_jsonl = dataset_dir / "train.jsonl"
            stats = _copy_prime_rl_jsonl(
                train_jsonl,
                transformed_train_jsonl,
                omit_tool_defs=tool_defs_mode == "omit",
                chat_template_kwargs=chat_template_kwargs,
                message_tail_truncation=message_tail_truncation,
                tokenizer=tokenizer,
                message_tail_max_area=message_tail_max_area,
            )
            dataset_dir = _finalize_staged_dataset_dir(work_dir, dataset_dir)
            transformed_train_jsonl = dataset_dir / "train.jsonl"
            resolved_spec = replace(spec, data=str(dataset_dir))
            return resolved_spec, PrimeRlSftDatasetPlan(
                source_data=source_data,
                resolved_data=str(dataset_dir),
                kind="local_dataset_dir_transformed",
                dataset_dir=str(dataset_dir),
                train_jsonl=str(transformed_train_jsonl),
                tool_defs_mode=tool_defs_mode,
                tool_defs_removed_rows=stats.tool_defs_removed_rows,
                chat_template_kwargs=(
                    dict(chat_template_kwargs) if chat_template_kwargs else None
                ),
                chat_template_kwargs_rows=stats.chat_template_kwargs_rows,
                message_tail_truncation=message_tail_truncation,
                message_tail_truncated_rows=stats.message_tail_truncated_rows,
                message_tail_max_area=stats.message_tail_max_area,
                message_tail_max_tokens_before=stats.message_tail_max_tokens_before,
                message_tail_max_tokens_after=stats.message_tail_max_tokens_after,
                custom_trainer_pretokenized_rows=(
                    stats.custom_trainer_pretokenized_rows
                ),
                custom_trainer_pretokenized_trainable_tokens=(
                    stats.custom_trainer_pretokenized_trainable_tokens
                ),
                validation=validation,
            )
        resolved_spec = replace(spec, data=str(source_path))
        return resolved_spec, PrimeRlSftDatasetPlan(
            source_data=source_data,
            resolved_data=str(source_path),
            kind="local_dataset_dir",
            dataset_dir=str(source_path),
            train_jsonl=str(train_jsonl) if train_jsonl.is_file() else None,
            tool_defs_mode=tool_defs_mode,
            chat_template_kwargs=(
                dict(chat_template_kwargs) if chat_template_kwargs else None
            ),
            message_tail_truncation=message_tail_truncation,
            validation=validation,
        )

    if source_path.suffix != ".jsonl":
        raise ValueError(
            f"--data local files must be Prime-SFT JSONL files, got {source_path}"
        )

    from benchflow.trajectories.export_prime_sft import validate_prime_sft_jsonl

    validation = validate_prime_sft_jsonl(source_path)
    dataset_dir = _staged_dataset_dir(work_dir)
    dataset_dir.mkdir(parents=True)
    train_jsonl = dataset_dir / "train.jsonl"
    stats = _JsonlTransformStats()
    if (
        tool_defs_mode == "omit"
        or chat_template_kwargs
        or message_tail_truncation != "off"
    ):
        stats = _copy_prime_rl_jsonl(
            source_path,
            train_jsonl,
            omit_tool_defs=tool_defs_mode == "omit",
            chat_template_kwargs=chat_template_kwargs,
            message_tail_truncation=message_tail_truncation,
            tokenizer=tokenizer,
            message_tail_max_area=message_tail_max_area,
        )
    else:
        shutil.copy2(source_path, train_jsonl)
    dataset_dir = _finalize_staged_dataset_dir(work_dir, dataset_dir)
    train_jsonl = dataset_dir / "train.jsonl"
    resolved_spec = replace(spec, data=str(dataset_dir))
    return resolved_spec, PrimeRlSftDatasetPlan(
        source_data=source_data,
        resolved_data=str(dataset_dir),
        kind="local_jsonl_packaged",
        dataset_dir=str(dataset_dir),
        train_jsonl=str(train_jsonl),
        tool_defs_mode=tool_defs_mode,
        tool_defs_removed_rows=stats.tool_defs_removed_rows,
        chat_template_kwargs=(
            dict(chat_template_kwargs) if chat_template_kwargs else None
        ),
        chat_template_kwargs_rows=stats.chat_template_kwargs_rows,
        message_tail_truncation=message_tail_truncation,
        message_tail_truncated_rows=stats.message_tail_truncated_rows,
        message_tail_max_area=stats.message_tail_max_area,
        message_tail_max_tokens_before=stats.message_tail_max_tokens_before,
        message_tail_max_tokens_after=stats.message_tail_max_tokens_after,
        custom_trainer_pretokenized_rows=stats.custom_trainer_pretokenized_rows,
        custom_trainer_pretokenized_trainable_tokens=(
            stats.custom_trainer_pretokenized_trainable_tokens
        ),
        validation=validation,
    )


def _initial_manifest(
    spec: PrimeRlSftSpec,
    argv: list[str],
    logs: list[str],
    exposure_plan: PrimeRlSftExposurePlan | None = None,
    dataset_plan: PrimeRlSftDatasetPlan | None = None,
    shim_plan: PrimeRlSftShimPlan | None = None,
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
    if shim_plan is not None:
        manifest.extra["prime_rl_sft_shim"] = shim_plan.to_dict()
    if spec.compat_profile:
        manifest.extra["prime_rl_sft_compat_profile"] = {
            "name": spec.compat_profile,
            "description": (
                "BenchFlow Mobile300 PR828 Prime-RL wrapper settings that match "
                "the historical custom-trainer run where Prime-SFT rows had "
                "tool_defs but no tools and max_steps counted batch-size-1 "
                "micro-batches before gradient accumulation."
            ),
            "resolved_settings": {
                "target_examples": spec.target_examples,
                "target_micro_steps": spec.target_micro_steps,
                "sync_scheduler_to_max_steps": spec.sync_scheduler_to_max_steps,
                "sync_ckpt_to_max_steps": spec.sync_ckpt_to_max_steps,
                "pack_function": spec.pack_function,
                "loss_mask": spec.loss_mask,
                "loss_normalization": spec.loss_normalization,
                "model_attn": spec.model_attn,
                "renderer_mode": spec.renderer_mode,
                "tool_defs_mode": spec.tool_defs_mode,
                "chat_template_kwargs": _parse_chat_template_kwargs(
                    spec.chat_template_kwargs
                ),
                "message_tail_truncation": spec.message_tail_truncation,
            },
            "known_prime_rl_gap": (
                "BenchFlow stages each row as the historical custom trainer's "
                "shifted input_ids/target_ids/loss_mask tensors and enables a "
                "run-local Prime-RL data shim so Prime-RL trains those tensors "
                "without message rerendering. BenchFlow also enables a run-local "
                "sample-mean loss shim because Prime-RL's native SFT loop "
                "normalizes by trainable token count, while the historical custom "
                "trainer averaged per-row losses. The Prime-RL optimizer, "
                "scheduler, checkpoint writer, and batch ordering are still "
                "Prime-RL implementations."
            ),
        }
    return manifest


def run_prime_rl_sft(spec: PrimeRlSftSpec) -> PrimeRlSftResult:
    spec = _apply_compat_profile(spec)
    if spec.cwd is not None and not spec.cwd.is_dir():
        raise ValueError(f"--prime-rl-dir not found: {spec.cwd}")
    config_path = _resolve_config_path(spec.config, spec.cwd)
    config_data = _load_toml(config_path)
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

    launch_spec, dataset_plan = _prepare_prime_rl_data(spec, work_dir, config_data)
    launch = build_prime_rl_sft_launch(launch_spec)
    argv = launch.argv
    shim_plan = _prepare_prime_rl_shim(launch_spec, work_dir)
    command_env = _command_env(shim_plan)
    command_prefix = _shell_quote_env(command_env)
    command_text = (
        f"{command_prefix} {_shell_quote(argv)}"
        if command_prefix
        else _shell_quote(argv)
    )
    command_path.write_text(command_text + "\n", encoding="utf-8")
    manifest = _initial_manifest(
        launch_spec,
        argv,
        [
            str(stdout_path.relative_to(work_dir)),
            str(stderr_path.relative_to(work_dir)),
        ],
        launch.exposure_plan,
        dataset_plan,
        shim_plan,
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
            env=_build_prime_rl_env(shim_plan),
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
