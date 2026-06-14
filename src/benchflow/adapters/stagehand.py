"""Inbound adapter for official Stagehand eval task slices.

Stagehand evals define benchmark tasks as TypeScript modules. This adapter
keeps the benchmark layer as a descriptor-only import: it translates a
``stagehand-task.json`` task slice into BenchFlow's native inbound contract,
while the Stagehand agent loop remains an agent adapter.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundSupportReport,
    InboundTask,
    UnsupportedInboundTaskError,
    carry_native_subtrees,
    manifest_from_task_config,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.task.config import TaskConfig

STAGEHAND_TASK_FILE = "stagehand-task.json"

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._/-]+")
_STRING_LITERAL = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`'
_COMPAT_KEYS = (
    "task_id",
    "benchmark",
    "category",
    "instruction",
    "start_url",
    "expected_answer",
    "expected_url",
    "success_check",
    "source_file",
    "upstream_commit",
    "max_steps",
    "original_runner",
)


class StagehandEvalAdapter:
    """Translate a Stagehand eval descriptor into an ``InboundTask``."""

    source = "stagehand-evals"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        root = Path(task_dir)
        descriptor_path = root / STAGEHAND_TASK_FILE
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"Stagehand task is missing {STAGEHAND_TASK_FILE}: {descriptor_path}"
            )

        raw = _load_descriptor(descriptor_path)
        task_id = _required_string(raw, "task_id", descriptor_path)
        instruction = _instruction(raw, descriptor_path)
        short_name = _task_slug(task_id)
        config = cls._build_config(name=f"stagehand/{short_name}", raw=raw)
        assert config.task is not None
        manifest = cls._load_manifest(root, name=config.task.name, config=config)
        files = cls._build_file_map(root)
        generated_files = _generated_files(raw)

        return InboundTask(
            name=short_name,
            source=cls.source,
            instruction=instruction,
            manifest=manifest,
            config=config,
            files=files,
            generated_files=generated_files,
            compatibility=InboundCompatibility(
                source=cls.source,
                config_extra=_compat_metadata(raw),
                config_extra_paths=(STAGEHAND_TASK_FILE,),
            ),
        )

    @staticmethod
    def _build_config(name: str, raw: dict[str, Any]) -> TaskConfig:
        timeout = _optional_positive_float(raw, "timeout_sec")
        agent_timeout = _optional_positive_float(raw, "agent_timeout_sec") or timeout
        verifier_timeout = (
            _optional_positive_float(raw, "verifier_timeout_sec") or timeout
        )
        task_id = _required_string(raw, "task_id", Path(STAGEHAND_TASK_FILE))
        metadata: dict[str, Any] = {
            "benchmark": raw.get("benchmark") or "stagehand-evals",
            "stagehand": {
                key: raw[key]
                for key in (
                    "task_id",
                    "category",
                    "start_url",
                    "expected_answer",
                    "expected_url",
                    "success_check",
                    "source_file",
                    "upstream_commit",
                    "max_steps",
                )
                if key in raw and raw[key] is not None
            },
        }
        payload: dict[str, Any] = {
            "schema_version": "1.3",
            "task": {
                "name": name,
                "description": f"Stagehand eval task {task_id}",
                "keywords": ["stagehand", "browser", "external-eval"],
            },
            "metadata": metadata,
            "source": StagehandEvalAdapter.source,
        }
        if agent_timeout is not None:
            payload["agent"] = {"timeout_sec": agent_timeout}
        verifier = _verifier_config(raw, verifier_timeout=verifier_timeout)
        if verifier:
            payload["verifier"] = verifier
        if raw.get("docker_image"):
            payload["environment"] = {"docker_image": str(raw["docker_image"])}
        return TaskConfig.model_validate(payload)

    @staticmethod
    def _load_manifest(
        root: Path, *, name: str, config: TaskConfig
    ) -> EnvironmentManifest:
        manifest_path = root / "environment.toml"
        if manifest_path.is_file():
            return EnvironmentManifest.model_validate_toml(manifest_path.read_text())
        return manifest_from_task_config(name=name, config=config)

    @staticmethod
    def _build_file_map(root: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}

        def _place(native: str, src: Path) -> None:
            existing = files.get(native)
            if existing is not None and existing != src:
                raise ValueError(
                    "Stagehand task file map collision for "
                    f"{native!r}: {existing} vs {src}"
                )
            files[native] = src

        carry_native_subtrees(root, _place)
        return files


def official_task_descriptor_from_source(
    source: str,
    *,
    source_file: str | Path | None = None,
    upstream_commit: str | None = None,
    benchmark: str = "stagehand-evals",
    timeout_sec: int = 1800,
) -> dict[str, Any]:
    """Extract the common official Stagehand agent-task shape.

    This intentionally supports a conservative subset: tasks with a static
    instruction and a deterministic URL success check. Stagehand verifier /
    expected-answer tasks are reported as unsupported until their reward
    mapping is explicit in BenchFlow.
    """

    task_id = _extract_task_id(source, source_file=source_file)
    instruction = _extract_instruction(source)
    if instruction is None:
        raise _unsupported(
            task_id=task_id,
            reason="static Stagehand agent instruction was not found",
            source_file=source_file,
            issue="missing-static-instruction",
        )

    descriptor: dict[str, Any] = {
        "task_id": task_id,
        "benchmark": benchmark,
        "category": task_id.split("/", 1)[0] if "/" in task_id else "stagehand",
        "instruction": instruction,
        "timeout_sec": timeout_sec,
        "original_runner": {
            "framework": "stagehand-evals",
            "task": task_id,
            "command": f"evals run {task_id}",
        },
    }
    start_url = _extract_start_url(source)
    if start_url is not None:
        descriptor["start_url"] = start_url
    max_steps = _extract_max_steps(source)
    if max_steps is not None:
        descriptor["max_steps"] = max_steps
    expected = _extract_expected_answer(source)
    if expected is not None:
        descriptor["expected_answer"] = expected
    success_check = _extract_success_check(source)
    if success_check is not None:
        descriptor["success_check"] = success_check
        if success_check["type"] in {"url_exact", "url_contains"}:
            descriptor["expected_url"] = success_check["value"]
    elif "runWithVerifier(" in source:
        issue = (
            "stagehand-expected-answer-verifier-not-mapped"
            if expected is not None
            else "stagehand-verifier-not-mapped"
        )
        raise _unsupported(
            task_id=task_id,
            reason=(
                "task uses Stagehand's verifier adapter, but BenchFlow does "
                "not yet map that verifier/reward contract"
            ),
            source_file=source_file,
            issue=issue,
        )
    else:
        raise _unsupported(
            task_id=task_id,
            reason="task has no deterministic success check mapped to BenchFlow",
            source_file=source_file,
            issue="stagehand-success-check-not-mapped",
        )

    if source_file is not None:
        descriptor["source_file"] = str(source_file)
    if upstream_commit is not None:
        descriptor["upstream_commit"] = upstream_commit
    return descriptor


def support_report_from_source(
    source: str,
    *,
    source_file: str | Path | None = None,
) -> InboundSupportReport:
    """Return provider-honest support status for one Stagehand task source."""

    try:
        descriptor = official_task_descriptor_from_source(
            source, source_file=source_file
        )
    except UnsupportedInboundTaskError as exc:
        return exc.report
    return InboundSupportReport(
        source=StagehandEvalAdapter.source,
        supported=True,
        task_id=str(descriptor["task_id"]),
        dataset="stagehand-evals",
        details={
            "task_shape": "stagehand-agent-task",
            "success_check": descriptor.get("success_check"),
        },
    )


def _load_descriptor(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Stagehand task JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Stagehand task JSON must be an object: {path}")
    return raw


def _required_string(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stagehand task JSON must define a non-empty {key!r}: {path}")
    return value.strip()


def _instruction(raw: dict[str, Any], path: Path) -> str:
    instruction = _required_string(raw, "instruction", path)
    lines: list[str] = []
    start_url = _optional_string(raw.get("start_url"))
    if start_url is not None:
        lines.append(f"Open {start_url}.")
    lines.append(instruction)
    expected = _optional_string(raw.get("expected_answer"))
    if expected is not None:
        lines.append(f"Final answer should match: {expected}")
    max_steps = raw.get("max_steps")
    if isinstance(max_steps, int) and max_steps > 0:
        lines.append(f"Stagehand max browser steps: {max_steps}.")
    return "\n\n".join(lines).strip() + "\n"


def _task_slug(task_id: str) -> str:
    slug = _TASK_ID_INVALID.sub("-", task_id.strip().lower()).strip("-._/")
    slug = slug.replace("/", "-")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_positive_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stagehand task {key!r} must be numeric") from exc
    if number <= 0:
        raise ValueError(f"Stagehand task {key!r} must be positive")
    return number


def _verifier_config(
    raw: dict[str, Any],
    *,
    verifier_timeout: float | None,
) -> dict[str, Any]:
    verifier: dict[str, Any] = {}
    if verifier_timeout is not None:
        verifier["timeout_sec"] = verifier_timeout
    raw_verifier = raw.get("verifier")
    if isinstance(raw_verifier, dict):
        verifier_type = _optional_string(raw_verifier.get("type"))
        if verifier_type is not None:
            verifier["type"] = verifier_type
        raw_timeout = _optional_positive_float(raw_verifier, "timeout_sec")
        if raw_timeout is not None:
            verifier["timeout_sec"] = raw_timeout
        judge = raw_verifier.get("judge")
        if isinstance(judge, dict):
            judge_payload: dict[str, Any] = {}
            for key in ("model", "rubric_path", "input_dir", "input_type", "context"):
                value = _optional_string(judge.get(key))
                if value is not None:
                    judge_payload[key] = value
            if judge_payload:
                verifier["judge"] = judge_payload
        env = raw_verifier.get("env")
        if isinstance(env, dict):
            verifier["env"] = {
                str(key): str(value)
                for key, value in env.items()
                if isinstance(key, str) and key
            }
    return verifier


def _compat_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in _COMPAT_KEYS if key in raw}


def _generated_files(raw: dict[str, Any]) -> dict[str, str | bytes]:
    success_check = raw.get("success_check")
    if not isinstance(success_check, dict):
        return {}
    check_type = success_check.get("type")
    value = success_check.get("value")
    if check_type not in {"url_exact", "url_contains"} or not isinstance(value, str):
        return {}
    return {"tests/test.sh": _url_verifier_script(check_type=check_type, value=value)}


def _url_verifier_script(*, check_type: object, value: str) -> str:
    predicate = (
        "current == expected" if check_type == "url_exact" else "expected in current"
    )
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json
        import sys
        from pathlib import Path

        expected = {value!r}
        Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
        Path("/logs/artifacts").mkdir(parents=True, exist_ok=True)


        def emit(reward, current, *, note=None):
            Path("/logs/verifier/reward.txt").write_text(f"{{reward}}\\n")
            Path("/logs/verifier/reward.json").write_text(
                json.dumps({{"reward": reward}}) + "\\n"
            )
            payload = {{
                "reward": reward,
                "current_url": current,
                "expected_url": expected,
            }}
            if note is not None:
                payload["note"] = note
            Path("/logs/artifacts/stagehand-url-verifier.json").write_text(
                json.dumps(payload, indent=2) + "\\n"
            )
            print(json.dumps({{"reward": reward, "current_url": current}}))
            sys.exit(0)


        artifact_path = Path("/logs/artifacts/browser-use-smoke-trace.json")
        # No trace (e.g. an anti-bot/Cloudflare page blocked the browser so the
        # agent produced no artifact) is a clean reward-0 outcome, not a crash:
        # the run stays comparable instead of going unscored.
        if not artifact_path.is_file():
            print(
                f"missing Stagehand artifact: {{artifact_path}} -> reward 0",
                file=sys.stderr,
            )
            emit(0.0, "", note="missing-trace")
        try:
            artifact = json.loads(artifact_path.read_text())
        except (OSError, ValueError) as exc:
            print(
                f"unreadable Stagehand artifact: {{artifact_path}}: {{exc}} -> reward 0",
                file=sys.stderr,
            )
            emit(0.0, "", note="unreadable-trace")
        current = str(artifact.get("stagehand_current_url") or "")
        reward = 1.0 if ({predicate}) else 0.0
        emit(reward, current)
        """
    )


def _extract_task_id(source: str, *, source_file: str | Path | None) -> str:
    match = re.search(
        rf"name\s*:\s*(?P<literal>{_STRING_LITERAL})",
        source,
        flags=re.DOTALL,
    )
    if match:
        return _decode_js_string(match.group("literal")).strip()
    if source_file is not None:
        path = Path(source_file)
        if path.parent.name:
            return f"{path.parent.name}/{path.stem}"
        return path.stem
    return "stagehand-task"


def _extract_instruction(source: str) -> str | None:
    direct = re.search(
        rf"agent\.execute\s*\(\s*\{{.*?instruction\s*:\s*(?P<value>{_STRING_LITERAL}|[A-Za-z_$][\w$]*)",
        source,
        flags=re.DOTALL,
    )
    if direct:
        value = direct.group("value")
        if _is_literal(value):
            return _decode_js_string(value).strip()
        resolved = _extract_const_string(source, value)
        if resolved is not None:
            return resolved.strip()
    if re.search(r"\binstruction\s*,", source):
        resolved = _extract_const_string(source, "instruction")
        if resolved is not None:
            return resolved.strip()
    return _extract_const_string(source, "instruction")


def _extract_start_url(source: str) -> str | None:
    match = re.search(
        rf"page\.goto\s*\(\s*(?P<value>{_STRING_LITERAL}|[A-Za-z_$][\w$]*)",
        source,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = match.group("value")
    if _is_literal(value):
        return _decode_js_string(value).strip()
    resolved = _extract_const_string(source, value)
    return resolved.strip() if resolved is not None else None


def _extract_expected_answer(source: str) -> str | None:
    if "expectedAnswer" not in source:
        return None
    expected = _extract_const_string(source, "expected")
    return expected.strip() if expected is not None else None


def _extract_success_check(source: str) -> dict[str, str] | None:
    exact = re.search(
        rf"\burl\s*===\s*(?P<literal>{_STRING_LITERAL})",
        source,
        flags=re.DOTALL,
    )
    if exact:
        return {
            "type": "url_exact",
            "value": _decode_js_string(exact.group("literal")).strip(),
        }
    contains = re.search(
        rf"(?:page\.url\(\)|\burl)\.includes\(\s*(?P<literal>{_STRING_LITERAL})",
        source,
        flags=re.DOTALL,
    )
    if contains:
        return {
            "type": "url_contains",
            "value": _decode_js_string(contains.group("literal")).strip(),
        }
    return None


def _extract_max_steps(source: str) -> int | None:
    match = re.search(
        r"AGENT_EVAL_MAX_STEPS\)\s*\|\|\s*(?P<steps>\d+)",
        source,
    )
    if not match:
        return None
    return int(match.group("steps"))


def _extract_const_string(source: str, name: str) -> str | None:
    match = re.search(
        rf"\bconst\s+{re.escape(name)}\s*=\s*(?P<literal>{_STRING_LITERAL})\s*;",
        source,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return _decode_js_string(match.group("literal"))


def _is_literal(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and stripped[0] in {"'", '"', "`"}


def _decode_js_string(value: str) -> str:
    stripped = value.strip()
    if len(stripped) < 2:
        return stripped
    quote = stripped[0]
    body = stripped[1:-1]
    if quote == "`":
        return body.replace("\\`", "`").replace("\\n", "\n")
    return json.loads(f'"{body.replace(chr(34), chr(92) + chr(34))}"')


def _unsupported(
    *,
    task_id: str,
    reason: str,
    source_file: str | Path | None,
    issue: str,
) -> UnsupportedInboundTaskError:
    return UnsupportedInboundTaskError(
        InboundSupportReport(
            source=StagehandEvalAdapter.source,
            supported=False,
            task_id=task_id,
            dataset="stagehand-evals",
            reason=reason,
            details={
                "issue": issue,
                "source_file": str(source_file) if source_file is not None else None,
                "required_mapping": (
                    "static instruction, setup URL, and deterministic or "
                    "verifier-backed reward mapping"
                ),
            },
        )
    )


def from_stagehand_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate a Stagehand task directory."""

    return StagehandEvalAdapter.from_task_dir(task_dir)
