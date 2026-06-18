"""Inbound adapter for public use-computer cookbook task slices."""

from __future__ import annotations

import json
import re
import shlex
import sys
import tomllib
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
from benchflow.task.imports import import_task_config_toml

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")
_KNOWN_DATASET_TAGS = {
    "cuagym",
    "macosworld",
    "osworld",
    "waa",
    "windowsagentarena",
}
_CUAGYM_SUPPORTED_SETUP_KINDS = {"command", "download", "execute", "launch", "sleep"}
_CUAGYM_SUPPORTED_POSTCONFIG_KINDS = {"command", "execute", "sleep"}
_CUAGYM_PYAUTOGUI_SAVE_COMMANDS = {
    ("python", "-c", 'import pyautogui; pyautogui.hotkey("ctrl", "s");'),
    ("python3", "-c", 'import pyautogui; pyautogui.hotkey("ctrl", "s");'),
}
_CUAGYM_UNMAPPED_SETUP_MARKERS = {
    "launch_gui(",
    "subprocess.Popen(",
    "xdg-open",
    " code ",
    "code ",
    "libreoffice",
    "soffice",
    "vlc",
    "evince",
    "firefox",
    "google-chrome",
    "chromium",
}
_CUAGYM_STDLIB_REWARD_IMPORTS = set(getattr(sys, "stdlib_module_names", ())) | {
    "ast",
    "csv",
    "glob",
    "hashlib",
    "io",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "shutil",
    "sqlite3",
    "subprocess",
    "sys",
    "time",
    "typing",
}
_CUAGYM_REWARD_DEPENDENCY_PACKAGES = {
    "PIL": "Pillow",
    "PyPDF2": "PyPDF2",
    "docx": "python-docx",
    "gimpformats": "gimpformats",
    "numpy": "numpy",
    "odf": "odfpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "pptx": "python-pptx",
    "pyperclip": "pyperclip",
}
_CUAGYM_SUPPORTED_REWARD_IMPORTS = _CUAGYM_STDLIB_REWARD_IMPORTS | set(
    _CUAGYM_REWARD_DEPENDENCY_PACKAGES
)


class UseComputerCookbookAdapter:
    """Translate use-computer cookbook task dirs into ``InboundTask`` records."""

    source = "use-computer-cookbook"

    @classmethod
    def is_task_dir(cls, task_dir: Path | str) -> bool:
        return cls.support_report(task_dir) is not None

    @classmethod
    def support_report(cls, task_dir: Path | str) -> InboundSupportReport | None:
        root = Path(task_dir)
        if (
            not (root / "task.toml").is_file()
            or not (root / "instruction.md").is_file()
        ):
            return None

        metadata = _task_toml_metadata(root)
        osworld_task = _load_osworld_task(root)
        dataset = _dataset(metadata, osworld_task=osworld_task)
        cuagym_smoke = _load_cuagym_smoke(root, dataset=dataset)
        cuagym_task = _load_cuagym_python_task(root, dataset=dataset)
        cuagym_issue = _cuagym_python_task_issue(root, dataset=dataset)
        if dataset == "use-computer":
            return None
        task_id = _task_id(
            root,
            osworld_task=osworld_task,
            cuagym_task=cuagym_task,
        )
        if cuagym_issue is not None and isinstance(cuagym_issue.get("task_id"), str):
            task_id = str(cuagym_issue["task_id"])
        if osworld_task is not None:
            return InboundSupportReport(
                source=cls.source,
                supported=True,
                task_id=task_id,
                dataset=dataset,
                details={"signature": "tests/osworld_task.json"},
            )
        if cuagym_smoke is not None:
            return InboundSupportReport(
                source=cls.source,
                supported=True,
                task_id=task_id,
                dataset=dataset,
                details={"signature": "tests/setup/pre_command.sh:cuagym-smoke"},
            )
        if cuagym_task is not None:
            return InboundSupportReport(
                source=cls.source,
                supported=True,
                task_id=task_id,
                dataset=dataset,
                details={"signature": "tests/cuagym/original/task.json:python-reward"},
            )
        if cuagym_issue is not None:
            details = {
                "tags": _metadata_tags(metadata),
                **dict(cuagym_issue.get("details") or {}),
            }
            return InboundSupportReport(
                source=cls.source,
                supported=False,
                task_id=task_id,
                dataset=dataset,
                reason=str(cuagym_issue["reason"]),
                details=details,
            )
        return InboundSupportReport(
            source=cls.source,
            supported=False,
            task_id=task_id,
            dataset=dataset,
            reason=_unsupported_reason(dataset),
            details={"tags": _metadata_tags(metadata)},
        )

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        root = Path(task_dir)
        config_path = root / "task.toml"
        instruction_path = root / "instruction.md"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"use-computer cookbook task is missing task.toml: {config_path}"
            )
        if not instruction_path.is_file():
            raise FileNotFoundError(
                "use-computer cookbook task is missing instruction.md: "
                f"{instruction_path}"
            )

        imported = import_task_config_toml(config_path.read_text(), source=cls.source)
        osworld_task = _load_osworld_task(root)
        dataset = _dataset(imported.config.metadata, osworld_task=osworld_task)
        cuagym_smoke = _load_cuagym_smoke(root, dataset=dataset)
        cuagym_task = _load_cuagym_python_task(root, dataset=dataset)
        support = cls.support_report(root)
        if support is not None and not support.supported:
            raise UnsupportedInboundTaskError(support)
        if osworld_task is None and cuagym_smoke is None and cuagym_task is None:
            raise UnsupportedInboundTaskError(
                InboundSupportReport(
                    source=cls.source,
                    supported=False,
                    task_id=root.name,
                    dataset=dataset,
                    reason=(
                        "missing supported task signature tests/osworld_task.json "
                        "or supported CUA-Gym setup/reward signature"
                    ),
                )
            )
        task_id = _task_id(
            root,
            osworld_task=osworld_task,
            cuagym_task=cuagym_task,
        )
        expected_result = _expected_result(osworld_task, cuagym_smoke, cuagym_task)
        instruction = _instruction(
            instruction_path.read_text(),
            expected_result=expected_result,
            append_expected=osworld_task is None,
        )
        config = _config_with_metadata(
            imported.config,
            root=root,
            task_id=task_id,
            dataset=dataset,
            expected_result=expected_result,
            osworld_task=osworld_task,
            cuagym_smoke=cuagym_smoke,
            cuagym_task=cuagym_task,
        )
        assert config.task is not None

        files = cls._build_file_map(
            root,
            osworld_task=osworld_task,
            cuagym_smoke=cuagym_smoke,
            cuagym_task=cuagym_task,
        )
        generated_files = _generated_files(
            osworld_task=osworld_task,
            cuagym_smoke=cuagym_smoke,
            cuagym_task=cuagym_task,
            expected_result=expected_result,
        )

        return InboundTask(
            name=_task_slug(task_id),
            source=cls.source,
            instruction=instruction,
            manifest=cls._load_manifest(root, name=config.task.name, config=config),
            config=config,
            files=files,
            generated_files=generated_files,
            compatibility=InboundCompatibility(
                source=cls.source,
                config_extra={
                    "task_id": task_id,
                    "dataset": dataset,
                    "expected_result": expected_result,
                    "osworld_task": osworld_task,
                    "cuagym_smoke": cuagym_smoke,
                    "cuagym_task": cuagym_task,
                    **imported.report.extra,
                },
                config_extra_paths=_compat_paths(
                    osworld_task,
                    cuagym_smoke,
                    cuagym_task,
                    imported.report.extra_paths,
                ),
            ),
        )

    @staticmethod
    def _load_manifest(
        root: Path, *, name: str, config: TaskConfig
    ) -> EnvironmentManifest:
        manifest_path = root / "environment.toml"
        if manifest_path.is_file():
            return EnvironmentManifest.model_validate_toml(manifest_path.read_text())
        return manifest_from_task_config(name=name, config=config)

    @staticmethod
    def _build_file_map(
        root: Path,
        *,
        osworld_task: dict[str, Any] | None,
        cuagym_smoke: dict[str, Any] | None,
        cuagym_task: dict[str, Any] | None,
    ) -> dict[str, Path]:
        files: dict[str, Path] = {}

        def _place(native: str, src: Path) -> None:
            if osworld_task is not None and native == "tests/test.sh":
                return
            if osworld_task is not None and native == "tests/osworld_task.json":
                return
            if cuagym_smoke is not None and native == "tests/test.sh":
                return
            if cuagym_task is not None and native == "tests/test.sh":
                return
            existing = files.get(native)
            if existing is not None and existing != src:
                raise ValueError(
                    "use-computer cookbook file map collision for "
                    f"{native!r}: {existing} vs {src}"
                )
            files[native] = src

        carry_native_subtrees(root, _place)
        return files


def _load_osworld_task(root: Path) -> dict[str, Any] | None:
    path = root / "tests" / "osworld_task.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"OSWorld task JSON must be an object: {path}")
    return raw


def _load_cuagym_smoke(root: Path, *, dataset: str) -> dict[str, Any] | None:
    if dataset != "cuagym":
        return None
    setup_path = root / "tests" / "setup" / "pre_command.sh"
    if not setup_path.is_file():
        return None
    try:
        setup = setup_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "runner-cuagym-setup-ok" not in setup:
        return None
    return {
        "setup_marker": "/tmp/runner-cuagym-setup-ok",
        "signature": "tests/setup/pre_command.sh:runner-cuagym-setup-ok",
    }


def _load_cuagym_python_task(root: Path, *, dataset: str) -> dict[str, Any] | None:
    task, _issue = _analyze_cuagym_python_task(root, dataset=dataset)
    return task


def _cuagym_python_task_issue(root: Path, *, dataset: str) -> dict[str, Any] | None:
    _task, issue = _analyze_cuagym_python_task(root, dataset=dataset)
    return issue


def _analyze_cuagym_python_task(
    root: Path,
    *,
    dataset: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if dataset != "cuagym":
        return None, None
    original = root / "tests" / "cuagym" / "original"
    task_json_path = original / "task.json"
    reward_path = original / "reward.py"
    if not task_json_path.is_file() or not reward_path.is_file():
        return None, None
    task_id = root.name
    signature = "tests/cuagym/original/task.json:python-reward"

    def issue(reason: str, code: str, **details: Any) -> tuple[None, dict[str, Any]]:
        return None, {
            "task_id": task_id,
            "reason": reason,
            "details": {
                "signature": signature,
                "issue": code,
                **details,
            },
        }

    try:
        task_json = json.loads(task_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return issue(
            f"CUA-Gym Python-reward task has invalid task.json: {exc}",
            "invalid-task-json",
        )
    if not isinstance(task_json, dict):
        return issue(
            "CUA-Gym Python-reward task.json must be an object",
            "invalid-task-json",
        )
    task_id = str(task_json.get("id") or root.name)
    evaluator = task_json.get("evaluator") or {}
    if not isinstance(evaluator, dict) or evaluator.get("type") != "python":
        return issue(
            "CUA-Gym Python-reward task uses an unsupported evaluator type",
            "unsupported-evaluator",
            evaluator_type=(
                evaluator.get("type") if isinstance(evaluator, dict) else None
            ),
        )
    if str(evaluator.get("url") or "") not in {"./reward.py", "reward.py"}:
        return issue(
            "CUA-Gym Python-reward task uses an unsupported reward.py location",
            "unsupported-reward-location",
            reward_url=str(evaluator.get("url") or ""),
        )
    postconfig, postconfig_issue = _cuagym_supported_postconfig(evaluator)
    if postconfig_issue is not None:
        return issue(
            str(postconfig_issue["reason"]),
            "unsupported-evaluator-postconfig",
            **dict(postconfig_issue.get("details") or {}),
        )
    app_type = str(task_json.get("app_type") or "")
    if "mock" in app_type.lower():
        return issue(
            "CUA-Gym Python-reward task uses a mock app type",
            "mock-app-type",
            app_type=app_type,
        )
    placeholders = _cuagym_placeholders_in_dir(original)
    if placeholders:
        return issue(
            "CUA-Gym Python-reward task contains unresolved CUA-Gym placeholders",
            "unresolved-placeholders",
            placeholders=sorted(placeholders),
        )
    setup_kinds = _cuagym_setup_kinds(task_json)
    unsupported_setup = setup_kinds - _CUAGYM_SUPPORTED_SETUP_KINDS
    if unsupported_setup:
        return issue(
            "CUA-Gym Python-reward task uses unsupported setup steps",
            "unsupported-setup-steps",
            setup_kinds=sorted(setup_kinds),
            unsupported_setup_kinds=sorted(unsupported_setup),
        )
    unmapped_setup_markers = _cuagym_unmapped_setup_markers(original, task_json)
    if unmapped_setup_markers:
        return issue(
            "CUA-Gym Python-reward task setup launches an unmapped desktop app/runtime",
            "unmapped-setup-launchers",
            unmapped_setup_markers=sorted(unmapped_setup_markers),
        )
    reward_source = reward_path.read_text(errors="replace")
    reward_compile_error = _reward_compile_error(reward_source, filename=reward_path)
    if reward_compile_error is not None:
        return issue(
            "CUA-Gym Python-reward task reward.py is not executable Python",
            "invalid-reward-python",
            reward_compile_error=reward_compile_error,
        )
    reward_imports = _reward_import_roots(reward_source)
    unsupported_imports = reward_imports - _CUAGYM_SUPPORTED_REWARD_IMPORTS
    if unsupported_imports:
        return issue(
            "CUA-Gym Python-reward task imports reward dependencies outside the stdlib allowlist",
            "unsupported-reward-imports",
            reward_imports=sorted(reward_imports),
            unsupported_reward_imports=sorted(unsupported_imports),
        )
    return {
        "task_id": task_id,
        "app_type": app_type,
        "difficulty": task_json.get("difficulty"),
        "setup_kinds": sorted(setup_kinds),
        "postconfig_kinds": postconfig["kinds"],
        "postconfig_save_hotkey": postconfig["save_hotkey"],
        "reward_imports": sorted(reward_imports),
        "reward_dependencies": _cuagym_reward_dependencies(reward_imports),
        "original_dir": "tests/cuagym/original",
    }, None


def _task_toml_metadata(root: Path) -> dict[str, Any]:
    try:
        raw = tomllib.loads((root / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    metadata = raw.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _cuagym_placeholders_in_dir(path: Path) -> set[str]:
    placeholders: set[str] = set()
    pattern = re.compile(r"__CUA_GYM_[A-Z0-9_]+__")
    for file_path in path.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in {
            ".json",
            ".md",
            ".py",
            ".sh",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        placeholders.update(pattern.findall(text))
    return placeholders


def _cuagym_setup_kinds(task_json: dict[str, Any]) -> set[str]:
    return _cuagym_step_kinds(task_json.get("config"))


def _cuagym_step_kinds(raw_steps: Any) -> set[str]:
    kinds: set[str] = set()
    steps = raw_steps if isinstance(raw_steps, list) else []
    for step in steps:
        if isinstance(step, dict):
            kinds.add(str(step.get("type") or ""))
    return kinds


def _cuagym_supported_postconfig(
    evaluator: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    postconfig = evaluator.get("postconfig") or []
    if not postconfig:
        return {"kinds": [], "save_hotkey": False}, None
    if not isinstance(postconfig, list):
        return (
            {"kinds": [], "save_hotkey": False},
            {
                "reason": "CUA-Gym Python-reward task has malformed evaluator postconfig",
                "details": {"postconfig_kinds": []},
            },
        )
    kinds = _cuagym_step_kinds(postconfig)
    unsupported_kinds = kinds - _CUAGYM_SUPPORTED_POSTCONFIG_KINDS
    if unsupported_kinds:
        return (
            {"kinds": sorted(kinds), "save_hotkey": False},
            {
                "reason": "CUA-Gym Python-reward task uses unsupported evaluator postconfig steps",
                "details": {
                    "postconfig_kinds": sorted(kinds),
                    "unsupported_postconfig_kinds": sorted(unsupported_kinds),
                },
            },
        )
    save_hotkey = False
    for step in postconfig:
        if not isinstance(step, dict):
            continue
        kind = str(step.get("type") or "")
        if kind == "sleep":
            continue
        params = step.get("parameters") or {}
        command = params.get("command") if isinstance(params, dict) else None
        if _is_cuagym_pyautogui_save_command(command):
            save_hotkey = True
            continue
        return (
            {"kinds": sorted(kinds), "save_hotkey": save_hotkey},
            {
                "reason": "CUA-Gym Python-reward task has an unsupported evaluator postconfig command",
                "details": {"postconfig_kinds": sorted(kinds)},
            },
        )
    return {"kinds": sorted(kinds), "save_hotkey": save_hotkey}, None


def _is_cuagym_pyautogui_save_command(command: Any) -> bool:
    if isinstance(command, list):
        return tuple(str(part) for part in command) in _CUAGYM_PYAUTOGUI_SAVE_COMMANDS
    return False


def _cuagym_reward_dependencies(reward_imports: set[str]) -> list[str]:
    packages = {
        package
        for import_name, package in _CUAGYM_REWARD_DEPENDENCY_PACKAGES.items()
        if import_name in reward_imports
    }
    return sorted(packages)


def _cuagym_unmapped_setup_markers(
    original: Path,
    task_json: dict[str, Any],
) -> set[str]:
    text_parts = [json.dumps(task_json.get("config") or [])]
    for step in task_json.get("config") or []:
        if not isinstance(step, dict):
            continue
        params = step.get("parameters") or {}
        if not isinstance(params, dict):
            continue
        for item in params.get("files") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            source = (
                original / url[2:]
                if url.startswith("./")
                else original / Path(url).name
            )
            if not source.is_file():
                continue
            try:
                text_parts.append(source.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    setup_text = "\n".join(text_parts)
    return {marker for marker in _CUAGYM_UNMAPPED_SETUP_MARKERS if marker in setup_text}


def _reward_import_roots(source: str) -> set[str]:
    imports: set[str] = set()
    for match in re.finditer(r"^from\s+([A-Za-z0-9_\.]+)\s+import\s+", source, re.M):
        imports.add(match.group(1).split(".", 1)[0])
    for match in re.finditer(r"^import\s+(.+)$", source, re.M):
        for part in match.group(1).split(","):
            token = part.strip().split(" as ", 1)[0].strip()
            if token:
                imports.add(token.split(".", 1)[0])
    return imports


def _reward_compile_error(source: str, *, filename: Path) -> str | None:
    try:
        compile(source, str(filename), "exec")
    except SyntaxError as exc:
        return str(exc)
    return None


def _metadata_tags(metadata: dict[str, Any]) -> list[str]:
    tags = metadata.get("tags", [])
    if not isinstance(tags, list):
        return []
    return [str(tag).lower() for tag in tags]


def _task_id(
    root: Path,
    *,
    osworld_task: dict[str, Any] | None,
    cuagym_task: dict[str, Any] | None = None,
) -> str:
    if osworld_task is not None:
        value = osworld_task.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if cuagym_task is not None:
        value = cuagym_task.get("task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return root.name


def _dataset(metadata: dict[str, Any], *, osworld_task: dict[str, Any] | None) -> str:
    tags = _metadata_tags(metadata)
    for tag in ("osworld", "cuagym", "macosworld", "waa", "windowsagentarena"):
        if tag in tags:
            return tag
    if osworld_task is not None:
        return "osworld"
    return "use-computer"


def _unsupported_reason(dataset: str) -> str:
    match dataset:
        case "cuagym":
            return (
                "CUA-Gym cookbook tasks need provider-honest setup/runtime and "
                "reward.py translation before BenchFlow can run them on Cua; "
                "only the infra-smoke setup-marker slice and strict no-mock "
                "Python-reward tasks with replayable setup and mapped reward "
                "dependencies are currently supported"
            )
        case "macosworld":
            return (
                "macOSWorld cookbook tasks need a macOS desktop provider mapping; "
                "BenchFlow Cua support currently covers Linux/Windows"
            )
        case "waa" | "windowsagentarena":
            return (
                "WindowsAgentArena cookbook tasks need Windows setup/evaluator "
                "translation before BenchFlow can run them on Cua"
            )
        case _:
            return "unsupported use-computer cookbook dataset"


def _expected_result(
    osworld_task: dict[str, Any] | None,
    cuagym_smoke: dict[str, Any] | None,
    cuagym_task: dict[str, Any] | None,
) -> str:
    if osworld_task is None and cuagym_smoke is None and cuagym_task is None:
        return "setup-ok"
    if cuagym_smoke is not None:
        return "setup-ok"
    if cuagym_task is not None:
        return "observed"
    assert osworld_task is not None
    evaluator = osworld_task.get("evaluator") or {}
    result = evaluator.get("result") if isinstance(evaluator, dict) else {}
    command = result.get("command") if isinstance(result, dict) else None
    if isinstance(command, str) and "runner-osworld-setup-ok" in command:
        return "setup-ok"
    return "observed"


def _instruction(raw: str, *, expected_result: str, append_expected: bool = True) -> str:
    text = raw.strip()
    # OSWorld tasks are scored by the real evaluator running the task's check
    # command against system state — NOT by the agent printing a magic string, so
    # we must not graft a "final answer must be exactly ..." onto their instruction.
    if append_expected and "Final answer must be exactly:" not in text:
        text += f"\n\nFinal answer must be exactly: {expected_result}"
    return text + "\n"


def _config_with_metadata(
    config: TaskConfig,
    *,
    root: Path,
    task_id: str,
    dataset: str,
    expected_result: str,
    osworld_task: dict[str, Any] | None,
    cuagym_smoke: dict[str, Any] | None,
    cuagym_task: dict[str, Any] | None,
) -> TaskConfig:
    payload = config.model_dump(by_alias=True, mode="json")
    task_payload = payload.get("task")
    if task_payload is None:
        task_payload = {
            "name": f"use-computer-cookbook/{_task_slug(task_id)}",
            "description": f"use-computer cookbook task {task_id}",
            "keywords": ["computer-use", dataset, "external-eval"],
        }
    else:
        keywords = list(task_payload.get("keywords") or [])
        for keyword in ("computer-use", dataset, "external-eval"):
            if keyword not in keywords:
                keywords.append(keyword)
        task_payload["keywords"] = keywords
    payload["task"] = task_payload
    payload["source"] = UseComputerCookbookAdapter.source
    payload.setdefault("metadata", {})
    payload["metadata"]["use_computer_cookbook"] = {
        "task_id": task_id,
        "dataset": dataset,
        "expected_result": expected_result,
        "osworld": osworld_task is not None,
        "cuagym_smoke": cuagym_smoke is not None,
        "cuagym_task": cuagym_task is not None,
        "source_dir_name": root.name,
    }
    environment = payload.setdefault("environment", {})
    if not environment.get("workdir"):
        environment["workdir"] = "/app"
    if cuagym_task is not None:
        verifier = payload.setdefault("verifier", {})
        verifier["user"] = "root"
    return TaskConfig.model_validate(payload)


def _generated_files(
    *,
    osworld_task: dict[str, Any] | None,
    cuagym_smoke: dict[str, Any] | None,
    cuagym_task: dict[str, Any] | None,
    expected_result: str,
) -> dict[str, str | bytes]:
    if cuagym_task is not None:
        task_id = str(cuagym_task["task_id"])
        return {
            "environment/Dockerfile": _default_dockerfile(),
            "solution/solve.sh": _oracle_script(expected_result),
            "tests/setup/pre_command.sh": _cuagym_setup_script(task_id),
            "tests/test.sh": _cuagym_reward_verifier_script(task_id),
        }

    if osworld_task is not None:
        # Real OSWorld eval: the verifier runs the task's actual evaluator
        # (postconfig + result check command + metric) in the sandbox, NOT a
        # "did the agent print 'observed'" string match.
        return _osworld_verifier_files(osworld_task)

    marker_path = (
        str(cuagym_smoke.get("setup_marker", "/tmp/runner-cuagym-setup-ok"))
        if cuagym_smoke is not None
        else "/tmp/runner-osworld-setup-ok"
    )
    return {
        "environment/Dockerfile": _default_dockerfile(),
        "solution/solve.sh": _oracle_script(expected_result),
        "tests/test.sh": _verifier_script(
            expected_result,
            setup_marker_path=marker_path,
        ),
    }


def _osworld_verifier_files(osworld_task: dict[str, Any]) -> dict[str, str | bytes]:
    """Generate a REAL OSWorld verifier package.

    Carries the (import-resilient) osworld_eval + osworld_metrics modules into the
    verifier dir alongside a runner that reads the task's own evaluator from
    ``osworld_task.json``, runs ``evaluate(task, run_command=<in-sandbox subprocess>)``,
    and writes the OSWorld reward to ``/logs/verifier/reward.{txt,json}`` — the
    benchflow verifier reward contract.
    """
    here = Path(__file__).parent
    files: dict[str, str | bytes] = {
        "environment/Dockerfile": _osworld_dockerfile(),
        "solution/solve.sh": _osworld_oracle_placeholder(),
        "tests/setup/pre_command.sh": _osworld_setup_script(osworld_task),
        "tests/osworld_task.json": json.dumps(osworld_task, indent=2) + "\n",
        "tests/osworld_metrics.py": (here / "osworld_metrics.py").read_text(),
        "tests/osworld_eval.py": (here / "osworld_eval.py").read_text(),
        "tests/osworld_vendor.py": (here / "osworld_vendor.py").read_text(),
        "tests/osworld_getters.py": (here / "osworld_getters.py").read_text(),
        "tests/run_osworld_verifier.py": _OSWORLD_RUNNER,
        "tests/test.sh": _OSWORLD_TEST_SH,
    }
    # Carry the vendored OSWorld evaluator suite (Apache-2.0) so the in-guest
    # verifier scores with OSWorld's own metric/getter code (exact parity). It
    # lands at tests/_osworld_vendor/, which is where osworld_vendor.py resolves
    # _VENDOR_ROOT relative to itself when carried as a sibling.
    vendor = here / "_osworld_vendor"
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.name == "LICENSE"):
            rel = path.relative_to(here).as_posix()  # _osworld_vendor/...
            files[f"tests/{rel}"] = path.read_text(errors="replace")
    return files


def _osworld_dockerfile() -> str:
    # python3 + pip + the vendored OSWorld evaluator-suite deps (spreadsheet/doc/
    # pdf/json/text metrics). The desktop + OSWorld apps come from the desktop
    # backend (Daytona computer-use, or the OSWorld VM), not this base image.
    osworld_deps = (
        "pandas openpyxl python-docx python-pptx pdfplumber PyPDF2 pymupdf "
        "rapidfuzz formulas lxml cssselect xmltodict tldextract Pillow numpy "
        "odfpy mutagen pyyaml"
    )
    return (
        "FROM ubuntu:24.04\n\n"
        "WORKDIR /app\n\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        python3 python3-pip ca-certificates curl \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        "RUN python3 -m pip install --no-cache-dir --break-system-packages \\\n"
        f"        {osworld_deps}\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /tmp /home/user\n"
    )


def _osworld_oracle_placeholder() -> str:
    # OSWorld ships no committable oracle trajectory; the agent eval does not run
    # the oracle. Keep a no-op so the task package is structurally complete.
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "# No oracle solution for real OSWorld tasks (scored by the real evaluator).\n"
    )


_OSWORLD_TEST_SH = (
    "#!/bin/bash\n"
    "set -euo pipefail\n\n"
    "mkdir -p /logs/verifier /logs/artifacts\n"
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'python3 "$HERE/run_osworld_verifier.py"\n'
)

_OSWORLD_RUNNER = '''\
#!/usr/bin/env python3
"""In-sandbox OSWorld verifier: run the task's real evaluator -> benchflow reward."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import osworld_eval  # noqa: E402  (sibling module carried into the verifier dir)


def run_command(command, shell):
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("HOME", "/home/user")
    if isinstance(command, list):
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
    else:
        proc = subprocess.run(
            command,
            shell=bool(shell),
            capture_output=True,
            text=True,
            env=env,
            executable="/bin/bash" if shell else None,
        )
    return proc.stdout


def main() -> None:
    task = json.loads((HERE / "osworld_task.json").read_text())
    password = os.environ.get("CLIENT_PASSWORD", "password")
    try:
        reward = float(osworld_eval.evaluate(task, run_command, password=password))
    except Exception as exc:  # never crash the verifier; score 0 and log
        sys.stderr.write(f"osworld verifier error: {exc}\\n")
        reward = 0.0
    verdir = pathlib.Path("/logs/verifier")
    verdir.mkdir(parents=True, exist_ok=True)
    (verdir / "reward.txt").write_text(f"{reward}\\n")
    (verdir / "reward.json").write_text(json.dumps({"reward": reward}) + "\\n")
    print(f"osworld reward: {reward}")


if __name__ == "__main__":
    main()
'''


def _default_dockerfile() -> str:
    return (
        "FROM ubuntu:24.04\n\n"
        "WORKDIR /app\n\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /tmp\n"
    )


def _oracle_script(expected_result: str) -> str:
    quoted = shlex.quote(expected_result)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        f"printf '%s\\n' {quoted} > /app/computer_use_result.txt\n"
        "cp /app/computer_use_result.txt /app/computer_use_roundtrip.txt\n"
    )


def _verifier_script(expected_result: str, *, setup_marker_path: str) -> str:
    quoted = shlex.quote(expected_result)
    marker = shlex.quote(setup_marker_path)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p /logs/verifier /logs/artifacts\n"
        f"expected={quoted}\n"
        "actual=''\n"
        "roundtrip=''\n"
        "[ -f /app/computer_use_result.txt ] && "
        "actual=\"$(tr -d '\\n' < /app/computer_use_result.txt)\"\n"
        "[ -f /app/computer_use_roundtrip.txt ] && "
        "roundtrip=\"$(tr -d '\\n' < /app/computer_use_roundtrip.txt)\"\n"
        "setup_output=''\n"
        f"[ -f {marker} ] && "
        f"setup_output=\"$(tr -d '\\n' < {marker})\"\n"
        "trace=/logs/artifacts/computer-use-smoke-trace.json\n"
        "reward=0.0\n"
        'if [ "$actual" = "$expected" ] '
        '&& [ "$roundtrip" = "$expected" ] '
        '&& [ "$setup_output" = "setup-ok" ] '
        "&& [ -s /logs/artifacts/computer-use-smoke.png ] "
        '&& [ -f "$trace" ] '
        '&& grep -q \'"screenshots_b64":\' "$trace"; then\n'
        "  reward=1.0\n"
        "fi\n"
        "printf '%s\\n' \"$reward\" > /logs/verifier/reward.txt\n"
        'printf \'{"reward": %s}\\n\' "$reward" > /logs/verifier/reward.json\n'
    )


def _cuagym_setup_script(task_id: str) -> str:
    quoted_id = shlex.quote(task_id)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"TASK_ID={quoted_id}\n"
        'SRC="$HERE/files/original"\n'
        'TASK_DIR="/home/user/cuagym/tasks/$TASK_ID"\n'
        'mkdir -p /home/user /logs/artifacts /logs/verifier "$TASK_DIR"\n'
        'cp -R "$SRC"/. "$TASK_DIR"/\n'
        "HOME=/home/user python3 - \"$TASK_DIR\" <<'PY'\n"
        "import json, os, pathlib, shlex, shutil, subprocess, sys, time\n\n"
        "task_dir = pathlib.Path(sys.argv[1])\n"
        "task_json = json.loads((task_dir / 'task.json').read_text())\n\n"
        "def source_path(source):\n"
        "    if source.startswith('./'):\n"
        "        return task_dir / source[2:]\n"
        "    return task_dir / pathlib.Path(source).name\n\n"
        "def run_command(command):\n"
        "    env = os.environ.copy()\n"
        "    env.setdefault('DISPLAY', ':1')\n"
        "    env.setdefault('HOME', '/home/user')\n"
        "    if isinstance(command, str):\n"
        "        subprocess.run(command, shell=True, check=True, env=env)\n"
        "    elif isinstance(command, list) and command:\n"
        "        subprocess.run([str(part) for part in command], check=True, env=env)\n\n"
        "def launch_command(command):\n"
        "    env = os.environ.copy()\n"
        "    env.setdefault('DISPLAY', ':1')\n"
        "    env.setdefault('HOME', '/home/user')\n"
        "    if isinstance(command, str):\n"
        "        argv = shlex.split(command)\n"
        "    elif isinstance(command, list) and command:\n"
        "        argv = [str(part) for part in command]\n"
        "    else:\n"
        "        return\n"
        "    if argv:\n"
        "        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)\n\n"
        "for step in task_json.get('config') or []:\n"
        "    kind = step.get('type')\n"
        "    params = step.get('parameters') or {}\n"
        "    if kind == 'download':\n"
        "        for item in params.get('files') or []:\n"
        "            src = source_path(str(item.get('url') or ''))\n"
        "            dst = pathlib.Path(str(item.get('path') or ''))\n"
        "            if not src.exists() or not str(dst):\n"
        "                raise FileNotFoundError(f'CUA-Gym setup copy failed: {src} -> {dst}')\n"
        "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "            shutil.copy2(src, dst)\n"
        "    elif kind in {'execute', 'command'}:\n"
        "        run_command(params.get('command') or [])\n"
        "    elif kind == 'launch':\n"
        "        launch_command(params.get('command') or [])\n"
        "    elif kind == 'sleep':\n"
        "        time.sleep(float(params.get('seconds', 1)))\n"
        "    else:\n"
        "        raise RuntimeError(f'unsupported CUA-Gym setup step: {kind}')\n"
        "print('CUA-Gym task setup complete:', task_json.get('id'))\n"
        "PY\n"
    )


def _cuagym_reward_verifier_script(task_id: str) -> str:
    quoted_id = shlex.quote(task_id)
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p /logs/verifier\n"
        f"TASK_ID={quoted_id}\n"
        'TASK_DIR="/home/user/cuagym/tasks/$TASK_ID"\n'
        'if [ ! -f "$TASK_DIR/reward.py" ]; then\n'
        "  printf '0.0\\n' > /logs/verifier/reward.txt\n"
        '  printf \'{"reward": 0.0, "error": "missing reward.py"}\\n\' > /logs/verifier/reward.json\n'
        "  exit 0\n"
        "fi\n"
        'cd "$TASK_DIR" && HOME=/home/user DISPLAY=:1 '
        "XAUTHORITY=/home/cua/.Xauthority python3 - <<'PY'\n"
        "import json, os, subprocess, time\n\n"
        "task_json = json.loads(open('task.json').read())\n"
        "postconfig = (task_json.get('evaluator') or {}).get('postconfig') or []\n\n"
        "def send_save_hotkey():\n"
        "    from Xlib import X, XK, display\n"
        "    from Xlib.ext import xtest\n"
        "    d = display.Display(':1')\n"
        "    ctrl = d.keysym_to_keycode(XK.string_to_keysym('Control_L'))\n"
        "    s_key = d.keysym_to_keycode(XK.string_to_keysym('s'))\n"
        "    xtest.fake_input(d, X.KeyPress, ctrl)\n"
        "    xtest.fake_input(d, X.KeyPress, s_key)\n"
        "    xtest.fake_input(d, X.KeyRelease, s_key)\n"
        "    xtest.fake_input(d, X.KeyRelease, ctrl)\n"
        "    d.sync()\n\n"
        "for step in postconfig:\n"
        "    kind = step.get('type')\n"
        "    params = step.get('parameters') or {}\n"
        "    if kind == 'sleep':\n"
        "        time.sleep(float(params.get('seconds', 1)))\n"
        "    elif kind in {'execute', 'command'}:\n"
        "        command = params.get('command') or []\n"
        "        if command in [\n"
        "            ['python', '-c', 'import pyautogui; pyautogui.hotkey(\"ctrl\", \"s\");'],\n"
        "            ['python3', '-c', 'import pyautogui; pyautogui.hotkey(\"ctrl\", \"s\");'],\n"
        "        ]:\n"
        "            send_save_hotkey()\n"
        "        else:\n"
        "            env = os.environ.copy()\n"
        "            env['DISPLAY'] = ':1'\n"
        "            env['XAUTHORITY'] = '/home/cua/.Xauthority'\n"
        "            env['HOME'] = '/home/user'\n"
        "            subprocess.run(command, check=True, env=env)\n"
        "    else:\n"
        "        raise RuntimeError(f'unsupported CUA-Gym postconfig step: {kind}')\n"
        "PY\n"
        "cd \"$TASK_DIR\" && HOME=/home/user python3 - <<'PY'\n"
        "import ast, importlib, pathlib, subprocess, sys\n\n"
        "dependency_packages = {'PIL': 'Pillow', 'PyPDF2': 'PyPDF2', 'docx': 'python-docx', 'gimpformats': 'gimpformats', 'numpy': 'numpy', 'odf': 'odfpy', 'openpyxl': 'openpyxl', 'pandas': 'pandas', 'pptx': 'python-pptx', 'pyperclip': 'pyperclip'}\n"
        "source = pathlib.Path('reward.py').read_text(errors='replace')\n"
        "tree = ast.parse(source)\n"
        "imports = set()\n"
        "for node in ast.walk(tree):\n"
        "    if isinstance(node, ast.Import):\n"
        "        for alias in node.names:\n"
        "            imports.add(alias.name.split('.', 1)[0])\n"
        "    elif isinstance(node, ast.ImportFrom) and node.module:\n"
        "        imports.add(node.module.split('.', 1)[0])\n"
        "for import_name in sorted(imports):\n"
        "    package = dependency_packages.get(import_name)\n"
        "    if package is None:\n"
        "        continue\n"
        "    try:\n"
        "        importlib.import_module(import_name)\n"
        "    except ImportError:\n"
        "        subprocess.check_call([\n"
        "            sys.executable,\n"
        "            '-m',\n"
        "            'pip',\n"
        "            'install',\n"
        "            '--quiet',\n"
        "            '--disable-pip-version-check',\n"
        "            '--root-user-action=ignore',\n"
        "            package,\n"
        "        ])\n"
        "PY\n"
        'output=$(cd "$TASK_DIR" && HOME=/home/user python3 reward.py 2>&1 || true)\n'
        "printf '%s\\n' \"$output\"\n"
        "REWARD_OUTPUT=\"$output\" python3 - <<'PY'\n"
        "import json, os, re\n"
        "text = os.environ.get('REWARD_OUTPUT', '')\n"
        "matches = re.findall(r'REWARD\\s*[:=]\\s*([0-9]*\\.?[0-9]+)', text, re.I)\n"
        "reward = float(matches[-1]) if matches else 0.0\n"
        "reward = max(0.0, min(1.0, reward))\n"
        "open('/logs/verifier/reward.txt', 'w').write(f'{reward}\\n')\n"
        "open('/logs/verifier/reward.json', 'w').write(json.dumps({'reward': reward}) + '\\n')\n"
        "PY\n"
    )


def _osworld_setup_script(osworld_task: dict[str, Any]) -> str:
    commands: list[str] = []
    for step in osworld_task.get("config") or []:
        if not isinstance(step, dict):
            continue
        if step.get("type") not in {"execute", "command"}:
            continue
        params = step.get("parameters") or {}
        command = params.get("command")
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            commands.append(shlex.join(command))
        elif isinstance(command, str):
            commands.append(command)
    if not commands:
        commands.append("printf 'setup-ok\\n' > /tmp/runner-osworld-setup-ok")
    body = "\n".join(commands)
    return "#!/bin/bash\nset -euo pipefail\n" + body + "\n"


def _compat_paths(
    osworld_task: dict[str, Any] | None,
    cuagym_smoke: dict[str, Any] | None,
    cuagym_task: dict[str, Any] | None,
    extra_paths: tuple[str, ...],
) -> tuple[str, ...]:
    paths = list(extra_paths)
    if osworld_task is not None:
        paths.append("tests/osworld_task.json")
    if cuagym_smoke is not None:
        paths.append("tests/setup/pre_command.sh")
    if cuagym_task is not None:
        paths.append("tests/cuagym/original/task.json")
        paths.append("tests/cuagym/original/reward.py")
    return tuple(sorted(paths))


def _task_slug(task_id: str) -> str:
    slug = _TASK_ID_INVALID.sub("-", task_id.strip().lower()).strip("-._")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug


def from_use_computer_cookbook_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate a use-computer cookbook task directory."""
    return UseComputerCookbookAdapter.from_task_dir(task_dir)
