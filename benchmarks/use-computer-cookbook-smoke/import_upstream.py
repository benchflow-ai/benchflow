#!/usr/bin/env python3
"""Import selected public use-computer cookbook smoke task dirs."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

SMOKE_TASKS = {
    "cuagym": Path("datasets/cuagym/smoke__ubuntu-infra"),
    "osworld": Path("datasets/osworld/ubuntu/smoke__ubuntu-osworld"),
}
CUAGYM_SUPPORTED_SETUP_KINDS = {"command", "download", "execute", "launch", "sleep"}
CUAGYM_SUPPORTED_POSTCONFIG_KINDS = {"command", "execute", "sleep"}
CUAGYM_PYAUTOGUI_SAVE_COMMANDS = {
    ("python", "-c", 'import pyautogui; pyautogui.hotkey("ctrl", "s");'),
    ("python3", "-c", 'import pyautogui; pyautogui.hotkey("ctrl", "s");'),
}
CUAGYM_UNMAPPED_SETUP_MARKERS = {
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
CUAGYM_STDLIB_REWARD_IMPORTS = set(getattr(sys, "stdlib_module_names", ())) | {
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
CUAGYM_REWARD_DEPENDENCY_PACKAGES = {
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
CUAGYM_SUPPORTED_REWARD_IMPORTS = CUAGYM_STDLIB_REWARD_IMPORTS | set(
    CUAGYM_REWARD_DEPENDENCY_PACKAGES
)
CUAGYM_TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
CUAGYM_PLACEHOLDER = re.compile(r"__CUA_GYM_[A-Z0-9_]+__")
CUAGYM_SUPPORT_REPORT_SCHEMA = "benchflow.cuagym-import-support-report.v1"


def import_tasks(
    upstream_repo: Path,
    out_dir: Path,
    *,
    datasets: list[str],
    overwrite: bool = False,
) -> list[Path]:
    outputs: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        source_rel = SMOKE_TASKS[dataset]
        source = upstream_repo / source_rel
        if not source.is_dir():
            raise FileNotFoundError(f"missing upstream cookbook task dir: {source}")
        dest = out_dir / source.name
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"destination exists: {dest}")
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        outputs.append(dest)
    return outputs


def import_cuagym_task(
    source_task_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    task_json_path = source_task_dir / "task.json"
    reward_path = source_task_dir / "reward.py"
    if not task_json_path.is_file() or not reward_path.is_file():
        raise FileNotFoundError(
            "CUA-Gym source task must contain task.json and reward.py: "
            f"{source_task_dir}"
        )
    task_json = json.loads(task_json_path.read_text())
    task_id = str(task_json.get("id") or source_task_dir.name)
    app_type = str(task_json.get("app_type") or "task")
    dest = out_dir / f"{_slug(app_type)}__{task_id}"
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {dest}")
        shutil.rmtree(dest)

    original = dest / "tests" / "cuagym" / "original"
    setup_files = dest / "tests" / "setup" / "files" / "original"
    original.mkdir(parents=True)
    setup_files.mkdir(parents=True)
    shutil.copytree(source_task_dir, original, dirs_exist_ok=True)
    shutil.copytree(source_task_dir, setup_files, dirs_exist_ok=True)
    (dest / "instruction.md").write_text(str(task_json.get("instruction") or ""))
    (dest / "task.toml").write_text(_cuagym_task_toml(task_json))
    return dest


def import_cuagym_tasks(
    source_tasks_root: Path,
    out_dir: Path,
    *,
    app_types: set[str] | None = None,
    difficulties: set[str] | None = None,
    limit: int = 1,
    overwrite: bool = False,
    support_report_out: Path | None = None,
) -> list[Path]:
    """Import supported raw CUA-Gym task dirs from an extracted dataset root."""
    if not source_tasks_root.is_dir():
        raise FileNotFoundError(f"missing CUA-Gym tasks root: {source_tasks_root}")
    outputs: list[Path] = []
    scanned = 0
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    supported_records: list[dict[str, object]] = []
    unsupported_records: list[dict[str, object]] = []
    filtered_records: list[dict[str, object]] = []
    for source_task_dir in sorted(
        path for path in source_tasks_root.iterdir() if path.is_dir()
    ):
        scanned += 1
        metadata = _cuagym_task_metadata(source_task_dir)
        supported, reason = raw_cuagym_task_support(source_task_dir)
        if not supported:
            skipped += 1
            issue = reason or "unsupported"
            skip_reasons[issue] += 1
            unsupported_records.append(
                _cuagym_support_record(
                    source_task_dir,
                    metadata=metadata,
                    status="unsupported",
                    reason=issue,
                )
            )
            continue
        app_type = str(metadata.get("app_type") or "")
        difficulty = str(metadata.get("difficulty") or "")
        if app_types is not None and app_type not in app_types:
            skipped += 1
            skip_reasons["filtered app_type"] += 1
            filtered_records.append(
                _cuagym_support_record(
                    source_task_dir,
                    metadata=metadata,
                    status="filtered",
                    reason="filtered app_type",
                )
            )
            continue
        if difficulties is not None and difficulty not in difficulties:
            skipped += 1
            skip_reasons["filtered difficulty"] += 1
            filtered_records.append(
                _cuagym_support_record(
                    source_task_dir,
                    metadata=metadata,
                    status="filtered",
                    reason="filtered difficulty",
                )
            )
            continue
        imported = import_cuagym_task(source_task_dir, out_dir, overwrite=overwrite)
        outputs.append(imported)
        supported_records.append(
            {
                **_cuagym_support_record(
                    source_task_dir,
                    metadata=metadata,
                    status="supported",
                    reason=None,
                ),
                "imported_path": str(imported),
            }
        )
        if limit > 0 and len(outputs) >= limit:
            break
    if support_report_out is not None:
        _write_cuagym_support_report(
            support_report_out,
            source_tasks_root=source_tasks_root,
            app_types=app_types,
            difficulties=difficulties,
            limit=limit,
            scanned=scanned,
            imported=outputs,
            supported=supported_records,
            unsupported=unsupported_records,
            filtered=filtered_records,
            skip_reasons=skip_reasons,
        )
    if not outputs:
        raise ValueError(
            "no supported CUA-Gym tasks matched filters "
            f"(scanned={scanned}, skipped={skipped})"
        )
    print(
        "CUA-Gym import summary: "
        f"scanned={scanned} imported={len(outputs)} skipped={skipped}",
        file=sys.stderr,
    )
    if skip_reasons:
        summary = "; ".join(
            f"{reason}={count}" for reason, count in skip_reasons.most_common(8)
        )
        print(f"CUA-Gym skip reasons: {summary}", file=sys.stderr)
    return outputs


def raw_cuagym_task_support(source_task_dir: Path) -> tuple[bool, str | None]:
    task_json_path = source_task_dir / "task.json"
    reward_path = source_task_dir / "reward.py"
    if not task_json_path.is_file() or not reward_path.is_file():
        return False, "missing task.json or reward.py"
    try:
        task_json = json.loads(task_json_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"invalid task.json: {exc}"
    evaluator = task_json.get("evaluator") or {}
    if not isinstance(evaluator, dict) or evaluator.get("type") != "python":
        return False, "non-python evaluator"
    if str(evaluator.get("url") or "") not in {"./reward.py", "reward.py"}:
        return False, "unsupported reward.py location"
    supported_postconfig, postconfig_reason = _cuagym_supported_postconfig(evaluator)
    if not supported_postconfig:
        return False, postconfig_reason or "unsupported evaluator postconfig"
    app_type = str(task_json.get("app_type") or "")
    if "mock" in app_type.lower():
        return False, "mock app type"
    setup_kinds = {
        str(step.get("type") or "")
        for step in (task_json.get("config") or [])
        if isinstance(step, dict)
    }
    unsupported_setup = setup_kinds - CUAGYM_SUPPORTED_SETUP_KINDS
    if unsupported_setup:
        return False, f"unsupported setup kinds: {sorted(unsupported_setup)}"
    unmapped_markers = _cuagym_unmapped_setup_markers(source_task_dir, task_json)
    if unmapped_markers:
        return False, f"unmapped setup launchers: {sorted(unmapped_markers)}"
    placeholders = _cuagym_placeholders_in_dir(source_task_dir)
    if placeholders:
        return False, f"unsupported placeholders: {sorted(placeholders)}"
    reward_source = reward_path.read_text(errors="replace")
    reward_compile_error = _reward_compile_error(
        reward_source,
        filename=reward_path,
    )
    if reward_compile_error is not None:
        return False, f"invalid reward.py python: {reward_compile_error}"
    reward_imports = _reward_import_roots(reward_source)
    unsupported_imports = reward_imports - CUAGYM_SUPPORTED_REWARD_IMPORTS
    if unsupported_imports:
        return False, f"unsupported reward imports: {sorted(unsupported_imports)}"
    return True, None


def _cuagym_task_metadata(source_task_dir: Path) -> dict[str, object]:
    task_json_path = source_task_dir / "task.json"
    metadata: dict[str, object] = {
        "task_id": source_task_dir.name,
        "app_type": "unknown",
        "difficulty": "unknown",
    }
    if not task_json_path.is_file():
        return metadata
    try:
        task_json = json.loads(task_json_path.read_text())
    except json.JSONDecodeError:
        return metadata
    if isinstance(task_json, dict):
        metadata["task_id"] = str(task_json.get("id") or source_task_dir.name)
        metadata["app_type"] = str(task_json.get("app_type") or "unknown")
        metadata["difficulty"] = str(task_json.get("difficulty") or "unknown")
    return metadata


def _cuagym_support_record(
    source_task_dir: Path,
    *,
    metadata: dict[str, object],
    status: str,
    reason: str | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "task_id": metadata.get("task_id") or source_task_dir.name,
        "source_dir": str(source_task_dir),
        "status": status,
        "app_type": metadata.get("app_type") or "unknown",
        "difficulty": metadata.get("difficulty") or "unknown",
    }
    if reason:
        record["reason"] = reason
        record["code"] = _support_issue_code(reason)
    return record


def _support_issue_code(reason: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")
    if normalized.startswith("unsupported-setup-kinds"):
        return "unsupported-setup-kind"
    if normalized.startswith("unsupported-evaluator-postconfig"):
        return "unsupported-evaluator-postconfig"
    if normalized.startswith("unsupported-reward-imports"):
        return "unsupported-reward-import"
    if normalized.startswith("unmapped-setup-launchers"):
        return "unmapped-setup-launcher"
    if normalized.startswith("unsupported-placeholders"):
        return "unsupported-placeholder"
    if normalized.startswith("invalid-reward-py-python"):
        return "invalid-reward-python"
    return normalized or "unsupported"


def _write_cuagym_support_report(
    path: Path,
    *,
    source_tasks_root: Path,
    app_types: set[str] | None,
    difficulties: set[str] | None,
    limit: int,
    scanned: int,
    imported: list[Path],
    supported: list[dict[str, object]],
    unsupported: list[dict[str, object]],
    filtered: list[dict[str, object]],
    skip_reasons: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": CUAGYM_SUPPORT_REPORT_SCHEMA,
        "source": {
            "type": "raw-cuagym-tasks-root",
            "path": str(source_tasks_root),
        },
        "filters": {
            "app_types": sorted(app_types) if app_types is not None else None,
            "difficulties": sorted(difficulties)
            if difficulties is not None
            else None,
            "limit": limit,
        },
        "counts": {
            "scanned": scanned,
            "imported": len(imported),
            "supported_seen": len(supported),
            "unsupported": len(unsupported),
            "filtered": len(filtered),
            "skipped": len(unsupported) + len(filtered),
        },
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "supported": supported,
        "unsupported": unsupported,
        "filtered": filtered,
        "plaintext_policy": (
            "report contains task identity, metadata, and support reasons only; "
            "raw instructions, setup files, reward.py, and screenshots are not copied"
        ),
    }
    path.write_text(json.dumps(report, indent=2) + "\n")


def _cuagym_task_toml(task_json: dict[str, object]) -> str:
    app_type = str(task_json.get("app_type") or "unknown")
    difficulty = str(task_json.get("difficulty") or "unknown")
    return (
        "[metadata]\n"
        'author_name = "CUA-Gym"\n'
        f"difficulty = {json.dumps(difficulty)}\n"
        'category = "desktop-automation"\n'
        f"tags = {json.dumps(['cuagym', app_type, difficulty])}\n\n"
        "[verifier]\n"
        "timeout_sec = 600\n"
        'user = "root"\n\n'
        "[agent]\n"
        "timeout_sec = 600\n\n"
        "[environment]\n"
        "cpus = 4\n"
        "memory_mb = 8192\n"
        "allow_internet = true\n"
    )


def _slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return slug or "task"


def _cuagym_placeholders_in_dir(path: Path) -> set[str]:
    placeholders: set[str] = set()
    for file_path in path.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in CUAGYM_TEXT_SUFFIXES:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        placeholders.update(CUAGYM_PLACEHOLDER.findall(text))
    return placeholders


def _cuagym_supported_postconfig(evaluator: dict[str, object]) -> tuple[bool, str | None]:
    postconfig = evaluator.get("postconfig") or []
    if not postconfig:
        return True, None
    if not isinstance(postconfig, list):
        return False, "malformed evaluator postconfig"
    kinds: set[str] = set()
    for step in postconfig:
        if isinstance(step, dict):
            step_data = cast(dict[str, Any], step)
            kinds.add(str(step_data.get("type") or ""))
    unsupported_kinds = kinds - CUAGYM_SUPPORTED_POSTCONFIG_KINDS
    if unsupported_kinds:
        return False, f"unsupported evaluator postconfig kinds: {sorted(unsupported_kinds)}"
    for step in postconfig:
        if not isinstance(step, dict):
            continue
        step_data = cast(dict[str, Any], step)
        kind = str(step_data.get("type") or "")
        if kind == "sleep":
            continue
        params = step_data.get("parameters") or {}
        command = params.get("command") if isinstance(params, dict) else None
        if isinstance(command, list) and (
            tuple(str(part) for part in command) in CUAGYM_PYAUTOGUI_SAVE_COMMANDS
        ):
            continue
        return False, "unsupported evaluator postconfig command"
    return True, None


def _cuagym_unmapped_setup_markers(
    source_task_dir: Path,
    task_json: dict[str, object],
) -> set[str]:
    config = task_json.get("config") or []
    if not isinstance(config, list):
        config = []
    text_parts = [json.dumps(config)]
    for step in config:
        if not isinstance(step, dict):
            continue
        step_data = cast(dict[str, Any], step)
        params = step_data.get("parameters") or {}
        if not isinstance(params, dict):
            continue
        params_data = cast(dict[str, Any], params)
        for item in params_data.get("files") or []:
            if not isinstance(item, dict):
                continue
            item_data = cast(dict[str, Any], item)
            url = str(item_data.get("url") or "")
            source = (
                source_task_dir / url[2:]
                if url.startswith("./")
                else source_task_dir / Path(url).name
            )
            if not source.is_file():
                continue
            try:
                text_parts.append(source.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    setup_text = "\n".join(text_parts)
    return {
        marker for marker in CUAGYM_UNMAPPED_SETUP_MARKERS if marker in setup_text
    }


def _reward_import_roots(source: str) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"<syntax-error>"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def _reward_compile_error(source: str, *, filename: Path) -> str | None:
    try:
        compile(source, str(filename), "exec")
    except SyntaxError as exc:
        return str(exc)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-repo", type=Path)
    parser.add_argument(
        "--cuagym-task-dir",
        type=Path,
        help="Raw extracted CUA-Gym task dir containing task.json/reward.py.",
    )
    parser.add_argument(
        "--cuagym-tasks-root",
        type=Path,
        help=(
            "Raw extracted CUA-Gym tasks/ root. Imports supported Python-reward "
            "tasks only; defaults to one task for smoke safety."
        ),
    )
    parser.add_argument(
        "--cuagym-limit",
        type=int,
        default=1,
        help="Maximum tasks to import from --cuagym-tasks-root. Use 0 for all matches.",
    )
    parser.add_argument(
        "--cuagym-app-type",
        action="append",
        dest="cuagym_app_types",
        help="Filter --cuagym-tasks-root by raw task app_type. Repeatable.",
    )
    parser.add_argument(
        "--cuagym-difficulty",
        action="append",
        dest="cuagym_difficulties",
        help="Filter --cuagym-tasks-root by raw task difficulty. Repeatable.",
    )
    parser.add_argument(
        "--support-report-out",
        type=Path,
        help=(
            "Write a scrubbed per-task support report for --cuagym-tasks-root "
            "imports."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(SMOKE_TASKS),
        dest="datasets",
        help="Dataset smoke slice to import. Repeatable; defaults to cuagym.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cuagym_task_dir is not None and args.cuagym_tasks_root is not None:
        parser.error("--cuagym-task-dir and --cuagym-tasks-root are mutually exclusive")

    if args.cuagym_task_dir is not None:
        print(
            import_cuagym_task(
                args.cuagym_task_dir,
                args.out_dir,
                overwrite=args.overwrite,
            )
        )
    elif args.cuagym_tasks_root is not None:
        outputs = import_cuagym_tasks(
            args.cuagym_tasks_root,
            args.out_dir,
            app_types=set(args.cuagym_app_types) if args.cuagym_app_types else None,
            difficulties=(
                set(args.cuagym_difficulties) if args.cuagym_difficulties else None
            ),
            limit=args.cuagym_limit,
            overwrite=args.overwrite,
            support_report_out=args.support_report_out,
        )
        for output in outputs:
            print(output)
    else:
        if args.upstream_repo is None:
            parser.error(
                "--upstream-repo is required unless a raw CUA-Gym import flag is set"
            )
        outputs = import_tasks(
            args.upstream_repo,
            args.out_dir,
            datasets=args.datasets or ["cuagym"],
            overwrite=args.overwrite,
        )
        for output in outputs:
            print(output)


if __name__ == "__main__":
    main()
