import json
import logging
import re
import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.task.acceptance_live import run_live_acceptance_checks
from benchflow.task.document import (
    TaskDocument,
    TaskDocumentParseError,
    render_normalized_task_md,
    render_task_md_from_legacy,
)
from benchflow.task.imports import import_task_config_toml
from benchflow.task.package import TaskPackage
from benchflow.task.paths import TaskPaths, local_script_strategy_files
from benchflow.task.verifier_document import (
    VERIFIER_DOCUMENT_FILENAME,
    VerifierDocument,
    VerifierStrategy,
)

logger = logging.getLogger(__name__)

LEGACY_REQUIRED_FILES = ["task.toml", "instruction.md"]
TASK_DOCUMENT_FILE = "task.md"
REQUIRED_DIRS = ["environment"]
OPTIONAL_FILES = ["environment/Dockerfile"]
OPTIONAL_DIRS = ["verifier", "oracle", "tests", "solution"]
TaskValidationLevel = Literal[
    "schema",
    "structural",
    "runtime-capability",
    "publication-grade",
    "acceptance",
    "acceptance-live",
]
_VALIDATION_LEVELS: set[str] = {
    "schema",
    "structural",
    "runtime-capability",
    "publication-grade",
    "acceptance",
    "acceptance-live",
}
_RUBRIC_SUFFIXES = {".md", ".toml", ".yaml", ".yml", ".json"}

# Placeholder marker written by init_task — must be replaced before the task
# is considered authored. Catching this in check_task prevents a freshly
# scaffolded task from being mistaken for a real benchmark (#360).
_PLACEHOLDER_MARKER = "[REPLACE:"


@dataclass(frozen=True)
class TaskMigrationResult:
    task_dir: Path
    task_md: Path
    removed_legacy: bool
    migrated_legacy_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskNormalizeResult:
    task_dir: Path
    task_md: Path
    normalized_text: str
    output_path: Path | None


def check_task(
    task_dir: Path,
    *,
    sandbox_type: str | None = None,
    validation_level: TaskValidationLevel = "structural",
    acceptance_live_report_output: Path | None = None,
    acceptance_live_write_report: bool = True,
) -> list[str]:
    """Validate a task directory structure. Returns list of issues (empty = valid)."""
    if validation_level not in _VALIDATION_LEVELS:
        raise ValueError(f"Unknown task validation level: {validation_level}")
    if (
        acceptance_live_report_output is not None
        and validation_level != "acceptance-live"
    ):
        return [
            "acceptance-live report output override requires --level acceptance-live"
        ]

    issues = []
    if not task_dir.is_dir():
        return [f"Not a directory: {task_dir}"]

    task_md = task_dir / TASK_DOCUMENT_FILE
    has_task_md = task_md.exists()

    if has_task_md:
        issues.extend(_check_task_document(task_md))
    else:
        for f in LEGACY_REQUIRED_FILES:
            if not (task_dir / f).exists():
                issues.append(f"Missing required file: {f}")

    # Validate task.toml
    # Note: [agent] and [agent].timeout_sec are optional at runtime
    # (AgentConfig defaults to timeout_sec=None → no wall-clock cap). We
    # only surface parse errors here so `bench tasks check` and
    # `bench eval create` agree on what a "valid" task looks like.
    # See #379.
    toml_path = task_dir / "task.toml"
    if toml_path.exists() and not has_task_md:
        try:
            with open(toml_path, "rb") as f:
                tomllib.load(f)
        except Exception as e:
            issues.append(f"task.toml parse error: {e}")

    # Check instruction.md is non-empty and has no placeholder markers
    instr = task_dir / "instruction.md"
    if instr.exists():
        if instr.stat().st_size == 0:
            issues.append("instruction.md is empty")
        elif _PLACEHOLDER_MARKER in instr.read_text():
            issues.append(
                f"instruction.md contains unreplaced placeholder "
                f"('{_PLACEHOLDER_MARKER} ...' markers) — replace them with "
                f"real task instructions"
            )

    if validation_level == "schema":
        return issues

    for d in REQUIRED_DIRS:
        if not (task_dir / d).is_dir():
            issues.append(f"Missing required directory: {d}/")

    # Check Dockerfile exists
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        issues.append("Missing environment/Dockerfile")

    paths = TaskPaths(task_dir)
    issues.extend(_check_compatibility_alias_drift(paths))

    # Check verifier code. Native tasks use verifier/; tests/ remains the
    # legacy alias for existing Harbor-style task packages.
    verifier_dir = paths.tests_dir
    verifier_label = _logical_dir_label(paths, kind="verifier")
    if verifier_dir.is_dir():
        if not any(verifier_dir.iterdir()):
            issues.append(f"{verifier_label}/ directory is empty")
        verifier_document = verifier_dir / VERIFIER_DOCUMENT_FILENAME
        if verifier_document.exists():
            try:
                VerifierDocument.from_path(verifier_document)
            except Exception as e:
                issues.append(f"{verifier_label}/verifier.md parse error: {e}")
        if not paths.has_verifier_entrypoint():
            issues.append(
                f"{verifier_label}/ has no runnable verifier entrypoint "
                "(expected test.sh or a valid verifier.md selected strategy)"
            )
        issues.extend(
            _check_unreplaced_verifier_placeholders(
                verifier_dir,
                verifier_label=verifier_label,
            )
        )
    else:
        issues.append(
            "Missing verifier/ directory (or legacy tests/; verifier needs "
            "test.sh or a verifier.md selected strategy)"
        )

    # Check CTRF output path consistency (ENG-153)
    test_sh = paths.test_path
    if test_sh.exists():
        issues.extend(_check_ctrf_path(test_sh))

    # Detect placeholder oracle scripts that have not been replaced (#360).
    for solve_sh in (
        paths.oracle_dir / "solve.sh",
        paths.legacy_solution_dir / "solve.sh",
    ):
        if solve_sh.exists() and _PLACEHOLDER_MARKER in solve_sh.read_text():
            label = "oracle" if solve_sh.parent == paths.oracle_dir else "solution"
            issues.append(
                f"{label}/solve.sh contains unreplaced placeholder — "
                "write a real oracle solution before running the task"
            )

    if validation_level == "runtime-capability" and sandbox_type is None:
        issues.append("runtime-capability validation requires --sandbox <backend>")

    if sandbox_type is not None:
        issues.extend(_check_runtime_capabilities(task_dir, sandbox_type=sandbox_type))

    if validation_level in {"publication-grade", "acceptance", "acceptance-live"}:
        issues.extend(_check_publication_grade(task_dir))
    if validation_level in {"acceptance", "acceptance-live"}:
        issues.extend(_check_acceptance_evidence(task_dir))
    if validation_level == "acceptance-live" and not issues:
        issues.extend(
            _check_live_acceptance_execution(
                task_dir,
                sandbox_type=sandbox_type,
                report_output=acceptance_live_report_output,
                write_report=acceptance_live_write_report,
            )
        )

    return issues


_CTRF_STANDARD_PATH = "/logs/verifier/ctrf.json"


def _check_unreplaced_verifier_placeholders(
    verifier_dir: Path,
    *,
    verifier_label: str,
) -> list[str]:
    """Find scaffold placeholders anywhere in the verifier package."""

    issues: list[str] = []
    for path in sorted(verifier_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if _PLACEHOLDER_MARKER not in text:
            continue
        relative = path.relative_to(verifier_dir).as_posix()
        issues.append(
            f"{verifier_label}/{relative} contains unreplaced placeholder — "
            "replace it with real verifier logic or rubric text"
        )
    return issues


def _check_ctrf_path(test_sh: Path) -> list[str]:
    """Warn when test.sh uses --ctrf with a non-standard output path."""
    try:
        text = test_sh.read_text()
    except OSError:
        return []
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    match = re.search(r"--ctrf[= ]([^\s\\]+)", uncommented)
    if not match:
        return []
    path_arg = match.group(1).strip('"').strip("'")
    if path_arg.startswith("$"):
        return []
    if path_arg != _CTRF_STANDARD_PATH:
        return [
            f"test.sh uses non-standard CTRF path '{path_arg}' "
            f"(expected '{_CTRF_STANDARD_PATH}')"
        ]
    return []


def _logical_dir_label(paths: TaskPaths, *, kind: Literal["verifier", "oracle"]) -> str:
    if kind == "verifier":
        return (
            TaskPaths.NATIVE_VERIFIER_DIRNAME
            if paths.uses_native_verifier_dir
            else TaskPaths.LEGACY_TESTS_DIRNAME
        )
    return (
        TaskPaths.NATIVE_ORACLE_DIRNAME
        if paths.uses_native_oracle_dir
        else TaskPaths.LEGACY_SOLUTION_DIRNAME
    )


_ALIAS_DRIFT_LOSS_PATHS = {
    "oracle|solution",
    "verifier|tests",
    "task.md|task.toml",
    "task.md|instruction.md",
}


def _check_compatibility_alias_drift(paths: TaskPaths) -> list[str]:
    """Fail structural checks when native and split aliases disagree."""

    if not paths.task_document_path.exists():
        return []
    issues = _check_partial_split_definition(paths)
    if not any(
        path.exists()
        for path in (
            paths.config_path,
            paths.instruction_path,
            paths.legacy_solution_dir,
            paths.legacy_tests_dir,
        )
    ):
        return issues

    try:
        from benchflow.task.export import build_compatibility_export_report

        report = build_compatibility_export_report(paths.task_dir)
    except Exception as exc:
        return [*issues, f"Compatibility alias drift check failed: {exc}"]

    issues.extend(
        f"Compatibility alias drift: {loss.path}: {loss.reason}"
        for loss in report.losses
        if loss.path in _ALIAS_DRIFT_LOSS_PATHS
    )
    return issues


def _check_partial_split_definition(paths: TaskPaths) -> list[str]:
    has_config = paths.config_path.exists()
    has_instruction = paths.instruction_path.exists()
    if has_config == has_instruction:
        return []
    missing = "instruction.md" if has_config else "task.toml"
    return [
        "Compatibility alias drift: task.md|legacy-split: native task.md "
        f"coexists with an incomplete legacy split definition (missing {missing})"
    ]


@dataclass(frozen=True)
class ScaffoldResult:
    """A freshly scaffolded task plus every file the scaffold wrote.

    ``files`` are relative POSIX paths under ``task_dir``, sorted, derived from
    what actually landed on disk — so a ``Created:`` summary can list the real
    scaffold instead of a hand-maintained subset that drifts from it.
    """

    task_dir: Path
    files: list[str]


def init_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_oracle: bool = False,
    task_format: Literal["legacy", "task-md"] = "task-md",
    *,
    no_solution: bool | None = None,
) -> Path:
    """Scaffold a new task directory; return its path.

    Thin wrapper over :func:`scaffold_task` for callers that only need the
    directory. Use :func:`scaffold_task` when you also need the exact list of
    files written (e.g. to print an accurate ``Created:`` summary).
    """
    return scaffold_task(
        name,
        parent_dir=parent_dir,
        no_pytest=no_pytest,
        no_oracle=no_oracle,
        task_format=task_format,
        no_solution=no_solution,
    ).task_dir


def scaffold_task(
    name: str,
    parent_dir: Path = Path("tasks"),
    no_pytest: bool = False,
    no_oracle: bool = False,
    task_format: Literal["legacy", "task-md"] = "task-md",
    *,
    no_solution: bool | None = None,
) -> ScaffoldResult:
    """Scaffold a new task directory with standard structure.

    Returns the directory together with every file written (see
    :class:`ScaffoldResult`).
    """
    if no_solution is not None:
        no_oracle = no_solution
    if task_format not in ("legacy", "task-md"):
        raise ValueError("task_format must be 'legacy' or 'task-md'")

    task_dir = parent_dir / name
    if task_dir.exists():
        raise FileExistsError(f"Task directory already exists: {task_dir}")

    task_dir.mkdir(parents=True)

    if task_format == "task-md":
        _write_task_md(task_dir, name)
    else:
        _write_legacy_task_files(task_dir, name)

    # environment/
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("""FROM ubuntu:24.04

# Install dependencies
RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Log directories
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
""")

    verifier_dirname = (
        TaskPaths.LEGACY_TESTS_DIRNAME
        if task_format == "legacy"
        else TaskPaths.NATIVE_VERIFIER_DIRNAME
    )
    oracle_dirname = (
        TaskPaths.LEGACY_SOLUTION_DIRNAME
        if task_format == "legacy"
        else TaskPaths.NATIVE_ORACLE_DIRNAME
    )

    # verifier/
    tests_dir = task_dir / verifier_dirname
    tests_dir.mkdir()
    # Verifier defaults to FAILURE (0.0) until the author replaces the
    # placeholder. A scaffold that auto-passes would silently inflate eval
    # results - see #360.
    (tests_dir / "test.sh").write_text("""#!/bin/bash
# Verifier script - write reward to /logs/verifier/reward.txt (float 0.0-1.0).
# Exit 0 after writing it; nonzero exit means verifier infrastructure failure.

# [REPLACE: write real verification logic here. The scaffold defaults to 0.0
# so an unedited task cannot accidentally count as a passing benchmark.]
echo "[REPLACE: write real verifier logic] - defaulting to failure" >&2
echo "0.0" > /logs/verifier/reward.txt
""")
    (tests_dir / "test.sh").chmod(0o755)

    if not no_pytest:
        # Fails by default until the author writes real assertions (#360).
        (
            tests_dir / "test_outputs.py"
        ).write_text("""\"\"\"Pytest-based verifier. Run by BenchFlow after agent completes.\"\"\"

import pytest


def test_placeholder():
    # [REPLACE: write real verification logic. Until then this test fails so
    # a scaffolded task cannot accidentally count as passing.]
    pytest.fail("[REPLACE: write real verifier assertions in this file]")
""")

    if task_format == "task-md":
        _write_task_md_verifier_package(task_dir, name)

    # oracle/ - placeholder MUST be replaced. The oracle solution should
    # cause the verifier in test.sh to write 1.0; the unedited scaffold
    # deliberately does not, so an init+check round-trip can't be mistaken
    # for a real benchmark (#360).
    if not no_oracle:
        sol_dir = task_dir / oracle_dirname
        sol_dir.mkdir()
        (sol_dir / "solve.sh").write_text(f"""#!/bin/bash
# Oracle solution - demonstrates the task is solvable.
# Used by: bench eval create --agent oracle --tasks-dir tasks/{name}

# [REPLACE: implement the oracle solution. It must satisfy the verifier in
# {verifier_dirname}/test.sh so that running solve.sh -> test.sh produces reward 1.0.]
echo "[REPLACE: implement oracle solution for {name}]" >&2
exit 1
""")
        (sol_dir / "solve.sh").chmod(0o755)

    written = sorted(
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    )
    return ScaffoldResult(task_dir=task_dir, files=written)


def migrate_task_to_task_md(
    task_dir: Path,
    *,
    overwrite: bool = False,
    remove_legacy: bool = False,
) -> TaskMigrationResult:
    """Convert a legacy task.toml + instruction.md pair into task.md.

    The migration is intentionally non-destructive by default: authors can
    inspect the generated document before deleting the legacy pair. Config
    equivalence is checked before writing so migration cannot silently lose
    supported task configuration.
    """

    task_dir = Path(task_dir)
    task_md = task_dir / TASK_DOCUMENT_FILE
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"

    if not task_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {task_dir}")
    missing = [path.name for path in (task_toml, instruction_md) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot migrate task without legacy files: " + ", ".join(missing)
        )
    if task_md.exists() and not overwrite:
        raise FileExistsError(
            f"{task_md} already exists; pass overwrite=True to replace it"
        )

    legacy_config = import_task_config_toml(
        task_toml.read_text(), source="legacy"
    ).config
    rendered = render_task_md_from_legacy(task_dir)
    document = TaskDocument.from_text(rendered, path=task_md)
    if document.config.model_dump() != legacy_config.model_dump():
        raise ValueError(
            "Generated task.md does not preserve task.toml config semantics"
        )
    if document.instruction != instruction_md.read_text().strip():
        raise ValueError(
            "Generated task.md does not preserve instruction.md prompt text"
        )

    task_md.write_text(rendered)
    migrated_legacy_dirs: tuple[str, ...] = ()
    if remove_legacy:
        task_toml.unlink()
        instruction_md.unlink()
        migrated_legacy_dirs = _promote_legacy_task_md_alias_dirs(task_dir)

    return TaskMigrationResult(
        task_dir=task_dir,
        task_md=task_md,
        removed_legacy=remove_legacy,
        migrated_legacy_dirs=migrated_legacy_dirs,
    )


def _promote_legacy_task_md_alias_dirs(task_dir: Path) -> tuple[str, ...]:
    """Adopt native directory names when removing split-format aliases."""

    migrated: list[str] = []
    for legacy_name, native_name in (
        (TaskPaths.LEGACY_TESTS_DIRNAME, TaskPaths.NATIVE_VERIFIER_DIRNAME),
        (TaskPaths.LEGACY_SOLUTION_DIRNAME, TaskPaths.NATIVE_ORACLE_DIRNAME),
    ):
        legacy_dir = task_dir / legacy_name
        native_dir = task_dir / native_name
        if not legacy_dir.is_dir() or native_dir.exists():
            continue
        legacy_dir.rename(native_dir)
        migrated.append(f"{legacy_name}/ -> {native_name}/")
    return tuple(migrated)


def normalize_task_md(
    task_dir: Path,
    *,
    output_path: Path | None = None,
    write: bool = False,
) -> TaskNormalizeResult:
    """Normalize a human-authored ``task.md`` into canonical machine form."""

    task_dir = Path(task_dir)
    task_md = task_dir / TASK_DOCUMENT_FILE
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {task_dir}")
    if not task_md.is_file():
        raise FileNotFoundError(f"Missing task.md: {task_md}")
    if output_path is not None and write:
        raise ValueError("Use either output_path or write=True, not both")

    normalized = render_normalized_task_md(task_md.read_text(), path=task_md)
    destination = task_md if write else output_path
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(normalized)

    return TaskNormalizeResult(
        task_dir=task_dir,
        task_md=task_md,
        normalized_text=normalized,
        output_path=destination,
    )


def _check_task_document(task_md: Path) -> list[str]:
    issues: list[str] = []
    try:
        document = TaskDocument.from_path(task_md)
    except TaskDocumentParseError as e:
        return [f"task.md parse error: {e}"]
    except Exception as e:
        return [f"task.md parse error: {e}"]

    text = task_md.read_text()
    if not document.instruction.strip():
        issues.append("task.md prompt is empty")
    if _PLACEHOLDER_MARKER in text:
        issues.append(
            "task.md contains unreplaced placeholder - replace "
            "task prompts, role prompts, and simulated-user notes"
        )
    return issues


def _check_runtime_capabilities(task_dir: Path, *, sandbox_type: str) -> list[str]:
    try:
        package = TaskPackage.from_task_dir(task_dir, sandbox=sandbox_type)
    except Exception as e:
        return [f"runtime capability parse error: {e}"]

    return [
        f"Unsupported runtime feature: {feature.format()}"
        for feature in package.runtime_issues
    ]


def _check_publication_grade(task_dir: Path) -> list[str]:
    """Static publication gate for the native task.md standard.

    This is intentionally narrower than acceptance testing. It proves the
    package has one native authoring entrypoint, an oracle proof location, and a
    verifier package/rubric contract. Running the oracle and calibration suite
    belongs to a later acceptance level.
    """

    issues: list[str] = []
    paths = TaskPaths(task_dir)

    if not paths.task_document_path.exists():
        issues.append(
            "publication-grade validation requires task.md as the authoritative "
            "entrypoint"
        )
    if paths.config_path.exists() or paths.instruction_path.exists():
        issues.append(
            "publication-grade tasks must not keep task.toml/instruction.md "
            "beside task.md; export split layouts through bench tasks export"
        )

    if not paths.oracle_dir.is_dir():
        issues.append("publication-grade validation requires native oracle/")
    elif not _has_regular_file(paths.oracle_dir):
        issues.append("publication-grade oracle/ must contain oracle evidence")
    if paths.legacy_solution_dir.exists():
        issues.append(
            "publication-grade tasks must use oracle/; solution/ is a "
            "compatibility export alias"
        )

    verifier_dir = paths.verifier_source_dir
    if not verifier_dir.is_dir():
        issues.append("publication-grade validation requires native verifier/")
        return issues
    if paths.legacy_tests_dir.exists():
        issues.append(
            "publication-grade tasks must use verifier/; tests/ is a "
            "compatibility export alias"
        )

    verifier_md = verifier_dir / VERIFIER_DOCUMENT_FILENAME
    if not verifier_md.exists():
        issues.append("publication-grade validation requires verifier/verifier.md")
        return issues

    try:
        document = VerifierDocument.from_path(verifier_md)
    except Exception as e:
        issues.append(f"publication-grade verifier/verifier.md parse error: {e}")
        return issues

    dimensions = document.rubric.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        issues.append(
            "publication-grade verifier/verifier.md must declare "
            "verifier.rubric.dimensions"
        )

    rubrics_dir = verifier_dir / "rubrics"
    if not rubrics_dir.is_dir() or not any(
        child.is_file() and child.suffix.lower() in _RUBRIC_SUFFIXES
        for child in rubrics_dir.rglob("*")
    ):
        issues.append(
            "publication-grade verifier packages require verifier/rubrics/ "
            "with at least one rubric file"
        )

    if not document.outputs.declared_reward_json:
        issues.append("publication-grade verifier outputs must declare reward_json")

    issues.extend(
        _check_selected_verifier_strategy_files(document, verifier_dir=verifier_dir)
    )
    return issues


def _check_acceptance_evidence(task_dir: Path) -> list[str]:
    """Static acceptance/calibration evidence gate for native task packages."""

    task_md = task_dir / TASK_DOCUMENT_FILE
    try:
        document = TaskDocument.from_path(task_md)
    except Exception as e:
        return [f"acceptance validation cannot parse task.md: {e}"]

    evidence = document.benchflow.get("evidence")
    if not isinstance(evidence, dict):
        return ["acceptance validation requires benchflow.evidence mapping"]
    evidence_mapping = cast(dict[str, object], evidence)

    issues: list[str] = []
    issues.extend(
        _check_oracle_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_verifier_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_review_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(
        _check_calibration_acceptance_evidence(evidence_mapping, task_dir=task_dir)
    )
    issues.extend(_check_listed_evidence_artifacts(evidence_mapping, task_dir=task_dir))
    issues.extend(_check_primary_evidence_pins(evidence_mapping))
    return issues


def _check_live_acceptance_execution(
    task_dir: Path,
    *,
    sandbox_type: str | None,
    report_output: Path | None,
    write_report: bool = True,
) -> list[str]:
    """Run declared live acceptance cases through the sandbox/verifier boundary."""

    if sandbox_type is None:
        return [
            "acceptance-live validation requires --sandbox <backend> to execute "
            "oracle and calibration checks"
        ]

    task_md = task_dir / TASK_DOCUMENT_FILE
    try:
        document = TaskDocument.from_path(task_md)
    except Exception as e:
        return [f"acceptance-live validation cannot parse task.md: {e}"]
    evidence = document.benchflow.get("evidence")
    if not isinstance(evidence, dict):
        return ["acceptance-live validation requires benchflow.evidence mapping"]
    return run_live_acceptance_checks(
        task_dir,
        sandbox_type=sandbox_type,
        evidence=cast(dict[str, object], evidence),
        report_output=report_output,
        write_report=write_report,
    )


def _check_oracle_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    oracle_runs = evidence.get("oracle_runs")
    if not isinstance(oracle_runs, dict):
        return ["acceptance validation requires benchflow.evidence.oracle_runs"]
    oracle_mapping = cast(dict[str, object], oracle_runs)
    issues: list[str] = []
    reward = oracle_mapping.get("required_reward")
    reward_value = _number_value(reward)
    if reward_value is None or reward_value < 0.99:
        issues.append(
            "acceptance oracle_runs.required_reward must be numeric and >= 0.99"
        )
    if not any(
        isinstance(oracle_mapping.get(key), str)
        and str(oracle_mapping.get(key)).strip()
        for key in ("last_job", "artifact")
    ):
        issues.append(
            "acceptance oracle_runs must include last_job or artifact evidence"
        )
    artifact = oracle_mapping.get("artifact")
    if artifact is not None:
        issues.extend(
            _check_declared_evidence_file(
                artifact,
                task_dir=task_dir,
                source="acceptance oracle_runs.artifact",
            )
        )
        if reward_value is not None and reward_value >= 0.99:
            issues.extend(
                _check_oracle_run_artifact(
                    artifact,
                    task_dir=task_dir,
                    required_reward=reward_value,
                )
            )
    return issues


def _check_oracle_run_artifact(
    value: object,
    *,
    task_dir: Path,
    required_reward: float,
) -> list[str]:
    source = "acceptance oracle_runs.artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    issues = []
    reward = _number_value(artifact.get("reward"))
    if reward is None:
        reward = _number_value(artifact.get("expected_reward"))
    if reward is None or reward < required_reward:
        issues.append(f"{source}.reward must be >= oracle_runs.required_reward")
    status = artifact.get("status")
    if status is not None and status != "passed":
        issues.append(f"{source}.status must be passed when present")
    return issues


def _check_verifier_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    verifier = evidence.get("verifier")
    if not isinstance(verifier, dict):
        return ["acceptance validation requires benchflow.evidence.verifier"]
    verifier_mapping = cast(dict[str, object], verifier)
    issues: list[str] = []
    reruns = verifier_mapping.get("reruns")
    if not isinstance(reruns, int) or isinstance(reruns, bool) or reruns < 3:
        issues.append("acceptance verifier.reruns must be an integer >= 3")
    flake_rate = verifier_mapping.get("flake_rate")
    flake_rate_value = _number_value(flake_rate)
    if flake_rate_value is None or not 0.0 <= flake_rate_value <= 0.05:
        issues.append("acceptance verifier.flake_rate must be numeric and <= 0.05")
    report = verifier_mapping.get("report")
    issues.extend(
        _check_declared_evidence_file(
            report,
            task_dir=task_dir,
            source="acceptance verifier.report",
        )
    )
    if (
        isinstance(reruns, int)
        and not isinstance(reruns, bool)
        and reruns >= 3
        and flake_rate_value is not None
        and 0.0 <= flake_rate_value <= 0.05
    ):
        issues.extend(
            _check_verifier_stability_report(
                report,
                task_dir=task_dir,
                declared_reruns=reruns,
                declared_flake_rate=flake_rate_value,
            )
        )
    return issues


def _check_verifier_stability_report(
    value: object,
    *,
    task_dir: Path,
    declared_reruns: int,
    declared_flake_rate: float,
) -> list[str]:
    source = "acceptance verifier.report"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    report = cast(dict[str, object], data)

    issues = []
    if report.get("kind") != "verifier-stability-report":
        issues.append(f"{source}.kind must be verifier-stability-report")

    report_reruns = report.get("reruns")
    if (
        not isinstance(report_reruns, int)
        or isinstance(report_reruns, bool)
        or report_reruns < declared_reruns
    ):
        issues.append(f"{source}.reruns must be an integer >= declared verifier.reruns")

    min_reward = _number_value(report.get("min_reward"))
    if min_reward is None or not 0.0 <= min_reward <= 1.0:
        issues.append(f"{source}.min_reward must be numeric within 0..1")

    report_flake_rate = _number_value(report.get("flake_rate"))
    if report_flake_rate is None or not 0.0 <= report_flake_rate <= declared_flake_rate:
        issues.append(
            f"{source}.flake_rate must be numeric and <= declared verifier.flake_rate"
        )

    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        issues.append(f"{source}.runs must be a non-empty list")
        return issues
    if (
        report_reruns is not None
        and isinstance(report_reruns, int)
        and len(runs) != report_reruns
    ):
        issues.append(f"{source}.runs length must match report reruns")
    if len(runs) < declared_reruns:
        issues.append(f"{source}.runs must include at least declared verifier.reruns")

    failures = 0
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            issues.append(f"{source}.runs[{index}] must be a mapping")
            failures += 1
            continue
        run_mapping = cast(dict[str, object], run)
        status = run_mapping.get("status")
        if status not in {"passed", "failed"}:
            issues.append(f"{source}.runs[{index}].status must be passed or failed")
            failures += 1
        reward = _number_value(run_mapping.get("reward"))
        if reward is None or not 0.0 <= reward <= 1.0:
            issues.append(f"{source}.runs[{index}].reward must be numeric within 0..1")
            failures += 1
        elif min_reward is not None and status == "passed" and reward < min_reward:
            issues.append(f"{source}.runs[{index}].reward is below min_reward")
            failures += 1
        elif status == "failed":
            failures += 1

    computed_flake_rate = failures / len(runs)
    if computed_flake_rate > declared_flake_rate:
        issues.append(
            f"{source}.runs imply flake_rate {computed_flake_rate:.3f}, "
            "above declared verifier.flake_rate"
        )
    if report_flake_rate is not None and report_flake_rate + 1e-9 < computed_flake_rate:
        issues.append(f"{source}.flake_rate is lower than failures in runs")
    return issues


def _check_review_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    review = evidence.get("review")
    if not isinstance(review, dict):
        return ["acceptance validation requires benchflow.evidence.review"]
    review_mapping = cast(dict[str, object], review)
    issues: list[str] = []
    for key in ("anti_cheat", "instruction_alignment"):
        if review_mapping.get(key) != "passed":
            issues.append(f"acceptance review.{key} must be passed")
    artifact = review_mapping.get("artifact")
    if artifact is None:
        issues.append("acceptance review.artifact must be declared")
    else:
        issues.extend(
            _check_review_artifact(
                artifact,
                task_dir=task_dir,
                declared_review=review_mapping,
            )
        )
    return issues


def _check_review_artifact(
    value: object,
    *,
    task_dir: Path,
    declared_review: dict[str, object],
) -> list[str]:
    source = "acceptance review.artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    issues = []
    if artifact.get("kind") not in {None, "acceptance-review"}:
        issues.append(f"{source}.kind must be acceptance-review when present")
    for key in ("anti_cheat", "instruction_alignment"):
        if artifact.get(key) != declared_review.get(key):
            issues.append(f"{source}.{key} must match benchflow.evidence.review.{key}")
    return issues


def _check_calibration_acceptance_evidence(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    calibration = evidence.get("calibration")
    if not isinstance(calibration, dict):
        return ["acceptance validation requires benchflow.evidence.calibration"]
    calibration_mapping = cast(dict[str, object], calibration)
    issues: list[str] = []
    no_op = calibration_mapping.get("no_op_reward_max")
    no_op_value = _number_value(no_op)
    if no_op_value is None or not 0.0 <= no_op_value <= 0.1:
        issues.append(
            "acceptance calibration.no_op_reward_max must be numeric and <= 0.1"
        )
    known_bad = calibration_mapping.get("known_bad_reward_max")
    known_bad_value = _number_value(known_bad)
    if known_bad_value is None or not 0.0 <= known_bad_value < 1.0:
        issues.append(
            "acceptance calibration.known_bad_reward_max must be numeric and < 1.0"
        )
    partial_range = calibration_mapping.get("partial_solution_range")
    partial_min: float | None = None
    partial_max: float | None = None
    if isinstance(partial_range, list) and len(partial_range) == 2:
        partial_min = _number_value(partial_range[0])
        partial_max = _number_value(partial_range[1])
    if (
        partial_min is None
        or partial_max is None
        or not (0.0 <= partial_min <= partial_max <= 1.0)
    ):
        issues.append(
            "acceptance calibration.partial_solution_range must be [min, max] "
            "within 0..1"
        )

    expected_rewards: list[float] = []
    examples = calibration_mapping.get("human_or_reference_examples")
    if not isinstance(examples, list) or not examples:
        issues.append(
            "acceptance calibration.human_or_reference_examples must be non-empty"
        )
    else:
        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                issues.append(
                    f"acceptance calibration.human_or_reference_examples[{index}] "
                    "must be a mapping"
                )
                continue
            example_mapping = cast(dict[str, object], example)
            expected = example_mapping.get("expected_reward")
            expected_value = _number_value(expected)
            if expected_value is None or not 0.0 <= expected_value <= 1.0:
                issues.append(
                    f"acceptance calibration.human_or_reference_examples[{index}] "
                    "expected_reward must be numeric within 0..1"
                )
            else:
                expected_rewards.append(expected_value)
            artifact = example_mapping.get("artifact")
            issues.extend(
                _check_declared_evidence_file(
                    artifact,
                    task_dir=task_dir,
                    source=(
                        "acceptance calibration.human_or_reference_examples"
                        f"[{index}].artifact"
                    ),
                )
            )
            if expected_value is not None and 0.0 <= expected_value <= 1.0:
                issues.extend(
                    _check_calibration_example_artifact(
                        artifact,
                        task_dir=task_dir,
                        expected_reward=expected_value,
                        index=index,
                    )
                )

    report = calibration_mapping.get("report")
    if report is None:
        issues.append("acceptance calibration.report must be declared")
    elif (
        no_op_value is not None
        and known_bad_value is not None
        and partial_min is not None
        and partial_max is not None
    ):
        issues.extend(
            _check_calibration_report(
                report,
                task_dir=task_dir,
                no_op_reward_max=no_op_value,
                known_bad_reward_max=known_bad_value,
                partial_min=partial_min,
                partial_max=partial_max,
                expected_rewards=expected_rewards,
            )
        )
    return issues


def _check_calibration_example_artifact(
    value: object,
    *,
    task_dir: Path,
    expected_reward: float,
    index: int,
) -> list[str]:
    source = f"acceptance calibration.human_or_reference_examples[{index}].artifact"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    artifact = cast(dict[str, object], data)
    reward = _number_value(artifact.get("expected_reward"))
    if reward is None:
        reward = _number_value(artifact.get("reward"))
    if reward is None:
        return [f"{source}.expected_reward must be numeric"]
    if abs(reward - expected_reward) > 1e-9:
        return [f"{source}.expected_reward must match declared expected_reward"]
    return []


def _check_calibration_report(
    value: object,
    *,
    task_dir: Path,
    no_op_reward_max: float,
    known_bad_reward_max: float,
    partial_min: float,
    partial_max: float,
    expected_rewards: list[float],
) -> list[str]:
    source = "acceptance calibration.report"
    data, issues = _load_declared_evidence_json(value, task_dir=task_dir, source=source)
    if issues:
        return issues
    if not isinstance(data, dict):
        return [f"{source} must be a JSON object"]
    report = cast(dict[str, object], data)

    issues = []
    if report.get("kind") != "calibration-report":
        issues.append(f"{source}.kind must be calibration-report")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        issues.append(f"{source}.cases must be a non-empty list")
        return issues

    has_no_op = False
    has_known_bad = False
    has_partial = False
    has_reference = False
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            issues.append(f"{source}.cases[{index}] must be a mapping")
            continue
        case_mapping = cast(dict[str, object], case)
        case_type = case_mapping.get("type")
        if case_type not in {"no-op", "known-bad", "partial", "reference"}:
            issues.append(
                f"{source}.cases[{index}].type must be no-op, known-bad, "
                "partial, or reference"
            )
            continue
        reward = _number_value(case_mapping.get("reward"))
        if reward is None or not 0.0 <= reward <= 1.0:
            issues.append(f"{source}.cases[{index}].reward must be numeric within 0..1")
            continue
        if case_type == "no-op":
            has_no_op = True
            if reward > no_op_reward_max:
                issues.append(f"{source}.cases[{index}] exceeds no_op_reward_max")
        elif case_type == "known-bad":
            has_known_bad = True
            if reward > known_bad_reward_max:
                issues.append(f"{source}.cases[{index}] exceeds known_bad_reward_max")
        elif case_type == "partial":
            has_partial = True
            if not partial_min <= reward <= partial_max:
                issues.append(f"{source}.cases[{index}] is outside partial range")
        elif case_type == "reference":
            has_reference = True
            if expected_rewards and all(
                abs(reward - expected) > 1e-9 for expected in expected_rewards
            ):
                issues.append(
                    f"{source}.cases[{index}] does not match a declared "
                    "reference expected_reward"
                )

    required_cases = [
        ("no-op", has_no_op),
        ("known-bad", has_known_bad),
        ("partial", has_partial),
        ("reference", has_reference),
    ]
    for case_type, present in required_cases:
        if not present:
            issues.append(f"{source}.cases must include a {case_type} case")
    return issues


def _check_listed_evidence_artifacts(
    evidence: dict[str, object],
    *,
    task_dir: Path,
) -> list[str]:
    issues: list[str] = []
    for list_key in ("trajectories", "artifacts"):
        items = evidence.get(list_key)
        if items is None:
            continue
        if not isinstance(items, list):
            issues.append(f"acceptance evidence.{list_key} must be a list")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                issues.append(
                    f"acceptance evidence.{list_key}[{index}] must be a mapping"
                )
                continue
            item_mapping = cast(dict[str, object], item)
            expected_sha256 = item_mapping.get("sha256")
            if not isinstance(expected_sha256, str) or not expected_sha256.strip():
                issues.append(
                    f"acceptance evidence.{list_key}[{index}].sha256 must be declared"
                )
            issues.extend(
                _check_declared_evidence_file(
                    item_mapping.get("path"),
                    task_dir=task_dir,
                    source=f"acceptance evidence.{list_key}[{index}].path",
                    expected_sha256=expected_sha256,
                )
            )
    return issues


def _check_primary_evidence_pins(evidence: dict[str, object]) -> list[str]:
    pinned_paths = _pinned_evidence_paths(evidence)
    required = _primary_evidence_paths(evidence)
    return [
        f"{source} must be listed in benchflow.evidence.artifacts or "
        "benchflow.evidence.trajectories with sha256"
        for source, path in required
        if path not in pinned_paths
    ]


def _pinned_evidence_paths(evidence: dict[str, object]) -> set[str]:
    pinned_paths: set[str] = set()
    for list_key in ("trajectories", "artifacts"):
        items = evidence.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_mapping = cast(dict[str, object], item)
            if (
                not isinstance(item_mapping.get("sha256"), str)
                or not str(item_mapping.get("sha256")).strip()
            ):
                continue
            path = _declared_evidence_path_key(item_mapping.get("path"))
            if path is not None:
                pinned_paths.add(path)
    return pinned_paths


def _primary_evidence_paths(evidence: dict[str, object]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []

    oracle_runs = evidence.get("oracle_runs")
    if isinstance(oracle_runs, dict):
        oracle_mapping = cast(dict[str, object], oracle_runs)
        _append_declared_evidence_path(
            paths,
            "acceptance oracle_runs.artifact",
            oracle_mapping.get("artifact"),
        )

    verifier = evidence.get("verifier")
    if isinstance(verifier, dict):
        verifier_mapping = cast(dict[str, object], verifier)
        _append_declared_evidence_path(
            paths,
            "acceptance verifier.report",
            verifier_mapping.get("report"),
        )

    review = evidence.get("review")
    if isinstance(review, dict):
        review_mapping = cast(dict[str, object], review)
        _append_declared_evidence_path(
            paths,
            "acceptance review.artifact",
            review_mapping.get("artifact"),
        )

    calibration = evidence.get("calibration")
    if isinstance(calibration, dict):
        calibration_mapping = cast(dict[str, object], calibration)
        _append_declared_evidence_path(
            paths,
            "acceptance calibration.report",
            calibration_mapping.get("report"),
        )
        examples = calibration_mapping.get("human_or_reference_examples")
        if isinstance(examples, list):
            for index, example in enumerate(examples):
                if not isinstance(example, dict):
                    continue
                example_mapping = cast(dict[str, object], example)
                _append_declared_evidence_path(
                    paths,
                    (
                        "acceptance calibration.human_or_reference_examples"
                        f"[{index}].artifact"
                    ),
                    example_mapping.get("artifact"),
                )

    return paths


def _append_declared_evidence_path(
    paths: list[tuple[str, str]],
    source: str,
    value: object,
) -> None:
    path = _declared_evidence_path_key(value)
    if path is not None:
        paths.append((source, path))


def _declared_evidence_path_key(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = _safe_relative_path(value)
    return path.as_posix() if path is not None else None


def _check_declared_evidence_file(
    value: object,
    *,
    task_dir: Path,
    source: str,
    expected_sha256: object | None = None,
) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{source} must be a non-empty relative path"]
    path = _safe_relative_path(value)
    if path is None:
        return [f"{source} must be a safe relative path"]
    local_path = task_dir / path
    if not local_path.is_file():
        return [f"{source} references missing file: {value}"]
    if expected_sha256 is None:
        return []
    if not isinstance(expected_sha256, str) or not expected_sha256.strip():
        return [f"{source} sha256 must be a non-empty string when declared"]
    digest = sha256(local_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        return [f"{source} sha256 mismatch for {value}"]
    return []


def _load_declared_evidence_json(
    value: object,
    *,
    task_dir: Path,
    source: str,
) -> tuple[object | None, list[str]]:
    issues = _check_declared_evidence_file(value, task_dir=task_dir, source=source)
    if issues:
        return None, issues
    assert isinstance(value, str)
    path = _safe_relative_path(value)
    assert path is not None
    try:
        return json.loads((task_dir / path).read_text()), []
    except json.JSONDecodeError as exc:
        return None, [f"{source} is not valid JSON: {exc}"]
    except OSError as exc:
        return None, [f"{source} cannot be read: {exc}"]


def _has_regular_file(root: Path) -> bool:
    return any(path.is_file() for path in root.rglob("*"))


def _number_value(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _check_selected_verifier_strategy_files(
    document: VerifierDocument,
    *,
    verifier_dir: Path,
) -> list[str]:
    strategy = document.selected_strategy
    prefix = (
        f"publication-grade selected verifier strategy "
        f"'{strategy.name}' ({strategy.type})"
    )

    if strategy.type == "script":
        issues = []
        scripts = local_script_strategy_files(
            strategy.command,
            verifier_dir=verifier_dir,
        )
        if not scripts:
            return [
                f"{prefix} must reference a packaged verifier artifact "
                "such as ./test.sh or verify.py"
            ]
        for script in scripts:
            if not script.is_file():
                issues.append(f"{prefix} references missing script: {script}")
        return issues
    elif strategy.type == "llm-judge":
        issues = []
        if not _strategy_file_exists(strategy.rubric_path, verifier_dir=verifier_dir):
            issues.append(f"{prefix} references missing rubric: {strategy.rubric_path}")
        if strategy.context_file is not None and not _strategy_file_exists(
            strategy.context_file,
            verifier_dir=verifier_dir,
        ):
            issues.append(
                f"{prefix} references missing context_file: {strategy.context_file}"
            )
        return issues
    elif strategy.type == "reward-kit":
        return _check_reward_kit_strategy_files(strategy, verifier_dir=verifier_dir)

    return []


def _check_reward_kit_strategy_files(
    strategy: VerifierStrategy,
    *,
    verifier_dir: Path,
) -> list[str]:
    issues: list[str] = []
    prefix = (
        f"publication-grade selected verifier strategy "
        f"'{strategy.name}' ({strategy.type})"
    )
    root = _safe_relative_path(strategy.root_path)
    if root is None:
        return [f"{prefix} must declare a safe relative root"]

    root_dir = verifier_dir / root
    if not root_dir.is_dir():
        issues.append(f"{prefix} references missing root: {strategy.root_path}")

    entrypoint = _safe_relative_path(strategy.entrypoint or "reward.py")
    if entrypoint is None:
        issues.append(f"{prefix} must declare a safe relative entrypoint")
    elif not (root_dir / entrypoint).is_file():
        issues.append(
            f"{prefix} references missing entrypoint: "
            f"{strategy.entrypoint or 'reward.py'}"
        )

    if strategy.criteria_path is not None:
        criteria = _safe_relative_path(strategy.criteria_path)
        if criteria is None or not (verifier_dir / criteria).is_file():
            issues.append(
                f"{prefix} references missing criteria: {strategy.criteria_path}"
            )
        else:
            try:
                criteria_aggregate_policy_from_rubric(verifier_dir / criteria)
            except ValueError as e:
                issues.append(f"{prefix} has invalid criteria: {e}")
    return issues


def _strategy_file_exists(value: str | None, *, verifier_dir: Path) -> bool:
    relative = _safe_relative_path(value)
    return relative is not None and (verifier_dir / relative).is_file()


def _safe_relative_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _write_legacy_task_files(task_dir: Path, name: str) -> None:
    (task_dir / "task.toml").write_text("""version = "1.0"

[metadata]
author_name = ""
difficulty = "medium"
category = "capability"
tags = []

[agent]
timeout_sec = 300

[verifier]
timeout_sec = 120

[environment]
cpus = 1
memory_mb = 2048
""")

    (task_dir / "instruction.md").write_text(f"""# {name}

[REPLACE: one-sentence summary of what the agent must do.]

## Goal

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific — list files to produce, commands to run, or behaviours to verify.]

## Success criteria

[REPLACE: list the conditions the verifier in tests/test.sh checks for.]
""")


def _write_task_md_verifier_package(task_dir: Path, name: str) -> None:
    verifier_dir = task_dir / TaskPaths.NATIVE_VERIFIER_DIRNAME
    rubrics_dir = verifier_dir / "rubrics"
    rubrics_dir.mkdir()
    (verifier_dir / VERIFIER_DOCUMENT_FILENAME).write_text(f"""---
document_version: "0.3"
verifier:
  name: {name}
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success: {{weight: 1.0, source: deterministic}}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## verifier intent

[REPLACE: describe what the verifier measures and which task outputs it reads.]
""")
    (rubrics_dir / "verifier.md").write_text(f"""# {name} Verifier Rubric

- `task_success`: [REPLACE: define the exact observable success condition the
  verifier checks.]
""")
    (rubrics_dir / "verifier.toml").write_text("""version = "0.1"

[[criteria]]
id = "task_success"
description = "[REPLACE: define the exact observable success condition the verifier checks.]"
weight = 1.0

[scoring]
method = "weighted_mean"
""")


def _write_task_md(task_dir: Path, name: str) -> None:
    (task_dir / TASK_DOCUMENT_FILE).write_text(f"""---
schema_version: "1.3"
metadata:
  author_name: ""
  difficulty: medium
  category: capability
  tags: []
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
---
# {name}

## prompt

[REPLACE: describe the goal, constraints, and expected outputs.
Be specific - list files to produce, commands to run, or behaviours to verify.]
""")
