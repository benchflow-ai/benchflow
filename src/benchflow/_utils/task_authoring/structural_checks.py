"""Read-only structural-check cluster for task packages.

Inspects an on-disk task directory and reports issues without mutating it:
verifier placeholders, CTRF output paths, compatibility alias drift, task.md /
runtime-capability parsing, the static publication-grade gate, and verifier
scripts that install pytest plugins where the plugin guard refuses them.
"""

import re
import shlex
import tomllib
from pathlib import Path
from typing import Literal

from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.task.document import TaskDocument
from benchflow.task.package import TaskPackage
from benchflow.task.paths import TaskPaths, local_script_strategy_files
from benchflow.task.verifier_document import (
    VERIFIER_DOCUMENT_FILENAME,
    VerifierDocument,
    VerifierStrategy,
)

from ._evidence_paths import (
    _PLACEHOLDER_MARKER,
    _has_regular_file,
    _safe_relative_path,
)

_RUBRIC_SUFFIXES = {".md", ".toml", ".yaml", ".yml", ".json"}

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


# Workspace plugin installs. The verifier's pytest plugin guard
# (sandbox/_pytest_plugin_guard.py) refuses plugin code from any place the
# agent could write: its workspace, its home, /tmp, /var/tmp, /logs and
# /testbed (lockdown._blocked_verifier_path_prefixes). A verifier script that
# builds its own Python environment in one of them and installs a pytest
# plugin into it therefore has that plugin refused every time, and because
# test.sh installed it after the agent stopped, every trial ends unscored
# (Terminal-Bench 2's mailman: ``uv venv .tb`` then ``uv pip install
# pytest-json-ctrf``; SkillsBench's powerlifting-coef-calc: ``uv init`` and
# ``uv add pytest-json-ctrf`` in its WORKDIR /root, the workspace; the same
# holds for an explicit /root/.venv or ~/.venv there).
_SCRIPT_COMMENT_RE = re.compile(r"^\s*#(?!!).*$|\s#.*$", re.MULTILINE)
_SCRIPT_COMMAND_SPLIT_RE = re.compile(r"\n|;|&&|\|\||\|")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PYTEST_PLUGIN_RE = re.compile(r"^pytest[-_][A-Za-z0-9][A-Za-z0-9_.-]*")
# Absolute places the guard refuses in the agent's sandbox whatever the image:
# BenchFlow's runtime directories and the sandbox user's home (/home/agent).
_GUARD_REFUSED_PREFIXES = ("/tmp", "/var/tmp", "/logs", "/testbed", "/home")
# The verifier runs as root, so its $HOME and ~ are /root.
_VERIFIER_HOME = "/root"
# Options of the environment-making commands that take a value.
_VENV_VALUE_OPTIONS = frozenset(
    {"-p", "--python", "--prompt", "--index-url", "--seed", "--relocatable"}
)
_TARGET_OPTIONS = ("--target", "-t", "--prefix")


def _resolve_script_path(path: str) -> str | None:
    """``path`` as the verifier sees it: ``$HOME``/``~`` expanded, or None when unknown."""
    for home in ("${HOME}", "$HOME", "~"):
        if path == home or path.startswith(home + "/"):
            return _VERIFIER_HOME + path[len(home) :]
    if path.startswith("$"):
        return None  # another variable: unknown
    return path


def _guard_refuses_path(path: str, workspace: str | None = None) -> bool:
    """A path the plugin guard refuses code from.

    Relative paths are the working directory (the image WORKDIR, normally the
    workspace); absolute ones are refused under the workspace (when known),
    the agent's home, /tmp, /var/tmp, /logs or /testbed.
    """
    resolved = _resolve_script_path(path) if path else None
    if not resolved:
        return False
    if not resolved.startswith("/"):
        return True
    prefixes = (*_GUARD_REFUSED_PREFIXES, *((workspace,) if workspace else ()))
    return any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )


def _environment_of(python_or_bin: str) -> str:
    """The environment a ``.../bin/python`` or ``.../bin/pip`` belongs to."""
    head, sep, _tail = python_or_bin.rpartition("/bin/")
    return head if sep else python_or_bin


_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_FROM_RE = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE | re.MULTILINE)
# Images whose WORKDIR is /, where BenchFlow makes /root the workspace.
_PLAIN_BASE_RE = re.compile(
    r"^(?:docker\.io/)?(?:library/)?(?:ubuntu|debian|python|alpine|node|golang|rust|"
    r"fedora|centos|rockylinux|almalinux|amazonlinux|buildpack-deps)(?:[:@]|$)"
)


def task_workspace(task_dir: Path) -> str | None:
    """The agent's workspace for a task, or None when it cannot be told statically.

    ``sandbox.workdir`` when the task declares one; else the Dockerfile's
    last WORKDIR (BenchFlow makes /root the workspace when it is /, as it is
    for plain distribution images). A prebuilt image's WORKDIR is unknown.
    """
    try:
        from benchflow.task.runtime_view import TaskRuntimeView

        workdir = getattr(
            getattr(TaskRuntimeView.from_task_dir(task_dir).config, "sandbox", None),
            "workdir",
            None,
        )
    except Exception:
        workdir = None
    if isinstance(workdir, str) and workdir.strip().startswith("/"):
        return workdir.strip().rstrip("/") or None
    dockerfile = task_dir / "environment" / "Dockerfile"
    try:
        text = dockerfile.read_text(errors="replace")
    except OSError:
        return None
    stages = _FROM_RE.split(text)
    if len(stages) < 3:
        return None
    base, last = stages[-2], stages[-1]
    current: str | None = None
    for match in _WORKDIR_RE.finditer(last):
        value = match.group(1).strip().strip("'\"")
        if value.startswith("$"):
            return None
        current = (
            value if value.startswith("/") else f"{current or ''}/{value}"
        ).rstrip("/") or "/"
    if current is None:
        return _VERIFIER_HOME if _PLAIN_BASE_RE.match(base) else None
    return _VERIFIER_HOME if current == "/" else current


def _command_words(command: str) -> list[str]:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    while words and (_ASSIGNMENT_RE.match(words[0]) or words[0] in {"env", "exec"}):
        words = words[1:]
    return words


def _first_argument(words: list[str]) -> str | None:
    skip = False
    for word in words:
        if skip:
            skip = False
        elif word.startswith("-"):
            skip = "=" not in word and word in _VENV_VALUE_OPTIONS
        else:
            return word
    return None


def _option_value(words: list[str], options: tuple[str, ...]) -> tuple[str, str] | None:
    """The first ``--opt VALUE`` or ``--opt=VALUE`` among ``options``: (option, value)."""
    for i, word in enumerate(words):
        for option in options:
            if word == option and i + 1 < len(words):
                return option, words[i + 1]
            if option.startswith("--") and word.startswith(option + "="):
                return option, word.split("=", 1)[1]
    return None


def _install_target(
    words: list[str], workspace: str | None = None, made: set[str] | None = None
) -> str | None:
    found = _option_value(words, _TARGET_OPTIONS)
    if found is not None:
        option, target = found
        if _guard_refuses_path(target, workspace):
            return f"--target {target}" if option == "-t" else f"{option} {target}"
    found = _option_value(words, ("--python", "-p"))
    if found is not None:
        env = _environment_of(found[1])
        if _guard_refuses_path(env, workspace) and env.rstrip("/") not in (made or ()):
            return f"{found[0]} {found[1]}"
    return None


def workspace_plugin_installs(
    text: str, *, workspace: str | None = None
) -> tuple[list[str], list[str]]:
    """What in a verifier script puts pytest plugins where the plugin guard refuses them.

    Returns the commands that make, activate or fill a Python environment in
    a place the agent could write (the working directory, ``workspace`` when
    known, the agent's home, /tmp, /var/tmp, /logs, /testbed; ``$HOME`` and
    ``~`` are the verifier's, /root), and the pytest plugins the script
    installs with anything but ``uvx`` (uv's cache, which BenchFlow moves to
    a directory the guard trusts). A script needs both to be flagged.
    """
    text = _SCRIPT_COMMENT_RE.sub("", text.replace("\\\n", " "))
    places: list[str] = []
    plugins: list[str] = []
    made: set[str] = set()  # environments a recorded command made

    def refused(path: str) -> bool:
        return _guard_refuses_path(path, workspace)

    def made_here(path: str) -> None:
        made.add(path.rstrip("/"))

    def new_place(env: str) -> bool:
        """An environment the script uses but no recorded command made."""
        return refused(env) and env.rstrip("/") not in made

    for command in _SCRIPT_COMMAND_SPLIT_RE.split(text):
        words = _command_words(command.strip())
        if not words:
            continue
        program = words[0].rsplit("/", 1)[-1]
        if program in {"source", "."} and len(words) > 1:
            if words[1].endswith("/bin/activate") and new_place(
                _environment_of(words[1])
            ):
                places.append(f"{program} {words[1]}")
        elif program == "uv" and len(words) > 1:
            subcommand, rest = words[1], words[2:]
            if subcommand == "venv":
                target = _first_argument(rest) or ".venv"
                if refused(target):
                    places.append(f"uv venv {target}")
                    made_here(target)
            elif subcommand in {"init", "add", "sync"}:
                project = _option_value(rest, ("--project", "--directory"))
                if project is None or refused(project[1]):
                    places.append(f"uv {subcommand}")
                if subcommand == "add":
                    plugins += [w for w in rest if _PYTEST_PLUGIN_RE.match(w)]
            elif subcommand == "pip" and rest[:1] == ["install"]:
                plugins += [w for w in rest[1:] if _PYTEST_PLUGIN_RE.match(w)]
                if target := _install_target(rest, workspace, made):
                    places.append(f"uv pip install {target}")
        elif program.startswith("pip") or (
            program.startswith("python") and words[1:3] == ["-m", "pip"]
        ):
            rest = words[1:] if program.startswith("pip") else words[3:]
            if rest[:1] == ["install"]:
                plugins += [w for w in rest[1:] if _PYTEST_PLUGIN_RE.match(w)]
                if target := _install_target(rest, workspace, made):
                    places.append(f"pip install {target}")
                if "/" in words[0] and refused(_environment_of(words[0])):
                    places.append(words[0])
        elif program.startswith("python") and words[1:3] == ["-m", "venv"]:
            target = _first_argument(words[3:])
            if target and refused(target):
                places.append(f"python -m venv {target}")
                made_here(target)
        elif program == "virtualenv":
            target = _first_argument(words[1:])
            if target and refused(target):
                places.append(f"virtualenv {target}")
                made_here(target)
    return list(dict.fromkeys(places)), list(dict.fromkeys(plugins))


# ``unset PYTEST_ADDOPTS`` or a fresh ``PYTEST_ADDOPTS=...``: drops the guard.
_CLEARS_PYTEST_ADDOPTS_RE = re.compile(
    r"\bunset\s+(?:-v\s+)?PYTEST_ADDOPTS\b"
    r"|(?<![\w$])PYTEST_ADDOPTS=(?![^\n;&|]*\$\{?PYTEST_ADDOPTS\b)"
)


def _check_workspace_plugin_installs(
    verifier_dir: Path, *, label: str, workspace: str | None = None
) -> list[str]:
    """Warn when a verifier script installs pytest plugins where the agent could write."""
    warnings = []
    for script in sorted(verifier_dir.glob("*.sh")):
        try:
            text = script.read_text(errors="replace")
        except OSError:
            continue
        places, plugins = workspace_plugin_installs(text, workspace=workspace)
        if not (places and plugins):
            continue
        if _CLEARS_PYTEST_ADDOPTS_RE.search(_SCRIPT_COMMENT_RE.sub("", text)):
            # The guard never runs, so nothing is refused: the plugin loads
            # unchecked from an environment the agent's workspace can seed.
            effect = (
                ", and clears PYTEST_ADDOPTS, so BenchFlow's pytest plugin "
                "guard never checks that code, which comes from a directory "
                "the agent could have prepared"
            )
        else:
            effect = (
                ": the verifier's pytest plugin guard refuses plugin code "
                "there, so every trial ends unscored"
            )
        where = f"; the workspace is {workspace}" if workspace else ""
        warnings.append(
            f"{label}/{script.name} installs pytest plugins "
            f"({', '.join(plugins)}) into a Python environment the agent could "
            f"write ({', '.join(places)}{where}){effect}. Install them with uvx "
            "into uv's default cache, which BenchFlow moves to a directory the "
            f"guard trusts (for example `uvx --with {plugins[0]} pytest ...`), "
            "or into the image"
        )
    return warnings


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


def task_document_parse_error(task_md: Path) -> str | None:
    """Return a human-readable parse error if ``task_md`` fails to parse, else None.

    The single source of truth for "does this task.md parse?" — used both by
    ``check_task``'s structural validation and by eval task-discovery, so a
    genuinely malformed task.md (a typo that would otherwise make the dir
    silently vanish from a batch, #3) can be told apart from a dir that simply
    isn't a task.
    """
    try:
        TaskDocument.from_path(task_md)
    except Exception as e:
        return f"task.md parse error: {e}"
    return None


def task_config_parse_error(task_toml: Path) -> str | None:
    """Return a human-readable parse error if ``task_toml`` fails to parse, else None.

    The legacy-format counterpart of :func:`task_document_parse_error`, shared
    the same way by ``check_task`` and eval task-discovery. Only TOML syntax
    is checked, as ``bench tasks check`` always has (#379).
    """
    try:
        with open(task_toml, "rb") as f:
            tomllib.load(f)
    except Exception as e:
        return f"task.toml parse error: {e}"
    return None


def _check_task_document(task_md: Path) -> list[str]:
    issues: list[str] = []
    parse_error = task_document_parse_error(task_md)
    if parse_error is not None:
        return [parse_error]
    document = TaskDocument.from_path(task_md)

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
        issues.append(
            "publication-grade validation requires native verifier/ — `bench tasks "
            "migrate` produces the native layout but not the verifier package; "
            "author verifier/ (verifier.md + rubrics/) per "
            "docs/task-authoring-task-md.md, then re-run check"
        )
        return issues
    if paths.legacy_tests_dir.exists():
        issues.append(
            "publication-grade tasks must use verifier/; tests/ is a "
            "compatibility export alias"
        )

    verifier_md = verifier_dir / VERIFIER_DOCUMENT_FILENAME
    if not verifier_md.exists():
        issues.append(
            "publication-grade validation requires verifier/verifier.md — author "
            "the verifier strategy document (see docs/task-authoring-task-md.md); "
            "`bench tasks migrate` does not generate it"
        )
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
