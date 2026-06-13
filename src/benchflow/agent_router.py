"""Benchmark adoption router — ``bench agent create | run | verify``.

This module is the real logic behind the ``bench agent`` router subcommands
that adopt an upstream benchmark into a BenchFlow benchmark. It sits downstream
of every environment framework: a benchmark is *routed* into the repo here,
while ``bench eval create`` *runs* the resulting tasks.

Three cohesive subcommands, registered onto the existing ``agent`` Typer group
by :func:`register_agent_router` (``cli/main.py`` only wires the call):

``create``  Deterministic scaffold of ``benchmarks/<name>/`` matching the
            reference layout (``benchmarks/programbench/``) and the contract in
            ``benchmarks/CONVERT.md``. Fail-closed: refuses to overwrite an
            existing benchmark and validates the slug.
``run``     Driver that assembles the adoption context (source + CONVERT.md +
            adoption skills) and launches the host ``codex`` CLI to drive the
            conversion toward a ``benchmarks/<name>/`` pull request. Context
            assembly and launch-command construction are pure functions so they
            are unit-testable with a fake exec layer; the live ``codex`` run is
            a manual-validation step.
``verify``  Closes the adopt->verify loop. Runs the parity gate for an adopted
            benchmark and emits a confidence verdict. The gate is *parity only*:
            a faithful translation must reproduce the original's behavior on
            identical inputs (including any reward-hackability the original
            has — parity never "improves" or sanitizes the source). On a
            divergence it prints a draft GitHub issue body for human support
            instead of filing anything automatically.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

# The parity gate (parsers, scoring, verdict) lives in agent_router_parity to
# keep this module focused on create/run/verify CLI wiring. Re-exported here
# (see __all__) so the public API is unchanged, e.g.
# ``from benchflow.agent_router import build_verify_report``.
from benchflow.agent_router_parity import (  # noqa: F401
    DEFAULT_REWARD_TOLERANCE,
    ConversionParity,
    CriterionComparison,
    RewardDistributionParity,
    RewardSample,
    Verdict,
    VerifyReport,
    build_verify_report,
    confidence_line,
    extract_criterion_comparisons,
    extract_reward_samples,
    render_divergence_issue,
)
from benchflow.agent_router_scaffold import (
    BENCHMARK_YAML_TEMPLATE,
    CONVERTER_TEMPLATE,
    JOB_YAML_TEMPLATE,
    MAIN_TEMPLATE,
    PARITY_TEST_TEMPLATE,
    README_TEMPLATE,
    RUNNER_TEMPLATE,
)
from benchflow.environment_adapter_parity import validate_environment_adapter_loop_state

# ── Errors ────────────────────────────────────────────────────────────


class InvalidBenchmarkName(ValueError):
    """Raised when a benchmark slug fails validation."""


class BenchmarkExistsError(FileExistsError):
    """Raised when scaffolding would overwrite an existing benchmark."""


class BenchmarkNotFound(FileNotFoundError):
    """Raised when an operation targets a benchmark that was never adopted."""


class ParityExperimentMissing(FileNotFoundError):
    """Raised when an adopted benchmark has no parity_experiment.json yet."""


class AdoptionReportInvalid(ValueError):
    """Raised when an optional adoption_report.json sidecar is malformed."""


class AdoptionLoopStateInvalid(ValueError):
    """Raised when an optional loop_state.json sidecar is malformed."""


class CodexLaunchError(RuntimeError):
    """Raised when the host codex CLI cannot be launched (e.g. no credentials)."""


# ── Paths ─────────────────────────────────────────────────────────────


def default_repo_root() -> Path:
    """Repo root inferred from this module's location (src/benchflow/...)."""
    return Path(__file__).resolve().parents[2]


def default_benchmarks_dir() -> Path:
    """The ``benchmarks/`` directory in the repo."""
    return default_repo_root() / "benchmarks"


def _default_codex_auth_file() -> Path:
    return Path.home() / ".codex" / "auth.json"


# ── Name validation ───────────────────────────────────────────────────

# Lowercase slug: starts with a letter, single internal hyphens, no traversal.
# Validated with ``fullmatch`` (not ``match`` + ``$``): in Python ``re`` the
# ``$`` anchor also matches just *before* a trailing newline, so an anchored
# ``match`` would accept ``"good\n"``. ``fullmatch`` requires the whole string
# to be consumed, rejecting any trailing newline outright.
_SLUG_RE = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")
_MAX_NAME_LEN = 64


def validate_benchmark_name(name: str) -> str:
    """Return ``name`` if it is a safe benchmark slug, else raise.

    Rejects uppercase, underscores, whitespace, path separators, leading
    digits/hyphens, trailing/consecutive hyphens and over-long names. The path
    separator check is the security floor: it keeps ``create``/``verify`` from
    being steered outside ``benchmarks/``.
    """
    if not name:
        raise InvalidBenchmarkName("benchmark name is empty")
    if len(name) > _MAX_NAME_LEN:
        raise InvalidBenchmarkName(
            f"benchmark name too long (>{_MAX_NAME_LEN} chars): {name!r}"
        )
    if not _SLUG_RE.fullmatch(name):
        raise InvalidBenchmarkName(
            f"invalid benchmark name {name!r}: use a lowercase slug like "
            "'my-bench' (letters/digits, single internal hyphens, leading letter)"
        )
    return name


def _title_from_slug(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("-"))


def _module_suffix(name: str) -> str:
    return name.replace("-", "_")


def derive_name_from_source(source: str) -> str:
    """Derive a benchmark slug from a source repo/path basename."""
    base = source.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[: -len(".git")]
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    return validate_benchmark_name(slug)


# ── create: deterministic scaffold ────────────────────────────────────


def _scaffold_parity_experiment(name: str) -> str:
    """Templated, empty parity_experiment.json (status ``template``).

    The schema is what ``bench agent verify`` reads: per-criterion verdict pairs
    (the deterministic conversion-faithfulness floor) and reward-distribution
    samples (the statistical legacy-vs-converted layer).
    """
    data = {
        "experiment": "side-by-side-parity",
        "benchmark": name,
        "status": "template",
        "judge_model": "",
        "conversion_parity": {
            "description": (
                "Per-criterion verdicts of original vs converted on identical "
                "inputs. Each task lists criteria_results with original_verdict, "
                "adapted_verdict, agreement."
            ),
            "tasks": [],
        },
        "reward_distribution_parity": {
            "description": (
                "Legacy vs converted pass-rate / reward deltas at agent scale "
                "(the statistical parity layer)."
            ),
            "samples": [],
        },
    }
    return json.dumps(data, indent=2) + "\n"


def build_scaffold_files(name: str) -> dict[str, str]:
    """Return ``{relative_path: contents}`` for a benchmark scaffold (pure)."""
    name = validate_benchmark_name(name)
    title = _title_from_slug(name)

    def render(template: str) -> str:
        return template.replace("{{NAME}}", name).replace("{{TITLE}}", title)

    return {
        "__init__.py": "",
        "benchflow.py": render(CONVERTER_TEMPLATE),
        "main.py": render(MAIN_TEMPLATE),
        "parity_test.py": render(PARITY_TEST_TEMPLATE),
        f"run_{_module_suffix(name)}.py": render(RUNNER_TEMPLATE),
        f"{name}.yaml": render(JOB_YAML_TEMPLATE),
        "benchmark.yaml": render(BENCHMARK_YAML_TEMPLATE),
        "parity_experiment.json": _scaffold_parity_experiment(name),
        "README.md": render(README_TEMPLATE),
    }


def create_benchmark(name: str, benchmarks_root: Path) -> tuple[Path, list[str]]:
    """Scaffold ``benchmarks/<name>/``. Fail-closed if it already exists.

    Returns ``(target_dir, sorted_relative_paths_written)``.
    """
    name = validate_benchmark_name(name)
    target = Path(benchmarks_root) / name
    if target.exists():
        raise BenchmarkExistsError(
            f"benchmark already exists: {target} (refusing to overwrite)"
        )

    files = build_scaffold_files(name)
    target.mkdir(parents=True)
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return target, sorted(files)


# ── run: adoption driver ──────────────────────────────────────────────


@dataclass(frozen=True)
class AdoptionSkill:
    """A reference to adoption guidance assembled into the codex context."""

    name: str
    reference: str


def collect_adoption_skills() -> list[AdoptionSkill]:
    """The adoption skills surfaced to the driver (static references, no I/O)."""
    return [
        AdoptionSkill("conversion-guide", "benchmarks/CONVERT.md"),
        AdoptionSkill(
            "reference-benchmark", "benchmarks/programbench/ (worked example)"
        ),
        AdoptionSkill(
            "parity-harness", "parity_test.py + parity_experiment.json (verify gate)"
        ),
    ]


def load_convert_guide(repo_root: Path) -> str:
    """Read ``benchmarks/CONVERT.md`` (fail-closed if missing)."""
    path = Path(repo_root) / "benchmarks" / "CONVERT.md"
    if not path.exists():
        raise FileNotFoundError(f"conversion guide not found: {path}")
    return path.read_text()


def assemble_adoption_context(
    source: str,
    name: str,
    *,
    convert_guide: str,
    skills: Sequence[AdoptionSkill],
) -> str:
    """Assemble the full codex prompt for adopting ``source`` (pure function).

    Includes the source, the target ``benchmarks/<name>/`` path, the adoption
    skills, and the embedded ``benchmarks/CONVERT.md`` guide.
    """
    skill_lines = "\n".join(f"- {s.name}: {s.reference}" for s in skills)
    return "\n".join(
        [
            f"# Benchmark adoption: {name}",
            "",
            "Adopt the source benchmark below into a BenchFlow benchmark by",
            "following the conversion guide. Produce the converter, parity tests,",
            "metadata, and task directories, then open a pull request.",
            "",
            f"Source benchmark: {source}",
            f"Target directory: benchmarks/{name}/",
            "",
            "## Adoption skills",
            skill_lines,
            "",
            "## Conversion guide (benchmarks/CONVERT.md)",
            "",
            convert_guide,
            "",
            "## Definition of done",
            f"- benchmarks/{name}/ has benchflow.py, parity_test.py,",
            f"  parity_experiment.json, benchmark.yaml, run_{_module_suffix(name)}.py,",
            "  README.md",
            f"- `bench agent verify {name}` reports parity-confirmed",
        ]
    )


def build_codex_launch_command(
    prompt: str,
    *,
    workdir: Path | str,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
) -> list[str]:
    """Construct the host ``codex exec`` argv for the adoption run (pure)."""
    command = [
        codex_bin,
        "exec",
        "--cd",
        str(workdir),
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
    ]
    if model:
        command += ["--model", model]
    command.append(prompt)
    return command


@dataclass(frozen=True)
class AdoptionLaunch:
    """Everything needed to launch (or dry-run) the codex adoption driver."""

    command: list[str]
    cwd: str
    prompt: str


def prepare_adoption_launch(
    source: str,
    name: str,
    *,
    repo_root: Path,
    convert_guide: str | None = None,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
) -> AdoptionLaunch:
    """Assemble context + build the codex command (no exec, no credentials)."""
    name = validate_benchmark_name(name)
    if convert_guide is None:
        convert_guide = load_convert_guide(repo_root)
    skills = collect_adoption_skills()
    prompt = assemble_adoption_context(
        source, name, convert_guide=convert_guide, skills=skills
    )
    command = build_codex_launch_command(
        prompt, workdir=repo_root, codex_bin=codex_bin, model=model, sandbox=sandbox
    )
    return AdoptionLaunch(command=command, cwd=str(repo_root), prompt=prompt)


def has_codex_credentials(env: Mapping[str, str], auth_file: Path) -> bool:
    """True if codex can authenticate via an API key or a login auth file."""
    if env.get("OPENAI_API_KEY") or env.get("CODEX_API_KEY"):
        return True
    return Path(auth_file).exists()


# exec_fn(command, *, cwd, env) -> exit code. Injected so tests use a fake.
ExecFn = Callable[..., int]


def _subprocess_exec(command: list[str], *, cwd: str, env: Mapping[str, str]) -> int:
    import subprocess

    return subprocess.run(command, cwd=cwd, env=dict(env)).returncode


def run_agent_adoption(
    source: str,
    name: str,
    *,
    repo_root: Path,
    exec_fn: ExecFn,
    env: Mapping[str, str] | None = None,
    auth_file: Path | None = None,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
) -> int:
    """Launch the host codex CLI to drive the adoption. Fail-closed on creds.

    The live codex invocation is the manual-validation step; here it is reached
    through ``exec_fn`` so the plumbing is unit-proven with a fake exec layer.
    """
    resolved_env = os.environ if env is None else env
    resolved_auth = _default_codex_auth_file() if auth_file is None else auth_file
    name = validate_benchmark_name(name)
    # Fail closed on missing credentials before assembling any context.
    if not has_codex_credentials(resolved_env, resolved_auth):
        raise CodexLaunchError(
            "codex needs credentials to launch: set OPENAI_API_KEY (or "
            "CODEX_API_KEY), or run `codex login` to create ~/.codex/auth.json"
        )
    launch = prepare_adoption_launch(
        source,
        name,
        repo_root=repo_root,
        codex_bin=codex_bin,
        model=model,
        sandbox=sandbox,
    )
    return exec_fn(launch.command, cwd=launch.cwd, env=resolved_env)


# ── verify: parity gate (parsers/scoring re-exported from _parity above) ──


def load_parity_experiment(benchmarks_root: Path, name: str) -> Any:
    """Load an adopted benchmark's parity_experiment.json (fail-closed).

    Returns whatever JSON the file holds — usually a mapping, but some adopted
    benchmarks ship a top-level array. The parity extractors tolerate either,
    so callers must not assume a mapping.
    """
    name = validate_benchmark_name(name)
    benchmark_dir = Path(benchmarks_root) / name
    if not benchmark_dir.exists():
        raise BenchmarkNotFound(
            f"benchmark not adopted: {benchmark_dir} — run "
            f"`bench agent create {name}` first"
        )
    parity_file = benchmark_dir / "parity_experiment.json"
    if not parity_file.exists():
        raise ParityExperimentMissing(
            f"no parity_experiment.json in {benchmark_dir} — run parity_test.py first"
        )
    try:
        return json.loads(parity_file.read_text())
    except json.JSONDecodeError as exc:
        raise ParityExperimentMissing(
            f"parity_experiment.json in {benchmark_dir} is not valid JSON: {exc}"
        ) from exc


def load_adoption_report(
    benchmarks_root: Path, name: str
) -> tuple[Path, Mapping[str, Any]] | None:
    """Load an optional scrubbed adoption_report.json sidecar.

    Older benchmark adoptions do not have this file, so missing means
    ``None``. If the file exists it must be parseable JSON object evidence; a
    malformed sidecar should not be silently ignored by adapter-adoption loops.
    """

    name = validate_benchmark_name(name)
    report_file = Path(benchmarks_root) / name / "adoption_report.json"
    if not report_file.exists():
        return None
    try:
        payload = json.loads(report_file.read_text())
    except json.JSONDecodeError as exc:
        raise AdoptionReportInvalid(
            f"adoption_report.json in {report_file.parent} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AdoptionReportInvalid(
            f"adoption_report.json in {report_file.parent} must be a JSON object"
        )
    return report_file, payload


def load_adoption_loop_state(
    benchmarks_root: Path, name: str
) -> tuple[Path, Mapping[str, Any]] | None:
    """Load an optional durable loop_state.json sidecar."""

    name = validate_benchmark_name(name)
    state_file = Path(benchmarks_root) / name / "loop_state.json"
    if not state_file.exists():
        return None
    try:
        payload = json.loads(state_file.read_text())
    except json.JSONDecodeError as exc:
        raise AdoptionLoopStateInvalid(
            f"loop_state.json in {state_file.parent} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AdoptionLoopStateInvalid(
            f"loop_state.json in {state_file.parent} must be a JSON object"
        )
    return state_file, payload


def build_adoption_loop_gate(
    report: VerifyReport,
    *,
    adoption_report: tuple[Path, Mapping[str, Any]] | None,
    loop_state: tuple[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the durable adapter-adoption sidecar as a scale gate.

    ``bench agent verify`` can confirm reward/criterion parity from
    ``parity_experiment.json`` alone. The 0.7 environment-adapter loop needs a
    stronger checkpoint before scale: architecture planes, artifact/timing
    shape, and cleanup evidence must be represented in the scrubbed
    ``adoption_report.json`` sidecar, and the resumable controller state must
    be represented in ``loop_state.json``.
    """

    issues: list[str] = []
    path: Path | None = None
    sidecar: Mapping[str, Any] = {}
    if adoption_report is None:
        issues.append("missing adoption_report.json sidecar")
    else:
        path, sidecar = adoption_report
        _check_adoption_report_shape(report, sidecar, issues)

    loop_path: Path | None = None
    loop_payload: Mapping[str, Any] = {}
    if loop_state is None:
        issues.append("missing loop_state.json sidecar")
    else:
        loop_path, loop_payload = loop_state
        _check_loop_state_shape(report, loop_payload, issues)

    status = "scale-ready" if not issues else "not-ready"
    payload: dict[str, Any] = {
        "schema": "benchflow.adapter-adoption-loop-gate.v1",
        "status": status,
        "passed": not issues,
        "scale_ready": not issues,
        "benchmark": report.name,
        "issues": issues,
        "required_criteria": [
            "trace-completeness",
            "artifact-shape",
            "timing-recorded",
            "cleanup",
        ],
    }
    if path is not None:
        payload["path"] = str(path)
    if sidecar:
        payload["planes"] = dict(_mapping(sidecar.get("planes")))
        payload["parity"] = dict(_mapping(sidecar.get("parity")))
    if loop_path is not None:
        payload["loop_state"] = {
            "path": str(loop_path),
            "status": loop_payload.get("status"),
            "roles": list(loop_payload.get("roles") or []),
            "queue": list(loop_payload.get("queue") or []),
        }
    return payload


def _check_loop_state_shape(
    report: VerifyReport, state: Mapping[str, Any], issues: list[str]
) -> None:
    for issue in validate_environment_adapter_loop_state(state):
        issues.append(f"loop_state.json {issue}")
    if state.get("benchmark") not in (None, report.name):
        issues.append("loop_state.json benchmark does not match verify target")
    if state.get("status") not in {"review-ready", "scale-ready"}:
        issues.append("loop_state.json status is not review-ready or scale-ready")


def _check_adoption_report_shape(
    report: VerifyReport, sidecar: Mapping[str, Any], issues: list[str]
) -> None:
    if sidecar.get("schema") != "benchflow.environment-adapter-adoption-report.v1":
        issues.append("adoption_report.json has unexpected schema")
    if sidecar.get("status") != report.verdict:
        issues.append("adoption_report.json status does not match parity verdict")
    if sidecar.get("benchmark") not in (None, report.name):
        issues.append("adoption_report.json benchmark does not match verify target")
    if report.verdict != "parity-confirmed":
        issues.append("parity verdict is not confirmed")

    planes = _mapping(sidecar.get("planes"))
    for key in (
        "sandbox_provider",
        "environment_adapter",
        "agent_adapter",
        "benchmark_adapter",
    ):
        if not planes.get(key):
            issues.append(f"adoption_report.json planes missing {key}")
    if planes.get("sandbox_provider") == "cua" and planes.get(
        "sandbox_provider_mode"
    ) not in {"local", "cloud-probed"}:
        issues.append("Cua adoption_report.json must declare sandbox_provider_mode")

    parity = _mapping(sidecar.get("parity"))
    compared = _number(parity.get("criteria_compared"))
    agreed = _number(parity.get("criteria_agreed"))
    if compared is None or compared <= 0:
        issues.append("adoption_report.json parity has no compared criteria")
    elif agreed != compared:
        issues.append("adoption_report.json parity criteria did not all agree")
    reward_delta = _number(parity.get("reward_delta"))
    if reward_delta is None:
        issues.append("adoption_report.json parity missing reward_delta")
    elif reward_delta > report.tolerance:
        issues.append("adoption_report.json reward_delta exceeds tolerance")

    comparison_ids = {item.criterion_id for item in report.conversion.comparisons}
    for criterion in (
        "trace-completeness",
        "artifact-shape",
        "timing-recorded",
        "cleanup",
    ):
        if criterion not in comparison_ids:
            issues.append(f"parity_experiment.json missing {criterion} criterion")

    artifacts = _artifact_index_by_id(sidecar.get("artifact_index"))
    requirements = _mapping(sidecar.get("artifact_requirements"))
    _check_original_runner_artifact(artifacts.get("original-runner"), issues)
    _check_benchflow_result_artifact(artifacts.get("benchflow-result"), issues)
    _check_benchflow_trace_artifact(
        artifacts.get("benchflow-agent-artifact"),
        issues,
        require_screenshot=not requirements,
    )
    _check_benchflow_eval_artifact(artifacts.get("benchflow-eval-summary"), issues)

    if requirements and requirements.get("ok") is not True:
        issues.append("adoption_report.json artifact requirements did not pass")

    unsupported = sidecar.get("unsupported_reports")
    if isinstance(unsupported, Sequence) and not isinstance(unsupported, (str, bytes)):
        for index, item in enumerate(unsupported):
            report_item = _mapping(item)
            if not (
                report_item.get("task_id")
                or report_item.get("task")
                or report_item.get("id")
            ):
                issues.append(f"unsupported report {index} missing task identity")
            if not (
                report_item.get("reason")
                or report_item.get("issue")
                or report_item.get("code")
            ):
                issues.append(f"unsupported report {index} missing reason")


def _artifact_index_by_id(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and item.get("id"):
            out[str(item["id"])] = item
    return out


def _check_original_runner_artifact(
    artifact: Mapping[str, Any] | None, issues: list[str]
) -> None:
    if artifact is None:
        issues.append("adoption_report.json missing original-runner artifact")
        return
    if _number(artifact.get("score")) is None:
        issues.append("original-runner artifact missing score")
    if not _positive_number(artifact.get("trace_steps")):
        issues.append("original-runner artifact missing trace steps")
    if artifact.get("error_present") is True:
        issues.append("original-runner artifact has an error")


def _check_benchflow_result_artifact(
    artifact: Mapping[str, Any] | None, issues: list[str]
) -> None:
    if artifact is None:
        issues.append("adoption_report.json missing benchflow-result artifact")
        return
    if _number(artifact.get("reward")) is None:
        issues.append("benchflow-result artifact missing reward")
    if not _positive_number(artifact.get("trajectory_steps")):
        issues.append("benchflow-result artifact missing trajectory steps")
    if not _positive_number(artifact.get("tool_calls")):
        issues.append("benchflow-result artifact missing tool calls")
    if artifact.get("timing_present") is not True:
        issues.append("benchflow-result artifact missing timing")
    if artifact.get("error_present") is True:
        issues.append("benchflow-result artifact has an error")


def _check_benchflow_trace_artifact(
    artifact: Mapping[str, Any] | None,
    issues: list[str],
    *,
    require_screenshot: bool,
) -> None:
    if artifact is None:
        issues.append("adoption_report.json missing benchflow-agent-artifact")
        return
    if not _positive_number(artifact.get("trace_steps")):
        issues.append("benchflow-agent-artifact missing trace steps")
    if require_screenshot and not _positive_number(
        artifact.get("screenshots_b64_count")
    ):
        issues.append("benchflow-agent-artifact missing screenshots")


def _check_benchflow_eval_artifact(
    artifact: Mapping[str, Any] | None, issues: list[str]
) -> None:
    if artifact is None:
        issues.append("adoption_report.json missing benchflow-eval-summary artifact")
        return
    if artifact.get("present") is not True:
        issues.append("benchflow-eval-summary artifact is not present")
    if artifact.get("status") != "completed" or artifact.get("ok") is not True:
        issues.append("benchflow-eval-summary did not complete cleanly")
    if not _positive_number(artifact.get("total")):
        issues.append("benchflow-eval-summary missing total")
    if _number(artifact.get("errored")) not in (0.0, None):
        issues.append("benchflow-eval-summary has errored tasks")
    if _number(artifact.get("verifier_errored")) not in (0.0, None):
        issues.append("benchflow-eval-summary has verifier errors")
    if artifact.get("timing_recorded") is not True:
        issues.append("benchflow-eval-summary missing timing")
    if artifact.get("summary_path_present") is not True:
        issues.append("benchflow-eval-summary missing summary path")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _positive_number(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0


def roundtrip_conformance_status(
    task_dir: Path,
    *,
    report_fn: Callable[..., Any] | None = None,
) -> tuple[str, list[str]]:
    """Surface the structural round-trip conformance harness for one task.

    Thin wiring to ``benchflow.task.build_harbor_roundtrip_conformance_report``
    (the existing parity utility). Returns ``(status, mismatch_reasons)``.

    ``verify`` runs this only when ``--roundtrip-task`` names a task directory;
    by default the verify gate scores the recorded ``parity_experiment.json``
    and this stays an opt-in structural check (the harness needs a concrete
    task tree, which the benchmark-level verdict does not require).
    """
    if report_fn is None:
        from benchflow.task import build_harbor_roundtrip_conformance_report

        report_fn = build_harbor_roundtrip_conformance_report
    report = report_fn(task_dir)
    reasons = [m.reason for m in getattr(report, "mismatches", [])]
    return report.status, reasons


def verify_report_payload(
    report: VerifyReport,
    *,
    roundtrip: Mapping[str, Any] | None = None,
    adoption_report: tuple[Path, Mapping[str, Any]] | None = None,
    adoption_loop: Mapping[str, Any] | None = None,
    issue_out: Path | None = None,
    issue_draft: str | None = None,
) -> dict[str, Any]:
    """Machine-readable verdict for adapter-adoption loop controllers."""

    reward = None
    if report.reward is not None:
        reward = {
            "samples": [
                {
                    "task_id": sample.task_id,
                    "legacy_reward": sample.legacy_reward,
                    "converted_reward": sample.converted_reward,
                    "delta": sample.delta,
                    "exceeds_tolerance": sample.delta > report.reward.tolerance,
                }
                for sample in report.reward.samples
            ],
            "sample_count": len(report.reward.samples),
            "max_abs_delta": report.reward.max_abs_delta,
            "tolerance": report.reward.tolerance,
            "within_tolerance": report.reward.within_tolerance,
        }

    payload: dict[str, Any] = {
        "status": report.verdict,
        "verdict": report.verdict,
        "passed": report.passed,
        "benchmark": report.name,
        "confidence": confidence_line(report),
        "conversion": {
            "compared": report.conversion.compared,
            "agreed": report.conversion.agreed,
            "agreement_rate": report.conversion.agreement_rate,
            "all_agree": report.conversion.all_agree,
            "disagreements": [
                {
                    "task_id": item.task_id,
                    "criterion_id": item.criterion_id,
                    "original_verdict": item.original_verdict,
                    "adapted_verdict": item.adapted_verdict,
                }
                for item in report.conversion.disagreements
            ],
        },
        "reward": reward,
    }
    if roundtrip is not None:
        payload["roundtrip"] = dict(roundtrip)
    if adoption_report is not None:
        path, report_payload = adoption_report
        payload["adoption_report"] = {
            "path": str(path),
            **dict(report_payload),
        }
    if adoption_loop is not None:
        payload["adoption_loop"] = dict(adoption_loop)
        if adoption_loop.get("passed") is False:
            payload["passed"] = False
    if issue_out is not None:
        payload["issue_out"] = str(issue_out)
    if issue_draft is not None:
        payload["issue_draft"] = issue_draft
    return payload


def verify_error_payload(name: str | None, reason: str) -> dict[str, Any]:
    """Machine-readable error payload for ``bench agent verify --json``."""

    payload: dict[str, Any] = {
        "status": "error",
        "verdict": "error",
        "passed": False,
        "reason": reason,
    }
    if name:
        payload["benchmark"] = name
    return payload


# ── CLI registration (thin; real logic lives above) ───────────────────


def register_agent_router(agent_app: typer.Typer) -> None:
    """Attach ``create`` / ``run`` / ``verify`` to the ``agent`` Typer group."""
    import shlex

    from rich.console import Console

    console = Console()

    @agent_app.command("create")
    def agent_create(
        name: Annotated[
            str, typer.Argument(help="Benchmark slug (lowercase, hyphenated)")
        ],
        benchmarks_dir: Annotated[
            Path | None,
            typer.Option("--benchmarks-dir", help="Target benchmarks/ directory"),
        ] = None,
    ) -> None:
        """Scaffold benchmarks/<name>/ for a new benchmark adoption."""
        root = benchmarks_dir or default_benchmarks_dir()
        try:
            target, written = create_benchmark(name, root)
        except (InvalidBenchmarkName, BenchmarkExistsError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Scaffolded[/green] {target}")
        for rel in written:
            console.print(f"  {rel}")

    @agent_app.command("run")
    def agent_run(
        source: Annotated[
            str, typer.Argument(help="Source benchmark repo or local path")
        ],
        name: Annotated[
            str | None,
            typer.Option("--name", help="Benchmark slug (default: from source)"),
        ] = None,
        model: Annotated[
            str | None, typer.Option("--model", help="Model for the codex driver")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Print the launch command, do not run"),
        ] = False,
        codex_bin: Annotated[
            str, typer.Option("--codex-bin", help="Host codex binary")
        ] = "codex",
    ) -> None:
        """Drive the CONVERT.md workflow by launching the host codex CLI."""
        repo_root = default_repo_root()
        try:
            resolved = name or derive_name_from_source(source)
            if dry_run:
                launch = prepare_adoption_launch(
                    source,
                    resolved,
                    repo_root=repo_root,
                    codex_bin=codex_bin,
                    model=model,
                )
                console.print(" ".join(shlex.quote(c) for c in launch.command))
                return
            code = run_agent_adoption(
                source,
                resolved,
                repo_root=repo_root,
                exec_fn=_subprocess_exec,
                codex_bin=codex_bin,
                model=model,
            )
        except (InvalidBenchmarkName, CodexLaunchError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        raise typer.Exit(code)

    @agent_app.command("verify")
    def agent_verify(
        name: Annotated[str, typer.Argument(help="Adopted benchmark slug")],
        benchmarks_dir: Annotated[
            Path | None,
            typer.Option("--benchmarks-dir", help="Target benchmarks/ directory"),
        ] = None,
        tolerance: Annotated[
            float,
            typer.Option("--tolerance", help="Max abs reward delta (statistical)"),
        ] = DEFAULT_REWARD_TOLERANCE,
        issue_out: Annotated[
            Path | None,
            typer.Option("--issue-out", help="Write the divergence issue draft here"),
        ] = None,
        roundtrip_task: Annotated[
            Path | None,
            typer.Option(
                "--roundtrip-task",
                help="Also run the structural round-trip check on this task dir",
            ),
        ] = None,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable parity verdict"),
        ] = False,
        require_adoption_report: Annotated[
            bool,
            typer.Option(
                "--require-adoption-report",
                help=(
                    "Fail unless adoption_report.json proves the adapter-adoption "
                    "loop is ready to scale"
                ),
            ),
        ] = False,
        loop_report_out: Annotated[
            Path | None,
            typer.Option(
                "--loop-report-out",
                help="Write the machine-readable adoption-loop verdict here",
            ),
        ] = None,
    ) -> None:
        """Run the parity gate for an adopted benchmark; emit a verdict."""
        root = benchmarks_dir or default_benchmarks_dir()
        try:
            name = validate_benchmark_name(name)
            data: Any = load_parity_experiment(root, name)
        except InvalidBenchmarkName as exc:
            if output_json:
                typer.echo(json.dumps(verify_error_payload(name, str(exc))))
                raise typer.Exit(1) from exc
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except BenchmarkNotFound as exc:
            if output_json:
                typer.echo(json.dumps(verify_error_payload(name, str(exc))))
                raise typer.Exit(1) from exc
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except ParityExperimentMissing as exc:
            if not output_json:
                console.print(f"[yellow]{exc}[/yellow]")
            data = {}

        try:
            adoption_report = load_adoption_report(root, name)
        except AdoptionReportInvalid as exc:
            if output_json:
                typer.echo(json.dumps(verify_error_payload(name, str(exc))))
                raise typer.Exit(1) from exc
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        try:
            adoption_loop_state = load_adoption_loop_state(root, name)
        except AdoptionLoopStateInvalid as exc:
            if output_json:
                typer.echo(json.dumps(verify_error_payload(name, str(exc))))
                raise typer.Exit(1) from exc
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        report = build_verify_report(name, data, tolerance=tolerance)
        if not output_json:
            console.print(f"[bold]Verdict:[/bold] {report.verdict}")
            console.print(
                f"  conversion: {report.conversion.agreed}/{report.conversion.compared} "
                f"criteria agree (rate {report.conversion.agreement_rate:.4f})"
            )
            if report.reward is not None:
                console.print(
                    f"  reward: max abs delta {report.reward.max_abs_delta:.4f} "
                    f"(tolerance {report.reward.tolerance:.4f})"
                )
            console.print(confidence_line(report))

        roundtrip_payload = None
        if roundtrip_task is not None:
            if not roundtrip_task.is_dir():
                if output_json:
                    typer.echo(
                        json.dumps(
                            verify_error_payload(
                                name, f"task dir not found: {roundtrip_task}"
                            )
                        )
                    )
                    raise typer.Exit(1)
                console.print(
                    f"[red]  round-trip: error: task dir not found: "
                    f"{roundtrip_task}[/red]"
                )
                raise typer.Exit(1)
            try:
                status, reasons = roundtrip_conformance_status(roundtrip_task)
            except OSError as exc:
                if output_json:
                    typer.echo(
                        json.dumps(
                            verify_error_payload(
                                name, f"could not check {roundtrip_task}: {exc}"
                            )
                        )
                    )
                    raise typer.Exit(1) from exc
                console.print(
                    f"[red]  round-trip: error: could not check "
                    f"{roundtrip_task}: {exc}[/red]"
                )
                raise typer.Exit(1) from exc
            roundtrip_payload = {
                "task": str(roundtrip_task),
                "status": status,
                "mismatch_reasons": reasons,
            }
            if not output_json:
                console.print(f"  round-trip: {status}")
                for reason in reasons:
                    console.print(f"    [yellow]- {reason}[/yellow]")

        adoption_loop = None
        if require_adoption_report or loop_report_out is not None:
            adoption_loop = build_adoption_loop_gate(
                report,
                adoption_report=adoption_report,
                loop_state=adoption_loop_state,
            )
            if not output_json:
                console.print(f"  adoption loop: {adoption_loop['status']}")
                for issue in adoption_loop["issues"]:
                    console.print(f"    [yellow]- {issue}[/yellow]")

        def _payload(
            *,
            issue_out_path: Path | None = None,
            issue_draft: str | None = None,
        ) -> dict[str, Any]:
            return verify_report_payload(
                report,
                roundtrip=roundtrip_payload,
                adoption_report=adoption_report,
                adoption_loop=adoption_loop,
                issue_out=issue_out_path,
                issue_draft=issue_draft,
            )

        def _write_loop_report(payload: Mapping[str, Any]) -> None:
            if loop_report_out is None:
                return
            loop_report_out.parent.mkdir(parents=True, exist_ok=True)
            loop_report_out.write_text(json.dumps(payload, indent=2) + "\n")

        if report.passed:
            payload = _payload()
            _write_loop_report(payload)
            if (
                require_adoption_report
                and adoption_loop is not None
                and not adoption_loop["passed"]
            ):
                if output_json:
                    typer.echo(json.dumps(payload))
                raise typer.Exit(1)
            if output_json:
                typer.echo(json.dumps(payload))
            return
        if report.verdict == "insufficient-evidence":
            # No parity data recorded yet — nothing diverged, so do not emit a
            # "parity could not be closed" divergence issue draft. confidence_line
            # above already told the author to run parity_test.py and record results.
            payload = _payload()
            _write_loop_report(payload)
            if output_json:
                typer.echo(json.dumps(payload))
            raise typer.Exit(1)
        issue = render_divergence_issue(report)
        if issue_out is not None:
            Path(issue_out).write_text(issue)
            if not output_json:
                console.print(f"[dim]Issue draft written to {issue_out}[/dim]")
        else:
            if not output_json:
                console.print(issue)
        payload = _payload(
            issue_out_path=issue_out,
            issue_draft=None if issue_out is not None else issue,
        )
        _write_loop_report(payload)
        if output_json:
            typer.echo(json.dumps(payload))
        raise typer.Exit(1)
