"""Benchmark adoption router behind ``bench eval adopt``.

This module is the real logic behind ``bench eval adopt``, the single command
that adopts an upstream benchmark into a BenchFlow benchmark (canonically
``bench eval adopt``; the legacy ``bench agent create|run|verify`` and the
intermediate ``bench adopt init|convert|verify`` remain as hidden deprecated
aliases). It sits downstream of every environment framework: a benchmark is
*routed* into the repo here, while ``bench eval create`` *runs* the tasks.

Three cohesive action bodies — module-level functions so the canonical command
and the deprecated alias closures share one implementation:

:func:`run_scaffold_action`  Deterministic scaffold of ``benchmarks/<name>/``
             matching the reference layout (``benchmarks/programbench/``).
             Fail-closed: refuses to overwrite an existing benchmark and
             validates the slug.
:func:`run_convert_action`   Driver that assembles the adoption context (source
             + adoption skills + conversion guide) and launches the host
             ``codex`` CLI to drive the conversion toward a ``benchmarks/<name>/``
             pull request. Context assembly and launch-command construction are
             pure functions so they are unit-testable with a fake exec layer; the
             live ``codex`` run is a manual-validation step.
:func:`run_verify_action`    Closes the adopt->verify loop. Runs the parity gate
             for an adopted benchmark and emits a confidence verdict. The gate is
             *parity only*: a faithful translation must reproduce the original's
             behavior on identical inputs (including any reward-hackability the
             original has — parity never "improves" or sanitizes the source). On
             a divergence it prints a draft GitHub issue body for human support
             instead of filing anything automatically.

The canonical surface is the single ``bench eval adopt`` command in
:mod:`benchflow.cli.adopt`; :func:`register_agent_router` registers the hidden
deprecated alias groups (``bench agent``, ``bench adopt``) that still expose the
three verbs and forward to these same action functions.
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
from rich.console import Console
from rich.markup import escape

# The parity gate (parsers, scoring, verdict) lives in agent_router_parity to
# keep this module focused on the adopt action wiring (scaffold/convert/verify).
# Re-exported here (see __all__) so the public API is unchanged, e.g.
# ``from benchflow.agent_router import build_verify_report``.
from benchflow.agent_router_guide import CONVERSION_GUIDE
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

# ── Errors ────────────────────────────────────────────────────────────


class InvalidBenchmarkName(ValueError):
    """Raised when a benchmark slug fails validation."""


class BenchmarkExistsError(FileExistsError):
    """Raised when scaffolding would overwrite an existing benchmark."""


class BenchmarkNotFound(FileNotFoundError):
    """Raised when an operation targets a benchmark that was never adopted."""


class ParityExperimentMissing(FileNotFoundError):
    """Raised when an adopted benchmark has no parity_experiment.json yet."""


class CodexLaunchError(RuntimeError):
    """Raised when the host codex CLI cannot be launched (e.g. no credentials)."""


class ParityRerunError(RuntimeError):
    """Raised when ``verify --rerun`` cannot independently re-execute parity."""


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

    The schema is what ``bench eval adopt --verify`` reads: per-criterion verdict pairs
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
        AdoptionSkill(
            "reference-benchmark", "benchmarks/programbench/ (worked example)"
        ),
        AdoptionSkill(
            "parity-harness", "parity_test.py + parity_experiment.json (verify gate)"
        ),
    ]


def assemble_adoption_context(
    source: str,
    name: str,
    *,
    skills: Sequence[AdoptionSkill],
    target_dir: str | None = None,
) -> str:
    """Assemble the full codex prompt for adopting ``source`` (pure function).

    Includes the source, the target benchmark path, the adoption skills
    (reference benchmark + parity harness), and the embedded conversion guide
    (:data:`agent_router_guide.CONVERSION_GUIDE`). ``target_dir`` overrides the
    default ``benchmarks/<name>/`` path (e.g. when ``--benchmarks-dir`` points the
    conversion at a non-default root) so the prompt matches where the package was
    scaffolded.
    """
    target = target_dir or f"benchmarks/{name}/"
    skill_lines = "\n".join(f"- {s.name}: {s.reference}" for s in skills)
    return "\n".join(
        [
            f"# Benchmark adoption: {name}",
            "",
            "Adopt the source benchmark below into a BenchFlow benchmark by",
            "following the conversion guide. Produce the converter, parity",
            "tests, metadata, and task directories, then open a pull request.",
            "",
            f"Source benchmark: {source}",
            f"Target directory: {target}",
            "",
            "## Adoption skills",
            skill_lines,
            "",
            CONVERSION_GUIDE,
        ]
    )


def build_codex_launch_command(
    prompt: str,
    *,
    workdir: Path | str,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
    config_overrides: Sequence[str] = (),
) -> list[str]:
    """Construct the host ``codex exec`` argv for the adoption run (pure).

    ``config_overrides`` are passed through as codex ``-c key=value`` flags, so
    host ``~/.codex/config.toml`` drift can be worked around per-run without
    editing the user's config (e.g. ``-c service_tier=flex`` when an installed
    codex version rejects a stale value).
    """
    command = [
        codex_bin,
        "exec",
        "--cd",
        str(workdir),
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
    ]
    for override in config_overrides:
        command += ["-c", override]
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
    benchmarks_dir: Path | None = None,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
    config_overrides: Sequence[str] = (),
) -> AdoptionLaunch:
    """Assemble context + build the codex command (no exec, no credentials).

    ``benchmarks_dir`` (when set) points the conversion at a non-default root, so
    the prompt's target path matches where the package was scaffolded.
    """
    name = validate_benchmark_name(name)
    skills = collect_adoption_skills()
    target_dir = f"{benchmarks_dir}/{name}/" if benchmarks_dir is not None else None
    prompt = assemble_adoption_context(
        source, name, skills=skills, target_dir=target_dir
    )
    command = build_codex_launch_command(
        prompt,
        workdir=repo_root,
        codex_bin=codex_bin,
        model=model,
        sandbox=sandbox,
        config_overrides=config_overrides,
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

    try:
        return subprocess.run(command, cwd=cwd, env=dict(env)).returncode
    except FileNotFoundError as exc:
        # Missing codex binary (default ``codex`` not on PATH, or a bad
        # --codex-bin) raises FileNotFoundError here; re-raise as the
        # already-handled launch error so the CLI prints a clean hint instead
        # of a raw traceback on every fresh/CI machine.
        raise CodexLaunchError(
            f"codex binary not found: {command[0]!r} — install codex or pass "
            "--codex-bin with the path to the binary"
        ) from exc


def run_agent_adoption(
    source: str,
    name: str,
    *,
    repo_root: Path,
    exec_fn: ExecFn,
    env: Mapping[str, str] | None = None,
    auth_file: Path | None = None,
    benchmarks_dir: Path | None = None,
    codex_bin: str = "codex",
    model: str | None = None,
    sandbox: str = "workspace-write",
    config_overrides: Sequence[str] = (),
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
        benchmarks_dir=benchmarks_dir,
        codex_bin=codex_bin,
        model=model,
        sandbox=sandbox,
        config_overrides=config_overrides,
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
            f"`bench eval adopt {name} --scaffold-only` first"
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


def rerun_parity_experiment(
    benchmarks_root: Path,
    name: str,
    *,
    runner: Callable[[list[str], Path], tuple[int, str, str]] | None = None,
) -> Any:
    """Independently re-execute a benchmark's ``parity_test.py`` and return its
    fresh side-by-side results, instead of trusting the recorded JSON.

    The default verify gate *scores the recorded* ``parity_experiment.json`` —
    fast, but it trusts an artifact the conversion produced about itself. This
    runs ``python <benchmark>/parity_test.py --mode side-by-side`` and parses the
    JSON it prints, so ``verify --rerun`` validates the conversion independently.

    CONTRACT: ``--mode side-by-side`` must emit (to stdout) the same shape the
    recorded ``parity_experiment.json`` uses and that :func:`build_verify_report`
    scores — i.e. per-criterion comparisons (``conversion_parity.tasks[].
    criteria_results[]`` or a recognized ``*_parity`` summary block) and/or
    reward samples. The scaffolded ``parity_test.py`` already prints this shape.

    Fail-closed: a missing script, a nonzero exit, unparseable output, OR output
    that parses but carries NO scoreable parity data all raise
    ``ParityRerunError`` — ``--rerun`` never silently falls back to stale/absent
    data, and never reports a misleading ``insufficient-evidence`` verdict on a
    shape the gate cannot read (a benchmark whose recorded JSON *would* score).

    ``runner`` is injected in tests; it returns ``(returncode, stdout, stderr)``.
    """
    name = validate_benchmark_name(name)
    benchmark_dir = Path(benchmarks_root) / name
    if not benchmark_dir.exists():
        raise BenchmarkNotFound(
            f"benchmark not adopted: {benchmark_dir} — run "
            f"`bench eval adopt {name} --scaffold-only` first"
        )
    script = benchmark_dir / "parity_test.py"
    if not script.exists():
        raise ParityRerunError(
            f"no parity_test.py in {benchmark_dir} — cannot --rerun "
            "(scaffold it with `bench eval adopt <name> --scaffold-only` and "
            "implement side-by-side)"
        )
    command = ["python", str(script), "--mode", "side-by-side"]
    returncode, stdout, stderr = (runner or _run_parity_script)(command, benchmark_dir)
    if returncode != 0:
        raise ParityRerunError(
            f"parity_test.py --mode side-by-side exited {returncode}: "
            f"{(stderr or stdout).strip()[-2000:]}"
        )
    data = _parse_parity_stdout(stdout)
    # Fail closed if the re-run output parses but is not in the scoreable shape:
    # without this, build_verify_report would report `insufficient-evidence` and
    # `--rerun` would silently FAIL the gate on a benchmark whose recorded JSON
    # would pass — defeating the feature (gh review on #694).
    if not extract_criterion_comparisons(data) and not extract_reward_samples(data):
        shape = (
            f"top-level keys {sorted(data)}"
            if isinstance(data, dict)
            else f"a top-level {type(data).__name__}"
        )
        raise ParityRerunError(
            "parity_test.py --mode side-by-side produced no scoreable parity data "
            "(no criterion comparisons and no reward samples) in the "
            f"parity_experiment.json shape the gate scores — got {shape}. Expected "
            "conversion_parity.tasks[].criteria_results[] (or a recognized "
            "*_parity summary block) and/or reward samples."
        )
    return data


def _parse_parity_stdout(stdout: str) -> Any:
    """Parse a ``parity_test.py`` JSON payload, tolerating leading log lines.

    Parses the whole (stripped) output first so a clean JSON object OR a
    top-level array parses directly; only falls back to slicing the outermost
    ``{...}`` object when the whole-string parse fails (e.g. log preamble)."""
    text = stdout.strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ParityRerunError(
            "could not parse parity_test.py --mode side-by-side output as JSON "
            f"(expected the recorded parity_experiment.json shape): {exc}"
        ) from exc


def _run_parity_script(
    command: list[str], cwd: Path, *, timeout_sec: int = 600
) -> tuple[int, str, str]:
    import subprocess
    import sys

    # Use the interpreter running benchflow so parity_test.py imports resolve.
    argv = [sys.executable, *command[1:]] if command[:1] == ["python"] else command
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired as exc:
        # A hung parity_test.py must not wedge the gate forever.
        raise ParityRerunError(
            f"parity_test.py --mode side-by-side timed out after {timeout_sec}s"
        ) from exc
    return proc.returncode, proc.stdout, proc.stderr


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


# ── Action bodies (shared by the canonical command and deprecated aliases) ──
#
# These are the real logic for the three adoption actions. The canonical single
# ``bench eval adopt`` command (cli/adopt.py) calls them directly; the deprecated
# alias closures in register_agent_router below call them after emitting a
# deprecation notice. Each takes an explicit ``console`` so the caller controls
# the output sink.


def run_scaffold_action(
    name: str,
    benchmarks_dir: Path | None,
    *,
    console: Console,
) -> None:
    """Scaffold ``benchmarks/<name>/`` for a new benchmark adoption."""
    root = benchmarks_dir or default_benchmarks_dir()
    try:
        target, written = create_benchmark(name, root)
    except (InvalidBenchmarkName, BenchmarkExistsError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Scaffolded[/green] {target}")
    for rel in written:
        console.print(f"  {rel}")


def run_convert_action(
    source: str,
    name: str | None,
    *,
    model: str | None,
    dry_run: bool,
    codex_bin: str,
    codex_config: list[str] | None,
    console: Console,
    benchmarks_dir: Path | None = None,
) -> None:
    """Drive the conversion workflow by launching the host codex CLI.

    ``benchmarks_dir`` (when set) is threaded into the codex prompt so the
    conversion target matches where the package was scaffolded.
    """
    import shlex

    repo_root = default_repo_root()
    overrides = tuple(codex_config or ())
    try:
        resolved = name or derive_name_from_source(source)
        if dry_run:
            launch = prepare_adoption_launch(
                source,
                resolved,
                repo_root=repo_root,
                benchmarks_dir=benchmarks_dir,
                codex_bin=codex_bin,
                model=model,
                config_overrides=overrides,
            )
            # Verbatim, copy-pasteable command: no console-width hard
            # wrapping (which would split tokens like the --cd path) and
            # no rich-markup interpretation of the prompt text.
            console.print(
                " ".join(shlex.quote(c) for c in launch.command),
                soft_wrap=True,
                markup=False,
            )
            return
        code = run_agent_adoption(
            source,
            resolved,
            repo_root=repo_root,
            exec_fn=_subprocess_exec,
            benchmarks_dir=benchmarks_dir,
            codex_bin=codex_bin,
            model=model,
            config_overrides=overrides,
        )
    except (InvalidBenchmarkName, CodexLaunchError) as exc:
        # escape(): CodexLaunchError embeds the user-supplied --codex-bin
        # and InvalidBenchmarkName the benchmark name — either can contain
        # Rich markup that would make this handler itself raise MarkupError.
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    raise typer.Exit(code)


def run_verify_action(
    name: str,
    *,
    benchmarks_dir: Path | None,
    tolerance: float,
    issue_out: Path | None,
    roundtrip_task: Path | None,
    rerun: bool,
    console: Console,
) -> None:
    """Run the parity gate for an adopted benchmark; emit a verdict."""
    root = benchmarks_dir or default_benchmarks_dir()
    try:
        name = validate_benchmark_name(name)
        if rerun:
            console.print("[dim]Re-executing parity_test.py --mode side-by-side…[/dim]")
            data: Any = rerun_parity_experiment(root, name)
        else:
            data = load_parity_experiment(root, name)
    except InvalidBenchmarkName as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except BenchmarkNotFound as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except ParityRerunError as exc:
        console.print(f"[red]--rerun failed: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except ParityExperimentMissing as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        data = {}

    report = build_verify_report(name, data, tolerance=tolerance)
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

    if roundtrip_task is not None:
        if not roundtrip_task.is_dir():
            console.print(
                f"[red]  round-trip: error: task dir not found: {roundtrip_task}[/red]"
            )
            raise typer.Exit(1)
        try:
            status, reasons = roundtrip_conformance_status(roundtrip_task)
        except OSError as exc:
            console.print(
                f"[red]  round-trip: error: could not check "
                f"{roundtrip_task}: {exc}[/red]"
            )
            raise typer.Exit(1) from exc
        console.print(f"  round-trip: {status}")
        for reason in reasons:
            console.print(f"    [yellow]- {reason}[/yellow]")

    if report.passed:
        return
    if report.verdict == "insufficient-evidence":
        # No parity data recorded yet — nothing diverged, so do not emit a
        # "parity could not be closed" divergence issue draft. confidence_line
        # above already told the author to run parity_test.py and record results.
        raise typer.Exit(1)
    issue = render_divergence_issue(report)
    if issue_out is not None:
        Path(issue_out).write_text(issue)
        console.print(f"[dim]Issue draft written to {escape(str(issue_out))}[/dim]")
    else:
        console.print(issue)
    raise typer.Exit(1)


# ── CLI registration (thin; real logic lives above) ───────────────────


# Canonical adoption verbs (the single ``bench eval adopt`` command flattens
# these into modes) and the deprecated ``bench adopt`` / ``bench agent`` alias
# groups that still expose them as subcommands. The same action functions back
# both surfaces.
ADOPT_VERBS = {"scaffold": "init", "drive": "convert", "verify": "verify"}
AGENT_ALIAS_VERBS = {"scaffold": "create", "drive": "run", "verify": "verify"}

# Where each deprecated verb now lives on the flattened canonical command.
_CANONICAL_HINTS = {
    "scaffold": "bench eval adopt <name> --scaffold-only",
    "drive": "bench eval adopt <source>",
    "verify": "bench eval adopt <name> --verify",
}


def register_agent_router(
    agent_app: typer.Typer,
    *,
    verbs: dict[str, str] | None = None,
    deprecated_as: str | None = None,
) -> None:
    """Register the deprecated benchmark-adoption alias subcommands onto ``agent_app``.

    The canonical surface is the single ``bench eval adopt`` command
    (:mod:`benchflow.cli.adopt`). This registers the hidden deprecated alias
    groups (``bench adopt init|convert|verify`` and ``bench agent
    create|run|verify``): ``deprecated_as`` is the alias group name, and each
    closure emits a one-line deprecation notice pointing at the new canonical
    shape before forwarding to the shared action function, so legacy scripts keep
    working through 0.6.
    """
    from benchflow.cli._shared import warn_deprecated

    console = Console()
    verbs = verbs or ADOPT_VERBS
    hidden = deprecated_as is not None

    def _maybe_warn(slot: str) -> None:
        if deprecated_as is not None:
            warn_deprecated(
                f"bench {deprecated_as} {verbs[slot]}",
                _CANONICAL_HINTS[slot],
            )

    @agent_app.command(verbs["scaffold"], hidden=hidden)
    def adopt_init(
        name: Annotated[
            str, typer.Argument(help="Benchmark slug (lowercase, hyphenated)")
        ],
        benchmarks_dir: Annotated[
            Path | None,
            typer.Option("--benchmarks-dir", help="Target benchmarks/ directory"),
        ] = None,
    ) -> None:
        """Scaffold benchmarks/<name>/ for a new benchmark adoption (deprecated alias)."""
        _maybe_warn("scaffold")
        run_scaffold_action(name, benchmarks_dir, console=console)

    @agent_app.command(verbs["drive"], hidden=hidden)
    def adopt_convert(
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
        codex_config: Annotated[
            list[str] | None,
            typer.Option(
                "--codex-config",
                "-c",
                help="Codex config override as key=value, passed to codex as "
                "`-c key=value`; repeatable (e.g. -c service_tier=flex to work "
                "around host ~/.codex/config.toml drift)",
            ),
        ] = None,
    ) -> None:
        """Drive the conversion workflow by launching the host codex CLI (deprecated alias)."""
        _maybe_warn("drive")
        run_convert_action(
            source,
            name,
            model=model,
            dry_run=dry_run,
            codex_bin=codex_bin,
            codex_config=codex_config,
            console=console,
        )

    @agent_app.command(verbs["verify"], hidden=hidden)
    def adopt_verify(
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
        rerun: Annotated[
            bool,
            typer.Option(
                "--rerun",
                help="Independently re-execute parity_test.py --mode side-by-side "
                "and score its fresh output, instead of the recorded "
                "parity_experiment.json",
            ),
        ] = False,
    ) -> None:
        """Run the parity gate for an adopted benchmark; emit a verdict (deprecated alias)."""
        _maybe_warn("verify")
        run_verify_action(
            name,
            benchmarks_dir=benchmarks_dir,
            tolerance=tolerance,
            issue_out=issue_out,
            roundtrip_task=roundtrip_task,
            rerun=rerun,
            console=console,
        )
