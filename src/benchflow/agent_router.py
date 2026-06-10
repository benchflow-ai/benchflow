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
from typing import Annotated, Any, Literal

import typer

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
_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
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
    if not _SLUG_RE.match(name):
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


def collect_adoption_skills(repo_root: Path) -> list[AdoptionSkill]:
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
    skills = collect_adoption_skills(repo_root)
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


# ── verify: parity gate + confidence verdict ──────────────────────────

DEFAULT_REWARD_TOLERANCE = 0.02

Verdict = Literal["parity-confirmed", "parity-divergent", "insufficient-evidence"]


@dataclass(frozen=True)
class CriterionComparison:
    """One per-criterion verdict pair from the deterministic parity floor."""

    task_id: str
    criterion_id: str
    original_verdict: str
    adapted_verdict: str
    agreement: bool


@dataclass(frozen=True)
class RewardSample:
    """One legacy-vs-converted reward delta from the statistical layer."""

    task_id: str
    legacy_reward: float | None
    converted_reward: float | None
    delta: float


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def extract_criterion_comparisons(data: Mapping[str, Any]) -> list[CriterionComparison]:
    """Pull per-criterion verdict pairs from a parity_experiment.json mapping.

    Tolerant of the scaffold shape (``conversion_parity.tasks``) and the
    CONVERT.md example shape (top-level ``tasks``).
    """
    task_lists: list[Any] = []
    conv = data.get("conversion_parity")
    if isinstance(conv, Mapping):
        task_lists.extend(_as_list(conv.get("tasks")))
    task_lists.extend(_as_list(data.get("tasks")))

    out: list[CriterionComparison] = []
    for task in task_lists:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id", ""))
        for crit in _as_list(task.get("criteria_results")):
            if not isinstance(crit, Mapping):
                continue
            original = str(crit.get("original_verdict", ""))
            adapted = str(crit.get("adapted_verdict", ""))
            agreement = bool(crit.get("agreement", original == adapted))
            out.append(
                CriterionComparison(
                    task_id=task_id,
                    criterion_id=str(crit.get("criterion_id", "")),
                    original_verdict=original,
                    adapted_verdict=adapted,
                    agreement=agreement,
                )
            )
    return out


def _reward_pair(result: Mapping[str, Any]) -> tuple[float | None, float | None, float]:
    legacy = result.get("legacy_reward", result.get("pb_reward"))
    converted = result.get("converted_reward", result.get("bf_reward"))
    if legacy is None and isinstance(result.get("programbench"), Mapping):
        legacy = result["programbench"].get("reward")
    if converted is None and isinstance(result.get("benchflow"), Mapping):
        converted = result["benchflow"].get("reward")

    if legacy is not None and converted is not None:
        delta = abs(float(converted) - float(legacy))
    else:
        delta = abs(float(result.get("reward_delta", 0.0)))
    legacy_f = float(legacy) if legacy is not None else None
    converted_f = float(converted) if converted is not None else None
    return legacy_f, converted_f, delta


def extract_reward_samples(data: Mapping[str, Any]) -> list[RewardSample]:
    """Pull legacy-vs-converted reward deltas (the statistical parity layer).

    Tolerant of the scaffold shape (``reward_distribution_parity.samples``) and
    the reference ``agent_parity.results`` (programbench/benchflow rewards).
    """
    result_lists: list[Any] = []
    rdp = data.get("reward_distribution_parity")
    if isinstance(rdp, Mapping):
        result_lists.extend(_as_list(rdp.get("samples")))
    agent = data.get("agent_parity")
    if isinstance(agent, Mapping):
        result_lists.extend(_as_list(agent.get("results")))

    out: list[RewardSample] = []
    for result in result_lists:
        if not isinstance(result, Mapping):
            continue
        legacy, converted, delta = _reward_pair(result)
        out.append(
            RewardSample(
                task_id=str(result.get("task_id", "")),
                legacy_reward=legacy,
                converted_reward=converted,
                delta=delta,
            )
        )
    return out


@dataclass(frozen=True)
class ConversionParity:
    """Deterministic conversion-faithfulness floor."""

    comparisons: list[CriterionComparison]

    @property
    def compared(self) -> int:
        return len(self.comparisons)

    @property
    def agreed(self) -> int:
        return sum(1 for c in self.comparisons if c.agreement)

    @property
    def all_agree(self) -> bool:
        return self.compared > 0 and self.agreed == self.compared

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    @property
    def disagreements(self) -> list[CriterionComparison]:
        return [c for c in self.comparisons if not c.agreement]


@dataclass(frozen=True)
class RewardDistributionParity:
    """Statistical legacy-vs-converted reward parity layer."""

    samples: list[RewardSample]
    tolerance: float

    @property
    def max_abs_delta(self) -> float:
        return max((s.delta for s in self.samples), default=0.0)

    @property
    def within_tolerance(self) -> bool:
        return self.max_abs_delta <= self.tolerance

    @property
    def exceeding(self) -> list[RewardSample]:
        return [s for s in self.samples if s.delta > self.tolerance]


@dataclass(frozen=True)
class VerifyReport:
    """Parity verdict for an adopted benchmark."""

    name: str
    conversion: ConversionParity
    reward: RewardDistributionParity | None
    verdict: Verdict
    tolerance: float = DEFAULT_REWARD_TOLERANCE

    @property
    def passed(self) -> bool:
        return self.verdict == "parity-confirmed"


def build_verify_report(
    name: str,
    data: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_REWARD_TOLERANCE,
) -> VerifyReport:
    """Score parity and assign a confidence verdict.

    Parity-only gate over two layers:

    * deterministic floor — every compared criterion's converted verdict must
      match the original's verdict on identical inputs;
    * statistical layer — every legacy-vs-converted reward delta must sit within
      ``tolerance``.

    A layer that has no data does not block the verdict. With no data at all the
    verdict is ``insufficient-evidence`` (the support path). The gate never
    "improves" the source: a faithful conversion reproduces the original's
    behavior, including any reward-hackability it has.
    """
    conversion = ConversionParity(extract_criterion_comparisons(data))
    samples = extract_reward_samples(data)
    reward = RewardDistributionParity(samples, tolerance=tolerance) if samples else None

    has_conversion = conversion.compared > 0
    has_reward = reward is not None

    if not has_conversion and not has_reward:
        verdict: Verdict = "insufficient-evidence"
    else:
        conversion_ok = (not has_conversion) or conversion.all_agree
        reward_ok = (not has_reward) or reward.within_tolerance
        verdict = (
            "parity-confirmed" if conversion_ok and reward_ok else ("parity-divergent")
        )

    return VerifyReport(
        name=name,
        conversion=conversion,
        reward=reward,
        verdict=verdict,
        tolerance=tolerance,
    )


def confidence_line(report: VerifyReport) -> str:
    """User-facing confidence framing (correctness/parity-based, no fixed %)."""
    if report.verdict == "parity-confirmed":
        return (
            "High-confidence: the converted evaluation reproduces the original's "
            "verdicts on every compared criterion and stays within reward "
            "tolerance."
        )
    if report.verdict == "parity-divergent":
        return (
            "Divergence found: the conversion does not yet reproduce the "
            "original's behavior — iterate, then open an issue for support."
        )
    return (
        "Insufficient evidence: no recorded parity comparisons. Run "
        "parity_test.py and record results before trusting the conversion."
    )


def render_divergence_issue(report: VerifyReport) -> str:
    """Render a draft GitHub issue body for a non-confirmed verdict.

    Printed/saved for a human to file — never auto-filed.
    """
    lines = [
        f"## Benchmark adoption parity: {report.name}",
        "",
        f"**Verdict:** {report.verdict}",
        "",
        confidence_line(report),
        "",
        "### Conversion parity (deterministic floor)",
        f"- criteria compared: {report.conversion.compared}",
        f"- agreed: {report.conversion.agreed}",
        f"- agreement rate: {report.conversion.agreement_rate:.4f}",
    ]
    for c in report.conversion.disagreements:
        lines.append(
            f"  - {c.task_id}/{c.criterion_id}: original={c.original_verdict} "
            f"converted={c.adapted_verdict}"
        )

    lines.append("")
    lines.append("### Reward-distribution parity (statistical layer)")
    if report.reward is None:
        lines.append("- no reward samples recorded")
    else:
        lines.append(f"- samples: {len(report.reward.samples)}")
        lines.append(f"- max abs delta: {report.reward.max_abs_delta:.4f}")
        lines.append(f"- tolerance: {report.reward.tolerance:.4f}")
        for s in report.reward.exceeding:
            lines.append(
                f"  - {s.task_id}: legacy={s.legacy_reward} "
                f"converted={s.converted_reward} delta={s.delta:.4f}"
            )

    lines += [
        "",
        "### Ask",
        "Parity could not be closed for this conversion. The translation must",
        "reproduce the original's behavior on identical inputs (including any",
        "reward-hackability it has). This draft has NOT been filed — review it,",
        "iterate on the converter, and open it manually if you need support.",
    ]
    return "\n".join(lines)


def load_parity_experiment(benchmarks_root: Path, name: str) -> dict:
    """Load an adopted benchmark's parity_experiment.json (fail-closed)."""
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
    return json.loads(parity_file.read_text())


def roundtrip_conformance_status(
    task_dir: Path,
    *,
    report_fn: Callable[..., Any] | None = None,
) -> tuple[str, list[str]]:
    """Surface the structural round-trip conformance harness for one task.

    Thin wiring to ``benchflow.task.build_harbor_roundtrip_conformance_report``
    (the existing parity utility). Returns ``(status, mismatch_reasons)``.
    """
    if report_fn is None:
        from benchflow.task import build_harbor_roundtrip_conformance_report

        report_fn = build_harbor_roundtrip_conformance_report
    report = report_fn(task_dir)
    reasons = [m.reason for m in getattr(report, "mismatches", [])]
    return report.status, reasons


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
    ) -> None:
        """Run the parity gate for an adopted benchmark; emit a verdict."""
        root = benchmarks_dir or default_benchmarks_dir()
        try:
            name = validate_benchmark_name(name)
            data: dict = load_parity_experiment(root, name)
        except InvalidBenchmarkName as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except BenchmarkNotFound as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except ParityExperimentMissing as exc:
            console.print(f"[yellow]{exc}[/yellow]")
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

        if report.passed:
            return
        issue = render_divergence_issue(report)
        if issue_out is not None:
            Path(issue_out).write_text(issue)
            console.print(f"[dim]Issue draft written to {issue_out}[/dim]")
        else:
            console.print(issue)
        raise typer.Exit(1)
