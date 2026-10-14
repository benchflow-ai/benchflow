"""Static skill checks: NVIDIA SkillEvaluator Tier 1 and NVIDIA SkillSpector.

`SkillEvaluator <https://github.com/NVIDIA/SkillEvaluator>`_ (Apache-2.0) is a
three-tier evaluation framework for agent skills. This module runs its Tier 1
static checks: schema, version, PII, license, code integrity, dependency
audit, Unicode smuggling, quality score and script lint.
`SkillSpector <https://github.com/NVIDIA/SkillSpector>`_ (Apache-2.0) is a
security scanner for agent skills. This module runs its static scan.

Design choices:

- **Subprocess, not import.** Both tools are driven through their CLIs and
  their JSON reports, so benchflow never imports their internals or inherits
  their Python pins. Neither is on PyPI, so they cannot be benchflow extras
  (PyPI rejects direct references); users install them with ``uv tool``.
- **SkillSpector runs once.** SkillEvaluator's own ``security`` check mainly
  reruns SkillSpector; its ``code-integrity`` check already runs Bandit (and
  Semgrep and Gitleaks when installed). benchflow runs SkillSpector directly,
  which keeps its risk score and verdict, and leaves ``security`` out of the
  default Tier 1 checks so the same scan does not run twice.
- **No model calls.** Both CLIs get ``--no-llm``. Tier 2 (embeddings) and
  Tier 3 (live agents) are not wired here; ``bench skills eval`` measures
  live lift. The ``dependency`` check does make network calls: pip-audit
  looks pinned packages up in a vulnerability database.
- **benchflow's gate.** A skill blocks when a schema, security or dependency
  finding is at or above ``fail_on`` (default ``high``), or when a checker
  fails to run. SkillEvaluator 0.5.0 reports dependency CVEs and banned
  packages only as legacy strings (``results[].legacy``), so they are read
  from there. SkillEvaluator's own verdict also fails on the quality score,
  on dead links and on missing scanners; those stay advisory here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SKILLEVALUATOR = "skillevaluator"
SKILLSPECTOR = "skillspector"

#: Environment variables that point at an executable that is not on ``PATH``.
BIN_ENV = {
    SKILLEVALUATOR: "BENCHFLOW_SKILLEVALUATOR_BIN",
    SKILLSPECTOR: "BENCHFLOW_SKILLSPECTOR_BIN",
}

#: Oldest versions whose JSON reports this module was checked against.
MIN_VERSIONS = {SKILLEVALUATOR: (0, 5, 0), SKILLSPECTOR: (2, 12, 0)}

INSTALL_COMMANDS = {
    SKILLEVALUATOR: (
        "uv tool install --python 3.13 "
        '"skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.5.0"'
    ),
    SKILLSPECTOR: (
        "uv tool install "
        '"skillspector @ git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0"'
    ),
}

#: SkillEvaluator 0.5.0's default ``tier1`` checks minus ``security`` (which
#: mainly reruns SkillSpector, run here as its own pass), plus the opt-in
#: ``dependency`` CVE audit (pip-audit; needs network access).
DEFAULT_TIER1_CHECKS: tuple[str, ...] = (
    "schema",
    "version",
    "pii",
    "license",
    "code-integrity",
    "dependency",
    "unicode",
    "quality",
    "lint",
)

SEVERITIES = ("info", "low", "medium", "high", "critical")
FAIL_ON_CHOICES = ("critical", "high", "medium", "low", "never")
DEFAULT_FAIL_ON = "high"

#: SkillEvaluator finding categories that gate. Everything else (quality,
#: lint, license, version, hygiene) is reported but advisory.
_SCHEMA_CATEGORIES = frozenset({"SCHEMA", "PLUGIN_SCHEMA"})
_SECURITY_CATEGORIES = frozenset(
    {"SECURITY", "BANDIT", "SEMGREP", "SECRET", "PII", "UNICODE", "DEPENDENCY"}
)

#: SkillEvaluator 0.5.0 reports dependency CVEs through its legacy string API
#: only: ``[CVE-HIGH] requests==2.19.0: PYSEC-2018-28 -> upgrade to 2.20.0``
#: in ``results[].legacy.errors`` (or ``warnings`` below its error
#: threshold), never in ``results[].findings``. Only these tags are read: the
#: other legacy strings repeat structured findings.
_LEGACY_DEPENDENCY = re.compile(
    r"\[(CVE|SAFETY)-(CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s*(.*)", re.DOTALL
)
_DEPENDENCY_VALIDATOR = "Dependency Vulnerability Audit"
#: Part of ``code-integrity``; reports banned packages and dead links only as
#: legacy error strings.
_HYGIENE_VALIDATOR = "Code Integrity & Hygiene"
_LEGACY_CATEGORIES = frozenset({"DEPENDENCY", "CVE", "SAFETY", "HYGIENE"})

#: Directories never searched for SKILL.md.
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv"})

RunFn = Callable[..., "subprocess.CompletedProcess[str]"]


class SkillToolMissing(RuntimeError):
    """Neither checker is installed."""


class SkillCheckError(RuntimeError):
    """A checker exited with an error or left no readable report."""


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(p) for p in version.split("."))
    return major, minor, patch


@dataclass(frozen=True)
class Tool:
    """One located checker executable."""

    name: str
    path: str
    version: str | None

    @property
    def minimum(self) -> str:
        return ".".join(str(p) for p in MIN_VERSIONS[self.name])

    def warning(self) -> str | None:
        """A one-line warning when the version is unknown or too old."""
        if self.version is None:
            return (
                f"could not read the {self.name} version; benchflow expects "
                f"{self.minimum} or newer"
            )
        if _version_tuple(self.version) < MIN_VERSIONS[self.name]:
            return (
                f"{self.name} {self.version} is older than {self.minimum}; its "
                f"report may not parse. Upgrade: {INSTALL_COMMANDS[self.name]}"
            )
        return None


def find_tool(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """Locate *name*, honoring its ``BENCHFLOW_*_BIN`` override."""
    source = os.environ if env is None else env
    override = source.get(BIN_ENV[name])
    return shutil.which(override) if override else shutil.which(name)


def tool_version(path: str, *, run: RunFn = subprocess.run) -> str | None:
    """Read ``X.Y.Z`` from ``<path> --version``; ``None`` when unreadable."""
    try:
        proc = run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", f"{proc.stdout}\n{proc.stderr}")
    return match.group(0) if match else None


def locate_tool(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
    run: RunFn = subprocess.run,
) -> Tool | None:
    path = find_tool(name, env)
    if path is None:
        return None
    return Tool(name=name, path=path, version=tool_version(path, run=run))


def install_hint(name: str) -> str:
    return (
        f"{name} is not installed. Install it with: {INSTALL_COMMANDS[name]} "
        f"(or set {BIN_ENV[name]} to the executable)."
    )


def iter_skill_dirs(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield each skill directory (one holding ``SKILL.md``) under *paths*.

    A path may be a skill directory, its ``SKILL.md``, or any directory, which
    is searched recursively. A skill's own subdirectories are not searched,
    and a skill reached through two paths is yielded once. A path with no
    name of its own (``.``, ``..``) is yielded resolved, so the skill keeps
    its directory name.
    """
    seen: set[Path] = set()
    for raw in paths:
        root = Path(raw)
        if root.is_file() and root.name == "SKILL.md":
            root = root.parent
        for skill_dir in _walk_skills(root):
            key = skill_dir.resolve()
            if key not in seen:
                seen.add(key)
                yield key if skill_dir.name in ("", "..") else skill_dir


def _walk_skills(root: Path) -> Iterator[Path]:
    if (root / "SKILL.md").is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        if "SKILL.md" in filenames:
            dirnames[:] = []
            yield Path(dirpath)
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)


def severity_rank(severity: str) -> int:
    """Index into :data:`SEVERITIES`; unknown labels rank as ``info``."""
    label = severity.lower()
    return SEVERITIES.index(label) if label in SEVERITIES else 0


@dataclass(frozen=True)
class Finding:
    """One finding from either checker, in one shape."""

    source: str
    severity: str
    category: str
    check: str
    message: str
    file: str = ""
    line: int | None = None

    @property
    def gates(self) -> bool:
        """Schema and security findings gate; quality, lint and the rest advise."""
        if self.source == SKILLSPECTOR:
            return True
        return self.category.upper() in _SCHEMA_CATEGORIES | _SECURITY_CATEGORIES

    @property
    def location(self) -> str:
        if not self.file:
            return ""
        return f"{self.file}:{self.line}" if self.line else self.file


@dataclass
class Tier1Result:
    """SkillEvaluator Tier 1 outcome for one skill."""

    status: str
    passed: bool
    quality_score: float | None = None
    quality_grade: str | None = None
    failed_checks: list[str] = field(default_factory=list)
    incomplete_scans: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report_path: Path | None = None


@dataclass
class SecurityScan:
    """SkillSpector static-scan outcome for one skill."""

    risk_score: int | None = None
    risk_severity: str | None = None
    recommendation: str | None = None
    analysis_status: str | None = None
    findings: list[Finding] = field(default_factory=list)
    report_path: Path | None = None


@dataclass
class SkillCheck:
    """Both checkers' results for one skill, plus anything that failed to run."""

    skill_path: Path
    tier1: Tier1Result | None = None
    security: SecurityScan | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def skill_name(self) -> str:
        return self.skill_path.resolve().name

    @property
    def findings(self) -> list[Finding]:
        return [
            *(self.tier1.findings if self.tier1 else []),
            *(self.security.findings if self.security else []),
        ]

    def blocking_findings(self, fail_on: str = DEFAULT_FAIL_ON) -> list[Finding]:
        if fail_on == "never":
            return []
        threshold = severity_rank(fail_on)
        blocking = [
            f
            for f in self.findings
            if f.gates and severity_rank(f.severity) >= threshold
        ]
        return sorted(blocking, key=lambda f: -severity_rank(f.severity))

    def is_blocking(self, fail_on: str = DEFAULT_FAIL_ON) -> bool:
        return bool(self.errors) or bool(self.blocking_findings(fail_on))

    def to_dict(self, fail_on: str = DEFAULT_FAIL_ON) -> dict[str, Any]:
        def _section(section: Tier1Result | SecurityScan | None) -> dict | None:
            if section is None:
                return None
            data = asdict(section)
            data["report_path"] = (
                str(section.report_path) if section.report_path else None
            )
            return data

        return {
            "skill_name": self.skill_name,
            "skill_path": str(self.skill_path),
            "blocking": self.is_blocking(fail_on),
            "blocking_findings": [asdict(f) for f in self.blocking_findings(fail_on)],
            "errors": list(self.errors),
            "skillevaluator": _section(self.tier1),
            "skillspector": _section(self.security),
        }


def _relative_file(raw: Any, skill_dir: Path) -> str:
    """Report paths relative to the skill; SkillEvaluator mixes cwd-relative ones."""
    text = str(raw or "")
    if not text:
        return ""
    try:
        return Path(text).resolve().relative_to(skill_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return text


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _legacy_findings(
    result: Mapping[str, Any], name: str, structured: list[Finding]
) -> list[Finding]:
    """Findings SkillEvaluator 0.5.0 reports only as ``legacy`` strings.

    Reads ``[CVE-*]`` and ``[SAFETY-*]`` strings from any validator, and the
    untagged errors of the hygiene validator (banned packages gate as
    dependency findings; dead links stay advisory). Skipped when the
    validator already reports such findings in structured form, so a newer
    release that does so is not counted twice. A failed dependency audit
    whose strings do not parse still blocks.
    """
    legacy = result.get("legacy")
    if not isinstance(legacy, Mapping):
        legacy = {}
    if any(f.category.upper() in _LEGACY_CATEGORIES for f in structured):
        return []
    found: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    def _add(severity: str, category: str, check: str, message: str) -> None:
        if (severity, message) not in seen:
            seen.add((severity, message))
            found.append(Finding(SKILLEVALUATOR, severity, category, check, message))

    for key in ("errors", "warnings"):
        for raw in legacy.get(key) or []:
            text = str(raw).strip()
            match = _LEGACY_DEPENDENCY.match(text)
            if match:
                tag, severity, message = match.groups()
                _add(severity.lower(), "DEPENDENCY", tag, message.strip())
            elif key == "errors" and name == _HYGIENE_VALIDATOR and not structured:
                if "banned package" in text.lower():
                    _add("high", "DEPENDENCY", "banned_package", text)
                else:
                    _add("high", "HYGIENE", "hygiene", text)

    if (
        name == _DEPENDENCY_VALIDATOR
        and result.get("status") == "failed"
        and not found
        and not any(f.gates for f in structured)
    ):
        errors = [str(e) for e in legacy.get("errors") or []]
        _add(
            "high",
            "DEPENDENCY",
            "dependency_audit_failed",
            errors[0] if errors else f"{name} failed",
        )
    return found


def parse_tier1_report(data: Mapping[str, Any], skill_dir: Path) -> Tier1Result:
    """Normalize a ``skillevaluator tier1 -r json`` report.

    Missing optional sections (``quality_summary``, ``results``) parse as
    empty so a partial report still reads.
    """
    quality_score: float | None = None
    quality_grade: str | None = None
    quality = data.get("quality_summary") or []
    if quality and isinstance(quality[0], Mapping):
        raw_score = quality[0].get("overall_score")
        if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
            quality_score = float(raw_score)
        quality_grade = str(quality[0].get("grade") or "") or None

    failed: list[str] = []
    findings: list[Finding] = []
    for result in data.get("results") or []:
        name = str(result.get("validator") or "?")
        if result.get("status") == "failed":
            failed.append(name)
        structured = [
            Finding(
                source=SKILLEVALUATOR,
                severity=str(f.get("severity") or "info").lower(),
                category=str(f.get("category") or ""),
                check=str(f.get("check_name") or ""),
                message=str(f.get("message") or ""),
                file=_relative_file(f.get("file_path"), skill_dir),
                line=_int_or_none(f.get("line_number")),
            )
            for f in result.get("findings") or []
        ]
        findings += structured
        findings += _legacy_findings(result, name, structured)

    status = str(
        data.get("overall_status")
        or ("passed" if data.get("overall_passed") else "failed")
    )
    return Tier1Result(
        status=status,
        passed=bool(data.get("overall_passed", status == "passed")),
        quality_score=quality_score,
        quality_grade=quality_grade,
        failed_checks=failed,
        incomplete_scans=[str(s) for s in data.get("incomplete_scans") or []],
        findings=findings,
    )


def parse_skillspector_report(data: Mapping[str, Any], skill_dir: Path) -> SecurityScan:
    """Normalize a ``skillspector scan --format json`` report."""
    risk = data.get("risk_assessment") or {}
    completeness = data.get("analysis_completeness") or {}
    findings: list[Finding] = []
    for issue in data.get("issues") or []:
        location = issue.get("location") or {}
        findings.append(
            Finding(
                source=SKILLSPECTOR,
                severity=str(issue.get("severity") or "info").lower(),
                category=str(issue.get("category") or ""),
                check=str(issue.get("id") or ""),
                message=str(issue.get("finding") or issue.get("explanation") or ""),
                file=_relative_file(location.get("file"), skill_dir),
                line=_int_or_none(location.get("start_line")),
            )
        )
    return SecurityScan(
        risk_score=_int_or_none(risk.get("score")),
        risk_severity=str(risk.get("severity") or "") or None,
        recommendation=str(risk.get("recommendation") or "") or None,
        analysis_status=(
            str(completeness.get("status") or "") or None
            if isinstance(completeness, Mapping)
            else None
        ),
        findings=findings,
    )


def build_tier1_command(
    executable: str,
    skill_dir: Path,
    output_dir: Path,
    *,
    checks: Iterable[str] = DEFAULT_TIER1_CHECKS,
    min_score: int | None = None,
) -> list[str]:
    """The ``skillevaluator tier1 validate`` argv (pure; used by tests).

    The explicit ``validate`` subcommand keeps a skill named like a Tier 1
    subcommand (``pii-scan``, ``quality-check``) from being read as one.
    """
    argv = [
        executable,
        "tier1",
        "validate",
        str(skill_dir),
        "--no-llm",
        "-r",
        "json",
        "-o",
        str(output_dir),
        "--checks",
        ",".join(checks),
    ]
    if min_score is not None:
        argv += ["--min-score", str(min_score)]
    return argv


def build_skillspector_command(
    executable: str, skill_dir: Path, output_file: Path
) -> list[str]:
    """The ``skillspector scan`` argv (pure; used by tests)."""
    return [
        executable,
        "scan",
        str(skill_dir),
        "--no-llm",
        "--format",
        "json",
        "--output",
        str(output_file),
    ]


def _run(
    argv: list[str],
    *,
    what: str,
    skill_dir: Path,
    timeout_sec: int,
    env: Mapping[str, str] | None,
    run: RunFn,
) -> subprocess.CompletedProcess[str]:
    try:
        return run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=None if env is None else dict(env),
        )
    except OSError as e:
        raise SkillCheckError(f"{what} could not start: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SkillCheckError(
            f"{what} timed out after {timeout_sec}s on {skill_dir}"
        ) from e


def _report_state(path: Path) -> str:
    """``execution_successful``, completeness and error from a run-error report."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(data, Mapping):
        return ""
    parts = []
    if "execution_successful" in data:
        parts.append(f"execution_successful={data['execution_successful']}")
    completeness = data.get("analysis_completeness")
    if isinstance(completeness, Mapping) and completeness.get("status"):
        parts.append(f"analysis {completeness['status']}")
    if isinstance(data.get("error"), str) and data["error"]:
        parts.append(f"error: {data['error'][:200]}")
    return f" ({', '.join(parts)})" if parts else ""


def _parse[T](
    parser: Callable[[Mapping[str, Any], Path], T],
    data: Mapping[str, Any],
    skill_dir: Path,
    *,
    what: str,
    path: Path | None,
) -> T:
    """Run *parser*, turning an unexpected report shape into one skill's error."""
    try:
        return parser(data, skill_dir)
    except (AttributeError, TypeError, KeyError, IndexError, ValueError) as e:
        raise SkillCheckError(
            f"unexpected {what} report shape in {path}: {type(e).__name__}: {e}"
        ) from e


def _read_report(
    path: Path | None,
    proc: subprocess.CompletedProcess[str],
    *,
    what: str,
    skill_dir: Path,
) -> dict[str, Any]:
    # Both CLIs exit 0 (clean) or 1 (policy failure) with a report; anything
    # else, or no report, is a run error. SkillSpector writes its report and
    # then exits 2 when the scan did not execute successfully.
    if proc.returncode not in (0, 1) or path is None or not path.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        detail = "\n".join(tail) if tail else "(no output)"
        if path is not None and path.is_file():
            state = f"exited {proc.returncode} (run error); report: {path}"
            state += _report_state(path)
        else:
            state = f"exited {proc.returncode} with no usable report"
        raise SkillCheckError(f"{what} {state} on {skill_dir}:\n{detail}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SkillCheckError(f"unreadable {what} report {path}: {e}") from e
    if not isinstance(data, dict):
        raise SkillCheckError(f"unexpected {what} report shape in {path}")
    return data


def run_tier1(
    skill_dir: Path,
    tool: Tool,
    output_dir: Path,
    *,
    checks: Iterable[str] = DEFAULT_TIER1_CHECKS,
    min_score: int | None = None,
    timeout_sec: int = 600,
    env: Mapping[str, str] | None = None,
    run: RunFn = subprocess.run,
) -> Tier1Result:
    """Run SkillEvaluator Tier 1 (no LLM) on one skill and parse its report."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("skillevaluator-*.json"):
            stale.unlink()  # a rerun into -o must not read the last run's report
    except OSError as e:
        raise SkillCheckError(f"cannot use report directory {output_dir}: {e}") from e
    argv = build_tier1_command(
        tool.path, skill_dir.resolve(), output_dir, checks=checks, min_score=min_score
    )
    proc = _run(
        argv,
        what=SKILLEVALUATOR,
        skill_dir=skill_dir,
        timeout_sec=timeout_sec,
        env=env,
        run=run,
    )
    reports = sorted(
        output_dir.glob("skillevaluator-*.json"), key=lambda p: p.stat().st_mtime
    )
    report_path = reports[-1] if reports else None
    data = _read_report(report_path, proc, what=SKILLEVALUATOR, skill_dir=skill_dir)
    result = _parse(
        parse_tier1_report, data, skill_dir, what=SKILLEVALUATOR, path=report_path
    )
    result.report_path = report_path
    return result


def run_skillspector(
    skill_dir: Path,
    tool: Tool,
    output_file: Path,
    *,
    timeout_sec: int = 600,
    env: Mapping[str, str] | None = None,
    run: RunFn = subprocess.run,
) -> SecurityScan:
    """Run the SkillSpector static scan (no LLM) on one skill and parse it."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)
    except OSError as e:
        raise SkillCheckError(f"cannot write report {output_file}: {e}") from e
    argv = build_skillspector_command(tool.path, skill_dir.resolve(), output_file)
    proc = _run(
        argv,
        what=SKILLSPECTOR,
        skill_dir=skill_dir,
        timeout_sec=timeout_sec,
        env=env,
        run=run,
    )
    data = _read_report(output_file, proc, what=SKILLSPECTOR, skill_dir=skill_dir)
    scan = _parse(
        parse_skillspector_report, data, skill_dir, what=SKILLSPECTOR, path=output_file
    )
    scan.report_path = output_file
    return scan


def check_skill(
    skill_dir: Path,
    *,
    skillevaluator: Tool | None,
    skillspector: Tool | None,
    checks: Iterable[str] = DEFAULT_TIER1_CHECKS,
    min_score: int | None = None,
    output_dir: Path | None = None,
    timeout_sec: int = 600,
    run: RunFn = subprocess.run,
) -> SkillCheck:
    """Run whichever checkers are given on one skill.

    Pass ``skillspector=None`` to skip the security scan. With *output_dir*
    the raw reports are kept there (``skillevaluator/`` and
    ``skillspector.json``); otherwise they go to a temporary directory and
    only the parsed results survive.
    """
    skill_dir = Path(skill_dir)
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"No SKILL.md in {skill_dir}")
    if skillevaluator is None and skillspector is None:
        raise SkillToolMissing(install_hint(SKILLEVALUATOR))

    # If the caller adds SkillEvaluator's own `security` check, point it at
    # the same SkillSpector benchflow found.
    env: dict[str, str] | None = None
    if skillspector is not None:
        env = {**os.environ, "SKILLEVALUATOR_SKILLSPECTOR_PATH": skillspector.path}

    result = SkillCheck(skill_path=skill_dir)
    with tempfile.TemporaryDirectory(prefix="benchflow-skill-check-") as tmp:
        out = Path(output_dir) if output_dir is not None else Path(tmp)
        if skillevaluator is not None:
            try:
                result.tier1 = run_tier1(
                    skill_dir,
                    skillevaluator,
                    out / "skillevaluator",
                    checks=checks,
                    min_score=min_score,
                    timeout_sec=timeout_sec,
                    env=env,
                    run=run,
                )
            except SkillCheckError as e:
                result.errors.append(str(e))
        if skillspector is not None:
            try:
                result.security = run_skillspector(
                    skill_dir,
                    skillspector,
                    out / "skillspector.json",
                    timeout_sec=timeout_sec,
                    run=run,
                )
            except SkillCheckError as e:
                result.errors.append(str(e))
        if output_dir is None:
            for section in (result.tier1, result.security):
                if section is not None:
                    section.report_path = None
    return result
