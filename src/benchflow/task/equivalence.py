"""Deterministic wrong-answer / equivalence battery for task verifiers.

``oracle = 1.0`` does not show that a correct answer can score, nor that a
wrong one cannot. This battery runs the oracle once in a sandbox, keeps the
files it wrote, and re-runs the task's own verifier on perturbed copies:

* the oracle output as-is must score 1.0 (and again at the end: a second
  score that differs marks the verifier non-deterministic);
* wrong answers (output removed, a file emptied or cut short, wrong numbers,
  off-by-one, wrong unit scale, row values swapped) must score below it — one
  that does not is a **false positive**;
* value-equivalent answers (re-indented or key-reordered JSON, a trailing
  newline, floats moved by one ulp, another row or evidence order, a graded prose
  literal paraphrased, another citation form, request-file ids only) must
  score like it — one that does not is a **false negative**, unless the
  instruction fixes that form (reported as a warning).

When the oracle output and an empty output score the same below 1.0, the
verifier fails before it reads the answer (a blocker) and no variant is run.

Everything is graded in one sandbox: the answer files are restored to the
oracle's bytes before each variant, then BenchFlow's normal verify path
(hardening, test.sh, reward parsing) runs. Variants are listed in
:mod:`benchflow.task.equivalence_variants`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from benchflow.task.equivalence_variants import (
    FAMILIES,
    Variant,
    build_variants,
)
from benchflow.task.paths import TaskPaths

logger = logging.getLogger(__name__)

Status = Literal[
    "clean", "defects", "blocked", "oracle-fails", "no-answer", "flaky", "error"
]
Verdict = Literal[
    "rejected",
    "accepted",
    "false-positive",
    "false-negative",
    "form-fixed-by-instruction",
    "verifier-error",
]

#: Files larger than this are not treated as answers (not downloaded or mutated).
MAX_ANSWER_BYTES = 5 * 1024 * 1024
#: At most this many oracle-written files are mutated.
MAX_ANSWER_FILES = 8
#: Extensions whose content the text/number variants may rewrite. Other files
#: (source code a workspace-edit oracle patched, binaries) only get the
#: removed / emptied variants.
TEXT_ANSWER_SUFFIXES = frozenset(
    {
        "",
        ".txt",
        ".csv",
        ".tsv",
        ".md",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".xml",
        ".out",
    }
)
_EPS = 1e-9


@dataclass(frozen=True)
class GradeOutcome:
    """One verifier run: a scalar reward, or the verifier's error."""

    reward: float | None
    error: str | None = None


class AnswerGrader(Protocol):
    """Scores the oracle output with ``changes`` applied (path -> bytes, or None to delete)."""

    async def grade(self, changes: Mapping[str, bytes | None]) -> GradeOutcome: ...


@dataclass(frozen=True)
class VariantResult:
    variant: Variant
    reward: float | None
    verdict: Verdict
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.variant.to_dict(),
            "reward": self.reward,
            "verdict": self.verdict,
            "error": self.error,
        }


@dataclass
class EquivalenceReport:
    """Per-task battery result. ``issues()`` is what ``bench tasks check`` prints."""

    task: str
    sandbox: str | None = None
    status: Status = "clean"
    reason: str = ""
    oracle_reward: float | None = None
    empty_output_reward: float | None = None
    oracle_rescore_reward: float | None = None
    answer_files: list[str] = field(default_factory=list)
    results: list[VariantResult] = field(default_factory=list)
    skipped_variants: int = 0

    @property
    def false_negatives(self) -> list[VariantResult]:
        return [r for r in self.results if r.verdict == "false-negative"]

    @property
    def false_positives(self) -> list[VariantResult]:
        return [r for r in self.results if r.verdict == "false-positive"]

    @property
    def warnings(self) -> list[VariantResult]:
        return [r for r in self.results if r.verdict == "form-fixed-by-instruction"]

    def issues(self) -> list[str]:
        if self.status in ("blocked", "oracle-fails", "no-answer", "error"):
            return [f"equivalence: {self.reason}"]
        out: list[str] = []
        oracle = _fmt(self.oracle_reward)
        for r in self.results:
            v = r.variant
            if r.verdict == "false-negative":
                out.append(
                    f"equivalence: false negative [{v.family}] {v.label} "
                    f"scored {_fmt(r.reward)} < oracle {oracle}"
                )
            elif r.verdict == "false-positive":
                out.append(
                    f"equivalence: false positive [{v.family}] {v.label} "
                    f"scored {_fmt(r.reward)}, not below oracle {oracle}"
                )
            elif r.verdict == "verifier-error":
                out.append(
                    f"equivalence: verifier error [{v.family}] {v.label}: {r.error}"
                )
        if self.status == "flaky":
            out.append(f"equivalence: {self.reason}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "sandbox": self.sandbox,
            "status": self.status,
            "reason": self.reason,
            "oracle_reward": self.oracle_reward,
            "empty_output_reward": self.empty_output_reward,
            "oracle_rescore_reward": self.oracle_rescore_reward,
            "answer_files": self.answer_files,
            "false_negatives": len(self.false_negatives),
            "false_positives": len(self.false_positives),
            "warnings": len(self.warnings),
            "skipped_variants": self.skipped_variants,
            "issues": self.issues(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"


def _fmt(value: float | None) -> str:
    return (
        "none"
        if value is None
        else f"{value:g}"
        if value not in (0.0, 1.0)
        else f"{value:.1f}"
    )


# ----------------------------------------------------------------- inputs


def task_instruction(task_dir: Path) -> str:
    """The agent-facing instruction (task.md body or instruction.md), or ""."""
    from benchflow.rollout._setup import _read_task_instruction

    try:
        return _read_task_instruction(task_dir)
    except (OSError, ValueError):
        return ""


def verifier_sources(task_dir: Path) -> dict[str, str]:
    """Python files of the task's verifier, keyed by task-relative path."""
    root = TaskPaths(task_dir).tests_dir
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        try:
            out[path.relative_to(task_dir).as_posix()] = path.read_text(
                errors="replace"
            )
        except OSError:
            continue
    return out


def request_file_ids(task_dir: Path) -> set[str]:
    """``id`` values of the request file under environment/, if any."""
    env = TaskPaths(task_dir).environment_dir
    for name in ("requests.json", "request.json", "inputs.json"):
        path = env / name
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError):
            return set()
        rows: Any = obj
        if isinstance(obj, dict):
            rows = next(
                (
                    obj[k]
                    for k in ("requests", "cases", "items", "queries")
                    if isinstance(obj.get(k), list)
                ),
                [],
            )
        if isinstance(rows, list):
            return {str(r["id"]) for r in rows if isinstance(r, dict) and "id" in r}
    return set()


# ---------------------------------------------------------------- battery


async def run_equivalence_battery(
    task_dir: Path,
    grader: AnswerGrader,
    answers: Mapping[str, bytes],
    *,
    families: Iterable[str] | None = None,
    max_variants: int = 60,
    sandbox: str | None = None,
    text_paths: Iterable[str] | None = None,
) -> EquivalenceReport:
    """Grade the oracle output, an empty output and every variant with ``grader``.

    ``answers`` maps sandbox paths to the oracle's bytes. ``text_paths`` limits
    content-rewriting variants to those paths (default: all answers).
    """
    task_dir = Path(task_dir)
    report = EquivalenceReport(
        task=task_dir.name, sandbox=sandbox, answer_files=sorted(answers)
    )
    wanted = set(families) if families is not None else None
    if wanted is not None and (unknown := wanted - set(FAMILIES)):
        raise ValueError(f"unknown equivalence families: {', '.join(sorted(unknown))}")

    control = await grader.grade({})
    report.oracle_reward = control.reward
    if control.reward is None:
        report.status = "oracle-fails"
        report.reason = f"oracle output produced no reward: {control.error}"
        return report
    empty_changes: dict[str, bytes | None] = {p: None for p in answers}
    empty = await grader.grade(empty_changes) if answers else None
    report.empty_output_reward = empty.reward if empty else None
    if control.reward < 1.0 - _EPS:
        if (
            empty is not None
            and empty.reward is not None
            and abs(empty.reward - control.reward) <= _EPS
        ):
            report.status = "blocked"
            report.reason = (
                f"blocker: the oracle output and an empty output both score "
                f"{_fmt(control.reward)}; the verifier fails before reading the answer"
            )
        else:
            report.status = "oracle-fails"
            report.reason = f"oracle output scores {_fmt(control.reward)}, not 1.0"
        return report
    if not answers:
        report.status = "no-answer"
        report.reason = (
            "the oracle wrote no file outside system paths; nothing to perturb"
        )
        return report

    rewritable = set(text_paths) if text_paths is not None else set(answers)
    variants = build_variants(
        {p: b for p, b in answers.items() if p in rewritable},
        instruction=task_instruction(task_dir),
        verifier_sources=verifier_sources(task_dir),
        request_ids=request_file_ids(task_dir),
    )
    # The empty output is graded above; binary / code answers still get "emptied".
    variants = [v for v in variants if v.family != "empty-output"]
    variants.insert(
        0,
        Variant(
            "empty-output",
            "reject",
            "every answer file removed: " + ", ".join(sorted(answers)),
            empty_changes,
        ),
    )
    for path in sorted(set(answers) - rewritable):
        variants.append(
            Variant(
                "empty-file",
                "reject",
                f"{Path(path).name}: emptied to 0 bytes",
                {path: b""},
            )
        )
    if wanted is not None:
        variants = [v for v in variants if v.family in wanted]
    if len(variants) > max_variants:
        report.skipped_variants = len(variants) - max_variants
        variants = variants[:max_variants]

    for variant in variants:
        outcome = (
            empty
            if variant.family == "empty-output" and empty is not None
            else await grader.grade(variant.changes)
        )
        report.results.append(_classify(variant, outcome, control.reward))

    rescore = await grader.grade({})
    report.oracle_rescore_reward = rescore.reward
    if rescore.reward is None or abs(rescore.reward - control.reward) > _EPS:
        report.status = "flaky"
        report.reason = (
            f"oracle output re-scored {_fmt(rescore.reward)} after the battery "
            f"(first {_fmt(control.reward)}); the verifier is not deterministic "
            "or a variant left state behind, so the results above are unreliable"
        )
    elif (
        report.false_negatives
        or report.false_positives
        or any(r.verdict == "verifier-error" for r in report.results)
    ):
        report.status = "defects"
        report.reason = (
            f"{len(report.false_negatives)} false negative(s), "
            f"{len(report.false_positives)} false positive(s)"
        )
    else:
        report.status = "clean"
        report.reason = f"oracle {_fmt(control.reward)}; every wrong answer scored lower and every equivalent one the same"
    return report


def _classify(variant: Variant, outcome: GradeOutcome, control: float) -> VariantResult:
    reward = outcome.reward
    if reward is None:
        return VariantResult(
            variant, None, "verifier-error", outcome.error or "no reward"
        )
    if variant.expect == "reject":
        verdict: Verdict = "false-positive" if reward >= control - _EPS else "rejected"
    elif reward < control - _EPS:
        verdict = (
            "form-fixed-by-instruction"
            if variant.instruction_fixes_form
            else "false-negative"
        )
    else:
        verdict = "accepted"
    return VariantResult(variant, reward, verdict)


# ---------------------------------------------------------------- sandbox

_MARKER = "/logs/.bf-equivalence-marker"
_PRUNE = [
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/var",
    "/usr",
    "/etc",
    "/boot",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/opt",
    "/logs",
    "/tests",
    "/solution",
    "/oracle",
    "/verifier",
    "/snap",
    "/nix",
]


def _find_new_files_cmd() -> str:
    prune = " -o ".join(f"-path {p}" for p in _PRUNE)
    names = " -o ".join(
        f"-name {shlex.quote(n)}"
        for n in (
            "__pycache__",
            "node_modules",
            ".git",
            "site-packages",
            ".cache",
            ".*",
        )
    )
    return (
        f"find / \\( {prune} -o {names} \\) -prune -o -type f -cnewer {_MARKER} "
        f"-size -{MAX_ANSWER_BYTES // 1024}k -print 2>/dev/null | head -n 200"
    )


def choose_answer_files(
    task_dir: Path, written: Iterable[str], declared: Iterable[str] = ()
) -> list[str]:
    """Answer files among the paths the oracle wrote.

    Declared ``[verifier] submission_files`` win; otherwise files the verifier
    code names (by path or basename) win; otherwise every written file.
    """
    written = sorted(set(written))
    declared = [p for p in declared if p in written]
    if declared:
        return declared[:MAX_ANSWER_FILES]
    code = "\n".join(verifier_sources(task_dir).values())
    tests_dir = TaskPaths(task_dir).tests_dir
    if tests_dir.is_dir():
        for sh in tests_dir.glob("*.sh"):
            with contextlib.suppress(OSError):
                code += "\n" + sh.read_text(errors="replace")
    named = [p for p in written if p in code or Path(p).name in code]
    return (named or written)[:MAX_ANSWER_FILES]


class SandboxAnswerGrader:
    """Grades answer variants in a live BenchFlow sandbox via the normal verify path."""

    def __init__(
        self,
        env: Any,
        task: Any,
        planes: Any,
        rollout_paths: Any,
        workspace: str,
        answers: Mapping[str, bytes],
        staging: Path,
    ) -> None:
        self.env = env
        self.task = task
        self.planes = planes
        self.rollout_paths = rollout_paths
        self.workspace = workspace
        self.answers = dict(answers)
        self.staging = staging

    async def _write(self, path: str, content: bytes | None) -> None:
        if content is None:
            await self.env.exec(
                f"rm -f {shlex.quote(path)}", user="root", timeout_sec=30
            )
            return
        local = self.staging / uuid.uuid4().hex
        local.write_bytes(content)
        try:
            await self.env.exec(
                f"mkdir -p {shlex.quote(str(Path(path).parent))}",
                user="root",
                timeout_sec=30,
            )
            await self.env.upload_file(local, path)
        finally:
            local.unlink(missing_ok=True)

    async def grade(self, changes: Mapping[str, bytes | None]) -> GradeOutcome:
        from benchflow.rollout import _verify_rollout

        for path, content in {**self.answers, **changes}.items():
            await self._write(path, content)
        rewards, error, _diag = await _verify_rollout(
            self.env,
            self.task,
            self.rollout_paths,
            {},
            self.planes,
            sandbox_user=None,
            workspace=self.workspace,
        )
        if error is not None:
            return GradeOutcome(None, error)
        reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
        if not isinstance(reward, int | float) or isinstance(reward, bool):
            return GradeOutcome(
                None, f"verifier returned no scalar reward: {rewards!r}"
            )
        return GradeOutcome(float(reward))


async def check_equivalence_async(
    task_dir: Path,
    *,
    sandbox_type: str,
    families: Iterable[str] | None = None,
    max_variants: int = 60,
) -> EquivalenceReport:
    """Run the oracle in a fresh ``sandbox_type`` sandbox, then the battery.

    The sandbox is deleted afterwards whatever happens.
    """
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import _resolve_agent_cwd, _start_env_and_upload
    from benchflow.rollout._setup import _run_oracle
    from benchflow.task.paths import RolloutPaths
    from benchflow.task.task import Task

    task_dir = Path(task_dir)
    report = EquivalenceReport(task=task_dir.name, sandbox=sandbox_type)
    task = Task(task_dir)
    if not task.paths.solve_path.exists():
        report.status = "oracle-fails"
        report.reason = (
            "no oracle: the battery needs oracle/solve.sh (or solution/solve.sh)"
        )
        return report
    planes = default_rollout_planes()
    with tempfile.TemporaryDirectory(prefix="benchflow-equivalence-") as tmp:
        rollout_paths = RolloutPaths(Path(tmp) / "rollout")
        rollout_paths.mkdir()
        staging = Path(tmp) / "staging"
        staging.mkdir()
        name = f"equivalence-{task_dir.name}-{uuid.uuid4().hex[:8]}"
        env = planes.create_environment(
            sandbox_type,
            task,
            task_dir,
            name,
            rollout_paths,
            preserve_agent_network=False,
            environment_manifest=None,
        )
        try:
            await _start_env_and_upload(env, task_dir, {})
            workspace = await _resolve_agent_cwd(env, task)
            await planes.prepare_log_dirs(env, sandbox_user=None)
            await env.exec(f": > {_MARKER} && sleep 1", user="root", timeout_sec=30)
            timeout = int(task.config.agent.timeout_sec or 1800)
            trajectory, _ = await _run_oracle(env, task_dir, timeout)
            rc = trajectory[0].get("return_code") if trajectory else None
            if rc != 0:
                report.status = "oracle-fails"
                report.reason = f"oracle solve.sh exited rc={rc}: {str(trajectory[0].get('stdout', ''))[-300:] if trajectory else ''}"
                return report
            found = await env.exec(_find_new_files_cmd(), user="root", timeout_sec=120)
            written = [
                line.strip()
                for line in (found.stdout or "").splitlines()
                if line.strip().startswith("/")
            ]
            declared = list(getattr(task.config.verifier, "submission_files", []) or [])
            chosen = choose_answer_files(task_dir, written, declared)
            answers: dict[str, bytes] = {}
            for path in chosen:
                local = staging / uuid.uuid4().hex
                await env.download_file(path, local)
                answers[path] = local.read_bytes()
                local.unlink(missing_ok=True)
            text_paths = [
                p for p in answers if Path(p).suffix.lower() in TEXT_ANSWER_SUFFIXES
            ]
            grader = SandboxAnswerGrader(
                env, task, planes, rollout_paths, workspace, answers, staging
            )
            return await run_equivalence_battery(
                task_dir,
                grader,
                answers,
                families=families,
                max_variants=max_variants,
                sandbox=sandbox_type,
                text_paths=text_paths,
            )
        except Exception as exc:
            logger.exception("equivalence battery failed")
            report.status = "error"
            report.reason = f"battery could not run: {exc}"
            return report
        finally:
            with contextlib.suppress(Exception):
                await env.stop(delete=True)


def check_equivalence(
    task_dir: Path,
    *,
    sandbox_type: str,
    families: Iterable[str] | None = None,
    max_variants: int = 60,
) -> EquivalenceReport:
    """Sync wrapper of :func:`check_equivalence_async` (fails inside a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            check_equivalence_async(
                task_dir,
                sandbox_type=sandbox_type,
                families=families,
                max_variants=max_variants,
            )
        )
    raise RuntimeError(
        "check_equivalence cannot run inside an active event loop; await check_equivalence_async instead"
    )
