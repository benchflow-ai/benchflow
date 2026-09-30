"""Read finished BenchFlow jobs and trials, and compare two jobs.

``bf.load_trial(path)`` reads one trial (rollout) directory: its result,
trajectory, verifier output, config, timing, checkpoints and branch lineage.
``bf.load_job(path)`` reads every trial under a job directory (or any folder
of trials). ``bf.compare(job_a, job_b)`` pairs their tasks.

Counting rules: a trial is *attempted*; it is *scored*
when it has a reward; an assessment error (the verifier failed) and an
unscored trial are counted apart; pass rates are given over scored and over
attempted trials; and control runs (the oracle, which runs the task's own
solution, and an empty/nop run) check the task rather than an agent, so they
are left out of the denominators unless ``include_controls=True``.

Both current and older layouts are read: ``rewards`` or a legacy top-level
``reward``, ``rollout_name`` or ``trial_name``, and the trajectory at
``trajectory/acp_trajectory.jsonl`` or ``agent/acp_trajectory.jsonl``.

>>> import benchflow as bf
>>> bf.Denominators(attempted=0).pass_rate_attempted is None
True
"""

from __future__ import annotations

import csv
import json
import logging
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from benchflow.branch_api import BranchChildResult
from benchflow.models import RolloutResult
from benchflow.pass_at_k import Sample, SolveRates
from benchflow.trajectories.viewer.models import RubricReview

logger = logging.getLogger(__name__)

__all__ = [
    "Comparison",
    "ComparisonRow",
    "ComparisonSummary",
    "Denominators",
    "Fork",
    "GROUP_KEYS",
    "GroupDenominators",
    "Job",
    "SETTINGS",
    "SettingCheck",
    "SettingMismatch",
    "Trial",
    "VerifierOutput",
    "compare",
    "load_job",
    "load_results_jsonl",
    "load_trial",
]

Control = Literal["oracle", "empty"]
Execution = Literal["completed", "errored", "timed_out", "integration_failed"]
Assessment = Literal["scored", "error", "unscored"]

_EMPTY_AGENTS = {"nop", "noop", "empty"}


@dataclass(frozen=True)
class GroupDenominators:
    """One group's denominators, e.g. key={"agent": ..., "model": ...}."""

    key: dict[str, Any]
    denominators: Denominators


_TRAJECTORY_PATHS = ("trajectory/acp_trajectory.jsonl", "agent/acp_trajectory.jsonl")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _normalize_legacy(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    if data.get("rewards") is None and isinstance(data.get("reward"), (int, float)):
        data["rewards"] = {"reward": float(data["reward"])}
    if not data.get("rollout_name") and data.get("trial_name"):
        data["rollout_name"] = data["trial_name"]
    if not data.get("task_name"):
        data["task_name"] = data.get("task_id") or ""
    return data


@dataclass(frozen=True)
class VerifierOutput:
    """What the verifier left in ``verifier/`` (None for a missing file)."""

    reward_text: str | None = None
    reward_json: Any = None
    stdout: str | None = None
    stderr: str | None = None
    ctrf: Any = None

    @classmethod
    def read(cls, verifier_dir: Path) -> VerifierOutput:
        """Read ``verifier/`` (reward, stdout, stderr, test report); missing files are None."""
        reward = _read_text(verifier_dir / "reward.txt")
        return cls(
            reward_text=reward.strip() if reward is not None else None,
            reward_json=_read_json(verifier_dir / "reward.json"),
            stdout=_read_text(verifier_dir / "test-stdout.txt"),
            stderr=_read_text(verifier_dir / "test-stderr.txt"),
            ctrf=_read_json(verifier_dir / "ctrf.json"),
        )


@dataclass(frozen=True)
class Fork:
    """One branch point recorded in the trial's ``tree.json``.

    ``value`` is V(checkpoint), the mean reward of the fork's children;
    child ``path`` values are relative to the trial directory.
    """

    id: str
    status: str | None
    value: float | None
    parent_restore: str | None
    children: list[BranchChildResult]
    snapshot: dict[str, Any] = field(default_factory=dict, repr=False)
    timing_sec: dict[str, Any] | None = field(default=None, repr=False)
    # "fork" or "retry" (bench eval run --retry-from-checkpoint); cost is
    # tokens/usd/usd_known/wall/parent/children/sandbox seconds.
    kind: str = "fork"
    parent_node: str | None = None
    cost: dict[str, Any] | None = field(default=None, repr=False)


def _forks(tree: Any) -> list[Fork]:
    if not isinstance(tree, dict):
        return []
    raw_forks = [f for f in tree.get("forks") or [] if isinstance(f, dict)]
    label_of = {
        c.get("node_id"): (c.get("intervention") or {}).get("label")
        for f in raw_forks
        for c in f.get("children") or []
    }
    forks = []
    for i, f in enumerate(raw_forks):
        children = [
            BranchChildResult(
                label=str(
                    (c.get("intervention") or {}).get("label") or c.get("node_id")
                ),
                status=str(c.get("status")),
                reward=c.get("reward"),
                reward_source=c.get("reward_source"),
                path=(c.get("artifacts") or {}).get("path"),
                fork_id=f.get("id"),
                parent_label=label_of.get(f.get("rollout")) if i else None,
                cost=c.get("cost") if isinstance(c.get("cost"), dict) else None,
                advantage=round(c["reward"] - f["value"], 6)
                if isinstance(c.get("reward"), int | float)
                and isinstance(f.get("value"), int | float)
                else None,
            )
            for c in f.get("children") or []
        ]
        forks.append(
            Fork(
                id=str(f.get("id")),
                status=f.get("status"),
                value=f.get("value"),
                parent_restore=f.get("parent_restore"),
                children=children,
                snapshot=f.get("snapshot") or {},
                timing_sec=f.get("timing_sec"),
                kind=str(f.get("kind") or "fork"),
                parent_node=f.get("parent_node"),
                cost=f.get("cost") if isinstance(f.get("cost"), dict) else None,
            )
        )
    return forks


@dataclass(repr=False)
class Trial:
    """One finished trial (rollout) directory.

    ``result`` is the typed :class:`RolloutResult`; ``execution`` is
    ``completed``, ``errored``, ``timed_out`` or ``integration_failed`` (the
    agent did nothing because its integration broke; see
    ``integration_failure``); ``assessment`` is ``scored``
    (a reward exists), ``error`` (the verifier failed) or ``unscored``;
    ``control`` is ``"oracle"``/``"empty"`` for control runs and None for
    agent runs. The trajectory is read on first use.
    """

    path: Path
    result: RolloutResult
    raw: dict[str, Any] = field(repr=False)
    config: dict[str, Any] = field(default_factory=dict, repr=False)
    timing: dict[str, Any] = field(default_factory=dict, repr=False)
    verifier: VerifierOutput = field(default_factory=VerifierOutput, repr=False)
    forks: list[Fork] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list, repr=False)
    # "result.json" for a trial folder; "results.jsonl" for a trial read from
    # a results.jsonl row (then ``path`` is that file and ``row`` its index).
    source: Literal["result.json", "results.jsonl"] = "result.json"
    row: int | None = None
    # The attempts of this trial's task in its Evaluation job (set by load_job).
    _attempts: list[Trial] | None = field(default=None, repr=False, compare=False)

    @property
    def attempts(self) -> list[Trial]:
        """Every rollout of this trial, oldest first, this one included.

        An Evaluation job retries a task in its own folder (and a resume
        re-runs it there), so a trial can take several rollouts: this lists
        them, for counting what a job ran or spent
        (``sum(len(t.attempts) for t in job.trials)``). One element when the
        task ran once, in a folder of independent rollouts (``bf.run_batch``),
        or when the trial was read on its own with :func:`load_trial`.
        """
        return list(self._attempts) if self._attempts else [self]

    @property
    def task_name(self) -> str:
        """The task this trial ran."""
        return self.result.task_name

    @property
    def agent(self) -> str:
        """The harness that ran (``result.agent``)."""
        return self.result.agent

    @property
    def model(self) -> str | None:
        """The model the agent used, or None."""
        return self.result.model

    @property
    def duration_sec(self) -> float | None:
        """Wall-clock seconds from start to finish, when both were recorded."""
        r = self.result
        if r.started_at is None or r.finished_at is None:
            return None
        return (r.finished_at - r.started_at).total_seconds()

    @property
    def reward(self) -> float | None:
        """The reward when the trial is scored; None when unscored (never 0 for a failure to score)."""
        if self.integration_failure is not None:
            return None
        reward = self.result.reward
        return reward if reward is not None and math.isfinite(reward) else None

    @cached_property
    def integration_failure(self) -> dict[str, Any] | None:
        """Why the agent did nothing useful, when its integration broke.

        Recorded at run time as ``integration_failure_info``; for results
        written before that existed it is detected here from the trajectory
        and agent logs (:mod:`benchflow.integration_health`). Such a trial is
        an execution failure (``integration_failed``) and unscored: its
        verifier reward is withheld, never counted as 0.
        """
        from benchflow.integration_health import (
            diagnose_trial_dir,
            stored_integration_failure,
        )

        stored = stored_integration_failure(self.raw)
        if stored is not None:
            return stored
        if (
            self.source != "result.json"
            or self.control is not None
            or self.result.n_tool_calls
            or self.result.reward is None
        ):
            return None
        finding = diagnose_trial_dir(self.path, self.raw)
        if finding is None:
            return None
        return {
            **finding.to_dict(),
            "reward_withheld": self.result.rewards,
            "detected": "on read",
        }

    @property
    def passed(self) -> bool:
        """Whether the trial passed (and its agent integration did not break)."""
        return self.result.passed and self.integration_failure is None

    def __repr__(self) -> str:
        # Compact for notebooks: the path and the full result stay attributes.
        return (
            f"Trial(task={self.task_name!r}, agent={self.agent!r}, "
            f"model={self.model!r}, reward={self.reward!r}, "
            f"execution={self.execution!r}, assessment={self.assessment!r})"
        )

    @property
    def cost_usd(self) -> float | None:
        """USD of this rollout, or None (see ``result.price_source``)."""
        return self.result.cost_usd

    @property
    def total_tokens(self) -> int | None:
        """Tokens of this rollout, or None when unknown."""
        return self.result.total_tokens

    @property
    def control(self) -> Control | None:
        """``oracle`` or ``empty`` for a control run, None for an agent run."""
        variant = str(
            self.config.get("task_variant") or self.raw.get("task_variant") or ""
        )
        if variant.startswith("oracle"):
            return "oracle"
        if variant.startswith("empty"):
            return "empty"
        # Control copies by name: a task copy suffixed __o is an oracle
        # control, one suffixed __e an empty-solution control.
        if self.task_name.endswith("__e"):
            return "empty"
        if self.task_name.endswith("__o"):
            return "oracle"
        agent = (self.result.agent or "").strip().lower()
        if agent == "oracle":
            return "oracle"
        if agent in _EMPTY_AGENTS and not self.result.model:
            return "empty"
        return None

    @property
    def execution(self) -> Execution:
        """How the run ended: completed, errored, timed_out or integration_failed."""
        if self.integration_failure is not None:
            return "integration_failed"
        if not self.result.error:
            return "completed"
        return "timed_out" if self.result.error_category == "timeout" else "errored"

    @property
    def assessment(self) -> Assessment:
        """Whether it got a score: scored, error (the verifier failed) or unscored."""
        from benchflow._utils.scoring import assessment_withholds_score

        if self.reward is not None and not assessment_withholds_score(self.raw):
            return "scored"
        if self.result.verifier_error:
            return "error"
        return "unscored"

    def solve_sample(self) -> Sample:
        """This trial as one pass@k sample: the reward when scored (else
        None, unscored), and the review gate's verdict when it has one."""
        from benchflow.pass_at_k import Sample

        scoring = self.result.scoring
        passed = (
            bool(scoring.passed)
            if scoring is not None and getattr(scoring, "status", None) == "complete"
            else None
        )
        return Sample(
            self.task_name,
            self.reward if self.assessment == "scored" else None,
            passed,
        )

    @cached_property
    def review(self) -> RubricReview | None:
        """The ``bench review`` rubric verdict for this trial, if any.

        Looked up as: the nearest ``review*/**/review_report.json``
        within four folders above the trial; a valid review wins over an
        invalid one. None for trials read from ``results.jsonl``.
        """
        if self.source != "result.json":
            return None
        from benchflow.trajectories.viewer.payload import _load_rubric

        return _load_rubric(self.path)

    @property
    def scoring(self) -> Any:
        """The automatic reviewer's gate verdict (``ScoringResult``), or None."""
        return self.result.scoring

    def settings(self) -> dict[str, Any]:
        """The run settings ``bf.compare`` checks between two sides."""
        import hashlib

        agent_env = self.config.get("agent_env")
        prompts = None
        if self.source == "result.json":
            try:
                raw = (self.path / "prompts.json").read_bytes()
                prompts = hashlib.sha256(raw).hexdigest()[:16]
            except OSError:
                prompts = None
        return {
            "task_digest": self.raw.get("task_digest")
            or self.config.get("task_digest"),
            "model": self.result.model,
            "harness": self.result.agent or None,
            "dataset_name": self.raw.get("dataset_name"),
            "dataset_version": self.raw.get("dataset_version"),
            "reasoning_effort": self.config.get("reasoning_effort"),
            "environment": self.config.get("environment"),
            "sandbox_user": self.config.get("sandbox_user"),
            "timeout_sec": self.config.get("timeout_sec"),
            "agent_variable_names": sorted(agent_env)
            if isinstance(agent_env, dict)
            else None,
            "prompts_sha256": prompts,
        }

    @cached_property
    def branch_view(self) -> dict[str, Any]:
        """This trial's ``benchflow.branch-view/1`` document
        (:func:`benchflow.branch_view.load_branch_view`)."""
        from benchflow.branch_view import load_branch_view

        return load_branch_view(self.path)

    @cached_property
    def trajectory(self) -> list[dict[str, Any]]:
        """The ACP events (current or older path); empty when none was captured."""
        if self.source != "result.json":
            return []
        for rel in _TRAJECTORY_PATHS:
            text = _read_text(self.path / rel)
            if text is not None:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
        return []

    def to_json_dict(
        self, *, include_trajectory: bool = True, include_verifier: bool = True
    ) -> dict[str, Any]:
        """The trial as a ``benchflow.trial`` document (see ``benchflow.job_export``).

        Validates against ``docs/reference/schemas/benchflow-trial.v1.schema.json``.
        """
        from benchflow.job_export import trial_export

        return trial_export(
            self,
            include_trajectory=include_trajectory,
            include_verifier=include_verifier,
        ).model_dump(mode="json")

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> Any:
        """:meth:`to_json_dict` as a JSON string, or written to ``path`` (returned)."""
        return _write_json(self.to_json_dict(), path, indent)

    def to_record(self) -> dict[str, Any]:
        """``RolloutResult.to_record()`` plus control, execution, assessment and
        the settings ``bf.compare`` checks (flattened; ``model`` and ``agent``
        already appear once)."""
        settings = {
            k: v for k, v in self.settings().items() if k not in ("model", "harness")
        }
        if isinstance(settings.get("agent_variable_names"), list):
            settings["agent_variable_names"] = ",".join(
                settings["agent_variable_names"]
            )
        return {
            **self.result.to_record(),
            **settings,
            "reward": self.reward,
            "control": self.control,
            "execution": self.execution,
            "assessment": self.assessment,
            "attempts": len(self.attempts),
            "forks": len(self.forks),
            "review_valid": self.review.review_valid if self.review else None,
            "source": self.source,
            "path": str(self.path),
        }


def _write_json(document: dict[str, Any], path: str | Path | None, indent: int) -> Any:
    text = json.dumps(document, indent=indent, allow_nan=False)
    if path is None:
        return text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


def load_trial(path: str | Path) -> Trial:
    """Read one trial directory (or its ``result.json``) into a :class:`Trial`."""
    from benchflow.checkpoints import load_checkpoints

    path = Path(path)
    trial_dir = path.parent if path.name == "result.json" else path
    data = _read_json(trial_dir / "result.json")
    if not isinstance(data, dict):
        raise FileNotFoundError(
            f"No readable result.json in {trial_dir}; pass a trial directory "
            "(jobs/<job>/<task>__<id>) or use bf.load_job for a job directory"
        )
    data = _normalize_legacy(data)
    timing = _read_json(trial_dir / "timing.json")
    try:
        checkpoints = load_checkpoints(trial_dir)
    except Exception:  # an unreadable checkpoints.json must not hide the trial
        checkpoints = []
    return Trial(
        path=trial_dir,
        result=RolloutResult.from_dict(data, rollout_dir=trial_dir),
        raw=data,
        config=_read_json(trial_dir / "config.json") or {},
        timing=timing if isinstance(timing, dict) else (data.get("timing") or {}),
        verifier=VerifierOutput.read(trial_dir / "verifier"),
        forks=_forks(_read_json(trial_dir / "tree.json")),
        checkpoints=list(checkpoints or []),
    )


def _row_error(error: Any) -> tuple[str | None, str | None]:
    """(agent error, verifier error) from a results.jsonl ``error`` field.

    BenchFlow's rows label the error (``agent_error``, ``verifier_error``,
    ``export_error``); an export error only means the row is not
    training-ready, so it is not a run error. A plain string (Verifiers
    output) is an agent error.
    """
    if isinstance(error, str) and error:
        return error, None
    if not isinstance(error, dict):
        return None, None
    text = str(error.get("error_chain_str") or error.get("error") or "")
    kind = error.get("error")
    if kind == "agent_error":
        return text, None
    if kind == "verifier_error":
        return None, text
    return None, None


def load_results_jsonl(path: str | Path) -> list[Trial]:
    """Read every row of a ``results.jsonl`` file as a :class:`Trial`.

    For jobs that have no trial folders (a copied-out results file, or a
    Verifiers-style export). Rows carry less than a trial folder: no
    trajectory, verifier output, lineage or review. A BenchFlow row written
    before ``info.schema_version`` 2 held reward 0.0 for an unscored rollout;
    it is read as unscored (its ``metrics`` has no ``reward``), never as 0.
    """
    path = Path(path)
    trials = []
    unreadable: list[int] = []
    unnamed: list[int] = []
    rows = 0
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except ValueError:
            unreadable.append(index)
            continue
        if not isinstance(row, dict):
            unreadable.append(index)
            continue
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        usage = (
            row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
        )
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        reward = row.get("reward")
        if (
            info.get("source") == "benchflow"
            and "schema_version" not in info
            and reward == 0.0
            and "reward" not in metrics
        ):
            # A version-1 BenchFlow row wrote 0.0 for an unscored rollout; a
            # scored one also has its reward in metrics. Unscored is never 0.
            reward = None
        error, verifier_error = _row_error(row.get("error"))

        def _int(value: Any) -> int | None:
            return int(value) if isinstance(value, (int, float)) else None

        task_name = info.get("task_name") or info.get("task_id")
        if not task_name and row.get("example_id") is not None:
            task_name = str(row["example_id"])
        if not task_name:
            unnamed.append(index)
            continue
        data: dict[str, Any] = {
            "task_name": str(task_name),
            "rollout_name": info.get("rollout_name") or "",
            "rewards": {"reward": float(reward)}
            if isinstance(reward, (int, float)) and not isinstance(reward, bool)
            else None,
            "agent": info.get("agent") or "",
            "agent_name": info.get("agent_name") or "",
            "model": info.get("model"),
            "error": error,
            "verifier_error": verifier_error,
            "n_tool_calls": _int(
                metrics.get("n_tool_calls", row.get("total_tool_calls"))
            )
            or 0,
            "n_input_tokens": _int(usage.get("input_tokens")),
            "n_output_tokens": _int(usage.get("output_tokens")),
            "total_tokens": _int(usage.get("total_tokens")),
        }
        trials.append(
            Trial(
                path=path,
                result=RolloutResult.from_dict(data),
                raw={**data, "row": row},
                timing=row.get("timing") if isinstance(row.get("timing"), dict) else {},
                source="results.jsonl",
                row=index,
            )
        )
    if unreadable and len(unreadable) == rows:
        raise ValueError(
            f"{path} is not JSON Lines (no row parses as one JSON object); "
            "pass a job directory or a results.jsonl file"
        )
    if unreadable:
        logger.warning(
            "Skipped %d unreadable row(s) of %s (first: line %d)",
            len(unreadable),
            path,
            unreadable[0] + 1,
        )
    if unnamed:
        raise ValueError(
            f"{len(unnamed)} of {rows} rows in {path} have no task name "
            "(info.task_name, info.task_id or example_id); this is not a "
            "BenchFlow or Verifiers results file, so it is not read (first "
            f"unnamed row: line {unnamed[0] + 1})"
        )
    return trials


def _results_jsonl_files(root: Path) -> list[Path]:
    """The outermost results.jsonl files under ``root`` (job level wins)."""
    files = sorted(root.rglob("results.jsonl"), key=lambda p: len(p.parts))
    chosen: list[Path] = []
    for f in files:
        if not any(c.parent in f.parents for c in chosen):
            chosen.append(f)
    return sorted(chosen)


@dataclass(frozen=True)
class Denominators:
    """Counts behind a pass rate.

    ``attempted`` = ``scored`` + ``assessment_errors`` + ``unscored``;
    ``execution_errors`` counts errored and timed-out runs among the
    attempted (a timed-out run can still be scored). ``clean_*`` leave out
    scored runs whose execution failed. ``controls_excluded`` is how many
    control runs were left out. ``integration_failures`` counts runs whose
    agent integration broke (also in ``unscored`` and ``execution_errors``).
    """

    attempted: int = 0
    scored: int = 0
    assessment_errors: int = 0
    unscored: int = 0
    execution_errors: int = 0
    passed: int = 0
    mean_reward: float | None = None
    clean_scored: int = 0
    clean_passed: int = 0
    controls_excluded: int = 0
    # Attempted runs whose agent integration broke (unscored, never 0); they
    # are also in ``unscored`` and ``execution_errors``.
    integration_failures: int = 0

    @property
    def pass_rate_scored(self) -> float | None:
        """Passed over scored trials; None when none was scored."""
        return self.passed / self.scored if self.scored else None

    @property
    def pass_rate_attempted(self) -> float | None:
        """Passed over attempted trials; None when none was attempted."""
        return self.passed / self.attempted if self.attempted else None

    @property
    def pass_rate_clean(self) -> float | None:
        """Passed over scored trials whose execution completed; None when none."""
        return self.clean_passed / self.clean_scored if self.clean_scored else None

    @classmethod
    def of(cls, trials: Iterable[Trial], *, controls_excluded: int = 0) -> Denominators:
        """Count ``trials`` (see the class docstring for each count)."""
        items = list(trials)
        scored = [t for t in items if t.assessment == "scored"]
        rewards = [t.reward for t in scored if t.reward is not None]
        clean = [t for t in scored if t.execution == "completed"]
        return cls(
            attempted=len(items),
            scored=len(scored),
            assessment_errors=sum(t.assessment == "error" for t in items),
            unscored=sum(t.assessment == "unscored" for t in items),
            execution_errors=sum(t.execution != "completed" for t in items),
            passed=sum(r == 1 for r in rewards),
            mean_reward=math.fsum(rewards) / len(rewards) if rewards else None,
            clean_scored=len(clean),
            clean_passed=sum(t.reward == 1 for t in clean),
            controls_excluded=controls_excluded,
            integration_failures=sum(t.integration_failure is not None for t in items),
        )


@dataclass(repr=False)
class Job:
    """Every trial under a job directory (or any folder of trials).

    ``kind`` is ``"evaluation"`` (an ``Evaluation`` job), ``"branch"`` (a
    ``bench eval branch`` / ``bf.branch`` job) or ``"directory"``.
    """

    path: Path
    trials: list[Trial]
    summary: dict[str, Any] | None = field(default=None, repr=False)
    evaluation: dict[str, Any] | None = field(default=None, repr=False)
    # Attempt folders (config.json written) that never wrote result.json: a
    # crash, a kill or a cancelled run. Their sandbox.json, when present, names
    # a sandbox that may not have been deleted.
    interrupted: list[Path] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return f"Job(path={str(self.path)!r}, trials={len(self.trials)}, kind={self.kind!r})"

    def __str__(self) -> str:
        return self.to_markdown()

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def to_markdown(self) -> str:
        """A short report of the job, also what ``print(job)`` shows.

        What ran (trials, tasks, agents and models), the solve rate with its
        95% interval and pass@k, the trials that got no score grouped by
        reason, the control runs left out, and what the rollouts cost (retried
        attempts included; estimates from an agent's session log counted and
        marked).
        """
        agents = self.agents()
        tasks = {t.task_name for t in agents}
        pairs = sorted({f"{t.agent} · {t.model or 'no model'}" for t in agents})
        shown = ", ".join(pairs[:3]) + (
            f" and {len(pairs) - 3} more" if len(pairs) > 3 else ""
        )
        lines = [
            f"**{self.path}** ({self.kind}): {len(agents)} trial(s) of "
            f"{len(tasks)} task(s)" + (f"; {shown}" if shown else "")
        ]
        rates = self.solve_rates(ks=[1])
        if rates.solve_rate is None:
            lines.append("- Solve rate: n/a (no scored trial)")
        else:
            interval = ""
            if rates.interval is not None:
                low, high = rates.interval
                interval = (
                    f", 95% interval {low:.1%} to {high:.1%} ({rates.interval_method})"
                )
            solved = round(rates.solve_rate * rates.trials)
            lines.append(
                f"- Solve rate: {rates.solve_rate:.1%} ({solved} of {rates.trials} "
                f"scored trials, {rates.success_rule}){interval}"
            )
            if rates.max_trials_per_task > 1:
                many = self.solve_rates()
                lines += [f"- {line}" for line in many.lines()]
        unscored = [t for t in agents if t.assessment != "scored"]
        if unscored:
            reasons: dict[str, list[str]] = {}
            for t in unscored:
                reasons.setdefault(_unscored_reason(t), []).append(t.task_name)
            parts = []
            for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                listed = ", ".join(sorted(set(names))[:3])
                more = len(set(names)) - 3
                parts.append(
                    f"{reason} x{len(names)} ({listed}{f', +{more}' if more > 0 else ''})"
                )
            lines.append(
                f"- Unscored: {len(unscored)} of {len(agents)} trial(s): "
                + "; ".join(parts)
            )
        controls = self.controls()
        if controls:
            kinds = ", ".join(sorted({t.control or "" for t in controls}))
            lines.append(
                f"- Control runs left out: {len(controls)} ({kinds}); they check "
                "the task, not an agent"
            )
        lines.append("- " + _cost_line([a for t in self.trials for a in t.attempts]))
        return "\n".join(lines) + "\n"

    @property
    def kind(self) -> Literal["evaluation", "branch", "directory"]:
        """evaluation (an Evaluation job), branch (a branch job) or directory."""
        if (self.summary or {}).get("kind") == "benchflow-branch-job":
            return "branch"
        if self.evaluation is not None or self.summary is not None:
            return "evaluation"
        return "directory"

    def agents(self) -> list[Trial]:
        """Agent runs (control runs left out)."""
        return [t for t in self.trials if t.control is None]

    def controls(self) -> list[Trial]:
        """Control runs (oracle, empty/nop)."""
        return [t for t in self.trials if t.control is not None]

    def by_task(self, *, include_controls: bool = False) -> dict[str, list[Trial]]:
        """Trials grouped by task (control runs left out unless ``include_controls``)."""
        out: dict[str, list[Trial]] = {}
        for t in self.trials if include_controls else self.agents():
            out.setdefault(t.task_name, []).append(t)
        return out

    def denominators_by(
        self, by: Iterable[str] = ("agent", "model"), *, include_controls: bool = False
    ) -> list[GroupDenominators]:
        """Denominators per group, e.g. per agent and model (see :data:`GROUP_KEYS`).

        Control runs are left out unless ``include_controls=True`` (they then
        form their own groups). Sorted by key.
        """
        keys = tuple(by)
        _check_group_keys(keys)
        groups: dict[str, list[Trial]] = {}
        for t in self.trials if include_controls else self.agents():
            groups.setdefault(
                json.dumps([_group_value(t, k) for k in keys]), []
            ).append(t)
        return [
            GroupDenominators(
                key=dict(zip(keys, json.loads(k), strict=True)),
                denominators=Denominators.of(trials),
            )
            for k, trials in sorted(groups.items())
        ]

    def denominators(self, *, include_controls: bool = False) -> Denominators:
        """Attempted, scored, errors and pass rates (control runs left out unless ``include_controls``)."""
        if include_controls:
            return Denominators.of(self.trials)
        return Denominators.of(self.agents(), controls_excluded=len(self.controls()))

    def solve_rates(
        self,
        *,
        ks: Iterable[int] | None = None,
        solve_threshold: float | None = None,
        include_controls: bool = False,
    ) -> SolveRates:
        """pass@k, pass^k and the solve rate over this job's trials.

        A task's samples are its scored trials; trials of the same task in
        different job folders (``--matrix --trials`` writes one per trial)
        and repeated rollouts in one ``bf.run_batch`` folder are separate
        samples, retries inside one Evaluation job are one (the
        ``attempts="best"`` rule of :func:`load_job`). Unscored trials are
        left out of ``n``, not counted as failures; control runs are left out
        unless ``include_controls=True``. ``ks`` defaults to 1, powers of two
        and multiples of five up to the smallest per-task ``n``; a task with
        ``n < k`` is left out of that ``k`` and listed in the caveats.
        ``solve_threshold`` counts a scored trial as solved when its reward is
        at least the threshold (partial credit); the default rule is
        "passed". See :mod:`benchflow.pass_at_k`.
        """
        from benchflow.pass_at_k import solve_rates

        trials = self.trials if include_controls else self.agents()
        return solve_rates(
            (t.solve_sample() for t in trials),
            ks=ks,
            solve_threshold=solve_threshold,
            controls_excluded=0 if include_controls else len(self.controls()),
        )

    @property
    def cost_usd(self) -> float | None:
        """USD over the job's trials that reported one (one attempt each; ``trial.attempts`` has the rest); None when none did."""
        costs = [t.cost_usd for t in self.trials if t.cost_usd is not None]
        return math.fsum(costs) if costs else None

    def to_json_dict(
        self, *, include_trajectories: bool = False, include_verifier: bool = True
    ) -> dict[str, Any]:
        """The job as a ``benchflow.job`` document (see ``benchflow.job_export``).

        Trajectories are left out unless ``include_trajectories=True``; verifier
        output (test logs, often most of the size) is left out with
        ``include_verifier=False``. Validates against
        ``docs/reference/schemas/benchflow-job.v1.schema.json``.
        """
        from benchflow.job_export import job_export

        return job_export(
            self,
            include_trajectories=include_trajectories,
            include_verifier=include_verifier,
        ).model_dump(mode="json")

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        include_trajectories: bool = False,
        include_verifier: bool = True,
        indent: int = 2,
    ) -> Any:
        """:meth:`to_json_dict` as a JSON string, or written to ``path`` (returned)."""
        return _write_json(
            self.to_json_dict(
                include_trajectories=include_trajectories,
                include_verifier=include_verifier,
            ),
            path,
            indent,
        )

    def branch_views(self) -> list[dict[str, Any]]:
        """The ``benchflow.branch-view/1`` document of every trial that
        branched (see docs/reference/branch-view.md)."""
        return [trial.branch_view for trial in self.trials if trial.forks]

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per trial, sorted by task, each labelled with ``job``
        (this job's path, so rows from several jobs stay distinguishable)."""
        return [
            {"job": str(self.path), **t.to_record()}
            for t in sorted(self.trials, key=lambda t: (t.task_name, str(t.path)))
        ]

    def to_csv(self, path: str | Path) -> Path:
        """Write :meth:`to_records` as CSV (one row per trial) and return the path."""
        records = self.to_records()
        # The header is the records' own keys, so a new record field can never
        # make the writer refuse a row.
        fields: list[str] = []
        for record in records:
            fields.extend(k for k in record if k not in fields)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        return path

    def to_jsonl(self, path: str | Path) -> Path:
        """Write :meth:`to_records` as JSON Lines and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in self.to_records()))
        return path


def _unscored_reason(trial: Trial) -> str:
    """Why a trial got no score, for a summary line."""
    failure = trial.integration_failure
    if failure is not None:
        return f"agent integration ({failure.get('cause') or 'unknown'})"
    r = trial.result
    if r.error:
        return r.error_category or "agent error"
    if r.verifier_error:
        return f"verifier: {r.verifier_error_category or 'error'}"
    return "no reward"


def _cost_line(rollouts: list[Trial]) -> str:
    """What ``rollouts`` cost: known USD, how much of it is estimated, and
    how many reported none."""
    priced = [t for t in rollouts if t.cost_usd is not None]
    estimated = [t for t in priced if t.result.price_source == "agent_session_log"]
    unknown = len(rollouts) - len(priced)
    if not priced:
        return (
            f"Cost: unknown ({len(rollouts)} rollout(s) reported no USD; a "
            "subscription run has none unless its agent's session log prices it)"
        )
    usd = math.fsum(t.cost_usd or 0.0 for t in priced)
    line = f"Cost: ${usd:.4f} over {len(priced)} rollout(s)"
    if estimated:
        line += f" ({len(estimated)} estimated from the agent's session log)"
    if unknown:
        line += f"; {unknown} rollout(s) reported no USD, so the total is a floor"
    return line


def _rank(trial: Trial) -> tuple[bool, float, str]:
    from benchflow._utils.result_paths import attempt_rank

    return attempt_rank(trial.path / "result.json", scored=trial.assessment == "scored")


def _final_attempts(trials: list[Trial]) -> list[Trial]:
    """One trial per retried task of an Evaluation job; every other rollout.

    Attempts of one task, agent and model in an Evaluation job folder are one
    trial, whose result is the best attempt (:func:`_rank`). Rollouts in any
    other folder (``bf.run_batch``, ``bf.run`` calls sharing a job name) are
    independent samples and all kept.
    """
    from benchflow._utils.result_paths import holds_attempts

    folders: dict[Path, bool] = {}
    kept: list[Trial] = []
    chains: dict[tuple[str, Path, str, str | None], list[Trial]] = {}
    for t in trials:
        folder = t.path.parent
        if folder not in folders:
            folders[folder] = holds_attempts(folder)
        if not folders[folder]:
            kept.append(t)
            continue
        chains.setdefault((t.task_name, folder, t.agent, t.model), []).append(t)
    for chain in chains.values():
        chain.sort(key=_started)
        best = chain[0]
        for t in chain:
            t._attempts = chain
            if _rank(t) >= _rank(best):  # ties go to the newer attempt
                best = t
        kept.append(best)
    return kept


def _started(trial: Trial) -> tuple[str, float, str, int]:
    """Chronological order of attempts: start time, then when result.json was
    written (an attempt starts after the one before it ended), then the row
    of a results.jsonl file."""
    started = trial.result.started_at
    return (
        started.isoformat() if started is not None else "",
        _rank(trial)[1],
        str(trial.path),
        trial.row or 0,
    )


def load_job(
    path: str | Path | Iterable[str | Path],
    *,
    attempts: Literal["best", "all"] = "best",
) -> Job:
    """Read every trial under ``path`` into a :class:`Job`.

    ``path`` is a job directory, a folder of job directories, one trial
    directory, or a list of any of these (merged into one Job, e.g. one arm
    of a paired run kept in per-task folders). Branch children are read as the parent trial's ``forks``, not
    as extra trials, and reviewer runs are skipped.

    Retries: an Evaluation job (a folder with ``evaluation.json`` or
    ``summary.json``) runs each task once and retries it in the same folder,
    so there the attempts of one task, agent and model are one trial. With
    ``attempts="best"`` (the default) that trial is its best attempt: the
    scored one first, then the newest, the rule resume and ``summary.json``
    use; its :attr:`Trial.attempts` lists every attempt, oldest first.
    ``attempts="all"`` keeps every attempt as a trial. Rollouts in any other folder
    (``bf.run_batch``, ``bf.run`` calls sharing a job name) are independent
    samples and are always all kept, so repeated rollouts of one task count
    as repeated trials in :meth:`Job.solve_rates`.
    """
    import os

    from benchflow._utils.result_paths import iter_task_result_paths

    roots = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    if not roots:
        raise FileNotFoundError("load_job needs at least one path")
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix != ".jsonl":
            raise ValueError(
                f"{root} is a file, not a job; pass a job directory (such as "
                f"{root.parent}) or a results.jsonl file"
            )
        if root.is_file():  # a results.jsonl file, read below
            continue
        if (root / "result.json").is_file():
            paths.append(root / "result.json")
        else:
            paths.extend(iter_task_result_paths(root))
    path = roots[0] if len(roots) == 1 else Path(os.path.commonpath(roots))
    if path.is_file():
        path = path.parent
    trials = []
    for result_path in paths:
        try:
            trials.append(load_trial(result_path.parent))
        except (FileNotFoundError, ValueError, TypeError) as exc:
            logger.warning("Skipping unreadable trial %s: %s", result_path.parent, exc)
    if not trials:
        # No trial folders: read results.jsonl rows (a copied-out results
        # file or a Verifiers-style export).
        for root in roots:
            files = [root] if root.is_file() else _results_jsonl_files(root)
            for results_file in files:
                trials.extend(load_results_jsonl(results_file))
    if not trials:
        raise FileNotFoundError(
            f"no trial (a folder with result.json) under {path}; pass a job "
            "directory such as jobs/<job_name>"
        )
    if attempts not in ("best", "all"):
        raise ValueError(f"attempts is 'best' or 'all', got {attempts!r}")
    kept = _final_attempts(trials)
    if attempts == "best":
        trials = kept
    trials.sort(key=lambda t: (t.task_name, str(t.path)))
    return Job(
        path=path,
        trials=trials,
        summary=_read_json(path / "summary.json"),
        evaluation=_read_json(path / "evaluation.json"),
        interrupted=[p for root in roots if root.is_dir() for p in _interrupted(root)],
    )


def _interrupted(root: Path) -> list[Path]:
    """Attempt folders under ``root`` with a config.json but no result.json.

    Folders inside a trial (branch children, reviewer runs, evidence) and
    folders awaiting review (solver.json) are not attempts of their own.
    """
    trial_dirs = {p.parent for p in root.rglob("result.json")}
    trial_dirs.update(p.parent for p in root.rglob("solver.json"))
    found = []
    for config in sorted(root.rglob("config.json")):
        folder = config.parent
        if folder in trial_dirs or any(p in trial_dirs for p in folder.parents):
            continue
        if any(t.is_relative_to(folder) for t in trial_dirs):
            continue  # a job folder's own config, above its trials
        data = _read_json(config)
        if not isinstance(data, dict) or data.get("purpose") == "reviewer":
            continue
        if "agent" not in data:
            continue
        found.append(folder)
    return found


# The settings bf.compare checks per paired task to decide that two trials
# ran the same configuration (Trial.settings() reads them).
SETTINGS = (
    "task_digest",
    "model",
    "harness",
    "dataset_name",
    "dataset_version",
    "reasoning_effort",
    "environment",
    "sandbox_user",
    "timeout_sec",
    "agent_variable_names",
    "prompts_sha256",
)


@dataclass(frozen=True)
class SettingCheck:
    """One setting compared across a paired task's trials.

    ``match`` is True when every trial on both sides recorded the same value,
    False when they differ, None when neither side recorded it; ``a``/``b``
    list the distinct values (up to three) when they differ.
    """

    setting: str
    match: bool | None
    a: list[Any] | None = None
    b: list[Any] | None = None


@dataclass(frozen=True)
class SettingMismatch:
    """A setting that differs between the sides on one paired task."""

    task: str
    setting: str
    a: list[Any]
    b: list[Any]


# Keys trials can be grouped or paired by: identity fields plus the settings.
GROUP_KEYS = (
    "agent",
    "model",
    "control",
    *[k for k in SETTINGS if k not in ("model", "harness")],
)


def _check_group_keys(by: tuple[str, ...]) -> None:
    unknown = sorted(set(by) - set(GROUP_KEYS))
    if unknown:
        raise ValueError(
            f"unknown key(s) in by: {', '.join(unknown)}; "
            f"choose from {', '.join(GROUP_KEYS)}"
        )


def _group_value(trial: Trial, key: str) -> Any:
    if key in ("agent", "model", "control"):
        return getattr(trial, key)
    return trial.settings().get(key)


def _setting_checks(ta: list[Trial], tb: list[Trial]) -> list[SettingCheck]:
    def values(trials: list[Trial], name: str) -> set[str]:
        return {json.dumps(t.settings().get(name), sort_keys=True) for t in trials}

    checks = []
    for name in SETTINGS:
        va, vb = values(ta, name), values(tb, name)
        if va | vb == {"null"}:
            checks.append(SettingCheck(name, None))
            continue
        same = va == vb and len(va) == 1
        checks.append(
            SettingCheck(
                name,
                same,
                None if same else [json.loads(v) for v in sorted(va)][:3],
                None if same else [json.loads(v) for v in sorted(vb)][:3],
            )
        )
    return checks


@dataclass(frozen=True)
class ComparisonRow:
    """One task in a comparison; rewards are means over each side's scored trials."""

    task: str
    status: Literal["paired", "only_a", "only_b"]
    n_a: int
    n_b: int
    scored_a: int
    scored_b: int
    reward_a: float | None
    reward_b: float | None
    checks: list[SettingCheck] = field(default_factory=list, repr=False)
    # The by= values this row pairs on (empty when pairing by task only).
    group: dict[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> float | None:
        """reward_b - reward_a, or None unless both sides were scored."""
        if self.reward_a is None or self.reward_b is None:
            return None
        return self.reward_b - self.reward_a


@dataclass(frozen=True)
class ComparisonSummary:
    tasks: int
    paired: int
    only_a: int
    only_b: int
    both_scored: int
    same_reward: int
    b_higher: int
    b_lower: int
    mean_delta: float | None
    one_run_per_side: bool
    # Paired tasks with at least one undeclared setting difference.
    setting_mismatches: int = 0


@dataclass
class Comparison:
    """Two jobs paired by task name (``a`` and ``b`` are their denominators)."""

    job_a: Job
    job_b: Job
    rows: list[ComparisonRow]
    summary: ComparisonSummary
    a: Denominators
    b: Denominators
    caveats: list[str]
    labels: tuple[str, str] = ("A", "B")
    mismatches: list[SettingMismatch] = field(default_factory=list)
    vary: tuple[str, ...] = ()
    include_controls: bool = False
    solve_rates_a: SolveRates | None = None
    solve_rates_b: SolveRates | None = None
    # Denominators over the tasks both sides ran (what a headline should
    # compare when the task sets differ).
    a_paired: Denominators | None = None
    b_paired: Denominators | None = None
    # The extra pairing keys (``compare(..., by=...)``).
    by: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """The comparison as a ``benchflow.comparison`` document.

        Validates against
        ``docs/reference/schemas/benchflow-comparison.v1.schema.json``.
        """
        from benchflow.job_export import comparison_export

        return comparison_export(self).model_dump(mode="json")

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> Any:
        """:meth:`to_json_dict` as a JSON string, or written to ``path`` (returned)."""
        return _write_json(self.to_json_dict(), path, indent)

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per row, with its delta and the settings that differ."""
        out = []
        for r in self.rows:
            record = {k: v for k, v in r.__dict__.items() if k != "checks"}
            record["delta"] = r.delta
            record["settings_differ"] = sorted(
                m.setting for m in self.mismatches if m.task == r.task
            )
            out.append(record)
        return out

    def to_markdown(self) -> str:
        """A short report: denominators, the per-task table and the caveats."""

        def rate(d: Denominators) -> str:
            pr = d.pass_rate_scored
            return (
                f"{d.passed}/{d.scored} scored passed"
                + (f" ({pr:.0%})" if pr is not None else "")
                + f", {d.attempted} attempted, {d.assessment_errors} verifier errors, "
                f"{d.unscored} unscored, {d.controls_excluded} control runs left out"
            )

        def num(x: float | None) -> str:
            return "" if x is None else f"{x:g}"

        s = self.summary
        la, lb = self.labels
        paired_line = []
        if self.a_paired is not None and self.b_paired is not None:
            paired_line = [
                f"On the {s.paired} tasks both sides ran: {la} {rate(self.a_paired)}; "
                f"{lb} {rate(self.b_paired)}.",
                "",
                "All tasks each side ran:",
            ]
        lines = [
            *paired_line,
            f"{la}: {self.job_a.path} — {rate(self.a)}",
            f"{lb}: {self.job_b.path} — {rate(self.b)}",
            "",
            f"{s.paired} tasks paired ({s.both_scored} scored on both sides): {lb} higher "
            f"on {s.b_higher}, lower on {s.b_lower}, same on {s.same_reward}; mean delta "
            f"({lb} - {la}) {num(s.mean_delta) or 'n/a'}. {s.only_a} only in {la}, "
            f"{s.only_b} only in {lb}.",
            "",
            *(
                [
                    f"| Task | Group | {la} | {lb} | Delta | Status |",
                    "|---|---|---|---|---|---|",
                    *(
                        f"| {r.task} | {', '.join(f'{k}={v}' for k, v in r.group.items())} "
                        f"| {num(r.reward_a)} | {num(r.reward_b)} | {num(r.delta)} | {r.status} |"
                        for r in self.rows
                    ),
                ]
                if any(r.group for r in self.rows)
                else [
                    f"| Task | {la} | {lb} | Delta | Status |",
                    "|---|---|---|---|---|",
                    *(
                        f"| {r.task} | {num(r.reward_a)} | {num(r.reward_b)} | {num(r.delta)} | {r.status} |"
                        for r in self.rows
                    ),
                ]
            ),
            "",
            *(f"- {c}" for c in self.caveats),
        ]
        rates = [
            (label, r)
            for label, r in ((la, self.solve_rates_a), (lb, self.solve_rates_b))
            if r is not None
        ]
        if rates:
            lines += ["", f"Solve rates (success: {rates[0][1].success_rule}):"]
            for label, r in rates:
                lines += [f"- {label}: {line}" for line in r.lines()]
                lines += [f"  - {c}" for c in r.caveats]
        return "\n".join(lines) + "\n"


def _default_labels(a: Path, b: Path) -> tuple[str, str]:
    """Folder names, or their parents' when the names are the same or are
    run timestamps (jobs/claude-cli/2026-01-01__12-00-00 -> claude-cli)."""
    import re

    stamp = re.compile(r"^\d{4}-\d{2}-\d{2}[_T]")

    def own(p: Path) -> str:
        return p.parent.name if stamp.match(p.name) else p.name

    for la, lb in ((own(a), own(b)), (a.parent.name, b.parent.name)):
        if la != lb and not (stamp.match(la) or stamp.match(lb)):
            return la, lb
    return "A", "B"


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def compare(
    job_a: Job | str | Path,
    job_b: Job | str | Path,
    *,
    include_controls: bool = False,
    labels: tuple[str, str] | None = None,
    vary: Iterable[str] = (),
    on_mismatch: Literal["warn", "raise", "ignore"] = "warn",
    ks: Iterable[int] | None = None,
    solve_threshold: float | None = None,
    by: Iterable[str] = (),
) -> Comparison:
    """Pair two jobs' trials by task name and summarise the differences.

    ``by`` adds pairing keys (see :data:`GROUP_KEYS`), e.g.
    ``by=("agent", "model")`` when each side holds several agents: each row
    then pairs one agent and model's trials on one task with the other
    side's, instead of averaging over them. Without it, a side that mixes
    models or harnesses on a task is reported (warning and caveat), since its
    per-task reward is then an average over them.

    ``labels`` name the two sides in reports (default: the jobs' folder
    names, or their parents' when those are the same or are run timestamps,
    else "A"/"B"). Deltas are B - A.

    Each paired task's trials are checked for comparable settings
    (:data:`SETTINGS`: task digest, model, harness, dataset, reasoning
    effort, sandbox, sandbox user, timeout, agent variable names, prompts
    hash). ``vary`` names the settings the comparison is about (e.g.
    ``("model",)`` to compare two models; declaring ``harness`` or ``model``
    also covers ``agent_variable_names``); any other difference is a
    mismatch: a ``UserWarning`` by default, a ``ValueError`` with
    ``on_mismatch="raise"``, silent with ``"ignore"``. Mismatches are listed
    in ``Comparison.mismatches`` in every mode.

    Rewards per side are means over that side's scored trials; a task scored
    on one side only has no delta. Control runs are left out unless
    ``include_controls=True``. No significance test is computed.

    ``solve_rates_a`` / ``solve_rates_b`` hold each side's pass@k, pass^k
    and solve rate (:meth:`Job.solve_rates` with ``ks`` and
    ``solve_threshold``).
    """
    vary = tuple(vary)
    by = tuple(by)
    _check_group_keys(by)
    unknown = sorted(set(vary) - set(SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown setting(s) in vary: {', '.join(unknown)}; "
            f"choose from {', '.join(SETTINGS)}"
        )
    if on_mismatch not in ("warn", "raise", "ignore"):
        raise ValueError("on_mismatch is 'warn', 'raise' or 'ignore'")
    a = job_a if isinstance(job_a, Job) else load_job(job_a)
    b = job_b if isinstance(job_b, Job) else load_job(job_b)

    def keyed(job: Job) -> dict[tuple[Any, ...], list[Trial]]:
        out: dict[tuple[Any, ...], list[Trial]] = {}
        for task, trials in job.by_task(include_controls=include_controls).items():
            for t in trials:
                key = (task, *(json.dumps(_group_value(t, k)) for k in by))
                out.setdefault(key, []).append(t)
        return out

    left, right = keyed(a), keyed(b)
    paired_a: list[Trial] = []
    paired_b: list[Trial] = []
    rows = []
    for key in sorted(set(left) | set(right)):
        task = key[0]
        ta, tb = left.get(key, []), right.get(key, [])
        if ta and tb:
            paired_a.extend(ta)
            paired_b.extend(tb)
        sa = [t.reward for t in ta if t.assessment == "scored" and t.reward is not None]
        sb = [t.reward for t in tb if t.assessment == "scored" and t.reward is not None]
        rows.append(
            ComparisonRow(
                task=task,
                status="paired" if ta and tb else "only_a" if ta else "only_b",
                n_a=len(ta),
                n_b=len(tb),
                scored_a=len(sa),
                scored_b=len(sb),
                reward_a=_mean(sa),
                reward_b=_mean(sb),
                checks=_setting_checks(ta, tb) if ta and tb else [],
                group={k: json.loads(v) for k, v in zip(by, key[1:], strict=True)},
            )
        )
    # A side that holds several models or harnesses on one task: its reward
    # for the task is an average over them, which "differ" would misdescribe.
    mixed: list[tuple[str, str, str, list[Any]]] = []
    for r in rows:
        for c in r.checks:
            if c.setting in ("model", "harness") and c.match is False:
                for side, values in (("A", c.a or []), ("B", c.b or [])):
                    if len(values) > 1:
                        mixed.append((r.task, c.setting, side, values))
    mixed_tasks = {m[0] for m in mixed}
    # A harness or model sets its own agent variables (ANTHROPIC_MODEL, ...),
    # so declaring either also covers agent_variable_names.
    covered = set(vary) | (
        {"agent_variable_names"} if {"harness", "model"} & set(vary) else set()
    )
    mismatches = [
        SettingMismatch(r.task, c.setting, c.a or [], c.b or [])
        for r in rows
        for c in r.checks
        if c.match is False
        and c.setting not in covered
        # the same set of models on both sides is mixing, reported below
        and not (
            r.task in mixed_tasks and c.setting in ("model", "harness") and c.a == c.b
        )
    ]
    both = [r for r in rows if r.delta is not None]
    paired = [r for r in rows if r.status == "paired"]
    one_each = all(r.n_a == 1 and r.n_b == 1 for r in paired)
    summary = ComparisonSummary(
        tasks=len(rows),
        paired=len(paired),
        only_a=sum(r.status == "only_a" for r in rows),
        only_b=sum(r.status == "only_b" for r in rows),
        both_scored=len(both),
        same_reward=sum(r.delta == 0 for r in both),
        b_higher=sum((r.delta or 0) > 0 for r in both),
        b_lower=sum((r.delta or 0) < 0 for r in both),
        mean_delta=_mean([r.delta for r in both if r.delta is not None]),
        one_run_per_side=one_each,
        setting_mismatches=len({m.task for m in mismatches}),
    )
    caveats = [
        "Tasks are paired by recorded task name"
        + (f" and {', '.join(by)}." if by else "."),
        (
            "Each task has one run per side (n = 1): agent variance alone can flip a "
            "pass to a fail, so a reward difference is a lead to investigate, not an "
            "effect estimate."
        )
        if one_each
        else "Some tasks have several runs per side; per-side rewards are means over scored runs.",
        "No statistical significance is computed or implied.",
    ]
    if not include_controls:
        caveats.append("Control runs (oracle, empty/nop) are left out.")
    if mismatches:
        names = sorted({m.setting for m in mismatches})
        message = (
            f"Settings differ between the sides on {summary.setting_mismatches} "
            f"paired task(s): {', '.join(names)}"
        )
        example = ", ".join(
            f"{m.setting} on {m.task}: {m.a} vs {m.b}"
            for m in [next(x for x in mismatches if x.setting == n) for n in names]
        )
        if on_mismatch == "raise":
            raise ValueError(
                f"{message} ({example}). Pass vary=({', '.join(repr(n) for n in names)},) "
                "if the comparison is about them."
            )
        caveats.append(f"{message}; see Comparison.mismatches.")
        if on_mismatch == "warn":
            warnings.warn(
                f"bf.compare: {message} ({example}). Pass vary=(...) to declare an "
                "intended difference, or on_mismatch='raise' to refuse.",
                UserWarning,
                stacklevel=2,
            )
    elif vary:
        caveats.append(f"Declared differences: {', '.join(vary)}.")
    if mixed:
        task, setting, side, values = mixed[0]
        what = "models" if setting == "model" else "harnesses"
        mix_message = (
            f"A side mixes several {what} on {len(mixed_tasks)} task(s) (e.g. "
            f"{side} on {task}: {values}), so its reward there is an average over "
            "them; pass by=('agent', 'model') to pair like with like"
        )
        caveats.append(mix_message + ".")
        if on_mismatch == "warn":
            warnings.warn(f"bf.compare: {mix_message}.", UserWarning, stacklevel=2)
    return Comparison(
        job_a=a,
        job_b=b,
        rows=rows,
        summary=summary,
        a=a.denominators(include_controls=include_controls),
        b=b.denominators(include_controls=include_controls),
        caveats=caveats,
        labels=labels or _default_labels(a.path, b.path),
        mismatches=mismatches,
        vary=vary,
        include_controls=include_controls,
        solve_rates_a=a.solve_rates(
            ks=ks, solve_threshold=solve_threshold, include_controls=include_controls
        ),
        solve_rates_b=b.solve_rates(
            ks=ks, solve_threshold=solve_threshold, include_controls=include_controls
        ),
        a_paired=Denominators.of(paired_a),
        b_paired=Denominators.of(paired_b),
        by=by,
    )
