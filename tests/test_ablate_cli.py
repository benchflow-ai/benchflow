"""Regression tests for ``bench eval ablate`` and its ablation library.

Guards "feat(cli): bench eval ablate — stage-level ablation over branch
children" (docs/rollout-branching-rfc.md §5 / WS-4c; FrontierPhysics#73). PR
number to be added on submission.

The branch machinery underneath (stage snapshots, per-child deltas, the
skill-mode fresh-rollout child) already had unit coverage; what it had no
surface for was *running an experiment*. These tests pin that surface: arm
specs map onto exactly the deltas the engine executes, a request the engine
would reject fails closed before the parent run costs anything, the report is
deterministic, per-arm errors are isolated (the arms that ran keep their
rewards) and exit 1, and the attribution line stays an observation rather than
a causal claim.

Unit tests against fakes and a patched engine — no Docker, Daytona, or API
keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar, cast

import click
import pytest
import typer
from typer.testing import CliRunner

from benchflow.ablate import (
    ARM_KIND_INJECT,
    ARM_KIND_SKILL_MODE,
    AblationReport,
    AblationRequest,
    AblationSpecError,
    ArmOutcome,
    attribute,
    parse_arm,
    parse_arms,
    resolve_ablation_task,
    run_ablation,
    validate_arms_for_stage,
    write_ablation_report,
)
from benchflow.branch import UNSCORED_KEY
from benchflow.branch_delta import BranchDelta
from benchflow.cli.main import app
from benchflow.rollout_branch import CHILD_WALL_CLOCK_KEY
from benchflow.trajectories.tree import RolloutTree

runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse Rich's terminal wrapping so a message asserts as one sentence."""
    return re.sub(r"\s+", " ", text)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLI_MD = _REPO_ROOT / "docs" / "reference" / "cli.md"
_DOC_HEADER = "### bench eval ablate"


def _task_dir(tmp_path: Path, name: str = "task") -> Path:
    task = tmp_path / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (task / "instruction.md").write_text("solve it\n", encoding="utf-8")
    return task


def _report(
    *,
    arms: list[ArmOutcome],
    parent_reward: float | None = 1.0,
    value: float | None = 0.5,
    error: str | None = None,
) -> AblationReport:
    return AblationReport(
        task_id="demo",
        task_path="/tasks/demo",
        stage="env-ready",
        snapshot_layers=["sandbox"],
        agent="claude-agent-acp",
        model="claude-sonnet",
        sandbox="docker",
        arms=arms,
        parent_reward=parent_reward,
        parent_run_dir="/out/ablation/demo",
        value=value,
        error=error,
    )


def _skill_outcomes(with_skill: float, no_skill: float) -> list[ArmOutcome]:
    return [
        ArmOutcome(
            name="with-skill",
            kind=ARM_KIND_SKILL_MODE,
            delta=BranchDelta(skill_mode="with-skill").provenance_dict(),
            delta_execution="fresh-rollout",
            reward=with_skill,
            wall_clock_sec=61.0,
            node_id="n1",
            artifacts="/out/ablation/demo/branches/root/children/n1",
        ),
        ArmOutcome(
            name="no-skill",
            kind=ARM_KIND_SKILL_MODE,
            delta=BranchDelta(skill_mode="no-skill").provenance_dict(),
            delta_execution="fresh-rollout",
            reward=no_skill,
            wall_clock_sec=44.0,
            node_id="n2",
            artifacts="/out/ablation/demo/branches/root/children/n2",
        ),
    ]


def _patched_run(monkeypatch, report: AblationReport) -> list[AblationRequest]:
    """Patch the engine and capture the request the CLI built for it."""
    seen: list[AblationRequest] = []

    async def fake_run_ablation(request: AblationRequest) -> AblationReport:
        seen.append(request)
        return report

    monkeypatch.setattr("benchflow.ablate.run_ablation", fake_run_ablation)
    return seen


# 1. Arm specs -> BranchDelta (the library-level mapping)


def test_arm_specs_map_onto_the_deltas_the_engine_executes(tmp_path: Path) -> None:
    """Each v1 arm kind lowers to exactly one executable BranchDelta field.

    ``skill_mode`` and ``injected_prompt`` are the only two fields the branch
    engine executes (RFC §3.3); an arm that lowered to anything else would
    fail closed at fork time, after a full parent run.
    """
    plan = tmp_path / "oracle-plan.md"
    plan.write_text("1. read the spec\n2. patch the module\n", encoding="utf-8")

    arms = parse_arms(f"with-skill,no-skill,inject:{plan}")

    assert [arm.name for arm in arms] == ["with-skill", "no-skill", f"inject:{plan}"]
    assert [arm.kind for arm in arms] == [
        ARM_KIND_SKILL_MODE,
        ARM_KIND_SKILL_MODE,
        ARM_KIND_INJECT,
    ]
    assert arms[0].delta == BranchDelta(skill_mode="with-skill")
    assert arms[1].delta == BranchDelta(skill_mode="no-skill")
    assert arms[2].delta.injected_prompt == plan.read_text()
    assert arms[2].delta.skill_mode is None
    assert arms[2].source == str(plan)
    # The injected text is hash-only in provenance (#908) — never the raw plan.
    provenance = arms[2].delta.provenance_dict()
    assert provenance["injected_prompt_sha256"].startswith("sha256:")
    assert "read the spec" not in json.dumps(provenance)


def test_whitespace_around_arms_is_tolerated_but_an_empty_entry_is_not() -> None:
    """A dropped arm would publish a table with a missing comparison."""
    assert [arm.name for arm in parse_arms(" with-skill , no-skill ")] == [
        "with-skill",
        "no-skill",
    ]
    with pytest.raises(AblationSpecError, match="empty arm"):
        parse_arms("with-skill,,no-skill")


def test_single_and_duplicate_arm_requests_fail_closed() -> None:
    """A fork needs >= 2 children, and the report keys arms by name."""
    with pytest.raises(AblationSpecError, match=">= 2 children"):
        parse_arms("with-skill")
    with pytest.raises(AblationSpecError, match="duplicate arm"):
        parse_arms("no-skill,no-skill")


def test_injection_arm_needs_a_readable_non_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(AblationSpecError, match="cannot read its injection file"):
        parse_arm(f"inject:{tmp_path / 'missing.md'}")
    with pytest.raises(AblationSpecError, match="empty injection file"):
        parse_arm(f"inject:{empty}")
    with pytest.raises(AblationSpecError, match="names no file"):
        parse_arm("inject:")


def test_skill_arms_are_rejected_away_from_env_ready() -> None:
    """The engine's own gate (skills are deployed by install_agent()), applied
    before the ablation pays for a full parent run."""
    arms = parse_arms("with-skill,no-skill")
    assert validate_arms_for_stage(arms, "env-ready") == "env-ready"
    with pytest.raises(AblationSpecError, match="needs --at-stage 'env-ready'"):
        validate_arms_for_stage(arms, "pre-verify")


def test_injection_arms_run_at_env_ready_and_at_later_boundaries(
    tmp_path: Path,
) -> None:
    """An ``inject:<file>`` arm at ``env-ready`` is supported, not rejected.

    WS-4c flagged that combination as unsound because the child was run in
    place on a world where ``install_agent()`` had been rolled back. The
    resolution routes it through the same fresh-rollout path the skills arms
    use rather than banning it, so the pre-flight must keep accepting it —
    a regression here would silently turn a supported ablation into a
    spec error.
    """
    plan = tmp_path / "plan.md"
    plan.write_text("Follow this plan.", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("Follow that plan.", encoding="utf-8")
    arms = parse_arms(f"inject:{plan},no-skill")
    assert validate_arms_for_stage(arms, "env-ready") == "env-ready"
    inject_only = parse_arms(f"inject:{plan},inject:{other}")
    assert validate_arms_for_stage(inject_only, "env-ready") == "env-ready"
    assert validate_arms_for_stage(inject_only, "pre-verify") == "pre-verify"


def test_env_ready_ablation_requires_the_container_layer(tmp_path: Path) -> None:
    """Every arm of an ``env-ready`` ablation re-installs the agent for itself,
    so the stage snapshot has to carry the container layer — the mirror of the
    engine's own gate, paid before a full parent run instead of after it."""
    plan = tmp_path / "plan.md"
    plan.write_text("Follow this plan.", encoding="utf-8")
    arms = parse_arms(f"inject:{plan},no-skill")

    with pytest.raises(AblationSpecError, match="sandbox"):
        validate_arms_for_stage(
            arms, "env-ready", snapshot_layers=frozenset({"environment"})
        )
    assert (
        validate_arms_for_stage(
            arms, "env-ready", snapshot_layers=frozenset({"environment", "sandbox"})
        )
        == "env-ready"
    )
    # The command's own default is the container layer, so the CLI path is
    # never the one that trips this.
    assert AblationRequest(
        task_path=tmp_path, arms=arms, agent="claude-agent-acp"
    ).snapshot_layers == frozenset({"sandbox"})


def test_tasks_dir_must_resolve_to_exactly_one_task(tmp_path: Path) -> None:
    """An ablation's axis is the arms; several tasks is a request it cannot
    answer, not a batch to expand."""
    collection = tmp_path / "tasks"
    _task_dir(collection, "alpha")
    assert resolve_ablation_task(collection / "alpha") == collection / "alpha"
    assert resolve_ablation_task(collection) == collection / "alpha"
    _task_dir(collection, "beta")
    with pytest.raises(AblationSpecError, match="holds 2 tasks"):
        resolve_ablation_task(collection)


# 2. Attribution: verdicts are observations, not causal claims


def test_attribution_pairs_the_skill_arms_and_names_the_stage() -> None:
    outcomes = _skill_outcomes(with_skill=1.0, no_skill=0.0)
    attribute(outcomes, parent_reward=0.0, stage="env-ready")

    assert [arm.reference for arm in outcomes] == ["no-skill", "with-skill"]
    assert outcomes[0].verdict == (
        "passes (1.00) where no-skill fails (0.00) at env-ready — this delta "
        "decides the outcome when applied at env-ready (1 run per arm)"
    )
    assert outcomes[1].verdict.startswith("fails (0.00) where with-skill passes (1.00)")
    # No claim about boundaries this ablation never forked.
    assert "at or before" not in outcomes[1].verdict


def test_attribution_reports_no_difference_and_falls_back_to_the_parent(
    tmp_path: Path,
) -> None:
    both_pass = _skill_outcomes(with_skill=1.0, no_skill=1.0)
    attribute(both_pass, parent_reward=1.0, stage="env-ready")
    assert "no difference in this comparison" in both_pass[0].verdict

    injected = ArmOutcome(name="inject:plan.md", kind=ARM_KIND_INJECT, reward=1.0)
    attribute([injected], parent_reward=0.0, stage="pre-verify")
    assert injected.reference == "parent"
    assert "where parent fails (0.00) at pre-verify" in injected.verdict

    orphan = ArmOutcome(name="inject:plan.md", kind=ARM_KIND_INJECT, reward=0.5)
    attribute([orphan], parent_reward=None, stage="pre-verify")
    assert orphan.reference is None
    assert "no counterpart arm or parent reward" in orphan.verdict


def test_report_json_is_deterministic_and_carries_arm_provenance(
    tmp_path: Path,
) -> None:
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)

    first = write_ablation_report(report, tmp_path / "out").read_text()
    second = write_ablation_report(report, tmp_path / "out").read_text()

    assert first == second
    payload = json.loads(first)
    assert payload["task"]["id"] == "demo"
    assert payload["stage"] == "env-ready"
    assert payload["parent"]["reward"] == 1.0
    assert [arm["delta"]["skill_mode"] for arm in payload["arms"]] == [
        "with-skill",
        "no-skill",
    ]
    assert [arm["status"] for arm in payload["arms"]] == ["pass", "fail"]
    assert all(arm["delta_execution"] == "fresh-rollout" for arm in payload["arms"])
    assert report.has_errors is False


# 3. The engine path: arms map onto forked children, failures stay isolated


class _FakeRollout:
    """A Rollout stand-in that forks children the way the branch engine does.

    Mirrors the engine contract WS-4a/WS-4b pinned: one child node per delta,
    attached and run in order, each carrying its ``delta`` provenance, its
    ``delta_execution`` and its ``reward``; a child that raises leaves an
    attached node with no reward and no further children.
    """

    rewards: ClassVar[list[Any]] = [1.0, 0.0]
    parent_rewards: ClassVar[dict | None] = {"reward": 1.0}

    def __init__(self, config) -> None:
        self.config = config
        self.tree = RolloutTree()
        self.result = None
        self.calls: list[str] = []
        self._rollout_dir = (
            Path(config.jobs_dir) / str(config.job_name) / str(config.rollout_name)
        )

    async def setup(self) -> None:
        self.calls.append("setup")

    async def start(self) -> None:
        self.calls.append("start")

    async def install_agent(self) -> None:
        self.calls.append("install_agent")

    async def connect(self) -> None:
        self.calls.append("connect")

    async def execute(self) -> None:
        self.calls.append("execute")

    async def verify(self) -> dict | None:
        self.calls.append("verify")
        return self.parent_rewards

    async def cleanup(self) -> None:
        self.calls.append("cleanup")

    async def branch_at_stage(self, stage, n, *, deltas=None) -> float:
        self.calls.append(f"branch_at_stage:{stage}")
        assert deltas is not None and len(deltas) == n
        returns: list[float] = []
        for delta, outcome in zip(deltas, self.rewards, strict=True):
            child = self.tree.attach(self.tree.root)
            child.state["delta"] = delta.provenance_dict()
            if delta.skill_mode is not None:
                child.state["delta_execution"] = "fresh-rollout"
            child.state[CHILD_WALL_CLOCK_KEY] = 12.5
            if isinstance(outcome, Exception):
                raise outcome
            child.state["reward"] = float(outcome)
            returns.append(float(outcome))
        return sum(returns) / len(returns)


def _patch_rollout(monkeypatch, cls: type = _FakeRollout) -> list[Any]:
    """Patch the Rollout the ablation builds; return the instances it built."""
    built: list[Any] = []

    class _Capturing(cls):  # type: ignore[valid-type, misc]
        def __init__(self, config) -> None:
            super().__init__(config)
            built.append(self)

    monkeypatch.setattr("benchflow.rollout.Rollout", _Capturing)
    return built


def _request(tmp_path: Path, arms: str = "with-skill,no-skill") -> AblationRequest:
    return AblationRequest(
        task_path=_task_dir(tmp_path),
        arms=parse_arms(arms),
        agent="claude-agent-acp",
        model="claude-sonnet",
        out_dir=tmp_path / "out",
    )


async def test_run_ablation_forks_one_child_per_arm_and_reads_their_rewards(
    tmp_path: Path, monkeypatch
) -> None:
    built = _patch_rollout(monkeypatch)
    request = _request(tmp_path)

    report = await run_ablation(request)

    # Not Rollout.run(): the branch has to happen while the sandbox is still
    # up, so cleanup is last and the fork comes before it.
    assert built[0].calls == [
        "setup",
        "start",
        "install_agent",
        "connect",
        "execute",
        "verify",
        "branch_at_stage:env-ready",
        "cleanup",
    ]
    assert [arm.name for arm in report.arms] == ["with-skill", "no-skill"]
    assert [arm.reward for arm in report.arms] == [1.0, 0.0]
    assert [arm.status for arm in report.arms] == ["pass", "fail"]
    assert [arm.delta["skill_mode"] for arm in report.arms] == [
        "with-skill",
        "no-skill",
    ]
    assert all(arm.wall_clock_sec == 12.5 for arm in report.arms)
    assert report.arms[0].artifacts.endswith("branches/root/children/n1")
    assert report.parent_reward == 1.0
    assert report.value == 0.5
    assert report.has_errors is False
    assert "decides the outcome when applied at env-ready" in report.arms[0].verdict


async def test_run_ablation_cleans_up_and_isolates_a_failing_arm(
    tmp_path: Path, monkeypatch
) -> None:
    """A child failure propagates out of branch_at_stage (it is never scored
    0.0), so the arm that raised carries the error, the arms after it report as
    skipped, and the arm that already ran keeps its reward."""
    monkeypatch.setattr(_FakeRollout, "rewards", [1.0, RuntimeError("no snapshot")])
    built = _patch_rollout(monkeypatch)
    request = _request(tmp_path, "with-skill,no-skill")

    report = await run_ablation(request)

    assert built[0].calls[-1] == "cleanup"
    assert [arm.status for arm in report.arms] == ["pass", "error"]
    assert report.arms[0].reward == 1.0
    assert "no snapshot" in report.arms[1].error
    assert report.arms[1].verdict == "errored before scoring — no reward to attribute"
    assert report.value is None
    assert report.has_errors is True


class _UnscoredRollout(_FakeRollout):
    """A fork whose children ran but produced no verifier reward.

    The engine contract this mirrors is pinned directly in
    ``tests/test_branch_skill_delta.py`` and ``tests/test_rollout_branch.py``:
    a child whose ``verify()`` yielded nothing gets ``UNSCORED_KEY`` (the
    reason) on its node and *no* ``reward`` key, and the branch returns ``None``
    because there is nothing to average. This is the live-run failure that
    reported two confident ``0.00``s.
    """

    reason: ClassVar[str] = (
        "branch child regex-email-parser produced no verifier reward — "
        "No reward file found at /out/verifier/reward.txt or reward.json"
    )

    async def branch_at_stage(self, stage, n, *, deltas=None) -> float | None:
        self.calls.append(f"branch_at_stage:{stage}")
        assert deltas is not None and len(deltas) == n
        for delta in deltas:
            child = self.tree.attach(self.tree.root)
            child.state["delta"] = delta.provenance_dict()
            child.state["delta_execution"] = "fresh-rollout"
            child.state[CHILD_WALL_CLOCK_KEY] = 118.7
            child.state[UNSCORED_KEY] = self.reason
        return None


async def test_an_unscored_child_is_an_arm_error_never_a_fabricated_zero(
    tmp_path: Path, monkeypatch
) -> None:
    """The regression that made a real ablation lie.

    Both children ran and neither was scored. Every arm must report ``error``
    with the reason, carry a ``null`` reward in ``ablation.json``, claim
    nothing in its verdict (never "no difference" between two arms that were
    never scored), leave V undefined, and make the command exit non-zero.
    """
    _patch_rollout(monkeypatch, _UnscoredRollout)

    report = await run_ablation(_request(tmp_path))

    assert [arm.status for arm in report.arms] == ["error", "error"]
    assert [arm.reward for arm in report.arms] == [None, None]
    assert all("No reward file found" in arm.error for arm in report.arms)
    assert all("no difference" not in arm.verdict for arm in report.arms)
    assert all("0.00" not in arm.verdict for arm in report.arms)
    assert report.value is None
    assert report.has_errors is True

    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert [arm["reward"] for arm in payload["arms"]] == [None, None]
    assert [arm["status"] for arm in payload["arms"]] == ["error", "error"]


def test_cli_exits_1_and_prints_no_score_for_an_unscored_arm(
    tmp_path: Path, monkeypatch
) -> None:
    """The table must not show a score for an arm that was never scored."""
    arms = _skill_outcomes(with_skill=0.0, no_skill=0.0)
    for arm in arms:
        arm.reward = None
        arm.error = _UnscoredRollout.reason
    report = _report(arms=arms, value=None)
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    _patched_run(monkeypatch, report)

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(_task_dir(tmp_path)),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "produced no verifier reward" in _flat(result.stderr)
    assert "no difference" not in _flat(result.output)
    payload = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert [arm["reward"] for arm in payload["arms"]] == [None, None]


async def test_run_ablation_records_a_branch_that_never_forked(
    tmp_path: Path, monkeypatch
) -> None:
    """A stage that was never captured fails before any child attaches: the
    report carries the error and every arm reports as skipped rather than 0.0."""

    class _NoStage(_FakeRollout):
        async def branch_at_stage(self, stage, n, *, deltas=None) -> float:
            raise LookupError(f"no snapshot recorded at stage {stage!r}")

    _patch_rollout(monkeypatch, _NoStage)

    report = await run_ablation(_request(tmp_path))

    assert "no snapshot recorded at stage 'env-ready'" in report.error
    assert [arm.status for arm in report.arms] == ["skipped", "skipped"]
    assert report.arms[0].verdict == "not run — an earlier arm errored"
    assert report.has_errors is True


async def test_run_ablation_keeps_the_arms_when_the_parent_leg_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """Attributing a failed run is the point (RFC §1): a parent that died after
    the boundary is recorded, and the arms still fork from the snapshot."""

    class _ParentFails(_FakeRollout):
        async def execute(self) -> None:
            raise RuntimeError("agent session died")

    built = _patch_rollout(monkeypatch, _ParentFails)

    report = await run_ablation(_request(tmp_path))

    assert built[0].calls[-1] == "cleanup"
    assert "agent session died" in report.parent_error
    assert report.parent_reward is None
    assert [arm.reward for arm in report.arms] == [1.0, 0.0]
    assert report.has_errors is False


async def test_run_ablation_raises_when_the_boundary_is_never_reached(
    tmp_path: Path, monkeypatch
) -> None:
    class _StartFails(_FakeRollout):
        async def start(self) -> None:
            raise RuntimeError("sandbox never came up")

    _patch_rollout(monkeypatch, _StartFails)

    with pytest.raises(Exception, match="nothing to branch"):
        await run_ablation(_request(tmp_path))


# 4. The CLI surface


def test_bad_arm_stage_and_empty_arms_fail_without_a_traceback(
    tmp_path: Path,
) -> None:
    task = _task_dir(tmp_path)
    for flags, expected in (
        (["--arms", "with-skill,turbo"], "unknown arm 'turbo'"),
        (["--arms", ""], "--arms is empty"),
        (["--arms", "with-skill"], "at least two arms"),
        (["--at-stage", "mid-flight"], "unknown --at-stage 'mid-flight'"),
        (["--at-stage", "post-research"], "cannot be captured by this command"),
        (["--at-stage", "pre-verify"], "needs --at-stage 'env-ready'"),
    ):
        result = runner.invoke(
            app, ["eval", "ablate", "--tasks-dir", str(task), *flags]
        )
        assert result.exit_code == 1, result.output
        assert expected in _flat(result.stderr)
        assert "Traceback (most recent call last)" not in result.output
        # Nothing ran: no report directory was created for a rejected request.
        assert not (tmp_path / "out").exists()


def test_missing_tasks_dir_fails_closed(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["eval", "ablate", "--tasks-dir", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1, result.output
    assert "is not a directory" in _flat(result.stderr)
    assert "Traceback (most recent call last)" not in result.output


def test_table_renders_arms_rewards_and_the_attribution_line(
    tmp_path: Path, monkeypatch
) -> None:
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    seen = _patched_run(monkeypatch, report)
    task = _task_dir(tmp_path)
    # Rich wraps cells to the terminal width; widen it so the attribution line
    # is asserted as one sentence instead of as wrap-dependent fragments.
    monkeypatch.setenv("COLUMNS", "300")

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(task),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    out = result.output
    assert "with-skill" in out and "no-skill" in out
    assert "1.00" in out and "0.00" in out
    assert "pass" in out and "fail" in out
    assert "61s" in out and "44s" in out
    assert "decides the outcome when applied at env-ready" in _flat(out)
    assert (tmp_path / "out" / "ablation.json").is_file()
    # The CLI built the request from its flags — defaults included.
    assert [arm.name for arm in seen[0].arms] == ["with-skill", "no-skill"]
    assert seen[0].stage == "env-ready"
    assert seen[0].task_path == task


def test_json_output_parses_and_carries_the_arm_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    _patched_run(monkeypatch, report)
    task = _task_dir(tmp_path)

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(task),
            "--out-dir",
            str(tmp_path / "out"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["stage"] == "env-ready"
    assert payload["task"]["id"] == "demo"
    assert payload["parent"]["reward"] == 1.0
    assert payload["value"] == 0.5
    assert [arm["name"] for arm in payload["arms"]] == ["with-skill", "no-skill"]
    assert [arm["delta"]["skill_mode"] for arm in payload["arms"]] == [
        "with-skill",
        "no-skill",
    ]
    assert [arm["reward"] for arm in payload["arms"]] == [1.0, 0.0]
    assert payload["report_path"] == str(tmp_path / "out" / "ablation.json")
    # stdout stays machine-readable: the Rich table never lands on it.
    assert "─" not in result.output


def test_exit_code_1_when_an_arm_errors(tmp_path: Path, monkeypatch) -> None:
    arms = _skill_outcomes(with_skill=1.0, no_skill=0.0)
    arms[1].reward = None
    arms[1].error = "sandbox 'ModalSandbox' does not implement snapshot/restore"
    report = _report(arms=arms, value=None)
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    _patched_run(monkeypatch, report)
    task = _task_dir(tmp_path)

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(task),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "does not implement snapshot/restore" in _flat(result.stderr)
    assert "Traceback (most recent call last)" not in result.output
    # The report is still written — the arm that did run keeps its reward.
    payload = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert payload["arms"][0]["reward"] == 1.0
    assert payload["arms"][1]["status"] == "error"


def test_engine_errors_surface_as_one_line_not_a_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    async def fake_run_ablation(request: AblationRequest) -> AblationReport:
        raise AblationSpecError("bench eval ablate needs an ACP agent")

    monkeypatch.setattr("benchflow.ablate.run_ablation", fake_run_ablation)

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(_task_dir(tmp_path)),
            "--agent",
            "oracle",
        ],
    )

    assert result.exit_code == 1
    assert "needs an ACP agent" in _flat(result.stderr)
    assert "Traceback (most recent call last)" not in result.output


# 5. Docs: the flag table in cli.md must match the parser


def _doc_section(header: str) -> str:
    doc = _CLI_MD.read_text()
    index = doc.index(header)
    level = len(header) - len(header.lstrip("#"))
    match = re.search(rf"\n#{{2,{level}}} ", doc[index + len(header) :])
    end = index + len(header) + match.start() if match else len(doc)
    return doc[index:end]


def test_ablate_flags_and_cli_md_are_set_equal() -> None:
    """Same bidirectional guard tests/test_cli_docs_drift.py holds `bench eval
    run` to (#731), extended to this command: a new flag cannot land
    undocumented, and a documented flag cannot rot out of the parser."""
    command = cast("click.Group", typer.main.get_command(app)).commands["eval"]
    ablate = cast("click.Group", command).commands["ablate"]
    cli = {
        opt
        for param in ablate.params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--")
    } - {"--help"}
    doc = set(re.findall(r"`(--[a-z0-9-]+)`", _doc_section(_DOC_HEADER)))
    assert cli == doc, (
        "bench eval ablate CLI↔cli.md flag drift:\n"
        f"  in CLI but UNDOCUMENTED: {sorted(cli - doc)}\n"
        f"  documented but NOT in CLI: {sorted(doc - cli)}"
    )


def test_ablate_is_documented_next_to_the_rfc() -> None:
    """The command's docs must point at the design it implements, and the RFC's
    validation plan must name the command that produces its evidence."""
    section = _doc_section(_DOC_HEADER)
    assert "rollout-branching-rfc" in section
    rfc = (_REPO_ROOT / "docs" / "rollout-branching-rfc.md").read_text()
    assert "bench eval ablate" in rfc
