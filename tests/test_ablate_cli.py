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

Section 6 guards "feat(ablate): per-test attribution so a scalar tie cannot
hide a behavioral difference" — the measured case where both arms of a skills
ablation scored 0.00 while their two sub-tests flipped in opposite directions,
and the tool printed "no difference in this comparison".

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
    ARM_KIND_CONFIG,
    ARM_KIND_ENV,
    ARM_KIND_INJECT,
    ARM_KIND_SKILL_MODE,
    AblationReport,
    AblationRequest,
    AblationSpecError,
    ArmOutcome,
    attribute,
    differing_tests,
    parse_arm,
    parse_arms,
    resolve_ablation_task,
    run_ablation,
    sub_test_attribution,
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


def test_config_arm_specs_map_onto_config_override_deltas(tmp_path: Path) -> None:
    """``config:<inline-or-@file>`` lowers to the config_override delta.

    Guards "feat(branch): execute config_override deltas as fresh rollouts
    from env-ready": the arm parses through the same loader as the run-level
    override (inline JSON or an ``@file`` ref), its commas stay content when
    nested in JSON braces, and its provenance records the #790 sha + keys —
    never the raw patch.
    """
    inline = 'config:{"agent": {"timeout_sec": 7}, "metadata": {"x": 1}}'
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text('{"agent": {"timeout_sec": 9}}', encoding="utf-8")

    arms = parse_arms(f"with-skill,{inline},config:@{overlay_file}")

    assert [arm.kind for arm in arms] == [
        ARM_KIND_SKILL_MODE,
        ARM_KIND_CONFIG,
        ARM_KIND_CONFIG,
    ]
    assert arms[1].name == inline  # the spec survives the comma-aware split
    assert arms[1].delta.config_override == {
        "agent": {"timeout_sec": 7},
        "metadata": {"x": 1},
    }
    assert arms[2].delta.config_override == {"agent": {"timeout_sec": 9}}
    assert arms[2].source == str(overlay_file)
    provenance = arms[1].delta.provenance_dict()
    assert provenance["config_override_sha256"].startswith("sha256:")
    assert provenance["config_override_keys"] == ["agent", "metadata"]


def test_braces_and_commas_inside_quoted_json_strings_stay_content() -> None:
    """The arm splitter tracks JSON string state, not just brace depth.

    Guards the quote-aware ``_split_arm_specs`` fix for the review finding on
    PR #1046: braces, brackets and commas inside a quoted inline-JSON string
    value are content — ``{"metadata": {"note": "a{b,c]d"}}`` is one arm, and
    an escaped quote (``\\"``) does not end the string. A bare quote at depth
    zero (e.g. in an ``inject:`` path) keeps its historical literal meaning.
    """
    # The close-braces inside the string zero out a depth-only counter, so the
    # comma right after them split the spec mid-JSON before the fix.
    spooky = 'config:{"metadata": {"close": "}}", "note": "a{b,c]d", "quote": "say \\"hi\\", ok"}}'
    arms = parse_arms(f"no-skill,{spooky}")
    assert [arm.kind for arm in arms] == [ARM_KIND_SKILL_MODE, ARM_KIND_CONFIG]
    assert arms[1].delta.config_override == {
        "metadata": {"close": "}}", "note": "a{b,c]d", "quote": 'say "hi", ok'}
    }

    from benchflow.ablate import _split_arm_specs

    # A bare quote at depth zero never opens a string: the comma still splits.
    assert _split_arm_specs('inject:some"quoted.md,no-skill') == [
        'inject:some"quoted.md',
        "no-skill",
    ]


def test_malformed_config_arm_specs_fail_closed(tmp_path: Path) -> None:
    """An unparsable, empty, or scorer-touching config arm dies at parse time
    — before the parent run costs anything, exactly where the engine's own
    allowlist gate would kill the fork."""
    with pytest.raises(AblationSpecError, match="carries no patch"):
        parse_arm("config:")
    with pytest.raises(AblationSpecError, match="cannot load its config patch"):
        parse_arm(f"config:@{tmp_path / 'missing.yaml'}")
    with pytest.raises(AblationSpecError, match="empty patch"):
        parse_arm("config:{}")
    with pytest.raises(AblationSpecError, match="may only patch"):
        parse_arm('config:{"verifier": {"timeout_sec": 1}}')


def test_config_arms_are_rejected_away_from_env_ready() -> None:
    """The engine's own stage gate (the child's setup() applies the patch),
    applied before the ablation pays for a full parent run."""
    arms = parse_arms('no-skill,config:{"agent": {"timeout_sec": 7}}')
    assert validate_arms_for_stage(arms, "env-ready") == "env-ready"
    config_only = parse_arms(
        'config:{"agent": {"timeout_sec": 7}},config:{"agent": {"timeout_sec": 9}}'
    )
    with pytest.raises(AblationSpecError, match="needs --at-stage 'env-ready'"):
        validate_arms_for_stage(config_only, "pre-verify")


def test_env_arm_specs_map_onto_environment_ref_deltas() -> None:
    """``env:<registry-ref>`` lowers to the environment_ref delta verbatim.

    Guards "feat(branch): execute service-level environment_ref deltas from
    env-ready": the ref is recorded exactly as written (the registry
    content-addresses it at resolution), an empty ref fails at parse time,
    and the arm is env-ready-only like every fresh-child delta.
    """
    arms = parse_arms("no-skill,env:env0@outage")
    assert arms[1].kind == ARM_KIND_ENV
    assert arms[1].delta == BranchDelta(environment_ref="env0@outage")
    assert arms[1].delta.provenance_dict()["environment_ref"] == "env0@outage"
    with pytest.raises(AblationSpecError, match="names no environment"):
        parse_arm("env:")
    env_only = parse_arms("env:env0@prod,env:env0@outage")
    assert validate_arms_for_stage(env_only, "env-ready") == "env-ready"
    with pytest.raises(AblationSpecError, match="needs --at-stage 'env-ready'"):
        validate_arms_for_stage(env_only, "post-verify")


async def test_an_env_arm_against_a_manifest_less_task_fails_before_the_parent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The env arm's content gates need the parent's manifest, so they run in
    run_ablation — but still *before* the parent run: a task that declares no
    environment has no plane to swap, and the request dies without building a
    single rollout."""
    built = _patch_rollout(monkeypatch)
    request = _request(tmp_path, "no-skill,env:env0@outage")

    with pytest.raises(AblationSpecError, match="no environment manifest"):
        await run_ablation(request)

    assert built == []  # nothing ran, nothing was paid for


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


async def test_the_ablation_parent_states_the_no_skill_mode_the_gate_requires(
    tmp_path: Path, monkeypatch
) -> None:
    """The parent an ablation runs is ``no-skill`` by statement, not by default.

    Guards the fix from "fix(branch): gate skill deltas on the parent's own
    skill mode". The arms fork the parent's own ``env-ready`` image, and a
    ``with-skill`` parent bakes its pack into that image — a ``no-skill`` arm
    would restore the pack and still be labelled ``no-skill``. The branch
    engine now refuses that fork, so this command was correct only for as long
    as ``RolloutConfig.from_legacy`` happened to default ``skill_mode`` to
    ``no-skill``. It says so itself now, and a change to that default cannot
    quietly move every ablation to the wrong side of the gate.
    """
    built = _patch_rollout(monkeypatch)

    await run_ablation(_request(tmp_path))

    assert built[0].config.skill_mode == "no-skill"
    assert built[0].config.recorded_skill_mode == "no-skill"


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
        (["--arms", "with-skill,config:"], "carries no patch"),
        (
            ["--arms", 'with-skill,config:{"verifier": {"timeout_sec": 1}}'],
            "may only patch",
        ),
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


# 6. Per-test (sub-outcome) attribution: a scalar tie cannot hide a difference


def _ctrf(tests: dict[str, str]) -> str:
    """A CTRF report (pytest-json-ctrf shape) naming ``{node id -> status}``."""
    return json.dumps(
        {
            "reportFormat": "CTRF",
            "results": {
                "tool": {"name": "pytest"},
                "tests": [
                    {"name": name, "status": status} for name, status in tests.items()
                ],
            },
        }
    )


def _tested_outcomes(
    with_skill: dict[str, str] | None, no_skill: dict[str, str] | None
) -> list[ArmOutcome]:
    """The measured lake-warming pair: both arms 0.00, per-test maps attached."""
    arms = _skill_outcomes(with_skill=0.0, no_skill=0.0)
    arms[0].tests = with_skill
    arms[1].tests = no_skill
    return arms


class _CtrfRollout(_FakeRollout):
    """A fork whose children leave a CTRF report where their kind leaves one.

    A **fresh-rollout** child (every child of ``env-ready``) has a run
    directory of its own, and that run directory *is* its artifact directory,
    so ``<child>/verifier/ctrf.json`` is where its verifier writes. An
    **in-place** child (``pre-verify`` / ``post-verify``) has no run directory:
    it writes through the parent's bind mounts, and the branch engine archives
    what it wrote under ``<child>/mounted/verifier/`` — the layout
    :mod:`benchflow.branch_artifacts` owns, which is why both paths are asked
    for here rather than spelled out.

    ``per_child`` is indexed by fork order; ``None`` writes no report at all
    (the verifier emitted no per-test data).
    """

    rewards: ClassVar[list[Any]] = [0.0, 0.0]
    per_child: ClassVar[list[dict[str, str] | None]] = [
        {"::T::test_trend_result": "passed", "::T::test_dominant_factor": "failed"},
        {"::T::test_trend_result": "failed", "::T::test_dominant_factor": "passed"},
    ]
    #: ``True`` runs the children the way a ``pre-verify`` fork does — in
    #: place, with their output preserved under ``mounted/``.
    in_place: ClassVar[bool] = False

    def _child_verifier_dir(self, child) -> Path:
        from benchflow.branch_artifacts import child_mount_dir
        from benchflow.branch_lineage import branch_child_dir

        if self.in_place:
            return (
                child_mount_dir(self._rollout_dir, child.parent.id, child.id)
                / "verifier"
            )
        return (
            branch_child_dir(self._rollout_dir, child.parent.id, child.id) / "verifier"
        )

    async def branch_at_stage(self, stage, n, *, deltas=None):
        value = await super().branch_at_stage(stage, n, deltas=deltas)
        children = [node for node in self.tree.nodes() if "delta" in node.state]
        for child, tests in zip(children, self.per_child, strict=True):
            if tests is None:
                continue
            verifier = self._child_verifier_dir(child)
            verifier.mkdir(parents=True, exist_ok=True)
            (verifier / "ctrf.json").write_text(_ctrf(tests), encoding="utf-8")
        return value


class _InPlaceCtrfRollout(_CtrfRollout):
    """The ``pre-verify`` fork: children run in place, artifacts under ``mounted/``."""

    in_place: ClassVar[bool] = True


async def test_per_test_outcomes_are_mined_from_each_child_verifier_report(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards "feat(ablate): per-test attribution so a scalar tie cannot hide a
    behavioral difference": every arm carries the outcome map its own verifier
    reported, read through the CLI's existing CTRF parser (no second parser,
    and the ``ClassName::test`` node ids are displayed by test name)."""
    _patch_rollout(monkeypatch, _CtrfRollout)

    report = await run_ablation(_request(tmp_path))

    assert [arm.reward for arm in report.arms] == [0.0, 0.0]
    assert report.arms[0].tests == {
        "test_dominant_factor": "failed",
        "test_trend_result": "passed",
    }
    assert report.arms[1].tests == {
        "test_dominant_factor": "passed",
        "test_trend_result": "failed",
    }
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert [arm["tests"] for arm in payload["arms"]] == [
        {"test_dominant_factor": "failed", "test_trend_result": "passed"},
        {"test_dominant_factor": "passed", "test_trend_result": "failed"},
    ]


def _in_place_request(tmp_path: Path) -> AblationRequest:
    """A ``pre-verify`` ablation: two injection arms, children run in place."""
    first = tmp_path / "plan-a.md"
    first.write_text("Warm the lake first.\n", encoding="utf-8")
    second = tmp_path / "plan-b.md"
    second.write_text("Cool the lake first.\n", encoding="utf-8")
    return AblationRequest(
        task_path=_task_dir(tmp_path),
        arms=parse_arms(f"inject:{first},inject:{second}"),
        agent="claude-agent-acp",
        model="claude-sonnet",
        stage="pre-verify",
        out_dir=tmp_path / "out",
    )


async def test_per_test_outcomes_are_mined_from_an_in_place_childs_mounted_output(
    tmp_path: Path, monkeypatch
) -> None:
    """An in-place branch child keeps its per-test attribution.

    Guards the fix from "fix(ablate): read per-test outcomes from preserved
    in-place child artifacts", reproduced live by @Galius5136 at ``--at-stage
    pre-verify``. A child that runs in place has no rollout directory of its
    own — its verifier output is the ``mounted/`` archive of what it wrote to
    the parent's shared bind mounts (the artifact isolation from "fix(branch):
    branch children no longer clobber the parent's artifacts"). The reader
    looked only at ``<child>/verifier/ctrf.json``, so the report fell back to
    scalar-only attribution while the per-test data sat one directory down.
    """
    _patch_rollout(monkeypatch, _InPlaceCtrfRollout)

    report = await run_ablation(_in_place_request(tmp_path))

    assert [arm.reward for arm in report.arms] == [0.0, 0.0]
    assert report.arms[0].tests == {
        "test_dominant_factor": "failed",
        "test_trend_result": "passed",
    }
    assert report.arms[1].tests == {
        "test_dominant_factor": "passed",
        "test_trend_result": "failed",
    }
    section = sub_test_attribution(report.arms)
    assert section["arms_without_tests"] == []
    assert section["summary"] == (
        "scalar rewards tie, but 2 sub-test outcome(s) differ: "
        "test_dominant_factor, test_trend_result"
    )


async def test_a_fresh_childs_own_verifier_output_wins_over_its_mounted_copy(
    tmp_path: Path, monkeypatch
) -> None:
    """Both roots can exist; the child's own run directory is authoritative.

    Guards the ordering in "fix(ablate): read per-test outcomes from preserved
    in-place child artifacts". A fresh-rollout child writes through the
    restored parent mounts *and* downloads its artifacts into its own run
    directory, so the ``mounted/`` archive can hold a second, staler copy —
    reading that one instead would attribute the wrong outcomes to the arm.
    """
    from benchflow.branch_artifacts import child_mount_dir

    class _BothRoots(_CtrfRollout):
        async def branch_at_stage(self, stage, n, *, deltas=None):
            value = await super().branch_at_stage(stage, n, deltas=deltas)
            for child in (node for node in self.tree.nodes() if "delta" in node.state):
                mounted = (
                    child_mount_dir(self._rollout_dir, child.parent.id, child.id)
                    / "verifier"
                )
                mounted.mkdir(parents=True, exist_ok=True)
                (mounted / "ctrf.json").write_text(
                    _ctrf({"::T::test_stale": "failed"}), encoding="utf-8"
                )
            return value

    _patch_rollout(monkeypatch, _BothRoots)

    report = await run_ablation(_request(tmp_path))

    assert [sorted(arm.tests or {}) for arm in report.arms] == [
        ["test_dominant_factor", "test_trend_result"],
        ["test_dominant_factor", "test_trend_result"],
    ]


def test_only_the_tests_whose_outcome_differs_are_attributed() -> None:
    """Tying tests are noise for attribution: they are counted, never listed.

    Guards the differences table of "feat(ablate): per-test attribution …" —
    a table that reprinted every test would bury the two rows that carry the
    signal.
    """
    arms = _tested_outcomes(
        {"test_a": "passed", "test_b": "failed", "test_same": "passed"},
        {"test_a": "failed", "test_b": "passed", "test_same": "passed"},
    )

    differences = differing_tests(arms)
    section = sub_test_attribution(arms)

    assert differences == [
        {"test": "test_a", "outcomes": {"no-skill": "failed", "with-skill": "passed"}},
        {"test": "test_b", "outcomes": {"no-skill": "passed", "with-skill": "failed"}},
    ]
    assert section["tying_tests"] == ["test_same"]
    assert section["arms_with_tests"] == ["with-skill", "no-skill"]
    assert section["arms_without_tests"] == []
    assert section["scalar_tie"] is True


def test_a_test_reported_for_one_arm_only_is_a_difference_not_a_failure() -> None:
    """A test the other arm never reported is recorded as ``None``.

    Guards "feat(ablate): per-test attribution …" against inventing rows: the
    honest reading of a missing entry is "not observed for that arm", never
    "failed there".
    """
    arms = _tested_outcomes(
        {"test_a": "passed"}, {"test_a": "passed", "test_b": "failed"}
    )

    assert differing_tests(arms) == [
        {"test": "test_b", "outcomes": {"no-skill": "failed", "with-skill": None}}
    ]


def test_scalar_tie_with_differing_sub_tests_says_so_in_the_verdict() -> None:
    """The exact regression: two arms at 0.00 whose sub-tests flip oppositely.

    Guards "feat(ablate): per-test attribution so a scalar tie cannot hide a
    behavioral difference". The measured lake-warming ablation printed "no
    difference in this comparison" while ``test_trend_result`` flipped with the
    skill pack and ``test_dominant_factor`` flipped against it. That wording
    must be unreachable whenever a sub-test difference was observed.
    """
    arms = _tested_outcomes(
        {"test_trend_result": "passed", "test_dominant_factor": "failed"},
        {"test_trend_result": "failed", "test_dominant_factor": "passed"},
    )

    attribute(arms, parent_reward=0.0, stage="env-ready")

    assert arms[0].verdict == (
        "matches no-skill at env-ready (both 0.00) — scalar rewards tie, but 2 "
        "sub-test outcome(s) differ: test_dominant_factor, test_trend_result"
    )
    assert all("no difference in this comparison" not in a.verdict for a in arms)
    assert sub_test_attribution(arms)["summary"] == (
        "scalar rewards tie, but 2 sub-test outcome(s) differ: "
        "test_dominant_factor, test_trend_result"
    )


def test_a_genuine_tie_keeps_todays_no_difference_wording() -> None:
    """Sub-tests that were observed *and* tie are not a difference.

    The other half of "feat(ablate): per-test attribution …": the qualification
    fires on measured disagreement only, so an ablation where nothing moved
    still reads as it always has.
    """
    arms = _tested_outcomes({"test_a": "failed"}, {"test_a": "failed"})

    attribute(arms, parent_reward=0.0, stage="env-ready")

    assert arms[0].verdict == (
        "matches no-skill at env-ready (both 0.00) — no difference in this comparison"
    )
    assert sub_test_attribution(arms)["summary"] == (
        "2 arms reported per-test outcomes and all 1 tie — no sub-test "
        "difference in this comparison"
    )


async def test_a_child_without_per_test_data_degrades_to_scalar_only(
    tmp_path: Path, monkeypatch
) -> None:
    """A verifier that emits no CTRF report yields no rows, and says so.

    Guards the degradation path of "feat(ablate): per-test attribution …":
    ``tests`` stays ``None`` (not ``{}``, not a fabricated row), the section
    names the arms it could not read, and the scalar verdict is left alone —
    nothing was measured to qualify it with.
    """
    monkeypatch.setattr(_CtrfRollout, "per_child", [{"::T::test_a": "passed"}, None])
    _patch_rollout(monkeypatch, _CtrfRollout)

    report = await run_ablation(_request(tmp_path))

    assert report.arms[0].tests == {"test_a": "passed"}
    assert report.arms[1].tests is None
    section = sub_test_attribution(report.arms)
    assert section["arms_with_tests"] == ["with-skill"]
    assert section["arms_without_tests"] == ["no-skill"]
    assert section["differing_tests"] == []
    assert section["summary"] == (
        "scalar-only attribution — 1 of 2 arms reported per-test outcomes, so "
        "no sub-test comparison was made"
    )
    assert "no difference in this comparison" in report.arms[0].verdict


def test_test_attribution_is_deterministic_and_carries_no_wall_clock(
    tmp_path: Path,
) -> None:
    """Same input, byte-identical report — the determinism guarantee of
    "feat(ablate): per-test attribution …" extended to the new section: test
    names sort, arm order follows the request, and nothing is stamped."""
    arms = _tested_outcomes(
        {"test_z": "failed", "test_a": "passed"},
        {"test_a": "failed", "test_z": "passed"},
    )
    report = _report(arms=arms, parent_reward=0.0, value=0.0)
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)

    first = write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    second = write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")

    assert first == second
    payload = json.loads(first)
    assert [
        entry["test"] for entry in payload["test_attribution"]["differing_tests"]
    ] == [
        "test_a",
        "test_z",
    ]
    assert list(payload["arms"][0]["tests"]) == ["test_a", "test_z"]


def test_table_lists_the_differing_sub_tests_and_omits_the_tying_ones(
    tmp_path: Path, monkeypatch
) -> None:
    """The console half of "feat(ablate): per-test attribution …".

    A reader of the table must be able to see the behavioral difference the
    two 0.00s hide, and must not have to scan tying rows to find it.
    """
    arms = _tested_outcomes(
        {
            "test_trend_result": "passed",
            "test_dominant_factor": "failed",
            "test_boring": "passed",
        },
        {
            "test_trend_result": "failed",
            "test_dominant_factor": "passed",
            "test_boring": "passed",
        },
    )
    report = _report(arms=arms, parent_reward=0.0, value=0.0)
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    _patched_run(monkeypatch, report)
    monkeypatch.setenv("COLUMNS", "300")

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

    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "Sub-test outcomes that differ (2)" in out
    assert "test_trend_result" in out and "test_dominant_factor" in out
    assert "test_boring" not in out
    assert "1 test(s) tie across the arms and are omitted." in out
    assert "scalar rewards tie, but 2 sub-test outcome(s) differ" in out
    assert "no difference in this comparison" not in out


def test_json_output_carries_the_per_test_maps_and_the_difference_section(
    tmp_path: Path, monkeypatch
) -> None:
    """``--json`` stays machine-readable and complete for "feat(ablate):
    per-test attribution …": the per-arm maps and the differences section both
    round-trip through stdout."""
    arms = _tested_outcomes(
        {"test_trend_result": "passed", "test_dominant_factor": "failed"},
        {"test_trend_result": "failed", "test_dominant_factor": "passed"},
    )
    report = _report(arms=arms, parent_reward=0.0, value=0.0)
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
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["arms"][0]["tests"] == {
        "test_dominant_factor": "failed",
        "test_trend_result": "passed",
    }
    section = payload["test_attribution"]
    assert section["scalar_tie"] is True
    assert [entry["test"] for entry in section["differing_tests"]] == [
        "test_dominant_factor",
        "test_trend_result",
    ]
    assert section["differing_tests"][0]["outcomes"] == {
        "no-skill": "passed",
        "with-skill": "failed",
    }
    assert "─" not in result.output


# 7. The task's declared environment reaches the parent and every arm


_ABLATE_MANIFEST = """\
environment:
  name: ablate-declared-env
  image: example/ablate-declared:latest
"""


def _manifest_task_dir(
    tmp_path: Path,
    *,
    declares: str = "environment.yaml",
    write_manifest: bool = True,
) -> Path:
    """A task whose ``task.md`` declares ``benchflow.environment.manifest``."""
    task = _task_dir(tmp_path)
    (task / "task.md").write_text(
        "---\n"
        'schema_version: "1.3"\n'
        "task:\n"
        "  name: benchflow/ablate-manifest-demo\n"
        "  description: A task that declares its own environment\n"
        "benchflow:\n"
        "  environment:\n"
        f"    manifest: {declares}\n"
        "---\n"
        "Solve it.\n",
        encoding="utf-8",
    )
    if write_manifest:
        (task / "environment.yaml").write_text(_ABLATE_MANIFEST, encoding="utf-8")
    return task


async def test_the_task_declared_environment_reaches_the_parent_and_the_arms(
    tmp_path: Path, monkeypatch
) -> None:
    """A manifest-backed task is ablated in the world it declares.

    Guards "fix(ablate): resolve a task-declared environment manifest for
    parent and arms". ``bench eval`` resolves ``benchflow.environment.manifest``
    per task before building the rollout config; ``bench eval ablate`` left
    ``environment_manifest=None``, so the parent — and therefore every arm
    forked from its snapshot — ran without the task's required image,
    services, provisioning and readiness gates, and the report compared a
    different environment than a normal evaluation of the same task.

    Both halves are asserted: the parent config the command builds, and the
    child config the branch engine derives from it for a fresh ``env-ready``
    child (:func:`benchflow.branch_skill.make_fresh_child_runner`, the
    production path for every arm at that boundary).
    """
    from types import SimpleNamespace

    import benchflow.branch_skill as branch_skill
    from benchflow.branch_skill import make_fresh_child_runner

    task = _manifest_task_dir(tmp_path)
    built = _patch_rollout(monkeypatch)
    request = AblationRequest(
        task_path=task,
        arms=parse_arms("with-skill,no-skill"),
        agent="claude-agent-acp",
        model="claude-sonnet",
        out_dir=tmp_path / "out",
    )

    await run_ablation(request)

    manifest = built[0].config.environment_manifest
    assert manifest is not None
    assert (manifest.name, manifest.image) == (
        "ablate-declared-env",
        "example/ablate-declared:latest",
    )

    child_configs: list[Any] = []

    async def _capture_fresh_child(rollout, config, *, prompts=None):
        child_configs.append(config)
        return 1.0

    monkeypatch.setattr(branch_skill, "run_fresh_child", _capture_fresh_child)
    tree = RolloutTree()
    child = tree.attach(tree.root)
    for arm in request.arms:
        run_child = make_fresh_child_runner(
            cast(Any, SimpleNamespace(_config=built[0].config)),
            delta=arm.delta,
            parent=tree.root,
            branch_stage="env-ready",
            run_dir=built[0]._rollout_dir,
        )
        await run_child(child)

    assert [cfg.environment_manifest for cfg in child_configs] == [manifest, manifest]


async def test_a_task_without_a_declared_environment_still_binds_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """The control for the fix: no declaration, no manifest — nothing invented.

    Guards "fix(ablate): resolve a task-declared environment manifest for
    parent and arms" from over-reaching: a task that declares no environment
    must keep running on the plain sandbox, exactly as before.
    """
    built = _patch_rollout(monkeypatch)

    await run_ablation(_request(tmp_path))

    assert built[0].config.environment_manifest is None


async def test_an_unresolvable_declared_environment_fails_before_the_parent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """A declaration that cannot be resolved is fatal, not silently dropped.

    Guards "fix(ablate): resolve a task-declared environment manifest for
    parent and arms". Degrading to ``None`` would run the whole experiment —
    parent and arms — in a world the task says is wrong and report it as if it
    were the right one, so the request fails closed before the parent run costs
    anything.
    """
    task = _manifest_task_dir(tmp_path, declares="missing.yaml", write_manifest=False)
    built = _patch_rollout(monkeypatch)
    request = AblationRequest(
        task_path=task,
        arms=parse_arms("with-skill,no-skill"),
        agent="claude-agent-acp",
        model="claude-sonnet",
        out_dir=tmp_path / "out",
    )

    with pytest.raises(AblationSpecError, match="environment manifest"):
        await run_ablation(request)

    assert built == []


# 8. Ablation self-description: the bound environment and the stage snapshot


def _request_with_manifest(
    tmp_path: Path,
    task: Path,
    *,
    arms: str = "with-skill,no-skill",
    environment_manifest: Path | str | None = None,
) -> AblationRequest:
    return AblationRequest(
        task_path=task,
        arms=parse_arms(arms),
        agent="claude-agent-acp",
        model="claude-sonnet",
        out_dir=tmp_path / "out",
        environment_manifest=environment_manifest,
    )


async def test_an_explicit_environment_manifest_beats_the_task_declared_one(
    tmp_path: Path, monkeypatch
) -> None:
    """--environment-manifest wins over task.md, same precedence as the run path.

    Guards "feat(ablate): bind an explicit environment manifest and stamp the
    bound environment". ``bench eval run`` consults
    ``manifest_from_task_document`` only when no explicit manifest was bound
    (``benchflow.evaluation``); the ablation must mirror that, or the same
    flag would mean two different worlds on the two commands.
    """
    from benchflow._utils.content_address import sha256_prefixed

    task = _manifest_task_dir(tmp_path)  # declares environment.yaml
    override = tmp_path / "override.yaml"
    override.write_text(
        "environment:\n  name: ablate-flag-env\n  image: example/flag:latest\n",
        encoding="utf-8",
    )
    built = _patch_rollout(monkeypatch)

    report = await run_ablation(
        _request_with_manifest(tmp_path, task, environment_manifest=override)
    )

    manifest = built[0].config.environment_manifest
    assert (manifest.name, manifest.image) == ("ablate-flag-env", "example/flag:latest")
    assert report.environment == {
        "name": "ablate-flag-env",
        "ref": str(override),
        "env_hash": sha256_prefixed(override.read_bytes()),
        "image": "example/flag:latest",
        "base_image": None,
    }


async def test_the_task_declared_environment_is_stamped_deterministically(
    tmp_path: Path, monkeypatch
) -> None:
    """The report says which world the arms compared in, machine-independently.

    Guards "feat(ablate): bind an explicit environment manifest and stamp the
    bound environment" — the reviewer's open question. The stamp's ``ref`` is
    the ``task.md`` declaration verbatim (never the resolved absolute path,
    which would make ablation.json differ per machine) and ``env_hash`` is the
    manifest's sha256 content address, the same identity the registry uses.
    """
    from benchflow._utils.content_address import sha256_prefixed

    task = _manifest_task_dir(tmp_path)
    _patch_rollout(monkeypatch)

    report = await run_ablation(_request_with_manifest(tmp_path, task))

    assert report.environment == {
        "name": "ablate-declared-env",
        "ref": "environment.yaml",
        "env_hash": sha256_prefixed((task / "environment.yaml").read_bytes()),
        "image": "example/ablate-declared:latest",
        "base_image": None,
    }
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert payload["environment"]["ref"] == "environment.yaml"
    assert str(tmp_path) not in json.dumps(payload["environment"])


async def test_a_manifest_less_ablation_stamps_no_environment(
    tmp_path: Path, monkeypatch
) -> None:
    """No bound world, no stamp — nothing invented."""
    _patch_rollout(monkeypatch)

    report = await run_ablation(_request(tmp_path))

    assert report.environment is None
    assert all(arm.environment is None for arm in report.arms)


async def test_an_unresolvable_explicit_manifest_fails_before_the_parent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The flag fails closed exactly like a broken task declaration."""
    built = _patch_rollout(monkeypatch)
    request = _request_with_manifest(
        tmp_path,
        _task_dir(tmp_path),
        environment_manifest=tmp_path / "nope.yaml",
    )

    with pytest.raises(AblationSpecError, match="--environment-manifest"):
        await run_ablation(request)

    assert built == []


def test_the_environment_manifest_flag_reaches_the_request(
    tmp_path: Path, monkeypatch
) -> None:
    """The CLI threads --environment-manifest through to the request — the
    same flag name and value forms as ``bench eval run``."""
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    seen = _patched_run(monkeypatch, report)

    result = runner.invoke(
        app,
        [
            "eval",
            "ablate",
            "--tasks-dir",
            str(_task_dir(tmp_path)),
            "--environment-manifest",
            "env0@prod",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen[0].environment_manifest == Path("env0@prod")


_ABLATE_PROD_MANIFEST = """\
environment:
  name: ablate-prod
  base_image: example/claws:latest
  owns_lifecycle: false
  services:
    - name: gmail
      command: serve-gmail
      port: 9001
    - name: gcal
      command: serve-gcal
      port: 9002
"""

_ABLATE_OUTAGE_MANIFEST = """\
environment:
  name: ablate-outage
  base_image: example/claws:latest
  owns_lifecycle: false
  services:
    - name: gmail
      command: serve-gmail
      port: 9001
"""


async def test_an_env_arms_swapped_environment_is_stamped_on_its_row(
    tmp_path: Path, monkeypatch
) -> None:
    """An ``env:`` arm's row names the world its delta swapped in.

    Guards "feat(ablate): bind an explicit environment manifest and stamp the
    bound environment": the top-level stamp is the parent's world, and the one
    arm that ran a *different* manifest carries its own stamp — name, the ref
    exactly as the arm spec wrote it, and the child manifest's content hash —
    while every inheriting arm carries none.
    """
    from benchflow._utils.content_address import sha256_prefixed

    task = _task_dir(tmp_path)
    prod = tmp_path / "prod.yaml"
    prod.write_text(_ABLATE_PROD_MANIFEST, encoding="utf-8")
    outage = tmp_path / "outage.yaml"
    outage.write_text(_ABLATE_OUTAGE_MANIFEST, encoding="utf-8")
    (task / "task.md").write_text(
        "---\n"
        'schema_version: "1.3"\n'
        "task:\n"
        "  name: benchflow/ablate-env-arm-demo\n"
        "  description: A task with a swappable service set\n"
        "benchflow:\n"
        "  environment:\n"
        f"    manifest: {prod}\n"
        "---\n"
        "Solve it.\n",
        encoding="utf-8",
    )
    _patch_rollout(monkeypatch)

    report = await run_ablation(
        _request_with_manifest(tmp_path, task, arms=f"no-skill,env:{outage}")
    )

    assert report.environment is not None
    assert report.environment["name"] == "ablate-prod"
    assert report.arms[0].environment is None
    assert report.arms[1].environment == {
        "name": "ablate-outage",
        "ref": str(outage),
        "env_hash": sha256_prefixed(outage.read_bytes()),
        "image": None,
        "base_image": "example/claws:latest",
    }
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert payload["arms"][1]["environment"]["name"] == "ablate-outage"


# 9. Stage UX: the snapshot refs travel with the report, and the post-research
#    rejection teaches the SDK path


class _StageRefRollout(_FakeRollout):
    """A fork that records its stage registry the way the real engine does."""

    async def branch_at_stage(self, stage, n, *, deltas=None):
        from benchflow.branch import StageSnapshot
        from benchflow.sandbox.protocol import SandboxImage

        self._stage_snapshots = {
            stage: StageSnapshot(
                environment_ref=None,
                sandbox_ref=SandboxImage(provider="fake", ref="bf-snap-root-1"),
                stage=stage,
            )
        }
        return await super().branch_at_stage(stage, n, deltas=deltas)


async def test_the_branched_stages_snapshot_refs_are_stamped_into_the_report(
    tmp_path: Path, monkeypatch
) -> None:
    """ablation.json records the stage's roll-back handles.

    Guards "feat(ablate): bind an explicit environment manifest and stamp the
    bound environment" (the stage-UX half): the committed sandbox image ref of
    the branched boundary is what a reader needs to restore that world and
    re-branch it by hand later, so it travels in the report — same refs as the
    parent run's ``stage_snapshots.json``, no second source of truth.
    """
    _patch_rollout(monkeypatch, _StageRefRollout)

    report = await run_ablation(_request(tmp_path))

    assert report.stage_snapshot == {
        "environment_ref": None,
        "sandbox_ref": "bf-snap-root-1",
        "layers": ["sandbox"],
        # Without --keep-snapshots the committed image dies with the run's
        # cleanup, and the report says so (see the retention tests below).
        "ephemeral": True,
        "exported": None,
    }
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert payload["stage_snapshot"]["sandbox_ref"] == "bf-snap-root-1"


# 10. Snapshot retention (RFC §3.6): --keep-snapshots exports, and a handle
#     cleanup destroyed is recorded as ephemeral, never as restorable


class _ExportingStageRefRollout(_StageRefRollout):
    """A stage-ref rollout whose sandbox can docker-save its snapshot image."""

    def __init__(self, config) -> None:
        super().__init__(config)
        rollout = self

        class _Sandbox:
            async def export_image(self, ref: str, target_path) -> None:
                # Recorded in the rollout's call log so the test can prove the
                # export happened BEFORE cleanup destroyed the image.
                rollout.calls.append(f"export:{ref}")
                Path(target_path).write_bytes(b"fake-docker-save-tar")

        self.env = _Sandbox()


async def test_keep_snapshots_exports_the_tar_before_cleanup_and_records_it(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards "feat(ablate): --keep-snapshots export; ephemeral handles
    recorded truthfully" (the durable half): with the flag, the branched
    stage's sandbox image is docker-saved to <out-dir>/snapshots/<ref>.tar
    *before* cleanup destroys it, and ablation.json records the tar's path
    and content sha256 with ephemeral: false."""
    import dataclasses
    import hashlib

    built = _patch_rollout(monkeypatch, _ExportingStageRefRollout)
    request = dataclasses.replace(_request(tmp_path), keep_snapshots=True)

    report = await run_ablation(request)

    # The export ran between the fork and cleanup — the one window in which
    # the committed image still exists.
    calls = built[0].calls
    assert calls.index("export:bf-snap-root-1") < calls.index("cleanup")
    tar_path = Path(request.out_dir) / "snapshots" / "bf-snap-root-1.tar"
    assert tar_path.read_bytes() == b"fake-docker-save-tar"
    assert report.stage_snapshot["ephemeral"] is False
    assert report.stage_snapshot["exported"] == {
        "path": str(tar_path),
        "sha256": "sha256:" + hashlib.sha256(b"fake-docker-save-tar").hexdigest(),
    }
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert payload["stage_snapshot"]["exported"]["path"] == str(tar_path)


async def test_without_keep_snapshots_the_handle_is_recorded_ephemeral(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards "feat(ablate): --keep-snapshots export; ephemeral handles
    recorded truthfully" (the truthful half): the real E2E report recorded
    bf-snap-…, and docker image inspect immediately confirmed the image no
    longer existed — cleanup runs before serialization. The default report
    now says the handle is ephemeral so a reader knows it does not resolve."""
    _patch_rollout(monkeypatch, _StageRefRollout)

    report = await run_ablation(_request(tmp_path))

    assert report.stage_snapshot["ephemeral"] is True
    assert report.stage_snapshot["exported"] is None
    assert "export_error" not in report.stage_snapshot
    payload = json.loads(
        write_ablation_report(report, tmp_path / "out").read_text(encoding="utf-8")
    )
    assert payload["stage_snapshot"]["ephemeral"] is True
    assert payload["stage_snapshot"]["exported"] is None


async def test_a_failed_export_is_recorded_and_does_not_cost_the_rewards(
    tmp_path: Path, monkeypatch
) -> None:
    """The failure path of "feat(ablate): --keep-snapshots export; ephemeral
    handles recorded truthfully": a backend that cannot export (here: no
    sandbox attached) records export_error and keeps the handle ephemeral —
    the arms' rewards and the report survive."""
    import dataclasses

    _patch_rollout(monkeypatch, _StageRefRollout)  # no .env on this fake
    request = dataclasses.replace(_request(tmp_path), keep_snapshots=True)

    report = await run_ablation(request)

    snap = report.stage_snapshot
    assert snap["ephemeral"] is True
    assert snap["exported"] is None
    assert "--keep-snapshots" in snap["export_error"]
    assert [arm.reward for arm in report.arms] == [1.0, 0.0]
    assert not (Path(request.out_dir) / "snapshots").exists()


def test_cli_keep_snapshots_reaches_the_request(tmp_path: Path, monkeypatch) -> None:
    """The flag half of "feat(ablate): --keep-snapshots export; ephemeral
    handles recorded truthfully": --keep-snapshots reaches
    AblationRequest.keep_snapshots, and its absence means False."""
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    seen = _patched_run(monkeypatch, report)

    base = [
        "eval",
        "ablate",
        "--tasks-dir",
        str(_task_dir(tmp_path, "with-flag")),
        "--out-dir",
        str(tmp_path / "out-a"),
    ]
    result = runner.invoke(app, [*base, "--keep-snapshots"])
    assert result.exit_code == 0, result.output
    assert seen[0].keep_snapshots is True

    base[3] = str(_task_dir(tmp_path, "without-flag"))
    result = runner.invoke(app, base)
    assert result.exit_code == 0, result.output
    assert seen[1].keep_snapshots is False


def test_the_table_prints_the_environment_and_the_snapshot_refs(
    tmp_path: Path, monkeypatch
) -> None:
    """The self-description reaches the human surface, not only the JSON."""
    report = _report(arms=_skill_outcomes(with_skill=1.0, no_skill=0.0))
    report.environment = {
        "name": "ablate-prod",
        "ref": "env0@prod",
        "env_hash": "sha256:abc123",
        "image": None,
        "base_image": "example/claws:latest",
    }
    report.stage_snapshot = {
        "environment_ref": None,
        "sandbox_ref": "bf-snap-root-1",
        "layers": ["sandbox"],
    }
    attribute(report.arms, parent_reward=report.parent_reward, stage=report.stage)
    _patched_run(monkeypatch, report)
    monkeypatch.setenv("COLUMNS", "300")

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

    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "Environment: ablate-prod (env0@prod, sha256:abc123)" in out
    assert "Stage snapshot: sandbox=bf-snap-root-1" in out


def test_post_research_rejection_names_the_sdk_branching_path() -> None:
    """The rejection teaches how post-research branching IS done.

    Guards "feat(ablate): bind an explicit environment manifest and stamp the
    bound environment" (the stage-UX half): the CLI keeps refusing the stage
    it cannot capture, but the message now states the working recipe —
    ``mark_stage`` at the cut point, then ``branch_at_stage`` — instead of a
    bare pointer at "the Python API".
    """
    arms = parse_arms("with-skill,no-skill")
    with pytest.raises(AblationSpecError) as excinfo:
        validate_arms_for_stage(arms, "post-research")
    message = str(excinfo.value)
    assert "cannot be captured by this command" in message
    assert "rollout.mark_stage('post-research')" in message
    assert "rollout.branch_at_stage('post-research'" in message
