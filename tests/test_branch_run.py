"""``bench eval branch`` and the branch-trial driver behind it.

Branching a real agent run used to need a hand-written script
(``docs/examples/branch-agent-run.py``); there was no CLI entry point. These
tests drive :func:`benchflow.branch_run.run_branch_trial` through the real
branch engine, with a scripted rollout whose sandbox is a list of prompts the
agent has "applied", so each child's starting world is observable.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from benchflow import branch_run
from benchflow.branch_run import (
    BranchPlan,
    BranchPlanError,
    ChildSpec,
    load_checkpoint_source,
    parse_child_spec,
    run_branch_trial,
    write_branch_job,
)
from benchflow.cli.main import app
from benchflow.rollout import Rollout
from benchflow.rollout._setup import _resolve_prompts
from benchflow.sandbox.protocol import SandboxImage
from benchflow.trajectories.tree import Step

# ── child specs ──────────────────────────────────────────────────────


def test_child_spec_label_only():
    assert parse_child_spec("label=baseline") == ChildSpec("baseline")


def test_child_prompt_takes_the_rest_including_commas():
    spec = parse_child_spec("label=hint,prompt=Rename a, then b, then stop")
    assert spec == ChildSpec("hint", "Rename a, then b, then stop")


def test_child_prompt_file(tmp_path):
    (tmp_path / "p.md").write_text("From a file, with commas.")
    spec = parse_child_spec("label=f,prompt-file=p.md", base_dir=tmp_path)
    assert spec.prompt == "From a file, with commas."


@pytest.mark.parametrize(
    "spec",
    [
        "baseline",
        "prompt=hi",
        "label=",
        "label=a,colour=red",
        "label=a,label=b",
        "label=a,prompt=",
        "label=a,prompt-file=/nonexistent/p.md",
    ],
)
def test_bad_child_specs_are_refused(spec):
    with pytest.raises(BranchPlanError):
        parse_child_spec(spec)


# ── plan validation ──────────────────────────────────────────────────


def _plan(tmp_path: Path, **overrides) -> BranchPlan:
    values = {
        "task_paths": [tmp_path / "task"],
        "agent": "claude-agent-acp",
        "model": None,
        "sandbox": "docker",
        "children": [ChildSpec("baseline"), ChildSpec("hint", "Use the hint.")],
        "checkpoint_after": 1,
        "jobs_dir": tmp_path / "jobs",
        "job_name": "job",
    }
    values.update(overrides)
    return BranchPlan(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"children": [ChildSpec("only")]}, "at least two"),
        ({"children": [ChildSpec("a"), ChildSpec("a")]}, "unique"),
        ({"checkpoint_after": 3, "prompts": ["a", "b"]}, "after the last"),
        ({"agent": "oracle", "checkpoint_after": 0}, "oracle takes no prompts"),
        (
            {
                "agent": "oracle",
                "checkpoint_after": 2,
                "children": [ChildSpec("a"), ChildSpec("b")],
            },
            "0 or 1",
        ),
    ],
)
def test_inconsistent_plans_are_refused(tmp_path, overrides, message):
    with pytest.raises(BranchPlanError, match=message):
        _plan(tmp_path, **overrides).validate()


# ── the scripted rollout ─────────────────────────────────────────────


class WorldSandbox:
    """A sandbox whose filesystem is the list of prompts the agent applied."""

    supports_snapshot = True

    # Provider storage, shared by every sandbox (isolated children restore
    # the parent's snapshot into their own sandbox).
    images: ClassVar[dict[str, list[str]]] = {}

    def __init__(self, rollout: ScriptedRollout) -> None:
        self.rollout = rollout
        self.restores: list[str] = []
        self.deleted: list[str] = []

    async def snapshot(self, name=None) -> SandboxImage:
        ref = f"bf-snap-{len(self.images)}"
        self.images[ref] = list(self.rollout.world)
        return SandboxImage(provider="docker", ref=ref)

    async def restore(self, image: SandboxImage) -> None:
        self.restores.append(image.ref)
        self.rollout.calls.append(("restore", image.ref))
        self.rollout.world = list(self.images.get(image.ref, ["kept-checkpoint"]))

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        self.deleted.append(image.ref)
        return True


class ScriptedRollout(Rollout):
    """The real branch engine over a scripted lifecycle."""

    last: ScriptedRollout | None = None
    subs: ClassVar[list[ScriptedRollout]] = []

    def __init__(self, config) -> None:
        super().__init__(config)
        self.calls: list[tuple] = []
        self.world: list[str] = []
        self.fail_on: str | None = None
        if config.rollout_name is None:
            ScriptedRollout.last = self
            ScriptedRollout.subs = []
            WorldSandbox.images.clear()
        else:
            ScriptedRollout.subs.append(self)

    async def setup(self) -> None:
        cfg = self._config
        self._rollout_name = cfg.rollout_name or f"{cfg.task_path.name}__scripted"
        self._rollout_dir = Path(cfg.jobs_dir) / cfg.job_name / self._rollout_name
        self._rollout_dir.mkdir(parents=True)
        self._resolved_prompts = _resolve_prompts(cfg.task_path, cfg.prompts)
        self._env = WorldSandbox(self)
        self.calls.append(("setup",))

    async def start(self) -> None:
        self.calls.append(("start",))

    async def install_agent(self) -> None:
        self.calls.append(("install_agent",))

    async def connect(self) -> None:
        self._acp_client = object()
        self._session = SimpleNamespace(session_id=f"s-{self._config.rollout_name}")
        self.calls.append(("connect", self._resume_session_id))

    async def disconnect(self) -> None:
        self._acp_client = None

    async def execute(self, prompts=None, *, node=None):
        self.calls.append(("execute", list(prompts), list(self.world)))
        if self.fail_on in prompts:
            raise RuntimeError("agent crashed")
        self.world.extend(prompts)
        step = Step(id=f"step-{len(self.calls)}", data={"event": None})
        if node is not None:
            self._cursor = self._tree.populate(node, step)
        elif self._branch_child_active and self._cursor.step_in is None:
            self._cursor = self._tree.populate(self._cursor, step)
        else:
            self._cursor = self._tree.advance(self._cursor, step)
        return [], 0

    async def verify(self):
        self._verify_calls += 1
        self.calls.append(("verify", list(self.world)))
        # The task passes when its instruction was applied last.
        self._rewards = {"reward": 1.0 if self.world[-1:] == ["Do it."] else 0.0}
        return self._rewards

    async def finalize(self):
        self.calls.append(("finalize",))

    async def cleanup(self):
        self.calls.append(("cleanup",))


@pytest.fixture
def task(tmp_path: Path) -> Path:
    path = tmp_path / "task"
    path.mkdir()
    (path / "instruction.md").write_text("Do it.")
    return path


def _named(calls, name):
    return [call for call in calls if call[0] == name]


async def test_continue_branches_after_n_prompts_and_finishes_the_parent(
    tmp_path, task
):
    plan = _plan(
        tmp_path,
        task_paths=[task],
        prompts=["Draft first.", "@instruction"],
        checkpoint_after=1,
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    rollout = ScriptedRollout.last
    executes = _named(rollout.calls, "execute")
    assert executes == [
        # parent up to the checkpoint
        ("execute", ["Draft first."], []),
        # baseline: the parent's remaining prompts, from the checkpoint
        ("execute", ["Do it."], ["Draft first."]),
        # hint: its own prompt, from the same checkpoint
        ("execute", ["Use the hint."], ["Draft first."]),
        # the parent continues from its restored world
        ("execute", ["Do it."], ["Draft first."]),
    ]
    assert outcome.error is None
    assert outcome.fork_status == "completed"
    assert outcome.value == 0.5
    assert outcome.parent_restore == "restored"
    assert outcome.parent_reward == 1.0
    assert [
        (c["label"], c["reward"], c["reward_source"]) for c in outcome.children
    ] == [
        ("baseline", 1.0, "verifier"),
        ("hint", 0.0, "verifier"),
    ]
    assert rollout.calls[-1] == ("finalize",)
    tree = json.loads((rollout._rollout_dir / "tree.json").read_text())
    interventions = [c["intervention"] for c in tree["forks"][0]["children"]]
    assert [i["label"] for i in interventions] == ["baseline", "hint"]
    digest = hashlib.sha256(b"Use the hint.").hexdigest()[:12]
    assert [(i["requested"], i["execution"]) for i in interventions] == [
        ("parent's remaining prompts (1)", "runner"),
        (f"own prompt (13 characters, sha256:{digest})", "runner"),
    ]


async def test_discard_skips_the_parent_restore_and_verify(tmp_path, task):
    plan = _plan(tmp_path, task_paths=[task], parent_mode="discard")
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    rollout = ScriptedRollout.last
    assert outcome.parent_restore == "skipped"
    assert outcome.parent_reward is None
    assert len(rollout._env.restores) == 2  # one per child, none for the parent
    # Only the two children verified; the parent was finalized unverified.
    assert len(_named(rollout.calls, "verify")) == 2
    assert rollout.calls[-1] == ("finalize",)


async def test_checkpoint_zero_branches_before_any_prompt(tmp_path, task):
    plan = _plan(tmp_path, task_paths=[task], checkpoint_after=0)
    await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    executes = _named(ScriptedRollout.last.calls, "execute")
    assert executes[0] == ("execute", ["Do it."], [])  # baseline from the start
    assert executes[1] == ("execute", ["Use the hint."], [])


async def test_oracle_children_run_solve_sh(tmp_path, task, monkeypatch):
    turns = []

    async def fake_oracle_turn(rollout, node=None):
        turns.append((node is not None, list(rollout.world)))
        rollout.world.append("Do it.")
        step = Step(id=f"oracle-{len(turns)}", data={"event": None})
        if node is not None:
            rollout._cursor = rollout._tree.populate(node, step)
        else:
            rollout._cursor = rollout._tree.advance(rollout._cursor, step)

    monkeypatch.setattr(branch_run, "_oracle_turn", fake_oracle_turn)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        agent="oracle",
        checkpoint_after=0,
        children=[ChildSpec("a"), ChildSpec("b")],
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    # two children and the continuing parent, each from the empty checkpoint
    assert turns == [(True, []), (True, []), (False, [])]
    assert not _named(ScriptedRollout.last.calls, "connect")
    assert outcome.value == 1.0
    assert outcome.parent_reward == 1.0


async def test_a_failing_trial_is_reported_and_finalized(tmp_path, task, monkeypatch):
    class Failing(ScriptedRollout):
        def __init__(self, config):
            super().__init__(config)
            self.fail_on = "Draft first."

    plan = _plan(tmp_path, task_paths=[task], prompts=["Draft first.", "@instruction"])
    outcome = await run_branch_trial(plan, task, rollout_factory=Failing)
    assert outcome.error == "RuntimeError: agent crashed"
    assert outcome.fork_status is None
    assert ScriptedRollout.last.calls[-1] == ("finalize",)


# ── branching again from a kept checkpoint ───────────────────────────


def _kept_trial(tmp_path: Path, *, retention="kept", layers=("sandbox",)) -> Path:
    trial = tmp_path / "old" / "task__old"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task_path": "task"}))
    (trial / "tree.json").write_text(
        json.dumps(
            {
                "kind": "benchflow-branch-tree",
                "schema_version": 1,
                "forks": [
                    {
                        "id": "f1",
                        "snapshot": {
                            "captured_layers": list(layers),
                            "retention": retention,
                            "sandbox": {"provider": "docker", "ref": "bf-snap-kept"},
                        },
                    }
                ],
            }
        )
    )
    return trial


def test_a_kept_checkpoint_is_found(tmp_path):
    source = load_checkpoint_source(_kept_trial(tmp_path), None)
    assert (source.fork_id, source.provider, source.ref, source.task_name) == (
        "f1",
        "docker",
        "bf-snap-kept",
        "task",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retention": "deleted"}, "no kept sandbox snapshot"),
        ({"layers": ("environment", "sandbox")}, "cannot be re-imported"),
    ],
)
def test_unusable_checkpoints_are_refused(tmp_path, kwargs, message):
    with pytest.raises(BranchPlanError, match=message):
        load_checkpoint_source(_kept_trial(tmp_path, **kwargs), None)


async def test_from_checkpoint_restores_the_kept_snapshot_before_install(
    tmp_path, task
):
    source = load_checkpoint_source(_kept_trial(tmp_path), None)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        checkpoint_after=0,
        parent_mode="discard",
        source=source,
    )
    plan.validate()
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    rollout = ScriptedRollout.last
    names = [call[0] for call in rollout.calls]
    assert names[:4] == ["setup", "start", "restore", "install_agent"]
    assert rollout.calls[2] == ("restore", "bf-snap-kept")
    # both children start from the kept checkpoint's world
    assert [call[2] for call in _named(rollout.calls, "execute")] == [
        ["kept-checkpoint"],
        ["kept-checkpoint"],
    ]
    # The source also says where the source
    # trial is and how many trajectory events precede the fork, so the
    # branch-tree export can stitch the prefix back on (None: not step-keyed).
    assert outcome.source == {
        "trial": "task__old",
        "trial_path": outcome.source["trial_path"],
        "fork_id": "f1",
        "provider": "docker",
        "ref": "bf-snap-kept",
        "prefix_events": None,
    }
    assert outcome.source["trial_path"].endswith("task__old")
    recorded = json.loads((rollout._rollout_dir / "checkpoint_source.json").read_text())
    # It also says what the sandbox reused (None here: the
    # scripted rollout does not run the real install_agent).
    assert recorded == {**outcome.source, "snapshot_start": None}


# ── the job folder ───────────────────────────────────────────────────


async def test_job_summary_counts_children(tmp_path, task):
    plan = _plan(tmp_path, task_paths=[task], checkpoint_after=0)
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    job_dir = write_branch_job(plan, [outcome])
    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["kind"] == "benchflow-branch-job"
    assert (summary["total"], summary["passed"], summary["score"]) == (2, 1, 0.5)
    assert summary["children_requested"] == [
        {"label": "baseline", "own_prompt": False, "parent": None},
        {"label": "hint", "own_prompt": True, "parent": None},
    ]
    assert summary["trials"][0]["value"] == 0.5


# ── the command ──────────────────────────────────────────────────────


HELLO_TASK = Path(__file__).parent / "examples" / "hello-world-task"


def test_cli_refuses_a_single_child():
    result = CliRunner().invoke(
        app, ["eval", "branch", "--tasks-dir", str(HELLO_TASK), "--child", "label=a"]
    )
    assert result.exit_code == 2
    assert "at least two" in result.output


def test_cli_runs_each_task_and_writes_the_job(tmp_path, monkeypatch):
    plans = []

    async def fake_trial(plan, task_path, **_):
        plans.append(plan)
        return branch_run.BranchTrialOutcome(
            task=task_path.name,
            value=0.5,
            fork_status="completed",
            parent_restore="restored",
            parent_reward=1.0,
            children=[
                {
                    "label": "baseline",
                    "node_id": "n2",
                    "status": "scored",
                    "reward": 1.0,
                    "reward_source": "verifier",
                    "path": "branches/f/children/n2",
                },
                {
                    "label": "hint",
                    "node_id": "n3",
                    "status": "scored",
                    "reward": 0.0,
                    "reward_source": "verifier",
                    "path": "branches/f/children/n3",
                },
            ],
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--prompt",
            "Draft first.",
            "--prompt",
            "@instruction",
            "--child",
            "label=baseline",
            "--child",
            "label=hint,prompt=Use the hint, then stop.",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "j",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = plans[0]
    assert plan.checkpoint_after == 1
    assert plan.parent_mode == "continue"
    assert plan.snapshot_layers == frozenset({"sandbox"})
    assert plan.children[1] == ChildSpec("hint", "Use the hint, then stop.")
    summary = json.loads((tmp_path / "jobs" / "j" / "summary.json").read_text())
    assert (summary["total"], summary["passed"]) == (2, 1)
    assert plan.task_paths == [HELLO_TASK]
    assert "V = 0.5" in result.output


def test_cli_exit_code_is_1_when_a_fork_did_not_complete(tmp_path, monkeypatch):
    async def fake_trial(plan, task_path, **_):
        return branch_run.BranchTrialOutcome(
            task=task_path.name, fork_status="partial", error="ValueError: x"
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 1


# ── isolated, parallel and nested children ───────────────────────────


def test_child_spec_parent():
    spec = parse_child_spec("label=a1,parent=a,prompt=Go on, carefully.")
    assert spec == ChildSpec("a1", "Go on, carefully.", parent="a")


@pytest.mark.parametrize(
    ("children", "message"),
    [
        (
            [ChildSpec("a"), ChildSpec("b"), ChildSpec("a1", "x", parent="a")],
            "at least two children under 'a'",
        ),
        (
            [
                ChildSpec("a"),
                ChildSpec("b"),
                ChildSpec("z1", parent="zz"),
                ChildSpec("z2", parent="zz"),
            ],
            "unknown parent 'zz'",
        ),
    ],
)
def test_bad_nesting_is_refused(tmp_path, children, message):
    with pytest.raises(BranchPlanError, match=message):
        _plan(tmp_path, children=children).validate()


def test_concurrency_or_nesting_selects_isolated_children(tmp_path):
    assert not _plan(tmp_path).isolated
    assert _plan(tmp_path, concurrency=2).isolated
    assert _plan(tmp_path, isolate_children=True).isolated
    nested = _plan(
        tmp_path,
        children=[
            ChildSpec("a"),
            ChildSpec("b"),
            ChildSpec("a1", "x", parent="a"),
            ChildSpec("a2", "y", parent="a"),
        ],
    )
    assert nested.isolated


async def test_parallel_children_each_start_from_the_checkpoint(tmp_path, task):
    plan = _plan(
        tmp_path,
        task_paths=[task],
        prompts=["Draft first.", "@instruction"],
        concurrency=2,
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    root = ScriptedRollout.last
    assert outcome.error is None, outcome.error
    assert len(ScriptedRollout.subs) == 2
    for sub in ScriptedRollout.subs:
        first = _named(sub.calls, "execute")[0]
        assert first[2] == ["Draft first."]
    # The parent ran nothing in its own sandbox after the checkpoint but its
    # own continuation.
    assert _named(root.calls, "execute") == [
        ("execute", ["Draft first."], []),
        ("execute", ["Do it."], ["Draft first."]),
    ]
    assert outcome.value == 0.5
    tree = json.loads((root._rollout_dir / "tree.json").read_text())
    assert tree["forks"][0]["children_mode"] == {
        "isolated": True,
        "concurrency": 2,
        "prewarm": 2,
        "child_retries": 1,
        "continue_after_child_failure": True,
    }


async def test_nested_children_fork_again_from_their_parent_child(tmp_path, task):
    plan = _plan(
        tmp_path,
        task_paths=[task],
        prompts=["Draft first.", "@instruction"],
        children=[
            ChildSpec("a"),
            ChildSpec("b", "Other."),
            ChildSpec("a1", "Do it.", parent="a"),
            ChildSpec("a2", "Other.", parent="a"),
        ],
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    assert outcome.error is None, outcome.error
    root = ScriptedRollout.last
    tree = json.loads((root._rollout_dir / "tree.json").read_text())
    assert len(tree["forks"]) == 2
    rows = {row["label"]: row for row in outcome.children}
    assert set(rows) == {"a", "b", "a1", "a2"}
    assert rows["a1"]["parent_label"] == rows["a2"]["parent_label"] == "a"
    assert rows["a"]["parent_label"] is None
    assert (rows["a"]["reward"], rows["a1"]["reward"], rows["a2"]["reward"]) == (
        1.0,
        1.0,
        0.0,
    )
    # a1 and a2 started from a's state: the draft plus a's own prompt.
    grandkids = [
        s
        for s in ScriptedRollout.subs
        if s._rollout_name in {rows["a1"]["node_id"], rows["a2"]["node_id"]}
    ]
    assert [_named(g.calls, "execute")[0][2] for g in grandkids] == [
        ["Draft first.", "Do it."]
    ] * 2
    assert outcome.value == 0.5  # the top-level fork's value


def test_cli_concurrency_and_isolation_reach_the_plan(tmp_path, monkeypatch):
    plans = []

    async def fake_trial(plan, task_path, **_):
        plans.append(plan)
        return branch_run.BranchTrialOutcome(
            task=task_path.name, fork_status="completed"
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--child",
            "label=a1,parent=a,prompt=x",
            "--child",
            "label=a2,parent=a,prompt=y",
            "--concurrency",
            "3",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert plans[0].concurrency == 3
    assert plans[0].isolated
    assert [c.parent for c in plans[0].children] == [None, None, "a", "a"]


async def test_branch_checkpoints_each_parent_prompt(tmp_path, task, monkeypatch):
    from benchflow.checkpoints import parse_checkpoint_policy

    seen = []

    async def fake_after_prompt(rollout, number):
        seen.append(
            (number, rollout._config.checkpoints is not None, list(rollout.world))
        )

    monkeypatch.setattr(branch_run, "after_prompt", fake_after_prompt)
    plan = _plan(
        tmp_path,
        task_paths=[task],
        prompts=["Draft first.", "Second.", "@instruction"],
        checkpoint_after=2,
        checkpoints=parse_checkpoint_policy("every-prompt", keep=2),
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    assert outcome.error is None
    # One execute per parent prompt, each followed by its checkpoint hook,
    # before the fork (1, 2) and in the continuation (3); the in-place
    # children (baseline, hint) run in between and are not checkpointed.
    root = ScriptedRollout.last
    assert [c[1] for c in _named(root.calls, "execute")] == [
        ["Draft first."],
        ["Second."],
        ["Do it."],
        ["Use the hint."],
        ["Do it."],
    ]
    assert seen == [
        (1, True, ["Draft first."]),
        (2, True, ["Draft first.", "Second."]),
        (3, True, ["Draft first.", "Second.", "Do it."]),
    ]


def test_cli_checkpoint_options(tmp_path, monkeypatch):
    plans = []

    async def fake_trial(plan, task_path, **_):
        plans.append(plan)
        return branch_run.BranchTrialOutcome(
            task=task_path.name, fork_status="completed"
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    base = [
        "eval",
        "branch",
        "--tasks-dir",
        str(HELLO_TASK),
        "--child",
        "label=a",
        "--child",
        "label=b",
        "--jobs-dir",
        str(tmp_path / "jobs"),
    ]
    result = CliRunner().invoke(
        app, [*base, "--checkpoints", "prompt:1", "--checkpoint-keep", "1"]
    )
    assert result.exit_code == 0, result.output
    assert (plans[0].checkpoints.after, plans[0].checkpoints.keep) == (
        frozenset({1}),
        1,
    )
    bad = CliRunner().invoke(app, [*base, "--checkpoints", "sometimes"])
    assert bad.exit_code == 2
    assert "--checkpoints" in bad.output


async def test_resume_session_children_resume_the_parents_session(tmp_path, task):
    plan = _plan(
        tmp_path,
        task_paths=[task],
        prompts=["Draft first.", "@instruction"],
        resume_session=True,
    )
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    assert outcome.error is None, outcome.error
    root = ScriptedRollout.last
    # The parent connects fresh twice (before and after the fork); both
    # in-place children connect with the parent's session id.
    assert [c[1] for c in _named(root.calls, "connect")] == [
        None,
        "s-None",
        "s-None",
        None,
    ]
    tree = json.loads((root._rollout_dir / "tree.json").read_text())
    assert tree["forks"][0]["snapshot"]["agent_session"] == "resumed"


def test_cli_resume_session_flag(tmp_path, monkeypatch):
    plans = []

    async def fake_trial(plan, task_path, **_):
        plans.append(plan)
        return branch_run.BranchTrialOutcome(
            task=task_path.name, fork_status="completed"
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--resume-session",
            "--jobs-dir",
            str(tmp_path / "j"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert plans[0].resume_session is True


# ── cost summary ───────────────────────────────────────


async def test_outcome_carries_per_fork_cost_and_totals(tmp_path, task):
    plan = _plan(tmp_path, task_paths=[task], checkpoint_after=0)
    outcome = await run_branch_trial(plan, task, rollout_factory=ScriptedRollout)
    assert outcome.cost is not None
    assert set(outcome.cost) >= {"tokens", "usd", "usd_known", "sandbox_seconds"}
    [fork] = outcome.forks
    assert fork["children"] == 2 and fork["value"] == 0.5
    assert isinstance(fork["cost"]["sandbox_seconds"], float)
    job = write_branch_job(plan, [outcome])
    summary = json.loads((job / "summary.json").read_text())
    assert summary["cost"]["sandbox_seconds"] == outcome.cost["sandbox_seconds"]
    assert summary["cost"]["usd"] is None


def test_cli_prints_a_cost_table(tmp_path, monkeypatch):
    async def fake_trial(plan, task_path, **_):
        return branch_run.BranchTrialOutcome(
            task=task_path.name,
            fork_status="completed",
            value=0.5,
            forks=[
                {
                    "id": "abcdef0123",
                    "from": "checkpoint",
                    "children": 2,
                    "value": 0.5,
                    "cost": {
                        "tokens": 79683,
                        "usd": None,
                        "usd_known": False,
                        "wall_seconds": 180.5,
                        "sandbox_seconds": 320.25,
                    },
                }
            ],
            cost={
                "tokens": 79683,
                "usd": None,
                "usd_known": False,
                "sandbox_seconds": 320.25,
            },
        )

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            str(tmp_path / "j"),
        ],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert "Sandbox-s" in result.output
    assert "79,683" in result.output
    assert "320" in result.output
    # Say why USD is missing and where the estimate is.
    assert "USD not reported" in result.output


def test_cli_refuses_a_misspelt_agent_before_any_trial(tmp_path, monkeypatch):
    """bf.branch refuses a close misspelling of a registered agent; the CLI
    used to start a sandbox for it."""
    trials = []

    async def fake_trial(plan, task_path, **_):
        trials.append(task_path)

    monkeypatch.setattr(branch_run, "run_branch_trial", fake_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "branch",
            "--tasks-dir",
            str(HELLO_TASK),
            "--agent",
            "claude-agnet-acp",
            "--sandbox",
            "daytona",
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code != 0
    assert trials == []
    assert "did you mean 'claude-agent-acp'" in result.output


async def test_branch_trial_loads_the_task_documents_environment_manifest(tmp_path):
    """A task.md that declares ``benchflow.environment.manifest`` gets its
    environment plane on ``bench eval branch`` too, as on ``bench eval run``
    (``evaluation._environment_manifest_from_task_document``).

    ``run_branch_trial`` used to build its RolloutConfig without the manifest, so framework-started
    services never started (and were never restarted after a restore) on
    branch runs.
    """
    task = tmp_path / "task"
    task.mkdir()
    (task / "environment.toml").write_text(
        '[environment]\nname = "svc"\nbase_image = "ubuntu:24.04"\n'
        "owns_lifecycle = false\n\n[[environment.services]]\n"
        'name = "svc"\ncommand = "python3 -m http.server 8000"\nport = 8000\n'
    )
    (task / "task.md").write_text(
        "---\nschema_version: '1.0'\nbenchflow:\n  environment:\n"
        "    manifest: environment.toml\n---\n\n## prompt\n\nDo it.\n"
    )
    await run_branch_trial(
        _plan(tmp_path, task_paths=[task]), task, rollout_factory=ScriptedRollout
    )
    manifest = ScriptedRollout.last._config.environment_manifest
    assert manifest is not None
    assert [s.name for s in manifest.services] == ["svc"]
