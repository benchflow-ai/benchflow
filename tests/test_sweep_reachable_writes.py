"""Guard CONFIRM-taskmd-rewards-R4-1: no controller write follows a sandbox-planted link.

The rollout's verifier/ and agent/ folders are /logs/verifier and /logs/agent
in a Docker sandbox. Each test plants a link (absolute or relative) at the
name a controller writer is about to write, the way a sandbox process still
running during soft verification or a pending judge request could, and
checks that a dummy controller marker outside the rollout keeps its bytes
while the name ends up a regular file with BenchFlow's content.
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest

from benchflow.rewards import builtins, llm
from benchflow.rollout import Rollout, _separate_verifier, _setup
from benchflow.task import verifier_core
from benchflow.taskmd import materialize
from benchflow.taskmd import verify as taskmd_verify
from tests._receipt_fakes import receipt_aware
from tests.reachable_fixture import directory, on_disk, reachable
from tests.test_sweep_judge_confirm2 import native_capture as native_capture

MARKER = "DUMMY_CONTROLLER_MARKER"


@pytest.fixture
def victim(tmp_path):
    controller = tmp_path / "controller-only"
    controller.mkdir()
    marker = controller / "marker.json"
    marker.write_text(MARKER)
    return marker


def plant(name_path, victim, form):
    """Replace ``name_path`` with a link to ``victim`` (absolute or relative)."""
    name_path = on_disk(name_path)
    if os.path.lexists(name_path):
        name_path.unlink()
    target = (
        victim
        if form == "absolute"
        else os.path.relpath(victim, start=name_path.parent)
    )
    name_path.symlink_to(target)


def untouched(victim):
    assert victim.read_text() == MARKER


def regular(path):
    assert path.is_file() and not path.is_symlink()
    return path.read_text()


FORMS = ["absolute", "relative"]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("route", ["llm-config", "llm-strategy", "agent-judge"])
async def test_native_judge_reward_files_never_follow_a_link_planted_while_pending(
    native_capture, tmp_path, monkeypatch, victim, route, form
):
    """R4-1: a link planted while the judge answers does not redirect reward writes."""
    verifier, _, source = native_capture
    (source / "answer.txt").write_text("correct")
    verifier._logger = logging.getLogger(__name__)
    verifier._task.config.verifier.env = {}
    verifier._task.config.verifier.judge.model = "scripted"
    verifier._task.instruction = "Read the answer."
    monkeypatch.setattr(verifier, "_resolve_rubric_path", lambda _: tmp_path / "rubric")
    monkeypatch.setattr(verifier, "_resolve_llm_judge_context", lambda _: "Read it.")
    paths = verifier._rollout_paths
    details_victim = victim.with_name("details-marker.json")
    details_victim.write_text(MARKER)

    def attack():
        plant(paths.reward_json_path, victim, form)
        plant(paths.reward_details_json_path, details_victim, form)

    class Judge:
        events = ()

        def __init__(self, **kwargs):
            pass

        async def score(self, directory):
            attack()
            return 1.0

    async def call_judge(model, prompt, **kwargs):
        attack()
        return '{"score": 1}'

    monkeypatch.setattr(builtins, "LLMJudgeRewardFunc", Judge)
    monkeypatch.setattr(llm, "call_judge", call_judge)
    if route == "agent-judge":
        strategy = SimpleNamespace(
            name="judge",
            role="reviewer",
            model=None,
            inputs=[str(source / "answer.txt")],
        )
        result = await verifier._verify_agent_judge(
            strategy=strategy, document=SimpleNamespace(role_prompts={})
        )
    else:
        strategy = (
            SimpleNamespace(input_dir=str(source), model=None)
            if route == "llm-strategy"
            else None
        )
        result = await verifier._verify_llm_judge(strategy=strategy)
    assert result.rewards["reward"] == 1.0
    untouched(victim)
    untouched(details_victim)
    assert json.loads(regular(paths.reward_json_path))["reward"] == 1.0
    assert json.loads(regular(paths.reward_details_json_path))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("planted", ["link", "fifo", "hard-link"])
async def test_native_judge_replaces_special_entries_at_reward_names(
    native_capture, tmp_path, monkeypatch, victim, planted, form
):
    """R4-1: a FIFO or a hard link at reward.json is replaced, never opened."""
    if planted != "link" and form == "relative":
        pytest.skip("only links have a relative form")
    verifier, _, source = native_capture
    (source / "answer.txt").write_text("correct")
    verifier._logger = logging.getLogger(__name__)
    verifier._task.config.verifier.env = {}
    verifier._task.config.verifier.judge.model = "scripted"
    monkeypatch.setattr(verifier, "_resolve_rubric_path", lambda _: tmp_path / "rubric")
    monkeypatch.setattr(verifier, "_resolve_llm_judge_context", lambda _: "Read it.")
    paths = verifier._rollout_paths
    # A hard link needs a file the sandbox can name: one in its own mounts.
    shared = paths.agent_dir / "shared.txt"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text(MARKER)

    class Judge:
        events = ()

        def __init__(self, **kwargs):
            pass

        async def score(self, directory):
            name = paths.reward_json_path
            if planted == "link":
                plant(name, victim, form)
            elif planted == "fifo":
                os.mkfifo(on_disk(name))
            else:
                os.link(on_disk(shared), on_disk(name))
            return 1.0

    monkeypatch.setattr(builtins, "LLMJudgeRewardFunc", Judge)
    result = await verifier._verify_llm_judge(strategy=None)
    assert result.rewards["reward"] == 1.0
    untouched(victim)
    assert shared.read_text() == MARKER
    assert json.loads(regular(paths.reward_json_path)) == {"reward": 1.0}


@pytest.mark.parametrize("form", FORMS)
async def test_reward_json_rewrite_never_follows_a_link(
    tmp_path, monkeypatch, victim, form
):
    """R4-1 sibling: the aggregate rewrite of reward.json replaces the entry."""
    from benchflow.task import RolloutPaths, Verifier

    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    paths.reward_json_path.write_text(json.dumps({"a": 1.0}))
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._reward_range = None

    def aggregate(rewards, **kwargs):
        plant(paths.reward_json_path, victim, form)
        return {"reward": 1.0, **rewards}

    monkeypatch.setattr(verifier_core, "apply_aggregate_policy", aggregate)
    monkeypatch.setattr(verifier, "_parse_reward_json", lambda **_: {"a": 1.0})
    rewards = verifier._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 1.0
    untouched(victim)
    assert json.loads(regular(paths.reward_json_path))["reward"] == 1.0


@pytest.mark.parametrize("form", FORMS)
async def test_plugin_guard_marker_copy_never_follows_a_link(
    tmp_path, monkeypatch, victim, form
):
    """R4-1 sibling: guard markers copied to verifier/ replace the entry."""
    from benchflow.sandbox.lockdown import GuardMarker
    from benchflow.task import RolloutPaths, Verifier

    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._logger = logging.getLogger(__name__)
    plant(paths.verifier_dir / "guard-crashed", victim, form)

    class Box:
        # Guard markers are read through an exec receipt since integ a43679696.
        @receipt_aware
        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout="", stderr="", return_code=0)

    verifier._sandbox = Box()
    monkeypatch.setattr(
        verifier_core,
        "parse_pytest_plugin_guard_markers",
        lambda *_: [GuardMarker("guard-crashed", "crashed", "trace")],
    )
    markers = await verifier._plugin_guard_markers("guard", "main")
    assert markers["crashed"]
    untouched(victim)
    assert regular(paths.verifier_dir / "guard-crashed") == "trace"


@pytest.mark.parametrize("form", FORMS)
async def test_recovered_verifier_outputs_never_follow_a_link(tmp_path, victim, form):
    """R4-1 sibling: per-file recovery downloads privately, then replaces the entry."""
    from benchflow.task import RolloutPaths, SandboxPaths, Verifier

    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._logger = logging.getLogger(__name__)
    plant(paths.reward_json_path, victim, form)

    class Box:
        async def download_file(self, source, target):
            if source.endswith("reward.json"):
                with open(target, "w") as stream:
                    stream.write('{"reward": 1.0}')
            else:
                raise FileNotFoundError(source)

    verifier._sandbox = Box()
    assert await verifier._recover_main_verifier_outputs(SandboxPaths())
    untouched(victim)
    assert json.loads(regular(paths.reward_json_path)) == {"reward": 1.0}


@pytest.mark.parametrize("form", FORMS)
async def test_taskmd_timed_out_record_never_follows_a_link(
    tmp_path, monkeypatch, victim, form
):
    """R4-1 sibling: the timed_out rewrite used a fixed temp name and write_text."""
    from benchflow.task import RolloutPaths

    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    paths.reward_json_path.write_text(json.dumps({"reward": 1.0}))
    plant(paths.reward_json_path.with_name("reward.json.timed-out.tmp"), victim, form)
    monkeypatch.setattr(materialize, "taskmd_metadata", lambda _: {})
    rollout = SimpleNamespace(
        _rollout_paths=paths,
        _rewards={"reward": 1.0},
        _config=SimpleNamespace(task_path=tmp_path),
        _agent_ran_out_of_time=True,
    )
    plant_name = paths.reward_json_path
    real_loads = json.loads

    def loads_then_plant(text, *args, **kwargs):
        value = real_loads(text, *args, **kwargs)
        plant(plant_name, victim, form)
        return value

    monkeypatch.setattr(json, "loads", loads_then_plant)
    Rollout._record_taskmd_timed_out(rollout)
    monkeypatch.setattr(json, "loads", real_loads)
    untouched(victim)
    assert json.loads(regular(paths.reward_json_path))["timed_out"] is True


@pytest.mark.parametrize("form", FORMS)
async def test_separate_verifier_publication_never_follows_a_link(
    tmp_path, victim, form
):
    """R4-1 sibling: a separate verifier's outputs replace entries in verifier/."""
    source = tmp_path / "separate" / "verifier"
    source.mkdir(parents=True)
    (source / "reward.json").write_text('{"reward": 1.0}')
    (source / "judge").mkdir()
    (source / "judge" / "notes.txt").write_text("notes")
    dest = tmp_path / "rollout" / "verifier"
    dest.mkdir(parents=True)
    plant(dest / "reward.json", victim, form)
    victim_dir = victim.parent / "victim-dir"
    victim_dir.mkdir()
    (dest / "judge").symlink_to(victim_dir, target_is_directory=True)
    _separate_verifier._publish_verifier_outputs(directory(source), dest.parent, {})
    untouched(victim)
    assert list(victim_dir.iterdir()) == []
    assert regular(dest / "reward.json") == '{"reward": 1.0}'
    assert regular(dest / "judge" / "notes.txt") == "notes"


@pytest.mark.parametrize("form", FORMS)
def test_trajectory_host_copy_never_follows_a_link(tmp_path, victim, form):
    """R4-1 sibling: the host copy of acp_trajectory.jsonl is placed via the held agent/."""
    agent = tmp_path / "rollout" / "agent"
    agent.mkdir(parents=True)
    plant(agent / "output.txt", victim, form)
    _setup._write_host_file(reachable(agent / "output.txt"), "reply\n")
    untouched(victim)
    assert regular(agent / "output.txt") == "reply\n"
    # The staging file is no longer made inside agent/.
    assert sorted(p.name for p in agent.iterdir()) == ["output.txt"]


@pytest.mark.parametrize("form", FORMS)
def test_taskmd_host_writes_never_follow_a_link(tmp_path, victim, form):
    """R4-1 sibling: a link at a taskmd name refuses; a hard link keeps its inode."""
    verifier_dir = tmp_path / "rollout" / "verifier"
    verifier_dir.mkdir(parents=True)
    plant(verifier_dir / "review.json", victim, form)
    with pytest.raises(taskmd_verify.TaskMdVerifierError):
        taskmd_verify._write_json(verifier_dir / "review.json", {"ok": True})
    untouched(victim)
    plant(verifier_dir / "judgments.jsonl", victim, form)
    with pytest.raises(taskmd_verify.TaskMdVerifierError):
        taskmd_verify._append_jsonl(verifier_dir / "judgments.jsonl", {"n": 1})
    untouched(victim)
    (verifier_dir / "judgments.jsonl").unlink()
    taskmd_verify._append_jsonl(verifier_dir / "judgments.jsonl", {"n": 1})
    taskmd_verify._append_jsonl(verifier_dir / "judgments.jsonl", {"n": 2})
    assert regular(verifier_dir / "judgments.jsonl") == '{"n": 1}\n{"n": 2}\n'


@pytest.mark.parametrize("append", [False, True])
def test_taskmd_host_writes_never_write_a_hard_linked_inode(tmp_path, append):
    """R4-1 sibling: O_TRUNC/O_APPEND on a hard link wrote another mounted file."""
    verifier_dir = tmp_path / "rollout" / "verifier"
    verifier_dir.mkdir(parents=True)
    shared = tmp_path / "rollout" / "agent" / "shared.txt"
    shared.parent.mkdir()
    shared.write_text(MARKER)
    name = verifier_dir / ("judgments.jsonl" if append else "review.json")
    os.link(on_disk(shared), on_disk(name))
    if append:
        taskmd_verify._append_jsonl(name, {"n": 1})
    else:
        taskmd_verify._write_json(name, {"ok": True})
    assert shared.read_text() == MARKER
    assert regular(name).startswith(MARKER if append else "{")
