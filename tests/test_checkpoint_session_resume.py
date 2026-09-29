"""Forks from a kept checkpoint can resume the checkpoint's conversation.

``--from-checkpoint`` used to be unable to use
``--resume-session`` because a kept checkpoint did not record the agent
session id, so children always started fresh sessions, and a checkpoint
retry could not continue the failed conversation either. Now each automatic checkpoint records ``agent_session_id`` (the transcript itself
is in the snapshot); ``load_checkpoint_source`` carries it;
``bench eval branch --from-checkpoint … --resume-session`` and
``bench eval run --retry-from-checkpoint … --retry-resume-session`` resume
it with ACP ``session/load``. A checkpoint without the id is refused for
resume, not silently started fresh.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.branch_run import BranchPlanError, load_checkpoint_source
from benchflow.checkpoints import after_prompt, parse_checkpoint_policy
from tests.test_auto_checkpoints import Sandbox
from tests.test_branch_restore_parent import _fork, _rollout


async def test_checkpoints_record_the_agent_session(tmp_path):
    run = tmp_path / "task__abc"
    run.mkdir()
    rollout = SimpleNamespace(
        _config=SimpleNamespace(checkpoints=parse_checkpoint_policy("every-prompt")),
        _env=Sandbox(),
        _rollout_dir=run,
        _rollout_name="task__abc",
        _cursor=SimpleNamespace(id="n3"),
        _session=SimpleNamespace(session_id="sess-1"),
    )
    await after_prompt(rollout, 1)
    rollout._session = None
    await after_prompt(rollout, 2)
    rows = json.loads((run / "checkpoints.json").read_text())["checkpoints"]
    assert [r["agent_session_id"] for r in rows] == ["sess-1", None]


def _trial(tmp_path, session_id):
    trial = tmp_path / "old" / "task__old"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task_path": "task"}))
    (trial / "checkpoints.json").write_text(
        json.dumps(
            {
                "kind": "benchflow-checkpoints",
                "checkpoints": [
                    {
                        "id": "prompt:1",
                        "after_prompt": 1,
                        "node_id": "n3",
                        "provider": "daytona",
                        "ref": "bf-snap-x",
                        "status": "kept",
                        "agent_session_id": session_id,
                    }
                ],
            }
        )
    )
    return trial


def test_the_source_carries_the_session_id(tmp_path):
    assert (
        load_checkpoint_source(_trial(tmp_path, "sess-1"), None).session_id == "sess-1"
    )


def _plan(tmp_path, source, **kw):
    from tests.test_branch_run import _plan as plan

    return plan(
        tmp_path,
        task_paths=[tmp_path / "task"],
        checkpoint_after=0,
        parent_mode="discard",
        source=source,
        sandbox="daytona",
        **kw,
    )


def test_resume_from_a_checkpoint_needs_its_session_id(tmp_path):
    with_id = load_checkpoint_source(_trial(tmp_path, "sess-1"), None)
    _plan(tmp_path, with_id, resume_session=True).validate()
    without = load_checkpoint_source(_trial(tmp_path / "b", None), None)
    with pytest.raises(BranchPlanError, match="session id"):
        _plan(tmp_path, without, resume_session=True).validate()


async def test_branch_can_resume_an_explicit_session(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    seen = []

    async def child(_node):
        seen.append(rollout._resume_session_id)
        return 1.0

    # No live session on this rollout (a trial restored from a checkpoint).
    await rollout.branch(
        2,
        child,
        snapshot_layers={"sandbox"},
        resume_session=True,
        resume_session_id="sess-from-checkpoint",
    )
    assert seen == ["sess-from-checkpoint"] * 2
    assert _fork(rollout)["snapshot"]["agent_session"] == "resumed"


def test_retry_policy_can_resume():
    from benchflow.checkpoint_retry import parse_retry_policy
    from benchflow.evaluation import EvaluationConfig

    policy = parse_retry_policy("on-failure", prompt=None, resume_session=True)
    assert policy.resume_session is True
    config = EvaluationConfig(
        checkpoints="prompt:1",
        retry_from_checkpoint="on-failure",
        retry_resume_session=True,
    )
    assert config.retry_policy().resume_session is True


async def test_a_retry_resumes_the_checkpoints_session(tmp_path):
    from benchflow.checkpoint_retry import parse_retry_policy, run_checkpoint_retry
    from tests.test_branch_isolated import IMAGES, IsoRollout
    from tests.test_checkpoint_retry import _finished_trial, _result

    IMAGES.clear()
    IsoRollout.all = []
    root, _ = await _finished_trial(tmp_path)
    path = root._rollout_dir / "checkpoints.json"
    document = json.loads(path.read_text())
    document["checkpoints"][0]["agent_session_id"] = "sess-failed-run"
    path.write_text(json.dumps(document))
    seen = []
    original = IsoRollout.connect

    async def connect(self):
        seen.append(self._resume_session_id)
        await original(self)

    IsoRollout.connect = connect
    try:
        await run_checkpoint_retry(
            root,
            _result(0.0),
            parse_retry_policy("on-failure", prompt=None, resume_session=True),
        )
    finally:
        IsoRollout.connect = original
    assert seen == ["sess-failed-run"]
    fork = json.loads((root._rollout_dir / "tree.json").read_text())["forks"][0]
    assert fork["snapshot"]["agent_session"] == "resumed"


def test_eval_run_carries_retry_resume_session(tmp_path):
    from types import SimpleNamespace as NS

    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan
    from benchflow.eval_sharding import _config_payload
    from benchflow.eval_worker import _evaluation_config

    task = __import__("pathlib").Path(__file__).parent / "examples" / "hello-world-task"
    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=task,
            agent="oracle",
            checkpoints="prompt:1",
            retry_from_checkpoint="on-failure",
            retry_resume_session=True,
        )
    )
    config = plan.make_eval_config()
    assert config.retry_policy().resume_session is True
    payload = _config_payload(config, shard=NS(concurrency=1, task_names=["t"]))
    assert _evaluation_config(payload).retry_resume_session is True
