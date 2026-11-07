"""Guards FF-sw-daytona-C8, the receipt-loss regression at da6d3b9f.

Exercise JSON persistence and the real Daytona snapshot/admission code. Only
provider creation, process transport and credential I/O are faked; no key is used.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.branch import StageSnapshot
from benchflow.branch_lineage import ForkRecord
from benchflow.branch_run import load_checkpoint_source, run_branch_trial
from benchflow.checkpoint_retry import parse_retry_policy, run_checkpoint_retry
from benchflow.checkpoints import (
    after_prompt,
    load_checkpoints,
    parse_checkpoint_policy,
)
from benchflow.models import RolloutResult
from benchflow.sandbox import daytona as sdk
from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.protocol import SandboxImage
from tests.test_branch_run import ScriptedRollout, WorldSandbox, _plan

pytest.importorskip("daytona")
sdk._load_daytona_sdk()
META = "benchflow_toolbox_startup_v1"


def daytona(monkeypatch, activation="/tmp/task-activation\n'quoted'"):
    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env.logger = logging.getLogger(__name__)
    env._strategy = _DaytonaDirect(env)
    env._auto_delete_interval = env._auto_stop_interval = 0
    env._network_block_all = False
    env._persistent_env = {}
    env._create_outcome_unknown = False
    env._toolbox_startup_receipt = dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, "")
    env._toolbox_startup_receipt["BASH_ENV"] = activation
    env.environment_name = "dummy-checkpoint"
    creates = []

    def handle():
        return SimpleNamespace(
            id="dummy-sandbox",
            delete=AsyncMock(),
            _experimental_create_snapshot=AsyncMock(),
            process=SimpleNamespace(
                exec=AsyncMock(
                    # The start-up probe's answer when no start-up name is
                    # set: no record, then its end mark.
                    return_value=SimpleNamespace(exit_code=0, result="Z")
                )
            ),
        )

    env._sandbox = handle()

    async def create(params):
        creates.append(params)
        env._sandbox = handle()

    env._create_sandbox = create
    # The restored sandbox's daemon log is made root-only by a root exec of
    # its own before anything else runs (K1-R1-B-1).
    env._sandbox_exec = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0, stdout="BF daemon-log root-only\n", stderr=""
        )
    )
    monkeypatch.setattr(env._strategy, "_credential_ops", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "benchflow.sandbox.daytona_strategies.scrub_credentials",
        AsyncMock(return_value=[]),
    )
    return env, creates


@pytest.fixture
def world(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Do it.")
    provider, creates = daytona(monkeypatch)
    admitted = []

    class ReceiptSandbox(WorldSandbox):
        start_ref = None

        async def snapshot(self, name=None):
            image = await provider.snapshot("dummy-checkpoint")
            self.images[image.ref] = list(self.rollout.world)
            return image

        def start_from_snapshot(self, image):
            admitted.append(image)
            accepted = provider.start_from_snapshot(image)
            self.start_ref = image.ref
            return accepted

        async def restore(self, image):
            admitted.append(image)
            await provider.restore(image)
            await super().restore(image)

    class ReceiptRollout(ScriptedRollout):
        async def setup(self):
            await super().setup()
            self._env = ReceiptSandbox(self)

        async def start(self):
            if self._env.start_ref:
                self.world = list(
                    self._env.images.get(self._env.start_ref, ["kept-checkpoint"])
                )
            await super().start()

    return SimpleNamespace(
        task=task,
        provider=provider,
        creates=creates,
        admitted=admitted,
        factory=ReceiptRollout,
    )


async def automatic_checkpoint(tmp_path, world):
    from benchflow.rollout import RolloutConfig

    root = world.factory(
        RolloutConfig(
            task_path=world.task,
            agent="dummy --agent",
            jobs_dir=tmp_path / "jobs",
            job_name="old",
            checkpoints=parse_checkpoint_policy("every-prompt"),
        )
    )
    await root.setup()
    root._resolved_prompts = ["Draft first.", "Do it."]
    root.world = ["draft"]
    (root._rollout_dir / "config.json").write_text(
        json.dumps({"task_path": str(world.task)})
    )
    await after_prompt(root, 1)
    return root


async def test_automatic_checkpoint_roundtrips_into_real_restore(tmp_path, world):
    root = await automatic_checkpoint(tmp_path, world)
    [row] = load_checkpoints(root._rollout_dir)
    # Deserialize as an external consumer, then use the actual admission path.
    image = SandboxImage(row["provider"], row["ref"], row.get("meta", {}))
    await world.provider.restore(image)
    assert world.provider._toolbox_startup_receipt["BASH_ENV"] == (
        "/tmp/task-activation\n'quoted'"
    )
    assert len(world.creates) == 1
    assert all(world.creates[0].env_vars[n] == "" for n in SYSTEM_SHELL_EMPTY_VARS)


@pytest.mark.parametrize("origin", ["automatic", "retained-fork"])
@pytest.mark.parametrize("isolated", [False, True])
async def test_from_checkpoint_admits_receipt_through_branch_driver(
    tmp_path, world, origin, isolated
):
    root = await automatic_checkpoint(tmp_path, world)
    if origin == "retained-fork":
        image = await world.provider.snapshot("dummy-checkpoint")
        event = ForkRecord(
            root,
            root._rollout_dir,
            "kept-fork",
            root._cursor,
            frozenset({"sandbox"}),
            1,
            ["kept"],
        )
        event.captured(StageSnapshot(environment_ref=None, sandbox_ref=image))
        event.record["snapshot"]["retention"] = "kept"
        event.persist()
    source = load_checkpoint_source(root._rollout_dir, None)
    assert source.fork_id == ("kept-fork" if origin == "retained-fork" else "prompt:1")
    plan = _plan(
        tmp_path,
        task_paths=[world.task],
        sandbox="daytona",
        source=source,
        checkpoint_after=0,
        parent_mode="discard",
        isolate_children=isolated,
    )
    outcome = await run_branch_trial(plan, world.task, rollout_factory=world.factory)
    assert outcome.error is None
    assert world.admitted and all(META in image.meta for image in world.admitted)
    recorded = json.loads(
        (ScriptedRollout.last._rollout_dir / "checkpoint_source.json").read_text()
    )
    assert recorded["meta"] == source.meta == world.admitted[0].meta
    fork = json.loads((ScriptedRollout.last._rollout_dir / "tree.json").read_text())[
        "forks"
    ][0]
    assert fork["snapshot"]["sandbox"]["meta"] == source.meta


@pytest.mark.parametrize("isolated", [False, True])
async def test_automatic_checkpoint_reuse_keeps_admission_state(
    tmp_path, world, isolated
):
    plan = _plan(
        tmp_path,
        task_paths=[world.task],
        sandbox="daytona",
        isolate_children=isolated,
        checkpoints=parse_checkpoint_policy("every-prompt"),
    )
    outcome = await run_branch_trial(plan, world.task, rollout_factory=world.factory)
    assert outcome.error is None
    assert world.admitted and all(META in image.meta for image in world.admitted)
    fork = json.loads((ScriptedRollout.last._rollout_dir / "tree.json").read_text())[
        "forks"
    ][0]
    assert fork["snapshot"]["reused"] is True
    assert fork["snapshot"]["sandbox"]["meta"] == world.admitted[0].meta


async def test_retry_roundtrips_receipt_into_actual_child_admission(tmp_path, world):
    root = await automatic_checkpoint(tmp_path, world)
    result = RolloutResult(task_name="task", rewards={"reward": 0.0})
    await run_checkpoint_retry(
        root, result, parse_retry_policy("on-failure", prompt=None)
    )
    assert result.retry["status"] == "completed"
    assert result.retry["reward"] == 1.0
    assert result.rewards == {"reward": 0.0}
    assert world.admitted and META in world.admitted[0].meta
    fork = json.loads((root._rollout_dir / "tree.json").read_text())["forks"][0]
    assert fork["snapshot"]["sandbox"]["meta"] == world.admitted[0].meta


@pytest.mark.parametrize("provider", ["daytona", "docker", "remote_docker"])
async def test_retention_deletion_preserves_opaque_provider_metadata(
    tmp_path, provider
):
    image = SandboxImage(provider, "dummy-checkpoint", {"opaque": "provider-state"})
    sandbox = SimpleNamespace(
        supports_snapshot=True,
        snapshot=AsyncMock(return_value=image),
        delete_snapshot=AsyncMock(return_value=True),
    )
    rollout = SimpleNamespace(
        _rollout_dir=tmp_path,
        _env=sandbox,
        _config=SimpleNamespace(
            checkpoints=parse_checkpoint_policy("every-prompt", keep=1)
        ),
    )
    await after_prompt(rollout, 1)
    await after_prompt(rollout, 2)
    sandbox.delete_snapshot.assert_awaited_once_with(image)
    assert load_checkpoints(tmp_path)[1]["meta"] == image.meta


@pytest.mark.parametrize("entry", ["restore", "branch"])
@pytest.mark.parametrize(
    "meta",
    [
        {},
        {META: "null"},
        {META: "broken"},
        {META: "{}"},
        # DAYTONA-GAPS rule 2: a loader value in the receipt is carried aside
        # (tests/test_daytona_toolbox_startup.py), so only malformed receipts
        # are refused here: a name missing, an added name, a non-string, a NUL.
        {META: json.dumps(dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS[1:], ""))},
        {META: json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "X": ""})},
        {META: json.dumps({**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "ENV": 1})},
        {
            META: json.dumps(
                {**dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""), "ENV": "a\0b"}
            )
        },
    ],
)
async def test_invalid_persisted_receipts_still_refuse_before_provider_call(
    tmp_path, monkeypatch, entry, meta
):
    env, creates = daytona(monkeypatch)
    live = env._sandbox
    # Include a disk boundary so malformed receipts cannot gain provenance.
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"provider": "daytona", "ref": "dummy", "meta": meta}))
    image = SandboxImage(**json.loads(path.read_text()))
    with pytest.raises(RuntimeError, match="startup state is missing or unsupported"):
        if entry == "restore":
            await env.restore(image)
        else:
            env.start_from_snapshot(image)
    assert creates == []
    live.delete.assert_not_awaited()


@pytest.mark.parametrize("meta", [None, [], "receipt", 1, {META: None}, {META: {}}])
@pytest.mark.parametrize("origin", ["automatic", "retained-fork"])
async def test_malformed_metadata_containers_cannot_become_checkpoint_sources(
    tmp_path, world, meta, origin
):
    root = await automatic_checkpoint(tmp_path, world)
    if origin == "automatic":
        path = root._rollout_dir / "checkpoints.json"
        document = json.loads(path.read_text())
        document["checkpoints"][0]["meta"] = meta
    else:
        path = root._rollout_dir / "tree.json"
        document = {
            "forks": [
                {
                    "id": "kept",
                    "snapshot": {
                        "retention": "kept",
                        "captured_layers": ["sandbox"],
                        "sandbox": {
                            "provider": "daytona",
                            "ref": "dummy",
                            "meta": meta,
                        },
                    },
                }
            ],
        }
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid sandbox checkpoint metadata"):
        load_checkpoint_source(root._rollout_dir, None)
    assert world.creates == world.admitted == []


@pytest.mark.parametrize("origin", ["automatic", "retained-fork"])
async def test_legacy_docker_checkpoint_sources_remain_usable(tmp_path, world, origin):
    root = await automatic_checkpoint(tmp_path, world)
    path = root._rollout_dir / "checkpoints.json"
    document = json.loads(path.read_text())
    row = document["checkpoints"][0]
    row["provider"] = "docker"
    row.pop("meta", None)
    path.write_text(json.dumps(document))
    if origin == "retained-fork":
        (root._rollout_dir / "tree.json").write_text(
            json.dumps(
                {
                    "forks": [
                        {
                            "id": "kept",
                            "snapshot": {
                                "retention": "kept",
                                "captured_layers": ["sandbox"],
                                "sandbox": {"provider": "docker", "ref": row["ref"]},
                            },
                        }
                    ]
                }
            )
        )
    source = load_checkpoint_source(root._rollout_dir, None)
    assert source.provider == "docker" and source.meta == {}
