"""Regression tests for executing config_override branch deltas.

Guards "feat(branch): execute config_override deltas as fresh rollouts from
env-ready" (docs/rollout-branching-rfc.md §3.3; FrontierPhysics#73). PR number
to be added on submission.

``config_override`` was schema- and provenance-stable but fail-closed: the
C-axis overlay is deep-merged into the task's resolved config by ``setup()``,
which only a fresh child rollout re-runs. A child forked from the ``env-ready``
snapshot therefore executes the delta through the same fresh-rollout path the
skills ablation uses (``use_prebuilt_env``): its RolloutConfig is the parent's
with the overlay deep-merged through the EXISTING allowlisted machinery
(``benchflow._utils.config_override``, #790) — same allowlist, same fail-closed
rejection of scorer-touching keys, same content addressing.

These tests pin that the child really runs under the merged config (the
effective agent timeout its execution receives, not just a recorded label),
that a non-allowlisted key and every non-env-ready branch point fail closed
before any child runs, that the delta composes with ``skill_mode`` and
``injected_prompt`` on one child, and that provenance carries the overlay's
sha256 and its allowlisted keys.

Unit tests against fakes — no Docker, Daytona, or API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchflow._utils.config_override import overlay_hash
from benchflow.branch_delta import BranchDelta
from benchflow.branch_skill import child_skill_config
from benchflow.rollout import RolloutConfig
from benchflow.rollout_branch import BranchDeltaNotSupported
from tests.test_branch_skill_delta import (
    AGENT_EVENTS,
    FakePlanes,
    _capture_env_ready,
    _fake_verifier,
    _parent,
    _task_dir,
)

#: The one controlled change the tests fork under: a 7-second agent budget.
#: Small and distinctive — no TaskConfig default is 7, so an assertion on the
#: child's effective timeout cannot pass by coincidence.
OVERLAY = {"agent": {"timeout_sec": 7}}


class TimeoutRecordingPlanes(FakePlanes):
    """FakePlanes that records the effective agent timeout each child ran under.

    The overlay's observable effect: ``setup()`` resolves ``self._timeout``
    from the (merged) task config's ``agent.timeout_sec``, and ``execute()``
    hands exactly that to ``execute_prompts``. Recording it here measures the
    world the child ran in, not the label its config carries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[int] = []
        self.prompts: list[list[str]] = []

    async def execute_prompts(self, client, session, prompts, timeout, **kwargs):
        self.timeouts.append(int(timeout))
        self.prompts.append(list(prompts))
        return list(AGENT_EVENTS), 1


# 1. The delta executes: the child runs fresh under the merged config


async def test_config_delta_child_runs_fresh_with_the_merged_config(
    tmp_path: Path, monkeypatch
):
    """The config ablation, executed for real.

    The control child runs under the task's own agent budget; the delta child
    runs under the overlay's — asserted on the timeout its execution actually
    received, which only differs if the child's own setup() re-resolved the
    merged config. The parent's config is untouched afterwards.
    """
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    value = await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(config_override=OVERLAY)],
    )

    assert value == 1.0
    control_timeout, child_timeout = planes.timeouts
    assert child_timeout == 7
    assert control_timeout != 7  # the task's own default, not the overlay
    assert rollout._config.config_override is None  # the parent is untouched
    # each child re-installed for itself — the fresh-rollout path, not in-place
    assert len(planes.deployments) == 2


async def test_config_delta_child_config_json_records_the_merged_overlay(
    tmp_path: Path, monkeypatch
):
    """The child is a first-class rollout, so its own config.json carries the
    #790 overlay record — keys, sha256, and the patch itself."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(config_override=OVERLAY)],
    )

    children_dir = rollout._rollout_dir / "branches" / "root" / "children"
    control = json.loads((children_dir / "n2" / "config.json").read_text())
    child = json.loads((children_dir / "n3" / "config.json").read_text())
    assert "config_override" not in control
    assert child["config_override"] == {
        "keys": ["agent"],
        "sha256": overlay_hash(OVERLAY),
        "patch": OVERLAY,
    }


async def test_config_delta_merges_over_the_parents_own_overlay(
    tmp_path: Path, monkeypatch
):
    """A parent that ran under an overlay of its own hands the child both:
    dropping the parent's overlay would silently vary two things and label the
    comparison as one delta."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(
        _task_dir(tmp_path),
        tmp_path,
        planes=planes,
        config_override={"metadata": {"tags": ["parent-run"]}},
    )
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(config_override=OVERLAY)],
    )

    children_dir = rollout._rollout_dir / "branches" / "root" / "children"
    control = json.loads((children_dir / "n2" / "config.json").read_text())
    child = json.loads((children_dir / "n3" / "config.json").read_text())
    # the control child inherits the parent's overlay unchanged (zero delta)
    assert control["config_override"]["patch"] == {"metadata": {"tags": ["parent-run"]}}
    # the delta child runs under parent's overlay + the one recorded change
    assert child["config_override"]["patch"] == {
        "metadata": {"tags": ["parent-run"]},
        "agent": {"timeout_sec": 7},
    }
    assert sorted(child["config_override"]["keys"]) == ["agent", "metadata"]
    assert planes.timeouts[1] == 7


# 2. Combinations: one child, one config_override plus another executable field


async def test_config_delta_composes_with_skill_mode_on_one_child(
    tmp_path: Path, monkeypatch
):
    """config_override + skill_mode are both fields on the child's config, so
    one child can carry both: it deploys the switched pack AND runs under the
    merged budget."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            BranchDelta(skill_mode="no-skill"),
            BranchDelta(skill_mode="with-skill", config_override=OVERLAY),
        ],
    )

    no_skill, with_skill = planes.deployments
    assert no_skill["skills_dir"] is None
    assert with_skill["skill_files"] == ["demo"]
    assert planes.timeouts[0] != 7  # the pure-skill arm keeps the task budget
    assert planes.timeouts[1] == 7  # the combined arm runs under the overlay


async def test_config_delta_composes_with_an_injected_prompt_on_one_child(
    tmp_path: Path, monkeypatch
):
    """config_override + injected_prompt: the injection rides the fresh-child
    prompt path (the child's continuation prompt) while the overlay rides the
    config — neither swallows the other."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[
            None,
            BranchDelta(config_override=OVERLAY, injected_prompt="Follow PLAN.md."),
        ],
    )

    assert planes.prompts == [["Solve the task."], ["Follow PLAN.md."]]
    assert planes.timeouts == [planes.timeouts[0], 7]


# 3. Fail closed before any child runs


async def test_a_non_allowlisted_key_fails_before_any_child_runs(tmp_path: Path):
    """The same fail-closed allowlist as the run-level overlay (#790): a
    scorer-touching patch dies at delta validation, with nothing quiesced,
    restored, or forked — never at child setup after a snapshot was consumed."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await _capture_env_ready(rollout)

    with pytest.raises(ValueError, match="verifier") as excinfo:
        await rollout.branch_at_stage(
            "env-ready",
            2,
            deltas=[
                None,
                BranchDelta(config_override={"verifier": {"timeout_sec": 1}}),
            ],
        )

    assert "deltas[1].config_override" in str(excinfo.value)
    assert rollout._env.restored == []
    assert rollout._environment.restored == []
    assert planes.deployments == []
    assert [node for node in rollout.tree.nodes() if "delta" in node.state] == []


async def test_config_delta_at_a_cursor_branch_names_env_ready(tmp_path: Path):
    """A cursor branch forks after the parent's setup() consumed its config —
    fail closed naming the boundary that works."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    env = rollout._environment

    with pytest.raises(BranchDeltaNotSupported, match="config_override") as excinfo:
        await rollout.branch(
            2,
            snapshot_layers={"environment", "sandbox"},
            deltas=[None, BranchDelta(config_override=OVERLAY)],
        )

    assert "env-ready" in str(excinfo.value)
    assert "setup()" in str(excinfo.value)
    assert env.snapshots == []
    assert rollout._cursor.children == []


async def test_config_delta_at_another_stage_names_env_ready(tmp_path: Path):
    """pre-verify is a recorded boundary too — and still the wrong one: the
    state the config governs was already consumed by then."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await rollout.mark_stage("pre-verify", snapshot_layers={"environment", "sandbox"})

    with pytest.raises(BranchDeltaNotSupported, match="'env-ready'") as excinfo:
        await rollout.branch_at_stage(
            "pre-verify",
            2,
            deltas=[None, BranchDelta(config_override=OVERLAY)],
        )

    assert "'pre-verify'" in str(excinfo.value)
    assert planes.deployments == []


async def test_config_delta_with_an_explicit_run_child_is_rejected(tmp_path: Path):
    """A caller-supplied runner owns the child's execution, so the engine
    cannot run setup() under the merged config."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    await _capture_env_ready(rollout)

    async def run_child(child):
        return 1.0

    with pytest.raises(ValueError, match="run_child"):
        await rollout.branch_at_stage(
            "env-ready",
            2,
            run_child=run_child,
            deltas=[None, BranchDelta(config_override=OVERLAY)],
        )

    assert planes.deployments == []


# 4. Provenance: the sha and the allowlisted keys flow into the lineage


async def test_provenance_carries_the_overlay_sha_and_keys(tmp_path: Path, monkeypatch):
    """Per-child provenance.json and tree.json record the overlay as its
    sha256 (#790's exact hash) plus the allowlisted keys it patched — never
    the raw patch, which lives only in the child's own config.json."""
    planes = TimeoutRecordingPlanes()
    rollout = _parent(_task_dir(tmp_path), tmp_path, planes=planes)
    _fake_verifier(monkeypatch)
    await _capture_env_ready(rollout)

    await rollout.branch_at_stage(
        "env-ready",
        2,
        deltas=[None, BranchDelta(config_override=OVERLAY)],
    )

    run_dir = rollout._rollout_dir
    children_dir = run_dir / "branches" / "root" / "children"
    control = json.loads((children_dir / "n2" / "provenance.json").read_text())
    child = json.loads((children_dir / "n3" / "provenance.json").read_text())
    assert control["delta"]["config_override_sha256"] is None
    assert "config_override_keys" not in control["delta"]  # zero-delta shape
    assert child["delta"]["config_override_sha256"] == overlay_hash(OVERLAY)
    assert child["delta"]["config_override_keys"] == ["agent"]
    assert child["delta_execution"] == "fresh-rollout"
    nodes = {
        node["id"]: node
        for node in json.loads((run_dir / "tree.json").read_text())["nodes"]
    }
    assert nodes["n3"]["delta"]["config_override_sha256"] == overlay_hash(OVERLAY)
    assert nodes["n3"]["delta"]["config_override_keys"] == ["agent"]


# 5. The derived child config


def test_child_config_merges_the_delta_overlay_over_the_parents(tmp_path: Path):
    """The merge lives in child_skill_config so every fresh child derives its
    config the same way: delta keys win, untouched parent keys survive."""
    task = _task_dir(tmp_path)
    parent = RolloutConfig.from_legacy(
        task_path=task,
        agent="oracle",
        jobs_dir=tmp_path / "jobs",
        config_override={"agent": {"timeout_sec": 30}, "metadata": {"x": 1}},
    )

    child = child_skill_config(
        parent,
        skill_mode="no-skill",
        jobs_dir=tmp_path / "jobs",
        job_name="children",
        rollout_name="n2",
        config_override={"agent": {"timeout_sec": 7}},
    )

    assert child.config_override == {
        "agent": {"timeout_sec": 7},
        "metadata": {"x": 1},
    }
    # the parent's dict is not mutated, and no-delta children inherit it as-is
    assert parent.config_override == {
        "agent": {"timeout_sec": 30},
        "metadata": {"x": 1},
    }
    inherited = child_skill_config(
        parent,
        skill_mode="no-skill",
        jobs_dir=tmp_path / "jobs",
        job_name="children",
        rollout_name="n3",
    )
    assert inherited.config_override == parent.config_override


def test_provenance_dict_records_keys_only_when_the_field_is_set() -> None:
    """``config_override_keys`` joins the provenance block only for a set
    overlay, so every other delta keeps the exact dict shape the RFC pinned."""
    unset: dict[str, Any] = BranchDelta(skill_mode="no-skill").provenance_dict()
    assert "config_override_keys" not in unset
    provenance = BranchDelta(
        config_override={"metadata": {}, "agent": {"timeout_sec": 7}}
    ).provenance_dict()
    assert provenance["config_override_keys"] == ["agent", "metadata"]
