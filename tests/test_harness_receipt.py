"""Installed harness versions: recorded per trial, compared on resume, kept
apart in preference arms, and never read back from the agent's folder."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.receipt import (
    install_identity,
    probe_receipt,
    receipt_matches_install,
)
from benchflow.batch import Results
from benchflow.evaluation import EvaluationConfig, _trial_settings_differ
from benchflow.models import RolloutResult
from benchflow.preferences.store import _arm, issue
from benchflow.rollout._results import _build_rollout_result


def receipt(version):
    return {
        "identity": {"agent": "codex-acp", "harness": "acp"},
        "engine": {"version": version},
    }


def _installed(package, version):
    return {"package": package, "version": version, "source": "installed-package"}


def _build(trial, **kwargs):
    return _build_rollout_result(
        rollout_dir=trial,
        task_name="t",
        rollout_name="t__1",
        agent="codex-acp",
        agent_name="codex",
        model="m",
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
        **kwargs,
    )


def test_result_roundtrip_keeps_receipt(tmp_path):
    value = receipt("0.159.3")
    result = _build(tmp_path, harness_versions=value)
    assert result.harness_versions == value
    assert (
        json.loads((tmp_path / "result.json").read_text())["harness_versions"] == value
    )
    assert RolloutResult.load(tmp_path).harness_versions == value


def test_a_result_built_again_takes_the_receipt_config_json_recorded(tmp_path):
    """A later process (a scoring resume) has no probe of its own."""
    value = receipt("0.159.3")
    (tmp_path / "config.json").write_text(json.dumps({"harness_versions": value}))
    assert _build(tmp_path).harness_versions == value


def test_an_agent_that_writes_garbage_at_its_receipt_breaks_nothing(tmp_path):
    """The agent can write /logs/agent/runtime.json during its run."""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "runtime.json").write_text("not json")
    value = receipt("0.159.3")
    assert _build(tmp_path, harness_versions=value).harness_versions == value


def test_an_agent_link_at_its_receipt_never_reaches_the_records(tmp_path):
    """A link the agent leaves at /logs/agent/runtime.json was followed on the
    host, copying a host file into result.json and config.json."""
    secret = tmp_path / "host-only" / "credentials.json"
    secret.parent.mkdir()
    secret.write_text(json.dumps({"token": "host-secret-value"}))
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    os.symlink(secret, trial / "agent" / "runtime.json")
    from benchflow.rollout._results import _record_harness_versions_in_config

    (trial / "config.json").write_text("{}")
    value = receipt("0.159.3")
    _record_harness_versions_in_config(trial, value)
    _build(trial, harness_versions=value)
    for name in ("result.json", "config.json"):
        assert "host-secret-value" not in (trial / name).read_text()
    assert json.loads((trial / "config.json").read_text())["harness_versions"] == value
    # Without a probe, nothing is read from the agent's folder either.
    _build(trial / "unprobed")
    assert "host-secret-value" not in (trial / "unprobed" / "result.json").read_text()


def test_resume_rejects_different_installed_engine(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"agent": "codex-acp", "harness_versions": receipt("0.1")})
    )
    (tmp_path / "result.json").write_text('{"rewards": {"reward": 1}}')
    assert "harness_versions" in (
        _trial_settings_differ(tmp_path, EvaluationConfig(agent="codex-acp")) or ""
    )


def test_preference_versions_make_distinct_arms_and_pairs(tmp_path):
    a = {"agent": "codex-acp", "model": "m", "harness_versions": receipt("0.1")}
    b = {**a, "harness_versions": receipt("0.2")}
    assert _arm(a, "agent+model") != _arm(b, "agent+model")
    for index, item in enumerate([a, b]):
        trial = tmp_path / "job" / f"t__{index}"
        (trial / "artifacts").mkdir(parents=True)
        (trial / "artifacts/out.txt").write_text("answer")
        (trial / "result.json").write_text(
            json.dumps(
                {
                    **item,
                    "task_name": "t",
                    "task_digest": "same",
                    "finished_at": "2026-11-04",
                    "rewards": {"reward": 1},
                }
            )
        )
        (trial / "config.json").write_text("{}")
    issue([tmp_path / "job"], tmp_path / "store")
    assert len((tmp_path / "store/pairs.v1.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_the_probe_returns_observed_versions_and_writes_nothing(tmp_path):
    observed = {
        role: _installed("pkg", "9.8.7") for role in ("adapter", "cli", "engine")
    }
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout=json.dumps(observed))
        )
    )
    value = await probe_receipt(env, "codex-acp", "acp")
    assert value["engine"]["version"] == "9.8.7"
    assert value["identity"] == install_identity("codex-acp")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        AttributeError("'SimpleNamespace' object has no attribute 'return_code'"),
        RuntimeError("sandbox gone"),
        SimpleNamespace(return_code=0, stdout="not json"),
        SimpleNamespace(return_code=0, stdout="[1, 2]"),
        SimpleNamespace(return_code=127, stdout=""),
    ],
    ids=["no-return-code", "exec-raises", "not-json", "not-an-object", "no-node"],
)
async def test_a_probe_that_fails_leaves_the_versions_unknown(answer):
    """The probe is optional: whatever it meets, the trial goes on."""
    run = AsyncMock(side_effect=answer if isinstance(answer, Exception) else None)
    if not isinstance(answer, Exception):
        run.return_value = answer
    value = await probe_receipt(SimpleNamespace(exec=run), "codex-acp", "acp")
    assert all(
        value[role] == {"version": None, "source": "unavailable"}
        for role in ("adapter", "cli", "engine")
    )
    assert receipt_matches_install(value, "codex-acp", "acp")


@pytest.mark.asyncio
async def test_the_rollout_keeps_its_probe_and_records_it(tmp_path):
    """config.json and the reader's copy in agent/ come from the rollout's
    own receipt; the copy is written through the reachable-folder helper."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task.paths import RolloutPaths

    observed = {
        role: _installed("pkg", "1.2.3") for role in ("adapter", "cli", "engine")
    }
    (tmp_path / "task").mkdir()
    rollout = Rollout(
        RolloutConfig.from_legacy(task_path=tmp_path / "task", agent="codex-acp")
    )
    rollout._env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout=json.dumps(observed))
        )
    )
    trial = tmp_path / "trial"
    rollout._rollout_paths = RolloutPaths(trial)
    rollout._rollout_paths.mkdir()
    rollout._rollout_dir = trial
    (trial / "config.json").write_text(json.dumps({"agent": "codex-acp"}))
    await rollout._record_harness_versions("codex-acp")
    value = rollout._harness_versions
    assert value is not None and value["cli"]["version"] == "1.2.3"
    assert json.loads((trial / "config.json").read_text()) == {
        "agent": "codex-acp",
        "harness_versions": value,
    }
    assert json.loads((trial / "agent" / "runtime.json").read_text()) == value


def test_probe_reads_actual_nested_engine(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from benchflow.agents import registry
    from benchflow.agents.receipt import probe_command

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the installed-metadata probe")
    prefix = tmp_path / "js-agents"
    modules = prefix / "lib/node_modules"
    for name, version in [
        ("@agentclientprotocol/codex-acp", "2.0.1"),
        ("@openai/codex", "0.159.3"),
        ("@agentclientprotocol/codex-acp/node_modules/@openai/codex", "0.159.9"),
    ]:
        folder = modules / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "package.json").write_text(
            json.dumps(
                {
                    "name": "@openai/codex"
                    if "codex-acp/node_modules" in name
                    else name,
                    "version": version,
                }
            )
        )
    monkeypatch.setattr(
        registry, "_BENCHFLOW_NODE_PREFIX", str(Path(node).resolve().parent.parent)
    )
    monkeypatch.setattr(registry, "_BENCHFLOW_JS_AGENT_PREFIX", str(prefix))
    command = probe_command("codex-acp")
    assert command is not None
    observed = json.loads(subprocess.check_output(command, shell=True, text=True))
    assert observed["adapter"]["version"] == "2.0.1"
    # codex-acp's install puts its pinned engine in the adapter's own
    # node_modules (registry._codex_acp_install): that is the one it runs.
    assert observed["engine"]["version"] == "0.159.9"
    command = probe_command("codex-acp", "native")
    assert command is not None
    native = json.loads(subprocess.check_output(command, shell=True, text=True))
    assert native["cli"]["version"] == "0.159.3"


def test_resume_accepts_equal_pins_and_rejects_engine_drift(tmp_path):
    """Compare observed versions even with the same installer."""
    from benchflow.agents.registry import pinned_npm_package

    name = "claude-agent-acp"
    value = {"identity": install_identity(name)}
    for role, client in [
        ("adapter", name),
        ("cli", "claude-code"),
        ("engine", "claude-code"),
    ]:
        package, version = pinned_npm_package(client)
        value[role] = _installed(package, version)
    config = EvaluationConfig(agent=name)
    (tmp_path / "result.json").write_text('{"rewards": {"reward": 1}}')
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": name, "harness_versions": value}))
    assert _trial_settings_differ(tmp_path, config) is None
    value["engine"]["version"] = "0.0.1"
    path.write_text(json.dumps({"agent": name, "harness_versions": value}))
    assert "harness_versions" in (_trial_settings_differ(tmp_path, config) or "")


def test_an_unchanged_gemini_install_is_reused_on_resume():
    """gemini's engine (@google/gemini-cli-core) comes with its CLI and is
    never pinned: it was compared against the installer and never matched,
    so every resumed gemini trial was set aside and rerun."""
    value = {
        "schema": 1,
        "identity": install_identity("gemini"),
        "adapter": _installed("@google/gemini-cli", "0.42.0"),
        "cli": _installed("@google/gemini-cli", "0.42.0"),
        "engine": _installed("@google/gemini-cli-core", "0.42.0"),
    }
    assert receipt_matches_install(value, "gemini", "acp")
    value["cli"] = _installed("@google/gemini-cli", "0.41.0")
    assert not receipt_matches_install(value, "gemini", "acp")


def test_a_failed_probe_is_unknown_not_a_change():
    value = {
        "schema": 1,
        "identity": install_identity("codex-acp"),
        **{
            role: {"version": None, "source": "unavailable"}
            for role in ("adapter", "cli", "engine")
        },
    }
    assert receipt_matches_install(value, "codex-acp", "acp")
    value["identity"] = {**value["identity"], "install_sha256": "0" * 64}
    assert not receipt_matches_install(value, "codex-acp", "acp")


def test_csv_and_records_flatten_the_receipt(tmp_path):
    """EvaluationResult.to_csv raised 'dict contains fields not in
    fieldnames: harness_versions' once results carried a receipt."""
    value = {
        "schema": 1,
        "identity": install_identity("codex-acp"),
        "adapter": _installed("@agentclientprotocol/codex-acp", "2.0.1"),
        "cli": _installed("@openai/codex", "0.159.3"),
        "engine": {
            "package": "@openai/codex",
            "version": None,
            "source": "unavailable",
        },
    }
    results = Results(
        [
            RolloutResult("t1", rewards={"reward": 1.0}, harness_versions=value),
            RolloutResult("t2", rewards={"reward": 0.0}),
        ]
    )
    record = results[0].to_record()
    assert record["harness_adapter"] == "@agentclientprotocol/codex-acp@2.0.1"
    assert record["harness_cli"] == "@openai/codex@0.159.3"
    assert record["harness_engine"] is None
    assert not any(isinstance(v, dict | list) for v in record.values())
    rows = list(csv.DictReader(results.to_csv(tmp_path / "r.csv").open()))
    assert rows[0]["harness_cli"] == "@openai/codex@0.159.3"
    assert rows[1]["harness_cli"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_scripted_trials_record_no_applicable_harness(harness):
    """Scripted trials need no executable version probe."""
    env = SimpleNamespace(exec=AsyncMock())
    value = await probe_receipt(env, "oracle", harness)
    assert all(
        value[role]["source"] == "not-applicable"
        for role in ("adapter", "cli", "engine")
    )
    assert receipt_matches_install(value, "oracle", harness)
    env.exec.assert_not_called()
