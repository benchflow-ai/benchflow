"""The Codex model catalogs BenchFlow ships for first-party providers.

Codex takes a model's tools, instructions and reasoning levels from its model
catalog. A model the pinned Codex does not bundle runs on fallback metadata
(plain function tools, no apply_patch), which BenchFlow refuses as an
``agent_model_catalog`` integration failure. DeepSeek publishes a Codex
catalog for its models in its Codex guide; BenchFlow ships those bytes
unchanged, points Codex at them with ``model_catalog_json`` and installs them
in the sandbox before Codex starts, in a root-owned folder the agent cannot
rewrite between turns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.agents import codex_provider_catalogs as catalogs
from benchflow.agents.codex_config import (
    CODEX_CONFIG_ENV,
    apply_codex_launch_config,
    apply_codex_provider_config,
    codex_home_config,
)
from benchflow.agents.codex_connector_policy import enforce_codex_apps_policy
from benchflow.agents.codex_models import codex_model_efforts
from benchflow.agents.env import resolve_agent_env
from benchflow.diagnostics import IntegrationFailureDiagnostic, RolloutDiagnostics
from benchflow.effort_request import (
    EffortRequestError,
    refuse_known_downgrade,
    refuse_unsupported_effort,
    settle_effort,
)
from benchflow.native_harness.codex import codex_launch
from benchflow.native_harness.spec import NativeTurn
from benchflow.rollout import Rollout, _classify_model_catalog_fallback
from benchflow.trajectories.model_calls import write_model_calls

DEEPSEEK_GUIDE = "https://api-docs.deepseek.com/quick_start/agent_integrations/codex/"


def _deepseek():
    return catalogs.CATALOGS["deepseek"]


# ── The shipped bytes ───────────────────────────────────────────────────────


def test_the_deepseek_catalog_is_the_published_bytes():
    catalog = _deepseek()
    data = catalog.data()
    assert hashlib.sha256(data).hexdigest() == catalog.sha256
    assert catalog.url == DEEPSEEK_GUIDE
    assert catalog.fetched == "2026-11-18"
    assert catalog.minimal_client_version == "0.164.0"
    models = json.loads(data)["models"]
    assert [m["slug"] for m in models] == ["deepseek-flash", "deepseek-v4-pro"]
    flash = catalog.entries()["deepseek-flash"]
    assert flash["default_reasoning_level"] == "high"
    assert [x["effort"] for x in flash["supported_reasoning_levels"]] == [
        "low",
        "high",
        "max",
    ]
    assert flash["apply_patch_tool_type"] == "freeform"
    assert catalog.install_path == (
        f"/etc/codex/model-catalogs/deepseek-{catalog.sha256[:12]}.json"
    )


def test_changed_catalog_bytes_are_refused(monkeypatch):
    catalog = _deepseek()
    monkeypatch.setattr(
        catalogs,
        "CATALOGS",
        {"deepseek": dataclasses.replace(catalog, sha256="0" * 64)},
    )
    with pytest.raises(catalogs.CodexModelCatalogError, match="sha256"):
        catalogs.CATALOGS["deepseek"].data()


@pytest.mark.parametrize(
    ("model", "found"),
    [
        ("deepseek/deepseek-flash", "deepseek"),
        ("deepseek-flash", "deepseek"),
        ("deepseek/deepseek-v4-pro", "deepseek"),
        ("deepseek/deepseek-flash[high]", "deepseek"),
        # The catalog describes DeepSeek's own endpoint, not another route
        # serving the same name.
        ("openai/deepseek-flash", None),
        ("deepseek/deepseek-chat", None),
        ("gpt-5.5", None),
        (None, None),
    ],
)
def test_a_catalog_is_found_by_provider_and_slug(model, found):
    catalog = catalogs.codex_provider_catalog(model)
    assert (catalog.provider if catalog else None) == found


# ── Codex is pointed at it, in both harnesses ───────────────────────────────


def _codex_env(model: str) -> dict[str, str]:
    return resolve_agent_env("codex-acp", model, {"DEEPSEEK_API_KEY": "ds-test"})


def _config(env: dict[str, str]) -> dict:
    return json.loads(env[CODEX_CONFIG_ENV])


def test_codex_config_names_the_shipped_catalog():
    env = _codex_env("deepseek/deepseek-flash")
    assert _config(env)["model_catalog_json"] == _deepseek().install_path
    # The model proxy's rewrite of the provider keeps it.
    apply_codex_provider_config(
        env,
        base_url="http://172.17.0.1:4000/v1",
        model="deepseek-flash",
        provider_name="litellm",
        strict=True,
    )
    assert _config(env)["model_catalog_json"] == _deepseek().install_path
    assert f'model_catalog_json = "{_deepseek().install_path}"' in codex_home_config(
        _config(env)
    )


def test_a_catalog_the_run_names_itself_is_kept():
    env = resolve_agent_env(
        "codex-acp",
        "deepseek/deepseek-flash",
        {
            "DEEPSEEK_API_KEY": "ds-test",
            CODEX_CONFIG_ENV: json.dumps({"model_catalog_json": "/opt/mine.json"}),
        },
    )
    assert _config(env)["model_catalog_json"] == "/opt/mine.json"


def test_the_acp_launch_carries_it():
    """codex-acp's app-server wrapper turns every CODEX_CONFIG key into a
    global ``-c``; the launch keeps the key."""
    env = _codex_env("deepseek/deepseek-flash")
    launch_env, _ = apply_codex_launch_config(
        "codex-acp", env, model="deepseek/deepseek-flash", reasoning_effort="high"
    )
    assert _config(launch_env)["model_catalog_json"] == _deepseek().install_path


def test_the_native_launch_carries_it():
    env = _codex_env("deepseek/deepseek-flash")
    argv = codex_launch(NativeTurn(cwd="/app"), _config(env)).argv
    assert f'model_catalog_json="{_deepseek().install_path}"' in argv
    assert (
        argv[argv.index(f'model_catalog_json="{_deepseek().install_path}"') - 1] == "-c"
    )


# ── The install in the sandbox ──────────────────────────────────────────────


class FakeSandbox:
    """Records commands and uploads; answers the read-back with *readback*."""

    def __init__(self, readback: str | None = None, fail: str | None = None):
        self.commands: list[tuple[str, str | None]] = []
        self.uploads: list[tuple[bytes, str, str | None, str]] = []
        self.readback = readback
        self.fail = fail

    async def exec(self, command, user=None, timeout_sec=None, env=None, **_):
        self.commands.append((command, user))
        rc, out = 0, ""
        if self.fail and self.fail in command:
            rc = 1
        elif "sha256sum" in command:
            out = f"{self.readback}\n"
        return SimpleNamespace(return_code=rc, stdout=out, stderr="")

    async def upload_file(self, source, target, mode=None, admission=None, **_):
        with open(source, "rb") as f:
            self.uploads.append((f.read(), str(target), mode, admission.reason))


async def test_the_install_writes_the_bytes_and_reads_them_back(tmp_path):
    catalog = _deepseek()
    env = _codex_env("deepseek/deepseek-flash")
    sandbox = FakeSandbox(readback=catalog.sha256)

    receipt = await catalogs.install_codex_model_catalog(
        sandbox, env, rollout_dir=tmp_path
    )

    assert len(sandbox.uploads) == 1
    data, target, _, reason = sandbox.uploads[0]
    assert data == catalog.data()
    assert "Codex model catalog" in reason
    stage = target.rsplit("/", 1)[0]
    assert stage.startswith("/.runtime-codex-catalog-")
    assert all(user == "root" for _, user in sandbox.commands)
    install = next(c for c, _ in sandbox.commands if "sha256sum" in c)
    for folder in ("/", "/etc", "/etc/codex", "/etc/codex/model-catalogs"):
        assert f"t {folder} " in install or f"t {folder};" in install
    assert catalog.install_path in install
    assert any(f"rmdir {stage}" in c for c, _ in sandbox.commands)
    assert receipt["path"] == catalog.install_path
    assert receipt["sha256"] == catalog.sha256
    assert json.loads((tmp_path / "codex_model_catalog.json").read_text()) == receipt


async def test_a_changed_read_back_stops_the_launch(tmp_path):
    env = _codex_env("deepseek/deepseek-flash")
    sandbox = FakeSandbox(readback="f" * 64)
    with pytest.raises(catalogs.CodexModelCatalogError, match="sha256"):
        await catalogs.install_codex_model_catalog(sandbox, env, rollout_dir=tmp_path)
    assert not (tmp_path / "codex_model_catalog.json").exists()


async def test_an_untrusted_folder_stops_the_launch(tmp_path):
    env = _codex_env("deepseek/deepseek-flash")
    sandbox = FakeSandbox(readback=_deepseek().sha256, fail="sha256sum")
    with pytest.raises(catalogs.CodexModelCatalogError):
        await catalogs.install_codex_model_catalog(sandbox, env, rollout_dir=tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model": "deepseek-chat"},
        {"model_catalog_json": "/opt/mine.json"},
    ],
)
async def test_no_shipped_catalog_installs_nothing(tmp_path, config):
    sandbox = FakeSandbox()
    env = {CODEX_CONFIG_ENV: json.dumps(config)}
    assert (
        await catalogs.install_codex_model_catalog(sandbox, env, rollout_dir=tmp_path)
        is None
    )
    assert sandbox.commands == [] and sandbox.uploads == []


@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_every_codex_connection_installs_it_before_launch(tmp_path, harness):
    """The step Codex connections already await before launch (both
    harnesses, every role) installs the catalog."""
    catalog = _deepseek()
    env = _codex_env("deepseek/deepseek-flash")
    sandbox = FakeSandbox(readback=catalog.sha256)
    await enforce_codex_apps_policy(
        sandbox,
        agent="codex-acp",
        agent_launch="",
        agent_env=env,
        sandbox_user="agent",
        policy="inherit",
        requested="inherit",
        rollout_dir=tmp_path,
        harness=harness,
    )
    assert [u[0] for u in sandbox.uploads] == [catalog.data()]
    assert json.loads((tmp_path / "codex_model_catalog.json").read_text())["path"] == (
        catalog.install_path
    )


@pytest.mark.parametrize(
    ("mode", "symlink", "trusted"),
    [
        (0o755, False, True),
        (0o555, False, True),
        (0o775, False, False),
        (0o757, False, False),
        (0o1777, False, False),
        (0o755, True, False),
    ],
)
def test_the_folder_check(tmp_path, mode, symlink, trusted):
    """The shell predicate the install runs on each folder: owned by the
    expected uid, no group or other write bit, not a symlink."""
    real = tmp_path / "real"
    real.mkdir()
    os.chmod(real, mode)
    path = tmp_path / "link" if symlink else real
    if symlink:
        path.symlink_to(real)
    script = catalogs.trusted_folder_function(str(os.getuid())) + f"; t {path}"
    result = subprocess.run(["sh", "-c", script], check=False)
    os.chmod(real, 0o755)
    assert (result.returncode == 0) is trusted


# ── Efforts come from the catalog ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "levels"),
    [
        ("deepseek/deepseek-flash", ("low", "high", "max")),
        ("deepseek-flash", ("low", "high", "max")),
        ("deepseek/deepseek-v4-pro", ("low", "high", "max")),
        ("openai/deepseek-flash", None),
        ("deepseek/deepseek-chat", None),
        ("gpt-5.5", ("low", "medium", "high", "xhigh")),
    ],
)
def test_codex_efforts_come_from_the_catalog(model, levels):
    assert codex_model_efforts(model) == levels


def test_codex_refuses_a_level_the_catalog_does_not_list():
    refuse_unsupported_effort("codex-acp", "deepseek/deepseek-flash", "max")
    with pytest.raises(EffortRequestError, match="it takes low, high, max"):
        refuse_unsupported_effort("codex-acp", "deepseek/deepseek-flash", "medium")


@pytest.mark.parametrize("agent", ["claude-agent-acp", "pi-acp", "opencode"])
def test_the_catalog_levels_bind_codex_alone(agent):
    """The catalog lists the levels Codex offers for the model. DeepSeek's API
    answers 200 to every level (low through ultra, none, minimal), so the
    catalog gives another agent on deepseek-flash no new refusal."""
    refuse_known_downgrade(agent, "deepseek/deepseek-flash", "medium")


def test_claude_on_deepseek_keeps_its_own_effort_answer():
    """``-a claude -m deepseek/deepseek-flash -e medium`` gets the answer it
    got before the catalog shipped: the gateway's DeepSeek route sends a
    thinking budget, not a level. The catalog's levels play no part."""
    with pytest.raises(EffortRequestError) as caught:
        settle_effort("claude-agent-acp", "deepseek/deepseek-flash", "medium", {})
    assert "thinking.budget_tokens" in str(caught.value)
    assert "low, high, max" not in str(caught.value)


# ── A model in no catalog is still refused ──────────────────────────────────


FALLBACK = [
    {"type": "function", "name": "exec_command"},
    {"type": "function", "name": "write_stdin"},
    {"type": "function", "name": "view_image"},
    {"type": "web_search"},
]


def test_a_model_in_no_catalog_still_fails_on_fallback(tmp_path):
    env = _codex_env("deepseek/deepseek-chat")
    assert "model_catalog_json" not in _config(env)
    trial = tmp_path / "t__1"
    (trial / "trajectory").mkdir(parents=True)
    row = {
        "request": {"body": {"model": "deepseek-chat", "tools": FALLBACK}},
        "response": {"status_code": 200, "body": {"status": "completed"}},
        "metadata": {"call_purpose": "agent"},
    }
    (trial / "trajectory/llm_trajectory.jsonl").write_text(json.dumps(row) + "\n")
    write_model_calls(trial)
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="codex-acp", primary_model="deepseek/deepseek-chat"
    )
    rollout._rollout_dir = trial
    rollout._diagnostics = RolloutDiagnostics()
    rollout._trajectory = []
    rollout._rewards = {"reward": 1.0}
    rollout._error = None
    rollout._bare_timeout = False
    rollout._judge_timed_out_agent = False
    assert _classify_model_catalog_fallback(rollout) is True
    info = rollout._diagnostics.get(IntegrationFailureDiagnostic.field)
    assert info.cause == "agent_model_catalog"
    assert rollout._rewards is None


# ── An upstream failure keeps its cause ─────────────────────────────────────


def _codex_trial(tmp_path, statuses, *, trajectory=()):
    trial = tmp_path / "t__2"
    (trial / "trajectory").mkdir(parents=True)
    rows = [
        {
            "request": {"body": {"model": "glm-5.3-flash", "tools": FALLBACK}},
            "response": {"status_code": status, "body": {"status": "completed"}},
            "metadata": {"call_purpose": "agent"},
        }
        for status in statuses
    ]
    (trial / "trajectory/llm_trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    write_model_calls(trial)
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="codex-acp", primary_model="vllm/glm-5.3-flash"
    )
    rollout._rollout_dir = trial
    rollout._diagnostics = RolloutDiagnostics()
    rollout._trajectory = list(trajectory)
    rollout._rewards = None
    rollout._error = "Native harness error (codex): unexpected status 404 Not Found"
    rollout._bare_timeout = False
    rollout._judge_timed_out_agent = False
    return rollout


NOTICE = {
    "type": "notice",
    "severity": "warning",
    "title": "Model metadata for `glm-5.3-flash` not found. Defaulting to fallback "
    "metadata; this can degrade performance and cause issues.",
}


@pytest.mark.parametrize("evidence", ["notice", "tools"])
@pytest.mark.parametrize("status", [404, 500])
def test_an_upstream_failure_keeps_its_cause(tmp_path, evidence, status):
    """Every agent call failed upstream (a route with no Responses API): the
    trial's error stays the provider's, not fallback model metadata."""
    rollout = _codex_trial(
        tmp_path,
        [status] * 3,
        trajectory=[NOTICE] if evidence == "notice" else [],
    )
    assert _classify_model_catalog_fallback(rollout) is False
    assert rollout._error.endswith("unexpected status 404 Not Found")
    assert rollout._diagnostics.get(IntegrationFailureDiagnostic.field) is None


def test_one_answered_call_still_shows_the_fallback(tmp_path):
    rollout = _codex_trial(tmp_path, [500, 200, 500], trajectory=[NOTICE])
    assert _classify_model_catalog_fallback(rollout) is True
    info = rollout._diagnostics.get(IntegrationFailureDiagnostic.field)
    assert info.cause == "agent_model_catalog"
