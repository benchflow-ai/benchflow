"""The agent install split: the cacheable part from the layer, the rest per trial.

``install_agent`` used to run the registry's whole install in every trial. The
cacheable part (system packages, Node, the agent's pinned package) now comes
from the agent's cached image layer (sandbox.env_layers) when the sandbox
started from one; the sandbox user, credentials, skills and lockdown still run
in every trial. These tests drive the real install planes against a sandbox
that lists every step it is sent.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents import registry
from benchflow.agents.install import (
    LAYER_STAMP_DIR,
    AgentLayerPlan,
    agent_layer_plan,
    install_from_layer,
)
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENTS,
    CACHEABLE_INSTALLS,
    cacheable_install,
)
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.env_layers import (
    AGENT,
    FRESH_INSTALL_ENV,
    BaseImage,
    LayerResult,
    fresh_install_requested,
)
from benchflow.task.config import HealthcheckConfig, SandboxConfig, SetupCommandConfig
from tests._receipt_fakes import receipt_aware

# --- the split markers in the registry ----------------------------------------

PINNED_SPECS = {
    "claude-agent-acp": (
        "@agentclientprotocol/claude-agent-acp@0.81.2",
        "@anthropic-ai/claude-code@2.1.316",
    ),
    "codex-acp": ("@agentclientprotocol/codex-acp@2.0.1",),
    "gemini": ("@google/gemini-cli@0.42.0",),
    "opencode": ("opencode-ai@1.19.5",),
    "mimo": ("@mimo-ai/cli@0.1.15",),
    # AG moved agy from 1.2.7 to 1.3.5 (antigravity_config.AGY_RELEASES).
    "antigravity": ("1.3.5",),
}


def test_every_agent_with_a_pinned_install_is_cacheable():
    assert set(CACHEABLE_INSTALLS) == set(PINNED_SPECS)
    for agent, specs in PINNED_SPECS.items():
        version, install = cacheable_install(agent) or ("", "")
        assert install == AGENTS[agent].install_cmd
        for spec in specs:
            assert spec in install, (agent, spec)
            assert spec.rsplit("@", 1)[-1] in version, (agent, version)


@pytest.mark.parametrize(
    "agent", ["pi-acp", "harvey-lab-harness", "deepagents", "openhands"]
)
def test_an_unpinned_install_runs_in_every_trial(agent):
    assert agent in AGENTS
    assert cacheable_install(agent) is None
    assert agent_layer_plan(agent) is None


def test_a_replaced_install_command_is_not_cached(monkeypatch):
    """A manifest or plugin that swaps the command loses the cache."""
    monkeypatch.setitem(AGENT_INSTALLERS, "codex-acp", "npm install -g other-agent")
    assert cacheable_install("codex-acp") is None
    assert agent_layer_plan("codex-acp") is None


def test_an_acpx_wrapped_agent_is_not_cached():
    wrapped = registry.resolve_agent("acpx/claude-agent-acp")
    assert cacheable_install(wrapped.name) is None


@pytest.fixture
def restore_registry():
    """Put back every table register_agent writes."""
    tables = (
        registry.AGENTS,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
        registry.CACHEABLE_INSTALLS,
        registry._CACHEABLE_INSTALL_CMDS,
    )
    saved = [dict(table) for table in tables]
    yield
    for table, copy in zip(tables, saved, strict=True):
        table.clear()
        table.update(copy)


def test_a_plugin_with_a_pinned_install_is_cached(restore_registry):
    """register_agent(cacheable_version=...) is how a plugin such as
    mini-swe-acp, whose install pins every package, gets the agent layer."""
    install = "pip install -r /opt/runtime/plugin-requirements.txt"
    registry.register_agent(
        "plugin-pinned", install, "plugin", cacheable_version="2.4.6+lock-1"
    )
    assert cacheable_install("plugin-pinned") == ("2.4.6+lock-1", install)
    plan = agent_layer_plan("plugin-pinned")
    assert plan is not None
    assert plan.install == install
    assert plan.recipe.version == "2.4.6+lock-1"
    assert plan.stamp_path == f"{LAYER_STAMP_DIR}/agent-plugin-pinned.recipe"


def test_a_plugin_without_a_version_installs_in_every_trial(restore_registry):
    registry.register_agent("plugin-unpinned", "pip install plugin", "plugin")
    assert cacheable_install("plugin-unpinned") is None
    assert agent_layer_plan("plugin-unpinned") is None


def test_a_pinned_plugin_loses_the_cache_when_its_command_changes(restore_registry):
    registry.register_agent(
        "plugin-pinned", "install v1", "plugin", cacheable_version="1"
    )
    # A manifest that swaps the command afterwards.
    AGENT_INSTALLERS["plugin-pinned"] = "install v2"
    assert cacheable_install("plugin-pinned") is None
    # Registering again without a version drops the old pin.
    registry.register_agent("plugin-pinned", "install v1", "plugin")
    assert cacheable_install("plugin-pinned") is None
    assert "plugin-pinned" not in CACHEABLE_INSTALLS


def test_a_builtin_replaced_by_a_plugin_keeps_its_own_pin_only(restore_registry):
    original = AGENTS["codex-acp"].install_cmd
    registry.register_agent("codex-acp", "npm install -g other-agent", "codex-acp")
    assert cacheable_install("codex-acp") is None
    # The built-in command again: its built-in pin applies again.
    registry.register_agent("codex-acp", original, "codex-acp")
    assert cacheable_install("codex-acp") == (CACHEABLE_INSTALLS["codex-acp"], original)
    # A plugin that pins its replacement gets its own version.
    registry.register_agent(
        "codex-acp", "npm install -g other@1", "codex-acp", cacheable_version="1"
    )
    assert cacheable_install("codex-acp") == ("1", "npm install -g other@1")


_PER_TRIAL_MARKERS = (
    "/home/",
    "useradd",
    "credentials",
    "auth.json",
    "ANTHROPIC_",
    "OPENAI_API_KEY",
    "skills",
    "BENCHFLOW_AGENT_HOME",
)


@pytest.mark.parametrize("agent", sorted(PINNED_SPECS))
def test_a_layer_recipe_is_the_install_and_its_stamp_nothing_per_trial(agent):
    plan = agent_layer_plan(agent)
    assert plan is not None
    install = AGENTS[agent].install_cmd
    assert plan.install == install
    assert plan.recipe.install_cmd.startswith(f"( {install} ) && ")
    assert plan.stamp_path == f"{LAYER_STAMP_DIR}/agent-{agent}.recipe"
    tail = plan.recipe.install_cmd[len(f"( {install} )") :]
    assert plan.stamp in tail and plan.stamp_path in tail
    assert plan.recipe.version == CACHEABLE_INSTALLS[agent]
    # What the trial does per trial is never in the cached part.
    for marker in _PER_TRIAL_MARKERS:
        if marker not in install:
            assert marker not in plan.recipe.install_cmd, (agent, marker)
    for field in ("credential_files", "skill_paths", "home_dirs"):
        for item in getattr(AGENTS[agent], field):
            assert str(item) not in plan.recipe.install_cmd


def test_the_native_harness_cli_joins_the_layer():
    from benchflow.native_harness.harnesses import NATIVE_HARNESSES

    acp = agent_layer_plan("codex-acp")
    native = agent_layer_plan("codex-acp", harness="native")
    assert acp is not None and native is not None
    cli = NATIVE_HARNESSES["codex-acp"].install_cmd
    assert cli and cli not in acp.install and cli in native.install
    assert native.stamp != acp.stamp
    assert native.recipe.version != acp.recipe.version
    # Claude Code's native CLI comes with the agent's own install.
    assert (
        agent_layer_plan("claude-agent-acp", harness="native").install
        == AGENTS["claude-agent-acp"].install_cmd
    )


def test_the_install_timeout_bounds_the_layer_build():
    plan = agent_layer_plan("claude-agent-acp")
    assert plan is not None
    assert plan.recipe.timeout_sec == AGENTS["claude-agent-acp"].install_timeout


@pytest.mark.parametrize(
    ("value", "flag", "expected"),
    [
        ("", None, False),
        ("0", None, False),
        ("1", None, True),
        ("true", None, True),
        ("", True, True),
    ],
)
def test_fresh_install_is_the_flag_or_the_environment(
    monkeypatch, value, flag, expected
):
    monkeypatch.setenv(FRESH_INSTALL_ENV, value)
    assert fresh_install_requested(flag) is expected


# --- a sandbox that lists every step -----------------------------------------


@dataclass
class _Result:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class StepSandbox:
    """A sandbox that starts from an agent layer and lists every step sent to it.

    ``holds`` is the stamp its image has (None: the image holds no layer's
    install). Every exec, upload and download is recorded in order.
    """

    def __init__(self, *, layer: LayerResult | None = None, holds: str | None = None):
        self.steps: list[tuple[str, str]] = []
        self.recipes: list[Any] = []
        self.layer_records: dict[str, dict[str, Any]] = {}
        self._layer = layer
        self._holds = holds
        self.default_user = None
        self.supports_system_shell = True

    def use_agent_layer(self, recipe: Any) -> None:
        self.recipes.append(recipe)

    @property
    def agent_layer(self) -> LayerResult | None:
        return self._layer

    @receipt_aware
    async def exec(self, command: str, **kwargs: Any) -> _Result:
        self.steps.append(("exec", command))
        if command.startswith(f"cat {LAYER_STAMP_DIR}/"):
            if self._holds is None:
                return _Result(stderr="No such file or directory", return_code=1)
            return _Result(stdout=self._holds + "\n")
        if command == "pwd":
            return _Result(stdout="/app\n")
        if "pwd" in command:
            return _Result(stdout="/home/agent\n")
        return _Result()

    async def upload_file(self, source: Any, target: str, **kwargs: Any) -> None:
        self.steps.append(("upload_file", str(target)))

    async def upload_dir(self, source: Any, target: str, **kwargs: Any) -> None:
        self.steps.append(("upload_dir", str(target)))

    async def download_file(self, source: str, target: Any, **kwargs: Any) -> None:
        self.steps.append(("download_file", source))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("")


def _layer_result(plan: AgentLayerPlan, status: str = "hit") -> LayerResult:
    base = BaseImage(
        ref="task:image",
        image_id="sha256:" + "1" * 64,
        digest="sha256:" + "2" * 64,
        arch="amd64",
        libc="glibc",
    )
    return LayerResult(
        kind=AGENT,
        key="3" * 64,
        tag="benchflow-layer:agent-" + "3" * 64,
        image="bf__layer__trial",
        status=status,
        base=base,
        seconds=0.05,
    )


def _trial(tmp_path: Path, name: str, agent: str, env: Any, **config: Any) -> Rollout:
    skills = tmp_path / "skills" / "demo"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "SKILL.md").write_text("---\nname: demo\n---\n")
    if agent == "codex-acp":
        # The Apps policy's own probes need a real sandbox; it is per trial
        # either way and has its own tests.
        config.setdefault("codex_apps_policy", "inherit")
    rollout_config = RolloutConfig.from_legacy(
        task_path=tmp_path / "task",
        agent=agent,
        prompts=[None],
        sandbox_user="agent",
        sandbox_locked_paths=["/app/solution"],
        **config,
    )
    trial = Rollout(rollout_config)
    trial._env = env
    trial._rollout_dir = tmp_path / name
    trial._rollout_dir.mkdir()
    (trial._rollout_dir / "config.json").write_text(json.dumps({"agent": agent}))
    trial._rollout_paths = MagicMock()
    trial._task = MagicMock()
    # This fixture has no tests requesting a verifier runner (FIX-49 r4).
    trial._task.task_dir = tmp_path / "task"
    (trial._task.task_dir / "tests").mkdir(parents=True, exist_ok=True)
    trial._task.config.environment = None
    trial._task.config.sandbox = SandboxConfig()
    trial._task.config.verifier = MagicMock()
    trial._effective_task_path = tmp_path / "task"
    trial._effective_skills_dir = tmp_path / "skills"
    trial._effective_skills_sandbox_dir = "/skills"
    trial._effective_locked = ["/app/solution"]
    trial._agent_env = {"ANTHROPIC_API_KEY": "sk-test-not-a-key"}
    if agent in AGENTS:
        trial._agent_launch = AGENTS[agent].launch_cmd
    return trial


async def _run_trial(tmp_path: Path, name: str, agent: str, sandbox: StepSandbox):
    trial = _trial(tmp_path, name, agent, sandbox)
    trial._request_agent_layer()
    trial._record_layers()
    await trial.install_agent()
    return trial


# A merged exec's marker tag is random per exec (sandbox/_exec_merge.py,
# PY8 U12), so two trials' identical steps differ only in it.
_MERGED_TAG = re.compile(r"@@benchflow-merged-[0-9a-f]+")


def _without_install(steps: list[tuple[str, str]], install: str, stamp_path: str):
    """The steps a trial ran besides the install (or its layer's stamp read),
    with each merged exec's random tag replaced by one fixed name."""
    out = []
    for kind, what in steps:
        if kind == "exec" and (what == install or what == f"cat {stamp_path}"):
            continue
        out.append((kind, _MERGED_TAG.sub("@@benchflow-merged-TAG", what)))
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "codex-acp"])
async def test_per_trial_steps_run_every_trial_and_the_install_comes_from_the_layer(
    tmp_path, monkeypatch, agent
):
    """One trial installs in the sandbox, three start from the layer.

    Every trial runs the same per-trial steps (user, credentials, skills,
    lockdown) in the same order; only the trials on the layer skip the
    install, which they replace by one read of the layer's stamp.
    """
    from benchflow.sandbox import lockdown

    # V7's pre-agent inspector has a fresh name in each trial. Hold only the
    # name fixed while comparing the steps, including the protected capture.
    monkeypatch.setattr(
        lockdown, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="1" * 32))
    )
    # Runtime skill staging also has a fresh protected name in each trial.
    # Normalize that name while retaining the full per-trial step comparison.
    monkeypatch.setattr(
        "benchflow.agents.install.uuid4", lambda: SimpleNamespace(hex="2" * 32)
    )
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    plan = agent_layer_plan(agent)
    assert plan is not None
    install = AGENT_INSTALLERS[agent]

    cold = StepSandbox()  # no layer: as before, and as on a backend without one
    cold_trial = _trial(tmp_path, "cold", agent, cold)
    cold_trial._agent_layer_plan = None
    await cold_trial.install_agent()

    warm = []
    for index in range(3):
        sandbox = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
        trial = _trial(tmp_path, f"warm-{index}", agent, sandbox)
        # What start() leaves when the sandbox started from the layer
        # (test_a_pinned_agent_asks_for_its_layer covers the asking).
        trial._agent_layer_plan = plan
        await trial.install_agent()
        warm.append((trial, sandbox))

    assert ("exec", install) in cold.steps
    cold_rest = _without_install(cold.steps, install, plan.stamp_path)
    kinds = " ".join(what for _, what in cold_rest)
    for step in ("useradd", "/app/solution", "/skills"):
        assert step in kinds, step

    for trial, sandbox in warm:
        assert ("exec", install) not in sandbox.steps
        assert ("exec", f"cat {plan.stamp_path}") in sandbox.steps
        assert _without_install(sandbox.steps, install, plan.stamp_path) == cold_rest
        assert trial._agent_cfg is AGENTS[agent]
        log = (trial._rollout_dir / "agent" / "install-stdout.txt").read_text()
        assert log.startswith(f"$ {install}\n")
        assert "cached agent layer" in log
        layers = json.loads((trial._rollout_dir / "config.json").read_text())["layers"]
        assert layers[AGENT]["install"] == "from layer"


@pytest.mark.asyncio
async def test_a_sandbox_without_the_layers_install_installs_in_the_trial(
    tmp_path, monkeypatch
):
    """A stamp that is missing or names another install: the trial installs."""
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    agent = "claude-agent-acp"
    plan = agent_layer_plan(agent)
    assert plan is not None
    for name, holds in (("none", None), ("other", "sha256:" + "0" * 64)):
        sandbox = StepSandbox(layer=_layer_result(plan), holds=holds)
        trial = await _run_trial(tmp_path, name, agent, sandbox)
        assert ("exec", AGENT_INSTALLERS[agent]) in sandbox.steps
        layers = json.loads((trial._rollout_dir / "config.json").read_text())["layers"]
        assert layers[AGENT]["install"] == "in trial"
        assert "holds no claude-agent-acp install" in layers[AGENT]["reason"]


@pytest.mark.asyncio
async def test_an_install_failure_after_a_bad_layer_still_raises(tmp_path, monkeypatch):
    from benchflow.models import AgentInstallError

    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    plan = agent_layer_plan("codex-acp")
    assert plan is not None

    class Failing(StepSandbox):
        @receipt_aware
        async def exec(self, command: str, **kwargs: Any) -> _Result:
            result = await super().exec(command, **kwargs)
            if command == AGENT_INSTALLERS["codex-acp"]:
                return _Result(stderr="npm ERR!", return_code=1)
            return result

    sandbox = Failing(layer=_layer_result(plan), holds=None)
    trial = _trial(tmp_path, "fail", "codex-acp", sandbox)
    trial._request_agent_layer()
    with pytest.raises(AgentInstallError):
        await trial.install_agent()


# --- what start() asks the sandbox for ---------------------------------------


def _ask(
    tmp_path: Path, name: str, agent: str, **config: Any
) -> tuple[Rollout, StepSandbox]:
    sandbox = StepSandbox()
    trial = _trial(tmp_path, name, agent, sandbox, **config)
    trial._request_agent_layer()
    trial._record_layers()
    return trial, sandbox


def _layers(trial: Rollout) -> dict[str, Any] | None:
    return json.loads((trial._rollout_dir / "config.json").read_text()).get("layers")


def test_a_pinned_agent_asks_for_its_layer(tmp_path, monkeypatch):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    trial, sandbox = _ask(tmp_path, "a", "claude-agent-acp")
    plan = agent_layer_plan("claude-agent-acp")
    assert plan is not None
    assert [r.install_cmd for r in sandbox.recipes] == [plan.recipe.install_cmd]
    assert trial._agent_layer_plan == plan


@pytest.mark.parametrize("how", ["env", "flag"])
def test_a_fresh_install_asks_for_no_layer_and_says_so(tmp_path, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv(FRESH_INSTALL_ENV, "1")
    else:
        monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    sandbox = StepSandbox()
    # The flag case builds the config with the declared field, the one the
    # run config sets from ``--fresh-install`` (an undeclared name fails here).
    config = {"fresh_install": True} if how == "flag" else {}
    trial = _trial(tmp_path, "f", "codex-acp", sandbox, **config)
    trial._request_agent_layer()
    trial._record_layers()
    assert sandbox.recipes == [None]
    assert trial._agent_layer_plan is None
    assert _layers(trial) == {
        AGENT: {"status": "fresh", "reason": "fresh install asked for"}
    }


@pytest.mark.asyncio
async def test_a_fresh_install_installs_in_the_trial(tmp_path, monkeypatch):
    monkeypatch.setenv(FRESH_INSTALL_ENV, "1")
    plan = agent_layer_plan("codex-acp")
    assert plan is not None
    # Even a sandbox whose image holds the install runs it again.
    sandbox = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = await _run_trial(tmp_path, "f", "codex-acp", sandbox)
    assert ("exec", AGENT_INSTALLERS["codex-acp"]) in sandbox.steps
    assert ("exec", f"cat {plan.stamp_path}") not in sandbox.steps
    assert _layers(trial)[AGENT]["status"] == "fresh"


def test_an_unpinned_agent_records_why_it_installs_in_the_trial(tmp_path, monkeypatch):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    trial, sandbox = _ask(tmp_path, "p", "pi-acp")
    assert sandbox.recipes == [None]
    assert _layers(trial) == {
        AGENT: {"status": "fresh", "reason": "pi-acp has no pinned install to cache"}
    }


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_a_scripted_agent_asks_for_nothing(tmp_path, monkeypatch, agent):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    trial, sandbox = _ask(tmp_path, agent, agent)
    assert sandbox.recipes == [None]
    assert _layers(trial) is None


def test_skip_agent_install_asks_for_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    trial, sandbox = _ask(tmp_path, "s", "claude-agent-acp", skip_agent_install=True)
    assert sandbox.recipes == [None]
    assert _layers(trial) is None


@pytest.mark.parametrize("why", ["snapshot", "caller"])
def test_a_snapshot_or_caller_owned_sandbox_asks_for_nothing(
    tmp_path, monkeypatch, why
):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    sandbox = StepSandbox()
    trial = _trial(tmp_path, why, "claude-agent-acp", sandbox)
    if why == "snapshot":
        trial._from_branch_snapshot = True
    else:
        trial._env_externally_owned = True
    trial._request_agent_layer()
    assert sandbox.recipes == [None]
    assert trial._agent_layer_plan is None


def test_a_sandbox_without_layers_is_left_alone(tmp_path, monkeypatch):
    """Backends that build no layers (Modal, Daytona's own path) get no call."""
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    sandbox = MagicMock()
    sandbox.exec = AsyncMock()
    trial = _trial(tmp_path, "m", "claude-agent-acp", sandbox)
    trial._request_agent_layer()
    trial._record_layers()
    assert trial._agent_layer_plan is None
    assert _layers(trial) is None
    assert not sandbox.use_agent_layer.called


def test_the_sandboxs_layer_record_lands_in_config_json(tmp_path, monkeypatch):
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    plan = agent_layer_plan("codex-acp")
    assert plan is not None
    sandbox = StepSandbox(layer=_layer_result(plan, "miss"), holds=plan.stamp)
    trial = _trial(tmp_path, "r", "codex-acp", sandbox)
    trial._request_agent_layer()
    sandbox.layer_records[AGENT] = _layer_result(plan, "miss").record()
    trial._record_layers()
    record = _layers(trial)[AGENT]
    assert record["status"] == "miss"
    assert record["key"] == "3" * 64
    # The rest of config.json is kept.
    assert json.loads((trial._rollout_dir / "config.json").read_text())["agent"] == (
        "codex-acp"
    )


@pytest.mark.asyncio
async def test_install_from_layer_reads_the_stamp_with_one_command(tmp_path):
    plan = agent_layer_plan("gemini")
    assert plan is not None
    sandbox = StepSandbox(holds=plan.stamp)
    assert await install_from_layer(sandbox, plan, tmp_path, layer_key="k") is None
    assert sandbox.steps == [("exec", f"cat {shlex.quote(plan.stamp_path)}")]


@pytest.mark.parametrize(
    "mutation", ["setup", "hook", "manifest", "upload", "healthcheck", "env"]
)
async def test_preinstall_mutation_never_trusts_an_intact_recipe_stamp(
    tmp_path, monkeypatch, mutation
):
    """Guards the pre-install mutation regression introduced by 6de3e5d3."""
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    plan = agent_layer_plan("gemini")
    sandbox = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = _trial(tmp_path, mutation, "gemini", sandbox)
    if mutation == "setup":
        trial._task.config.sandbox.setup_commands = [
            SetupCommandConfig(command="printf setup")
        ]
    elif mutation == "hook":
        trial._config.pre_agent_hooks = [AsyncMock()]
    elif mutation == "manifest":
        trial._config.environment_manifest = MagicMock()
    elif mutation == "upload":
        trial._config.uploads = {"fixture": "/opt/runtime/bin/gemini"}
    elif mutation == "healthcheck":
        trial._task.config.sandbox.healthcheck = HealthcheckConfig(command="true")
    else:
        trial._task.config.sandbox.env = {"npm_config_registry": "synthetic"}
    trial._request_agent_layer()
    assert sandbox.recipes == [None]
    # Also check the public split lifecycle if a mutator was added late.
    trial._agent_layer_plan = plan
    await trial.install_agent()
    assert ("exec", AGENT_INSTALLERS["gemini"]) in sandbox.steps
    assert ("exec", f"cat {plan.stamp_path}") not in sandbox.steps
    assert trial._layer_records[AGENT]["install"] == "in trial"


async def test_caller_install_plane_always_runs(tmp_path, monkeypatch):
    """Guards the caller install-plane regression introduced by 6de3e5d3."""
    from benchflow.rollout_planes import DefaultRolloutPlanes

    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    called = []

    class CustomPlanes(DefaultRolloutPlanes):
        async def install_agent(self, env, agent, rollout_dir, **kwargs):
            called.append(agent)
            return AGENTS[agent]

    plan = agent_layer_plan("gemini")
    sandbox = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = _trial(tmp_path, "custom", "gemini", sandbox, planes=CustomPlanes())
    trial._request_agent_layer()
    assert sandbox.recipes == [None]
    trial._agent_layer_plan = plan
    await trial.install_agent()
    assert called == ["gemini"]
    assert "plane" in trial._layer_records[AGENT]["reason"]


@pytest.mark.parametrize("harness", ["acp", "native"])
async def test_installer_override_after_selection_runs_current_recipe(
    tmp_path, monkeypatch, harness
):
    """Guards the installer-override regression introduced by 6de3e5d3."""
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    plan = agent_layer_plan("codex-acp", harness=harness)
    sandbox = StepSandbox(layer=_layer_result(plan), holds=plan.stamp)
    trial = _trial(tmp_path, harness, "codex-acp", sandbox, harness=harness)
    trial._request_agent_layer()
    command = "printf synthetic-replacement-installer"
    monkeypatch.setitem(AGENT_INSTALLERS, "codex-acp", command)
    await trial.install_agent()
    assert ("exec", command) in sandbox.steps
    assert ("exec", f"cat {plan.stamp_path}") not in sandbox.steps
    assert trial._layer_records[AGENT]["install"] == "in trial"
    assert "changed" in trial._layer_records[AGENT]["reason"]


@pytest.mark.parametrize("source", ["dict", "yaml"])
def test_fresh_install_config_round_trip_reaches_layer_selection(
    tmp_path, monkeypatch, source
):
    """Guards the fresh-install round trip; already fixed by 5c95d43f in 58982aeb."""
    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    trial, sandbox = _ask(tmp_path, source, "gemini", fresh_install=True)
    if source == "dict":
        restored = RolloutConfig.from_dict(trial._config.to_dict())
    else:
        path = tmp_path / "fresh.yaml"
        path.write_text(
            "task_path: /tmp/synthetic-task\nagent: gemini\nfresh_install: true\n"
        )
        restored = RolloutConfig.from_yaml(path)
    assert restored.fresh_install is True
    trial._config = restored
    sandbox.recipes.clear()
    trial._request_agent_layer()
    assert sandbox.recipes == [None]
    assert trial._agent_layer_plan is None


@pytest.mark.parametrize("mutator", ["setup", "healthcheck", "env"])
async def test_native_task_install_inputs_bypass_agent_layers(
    tmp_path, monkeypatch, mutator
):
    """Guards native task inputs: 6de3e5d3 must use the normalized sandbox fields."""
    from benchflow import rollout_planes

    monkeypatch.delenv(FRESH_INSTALL_ENV, raising=False)
    task = tmp_path / "native-task"
    shutil.copytree(
        Path(__file__).parent / "examples" / "hello-world-task",
        task,
        ignore=shutil.ignore_patterns("task.md"),
    )
    addition = {
        "setup": '\n[[environment.setup_commands]]\ncommand = "printf setup"\n',
        "healthcheck": '\n[environment.healthcheck]\ncommand = "true"\n',
        "env": '\n[environment.env]\nnpm_config_registry = "synthetic"\n',
    }[mutator]
    with (task / "task.toml").open("a") as handle:
        handle.write(addition)
    sandbox = StepSandbox()
    monkeypatch.setattr(
        rollout_planes, "_create_environment", lambda *_a, **_kw: sandbox
    )
    trial = await Rollout.create(
        RolloutConfig(
            task_path=task, agent="gemini", environment="docker", jobs_dir=tmp_path
        )
    )
    await trial.setup()
    trial._request_agent_layer()
    assert sandbox.recipes == [None]
    assert trial._agent_layer_plan is None
