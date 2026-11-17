"""EN-698: every start of a trial's sandbox goes through one function,
:func:`benchflow.sandbox.service_cover.start_sandbox`, which asks for the
start holds before the sandbox starts (R698 07:24Z: ``Runtime.execute`` with
a sandbox it starts, the equivalence battery and acceptance-live started one
without them).

The first tests read src: every ``.start(...)`` call whose result is used
(awaited, handed on, kept) is listed, and each one is the start inside
``start_sandbox`` or one named below as not a sandbox's start. A call made
as a bare statement discards its result, and every sandbox's ``start`` is a
coroutine function (checked here), so such a call cannot start a sandbox.
The rest drive each start path with a stand-in sandbox that records, at its
own start, which holds it was asked for. Real Docker trials are in
tests/test_en698_policy_first_instruction_docker.py."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SRC = Path(__file__).parents[1] / "src" / "benchflow"
HELPER = ("sandbox/service_cover.py", "start_sandbox", "sandbox")

#: Every other ``.start(...)`` in src whose result is used, by (file, the
#: function it is in, what it is called on), with what it starts.
NOT_A_SANDBOX_START = {
    # A Rollout's or TaskRuntime's start: Rollout.start, whose sandbox start
    # is _start_env_and_upload's (start_sandbox).
    ("agent_start.py", "_probe", "rollout"): "Rollout.start",
    ("branch_run.py", "run_branch_trial", "rollout"): "Rollout.start",
    (
        "continue_run/orchestrator.py",
        "_continue_run_with_sandbox_proxy",
        "rollout",
    ): "Rollout.start",
    ("eval_regrade.py", "run_verifier_on_frozen_trial", "child"): "Rollout.start",
    ("rollout/__init__.py", "Rollout._prestart_steps", "self"): "Rollout.start",
    ("rollout/__init__.py", "Rollout._setup_and_start", "self"): "Rollout.start",
    ("rollout/_verifier_recovery.py", "_recover_verifier", "child"): "Rollout.start",
    ("rollout/task_runtime.py", "TaskRuntime.start", "rollout"): "Rollout.start",
    ("rollout_branch.py", "_run_isolated_children.one", "sub"): "Rollout.start",
    ("rollout/task_runtime.py", "TaskRuntime.create", "runtime"): "TaskRuntime",
    ("rollout/task_runtime.py", "TaskRuntime.__aenter__", "self"): "TaskRuntime",
    # Environment.start, whose sandbox start is start_sandbox's.
    ("runtime.py", "Environment.__aenter__", "self"): "Environment.start",
    ("runtime.py", "Runtime.execute", "self.env"): "Environment.start",
    # Inside a sandbox's own start(), which start_sandbox called.
    ("sandbox/daytona.py", "DaytonaSandbox.start", "self._strategy"): "own start",
    ("sandbox/remote_docker.py", "RemoteDockerSandbox.start", "super()"): "own start",
    ("sandbox/docker.py", "DockerSandbox._direct_up", "plan"): "own start",
    # A process, transport or helper in a sandbox already started.
    ("acp/client.py", "ACPClient.connect", "self._transport"): "agent transport",
    (
        "acp/container_transport.py",
        "ContainerTransport.start",
        "self._cp",
    ): "agent process",
    ("native_harness/client.py", "NativeCLIClient.prompt", "process"): "agent process",
    ("sandbox/docker.py", "DockerSandbox.egress_container_start", "box"): "helper",
    ("sandbox/docker.py", "DockerSandbox.bfx_session", "serve"): "bfx serve",
    ("sandbox/docker.py", "DockerSandbox._start_exec_channel", "channel"): "exec",
    # Host-side proxies, watchdogs and budget guards.
    (
        "continue_run/orchestrator.py",
        "continue_run",
        "ReplayProxy(router, host=bind_host, advertise_host=advertise_host)",
    ): "replay proxy",
    (
        "continue_run/orchestrator.py",
        "_continue_run_with_sandbox_proxy",
        "SandboxReplayProxy",
    ): "replay proxy in a started sandbox",
    ("continue_run/replay_proxy.py", "ReplayProxy.__enter__", "self"): "proxy",
    ("acp/runtime.py", "_prompt_with_idle_watchdog", "IdleWatchdog"): "watchdog",
    ("budget.py", "BudgetGuard.admit", "self"): "budget guard",
    ("evaluation.py", "Evaluation._run_sequential_shared", "guard"): "budget guard",
    # A regular expression match's offset.
    ("effort_request.py", "split_model_effort", "match"): "re.Match",
    ("integration_health.py", "_marker_line", "match"): "re.Match",
    ("sandbox/lockdown.py", "_verifier_named_dependency_files", "match"): "re.Match",
    ("sandbox/setup.py", "_modal_rewrite_heredoc_line", "match"): "re.Match",
    ("sandbox/setup.py", "_modal_rewrite_heredoc_line", "cat_match"): "re.Match",
    (
        "task/_document_parse.py",
        "_extract_prompt_sections",
        "matches[index + 1]",
    ): "re.Match",
    ("task/_document_parse.py", "_extract_prompt_sections", "matches[0]"): "re.Match",
    ("task/equivalence_variants.py", "_text_variants", "m"): "re.Match",
    (
        "task/verifier_document.py",
        "_extract_role_prompts",
        "matches[index + 1]",
    ): "re.Match",
}


def _calls(attr: str) -> list[tuple[tuple[str, str, str], ast.Call, ast.AST]]:
    """Every ``<x>.<attr>(...)`` call in src, keyed by (file, the function
    it is in, ``<x>``), with its parent node."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attr
            ):
                continue
            names, up = [], node
            while up in parent:
                up = parent[up]
                if isinstance(
                    up, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    names.append(up.name)
            key = (
                path.relative_to(SRC).as_posix(),
                ".".join(reversed(names)) or "<module>",
                ast.unparse(node.func.value),
            )
            found.append((key, node, parent.get(node)))
    return found


def test_every_sandbox_start_in_src_goes_through_start_sandbox():
    """A start anywhere else fails here until it goes through start_sandbox
    (or, if it starts no sandbox, is named in NOT_A_SANDBOX_START)."""
    used = {key for key, _node, up in _calls("start") if not isinstance(up, ast.Expr)}
    unknown = sorted(used - {HELPER} - set(NOT_A_SANDBOX_START))
    assert unknown == [], f"a start outside start_sandbox: {unknown}"
    assert HELPER in used, "start_sandbox no longer starts the sandbox"
    stale = sorted(set(NOT_A_SANDBOX_START) - used)
    assert stale == [], f"named starts no longer in src: {stale}"


def test_the_start_holds_are_asked_only_by_start_sandbox():
    asked = {key[:2] for key, _node, _up in _calls("ask_start_holds")}
    direct = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "ask_start_holds"
                ):
                    direct.add((path.relative_to(SRC).as_posix(), fn.name))
    assert asked == set()
    assert direct == {("sandbox/service_cover.py", "start_sandbox")}


def test_every_start_path_calls_start_sandbox():
    """The functions that start a sandbox: each one's start is start_sandbox."""
    callers = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [None, *[n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]]:
            body = tree.body if cls is None else cls.body
            for fn in body:
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "start_sandbox"
                    for n in ast.walk(fn)
                ):
                    name = fn.name if cls is None else f"{cls.name}.{fn.name}"
                    callers.add((path.relative_to(SRC).as_posix(), name))
    assert callers == {
        ("rollout/_setup.py", "_start_env_and_upload"),
        ("rollout/_separate_verifier.py", "VerifierSandboxRun._start"),
        ("runtime.py", "Environment.start"),
    }


def test_every_sandbox_start_is_a_coroutine_function():
    """So a ``.start()`` made as a bare statement starts no sandbox."""
    from benchflow.runtime import Environment
    from benchflow.sandbox import _base, protocol

    seen = []
    for path in sorted((SRC / "sandbox").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "start":
                    seen.append((path.name, cls.name))
    assert seen == []
    import inspect

    for owner in (protocol.Sandbox, _base.BaseSandbox, Environment):
        assert inspect.iscoroutinefunction(owner.start), owner


# --- each start path, with a stand-in sandbox ---------------------------------


class _Started(Exception):
    """The stand-in's start: the holds it was asked for are recorded."""


class _Box:
    def __init__(self) -> None:
        self.holds: dict | None = None
        self.task_env_config = None

    async def start(self, force_build: bool = False) -> None:
        self.holds = {
            "services": getattr(self, "_hold_every_service", False),
            "main": getattr(self, "_start_hold_mode", ""),
            "cover": (getattr(self, "_start_cover", None) or {}).get("mode"),
        }
        raise _Started


def _task_dir(root: Path, name: str = "task") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    (task / "instruction.md").write_text("Probe.\n")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[agent]\ntimeout_sec = 60\n\n'
        "[verifier]\ntimeout_sec = 60\n\n"
        '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.com"]\n'
    )
    return task


HELD = {"services": True, "main": "denylist", "cover": "denylist"}
SERVICES_ONLY = {"services": True, "main": "", "cover": "denylist"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "verifier_only", "expected"),
    [
        ("oracle", False, HELD),
        ("nop", False, HELD),
        ("script:/op.sh", False, HELD),
        ("claude-agent-acp", False, SERVICES_ONLY),
        (None, False, SERVICES_ONLY),
        # A verifier-only rollout (regrade, verifier recovery) gets a normal
        # oracle trial's holds (EN-698 L4); before, main was left open.
        ("oracle", True, HELD),
        ("claude-agent-acp", True, HELD),
    ],
)
async def test_start_sandbox_asks_for_the_holds_before_the_start(
    tmp_path, agent, verifier_only, expected
):
    from benchflow.sandbox.service_cover import start_sandbox
    from benchflow.task import Task

    box = _Box()
    with pytest.raises(_Started):
        await start_sandbox(
            box, Task(_task_dir(tmp_path)), agent=agent, verifier_only=verifier_only
        )
    assert box.holds == expected


@pytest.mark.asyncio
async def test_a_separate_verifier_sandbox_is_held_under_the_verifiers_network(
    tmp_path,
):
    from benchflow.sandbox.service_cover import start_sandbox
    from benchflow.task import Task

    task = _task_dir(tmp_path)
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[verifier]\n", '[verifier]\nnetwork_mode = "no-network"\n'
        )
    )
    box = _Box()
    with pytest.raises(_Started):
        await start_sandbox(box, Task(task), agent=None, separate_verifier=True)
    assert box.holds == {
        "services": True,
        "main": "no-network",
        "cover": "no-network",
    }


@pytest.mark.asyncio
async def test_runtime_execute_starts_its_sandbox_under_the_holds(
    tmp_path, monkeypatch
):
    """R698's SDK path: Environment.from_task's sandbox, which
    Runtime.execute starts itself, for the oracle (before: no holds, so
    main's ENTRYPOINT and the sidecar ran before any rule)."""
    from benchflow.runtime import Agent, Environment, Runtime

    box = _Box()
    env = Environment(box, _task_dir(tmp_path), "docker")
    monkeypatch.setattr(Runtime, "_ready_for_cloud_metadata_block", lambda self: None)
    monkeypatch.setattr(
        "benchflow.rollout.Rollout.create", AsyncMock(return_value=MagicMock())
    )
    with pytest.raises(_Started):
        await Runtime(env, Agent("oracle", model="")).execute()
    assert box.holds == HELD


@pytest.mark.asyncio
async def test_an_environment_started_by_its_caller_is_held(tmp_path):
    """``async with Environment(...)``: its services always; main too once
    prepare_kept_services names an oracle."""
    from benchflow.runtime import Environment

    box = _Box()
    with pytest.raises(_Started):
        await Environment(box, _task_dir(tmp_path, "a"), "docker").start()
    assert box.holds == SERVICES_ONLY
    box = _Box()
    env = Environment(box, _task_dir(tmp_path, "b"), "docker")
    env.prepare_kept_services("oracle")
    with pytest.raises(_Started):
        await env.start()
    assert box.holds == HELD


@pytest.mark.asyncio
async def test_a_rollout_starts_its_sandbox_under_the_holds(tmp_path):
    from benchflow.rollout import Rollout
    from benchflow.task import Task

    task = _task_dir(tmp_path)
    rollout = Rollout.__new__(Rollout)
    rollout._env = _Box()
    rollout._task = Task(task)
    rollout._config = SimpleNamespace(
        task_path=task, uploads=None, primary_agent="oracle", sandbox_user="agent"
    )
    rollout._timing = {}
    rollout._env_externally_owned = False
    rollout._from_branch_snapshot = False
    rollout._skills_agent_derived = False
    rollout._skills_baked = False
    for name in ("_configure_daytona_layers", "_request_layers", "_reserve_host_proxy"):
        setattr(rollout, name, lambda: None)
    with pytest.raises(_Started):
        await Rollout.start(rollout)
    assert rollout._env.holds == HELD


@pytest.mark.asyncio
async def test_the_equivalence_battery_starts_its_sandbox_under_the_holds(
    tmp_path, monkeypatch
):
    from benchflow.task import equivalence

    box = _Box()
    planes = MagicMock()
    planes.create_environment = MagicMock(return_value=box)
    box.stop = AsyncMock()
    monkeypatch.setattr("benchflow.contracts.default_rollout_planes", lambda: planes)
    await equivalence.check_equivalence_async(
        _task_dir(tmp_path), sandbox_type="docker"
    )
    assert box.holds == HELD


@pytest.mark.asyncio
async def test_acceptance_live_starts_its_sandbox_under_the_holds(
    tmp_path, monkeypatch
):
    from benchflow.task import acceptance_live

    box = _Box()
    box.stop = AsyncMock()
    planes = MagicMock()
    planes.create_environment = MagicMock(return_value=box)
    monkeypatch.setattr(acceptance_live, "default_rollout_planes", lambda: planes)
    case = SimpleNamespace(case_type="command", name="c", command="true")
    result = await acceptance_live._run_single_case(
        _task_dir(tmp_path),
        sandbox_type="docker",
        workspace=SimpleNamespace(target="/app"),
        staged_worktree=None,
        case=case,
        run_index=0,
    )
    assert result.reward is None
    assert box.holds == HELD


@pytest.mark.asyncio
async def test_a_separate_verifier_start_is_held_whatever_made_its_sandbox(
    tmp_path, monkeypatch
):
    """VerifierSandboxRun._start asks for the holds at the start, so a
    custom ``create`` (not only the default one) gets them."""
    from benchflow.rollout import _separate_verifier as sv
    from benchflow.sandbox.verifier_network import separate_verifier_network
    from benchflow.task import Task

    task = Task(_task_dir(tmp_path))
    box = _Box()
    rollout = SimpleNamespace(
        _rollout_dir=tmp_path / "trial",
        _timing={},
        _config=SimpleNamespace(
            environment="docker", context_root=None, base_image_override=None
        ),
        _task=task,
        _planes=MagicMock(),
        _agent_cwd="/app",
    )
    monkeypatch.setattr(sv, "_verifier_task", lambda task, plan: task)
    monkeypatch.setattr(sv, "_stage_context", lambda *a, **k: tmp_path)
    monkeypatch.setattr(sv, "_use_verifier_layer", lambda *a: None)
    run = sv.VerifierSandboxRun(rollout, create_environment=lambda *a: box)
    run.plan = MagicMock()
    with pytest.raises(sv.SeparateVerifierError, match="_Started"):
        await run._start()
    mode = separate_verifier_network(task.config)
    assert mode
    assert box.holds == {"services": True, "main": mode, "cover": mode}


# --- a sandbox its caller started before the run named its agent --------------


class _CallerStarted:
    """A Docker sandbox its caller started, with main's image ENTRYPOINT."""

    def __init__(self, held: str = "") -> None:
        self._bootstrap_configs = {"main": {"Entrypoint": ["/entry.sh"]}}
        if held:
            self._start_hold_mode = held

    async def metadata_rules_install(self, command):  # pragma: no cover
        raise AssertionError


@pytest.mark.parametrize(
    ("agent", "held", "refused"),
    [
        ("oracle", "", True),
        ("nop", "", True),
        ("oracle", "denylist", False),
        ("claude-agent-acp", "", False),
    ],
)
def test_a_caller_started_sandbox_refuses_an_oracle_its_start_did_not_hold(
    tmp_path, agent, held, refused
):
    from benchflow.sandbox.service_cover import caller_start_refusal
    from benchflow.task import Task

    reason = caller_start_refusal(
        _CallerStarted(held), Task(_task_dir(tmp_path)), agent=agent
    )
    if refused:
        assert reason is not None
        assert reason.startswith("sandbox startup refused: this sandbox was started")
        assert "(/entry.sh)" in reason and "(denylist)" in reason
    else:
        assert reason is None


def test_the_caller_start_refusal_is_not_retried():
    from benchflow._utils.scoring import is_unrecoverable_startup_error

    assert is_unrecoverable_startup_error(
        "sandbox startup refused: this sandbox was started before the run named "
        "its agent, so main's image ENTRYPOINT (/entry.sh) ran outside"
    )
