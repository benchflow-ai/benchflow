"""Guards the fix on py8/script-agent against the missing runner at fc2b9e9a."""

import base64
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.agents.registry import is_scripted_agent
from benchflow.eval_plan import (
    EvalCreateRequest,
    EvalPlanError,
    _normalize_eval_agent,
    build_eval_plan,
)
from benchflow.evaluation import EvaluationConfig, effective_model
from benchflow.rollout import Rollout, RolloutConfig
from tests._trust_admission import admitted
from tests.receipt_fakes import receipt_answer


@pytest.fixture
def inputs(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 1\n[environment]\nworkdir = "/app"\n'
    )
    (task / "instruction.md").write_text("Write hello.txt\n")
    script = tmp_path / "agents" / "write hello.sh"
    script.parent.mkdir()
    script.write_text("printf hello > hello.txt\nexit 7\n")
    return task, script, f"script:{script}"


def test_script_paths_parse_and_need_no_model_or_credentials(inputs):
    task, _, spec = inputs
    assert _normalize_eval_agent(spec) == spec
    assert is_scripted_agent(spec)
    assert effective_model(spec, "fake/model") is None
    assert EvaluationConfig(agent=spec).model is None
    cfg = RolloutConfig.from_legacy(task_path=task, agent=spec, model="fake/model")
    assert cfg.primary_model is None
    assert cfg.primary_agent == spec


@pytest.mark.parametrize("user", [None, "", " ", "root", "0", "none"])
def test_script_refuses_root_before_sandbox(inputs, user):
    task, _, spec = inputs
    with pytest.raises(ValueError, match="non-root"):
        RolloutConfig(task_path=task, agent=spec, sandbox_user=user)


def test_script_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(EvalPlanError, match="regular file"):
        _normalize_eval_agent(f"script:{tmp_path}/missing.sh")


def test_script_size_is_capped_before_any_sandbox(tmp_path):
    script = tmp_path / "too-large.sh"
    with script.open("wb") as source:
        source.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(EvalPlanError, match="16 MiB"):
        _normalize_eval_agent(f"script:{script}")


@pytest.mark.parametrize("legacy", [False, True])
def test_script_refuses_document_scenes(inputs, legacy):
    task, _, spec = inputs
    (task / "task.md").write_text(
        '---\nversion: "1.0"\nagents:\n  roles:\n    solver:\n      agent: codex\nscenes:\n  - name: solve\n    roles: [solver]\n---\n## prompt\n\nWrite hello.txt.\n'
    )
    constructor = RolloutConfig.from_legacy if legacy else RolloutConfig
    with pytest.raises(ValueError, match="document scenes"):
        constructor(task_path=task, agent=spec)


@pytest.mark.parametrize("options", [{"harness": "acp"}, {"agent_env": {"X": "1"}}])
def test_script_cli_refuses_harness_and_agent_options(inputs, options):
    task, _, spec = inputs
    with pytest.raises(EvalPlanError, match=r"drop.*choose a model agent"):
        build_eval_plan(EvalCreateRequest(tasks_dir=task.parent, agent=spec, **options))


def test_script_refuses_role_and_user_loop(inputs):
    from benchflow._types import Scene
    from benchflow.loop_strategies import parse_loop_strategy_spec

    task, _, spec = inputs
    with pytest.raises(ValueError, match="role agent"):
        RolloutConfig(task_path=task, scenes=[Scene.single(agent=spec)])
    with pytest.raises(ValueError, match=r"user.*loop"):
        RolloutConfig(
            task_path=task,
            agent=spec,
            loop_strategy=parse_loop_strategy_spec("verify-retry:k=2"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_script_dispatch_records_user_output_exit_and_timeout(
    inputs, tmp_path, monkeypatch, timeout
):

    task, _, spec = inputs
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    trial = tmp_path / "trial"
    trial.mkdir()
    rollout._rollout_dir = trial
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._timeout = 1
    rollout._agent_cwd = "/app"
    rollout._egress_denylist = None
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()
    rollout.cleanup = AsyncMock()
    rollout._run_steps = AsyncMock(side_effect=AssertionError("must not launch ACP"))
    calls = []

    async def execute(command, **kwargs):
        calls.append((command, kwargs))
        # Account reads cross as a receipt (its NUL separator).
        return await receipt_answer(command, lambda read: answer(read, kwargs))

    async def answer(command, kwargs):
        if command.startswith("cat /etc/passwd"):
            return SimpleNamespace(return_code=0, stdout=ACCOUNT_FILES, stderr="")
        if "su -s /bin/bash" in command:
            if timeout:
                raise RuntimeError("Command timed out after 16 seconds")
            return SimpleNamespace(return_code=7, stdout="", stderr="")
        if is_marker_read(command):
            return SimpleNamespace(
                return_code=0, stdout=marker_reply(command, "1:7"), stderr=""
            )
        if command.startswith("wc -c"):
            data = b"agent\noutput\n" if "script.stdout" in command else b"err  \n"
            return SimpleNamespace(
                return_code=0,
                stdout=f"{len(data)}\n" + base64.b64encode(data).decode(),
                stderr="",
            )
        return SimpleNamespace(return_code=0, stdout="agent\noutput\n", stderr="")

    trusted_execute = AsyncMock(side_effect=execute)
    rollout._env = SimpleNamespace(
        exec=execute,
        upload_file=AsyncMock(),
        script_runner_exec=trusted_execute,
        script_runner_upload=AsyncMock(),
    )
    kill = AsyncMock()
    monkeypatch.setattr(rollout._planes, "quiesce_agent", kill)
    firewall = AsyncMock()
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", firewall, raising=False
    )

    async def verify():
        if timeout:
            kill.assert_awaited_once()
            transport, user = kill.await_args.args
            assert user == "agent"
            await transport.exec("true", user="root")
            trusted_execute.assert_awaited_with("true")
        rollout._rewards = {"reward": 0.0}

    rollout.verify = AsyncMock(side_effect=verify)
    result = await rollout.run()
    assert result.error is None
    assert result.rewards == {"reward": 0.0}
    assert result.agent == "script:write hello.sh"
    assert result.agent_name == "script:write hello.sh"
    assert result.model is None
    assert result.trajectory == [
        {
            "type": "script",
            "command": "script:write hello.sh",
            "return_code": 124 if timeout else 7,
            "stdout": "agent\noutput\nerr",
            "timed_out": timeout,
        }
    ]
    command, kwargs = next(c for c in calls if "su -s /bin/bash" in c[0])
    assert command.startswith("timeout -s TERM -k 1 1 /bin/bash ")
    assert "su -s /bin/bash agent" in command
    assert "env -i" in command
    assert kwargs["cwd"] == "/app"
    assert kwargs["timeout_sec"] == 16
    assert any("chown root:root" in c and "chmod 755" in c for c, _ in calls)
    firewall.assert_awaited_once_with(rollout._env, "agent", {})
    rollout.verify.assert_awaited_once()
    saved = json.loads((trial / "result.json").read_text())
    assert saved["scenes"][0]["roles"][0]["agent"] == "script:write hello.sh"


def test_script_activity_is_not_a_provider_failure(inputs):
    from benchflow.integration_health import diagnose

    task, _, spec = inputs
    assert (
        diagnose(
            [{"type": "script"}],
            agent=spec,
            n_tool_calls=0,
            output_tokens=0,
            logs={},
            agent_seconds=1,
            prompt_sent=True,
        )
        is None
    )
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    rollout._rewards = {"reward": 0.0}
    rollout._executed_prompts = ["test"]
    rollout._maybe_classify_api_error()
    assert rollout._error is None
    assert rollout._rewards == {"reward": 0.0}


@pytest.mark.parametrize("agent", ["EMPTY", " empty ", "EmPtY"])
def test_legacy_empty_control_normalization_survives_prefix_predicate(agent):
    """Guards b18ed01d's predicate change without changing fc2b9e9a's empty alias."""
    from benchflow.integration_health import diagnose

    assert (
        diagnose(
            [],
            agent=agent,
            n_tool_calls=0,
            output_tokens=0,
            logs={},
            agent_seconds=1,
            prompt_sent=True,
            truncated=True,
        )
        is None
    )


def test_script_integrity_audit_does_not_invent_agent_actions(tmp_path, monkeypatch):
    import benchflow.integrity.emit as module
    from benchflow.integrity.contract import ContractFacts, derive_contract
    from benchflow.integrity.emit import TrialEvidence, emit_integrity

    def no_agent_lane(*args, **kwargs):
        raise AssertionError("an operator script has no observed tool-call lane")

    monkeypatch.setattr(module, "action_records_from_trajectory", no_agent_lane)
    evidence = TrialEvidence(
        trial_dir=tmp_path,
        run_id="script-trial",
        mode="audit",
        contract=derive_contract(ContractFacts(task_id="t", workspace="/app")),
        agent="script:wrong.sh",
        trajectory=[{"type": "script", "return_code": 0}],
        rewards={"reward": 0.0},
    )
    emit_integrity(evidence)
    assert (tmp_path / "integrity" / "action_records.jsonl").read_text() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["offline", "denylist", "allowlist"])
async def test_script_setup_keeps_container_block_and_agent_policy(
    inputs, tmp_path, policy
):
    from benchflow.contracts import default_rollout_planes

    task, _, spec = inputs
    extra = {
        "offline": "allow_internet = false\n",
        "denylist": 'network_mode = "denylist"\nblocked_hosts = ["blocked.bench.invalid"]\n',
        "allowlist": 'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
    }[policy]
    with (task / "task.toml").open("a") as config:
        config.write(extra)
    planes = MagicMock(wraps=default_rollout_planes())
    planes.resolve_agent_env.return_value = {}
    planes.create_environment.return_value = SimpleNamespace()
    rollout = Rollout(
        RolloutConfig(
            task_path=task, agent=spec, jobs_dir=tmp_path / "jobs", planes=planes
        )
    )
    await rollout.setup()
    assert planes.create_environment.call_args.kwargs["preserve_agent_network"] is False
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in rollout._agent_env
    planes.agent_launch.assert_not_called()
    if policy == "offline":
        assert rollout._egress_denylist is None
    else:
        assert rollout._egress_denylist.mode == policy


@pytest.mark.asyncio
async def test_script_under_an_offline_agent_phase_gets_the_offline_sandbox(
    inputs, tmp_path
):
    """Fails on c0d0122e, where a script: agent under [agent] no-network kept the sandbox's whole network."""
    from benchflow.contracts import default_rollout_planes

    task, _, spec = inputs
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 1\nnetwork_mode = "no-network"\n'
        '[environment]\nworkdir = "/app"\n'
    )
    planes = MagicMock(wraps=default_rollout_planes())
    planes.resolve_agent_env.return_value = {}
    planes.create_environment.return_value = SimpleNamespace()
    rollout = Rollout(
        RolloutConfig(
            task_path=task, agent=spec, jobs_dir=tmp_path / "jobs", planes=planes
        )
    )
    await rollout.setup()
    # The sandbox is made as for an offline model agent: NET_ADMIN for the
    # uid firewall, the sandbox itself still online for setup and the verifier.
    assert planes.create_environment.call_args.kwargs["preserve_agent_network"] is True
    assert rollout._script_agent_offline is True
    assert rollout._egress_denylist is None
    assert "BENCHFLOW_DISALLOW_WEB_TOOLS" not in rollout._agent_env


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["script", "oracle", "nop"])
async def test_script_install_guards_daemon_keeps_lockdown_and_skips_credentials(
    inputs, tmp_path, monkeypatch, agent
):
    import benchflow.rollout as module
    from benchflow.contracts import default_rollout_planes
    from benchflow.task import Task

    task, _, spec = inputs
    planes = MagicMock(wraps=default_rollout_planes())
    for name in (
        "snapshot_build_config",
        "seed_verifier_workspace",
        "prepare_verifier_runtime",
        "deploy_skills",
        "lockdown_paths",
    ):
        setattr(planes, name, AsyncMock())
    planes.setup_sandbox_user = AsyncMock(return_value="/mapped-workspace")
    rollout = Rollout(
        RolloutConfig(
            task_path=task, agent=spec if agent == "script" else agent, planes=planes
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._task = Task(task)
    rollout._effective_task_path = task
    rollout._effective_locked = ["/solution", "/tests"]
    rollout._env = SimpleNamespace()
    monkeypatch.setattr(module, "_resolve_agent_cwd", AsyncMock(return_value="/app"))
    monkeypatch.setattr(module, "prepare_capture_runtime", AsyncMock())
    await rollout.install_agent()
    assert planes.setup_sandbox_user.call_args.kwargs["guard_daemon"] is (
        agent == "script"
    )
    assert rollout._agent_cwd == ("/mapped-workspace" if agent == "script" else "/app")
    planes.lockdown_paths.assert_awaited_once_with(
        rollout._env, ["/solution", "/tests"]
    )
    planes.install_agent.assert_not_called()
    planes.write_credential_files.assert_not_called()


@pytest.mark.asyncio
async def test_script_proxy_firewall_exec_stop_verify_order(
    inputs, tmp_path, monkeypatch
):
    import benchflow.rollout as module
    from benchflow.sandbox.egress_denylist import EgressDenylist

    task, _, spec = inputs
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace()
    rollout._egress_denylist = EgressDenylist((), ("blocked.bench.invalid",))
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    calls = []

    async def start(*args):
        calls.append("proxy")

    async def firewall(env, user, variables):
        assert user == "agent" and variables["HTTPS_PROXY"].startswith(
            "http://127.0.0.1:"
        )
        assert "SOLUTION_SECRET" not in variables
        calls.append("firewall")

    async def run(env, data, label, timeout, user, cwd, variables, **kwargs):
        assert data and cwd == "/app" and "HTTPS_PROXY" in variables
        calls.append("script")
        return [{"type": "script", "return_code": 0, "timed_out": False}], label

    async def stop():
        calls.append("stop")

    async def verify():
        calls.append("verify")
        rollout._rewards = {"reward": 1.0}

    rollout._agent_env = {"SOLUTION_SECRET": "not-forwarded"}
    rollout._start_egress_denylist = start
    rollout._stop_active_egress = stop
    rollout.verify = verify
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", firewall, raising=False
    )
    monkeypatch.setattr(module, "_run_script", run, raising=False)
    result = await rollout.run()
    assert result.error is None and result.rewards == {"reward": 1.0}
    assert calls == ["proxy", "firewall", "script", "stop", "verify"]


@pytest.mark.asyncio
async def test_an_offline_script_starts_behind_the_uid_firewall(
    inputs, tmp_path, monkeypatch
):
    """Fails on c0d0122e: the loopback-only uid firewall must be armed before an offline script runs."""
    import benchflow.rollout as module

    task, _, spec = inputs
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace()
    rollout._egress_denylist = None
    rollout._script_agent_offline = True
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    calls = []

    async def policy_firewall(env, user, variables):
        assert variables == {}
        calls.append("policy")

    async def uid_firewall(env, user):
        assert env is rollout._env
        calls.append(f"uid firewall for {user}")

    async def run(env, data, label, timeout, user, cwd, variables, **kwargs):
        calls.append("script")
        return [{"type": "script", "return_code": 0, "timed_out": False}], label

    async def verify():
        calls.append("verify")
        rollout._rewards = {"reward": 0.0}

    rollout.verify = verify
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", policy_firewall, raising=False
    )
    monkeypatch.setattr(
        rollout._planes, "enforce_offline_uid_egress", uid_firewall, raising=False
    )
    monkeypatch.setattr(module, "_run_script", run, raising=False)
    result = await rollout.run()
    assert result.error is None
    assert calls == ["policy", "uid firewall for agent", "script", "verify"]


@pytest.mark.asyncio
async def test_oracle_exec_is_unchanged(inputs):
    from benchflow.rollout._setup import _run_oracle

    task, _, _ = inputs
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("true\n")
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=""))
    )
    await _run_oracle(env, task, 1)
    assert env.exec.call_args_list[0].args == (
        "bash /solution/solve.sh > /logs/agent/oracle.txt 2>&1",
    )
    assert env.exec.call_args_list[0].kwargs == {
        "env": {"DEBIAN_FRONTEND": "noninteractive"},
        "timeout_sec": 1,
        "cwd": "/app",
    }


def is_marker_read(command):
    return command.startswith("printf") and "/exit-code 2>/dev/null" in command


def marker_reply(command, body, before=""):
    """What the marker read prints: *before* (stray output), then *body*
    on the line tagged with the read's own nonce."""
    tag = re.search(r"benchflow-script-marks-[0-9a-f]{32}", command).group()
    return f"{before}\n{tag}:{body}\n"


ACCOUNT_FILES = (
    "root:x:0:0::/root:/bin/bash\nagent:x:1000:1000::/tmp/agent:/bin/bash\n"
    "nobody:x:65534:65534::/tmp/nobody:/bin/sh\n"
    "grader:x:2000:2000::/tmp/grader:/bin/sh\n"
    "\0root:x:0:\nagent:x:1000:\ngrader:x:2000:\nnogroup:x:65534:\n"
)


class FakeShell:
    """A sandbox whose runner, start and exit-code markers and publish step
    are scripted."""

    def __init__(self, *, code=0, marker="", publish=None, started=True, stray=""):
        self.code, self.marker, self.publish = code, marker, publish
        self.started, self.stray = started, stray
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        # Account reads cross as a receipt (its NUL separator).
        return await receipt_answer(command, lambda read: self._answer(read, kwargs))

    async def _answer(self, command, kwargs):
        if command.startswith("cat /etc/passwd"):
            return SimpleNamespace(return_code=0, stdout=ACCOUNT_FILES, stderr="")
        if "su -s /bin/bash" in command:
            return SimpleNamespace(return_code=self.code, stdout="", stderr="")
        if is_marker_read(command):
            proof = "1" if self.started else ""
            return SimpleNamespace(
                return_code=0,
                stdout=marker_reply(
                    command, f"{proof}:{self.marker or ''}", self.stray
                ),
                stderr="",
            )
        if "/logs/agent/script.txt" in command and self.publish is not None:
            raise self.publish
        if command.startswith("wc -c"):
            return SimpleNamespace(return_code=0, stdout="0\n", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target, **_kwargs):
        return None

    async def script_runner_exec(self, command, **kwargs):
        return await self.exec(command, user="root", **kwargs)

    async def script_runner_upload(self, source, target, **kwargs):
        return await self.upload_file(source, target, **kwargs)

    def runner(self):
        return next(c for c in self.calls if "su -s /bin/bash" in c[0])


async def run_fake(shell, timeout=30):
    from benchflow.rollout._setup import _run_script

    planes = SimpleNamespace(quiesce_agent=AsyncMock())
    events, _ = await _run_script(
        shell, b"true\n", "script:x.sh", timeout, "agent", "/app", {}, planes=planes
    )
    return events[0], planes.quiesce_agent


@pytest.mark.asyncio
async def test_script_runner_refuses_an_untrusted_backend_transport():
    """Never clean env after a root shell starts."""
    from benchflow.sandbox._base import BaseSandbox

    shell = FakeShell()
    with pytest.raises(RuntimeError, match="trusted root runner transport"):
        await BaseSandbox.script_runner_exec(shell, "true")
    assert not shell.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote_docker"])
async def test_script_root_transport_clears_every_configured_variable(backend):
    """Guards the fix and the partial loader-hook list in 886d70bb."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    cls = DockerSandbox if backend == "docker" else RemoteDockerSandbox
    shell = object.__new__(cls)
    entries = [
        "PATH=/app/bin",
        "BASH_ENV=/app/startup",
        "ENV=/app/startup",
        "LD_PRELOAD=/app/hook.so",
        "LD_LIBRARY_PATH=/app/lib",
        "LD_AUDIT=/app/audit.so",
        "GLIBC_TUNABLES=value",
        "GCONV_PATH=/app/gconv",
        "LOCPATH=/app/locale",
        "BASH_FUNC_helper%%=() { :; }",
        "FUTURE_RUNTIME_HOOK=/app/next",
        "LD_FUTURE_FLAG=1",
        "EMPTY=",
        # S1's inert bootstrap leaves cleared names bare in Config.Env.
        "LD_SHOW_AUXV",
        "LD_DEBUG",
    ]
    shell.script_container_config = AsyncMock(return_value={"Env": entries})
    shell._run_docker_compose_command = AsyncMock()
    await shell.script_runner_exec("printf safe", cwd="/app", timeout_sec=10)
    argv = shell._run_docker_compose_command.call_args.args[0]
    flags = argv[: argv.index("main")]
    cleared = {flags[i + 1] for i, value in enumerate(flags) if value == "-e"}
    loader = {"LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_FUTURE_FLAG"}
    loader |= {"GLIBC_TUNABLES", "GCONV_PATH", "LOCPATH"}
    # SCRIPT-AGENT-DOCKER-F1: a loader name is removed (bare -e NAME), never
    # set empty, and a name the container does not have is never revived.
    assert loader <= cleared
    others = {e.partition("=")[0] for e in entries if "=" in e} - loader
    assert {name + "=" for name in others} <= cleared
    assert not {"LD_SHOW_AUXV", "LD_DEBUG", "LD_SHOW_AUXV=", "LD_DEBUG="} & cleared
    assert not [
        flag for flag in cleared if flag.endswith("=") and flag.startswith("LD_")
    ]
    assert argv[argv.index("main") + 1 :] == [
        "/usr/bin/env",
        "-i",
        "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/root",
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-p",
        "-c",
        "printf safe",
    ]
    assert shell._run_docker_compose_command.call_args.kwargs == {
        "check": False,
        "timeout_sec": 10,
        "merge_stderr": False,
        "raw_output": True,
        # Compose must not fill a removed name in from its own environment.
        "unset_env": sorted(name for name in cleared if "=" not in name),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("before", ["", "AT_SYSINFO_EHDR: 0x7ffd\nAT_EXECFN: /x\n"])
async def test_script_marker_read_survives_stray_output(before):
    """Guards SCRIPT-AGENT-DOCKER-F1: output ahead of the receipt line (a
    glibc aux-vector dump) neither hides the start receipt nor fakes one."""
    event, quiesce = await run_fake(FakeShell(code=3, marker="3", stray=before))
    assert event["return_code"] == 3 and event["timed_out"] is False
    quiesce.assert_not_awaited()


@pytest.mark.asyncio
async def test_script_marker_line_without_the_reads_nonce_is_no_receipt():
    """A line shaped like the receipt but without this read's nonce is not one."""

    class Forged(FakeShell):
        async def exec(self, command, **kwargs):
            if is_marker_read(command):
                self.calls.append((command, kwargs))
                forged = "benchflow-script-marks-" + "0" * 32
                return SimpleNamespace(
                    return_code=0, stdout=f"1:0\n{forged}:1:0\n", stderr=""
                )
            return await super().exec(command, **kwargs)

    with pytest.raises(RuntimeError, match="before the script started"):
        await run_fake(Forged(code=0, marker="0"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module,classname",
    [
        ("daytona", "DaytonaSandbox"),
        ("modal_impl", "ModalSandbox"),
        ("agentcore", "AgentCoreSandbox"),
        ("apple_container", "AppleContainerSandbox"),
    ],
)
async def test_script_backends_without_root_transport_fail_closed(module, classname):
    """Guards the fix at 90d0c606 across non-Docker backend siblings."""
    from importlib import import_module

    cls = getattr(import_module(f"benchflow.sandbox.{module}"), classname)
    shell = object.__new__(cls)
    shell.exec = AsyncMock()
    shell.upload_file = AsyncMock()
    with pytest.raises(RuntimeError, match="trusted root runner transport"):
        await shell.script_runner_exec("true")
    with pytest.raises(RuntimeError, match="trusted runner upload"):
        await shell.script_runner_upload("unused", "unused", admission=admitted(shell))
    shell.exec.assert_not_called()
    shell.upload_file.assert_not_called()


@pytest.mark.asyncio
async def test_script_publish_timeout_keeps_trial_scored():
    """Guards DT2-R1-A-1: an agent FIFO at script.txt made publishing raise."""
    shell = FakeShell(
        code=0, marker="0", publish=RuntimeError("Command timed out after 10 seconds")
    )
    event, _ = await run_fake(shell)
    assert event["return_code"] == 0 and event["timed_out"] is False
    publish, kwargs = next(c for c in shell.calls if "script.txt" in c[0])
    assert publish.startswith("timeout -s KILL 5 /bin/sh -c ")
    assert "rm -f /logs/agent/script.txt" in publish
    assert kwargs["user"] == "agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1, 125, 127])
async def test_script_runner_failure_is_an_error_not_a_score(code):
    """Guards DT2-R1-A-3: BusyBox rejected GNU options (exit 1, no marker) and
    the untouched workspace was scored as the script's attempt."""
    shell = FakeShell(code=code, marker=None)
    with pytest.raises(RuntimeError, match=f"runner exited {code}"):
        await run_fake(shell)
    command, kwargs = shell.runner()
    assert command.startswith("timeout -s TERM -k 1 30 /bin/bash ")
    assert "--signal" not in command and "--kill-after" not in command
    assert kwargs["timeout_sec"] == 45


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [124, 137, 143])
async def test_script_watchdog_codes_without_marker_are_timeouts(code):
    event, quiesce = await run_fake(FakeShell(code=code, marker=None))
    assert event["return_code"] == 124 and event["timed_out"] is True
    quiesce.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [124, 143])
async def test_script_own_watchdog_like_exit_is_kept(code):
    event, quiesce = await run_fake(FakeShell(code=code, marker=str(code)))
    assert event["return_code"] == code and event["timed_out"] is False
    quiesce.assert_not_awaited()


@pytest.mark.asyncio
async def test_script_runner_failure_makes_rollout_error_without_verify(
    inputs, tmp_path, monkeypatch
):
    """Guards DT2-R1-A-3 end to end: no score for a script that never ran."""
    task, _, spec = inputs
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._timeout = 30
    rollout._agent_cwd = "/app"
    rollout._egress_denylist = None
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    rollout._env = FakeShell(code=1, marker=None)
    monkeypatch.setattr(rollout._planes, "quiesce_agent", AsyncMock())
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", AsyncMock(), raising=False
    )
    rollout.verify = AsyncMock()
    result = await rollout.run()
    assert "runner exited 1" in (result.error or "")
    assert result.rewards is None
    rollout.verify.assert_not_awaited()


@pytest.mark.asyncio
async def test_script_without_agent_timeout_runs_unbounded(
    inputs, tmp_path, monkeypatch
):
    from benchflow.task.task import Task

    """Guards DT2-R1-B-1: no [agent] timeout_sec gave `timeout 1s` and a 0."""
    task, _, spec = inputs
    (task / "task.toml").write_text(
        'version = "1.0"\n[environment]\nworkdir = "/app"\n'
    )
    rollout = Rollout(RolloutConfig(task_path=task, agent=spec))
    assert rollout._config.timeout is None
    rollout._task = Task(task)
    assert rollout._task.config.agent.timeout_sec is None
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "script-trial"
    rollout._started_at = datetime.now()
    rollout._timeout = int(rollout._task.config.agent.timeout_sec or 0)
    rollout._agent_cwd = "/app"
    rollout._egress_denylist = None
    for name in ("setup", "start", "install_agent", "cleanup"):
        setattr(rollout, name, AsyncMock())
    shell = FakeShell(code=0, marker="0")
    rollout._env = shell
    monkeypatch.setattr(
        rollout._planes, "enforce_agent_egress_firewall", AsyncMock(), raising=False
    )

    async def verify():
        rollout._rewards = {"reward": 1.0}

    rollout.verify = AsyncMock(side_effect=verify)
    result = await rollout.run()
    assert result.error is None and result.rewards == {"reward": 1.0}
    assert result.trajectory[0]["timed_out"] is False
    command, kwargs = shell.runner()
    assert command.startswith("/bin/bash --noprofile --norc -c ")
    assert kwargs["timeout_sec"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1, 127])
async def test_script_su_failure_is_an_error_not_a_score(code):
    """Guards DT2-R2-A-1: a failed su wrote its own exit (127 without su, 1
    without SETUID/SETGID) to the marker, so the unattempted workspace was
    scored as the script's attempt."""
    shell = FakeShell(code=code, marker=str(code), started=False)
    with pytest.raises(RuntimeError, match="before the script started"):
        await run_fake(shell)
    command, _ = shell.runner()
    directory = re.search(r"/\.benchflow-script-[0-9a-f]{32}", command)[0]
    assert f"3> {directory}/started" in command
    assert f"BASH_ENV={directory}/started.bash /bin/bash" in command
    prepare = shell.calls[0][0]
    assert f"mkdir -m 755 {directory}" in prepare
    assert "exit-code" not in prepare


@pytest.mark.asyncio
async def test_script_receives_the_container_start_environment():
    """Guards DT2-R2-A-2 at the command level (the Docker tests run it):
    the script starts from PID 1's environment, not a PATH/locale whitelist,
    with the root shell's PATH and locale only when that is unreadable."""
    shell = FakeShell(code=0, marker="0")
    await run_fake(shell)
    ready = next(c for c, _ in shell.calls if "/proc/1/environ" in c)
    command, _ = shell.runner()
    directory = re.search(r"/\.benchflow-script-[0-9a-f]{32}", command)[0]
    assert f"cat /proc/1/environ > {directory}/image.env" in ready
    assert f"done < {directory}/image.env || exit 125" in command
    assert (
        f"exec /usr/bin/env -i PATH=/usr/bin:/bin BASH_ENV={directory}/started.bash "
        '/bin/bash --noprofile --norc /dev/fd/6 "${image[@]}" HOME=/home/agent'
    ) in command
    assert "LANG=$BENCHFLOW_SCRIPT_LANG" in command


@pytest.mark.asyncio
@pytest.mark.parametrize("container_id,config", [(None, {}), ("box", None)])
async def test_script_docker_missing_config_is_an_error(container_id, config):
    """Guards DT2-R3-A-1 at 0490a06c: never score a missing Docker env."""
    from benchflow.sandbox.docker import DockerSandbox

    class DockerShell(FakeShell, DockerSandbox):
        pass

    shell = DockerShell()
    shell._main_container_id = AsyncMock(return_value=container_id)
    shell._inspect_container = AsyncMock(return_value={"Config": config})
    with pytest.raises(RuntimeError, match="script: could not"):
        await run_fake(shell)
    assert not any("su -s /bin/bash" in command for command, _ in shell.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entries,expected",
    [
        (None, {}),
        (["JAVA_HOME=/opt/jdk"], {"JAVA_HOME": "/opt/jdk"}),
        (
            ["PATH=/opt/bin", "HOSTNAME=override", "EQ=a=b\nc=d"],
            {"PATH": "/opt/bin", "HOSTNAME": "override", "EQ": "a=b\nc=d"},
        ),
    ],
    ids=["null-env", "no-path", "overrides"],
)
async def test_script_docker_env_starts_with_daemon_defaults(entries, expected):
    """Guards DT2-RC1-A-2/B-1 at 5756d715: exec adds PATH before Config.Env."""
    from benchflow.sandbox.docker import DockerSandbox

    class DockerShell(FakeShell, DockerSandbox):
        async def upload_file(self, source, target, **_kwargs):
            if target.endswith("/image.env"):
                self.image_env = Path(source).read_bytes()

    shell = DockerShell(code=0, marker="0")
    shell.script_container_config = AsyncMock(
        return_value={"Env": entries, "Hostname": "task-host"}
    )
    await run_fake(shell)
    pairs = shell.image_env.rstrip(b"\0").decode().split("\0")
    actual = dict(pair.split("=", 1) for pair in pairs)
    assert len(pairs) == len(actual)
    assert actual == {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOSTNAME": "task-host",
        **expected,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("container_id", ["ab12cd34ef56", "a1" * 32])
async def test_docker_main_container_id_ignores_compose_warnings(container_id):
    """Guards DT2-RC1-A-1 at 5756d715: a warning was used as the container id."""
    from benchflow.sandbox.docker import DockerSandbox

    shell = DockerSandbox.__new__(DockerSandbox)
    shell._run_docker_compose_command = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0,
            stdout=f"warning: obsolete version\n{'a' * 13}\n {container_id}\n",
            stderr="warning: unset variable\n",
        )
    )
    shell._docker_cli = AsyncMock(
        return_value=SimpleNamespace(
            return_code=0,
            stdout=f" {container_id}\n",
            stderr="warning: unset variable\n",
        )
    )
    shell._compose_project = "warning-task"
    assert await shell._main_container_id() == container_id
    shell._run_docker_compose_command.assert_not_awaited()
    shell._docker_cli.assert_awaited_once_with(
        [
            "ps",
            "-q",
            "--no-trunc",
            "--filter",
            "label=com.docker.compose.project=warning-task",
            "--filter",
            "label=com.docker.compose.service=main",
        ],
        check=False,
        raw_output=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,code",
    [
        ("warning: obsolete version\n", 0),
        ("a" * 64, 1),
        ("warning: obsolete version\n" + "a" * 64, 0),
        ("a" * 64 + "\n" + "b" * 64, 0),
    ],
)
async def test_docker_main_container_id_rejects_missing_or_failed_lookup(output, code):
    """Guards DT2-RC1-A-1 at 5756d715: diagnostics are never container ids."""
    from benchflow.sandbox.docker import DockerSandbox

    shell = DockerSandbox.__new__(DockerSandbox)
    shell._run_docker_compose_command = AsyncMock(
        return_value=SimpleNamespace(return_code=code, stdout=output)
    )
    shell._docker_cli = AsyncMock(
        return_value=shell._run_docker_compose_command.return_value
    )
    shell._compose_project = "warning-task"
    assert await shell._main_container_id() is None


@pytest.mark.asyncio
async def test_script_dind_uses_inner_container_config_without_live_daytona():
    """Guards DT2-R3-A-1 at 0490a06c for the same Docker gap inside DinD."""
    from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD, _DaytonaDirect

    facade = DaytonaSandbox.__new__(DaytonaSandbox)
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    facade._strategy = strategy
    strategy._compose_exec = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout="inner-box\n")
    )
    config = {"Env": ["JAVA_HOME=/opt/jdk"], "Hostname": "inner-box"}
    strategy._vm_exec = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout=json.dumps(config))
    )
    assert await facade.script_container_config() == config
    strategy._compose_exec.assert_awaited_once_with(
        ["ps", "-q", "main"], timeout_sec=30
    )
    strategy._vm_exec.assert_awaited_once_with(
        "docker inspect --format '{{json .Config}}' inner-box", timeout_sec=30
    )
    strategy._vm_exec.return_value.stdout = "invalid JSON with private data"
    with pytest.raises(
        RuntimeError, match=r"^script: invalid Docker container configuration$"
    ):
        await facade.script_container_config()
    facade._strategy = _DaytonaDirect.__new__(_DaytonaDirect)
    assert await facade.script_container_config() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["docker", "remote_docker"])
async def test_script_root_transport_keeps_removed_names_out_of_compose_env(
    backend, monkeypatch
):
    """Guards SCRIPT-AGENT-DOCKER-F1: Compose fills a bare ``-e NAME`` in from
    its own environment, so the protected exec's Compose runs without every
    name the argv removes, whatever the operator's or task's values."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    cls = DockerSandbox if backend == "docker" else RemoteDockerSandbox
    shell = object.__new__(cls)
    shell.script_container_config = AsyncMock(
        return_value={"Env": ["LD_PRELOAD=/app/hook.so", "LD_SHOW_AUXV"]}
    )
    client = {"PATH": "/usr/bin", "LD_PRELOAD": "/host.so", "LD_SHOW_AUXV": "1"}
    monkeypatch.setattr(
        shell,
        "_docker_process_env",
        lambda environment=None: dict(client if environment is None else environment),
    )
    monkeypatch.setattr(shell, "_compose_command", lambda command: ["docker", *command])
    if backend == "remote_docker":
        monkeypatch.setattr(shell, "_redact_result", lambda result, **kw: result)
    shell._run_docker_process = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
    )
    await shell.script_runner_exec("true", timeout_sec=10)
    environment = shell._run_docker_process.call_args.kwargs["environment"]
    assert environment == {"PATH": "/usr/bin"}
    argv = shell._run_docker_process.call_args.args[0]
    assert "-e" in argv and "LD_PRELOAD" in argv
    assert not [a for a in argv if a.startswith("LD_") and "=" in a]


@pytest.mark.asyncio
async def test_dind_identity_exec_removes_loader_names_without_compose(tmp_path):
    """DinD's identity exec is a plain docker exec in the VM's shell with the
    removed names unset there: nothing (the VM's environment, a task .env
    Compose would read) can fill a removed loader name back in."""
    import subprocess as sp

    from benchflow.sandbox.daytona import _DaytonaDinD

    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    strategy._env = SimpleNamespace()
    container = "c" * 64
    (tmp_path / ".env").write_text("LD_PRELOAD=/app/hook.so\n")
    # A docker that prints its own environment, then its arguments.
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\n/usr/bin/env\nprintf "ARG %s\\n" "$@"\n')
    fake.chmod(0o755)
    calls = []

    async def compose_exec(subcommand, **kwargs):
        calls.append(subcommand)
        assert subcommand == ["ps", "-q", "main"]
        return SimpleNamespace(return_code=0, stdout=container + "\n", stderr="")

    async def vm_exec(command, env=None, timeout_sec=None, **kwargs):
        if command.startswith("docker inspect"):
            assert command.endswith(container)
            config = {"Env": ["LD_PRELOAD=/app/hook.so", "LD_SHOW_AUXV", "KEEP=1"]}
            return SimpleNamespace(return_code=0, stdout=json.dumps(config), stderr="")
        done = sp.run(
            ["/bin/sh", "-c", command],
            cwd=tmp_path,
            env={
                "PATH": f"{fake.parent}:/usr/bin:/bin",
                "LD_PRELOAD": "/vm.so",
                "LD_SHOW_AUXV": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    strategy._compose_exec = compose_exec
    strategy._vm_exec = vm_exec
    from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute

    route = ExecRoute(pin=SYSTEM_PATH, system_shell=True, task_root=False)
    with patch(
        "benchflow.sandbox.daytona_dind.system_shell_route",
        AsyncMock(return_value=route),
    ):
        result = await strategy.identity_exec("true")
    assert result.return_code == 0, result.stderr
    lines = result.stdout.splitlines()
    names = {line.partition("=")[0] for line in lines if not line.startswith("ARG ")}
    args = [line[4:] for line in lines if line.startswith("ARG ")]
    assert "LD_PRELOAD" not in names and "LD_SHOW_AUXV" not in names
    assert args[0] == "exec" and "-T" not in args and container in args
    assert "LD_PRELOAD" in args and "KEEP=" in args
    assert not [a for a in args if a.startswith("LD_") and "=" in a]
    assert calls == [["ps", "-q", "main"]]


def test_image_ld_show_auxv_is_never_given_back():
    """Guards SCRIPT-AGENT-DOCKER-F1: an image's LD_SHOW_AUXV=1 came back for
    root's task commands, and the aux vector glibc then printed at every
    program start broke each exec receipt. Other kept values still return."""
    import subprocess as sp

    from benchflow.sandbox import _inert_bootstrap
    from benchflow.sandbox._base import IMAGE_ENV_PREFIX, restore_image_env

    # The plain route (no kit): the kit route skips the same names before
    # its value check, in the same loop.
    script = restore_image_env(()) + "/usr/bin/env"
    done = sp.run(
        ["/bin/sh", "-c", script],
        env={
            "PATH": "/usr/bin:/bin",
            f"{IMAGE_ENV_PREFIX}LD_SHOW_AUXV": "1",
            f"{IMAGE_ENV_PREFIX}TAR_OPTIONS": "--no-same-owner",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    names = {line.partition("=")[0] for line in done.stdout.splitlines()}
    assert "TAR_OPTIONS" in names
    assert "LD_SHOW_AUXV" not in names and not any("AT_" in n for n in names)
    assert not {n for n in names if n.startswith(IMAGE_ENV_PREFIX)}
    sandbox = SimpleNamespace(
        _bootstrap_configs={
            "main": {"Env": ["LD_SHOW_AUXV=1", "TAR_OPTIONS=--no-same-owner"]}
        }
    )
    assert _inert_bootstrap.restored_env(sandbox) == {"TAR_OPTIONS": "--no-same-owner"}
