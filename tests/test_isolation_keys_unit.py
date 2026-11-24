"""Task-file and working-folder settings that weakened isolation by
themselves on 0.10.0, as the census found them (the PR's table): the rows
that need no sandbox.

- A Compose ``security_opt`` other than ``no-new-privileges*`` and
  BenchFlow's own ``writable-cgroups=true`` (an AppArmor profile, an SELinux
  label, a seccomp profile) changes how the kernel confines the container:
  refused without ``BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1``, as ``unconfined``
  already was.
- A ``.env`` in the folder ``bench`` runs from may belong to the benchmark: no
  ``BENCHFLOW_ALLOW_*`` or ``BENCHFLOW_METADATA_*`` switch is taken from it.
- A setup command's capture writes into BenchFlow's own environment (and a
  dotenv file an operator variable names): only for a variable the operator
  lists with ``--allow-task-env`` (the file's variable too), never a
  ``BENCHFLOW_*`` name; otherwise the trial is refused before anything starts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tests.test_compose_oracle import HOST_ACCESS, _compose_task, _gate, _refused

# --------------------------------------------------------------- security_opt


@pytest.mark.parametrize(
    "option",
    [
        "apparmor=bf-test-profile",
        "apparmor:lsb_release",
        "label=type:spc_t",
        "label:user:USER",
        "no-new-privileges:false",
        "no-new-privileges=false",
        "seccomp=profile.json",
        "seccomp:/etc/docker/seccomp.json",
        "systempaths=unconfined",
        "mask=/proc/kcore",
    ],
)
def test_a_security_opt_that_changes_the_confinement_needs_host_access(
    monkeypatch, tmp_path, option
):
    task = _compose_task(
        tmp_path / "task",
        f'services:\n  main:\n    image: alpine\n    security_opt: ["{option}"]\n',
    )
    message = _refused(task)
    assert f"runs service main with security_opt {option}" in message, message
    assert option in _refused(task, "daytona")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


@pytest.mark.parametrize(
    "option",
    [
        "no-new-privileges",
        "no-new-privileges:true",
        "no-new-privileges=true",
        "apparmor=docker-default",
    ],
)
def test_no_new_privileges_and_docker_s_own_profile_pass(tmp_path, option):
    task = _compose_task(
        tmp_path / "task",
        f'services:\n  main:\n    image: alpine\n    security_opt: ["{option}"]\n',
    )
    _gate(task)


@pytest.mark.parametrize("service", ["main", "side"])
@pytest.mark.parametrize("option", ["writable-cgroups=true", "writable-cgroups:true"])
def test_a_task_asking_for_writable_cgroups_needs_host_access(
    monkeypatch, tmp_path, option, service
):
    """BenchFlow's kept-service freezer sets it on main itself;
    a task's own, on main or a sidecar, is refused without host access."""
    body = "services:\n  main:\n    image: alpine\n"
    if service == "side":
        body += "  side:\n    image: alpine\n"
    body += f'    security_opt: ["{option}"]\n'
    task = _compose_task(tmp_path / "task", body)
    message = _refused(task)
    assert f"runs service {service} with security_opt {option}" in message, message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


@pytest.mark.parametrize(
    ("service", "freezer", "refused"),
    [("main", True, False), ("main", False, True), ("side", True, True)],
    ids=["main-under-the-freezer", "main-without-it", "a-sidecar-under-it"],
)
def test_the_launch_gate_admits_writable_cgroups_only_where_the_freezer_puts_it(
    service, freezer, refused
):
    """The launch gate reads the run's whole project, BenchFlow's freezer
    overlay included: its writable-cgroups on main is admitted where the run
    stacks the freezer, nowhere else."""
    from benchflow.task.compose_env import _host_settings

    document = {
        "services": {
            "main": {"image": "alpine"},
            "side": {"image": "alpine"},
        }
    }
    document["services"][service]["security_opt"] = ["writable-cgroups=true"]
    said = list(_host_settings(document, freezer_main=freezer))
    assert bool(said) is refused, said


# ----------------------------------------------------------------- .env keys

DOTENV_SWITCHES = (
    "BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD",
    "BENCHFLOW_ALLOW_SOMETHING_LATER",
    "BENCHFLOW_METADATA_HELPER",
    "BENCHFLOW_METADATA_MAIN_NET_ADMIN",
)


def test_a_dotenv_sets_no_isolation_switch(monkeypatch, tmp_path):
    """``bench`` copies the working folder's .env into its environment; none
    of BenchFlow's isolation switches comes from there."""
    from benchflow.cli._shared import _apply_dotenv_to_process_env

    dotenv = tmp_path / "benchmark.env"
    dotenv.write_text(
        "".join(f"{name}=1\n" for name in DOTENV_SWITCHES) + "S1_PLAIN=from-dotenv\n"
    )
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    for name in (*DOTENV_SWITCHES, "S1_PLAIN"):
        monkeypatch.setenv(name, "x")  # so teardown restores each
        monkeypatch.delenv(name)
    _apply_dotenv_to_process_env()
    assert os.environ["S1_PLAIN"] == "from-dotenv"
    taken = [name for name in DOTENV_SWITCHES if name in os.environ]
    assert not taken, f"taken from .env: {taken}"


def test_an_exported_switch_still_counts(monkeypatch, tmp_path):
    """Guard: the operator's own exported value is untouched."""
    from benchflow.cli._shared import _apply_dotenv_to_process_env

    dotenv = tmp_path / "benchmark.env"
    dotenv.write_text("BENCHFLOW_METADATA_HELPER=1\n")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    monkeypatch.setenv("BENCHFLOW_METADATA_HELPER", "0")
    _apply_dotenv_to_process_env()
    assert os.environ["BENCHFLOW_METADATA_HELPER"] == "0"


# ------------------------------------------------------------- setup capture


def _capture_task(root: Path, name: str, file_var: str | None = None) -> Path:
    task = root / "capture"
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    line = f'capture_dir_b64_env_file_var = "{file_var}"\n' if file_var else ""
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 60.0\n\n'
        "[verifier]\ntimeout_sec = 60.0\n\n"
        '[environment]\ndocker_image = "python:3.12-slim"\n\n'
        "[[environment.setup_commands]]\n"
        'command = "mkdir -p /tmp/cap && echo hi > /tmp/cap/x"\n'
        'capture_dir = "/tmp/cap"\n'
        f'capture_dir_b64_env = "{name}"\n' + line
    )
    (task / "instruction.md").write_text("Nothing.\n")
    for path in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        path.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
        path.chmod(0o755)
    return task


def _dry_run(task: Path) -> None:
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    dry_run_plan_gate(
        RolloutConfig(task_path=task, agent="oracle", environment="docker"),
        default_rollout_planes(),
    )


@pytest.mark.parametrize(
    ("name", "file_var", "allowed", "says"),
    [
        ("S1_CAPTURE", None, "", "--allow-task-env S1_CAPTURE"),
        ("S1_CAPTURE", "S1_DOTENV", "S1_CAPTURE", "--allow-task-env S1_DOTENV"),
        (
            "BENCHFLOW_ALLOW_TASK_HOST_ACCESS",
            None,
            "BENCHFLOW_ALLOW_TASK_HOST_ACCESS",
            "never a BENCHFLOW_ name",
        ),
        (
            "S1_CAPTURE",
            "BENCHFLOW_DOTENV_PATH",
            "S1_CAPTURE,BENCHFLOW_DOTENV_PATH",
            "never a BENCHFLOW_ name",
        ),
    ],
    ids=["unlisted", "dotenv-unlisted", "benchflow-name", "benchflow-file-var"],
)
def test_a_setup_capture_the_operator_did_not_list_is_refused(
    monkeypatch, tmp_path, name, file_var, allowed, says
):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", allowed)
    task = _capture_task(tmp_path, name, file_var)
    with pytest.raises(Exception) as caught:
        _dry_run(task)
    text = str(caught.value)
    assert says in text and "capture" in text, text


@pytest.mark.parametrize("file_var", [None, "S1_DOTENV"])
def test_a_listed_setup_capture_passes(monkeypatch, tmp_path, file_var):
    monkeypatch.setenv(
        "BENCHFLOW_ALLOW_TASK_ENV", "S1_CAPTURE" + (f",{file_var}" if file_var else "")
    )
    _dry_run(_capture_task(tmp_path, "S1_CAPTURE", file_var))


class _CaptureEnv:
    """A sandbox whose captured folder holds one file."""

    async def download_dir(self, source, local_dir, service="main"):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "x").write_text("hi\n")


@pytest.mark.parametrize(
    ("name", "allowed"),
    [
        ("S1_PREWRITE", ""),
        ("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"),
    ],
    ids=["unlisted", "benchflow-name"],
)
async def test_the_capture_checks_again_before_it_writes(monkeypatch, name, allowed):
    """Behind the refusal before setup: the capture itself writes nothing
    the operator did not list."""
    from benchflow.rollout import _capture_setup_command_dir

    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", allowed)
    monkeypatch.setenv(name, "before")
    with pytest.raises(Exception) as caught:
        await _capture_setup_command_dir(
            _CaptureEnv(),
            source_dir="/tmp/cap",
            target_env_var=name,
            target_dotenv_path_env_var=None,
            service="main",
        )
    assert os.environ[name] == "before"
    assert "capture" in str(caught.value), caught.value


# -------------------------------------------------------------------- runtime


@pytest.mark.parametrize("runtime", ["runc", "nvidia", "${S1_RT:-runc}"])
def test_a_compose_runtime_needs_host_access(monkeypatch, tmp_path, runtime):
    """On a host whose default runtime sandboxes containers (gVisor, Kata), a
    task's ``runtime: runc`` would leave it: every ``runtime`` a task's
    Compose file names needs host access."""
    task = _compose_task(
        tmp_path / "task",
        f"services:\n  main:\n    image: alpine\n    runtime: {runtime}\n",
    )
    message = _refused(task)
    assert "runs service main under the container runtime" in message, message
    assert "under the container runtime" in _refused(task, "daytona")
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


# ------------------------------------- an operator's network over phase keys


def _network_config(extra: str):
    import tomllib

    from benchflow.task.config import TaskConfig

    return TaskConfig.model_validate(tomllib.loads('version = "1.0"\n' + extra))


NO_NETWORK = {"sandbox": {"network_mode": "no-network"}}


def test_an_operator_no_network_overlay_holds_the_verifier_too():
    """The operator's ``--config-override`` sets no-network; the task's own
    ``[verifier] network_mode = "public"`` no longer opens grading."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.sandbox.verifier_network import verifier_network_mode

    config = _network_config('[verifier]\nnetwork_mode = "public"\n')
    patched = apply_config_override(config, NO_NETWORK)
    assert verifier_network_mode(patched) == "no-network"
    from benchflow._utils.config_override import operator_network_replacements

    assert operator_network_replacements(config, NO_NETWORK) == [
        "verifier.network_mode = public"
    ]


def test_an_agent_allowlist_does_not_fold_over_an_operator_no_network():
    from benchflow._utils.config_override import apply_config_override
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config

    config = _network_config(
        '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n'
    )
    patched = apply_config_override(config, NO_NETWORK)
    assert str(agent_network_sandbox_config(patched).network_mode.value) == "no-network"


def test_a_separate_verifier_sandbox_takes_the_operator_network():
    from benchflow._utils.config_override import apply_config_override
    from benchflow.sandbox.verifier_network import separate_verifier_network

    config = _network_config('[verifier.sandbox]\nnetwork_mode = "public"\n')
    patched = apply_config_override(config, NO_NETWORK)
    assert separate_verifier_network(patched) == "no-network"


def test_an_operator_allowlist_replaces_a_task_s_other_verifier_allowlist():
    from benchflow._utils.config_override import apply_config_override

    config = _network_config(
        '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n'
    )
    overlay = {
        "sandbox": {"network_mode": "allowlist", "allowed_hosts": ["example.com"]}
    }
    patched = apply_config_override(config, overlay)
    assert patched.verifier.allowed_hosts == ["example.com"]


STEPS = (
    '[[steps]]\nname = "first"\n[steps.verifier]\nnetwork_mode = "public"\n'
    '[[steps]]\nname = "second"\n[steps.agent]\nnetwork_mode = "allowlist"\n'
    'allowed_hosts = ["example.com"]\n'
    '[[steps]]\nname = "third"\n[steps.agent]\nnetwork_mode = "no-network"\n'
)


def test_an_operator_no_network_overlay_holds_every_step_s_phase_keys():
    """A multi-step task's own step keys (a step's verifier on
    ``public``, a step's agent allowlist) take the operator's no-network; a
    step's own stricter key stays, and the record names each replacement."""
    from benchflow._utils.config_override import (
        apply_config_override,
        operator_network_replacements,
    )

    config = _network_config(STEPS)
    patched = apply_config_override(config, NO_NETWORK)
    first, second, third = patched.steps
    assert (
        str(getattr(first.verifier.network_mode, "value", first.verifier.network_mode))
        == "no-network"
    )
    assert (
        str(getattr(second.agent.network_mode, "value", second.agent.network_mode))
        == "no-network"
    )
    assert not second.agent.allowed_hosts
    assert (
        str(getattr(third.agent.network_mode, "value", third.agent.network_mode))
        == "no-network"
    )
    assert operator_network_replacements(config, NO_NETWORK) == [
        "steps[0].verifier.network_mode = public",
        "steps[1].agent.network_mode = allowlist",
    ]


def test_an_operator_allowlist_replaces_a_step_s_other_allowlist():
    from benchflow._utils.config_override import apply_config_override

    config = _network_config(STEPS)
    overlay = {"sandbox": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]}}
    patched = apply_config_override(config, overlay)
    assert patched.steps[1].agent.allowed_hosts == ["pypi.org"]
    assert (
        str(
            getattr(
                patched.steps[0].verifier.network_mode,
                "value",
                patched.steps[0].verifier.network_mode,
            )
        )
        == "allowlist"
    )


@pytest.mark.parametrize(
    ("task", "overlay", "kept"),
    [
        ('[verifier]\nnetwork_mode = "public"\n', None, "public"),
        (
            '[verifier]\nnetwork_mode = "public"\n',
            {"agent": {"timeout_sec": 30}},
            "public",
        ),
        (
            '[verifier]\nnetwork_mode = "no-network"\n',
            {"sandbox": {"network_mode": "public"}},
            "no-network",
        ),
    ],
    ids=["no-overlay", "overlay-without-network", "stricter-phase-key-stays"],
)
def test_the_task_s_own_phase_network_stays_otherwise(task, overlay, kept):
    """Guards: without an operator network the task's verifier network is its
    choice; an overlay never loosens a phase key that is stricter."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.sandbox.verifier_network import verifier_network_mode

    config = _network_config(task)
    assert verifier_network_mode(apply_config_override(config, overlay)) == kept


# ------------------------------------------ a service sharing main's namespaces


@pytest.mark.parametrize(
    ("compose", "shown"),
    [
        (
            "services:\n  main:\n    image: alpine\n"
            "  side:\n    image: alpine\n    pid: service:main\n    cap_add: [SYS_PTRACE]\n",
            ["shares main's pid namespace with service side"],
        ),
        (
            "services:\n  main:\n    image: alpine\n"
            "  side:\n    image: alpine\n    ipc: service:main\n",
            ["shares main's ipc namespace with service side"],
        ),
        (
            "services:\n  main:\n    image: alpine\n"
            "  side:\n    image: alpine\n    pid: service:main\n"
            "  side2:\n    image: alpine\n    pid: service:side\n",
            [
                "shares main's pid namespace with service side",
                "shares main's pid namespace with service side2",
            ],
        ),
        (
            "services:\n  main:\n    image: alpine\n    pid: service:side\n"
            "  side:\n    image: alpine\n    cap_add: [SYS_PTRACE]\n",
            ["shares main's pid namespace with service side"],
        ),
        (
            "services:\n  main:\n    image: alpine\n    ipc: service:side\n"
            "  side:\n    image: alpine\n",
            ["shares main's ipc namespace with service side"],
        ),
        (
            "services:\n  main:\n    image: alpine\n    pid: service:anchor\n"
            "  side:\n    image: alpine\n    pid: service:anchor\n"
            "    cap_add: [SYS_PTRACE]\n"
            "  anchor:\n    image: alpine\n",
            [
                "shares main's pid namespace with service anchor",
                "shares main's pid namespace with service side",
            ],
        ),
        (
            "services:\n  main:\n    image: alpine\n    ipc: service:anchor\n"
            "  side:\n    image: alpine\n    ipc: service:anchor\n"
            "  anchor:\n    image: alpine\n",
            [
                "shares main's ipc namespace with service anchor",
                "shares main's ipc namespace with service side",
            ],
        ),
        (
            "services:\n  main:\n    image: alpine\n    pid: service:side2\n"
            "  side:\n    image: alpine\n    pid: service:side2\n"
            "  side2:\n    image: alpine\n    pid: service:anchor\n"
            "  anchor:\n    image: alpine\n",
            [
                "shares main's pid namespace with service anchor",
                "shares main's pid namespace with service side",
                "shares main's pid namespace with service side2",
            ],
        ),
    ],
    ids=[
        "side-joins-main-pid",
        "side-joins-main-ipc",
        "chain-to-main-pid",
        "c3e-main-joins-side-pid",
        "main-joins-side-ipc",
        "c3f-both-join-anchor-pid",
        "both-join-anchor-ipc",
        "main-chain-to-anchor-pid",
    ],
)
def test_any_pid_or_ipc_namespace_main_shares_needs_host_access(
    monkeypatch, tmp_path, compose, shown
):
    """Whichever way the join goes (a service joining main, main joining a
    service, or both joining a third), the two share one pid namespace, and a
    service with SYS_PTRACE there reaches the agent's processes: refused
    without host access, on every backend."""
    task = _compose_task(tmp_path / "task", compose)
    for message in (_refused(task), _refused(task, "daytona")):
        found = re.findall(
            r"shares main's (?:pid|ipc) namespace with service \w+", message
        )
        assert sorted(found) == sorted(shown), message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


@pytest.mark.parametrize(
    "compose",
    [
        "  side:\n    image: alpine\n    ipc: service:side2\n  side2:\n    image: alpine\n",
        "  side:\n    image: alpine\n    network_mode: service:main\n",
    ],
    ids=[
        "ipc-between-sidecars",
        "network-service-main",
    ],
)
def test_namespaces_between_sidecars_and_main_s_network_join_pass(tmp_path, compose):
    task = _compose_task(
        tmp_path / "task", "services:\n  main:\n    image: alpine\n" + compose
    )
    _gate(task)


@pytest.mark.parametrize(
    ("compose", "shown"),
    [
        (
            "  side:\n    image: alpine\n  other:\n    image: alpine\n    pid: service:side\n",
            ["shares service side's pid namespace with service other"],
        ),
        (
            "  side:\n    image: alpine\n    pid: service:anchor\n"
            "  other:\n    image: alpine\n    pid: service:anchor\n  anchor:\n    image: alpine\n",
            [
                "shares service anchor's pid namespace with service other",
                "shares service anchor's pid namespace with service side",
            ],
        ),
    ],
    ids=["pid-between-sidecars", "sidecars-share-an-anchor"],
)
def test_pid_sharing_between_any_services_needs_host_access(
    monkeypatch, tmp_path, compose, shown
):
    """Every service waits in its hold until its rules are in,
    and a process in a shared pid namespace could release a peer's hold early
    (root's signal reaches any waiter there), so sharing one between sidecars
    needs host access too, not only with main."""
    task = _compose_task(
        tmp_path / "task", "services:\n  main:\n    image: alpine\n" + compose
    )
    message = _refused(task)
    assert all(line in message for line in shown), message
    # The reason is the hold's: a peer could release a held waiter early,
    # not that it reaches this machine.
    assert "signal the other's held waiter and release it" in message, message
    assert "held waiter and release it before its rules are in, which" not in message
    assert "a pid namespace shared between services" in message, message
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


def test_main_joining_a_sidecar_s_network_passes(tmp_path):
    """``network_mode: service:`` sharing stays as it is (ruling E)."""
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: alpine\n    network_mode: service:side\n"
        "  side:\n    image: alpine\n",
    )
    _gate(task)
