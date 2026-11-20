"""Restricted network policies on Daytona and Modal are refused before start
(the lead's rulings of 2026-11-15), in the dry run as in the run.

Daytona's rules go in from main itself, where root holds NET_ADMIN, so the
agent phase there holds only a main with no network at all (no-network with
the oracle or a script) or an open one. Allowlist, denylist, the offline
shape (a no-network main kept open for its verifier) and no-network with a
model agent are refused, natively and on Docker in Docker. Modal (gVisor,
no iptables) started a main kept open for a model agent with its network
open (PROBE): no-network with a model agent and the offline shape are refused
there, and so are allowlist and denylist (its capability table refuses them
too, but only after the model's credentials are read: NETSCAN's key-less dry
run named the missing login instead).
"""

# The dry run fixtures are imported from test_dry_run_capabilities.
# ruff: noqa: F811

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.task.config import TaskConfig
from benchflow.task.runtime_capabilities import (
    ALLOWLIST_POLICY,
    DENYLIST_POLICY,
    NO_NETWORK_AGENT_POLICY,
    NO_NETWORK_SCRIPTED_POLICY,
    OFFLINE_POLICY,
    OPEN_POLICY,
    agent_network_policy,
    network_policy_issue,
)
from tests.test_dry_run_capabilities import (  # noqa: F401 (fixtures)
    REACHED,
    _both,
    _plan,
    _task,
    bench,
    sandbox_reached,
)

#: (task.toml lines, an agent that launches, the policy it is).
CELLS = {
    "allowlist": (
        '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        True,
        ALLOWLIST_POLICY,
    ),
    "denylist": (
        '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.org"]\n',
        True,
        DENYLIST_POLICY,
    ),
    "offline": (
        '[environment]\nallow_internet = false\n[verifier]\nnetwork_mode = "public"\n',
        False,
        OFFLINE_POLICY,
    ),
    "no-network-model-agent": (
        "[environment]\nallow_internet = false\n",
        True,
        NO_NETWORK_AGENT_POLICY,
    ),
    "agent-none-online-sandbox": (
        '[agent]\nnetwork_mode = "no-network"\n[environment]\nallow_internet = true\n',
        True,
        NO_NETWORK_AGENT_POLICY,
    ),
    "no-network-oracle": (
        "[environment]\nallow_internet = false\n",
        False,
        NO_NETWORK_SCRIPTED_POLICY,
    ),
    "open": ("", True, OPEN_POLICY),
}
REFUSED = {
    "daytona": {
        "allowlist",
        "denylist",
        "offline",
        "no-network-model-agent",
        "agent-none-online-sandbox",
    },
    "modal": {
        "allowlist",
        "denylist",
        "offline",
        "no-network-model-agent",
        "agent-none-online-sandbox",
    },
}
#: Refused on Modal by its capability table (validate_task_runtime_support).
REFUSED_BY_MODAL_TABLE = {"allowlist", "denylist"}
#: With BENCHFLOW_EGRESS_GATEWAY set, Modal holds the denylist at the egress
#: gateway (benchflow.sandbox.egress_gateway); every other refusal stands.
REFUSED_WITH_GATEWAY = {**REFUSED, "modal": REFUSED["modal"] - {"denylist"}}
GATEWAY_ENV = "BENCHFLOW_EGRESS_GATEWAY"
#: The gateway cells: an [agent] network_mode leaves the agent's phase off the
#: sandbox's denylist, so Modal's capability table still refuses that denylist.
GATEWAY_CELLS = {
    **CELLS,
    "denylist-agent-none-separate-verifier": (
        '[agent]\nnetwork_mode = "no-network"\n[environment]\n'
        'network_mode = "denylist"\nblocked_hosts = ["example.org"]\n'
        '[verifier]\nsandbox_mode = "separate"\n',
        False,
        NO_NETWORK_SCRIPTED_POLICY,
    ),
}
REFUSED_BY_MODAL_TABLE_WITH_GATEWAY = {
    "allowlist",
    "denylist-agent-none-separate-verifier",
}


@pytest.fixture(autouse=True)
def _no_egress_gateway(monkeypatch):
    """The refusals without a gateway are main's; a test that names one sets it."""
    monkeypatch.delenv(GATEWAY_ENV, raising=False)


def _config(toml: str) -> TaskConfig:
    return TaskConfig.model_validate_toml('version = "1.0"\n' + toml)


def _task_dir(root: Path, compose: bool) -> Path:
    environment = root / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose:
        (environment / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    return root / "task"


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize(
    ("backend", "compose", "label"),
    [
        ("daytona", False, "Daytona"),
        ("daytona", True, "Daytona (Docker in Docker)"),
        ("modal", False, "Modal"),
        ("docker", False, None),
        ("docker", True, None),
    ],
    ids=["daytona", "daytona-dind", "modal", "docker", "docker-compose"],
)
def test_each_backend_and_policy(tmp_path, cell, backend, compose, label):
    toml, launches, policy = CELLS[cell]
    config = _config(toml)
    assert agent_network_policy(config, agent_launches=launches) == policy
    issue = network_policy_issue(
        config,
        sandbox=backend,
        agent_launches=launches,
        task_dir=_task_dir(tmp_path, compose),
    )
    if cell in REFUSED.get(backend, ()):
        assert issue is not None
        assert issue.reason == (
            f"the task's network policy ({policy}) is not held on {label}: run "
            "restricted policies on Docker, or choose `open` explicitly "
            "(recorded as open)"
        )
        assert issue.path == "sandbox.network_mode" and issue.sandbox == backend
    else:
        assert issue is None
    if backend == "modal":
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        table = validate_task_runtime_support(config, sandbox="modal")
        assert bool(table) is (cell in REFUSED_BY_MODAL_TABLE), table


@pytest.mark.parametrize("cell", sorted(GATEWAY_CELLS))
@pytest.mark.parametrize(
    ("backend", "compose", "label"),
    [
        ("daytona", False, "Daytona"),
        ("daytona", True, "Daytona (Docker in Docker)"),
        ("modal", False, "Modal"),
        ("docker", False, None),
    ],
    ids=["daytona", "daytona-dind", "modal", "docker"],
)
def test_each_backend_and_policy_with_the_egress_gateway(
    tmp_path, monkeypatch, cell, backend, compose, label
):
    """Guards the PR that adds BENCHFLOW_EGRESS_GATEWAY: with a gateway named,
    Modal admits the denylist (its plan gate and its capability table) and
    nothing else; Daytona and Docker are unchanged. Its second review round:
    the table still refuses a denylist under an [agent] network_mode."""
    monkeypatch.setenv(GATEWAY_ENV, "/run/egress-gateway/control.sock")
    toml, launches, policy = GATEWAY_CELLS[cell]
    config = _config(toml)
    issue = network_policy_issue(
        config,
        sandbox=backend,
        agent_launches=launches,
        task_dir=_task_dir(tmp_path, compose),
    )
    if cell in REFUSED_WITH_GATEWAY.get(backend, ()):
        assert issue is not None
        assert issue.reason.startswith(
            f"the task's network policy ({policy}) is not held on {label}: "
        )
    else:
        assert issue is None
    if backend == "modal":
        from benchflow.task.runtime_capabilities import validate_task_runtime_support

        table = validate_task_runtime_support(config, sandbox="modal")
        assert bool(table) is (cell in REFUSED_BY_MODAL_TABLE_WITH_GATEWAY), table


#: (task.toml lines under a denylist sandbox, the verifier label refused or None).
GATEWAY_VERIFIER_CELLS = {
    "inherited": ("", None),
    "public": ('[verifier]\nnetwork_mode = "public"\n', None),
    "no-network": ('[verifier]\nnetwork_mode = "no-network"\n', "no-network"),
    "allowlist": (
        '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        "allowlist",
    ),
}


@pytest.mark.parametrize("launches", [True, False], ids=["model-agent", "oracle"])
@pytest.mark.parametrize("cell", sorted(GATEWAY_VERIFIER_CELLS))
def test_a_denylist_tasks_shared_verifier_at_the_egress_gateway(
    tmp_path, monkeypatch, cell, launches
):
    """The verifier of a denylist task inherits its denylist (0.10; a task
    cannot declare a verifier denylist of its own): at the gateway it is the
    phase's, held. A shared verifier's no-network or allowlist is still
    refused, as on main, and Daytona refuses them all."""
    monkeypatch.setenv(GATEWAY_ENV, "/run/egress-gateway/control.sock")
    toml, refused = GATEWAY_VERIFIER_CELLS[cell]
    config = _config(
        '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.org"]\n'
        + toml
    )
    task_dir = _task_dir(tmp_path, False)
    issue = network_policy_issue(
        config, sandbox="modal", agent_launches=launches, task_dir=task_dir
    )
    if refused is None:
        assert issue is None, issue
    else:
        assert issue is not None
        assert issue.reason.startswith(
            f"the task's network policy ({refused}, the verifier's) is not held on Modal: "
        )
    daytona = network_policy_issue(
        config, sandbox="daytona", agent_launches=launches, task_dir=task_dir
    )
    assert daytona is not None


@pytest.fixture
def modal_sdk():
    pytest.importorskip("modal", reason="Modal's SDK is an optional extra")


def test_no_network_with_a_model_agent_on_modal_errors_the_same_in_both(
    tmp_path, bench, sandbox_reached, modal_sdk, monkeypatch
):
    """PROBE on rc: Modal started this main with its network open; now no
    sandbox is created, in the dry run as in the run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-a-key")
    _task(tmp_path / "tasks", "nonet", environment="allow_internet = false\n")
    dry, run, recorded = _both(
        tmp_path, bench, "modal", "--model", "claude-opus-5-5", agent="claude-agent-acp"
    )
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert (
        "the task's network policy (no-network, with a model agent) is not held on Modal"
        in recorded
    )
    assert run.exit_code == 1 and dry.exit_code == 1, (run.output, dry.output)
    assert _plan(dry)["errors"] == {"nonet": recorded}


@pytest.fixture
def daytona_sdk():
    pytest.importorskip("daytona", reason="Daytona's SDK is an optional extra")


def test_a_refused_policy_errors_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox_reached, daytona_sdk
):
    _task(
        tmp_path / "tasks",
        "allow",
        environment='network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    )
    dry, run, recorded = _both(tmp_path, bench, "daytona")
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert "the task's network policy (allowlist) is not held on Daytona" in recorded
    assert run.exit_code == 1, run.output
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"allow": recorded}


def test_no_network_with_the_oracle_runs_on_daytona_in_both(
    tmp_path, bench, sandbox_reached, daytona_sdk
):
    _task(tmp_path / "tasks", "nonet", environment="allow_internet = false\n")
    dry, _run, recorded = _both(tmp_path, bench, "daytona")
    assert recorded == REACHED, recorded
    assert dry.exit_code == 0, dry.output


def _compose_available() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        done = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _refusal(output: str) -> str:
    """The refusal block the CLI prints last, from its ``refused:`` line on
    (a dry run prints its plan before it, whose errors quote it)."""
    return output[output.rindex("refused:") :]


@pytest.mark.skipif(
    not _compose_available(), reason="the gate runs docker compose config"
)
@pytest.mark.parametrize(
    ("compose", "words"),
    [
        (
            "services:\n  main:\n    build: .\n    volumes:\n"
            "      - ${HOME}/.config/gcloud:/root/.config/gcloud:ro\n",
            ".config/gcloud",
        ),
        (
            "services:\n  main:\n    build:\n      context: .\n      network: host\n",
            "builds service main on the host network",
        ),
    ],
    ids=["gcloud-bind", "build-network-host"],
)
def test_the_compose_gates_refusals_are_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox_reached, monkeypatch, compose, words
):
    """check_compose's host-access refusals: the dry run's job check and the
    run's are the same, word for word."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    task = _task(tmp_path / "tasks", "hostly")
    (task / "environment" / "docker-compose.yaml").write_text(compose)
    dry = bench("docker", "--dry-run", jobs="dry")
    run = bench("docker", jobs="run")
    assert dry.exit_code == run.exit_code == 1, (dry.output, run.output)
    assert words in _refusal(run.output), run.output
    assert _refusal(dry.output) == _refusal(run.output)
    (job,) = _plan(dry)["errors"].values()
    assert words in job


@pytest.mark.parametrize(
    ("name", "environment", "policy"),
    [
        (
            "deny",
            'network_mode = "denylist"\nblocked_hosts = ["example.org"]\n',
            "denylist",
        ),
        (
            "allow",
            'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
            "allowlist",
        ),
        ("nonet", "allow_internet = false\n", "no-network, with a model agent"),
    ],
    ids=["denylist", "allowlist", "no-network"],
)
def test_a_key_less_dry_run_on_modal_names_the_policy_not_the_login(
    tmp_path, bench, sandbox_reached, modal_sdk, monkeypatch, name, environment, policy
):
    """NETSCAN on 476491c9d: with no model key, a Modal dry run of a denylist
    or allowlist task said "No Claude login found" (the capability table runs
    after the model's credentials are read); with a key it refused. The
    policy needs no credentials, so it is refused first."""
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _task(tmp_path / "tasks", name, environment=environment)
    dry = bench(
        "modal",
        "--dry-run",
        "--model",
        "anthropic/claude-opus-5-5",
        jobs="dry",
        agent="claude-agent-acp",
    )
    assert dry.exit_code == 1, dry.output
    (error,) = _plan(dry)["errors"].values()
    assert error.startswith("UnsupportedTaskFeatureError: "), error
    assert f"the task's network policy ({policy}) is not held on Modal" in error
    assert "login" not in error.lower()


#: (task.toml lines, an agent that launches) for a shared verifier's own
#: restricted network under an open agent phase (RCI's B-1).
VERIFIER_CELLS = {
    "no-network-model-agent": (
        '[verifier]\nnetwork_mode = "no-network"\n',
        True,
        "no-network",
    ),
    "no-network-oracle": (
        '[verifier]\nnetwork_mode = "no-network"\n',
        False,
        "no-network",
    ),
    "allowlist-oracle": (
        '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        False,
        "allowlist",
    ),
}


@pytest.mark.parametrize("cell", sorted(VERIFIER_CELLS))
@pytest.mark.parametrize("separate", [False, True], ids=["shared", "separate"])
@pytest.mark.parametrize(
    ("backend", "compose", "label"),
    [
        ("daytona", False, "Daytona"),
        ("daytona", True, "Daytona (Docker in Docker)"),
        ("modal", False, "Modal"),
        ("docker", False, None),
    ],
    ids=["daytona", "daytona-dind", "modal", "docker"],
)
def test_a_shared_verifiers_own_network_on_each_backend(
    tmp_path, cell, separate, backend, compose, label
):
    """A shared verifier's rules go in from main, where root holds NET_ADMIN
    on Daytona and where Modal has none: refused there before start. A
    separate verifier sandbox takes its own network at creation (Daytona's
    block_all, Modal's block_network) and runs. Docker's helper holds both."""
    toml, launches, mode = VERIFIER_CELLS[cell]
    if separate:
        toml = toml.replace("[verifier]\n", '[verifier]\nsandbox_mode = "separate"\n')
    config = _config("[environment]\nallow_internet = true\n" + toml)
    issue = network_policy_issue(
        config,
        sandbox=backend,
        agent_launches=launches,
        task_dir=_task_dir(tmp_path, compose),
    )
    if backend != "docker" and not separate:
        assert issue is not None
        assert issue.reason.startswith(
            f"the task's network policy ({mode}, the verifier's) is not held on {label}: "
        ), issue.reason
    else:
        assert issue is None


def test_a_no_network_main_gives_its_shared_verifier_none_too(tmp_path):
    """no-network with the oracle: main has no network at all, and neither
    has its shared no-network verifier. Nothing to hold, so it runs."""
    config = _config(
        '[environment]\nallow_internet = false\n[verifier]\nnetwork_mode = "no-network"\n'
    )
    for backend in ("daytona", "modal"):
        assert (
            network_policy_issue(
                config,
                sandbox=backend,
                agent_launches=False,
                task_dir=_task_dir(tmp_path / backend, False),
            )
            is None
        )


def test_a_shared_no_network_verifier_errors_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox_reached, daytona_sdk
):
    """RCI's B-1: the 16 vintage-forensics variants' shape (public task,
    [verifier] network_mode = "no-network", shared), with the oracle."""
    _task(tmp_path / "tasks", "vnonet", verifier='network_mode = "no-network"\n')
    dry, run, recorded = _both(tmp_path, bench, "daytona")
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert (
        "the task's network policy (no-network, the verifier's) is not held on Daytona"
        in recorded
    )
    assert run.exit_code == 1 and dry.exit_code == 1, (run.output, dry.output)
    assert _plan(dry)["errors"] == {"vnonet": recorded}


ONLINE_AGENT_NONE = (
    '[agent]\nnetwork_mode = "no-network"\n[environment]\nallow_internet = true\n'
    '[verifier]\nnetwork_mode = "no-network"\n'
)


@pytest.mark.parametrize("agent", ["oracle", "nop", "task-runtime"])
@pytest.mark.parametrize(
    ("backend", "compose", "label"),
    [
        ("daytona", False, "Daytona"),
        ("daytona", True, "Daytona (Docker in Docker)"),
        ("modal", False, "Modal"),
    ],
    ids=["daytona", "daytona-dind", "modal"],
)
def test_an_online_sandbox_with_an_offline_agent_still_holds_its_verifier_from_main(
    tmp_path, agent, backend, compose, label
):
    """RCI's B-2 on a5bf299d8: an online sandbox whose [agent] network is
    no-network gave a scripted agent the no-network label, and the shared
    no-network verifier was let through, though main has a network. The
    preflight's own reading of the agent (rollout.preflight_rollout_network_policy)
    decides who launches."""
    from types import SimpleNamespace

    from benchflow.rollout import preflight_rollout_network_policy
    from benchflow.task.runtime_capabilities import UnsupportedTaskFeatureError

    task_dir = _task_dir(tmp_path, compose)
    (task_dir / "task.toml").write_text('version = "1.0"\n' + ONLINE_AGENT_NONE)
    task = SimpleNamespace(config=_config(ONLINE_AGENT_NONE))
    cfg = SimpleNamespace(environment=backend, primary_agent=agent, task_path=task_dir)
    with pytest.raises(UnsupportedTaskFeatureError) as refused:
        preflight_rollout_network_policy(task, cfg)
    assert (
        f"the task's network policy (no-network, the verifier's) is not held on {label}: "
        in str(refused.value)
    )


def _online_agent_none_task(root: Path, name: str) -> Path:
    task = _task(
        root,
        name,
        environment="allow_internet = true\n",
        verifier='network_mode = "no-network"\n',
    )
    toml = task / "task.toml"
    text = toml.read_text()
    assert "[agent]\ntimeout_sec = 300\n" in text
    toml.write_text(
        text.replace(
            "[agent]\ntimeout_sec = 300\n",
            '[agent]\ntimeout_sec = 300\nnetwork_mode = "no-network"\n',
        )
    )
    return task


@pytest.mark.parametrize("agent", ["oracle", "nop"])
@pytest.mark.parametrize("backend", ["daytona", "modal"])
def test_an_online_sandbox_with_an_offline_agent_errors_the_same_in_both(
    tmp_path, bench, sandbox_reached, agent, backend
):
    pytest.importorskip(backend, reason=f"{backend}'s SDK is an optional extra")
    _online_agent_none_task(tmp_path / "tasks", "agentnone")
    dry, run, recorded = _both(tmp_path, bench, backend, agent=agent)
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert "the task's network policy (no-network, the verifier's)" in recorded
    assert run.exit_code == 1 and dry.exit_code == 1, (run.output, dry.output)
    assert _plan(dry)["errors"] == {"agentnone": recorded}


def test_a_truly_offline_sandbox_with_the_oracle_still_runs_on_daytona(
    tmp_path, bench, sandbox_reached, daytona_sdk
):
    """The control: an offline sandbox and its shared no-network verifier,
    with the oracle. Main has no network at all, and neither has test.sh."""
    _task(
        tmp_path / "tasks",
        "offline",
        environment="allow_internet = false\n",
        verifier='network_mode = "no-network"\n',
    )
    dry, _run, recorded = _both(tmp_path, bench, "daytona")
    assert recorded == REACHED, recorded
    assert dry.exit_code == 0, dry.output


def test_no_label_exempts_a_shared_verifier_while_the_sandbox_has_a_network(tmp_path):
    """The lead's guard (11:06Z): the exemption keys on the sandbox's real
    network (allow_internet False), never on a label. Every label
    agent_network_policy returns, built with an online sandbox and a shared
    no-network verifier, is refused on each backend whose row holds the
    shared verifier. A label constant this grid does not reach fails here,
    so a future label cannot reopen RCI's B-2."""
    import itertools

    import benchflow.task.runtime_capabilities as rc

    labels = {
        getattr(rc, name)
        for name in dir(rc)
        if name.endswith("_POLICY") and name != "SHARED_VERIFIER_POLICY"
    }
    seen_online: set[str] = set()
    seen_offline: set[str] = set()
    sandboxes = {
        "open": "allow_internet = true\n",
        "allowlist": 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
        "denylist": 'network_mode = "denylist"\nblocked_hosts = ["example.org"]\n',
        "offline": "allow_internet = false\n",
    }
    agents = {
        "unset": "",
        "none": 'network_mode = "no-network"\n',
        "allowlist": 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    }
    verifiers = {
        "no-network": 'network_mode = "no-network"\n',
        "public": 'network_mode = "public"\n',
    }
    for (s, sbx), (a, agt), (v, ver), launches in itertools.product(
        sandboxes.items(), agents.items(), verifiers.items(), (True, False)
    ):
        toml = f"[agent]\n{agt}[environment]\n{sbx}[verifier]\n{ver}"
        try:
            config = _config(toml)
        except Exception:
            continue  # a combination the task format refuses
        label = agent_network_policy(config, agent_launches=launches)
        if config.sandbox.allow_internet is False:
            seen_offline.add(label)
            continue
        seen_online.add(label)
        if v != "no-network":
            continue
        for backend in ("daytona", "modal"):
            issue = network_policy_issue(
                config, sandbox=backend, agent_launches=launches, task_dir=None
            )
            assert issue is not None, (backend, s, a, launches, label)
    assert labels <= seen_online | seen_offline, labels - seen_online - seen_offline
    # The offline shape is the only label that needs an offline sandbox.
    assert labels - {OFFLINE_POLICY} <= seen_online, labels - seen_online
