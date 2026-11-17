"""Check what runs (the lead's 07:19Z and 07:25Z rulings): the
network-policy check and main's namespace topology read the Compose project
the run itself resolves and freezes (``docker compose ... config --format
json`` from the run's files, env and project: ``DockerSandbox.freeze_launch``),
never the raw docker-compose.yaml.

Both operator host-access flags are set here, so the host-access check stays
out of the way and the policy check is what refuses: a restricted policy is
refused whatever those flags say.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox.service_cover import ServiceCoverError, restricted_policy


def _compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        found = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


pytestmark = pytest.mark.skipif(
    not _compose_available(),
    reason="the run resolves its project with `docker compose config`",
)


@pytest.fixture(autouse=True)
def _operator(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    monkeypatch.setenv("BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD", "1")
    for name in ("NETWORK_MODE", "CAP"):
        monkeypatch.delenv(name, raising=False)


ALLOWLIST = '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n'


def _task(root: Path, compose: str, toml: str = ALLOWLIST, files=None) -> Path:
    environment = root / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    (environment / "docker-compose.yaml").write_text(compose)
    for name, text in (files or {}).items():
        (environment / name).write_text(text)
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (root / "instruction.md").write_text("Say hi.\n")
    (root / "task.toml").write_text('version = "1.0"\n' + toml)
    return root


def _frozen(task: Path):
    """The run's own freeze, with the policy the trial factory sets."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task import Task

    config = Task(task).config
    sandbox = DockerSandbox(task / "environment", "t", "t__1", None, config.sandbox)
    sandbox._network_policy = restricted_policy(config)
    # As the factory's sandbox runs: every rule from the helper, so no
    # BenchFlow overlay gives main NET_ADMIN (DockerSandbox.rules_by_helper).
    sandbox._rules_by_helper = True
    sandbox.freeze_launch()
    return sandbox


def _main(sandbox) -> dict:
    return sandbox._resolved_compose["services"]["main"]


def test_the_corpus_default_resolves_to_bridge_and_runs(tmp_path):
    """drone-planning-control: ``network_mode: ${NETWORK_MODE:-bridge}``."""
    task = _task(
        tmp_path, "services:\n  main:\n    network_mode: ${NETWORK_MODE:-bridge}\n"
    )
    sandbox = _frozen(task)
    assert _main(sandbox)["network_mode"] == "bridge"
    assert sandbox._resolved_topology.main_shares_outside_network is False


def test_host_in_the_runs_env_refuses(tmp_path):
    task = _task(
        tmp_path,
        "services:\n  main:\n    network_mode: ${NETWORK_MODE:-bridge}\n",
        ALLOWLIST + 'env = { NETWORK_MODE = "host" }\n',
    )
    with pytest.raises(ServiceCoverError, match=r"main the host's network"):
        _frozen(task)


def test_host_in_the_operators_shell_reaches_neither_the_check_nor_the_up(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NETWORK_MODE", "host")
    task = _task(
        tmp_path, "services:\n  main:\n    network_mode: ${NETWORK_MODE:-bridge}\n"
    )
    sandbox = _frozen(task)
    assert _main(sandbox)["network_mode"] == "bridge"
    # The launch every later Compose call uses: its own frozen file and env.
    launch = sandbox._launch
    assert "NETWORK_MODE" not in launch.env
    frozen = json.loads(Path(launch.path).read_text())
    assert frozen["services"]["main"]["network_mode"] == "bridge"


def test_the_up_reads_the_document_that_was_checked(tmp_path):
    """Check and up cannot see different values: every later Compose call
    reads the one frozen file, whose services are the checked model's."""
    task = _task(
        tmp_path, "services:\n  main:\n    network_mode: ${NETWORK_MODE:-bridge}\n"
    )
    sandbox = _frozen(task)
    launch = sandbox._launch
    assert sandbox._docker_compose_paths == [Path(launch.path)]
    frozen = json.loads(Path(launch.path).read_text())
    for name, service in sandbox._resolved_compose["services"].items():
        assert frozen["services"][name].get("network_mode") == service.get(
            "network_mode"
        )
        assert frozen["services"][name].get("cap_add") == service.get("cap_add")


def test_an_unresolvable_model_refuses(tmp_path):
    from benchflow.task.env import TaskEnvNotAllowedError

    task = _task(
        tmp_path, "services:\n  main:\n    network_mode: ${NETWORK_MODE:?set it}\n"
    )
    with pytest.raises(TaskEnvNotAllowedError, match="does not resolve") as refused:
        _frozen(task)
    # Its own reason: nothing here reaches outside the task.
    head = str(refused.value).splitlines()[0]
    assert head == "refused: the task has Compose files that do not resolve", head
    assert "NETWORK_MODE is missing a value" in str(refused.value)


BASE = "services:\n  base:\n    image: alpine:3.20\n    privileged: true\n"


@pytest.mark.parametrize(
    ("compose", "files", "named"),
    [
        (
            "services:\n  main:\n    extends:\n      file: base.yaml\n      service: base\n",
            {"base.yaml": BASE},
            "main privileged",
        ),
        (
            "include:\n  - side.yaml\nservices:\n  main: {}\n",
            {
                "side.yaml": "services:\n  side:\n    image: alpine:3.20\n    network_mode: host\n"
            },
            "side the host's network",
        ),
        (
            'services:\n  main:\n    cap_add: ["${CAP:-NET_ADMIN}"]\n',
            {},
            "main NET_ADMIN",
        ),
        (
            "include:\n  - side.yaml\nservices:\n  main: {}\n",
            {
                "side.yaml": "services:\n  side:\n    image: alpine:3.20\n    privileged: true\n"
            },
            "side privileged",
        ),
        # RCI's interp-env: the task's own .env, which the run's Compose reads.
        (
            'services:\n  main:\n    cap_add: ["${RCI_CAP:-CHOWN}"]\n',
            {".env": "RCI_CAP=NET_ADMIN\n"},
            "main NET_ADMIN",
        ),
        # RCI's extends-na.
        (
            "services:\n  main:\n    extends:\n      file: base.yaml\n      service: base\n",
            {
                "base.yaml": "services:\n  base:\n    image: alpine:3.20\n    cap_add: [NET_ADMIN]\n"
            },
            "main NET_ADMIN",
        ),
    ],
    ids=[
        "extends-privileged",
        "include-host-network",
        "interpolated-cap-add",
        "include-bad",
        "interp-env",
        "extends-na",
    ],
)
def test_what_compose_resolves_is_what_is_refused(tmp_path, compose, files, named):
    task = _task(tmp_path, compose, files=files)
    with pytest.raises(ServiceCoverError, match="holds every service") as refused:
        _frozen(task)
    assert named in str(refused.value)


DENYLIST = '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.org"]\n'


def test_a_clean_include_passes(tmp_path):
    task = _task(
        tmp_path,
        "include:\n  - side.yaml\nservices:\n  main: {}\n",
        toml=DENYLIST,
        files={"side.yaml": "services:\n  side:\n    image: alpine:3.20\n"},
    )
    sandbox = _frozen(task)
    assert "side" in sandbox._resolved_compose["services"]
    assert sandbox._resolved_topology.main_shares_outside_network is False


def test_an_allowlists_included_service_with_a_route_out_is_refused_at_the_freeze(
    tmp_path,
):
    """Under allowlist a service the run's project includes, on a
    network that is not internal, is refused when the launch freezes, before
    any container starts (the plan's read of the task's own file cannot see
    an included one); on an internal network it passes."""
    from benchflow.sandbox.egress_denylist import TaskServiceEgressError

    task = _task(
        tmp_path,
        "include:\n  - side.yaml\nservices:\n  main: {}\n",
        files={"side.yaml": "services:\n  side:\n    image: alpine:3.20\n"},
    )
    with pytest.raises(TaskServiceEgressError, match="'side'"):
        _frozen(task)
    inner = _task(
        tmp_path / "inner",
        "include:\n  - side.yaml\nservices:\n  main: {}\n",
        files={
            "side.yaml": "services:\n  side:\n    image: alpine:3.20\n"
            "    networks: [inner]\nnetworks:\n  inner:\n    internal: true\n"
        },
    )
    assert "side" in _frozen(inner)._resolved_compose["services"]


def test_a_file_the_run_does_not_pass_changes_nothing(tmp_path):
    """docker-compose.override.yaml: the run names its files (-f), so
    Compose never reads it, and neither does the check."""
    task = _task(
        tmp_path,
        "services:\n  main: {}\n",
        files={
            "docker-compose.override.yaml": "services:\n  main:\n    network_mode: host\n"
        },
    )
    assert _main(_frozen(task)).get("network_mode") != "host"


@pytest.mark.parametrize(
    "lines",
    [
        "    cap_add: [NET_ADMIN]\n",
        "    cap_add: [NET_RAW]\n",
        "    cap_add: [ALL]\n",
        "    privileged: true\n",
        "    network_mode: host\n",
    ],
    ids=["net-admin", "net-raw", "all", "privileged", "host-network"],
)
def test_the_metadata_opt_out_never_lifts_the_policy(tmp_path, lines):
    task = _task(
        tmp_path,
        "services:\n  main:\n" + lines,
        ALLOWLIST + "allow_cloud_metadata = true\n",
    )
    with pytest.raises(ServiceCoverError, match="opt-out lifts only the metadata rule"):
        _frozen(task)


@pytest.mark.parametrize(
    "toml",
    [
        '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.org"]\n',
        ALLOWLIST,
    ],
    ids=["denylist", "allowlist"],
)
def test_a_restricted_task_with_its_own_compose_starts_without_either(tmp_path, toml):
    """NETSCAN's corpus shape (its denylist tasks): a task compose file
    under a restricted policy is not refused, and the main the run starts
    holds neither NET_ADMIN nor NET_RAW."""
    task = _task(tmp_path, "services:\n  main:\n    environment:\n      A: b\n", toml)
    main = _main(_frozen(task))
    assert not {"NET_ADMIN", "NET_RAW", "ALL"} & set(main.get("cap_add") or [])
    assert "NET_RAW" in (main.get("cap_drop") or [])


def test_an_open_network_keeps_them(tmp_path):
    task = _task(tmp_path, "services:\n  main:\n    cap_add: [NET_ADMIN]\n", toml="")
    assert _main(_frozen(task))["cap_add"] == ["NET_ADMIN"]


@pytest.mark.asyncio
async def test_a_running_service_that_differs_from_the_checked_model_stops_the_trial():
    """After start, each running service's NetworkMode, CapAdd and Privileged
    (as the daemon reports them) against the model that was checked and
    frozen: a power the model did not give stops the trial."""
    from benchflow.sandbox.service_cover import cover_task_services, running_power_drift

    model = {"services": {"main": {}, "db": {"cap_add": ["NET_ADMIN"]}}}
    targets = [
        {
            "service": "main",
            "id": "m" * 64,
            "network_mode": "host",
            "cap_add": [],
            "privileged": False,
        },
        {
            "service": "db",
            "id": "d" * 64,
            "network_mode": "bf_default",
            "cap_add": ["CAP_NET_ADMIN"],
            "privileged": False,
        },
    ]
    assert running_power_drift(targets, model) == {"main": ["the host's network"]}

    class Env:
        _resolved_compose = model

        async def metadata_service_targets(self, include_main=False):
            return targets

        def metadata_service_close(self):
            pass

    with pytest.raises(ServiceCoverError, match="did not give it"):
        await cover_task_services(
            Env(), None, mode="public", metadata=True, opt_out=False, main_metadata=True
        )
