"""S1-B130-1: a1a5c818 loses native prototypes after a subnet retry's down."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _inert_bootstrap as bootstrap
from benchflow.sandbox._docker_sweep import release_project
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.fixture(autouse=True)
def _without_bfx(monkeypatch):
    # These fake Docker's answers, main's inspect included, which has no
    # bfx mount; main's /.bfx check (present when the pinned binaries are
    # built) is tested in tests/test_bfx_*.py.
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")


def test_launch_admits_the_checked_log_paths_sent_to_compose(tmp_path, monkeypatch):
    """Guards merge 448f3a59: ReachablePath diagnostics are not bind sources."""
    from benchflow.task import compose_env

    task = tmp_path / "task"
    task.mkdir()
    (task / "docker-compose.yaml").write_text("services: {main: {image: trusted}}")
    paths = RolloutPaths(tmp_path / "run")
    paths.mkdir()
    env = DockerSandbox(task, "logs", "logs", paths, SandboxConfig())
    monkeypatch.setattr(
        env, "_stage", lambda: SimpleNamespace(environment_dir=task, task_dir=task)
    )
    for key in ("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env, "_docker_compose_global_args", lambda: [])

    def freeze(**kwargs):
        mounts = [
            {"type": "bind", "source": kwargs["env"][key], "target": "/logs/" + name}
            for key, name in (
                ("HOST_VERIFIER_LOGS_PATH", "verifier"),
                ("HOST_AGENT_LOGS_PATH", "agent"),
                ("HOST_ARTIFACTS_PATH", "artifacts"),
            )
        ]
        assert {m["source"] for m in mounts} == {
            str(paths.rollout_dir / name) for name in ("verifier", "agent", "artifacts")
        }
        refused = kwargs["check"].check({"services": {"main": {"volumes": mounts}}})
        return SimpleNamespace(refused=refused, launch=object())

    monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    env.freeze_launch()
    assert env._launch is not None


@pytest.mark.parametrize("kind", ["first", "restart", "recovery", "checkpoint"])
@pytest.mark.parametrize("refusal", ["create", "up", None])
async def test_subnet_retry_prepares_and_releases_every_service(
    tmp_path, monkeypatch, kind, refusal
):
    """Guards a1a5c818: a retry must keep the native peer and the bind barrier."""
    (tmp_path / "Dockerfile").write_text("FROM trusted")
    env = DockerSandbox(tmp_path, "retry", "retry", None, SandboxConfig())
    project = {"services": {"main": {}, "peer": {}}}
    records = {}
    prepared = []
    released = set()
    events = []
    generation = 0
    refused = False
    proved = False
    config = {
        "Entrypoint": ["/native-service"],
        "Cmd": [],
        "Env": ["BASH_ENV=/task/startup"],
    }

    def create(name):
        cid = f"{generation * 2 + (1 if name == 'main' else 2):064x}"
        original = {**config, "Labels": {"com.docker.compose.service": name}}
        if kind == "checkpoint" and name == "main":
            original = {
                **original,
                "Entrypoint": bootstrap.entrypoint(original),
                "Env": [],
                "Labels": {
                    **original["Labels"],
                    bootstrap.LABEL: bootstrap.config_receipt(original),
                },
            }
        records[cid] = {
            "Image": "trusted",
            "State": {"Running": False, "ExitCode": 0},
            "Config": copy.deepcopy(original),
            "Mounts": [],
        }
        return cid

    def names():
        return {
            r["Config"]["Labels"]["com.docker.compose.service"]: cid
            for cid, r in records.items()
        }

    async def compose(args, **kwargs):
        nonlocal generation, refused
        events.append(args[0])
        if args[0] == refusal and not refused:
            refused = True
            raise RuntimeError("Pool overlaps with other one on this address space")
        output = ""
        if args[0] == "create":
            # Preparation must read the original project, including readiness
            # dependencies, instead of the preceding bootstrap overlay.
            assert env._bootstrap_override is None
            records.clear()
            generation += 1
            for name in project["services"]:
                create(name)
            prepared.append(set(records))
        elif args[0] == "down":
            records.clear()
        elif args[0] == "config":
            output = json.dumps(project)
        elif args[0] == "ps":
            output = "\n".join(records)
        elif args[0] == "up":
            # Compose up recreates only explicitly selected services.
            selected = [name for name in project["services"] if name in args]
            assert selected
            for name in selected:
                cid = names().get(name) or create(name)
                if name == "peer":
                    assert proved, "native sidecar ran before the bind proof"
                    assert "--no-recreate" in args
                    assert cid in prepared[-1]
                    released.add(cid)
                records[cid]["State"]["Running"] = True
        return ExecResult(return_code=0, stdout=output, stderr="")

    async def cli(args, **kwargs):
        if args[0] == "kill":
            assert proved, "guarded command ran before the bind proof"
            released.add(args[-1])
        return ExecResult(return_code=0, stdout="", stderr="")

    async def proof():
        nonlocal proved
        # Use the real service-set/bind-baseline check, then mark release safe.
        await DockerSandbox._pin_or_verify_bind_targets(env)
        proved = True
        events.append("proof")

    async def protected(_env, _project, _prototypes):
        return {"main"}  # main's log bind; peer has no shared mount/namespace

    monkeypatch.setattr(bootstrap, "protected_services", protected)
    monkeypatch.setattr(env, "freeze_launch", lambda *args: None)
    for name in (
        "_run_pre_compose_hook",
        "_build_task_image",
        "_start_from_layers",
        "_choose_subnets",
        "_name_trial_volumes",
        "_note_task_image_pull",
        "_learn_main_container",
        "_prepare_log_dirs_after_up",
        "_record_mount_inventory",
    ):
        monkeypatch.setattr(env, name, AsyncMock(), raising=False)
    env._absent_task_image = AsyncMock(return_value=None)
    env._run_docker_compose_command = compose
    env._docker_cli = cli
    env._inspect_container = AsyncMock(side_effect=lambda cid: records[cid])
    env._main_container_id = AsyncMock(side_effect=lambda: names()["main"])
    env._pin_or_verify_bind_targets = proof
    env._subnets = {"default": "192.0.2.0/28"}
    if kind in {"restart", "recovery", "checkpoint"}:
        env._bind_baseline = {"main": {}, "peer": {}}
        env._compose_up_attempted = True
    if kind == "recovery":
        env._recovery_baseline = SimpleNamespace(
            effective_allow_internet=True, image_id="trusted"
        )
        monkeypatch.setattr("benchflow.sandbox.docker.validate_baseline", AsyncMock())
    try:
        await env.start(False)
        assert set(names()) == {"main", "peer"}
        assert released == set(records) == prepared[-1]
        assert all(record["State"]["Running"] for record in records.values())
        assert env._bootstrap_configs["main"]["Entrypoint"] == ["/native-service"]
        assert env._bootstrap_configs["main"]["Env"] == config["Env"]
        assert refused == (refusal is not None)
        assert len(prepared) == (2 if refusal == "up" else 1)
        assert events.index("proof") > max(
            i for i, event in enumerate(events) if event == "create"
        )
    finally:
        release_project(env.compose_project_name)
        if getattr(env, "_bootstrap_folder", None):
            env._bootstrap_folder.cleanup()
