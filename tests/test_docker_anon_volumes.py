"""Every volume a trial makes is named and labelled at creation (DOCKER-ANON-VOLUME-LEAK r3).

Rounds 1 and 2 inferred after teardown which anonymous volumes a trial
made, and both directions failed (an operator's inherited volume removed;
the trial's own kept). Now Docker makes no anonymous volume for a trial:
:mod:`benchflow.sandbox._trial_volumes` names each one in a Compose overlay,
with the project's label, before any container is created, and teardown
removes by that label. These tests pin the overlay and how the sandbox
reads its inputs; ``test_anon_volume_container.py`` runs it on Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _trial_volumes
from benchflow.sandbox._trial_volumes import LABEL, Unreadable, overlay
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult, SandboxStartupError


def _project(**services):
    return {"name": "p", "services": services}


def _mounts(document, service):
    return {
        m["target"]: m
        for m in document.get("services", {}).get(service, {}).get("volumes", [])
    }


def test_an_image_volume_gets_a_labelled_named_volume():
    document = overlay(_project(main={"image": "img"}), {"img": ["/data/"]}, {}, 1)
    assert _mounts(document, "main") == {
        "/data": {"type": "volume", "target": "/data", "source": "bfvol1_main_0"}
    }
    assert document["volumes"] == {"bfvol1_main_0": {"labels": {LABEL: "p"}}}


def test_a_compose_anonymous_volume_keeps_its_options():
    main = {
        "image": "img",
        "volumes": [
            {
                "type": "volume",
                "target": "/cache",
                "read_only": True,
                "volume": {"nocopy": True},
            },
            {"type": "volume", "source": "named", "target": "/named"},
            {"type": "bind", "source": "/host", "target": "/bind"},
        ],
    }
    document = overlay(_project(main=main), {"img": []}, {}, 3)
    assert _mounts(document, "main") == {
        "/cache": {
            "type": "volume",
            "target": "/cache",
            "read_only": True,
            "volume": {"nocopy": True},
            "source": "bfvol3_main_0",
        }
    }


def test_a_path_the_service_mounts_already_is_left_alone():
    main = {
        "image": "img",
        "volumes": [
            {"type": "bind", "source": "/host", "target": "/a"},
            {"type": "volume", "source": "named", "target": "/b"},
        ],
        "tmpfs": ["/c:size=1m"],
    }
    document = overlay(_project(main=main), {"img": ["/a", "/b", "/c", "/d"]}, {}, 1)
    assert set(_mounts(document, "main")) == {"/d"}


def test_a_project_volumes_from_shares_the_sources_volumes():
    """The source names its own; the inheritor gets the source's through
    volumes_from, as before, and names only the paths no source gives."""
    project = _project(
        main={"image": "img", "volumes_from": ["side:ro"]},
        side={
            "image": "side-img",
            "volumes": [{"type": "volume", "target": "/shared"}],
            "volumes_from": ["base"],
        },
        base={"image": "base-img"},
    )
    images = {
        "img": ["/data", "/shared", "/base", "/own"],
        "side-img": ["/data"],
        "base-img": ["/base"],
    }
    document = overlay(project, images, {}, 1)
    assert set(_mounts(document, "side")) == {"/shared", "/data"}
    assert set(_mounts(document, "base")) == {"/base"}
    assert set(_mounts(document, "main")) == {"/own"}
    assert "volumes_from" not in document["services"]["main"]


@pytest.mark.parametrize(
    ("side", "passed"),
    [
        # Not mount points: Docker does not copy them, so main's VOLUME at
        # /data gets a volume, which must be named (round 3 review A, M1).
        ({"image": "plain", "tmpfs": ["/data"]}, False),
        ({"image": "plain", "volumes": [{"type": "tmpfs", "target": "/data"}]}, False),
        # The source's own VOLUME under its own tmpfs gives it no volume.
        ({"image": "img", "tmpfs": "/data:size=1m"}, False),
        ({"image": "plain", "secrets": [{"source": "env", "target": "/data"}]}, False),
        (
            {"image": "plain", "configs": [{"source": "inline", "target": "/data"}]},
            False,
        ),
        # Mount points: Docker copies them, so main gets no volume there.
        ({"image": "img"}, True),
        ({"image": "plain", "volumes": [{"type": "volume", "target": "/data"}]}, True),
        (
            {
                "image": "plain",
                "volumes": [{"type": "volume", "source": "op", "target": "/data"}],
            },
            True,
        ),
        (
            {
                "image": "plain",
                "volumes": [{"type": "bind", "source": "/h", "target": "/data"}],
            },
            True,
        ),
        (
            {
                "image": "plain",
                "volumes": [{"type": "image", "source": "img", "target": "/data"}],
            },
            True,
        ),
        # Compose bind-mounts file secrets and configs.
        ({"image": "plain", "secrets": [{"source": "file", "target": "/data"}]}, True),
        ({"image": "plain", "configs": [{"source": "file", "target": "/data"}]}, True),
        ({"image": "plain", "configs": [{"source": "file", "target": "data"}]}, True),
    ],
    ids=[
        "tmpfs",
        "tmpfs-entry",
        "own-volume-under-tmpfs",
        "env-secret",
        "content-config",
        "image-volume",
        "anonymous",
        "named",
        "bind",
        "image-mount",
        "file-secret",
        "file-config",
        "file-config-relative",
    ],
)
def test_volumes_from_passes_only_the_sources_mount_points(side, passed):
    """What Docker copies through ``volumes_from`` (checked on Docker 29.1.3,
    Compose 2.40.3, results/ANONVOL.md round 4): volume, bind and image
    mounts and the binds of file secrets and configs; never a tmpfs."""
    project = {
        **_project(main={"image": "img", "volumes_from": ["side"]}, side=side),
        "secrets": {"file": {"file": "/s"}, "env": {"environment": "HOME"}},
        "configs": {"file": {"file": "/c"}, "inline": {"content": "x"}},
    }
    document = overlay(project, {"img": ["/data"], "plain": []}, {}, 1)
    assert ("/data" in _mounts(document, "main")) is not passed


def test_a_file_secret_or_config_covers_the_services_own_volume_path():
    main = {
        "image": "img",
        "secrets": [{"source": "s", "target": "/run/secrets/s"}, "t"],
        "configs": [{"source": "c"}, {"source": "inline", "target": "/i"}],
    }
    project = {
        **_project(main=main),
        "secrets": {"s": {"file": "/s"}, "t": {"file": "/t"}},
        "configs": {"c": {"file": "/c"}, "inline": {"content": "x"}},
    }
    images = {"img": ["/run/secrets/s", "/run/secrets/t", "/c", "/i"]}
    assert set(_mounts(overlay(project, images, {}, 1), "main")) == {"/i"}


def test_an_external_volumes_from_path_is_left_to_its_source():
    project = _project(main={"image": "img", "volumes_from": ["container:operator:ro"]})
    document = overlay(
        project, {"img": ["/opdata", "/data"]}, {"operator": ["/opdata"]}, 1
    )
    assert set(_mounts(document, "main")) == {"/data"}


@pytest.mark.parametrize(
    ("project", "images", "outside"),
    [
        (_project(main={"image": "img"}), {}, {}),
        (
            _project(main={"image": "img", "volumes_from": ["container:gone"]}),
            {"img": []},
            {},
        ),
        (_project(main={"image": "img", "volumes_from": ["nosuch"]}), {"img": []}, {}),
        (
            _project(
                main={"image": "img", "volumes_from": ["side"]},
                side={"image": "img", "volumes_from": ["main"]},
            ),
            {"img": []},
            {},
        ),
        (
            {
                "name": "p",
                "services": {"main": {"image": "img"}},
                "volumes": {"bfvol1_main_0": {}},
            },
            {"img": ["/data"]},
            {},
        ),
    ],
    ids=["image-unread", "source-unread", "no-service", "cycle", "name-taken"],
)
def test_what_cannot_be_read_stops_the_overlay(project, images, outside):
    with pytest.raises(Unreadable):
        overlay(project, images, outside, 1)


def test_nothing_to_name_is_no_overlay():
    main = {
        "image": "img",
        "volumes": [{"type": "bind", "source": "/h", "target": "/b"}],
    }
    assert overlay(_project(main=main), {"img": []}, {}, 1) == {}


def test_a_build_without_image_reads_composes_default_name():
    assert _trial_volumes.image_of(_project(main={"build": {}}), "main") == "p-main"


# --- DockerSandbox._name_trial_volumes ---------------------------------------


def _sandbox(
    tmp_path: Path,
    project: dict,
    images: dict,
    containers=None,
    pulled=None,
    pull_error=None,
):
    """A DockerSandbox whose frozen launch is *project*; the daemon has
    *images* (name -> VOLUME paths) and outside *containers* (name ->
    mount destinations); a compose pull adds *pulled*, or fails with
    *pull_error*."""
    sandbox = object.__new__(DockerSandbox)
    sandbox._compose_project = "p"
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(project))
    sandbox._launch = SimpleNamespace(path=path)
    calls: list[list[str]] = []
    images = dict(images)

    async def cli(args, check=True, *, raw_output=False):
        calls.append(list(args))
        if args[:2] == ["image", "inspect"]:
            if args[-1] not in images:
                return ExecResult(stdout="", stderr="No such image", return_code=1)
            body = {path: {} for path in images[args[-1]]} or None
            return ExecResult(stdout=json.dumps(body), stderr="", return_code=0)
        if args[0] == "inspect":
            found = (containers or {}).get(args[-1])
            if found is None:
                return ExecResult(
                    stdout="[]", stderr="No such container", return_code=1
                )
            body = [{"Destination": destination} for destination in found]
            return ExecResult(stdout=json.dumps(body), stderr="", return_code=0)
        raise AssertionError(args)

    async def compose(command, check=True, timeout_sec=None, **_):
        calls.append(["compose", *command])
        if pull_error is not None:
            return ExecResult(stdout=pull_error, stderr="", return_code=1)
        images.update(pulled or {})
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox._docker_cli = cli
    sandbox._run_docker_compose_command = compose
    return sandbox, calls


def _written(sandbox):
    path = sandbox._trial_volumes_compose_path
    return json.loads(path.read_text()) if path is not None else None


@pytest.mark.asyncio
async def test_the_overlay_is_in_every_compose_call_and_new_for_each_attempt(
    tmp_path,
):
    sandbox, calls = _sandbox(
        tmp_path, _project(main={"image": "img"}), {"img": ["/data"]}
    )
    await sandbox._name_trial_volumes()
    first = sandbox._trial_volumes_compose_path
    assert first in sandbox._docker_compose_paths
    assert set(_written(sandbox)["volumes"]) == {"bfvol1_main_0"}
    # A subnet retry (or a second start) gets volumes no earlier attempt used.
    sandbox._write_trial_volumes_overlay()
    assert not first.exists()
    assert set(_written(sandbox)["volumes"]) == {"bfvol2_main_0"}
    sandbox._drop_trial_volumes_overlay()
    assert sandbox._trial_volumes_compose_path is None
    assert not any(c[0] == "compose" for c in calls)


@pytest.mark.asyncio
async def test_an_absent_image_is_pulled_before_it_is_read(tmp_path):
    project = _project(
        main={"image": "img"}, side={"image": "side", "pull_policy": "always"}
    )
    sandbox, calls = _sandbox(
        tmp_path, project, {"side": []}, pulled={"img": ["/data"], "side": ["/s"]}
    )
    await sandbox._name_trial_volumes()
    assert ["compose", "pull", "--quiet", "--ignore-buildable", "main", "side"] in calls
    assert set(_mounts(_written(sandbox), "main")) == {"/data"}
    assert set(_mounts(_written(sandbox), "side")) == {"/s"}


@pytest.mark.asyncio
async def test_an_image_that_cannot_be_read_stops_the_start(tmp_path):
    sandbox, _ = _sandbox(tmp_path, _project(main={"image": "img"}), {})
    with pytest.raises(SandboxStartupError, match="VOLUMEs of image 'img'"):
        await sandbox._name_trial_volumes()


@pytest.mark.asyncio
async def test_a_failed_pull_stops_the_start_with_the_registrys_reason(tmp_path):
    """Round 3 review A, L2: the reason (a typo, a login, a rate limit) is
    what the user acts on."""
    reason = (
        "Error response from daemon: pull access denied for nosuch, repository "
        "does not exist or may require 'docker login'"
    )
    sandbox, _ = _sandbox(
        tmp_path, _project(main={"image": "nosuch"}), {}, pull_error=reason
    )
    with pytest.raises(SandboxStartupError) as raised:
        await sandbox._name_trial_volumes()
    assert str(raised.value) == f"Cannot pull image 'nosuch': {reason}"
    assert isinstance(raised.value, RuntimeError)


@pytest.mark.asyncio
async def test_an_outside_source_is_read_as_a_container(tmp_path):
    """Round 2 review B: an untyped inspect of the source's name returned an
    image of that name. The source is read with ``--type container``."""
    project = _project(main={"image": "img", "volumes_from": ["container:op"]})
    sandbox, calls = _sandbox(
        tmp_path,
        project,
        {"img": ["/data", "/opdata"]},
        containers={"op": ["/opdata"]},
    )
    await sandbox._name_trial_volumes()
    inspect = ["inspect", "--type", "container", "--format", "{{json .Mounts}}", "op"]
    assert inspect in calls
    assert set(_mounts(_written(sandbox), "main")) == {"/data"}


@pytest.mark.asyncio
async def test_an_outside_source_that_cannot_be_read_stops_the_start(tmp_path):
    project = _project(main={"image": "img", "volumes_from": ["container:op"]})
    sandbox, _ = _sandbox(tmp_path, project, {"img": ["/data"]})
    with pytest.raises(SandboxStartupError, match="volumes_from"):
        await sandbox._name_trial_volumes()


# --- the direct Engine path makes no volume ------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        # What the overlay adds for an image VOLUME or an anonymous entry.
        lambda p: overlay(p, {_trial_volumes.image_of(p, "main"): ["/data"]}, {}, 1),
        # A Compose anonymous volume as the task writes it.
        lambda p: {
            "services": {"main": {"volumes": [{"type": "volume", "target": "/c"}]}}
        },
        # An external volumes_from.
        lambda p: {"services": {"main": {"volumes_from": ["container:op"]}}},
    ],
    ids=["overlay", "anonymous", "volumes_from"],
)
def test_the_direct_path_takes_no_project_that_gets_a_volume(change):
    """PERF-E's path renders only binds, one service and no top-level
    volumes (and sends an image that declares VOLUME to Compose,
    test_direct_docker_container), so it never makes a volume."""
    from benchflow.sandbox import _direct_docker as dd
    from tests.test_direct_docker import _project as direct_project

    project = direct_project()
    dd.check_project(project, project["name"])  # taken as it is
    added = change(project)
    main = project["services"]["main"]
    main["volumes"] = [
        *main.get("volumes", []),
        *added["services"]["main"].get("volumes", []),
    ]
    if "volumes_from" in added["services"]["main"]:
        main["volumes_from"] = added["services"]["main"]["volumes_from"]
    if added.get("volumes"):
        project["volumes"] = added["volumes"]
    with pytest.raises(dd.Fallback):
        dd.check_project(project, project["name"])


@pytest.mark.asyncio
async def test_a_trial_whose_volumes_are_named_says_so_and_stays_on_compose(
    tmp_path,
):
    """With the overlay written, the direct plan refuses before it asks
    Compose for the project, with the reason, not the key that refused."""
    from benchflow.sandbox import _direct_docker as dd

    sandbox, calls = _sandbox(
        tmp_path, _project(main={"image": "img"}), {"img": ["/data"]}
    )
    await sandbox._name_trial_volumes()
    calls.clear()
    with pytest.raises(dd.Fallback, match="BenchFlow names this trial's volumes"):
        await sandbox._direct_plan()
    assert calls == []


# --- the layer build's libc probe ----------------------------------------------


def _layers(run):
    """DockerLayers whose ``docker run`` is *run*; ``rm`` and ``volume rm``
    are recorded, and a volume stays in use until its container is gone and
    twice more (its ``--rm`` still removing it)."""
    from benchflow.sandbox._docker_subnets import daemon_key
    from benchflow.sandbox.env_layers import DockerLayers, _Run

    layers = DockerLayers(daemon=daemon_key(None))
    calls: list[tuple[str, ...]] = []
    busy = {"tries": 2}

    async def docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "run":
            return await run(args)
        if args[:2] == ("volume", "rm") and busy["tries"]:
            busy["tries"] -= 1
            return _Run("", "Error: volume is in use", 1)
        return _Run("", "", 0)

    async def nothing(*_):
        return None

    layers._docker = docker
    layers.daemon_id = nothing
    layers._libc_from_layers = nothing
    return layers, calls


def _probe_cleanup(calls):
    (run,) = [c for c in calls if c[0] == "run"]
    container = run[run.index("--name") + 1]
    volumes = [
        a.split("src=")[1].split(",")[0]
        for i, a in enumerate(run)
        if i and run[i - 1] == "--mount"
    ]
    after = calls[calls.index(run) + 1 :]
    assert after[0] == ("rm", "--force", container), after
    removed = [c[2] for c in after if c[:2] == ("volume", "rm")]
    return volumes, removed


@pytest.mark.asyncio
async def test_the_libc_probe_removes_its_container_then_its_volumes():
    from benchflow.sandbox.env_layers import _Run

    async def run(args):
        return _Run("glibc\n", "", 0)

    layers, calls = _layers(run)
    assert await layers._libc("sha256:img", "d", ["/data", "/cache"]) == "glibc"
    volumes, removed = _probe_cleanup(calls)
    assert len(volumes) == 2 and set(removed) == set(volumes)
    assert len(removed) == 4  # the first two met "in use" and were tried again


@pytest.mark.asyncio
async def test_a_cancelled_libc_probe_still_removes_its_volumes():
    """Round 3 review A, L3: the cancel kills the CLI only; the container
    goes by name first, so its volumes can go."""
    import asyncio

    started = asyncio.Event()

    async def run(args):
        started.set()
        await asyncio.sleep(3600)

    layers, calls = _layers(run)
    probe = asyncio.ensure_future(layers._libc("sha256:img", "d", ["/data"]))
    await started.wait()
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    volumes, removed = _probe_cleanup(calls)
    assert volumes and removed[-1] == volumes[0] and set(removed) == set(volumes)
