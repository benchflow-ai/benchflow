"""Every start of a held compose service is BenchFlow's, on a real Docker
daemon against the VM's real metadata server.

Docker never starts a held container by itself (its restart policy is
``no``, whatever the task set). Where the task's policy would start a service
again at start-up, BenchFlow does: it reads where each of the stopped
container's mounts lands, starts the same container, proves its binds, covers
it and releases it. After start-up an exit is final: the service stays down,
and the trial is scored and flagged. A command BenchFlow runs in a held
service (its healthcheck, a collect hook) runs only once it is released.

Each service here reports to a server on the host what it got from the
metadata server's instance id (never a token), who asked, and what its PID 1
was. The rows run only where the host gets that answer (a GCE VM, meant to
have no service account).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests._report_server import ReportServer
from tests.test_metadata_optin_docker import (
    _docker_ready,
    _host_reaches_metadata,
    _refusal_text,
    _sdk_run,
    _task,
    build_probe_image,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
    pytest.mark.skipif(
        not _host_reaches_metadata(),
        reason="the host gets no answer from a GCE metadata server",
    ),
]

LABEL = "bf.test.heldstarts"


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


def _build(tmp_path_factory, base: str, lines: str) -> str:
    context = tmp_path_factory.mktemp("held-starts-image")
    (context / "Dockerfile").write_text(f"FROM {base}\n{lines}")
    tag = f"bf-test-heldstarts:{uuid.uuid4().hex[:12]}"
    built = subprocess.run(
        ["docker", "build", "--quiet", "--label", f"{LABEL}=image", "-t", tag, "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if built.returncode != 0:
        pytest.fail(f"cannot build {tag}: {built.stderr[-300:]}", pytrace=False)
    return tag


@pytest.fixture(scope="module")
def linked_image(probe_image, tmp_path_factory):
    """The probe image with a link to /.bfx in it."""
    tag = _build(tmp_path_factory, probe_image, "RUN ln -s /.bfx /x\n")
    yield tag
    subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


@pytest.fixture(scope="module")
def usr1_image(probe_image, tmp_path_factory):
    """The probe image whose stop signal is the waiter's release signal."""
    tag = _build(tmp_path_factory, probe_image, "STOPSIGNAL SIGUSR1\n")
    yield tag
    subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


class Reports(ReportServer):
    """A server on the host each service tells what it got."""

    def __init__(self) -> None:
        super().__init__()
        self.host = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, check=True
        ).stdout.split()[0]

    def program(self, who: str, tail: str = "") -> str:
        """Python that reports once as *who*, then runs *tail* (which may
        report too, to ``http://__HOST__:__PORT__/r?...``)."""
        return (
            (REPORT + tail)
            .replace("__WHO__", who)
            .replace("__HOST__", self.host)
            .replace("__PORT__", str(self.port))
        )

    def by(self, who: str) -> list[dict]:
        return [h for h in self.hits if h.get("who") == who]

    def reached(self) -> list[dict]:
        return [h for h in self.hits if str(h.get("res", "")).startswith("200")]


REPORT = r"""
import os, urllib.request, urllib.parse
def fetch():
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"}), timeout=3)
        return "200"
    except Exception as e:
        return "ERR:" + type(e).__name__
pid1 = open("/proc/1/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:60]
q = urllib.parse.urlencode({"who": "__WHO__", "res": fetch(), "pid1": pid1,
     "boot": open("/proc/1/stat").read().split()[21], "uid": os.getuid()})
try:
    urllib.request.urlopen("http://__HOST__:__PORT__/r?" + q, timeout=3)
except Exception:
    pass
"""


@pytest.fixture
def reports():
    server = Reports()
    yield server
    server.close()


def _compose(task: Path, services: dict, **top: dict) -> None:
    (task / "environment" / "docker-compose.yaml").write_text(
        json.dumps({"services": services, **top})
    )


def _script(task: Path, name: str, body: str) -> None:
    path = task / ("tests" if name == "test.sh" else "solution") / name
    path.write_text(body)
    path.chmod(0o755)


def _run(tmp_path: Path, task: Path):
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    result = json.loads((trial / "result.json").read_text()) if trial else {}
    return outcome, result


def _reward(result: dict):
    return (result.get("rewards") or {}).get("reward")


def _boots(hits: list[dict]) -> set[str]:
    return {h.get("boot") for h in hits}


PASS = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
FAIL = "#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n"


# A program that reports, then exits 1 after *n* s (0.3 or 7: past the 5 s
# that start-up watches a restarted service for).
def _exits_after(reports: Reports, seconds: float) -> list[str]:
    return [
        "python3",
        "-c",
        reports.program(
            "program", f"import time; time.sleep({seconds}); raise SystemExit(1)"
        ),
    ]


def _failing_check(reports: Reports) -> dict:
    return {
        "test": [
            "CMD",
            "python3",
            "-c",
            reports.program("healthcheck", "raise SystemExit(1)"),
        ],
        "interval": "1s",
        "timeout": "5s",
        "retries": 8,
    }


# --------------------------------------------------- restarts at start-up


@pytest.mark.parametrize("shape", ["depends-on", "late"])
def test_the_healthcheck_of_a_service_that_keeps_exiting_runs_only_once_released(
    tmp_path, probe_image, reports, shape
):
    """A sidecar with ``restart: always`` whose program keeps exiting, and
    whose healthcheck BenchFlow runs: main waits for it healthy
    (``depends-on``), or it exits 7 s in, after the start's watch
    (``late``). The check runs, the service starts again (each start a new
    boot), and nothing reaches the server; the start fails closed."""
    task = _task(tmp_path, probe_image)
    side = {
        "image": probe_image,
        "command": _exits_after(reports, 7 if shape == "late" else 0.3),
        "healthcheck": _failing_check(reports),
        "restart": "always",
    }
    main = (
        {}
        if shape == "late"
        else {"depends_on": {"side": {"condition": "service_healthy"}}}
    )
    _compose(task, {"main": main, "side": side})
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    assert len(_boots(reports.by("program"))) >= 2, reports.hits
    assert reports.by("healthcheck"), reports.hits
    assert not reports.reached(), reports.reached()
    assert _reward(result) is None, (result.get("rewards"), _refusal_text(outcome))


def test_a_crash_looping_sidecar_fails_closed_at_the_restart_limit(
    tmp_path, probe_image, reports
):
    """The same sidecar with no healthcheck: BenchFlow starts it again as
    its policy says, up to the limit, and the start is refused."""
    task = _task(tmp_path, probe_image)
    side = {
        "image": probe_image,
        "command": _exits_after(reports, 0.3),
        "restart": "always",
    }
    _compose(task, {"main": {}, "side": side})
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "times during startup" in said, said
    assert not reports.reached(), reports.reached()
    assert len(_boots(reports.by("program"))) >= 3, reports.hits


READY_AFTER_TWO = r"""
import os, pathlib, time, http.server, socketserver
count = pathlib.Path("/tmp/starts")
n = int(count.read_text()) + 1 if count.exists() else 1
count.write_text(str(n))
if n <= 2:
    time.sleep(0.5)
    raise SystemExit(1)
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 8080), http.server.SimpleHTTPRequestHandler).serve_forever()
"""


def test_a_sidecar_that_needs_two_restarts_comes_up_through_benchflow(
    tmp_path, probe_image, reports
):
    """The restart-until-ready shape: the sidecar exits twice, then serves.
    BenchFlow starts the same container again each time (its own file of
    starts kept), covers and releases it, main waits for it healthy, and the
    oracle passes; none of its three starts reaches the server."""
    task = _task(tmp_path, probe_image)
    side = {
        "image": probe_image,
        "command": ["python3", "-c", reports.program("program", READY_AFTER_TWO)],
        "healthcheck": {
            "test": [
                "CMD",
                "python3",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=1)",
            ],
            "interval": "1s",
            "retries": 10,
        },
        "restart": "on-failure",
    }
    _compose(
        task,
        {
            "main": {"depends_on": {"side": {"condition": "service_healthy"}}},
            "side": side,
        },
    )
    _script(
        task,
        "test.sh",
        "#!/bin/bash\npython3 -c \"import urllib.request; urllib.request.urlopen('http://side:8080/', timeout=5)\" "
        "&& echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n",
    )
    outcome, result = _run(tmp_path, task)
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))
    assert len(_boots(reports.by("program"))) == 3, reports.hits
    assert not reports.reached(), reports.reached()
    assert not result.get("held_exited") and not result.get("held_after_restart"), (
        result
    )


RELINK = r"""
import os, time
if not os.path.islink("/data"):
    os.rename("/data", "/data.old")
    os.symlink("/.bfx", "/data")
time.sleep(1)
raise SystemExit(1)
"""


def test_a_service_that_relinks_its_own_mount_is_not_started_again(
    tmp_path, probe_image, reports
):
    """Its program (root in its own container) moves the folder above its
    mount and leaves a link to /.bfx there, then exits. BenchFlow reads where
    the mount lands before starting it again, does not start it, and the
    trial goes on: scored as test.sh says and flagged, the service
    held_exited from the start-up with the reason; the task's file never
    runs."""
    task = _task(tmp_path, probe_image)
    evil = task / "environment" / "g"
    evil.write_text("#!/usr/local/bin/python3\n" + reports.program("task-file"))
    evil.chmod(0o755)
    side = {
        "image": probe_image,
        "command": ["python3", "-c", reports.program("program", RELINK)],
        "volumes": ["./g:/data/bfx:ro"],
        "restart": "always",
    }
    _compose(task, {"main": {}, "side": side})
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    assert not reports.by("task-file"), reports.hits
    assert not reports.reached(), reports.reached()
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))
    exited = result.get("held_exited") or []
    assert [e.get("service") for e in exited] == ["side"], exited
    assert "lands on /.bfx/bfx" in str(exited[0].get("start_refused")), exited
    assert any("not started again" in w for w in result.get("warnings") or []), result


READ_DATA_ONCE = r"""
import os, pathlib, time, urllib.request
swapped = os.path.exists("/data/swapped")
urllib.request.urlopen(f"http://__HOST__:__PORT__/r?who=data&swapped={swapped}", timeout=3)
if not pathlib.Path("/tmp/started").exists():
    pathlib.Path("/tmp/started").touch()
    time.sleep(1.5)
    raise SystemExit(1)
time.sleep(3600)
"""


def test_a_bind_whose_source_changed_between_starts_is_not_started_again(
    tmp_path, probe_image, reports
):
    """The sidecar exits once at start-up, and its bind's folder on the host
    is swapped for another before BenchFlow starts it again. Its binds are
    proven again at that start, as at the first: the changed bind is
    refused, the service stays down (held_exited from the start-up, with
    the reason), and nothing of it reads the other folder."""
    task = _task(tmp_path, probe_image)
    (task / "environment" / "data").mkdir()
    side = {
        "image": probe_image,
        "command": ["python3", "-c", reports.program("program", READ_DATA_ONCE)],
        "volumes": ["./data:/data:ro"],
        "restart": "on-failure",
    }
    _compose(task, {"main": {}, "side": side})
    _script(task, "test.sh", PASS)

    def swap() -> None:
        # The bind's source as Docker has it (BenchFlow's staged copy of the
        # task), swapped once the first start has read it.
        end = time.time() + 300
        while time.time() < end and not reports.by("data"):
            time.sleep(0.05)
        listed = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=com.docker.compose.service=side"],
            capture_output=True,
            text=True,
        ).stdout.split()
        for cid in listed:
            mounts = json.loads(
                subprocess.run(
                    ["docker", "inspect", "-f", "{{json .Mounts}}", cid],
                    capture_output=True,
                    text=True,
                ).stdout
                or "[]"
            )
            for mount in mounts:
                source = Path(mount.get("Source") or "")
                if mount.get("Destination") == "/data" and source.name == "data":
                    source.rename(source.with_name("data.first"))
                    source.mkdir()
                    (source / "swapped").write_text("")
                    return

    threading.Thread(target=swap, daemon=True).start()
    outcome, result = _run(tmp_path, task)
    assert [h.get("swapped") for h in reports.by("data")] == ["False"], reports.hits
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))
    exited = result.get("held_exited") or []
    assert [e.get("service") for e in exited] == ["side"], exited
    assert "bind identity changed" in str(exited[0].get("start_refused")), exited


# ------------------------------------------------- before the first start


def _pinned_copy(path: Path) -> None:
    from benchflow._bfx import locate

    shutil.copyfile(locate.binary_path("x86_64"), path)
    path.chmod(0o755)


@pytest.mark.parametrize("what", ["task-file", "pinned-copy", "secret"])
def test_a_mount_that_lands_on_bfx_through_the_images_link_is_refused_before_start(
    tmp_path, linked_image, reports, what
):
    """The image links /x to /.bfx; the task mounts a file at /x/bfx (or a
    secret whose target is /x/bfx). Docker follows the link when it mounts,
    so the mount lands on bfx. A copy of the pinned bfx there passes a hash;
    where the mount lands does not. Refused before anything starts, by
    name."""
    task = _task(tmp_path, linked_image)
    source = task / "environment" / "f"
    if what == "task-file":
        source.write_text("#!/usr/local/bin/python3\n" + reports.program("task-file"))
        source.chmod(0o755)
    else:
        _pinned_copy(source)
    side: dict = {"image": linked_image, "command": ["sleep", "infinity"]}
    top: dict = {}
    if what == "secret":
        side["secrets"] = [{"source": "f", "target": "/x/bfx"}]
        top["secrets"] = {"f": {"file": "./f"}}
    else:
        side["volumes"] = ["./f:/x/bfx:ro"]
    _compose(task, {"main": {}, "side": side}, **top)
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "lands on /.bfx/bfx" in said, said
    assert not reports.hits, reports.hits
    assert _reward(result) is None


def test_a_volume_another_service_writes_mounted_over_bfx_is_refused(
    tmp_path, linked_image, reports
):
    """A named volume that one service mounts writable, which another mounts
    at /x (the image's link to /.bfx): once the first is released it could
    rewrite what the second runs as its waiter on a later start. Refused
    before anything starts."""
    task = _task(tmp_path, linked_image)
    _compose(
        task,
        {
            "main": {},
            "writer": {
                "image": linked_image,
                "command": ["sleep", "infinity"],
                "volumes": ["shared:/shared"],
            },
            "side": {
                "image": linked_image,
                "command": ["sleep", "infinity"],
                "volumes": ["shared:/x"],
            },
        },
        volumes={"shared": {}},
    )
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "lands on /.bfx" in said, said
    assert not reports.hits


def _hidden_task(tmp_path, image, *, outer: str, side: dict | None = None, **top):
    """The ml-dev-bench shape: the task's folder bound into main at
    /opt/task-src (*outer*: its mode, or a whole volume entry), its solution
    and tests hidden under anonymous volumes; test.sh passes when they are
    hidden."""
    task = _task(tmp_path, image)
    src = task / "environment" / "src"
    for sub in ("solution", "tests"):
        (src / sub).mkdir(parents=True)
        (src / sub / "secret").write_text("x\n")
    main = {
        "volumes": [
            f"./src:/opt/task-src{outer}",
            "/opt/task-src/solution",
            "/opt/task-src/tests",
        ]
    }
    services = {"main": main}
    if side is not None:
        services["side"] = side
    _compose(task, services, **top)
    _script(
        task,
        "test.sh",
        "#!/bin/bash\n[ -d /opt/task-src/solution ] && [ ! -e /opt/task-src/solution/secret ] "
        "&& echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n",
    )
    return task


def test_a_read_only_folder_with_hidden_subfolders_starts(tmp_path, probe_image):
    """The task's folder bound read-only, its solution and tests hidden under
    anonymous volumes inside it (ml-dev-bench's shape): nothing of the
    trial writes the folder, so the mounts inside it are no risk."""
    task = _hidden_task(tmp_path, probe_image, outer=":ro")
    outcome, result = _run(tmp_path, task)
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))


def test_the_same_folder_bound_writable_is_refused(tmp_path, probe_image):
    task = _hidden_task(tmp_path, probe_image, outer="")
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "may not sit inside one another (it is writable in its service)" in said, (
        said
    )


def test_a_read_only_folder_another_service_writes_is_refused(tmp_path, probe_image):
    side = {
        "image": probe_image,
        "command": ["sleep", "infinity"],
        "volumes": ["./src:/data"],
    }
    task = _hidden_task(tmp_path, probe_image, outer=":ro", side=side)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "sandbox startup refused" in said, said
    assert _reward(result) is None


def test_a_read_only_folder_beneath_another_service_s_writable_one_is_refused(
    tmp_path, probe_image
):
    side = {
        "image": probe_image,
        "command": ["sleep", "infinity"],
        "volumes": ["./:/env"],
    }
    task = _hidden_task(tmp_path, probe_image, outer=":ro", side=side)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "sandbox startup refused" in said or "may reach only" in said, said
    assert _reward(result) is None


def test_a_local_volume_bound_to_the_folder_is_refused(tmp_path, probe_image):
    side = {
        "image": probe_image,
        "command": ["sleep", "infinity"],
        "volumes": ["shared:/data"],
    }
    source = tmp_path / "mdoptin-public-nokey" / "environment" / "src"
    bound = {
        "driver": "local",
        "driver_opts": {"type": "none", "o": "bind", "device": str(source)},
    }
    task = _hidden_task(
        tmp_path, probe_image, outer=":ro", side=side, volumes={"shared": bound}
    )
    assert source.is_dir()
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "refused" in said, said
    assert _reward(result) is None


@pytest.fixture(scope="module")
def linked_folder_image(probe_image, tmp_path_factory):
    """The probe image with /opt/task-src/solution a link to /.bfx."""
    tag = _build(
        tmp_path_factory,
        probe_image,
        "RUN mkdir -p /opt/task-src/tests && ln -s /.bfx /opt/task-src/solution\n",
    )
    yield tag
    subprocess.run(["docker", "rmi", "--force", tag], capture_output=True)


def test_a_link_inside_a_read_only_folder_to_bfx_is_refused(
    tmp_path, linked_folder_image
):
    """A read-only volume that nothing writes may hold mounts, but where each
    lands is still read: the image fills the volume with solution -> /.bfx,
    and the mount hiding solution would land on bfx. (A link in the task's
    own folder never gets that far: the task is refused as not runnable.)"""
    task = _task(tmp_path, linked_folder_image)
    main = {
        "volumes": [
            "src:/opt/task-src:ro",
            "/opt/task-src/solution",
            "/opt/task-src/tests",
        ]
    }
    _compose(task, {"main": main}, volumes={"src": {}})
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "lands on /.bfx" in said, said
    assert _reward(result) is None


def test_a_task_volume_inside_one_of_benchflow_s_mounts_is_refused(
    tmp_path, probe_image
):
    """main's /logs/verifier is BenchFlow's; a task volume inside it is the
    task's, and refused before start like any other mount inside another."""
    task = _task(tmp_path, probe_image)
    _compose(
        task,
        {"main": {"volumes": ["extra:/logs/verifier/extra"]}},
        volumes={"extra": {}},
    )
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    said = _refusal_text(outcome) + str(result.get("error") or "")
    assert "may not sit inside one another" in said, said
    assert "/logs/verifier/extra" in said, said


def test_a_mount_under_var_run_on_a_debian_image_starts(tmp_path, probe_image):
    """/var/run is a link to /run on Debian: following it is no refusal."""
    task = _task(tmp_path, probe_image)
    side = {
        "image": probe_image,
        "command": ["sleep", "infinity"],
        "volumes": ["sock:/var/run/app"],
    }
    _compose(task, {"main": {}, "side": side}, volumes={"sock": {}})
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))


@pytest.mark.timeout(7200)
def test_starts_in_a_row_with_every_check_on_never_hang(tmp_path, probe_image):
    """A sidecar with a bind, a volume and a tmpfs, beside main: every start
    reads where each held container's mounts land (main's and the sidecar's
    at once) and their bfx. Trials in a row each start, score and end within
    their bound; none waits on a read it stopped.

    Three by default; ``BENCHFLOW_TEST_HELD_START_RUNS=50`` is the soak (about
    30 minutes on 32 vCPUs). How a stopped read ends is held deterministically
    by tests/test_held_start_bounds.py (a paused pipe, a read that never
    answers, a stalled digest read)."""
    import time as clock

    runs = int(os.environ.get("BENCHFLOW_TEST_HELD_START_RUNS", "3"))
    for n in range(runs):
        root = tmp_path / f"run{n}"
        root.mkdir()
        task = _task(root, probe_image)
        (task / "environment" / "conf").write_text("x\n")
        side = {
            "image": probe_image,
            "command": ["sleep", "infinity"],
            "volumes": ["./conf:/etc/app/conf:ro", "data:/var/lib/app"],
            "tmpfs": ["/tmp"],
        }
        _compose(task, {"main": {}, "app": side}, volumes={"data": {}})
        _script(task, "test.sh", PASS)
        began = clock.monotonic()
        outcome, result = _run(root, task)
        assert _reward(result) == 1.0, (
            n,
            result.get("rewards"),
            _refusal_text(outcome),
        )
        assert clock.monotonic() - began < 300, n


# -------------------------------------------------------- after start-up


def test_a_restart_always_sidecar_that_exits_in_the_turn_stays_down(
    tmp_path, probe_image, reports
):
    """A sidecar with ``restart: always`` exits 25 s in, during the agent's
    turn, and a ``[[verifier.collect]]`` hook runs in it at the end. It
    stays down: the hook does not run, nothing reaches the server, the trial
    is scored as test.sh says and flagged, the service held_exited in the
    agent's turn."""
    task = _task(tmp_path, probe_image)
    (task / "environment" / "hook.py").write_text(reports.program("collect-hook"))
    toml = (
        (task / "task.toml")
        .read_text()
        .replace(
            "[environment]",
            '[[verifier.collect]]\nservice = "side"\ncommand = "python3 /hook/hook.py"\n'
            "timeout_sec = 30\n\n[environment]",
        )
    )
    (task / "task.toml").write_text(toml)
    side = {
        "image": probe_image,
        "command": _exits_after(reports, 25),
        "volumes": ["./hook.py:/hook/hook.py:ro"],
        "restart": "always",
    }
    _compose(task, {"main": {}, "side": side})
    _script(task, "solve.sh", "#!/bin/bash\nsleep 40\n")
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    assert not reports.by("collect-hook"), reports.hits
    assert not reports.reached(), reports.reached()
    assert len(_boots(reports.by("program"))) == 1, reports.hits
    assert _reward(result) == 1.0, (result.get("rewards"), _refusal_text(outcome))
    exited = result.get("held_exited") or []
    assert [(e.get("service"), e.get("phase")) for e in exited] == [
        ("side", "the agent's turn")
    ], exited
    assert not result.get("held_after_restart"), result


SERVER_WITH_EXIT = r"""
import http.server, os, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        if self.path == "/exit":
            os._exit(0)
    def log_message(self, *a):
        pass
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 8080), H).serve_forever()
"""
ASK_EXIT = (
    'python3 -c "import urllib.request; '
    "urllib.request.urlopen('http://side:8080/exit', timeout=5)\" || true\n"
)


@pytest.mark.parametrize(
    "case, solve, test, want",
    [
        ("exits", ASK_EXIT, FAIL, 0.0),
        ("control", "", FAIL, 0.0),
        ("passes", "", PASS, 1.0),
    ],
    ids=["exits", "control", "passes"],
)
def test_test_sh_s_own_service_exiting_in_the_turn_is_a_scored_failure(
    tmp_path, probe_image, case, solve, test, want
):
    """test.sh runs in a compose service (``[verifier].service``), and the
    agent's turn has that service exit through its own API. The trial is a
    0, flagged, the service held_exited in the agent's turn; without the
    exit, 0 or 1 as test.sh says."""
    task = _task(tmp_path, probe_image)
    toml = (
        (task / "task.toml")
        .read_text()
        .replace("[verifier]\n", '[verifier]\nservice = "side"\n')
    )
    (task / "task.toml").write_text(toml)
    _compose(
        task,
        {
            "main": {},
            "side": {
                "image": probe_image,
                "command": ["python3", "-c", SERVER_WITH_EXIT],
            },
        },
    )
    _script(task, "solve.sh", "#!/bin/bash\n" + solve + "sleep 2\n")
    _script(task, "test.sh", test)
    outcome, result = _run(tmp_path, task)
    assert _reward(result) == want, (
        result.get("rewards"),
        result.get("verifier_error"),
        _refusal_text(outcome),
    )
    assert not result.get("verifier_error"), result.get("verifier_error")
    exited = result.get("held_exited") or []
    if case == "exits":
        assert [(e.get("service"), e.get("phase")) for e in exited] == [
            ("side", "the agent's turn")
        ], exited
        assert any(
            "scored 0" in w or "exited" in w for w in result.get("warnings") or []
        )
    else:
        assert not exited, exited


def test_test_sh_s_own_service_exiting_under_test_sh_is_a_scored_failure(
    tmp_path, probe_image
):
    """test.sh, in its compose service, asks the service's program to exit
    after the verifier's rules are in: the container stops under test.sh,
    nothing starts it again, and the trial is a 0, flagged, the service
    held_exited in test.sh."""
    task = _task(tmp_path, probe_image)
    toml = (
        (task / "task.toml")
        .read_text()
        .replace("[verifier]\n", '[verifier]\nservice = "side"\n')
    )
    (task / "task.toml").write_text(toml)
    _compose(
        task,
        {
            "main": {},
            "side": {
                "image": probe_image,
                "command": ["python3", "-c", SERVER_WITH_EXIT],
                "restart": "always",
            },
        },
    )
    _script(
        task,
        "test.sh",
        "#!/bin/bash\n"
        "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/exit', timeout=5)\" || true\n"
        "sleep 5\necho 1 > /logs/verifier/reward.txt\n",
    )
    outcome, result = _run(tmp_path, task)
    assert _reward(result) == 0.0, (
        result.get("rewards"),
        result.get("verifier_error"),
        _refusal_text(outcome),
    )
    assert not result.get("verifier_error"), result.get("verifier_error")
    exited = result.get("held_exited") or []
    assert [(e.get("service"), e.get("phase")) for e in exited] == [
        ("side", "test.sh")
    ], exited


# ------------------------------------------------------------- stopping


@pytest.mark.parametrize("how", ["compose-stop-signal", "image-stopsignal"])
def test_a_stop_never_releases_a_waiting_service(
    tmp_path, probe_image, usr1_image, reports, how
):
    """The service's stop signal is the waiter's release signal (SIGUSR1),
    set by the task's ``stop_signal`` or by the image's STOPSIGNAL; its
    healthcheck never passes, so the start fails while main still waits.
    Every held container stops on SIGTERM and is killed first: nothing
    reaches the server."""
    image = usr1_image if how == "image-stopsignal" else probe_image
    task = _task(tmp_path, image)
    side = {
        "image": image,
        "command": _exits_after(reports, 0.3),
        "healthcheck": {
            "test": ["CMD", "python3", "-c", "raise SystemExit(1)"],
            "interval": "1s",
            "retries": 3,
        },
        "restart": "always",
    }
    if how == "compose-stop-signal":
        side.update({"stop_signal": "SIGUSR1", "stop_grace_period": "20s"})
    _compose(
        task,
        {
            "main": {"depends_on": {"side": {"condition": "service_healthy"}}},
            "side": side,
        },
    )
    _script(task, "test.sh", PASS)
    outcome, result = _run(tmp_path, task)
    time.sleep(5)
    assert not reports.reached(), reports.reached()
    assert _reward(result) is None, _refusal_text(outcome)


def test_held_containers_stop_on_sigterm_and_never_restart(
    tmp_path, usr1_image, monkeypatch
):
    """After the release, every held container (main included) has Docker's
    restart policy ``no`` and SIGTERM as its stop signal, whatever the task
    and the image set (main's restart policy is refused otherwise); a
    teardown that keeps its containers kills them first."""
    from benchflow.sandbox import _inert_bootstrap
    from benchflow.sandbox.docker import DockerSandbox

    seen: dict[str, tuple] = {}
    release = _inert_bootstrap.release

    async def watched(env, compose, **kwargs):
        await release(env, compose, **kwargs)
        for cid in await _inert_bootstrap.compose_container_ids(compose):
            inspected = await env._inspect_container(cid)
            name = inspected["Config"]["Labels"]["com.docker.compose.service"]
            seen[name] = (
                inspected["HostConfig"]["RestartPolicy"]["Name"],
                inspected["Config"].get("StopSignal"),
            )

    stop = DockerSandbox.stop
    projects: list[str] = []

    async def kept(self, delete):
        self._keep_containers = True
        projects.append(self.compose_project_name)
        await stop(self, delete)

    monkeypatch.setattr(_inert_bootstrap, "release", watched)
    monkeypatch.setattr(DockerSandbox, "stop", kept)
    task = _task(tmp_path, usr1_image)
    _compose(
        task,
        {
            "main": {},
            "side": {
                "image": usr1_image,
                "command": ["sleep", "infinity"],
                "restart": "always",
                "stop_signal": "SIGUSR1",
            },
        },
    )
    _script(task, "test.sh", PASS)
    try:
        outcome, result = _run(tmp_path, task)
        assert _reward(result) == 1.0, _refusal_text(outcome)
        assert seen == {"main": ("no", "SIGTERM"), "side": ("no", "SIGTERM")}, seen
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label=com.docker.compose.project={projects[0]}",
                "--format",
                "{{.Names}} {{.Status}}",
            ],
            capture_output=True,
            text=True,
        ).stdout
        # Killed, then stopped: SIGKILL's 137.
        assert listed and all("Exited (137)" in line for line in listed.splitlines()), (
            listed
        )
    finally:
        for project in projects:
            ids = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ],
                capture_output=True,
                text=True,
            ).stdout.split()
            if ids:
                subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
