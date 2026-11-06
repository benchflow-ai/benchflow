"""BenchFlow's Terminal-Bench 2 list of kept services (K2).

The list is the Rust ``bench``'s (round P1's ``tb2-services.txt``: 10 tasks
of Terminal-Bench 2 at 69671fba, each with its split-layout package digest).
It applies only when a task folder's content digest matches: an edited task,
or another task with the same name, gets no key. The fixture is
Terminal-Bench 2's ``pypi-server`` byte for byte (its ``test_outputs.py`` is
stored as ``test_outputs.py.fixture`` so the suite neither collects nor lints
it, and copied back under its name before each test).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task import Task


def _ks():
    """The module under test, imported in each test, so that each test fails on
    its own where the module is missing (the base)."""
    from benchflow.task import keep_services

    return keep_services


FIXTURE = Path(__file__).resolve().parent / "fixtures/keep_services/tb2-pypi-server"
PYPI_DIGEST = "sha256:0644861427264cd213466f25e4f5be985e53124dff843f42a0ecc1261830f7a6"

# Rust bench's list (crates/benchflow-task/src/services.rs at 518009b0, from
# round P1's results/P1.1/tb2-services.txt), written out.
RUST_LIST = {
    "hf-model-inference": (
        "sha256:c3bddc31278033ec47e718cb0174c9733236a57f468ccd542eb0f20409aededb",
        [(5000, "sentiment API")],
    ),
    "pypi-server": (PYPI_DIGEST, [(8080, "package index")]),
    "kv-store-grpc": (
        "sha256:97b699f1e6c85c5644731cade9888f5fc5a164a7deb7799ab4bcfdb1d2914d49",
        [(5328, "gRPC server")],
    ),
    "nginx-request-logging": (
        "sha256:03151a6d3b73236d55736fc7c110feea03d716affd8ca344abf3355ff1ed7f61",
        [(8080, "nginx")],
    ),
    "git-multibranch": (
        "sha256:7af260b6f0f84d57d8160aecd020d7c7b655e1bb4786ab180f2bf99a5d90d1e3",
        [(22, "ssh"), (8443, "https")],
    ),
    "configure-git-webserver": (
        "sha256:367c8c1983aabd745ac1bfd54b72877dcc1d3136f0f81603c6a6e8713f2580ab",
        [(22, "ssh"), (8080, "web server")],
    ),
    "qemu-alpine-ssh": (
        "sha256:dd8c3eae85efa4d91e49947555f691b6ba0c19a3ce687efe6e8e57525ce9e431",
        [(2222, "ssh into the VM")],
    ),
    "qemu-startup": (
        "sha256:8ca9793ec30d3b26ac351f5b9ac0a2631003c365c60be9fcafc0451256a072a7",
        [(6665, "VM console")],
    ),
    "install-windows-3.11": (
        "sha256:8c94fd5fb4cc3c0a44ee7fea04ce2b295319a6c7ea9c071ec4fb08d456c19f7f",
        [(5901, "VNC"), (80, "nginx")],
    ),
    "mailman": (
        "sha256:217914a8f3ce49066e39dd2898e9f161e8136a198503f23bb4df7c5f388ec9de",
        [(25, "smtp")],
    ),
}


def _copy(tmp_path: Path, name: str = "pypi-server") -> Path:
    dest = tmp_path / name
    shutil.copytree(FIXTURE, dest)
    stored = dest / "tests/test_outputs.py.fixture"
    stored.rename(dest / "tests/test_outputs.py")
    return dest


def test_the_list_is_the_rust_list():
    got = {
        t.task: (t.digest, [(s.port, s.name) for s in t.services])
        for t in _ks().tb2_list()
    }
    assert got == RUST_LIST


def test_a_published_task_gets_the_lists_ports_and_its_source(tmp_path):
    task = _copy(tmp_path)
    assert _ks().task_digest(task) == PYPI_DIGEST
    kept = _ks().for_task(Task(task))
    assert kept is not None
    assert kept.services == (_ks().KeepService(8080, "package index"),)
    assert kept.source == _ks().SOURCE_TB2
    assert kept.listed_task == "pypi-server"
    assert kept.describe_source() == "BenchFlow's Terminal-Bench 2 list (pypi-server)"


def test_only_the_digest_decides(tmp_path):
    # Another folder name, same bytes: the list applies.
    assert _ks().for_task(Task(_copy(tmp_path, "renamed-copy"))) is not None
    # Python caches and editor files are not published: they change nothing.
    task = _copy(tmp_path / "a")
    (task / "tests/__pycache__").mkdir()
    (task / "tests/__pycache__/x.cpython-312.pyc").write_bytes(b"junk")
    (task / "solution/solve.sh~").write_text("backup")
    assert _ks().task_digest(task) == PYPI_DIGEST
    # A file outside the published folders changes nothing either.
    (task / "notes.txt").write_text("local notes")
    assert _ks().task_digest(task) == PYPI_DIGEST
    # The name alone is not a digest.
    assert _ks().listed("pypi-server") is None


def test_an_edited_task_gets_no_key(tmp_path):
    for i, rel in enumerate(
        ("instruction.md", "tests/test_outputs.py", "environment/Dockerfile")
    ):
        task = _copy(tmp_path / str(i))
        with open(task / rel, "a") as f:
            f.write("\n")
        assert _ks().task_digest(task) != PYPI_DIGEST, rel
        assert _ks().for_task(Task(task)) is None, rel
    task = _copy(tmp_path / "gitignore")
    (task / ".gitignore").write_text("*.log\n")
    assert _ks().for_task(Task(task)) is None


def test_a_task_of_its_own_key_wins_over_the_list(tmp_path):
    task = _copy(tmp_path)
    toml = (task / "task.toml").read_text()
    (task / "task.toml").write_text(
        toml.replace(
            "[metadata]\n", "[metadata]\nx-keep-services = [{ port = 9000 }]\n", 1
        )
    )
    kept = _ks().for_task(Task(task))
    assert kept is not None and kept.ports == (9000,) and kept.source == "the task"


def test_bench_tasks_check_names_the_list_as_the_source(tmp_path):
    task = _copy(tmp_path)
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    flat = " ".join(result.output.split())
    assert (
        "keeps the agent's services on port 8080 (package index) for grading "
        "(x-keep-services, from BenchFlow's Terminal-Bench 2 list (pypi-server))"
    ) in flat, result.output
    edited = _copy(tmp_path / "edited")
    (edited / "instruction.md").write_text("Something else.\n")
    result = CliRunner().invoke(app, ["tasks", "check", str(edited)])
    assert "keeps the agent's services" not in result.output


def test_the_trial_records_where_the_names_came_from(tmp_path):
    """The kill's record carries the source: the list, with the task it matched."""
    from benchflow._utils import kill_record
    from benchflow.sandbox import lockdown

    task = Task(_copy(tmp_path))
    kept = _ks().for_task(task)
    assert kept is not None
    record: dict = {}
    outcome = lockdown.KeepingKill(
        killed=2,
        kept=(
            lockdown.KeptProcess(
                port=8080, pid=77, command="pypi-server run " + "x" * 300
            ),
        ),
        waited_sec=0.0,
    )
    with kill_record.recording_into(record):
        lockdown._record_kept(kept, outcome)
    assert record["kept_services"] == [
        {
            "port": 8080,
            "pid": 77,
            "command": ("pypi-server run " + "x" * 300)[:200],
            "source": "BenchFlow's Terminal-Bench 2 list (pypi-server)",
        }
    ]
    assert (
        record["kept_services_source"]
        == "BenchFlow's Terminal-Bench 2 list (pypi-server)"
    )
