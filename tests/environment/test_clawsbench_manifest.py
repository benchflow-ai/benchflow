"""The ClawsBench manifest parses and declares the expected environment."""

from pathlib import Path

from benchflow.environment.manifest import load_manifest

MANIFEST_PATH = Path("benchmarks/clawsbench/environment.toml")


def test_clawsbench_manifest_loads():
    m = load_manifest(MANIFEST_PATH)
    assert m.name == "clawsbench"
    assert m.base_image == "kywch/smolclaws-base:latest"
    assert m.image is None
    assert m.isolation == "per_task"


def test_clawsbench_manifest_is_framework_started():
    m = load_manifest(MANIFEST_PATH)
    assert m.owns_lifecycle is False
    assert m.services, "framework-started env must declare services"


def test_clawsbench_manifest_declares_the_five_claw_services():
    m = load_manifest(MANIFEST_PATH)
    names = {s.name for s in m.services}
    assert names == {"gmail", "slack", "gcal", "gdoc", "gdrive"}
    assert m.all_ports == [9001, 9002, 9003, 9004, 9005]


def test_clawsbench_manifest_uses_image_task_selection():
    m = load_manifest(MANIFEST_PATH)
    assert m.task_selection.mechanism == "image"


def test_clawsbench_manifest_declares_sqlite_state_scoped_to_gmail():
    """v0.5 Phase 2: ClawsBench is now stateful so snapshot/restore has real
    coverage — scoped to /data/gmail.db only (the sole DB the one shipping task
    seeds; declaring an absent path would make `sqlite3 .backup` fabricate an
    empty DB). Guards the manifest against silently re-widening to unseeded DBs.
    """
    m = load_manifest(MANIFEST_PATH)
    assert m.state is not None, "ClawsBench must declare [environment.state]"
    assert m.state.kind == "sqlite"
    assert m.state.paths == ["/data/gmail.db"]


def test_clawsbench_manifest_service_commands_are_runnable():
    m = load_manifest(MANIFEST_PATH)
    for svc in m.services:
        assert svc.command.startswith(f"claw-{svc.name}")
        assert "serve" in svc.command
        assert f"--port {svc.port}" in svc.command
