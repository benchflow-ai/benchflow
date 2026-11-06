"""The host trust state (TRUSTED-RUNTIME v2 §3 I2, §5)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.trust import state as trust
from benchflow.trust.state import (
    Binding,
    IngressKind,
    LineageMissingError,
    TrustState,
    TrustStateError,
)


@pytest.fixture
def private_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "private"
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(root))
    return root


def _sandbox(tmp_path: Path, name: str = "s1") -> SimpleNamespace:
    rollout = tmp_path / f"trial-{name}"
    rollout.mkdir()
    return SimpleNamespace(
        session_id=name,
        rollout_paths=SimpleNamespace(rollout_dir=rollout),
        environment_dir=tmp_path / f"env-{name}",
    )


BINDING = Binding(image_digest="sha256:aa", setup="s", credentials="c", bfx_version="b")


def test_fresh_generation_starts_pristine(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING, services=["db"])
    book = trust.ledger(box)
    assert {n: s.state for n, s in book.services.items()} == {
        "main": TrustState.PRISTINE,
        "db": TrustState.PRISTINE,
    }
    trust.require_pristine(box, "main")


def test_admit_marks_tainted_and_writes_ledger_before_returning(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    record = json.loads(trust.ledger(box).path.read_text())
    assert record["services"]["main"]["state"] == "tainted"
    assert record["services"]["main"]["history"][-1]["kind"] == "agent-launch"


def test_scan_refused_outside_pristine(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")
    with pytest.raises(TrustStateError):
        trust.record_manifest(box, "main", {"x": 1}, checklist_passed=True)


def test_scan_refused_before_any_start(private_root, tmp_path):
    box = _sandbox(tmp_path)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")


def test_admit_before_start_leaves_sandbox_tainted(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.admit(box, "main", IngressKind.SDK_EXEC)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")


def test_manifest_then_taint_keeps_manifest_and_seals(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.record_manifest(box, "main", {"facts": 1}, checklist_passed=True)
    assert trust.ledger(box).services["main"].state is TrustState.MANIFEST
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    assert trust.ledger(box).services["main"].state is TrustState.TAINTED
    assert trust.sealed(box, "main")


@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_checklist_value_other_than_true_is_not_sealed(private_root, tmp_path, value):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.record_manifest(box, "main", {}, checklist_passed=value)
    assert not trust.sealed(box, "main")


def test_unscanned_service_is_not_sealed(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING, services=["db"])
    trust.record_manifest(box, "main", {}, checklist_passed=True)
    assert not trust.sealed(box, "db")
    assert not trust.sealed(_sandbox(tmp_path, "other"), "main")


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {},
        {"services": {}},
        {"services": {"main": {}}},
        {"services": {"main": {"checklist_passed": None}}},
        {"services": {"main": {"checklist_passed": "yes"}}},
        {"services": {"other": {"checklist_passed": True}}},
    ],
)
def test_manifest_checklist_input_fails_closed(manifest):
    assert trust.checklist_passed(manifest, "main") is False


def test_manifest_checklist_input_true():
    manifest = {"services": {"main": {"checklist_passed": True}}}
    assert trust.checklist_passed(manifest, "main") is True


@pytest.mark.parametrize(
    "field", ["image_digest", "setup", "credentials", "bfx_version"]
)
def test_binding_change_unseals(private_root, tmp_path, field):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.record_manifest(box, "main", {}, checklist_passed=True)
    book = trust.ledger(box)
    book.binding = Binding(**{**BINDING.to_record(), "lineage": (), field: "other"})
    assert not trust.sealed(box, "main")


def test_shared_mount_taints_every_service(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING, services=["db", "web"])
    trust.admit(box, "db", IngressKind.SHARED_MOUNT)
    states = {n: s.state for n, s in trust.ledger(box).services.items()}
    assert states == dict.fromkeys(("main", "db", "web"), TrustState.TAINTED)


def test_ingress_without_backend_service_map_reaches_every_service(
    private_root, tmp_path
):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING, services=["db"])
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    assert trust.ledger(box).services["db"].state is TrustState.TAINTED


def test_backend_named_shared_services_are_tainted(private_root, tmp_path):
    class Box(SimpleNamespace):
        def trust_shared_services(self, service):
            return ["db"] if service == "main" else []

    box = Box(**vars(_sandbox(tmp_path)))
    trust.begin_fresh(box, BINDING, services=["db", "web"])
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    states = {n: s.state for n, s in trust.ledger(box).services.items()}
    assert states["db"] is TrustState.TAINTED
    assert states["web"] is TrustState.PRISTINE


def test_service_first_seen_after_main_taint_is_tainted(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "late")


def test_restore_inherits_manifest_and_is_tainted(private_root, tmp_path):
    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    trust.record_manifest(parent, "main", {"m": 1}, checklist_passed=True)
    lineage = trust.lineage_record(parent, "img-1")
    child = _sandbox(tmp_path, "child")
    trust.begin_restored(child, lineage, image_ref="img-1")
    book = trust.ledger(child)
    assert book.services["main"].state is TrustState.TAINTED
    assert book.binding.lineage[-1] == lineage
    assert trust.sealed(child, "main")
    with pytest.raises(TrustStateError):
        trust.require_pristine(child, "main")


def test_frozen_lineage_ignores_later_parent_ingress(private_root, tmp_path):
    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    lineage = trust.lineage_record(parent, "img-1")
    trust.admit(parent, "main", IngressKind.AGENT_LAUNCH)
    record = trust.load_record(lineage)
    assert record is not None
    assert record["services"]["main"]["state"] == "pristine"


@pytest.mark.parametrize("lineage", [None, "", "no-such-record", "../escape"])
def test_restore_without_lineage_is_tainted_unsealed_and_refuses_scan(
    private_root, tmp_path, lineage
):
    child = _sandbox(tmp_path, "child")
    trust.begin_fresh(child, BINDING)
    trust.record_manifest(child, "main", {}, checklist_passed=True)
    trust.begin_restored(child, lineage)
    book = trust.ledger(child)
    assert book.lineage_missing
    assert book.services["main"].state is TrustState.TAINTED
    assert not trust.sealed(child, "main")
    with pytest.raises(TrustStateError):
        trust.require_pristine(child, "main")
    with pytest.raises(LineageMissingError):
        trust.require_sealed(child, "main")


@pytest.mark.parametrize("lineage", [None, "no-such-record"])
def test_strict_restore_without_lineage_refuses(private_root, tmp_path, lineage):
    child = _sandbox(tmp_path, "child")
    with pytest.raises(LineageMissingError):
        trust.begin_restored(child, lineage, strict=True)


def test_lineage_bound_to_another_image_is_no_lineage(private_root, tmp_path):
    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    trust.record_manifest(parent, "main", {}, checklist_passed=True)
    lineage = trust.lineage_record(parent, "img-1")
    child = _sandbox(tmp_path, "child")
    trust.begin_restored(child, lineage, image_ref="img-2")
    assert trust.ledger(child).lineage_missing
    assert not trust.sealed(child, "main")


def test_missing_lineage_passes_to_descendants(private_root, tmp_path):
    first = _sandbox(tmp_path, "first")
    trust.begin_restored(first, None)
    lineage = trust.lineage_record(first, "img-1")
    second = _sandbox(tmp_path, "second")
    trust.begin_restored(second, lineage, image_ref="img-1")
    with pytest.raises(LineageMissingError):
        trust.require_sealed(second, "main")


def test_snapshot_meta_round_trip(private_root, tmp_path):
    from benchflow.sandbox.protocol import SandboxImage

    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    trust.record_manifest(parent, "main", {}, checklist_passed=True)
    image = trust.stamp_snapshot(parent, SandboxImage(provider="docker", ref="img-1"))
    record = SandboxImage.from_record(image.to_record())
    child = _sandbox(tmp_path, "child")
    trust.restore_from(child, record)
    trust.require_sealed(child, "main")


def test_require_sealed_without_manifest_raises(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    with pytest.raises(trust.NotSealedError):
        trust.require_sealed(box, "main")


def test_authoritative_ledger_is_private_and_report_is_a_copy(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    book = trust.ledger(box)
    assert book.path.parent == Path(os.path.realpath(private_root))
    assert stat.S_IMODE(os.stat(private_root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(book.path).st_mode) == 0o600
    report = box.rollout_paths.rollout_dir / trust.REPORT_NAME
    assert json.loads(report.read_text()) == json.loads(book.path.read_text())


def test_report_copy_is_never_read_back(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    report = box.rollout_paths.rollout_dir / trust.REPORT_NAME
    record = json.loads(report.read_text())
    record["services"]["main"]["state"] = "pristine"
    report.write_text(json.dumps(record))
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")


@pytest.mark.parametrize("where", ["rollout", "environment"])
def test_private_state_a_sandbox_can_reach_moves_to_a_private_temp_folder(
    tmp_path, monkeypatch, where
):
    box = _sandbox(tmp_path)
    box.environment_dir.mkdir()
    reachable = (
        box.rollout_paths.rollout_dir if where == "rollout" else box.environment_dir
    )
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(reachable / "state"))
    monkeypatch.setattr(trust.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    book = trust.ledger(box)
    real = Path(os.path.realpath(reachable))
    assert real not in book.root.parents and book.root != real
    assert stat.S_IMODE(os.stat(book.root).st_mode) == 0o700


def test_private_state_refuses_when_every_choice_is_reachable(tmp_path, monkeypatch):
    box = _sandbox(tmp_path)
    reachable = box.rollout_paths.rollout_dir
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(reachable / "state"))
    monkeypatch.setattr(trust.tempfile, "tempdir", str(reachable))
    with pytest.raises(TrustStateError):
        trust.ledger(box)


def test_an_object_without_attributes_keeps_a_ledger(private_root):
    sandbox = "a sandbox handle"
    trust.admit(sandbox, "main", IngressKind.SESSION_FACTORY)
    with pytest.raises(TrustStateError):
        trust.require_pristine(sandbox, "main")
    assert not trust.sealed(sandbox, "main")


def test_stop_and_start_of_same_container_keeps_taint(private_root, tmp_path):
    """Only begin_fresh opens a pristine generation; nothing else resets it."""
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    generation = trust.ledger(box).generation
    trust.admit(box, "main", IngressKind.SDK_EXEC)
    assert trust.ledger(box).generation == generation
    assert trust.ledger(box).services["main"].state is TrustState.TAINTED


def test_repeated_ingress_counts_without_rewriting(private_root, tmp_path):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.SDK_EXEC)
    path = trust.ledger(box).path
    before = path.stat().st_mtime_ns
    for _ in range(50):
        trust.admit(box, "main", IngressKind.SDK_EXEC)
    assert path.stat().st_mtime_ns == before
    assert trust.ledger(box).services["main"].admitted["sdk-exec"] == 51
    assert len(trust.ledger(box).services["main"].history) == 2


def test_mock_sandbox_is_tainted(private_root):
    from unittest.mock import MagicMock

    box = MagicMock()
    trust.admit(box, "main", IngressKind.SDK_EXEC)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")


# -- the ledger stays outside every folder the sandbox can reach (m1 A M1) --


def _bound(tmp_path: Path, source: Path, name: str = "s1") -> SimpleNamespace:
    box = _sandbox(tmp_path, name)
    box._mounts_json = [{"type": "bind", "source": str(source), "target": "/data"}]
    return box


@pytest.mark.parametrize("which", ["state folder", "ancestor"])
def test_ledger_is_outside_operator_bind_mounts(tmp_path, monkeypatch, which):
    data = tmp_path / "operator-data"
    state = data / "state"
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(state))
    monkeypatch.setattr(trust.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    state.mkdir(parents=True)
    box = _bound(tmp_path, state if which == "state folder" else data)
    trust.begin_fresh(box, BINDING)
    book = trust.ledger(box)
    real = Path(os.path.realpath(data))
    assert real not in book.path.parents
    assert not any(state.iterdir())


class _BackendWithBinds(SimpleNamespace):
    def trust_host_mount_sources(self) -> list[str]:
        return list(self.binds)


def test_ledger_moves_out_of_binds_the_backend_adds_after_it_was_created(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(state))
    monkeypatch.setattr(trust.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    plain = _sandbox(tmp_path)
    box = _BackendWithBinds(**vars(plain), binds=[])
    trust.begin_fresh(box, BINDING)
    first = trust.ledger(box).path
    assert first.parent == Path(os.path.realpath(state))
    # The frozen launch now binds a parent of the state folder.
    box.binds = [str(tmp_path)]
    monkeypatch.setattr(trust.tempfile, "tempdir", "/nonexistent-tmp-root")
    with pytest.raises((TrustStateError, OSError)):
        trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    monkeypatch.setattr(trust.tempfile, "tempdir", str(tmp_path / "tmp"))
    box.binds = [str(state)]
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    book = trust.ledger(box)
    assert Path(os.path.realpath(state)) not in book.path.parents
    assert json.loads(book.path.read_text())["services"]["main"]["state"] == "tainted"


def test_a_root_the_caller_chose_refuses_a_write_once_reachable(tmp_path):
    root = tmp_path / "chosen"
    root.mkdir(mode=0o700)
    plain = _sandbox(tmp_path)
    box = _BackendWithBinds(**vars(plain), binds=[])
    book = trust.TrustLedger(box, root=root)
    book.begin_fresh(BINDING, ())
    box.binds = [str(root)]
    with pytest.raises(TrustStateError):
        book.admit(["main"], IngressKind.AGENT_LAUNCH)
    assert book.path.parent == root


def test_unlistable_backend_binds_refuse_the_ledger(tmp_path, private_root):
    class Broken(SimpleNamespace):
        def trust_host_mount_sources(self) -> list[str]:
            raise OSError("frozen launch unreadable")

    box = Broken(**vars(_sandbox(tmp_path)))
    with pytest.raises(TrustStateError):
        trust.begin_fresh(box, BINDING)


def test_docker_lists_operator_and_frozen_launch_binds(tmp_path):
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.compose_env import ComposeLaunch

    frozen = tmp_path / "launch.json"
    frozen.write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": "/srv/task-data",
                                "target": "/d",
                            },
                            {"type": "volume", "source": "named", "target": "/v"},
                        ]
                    }
                },
                "volumes": {
                    "dev": {
                        "driver": "local",
                        "driver_opts": {
                            "type": "none",
                            "o": "bind",
                            "device": "/srv/dev",
                        },
                    },
                    "plain": {},
                },
            }
        )
    )
    launch = ComposeLaunch(
        path=frozen,
        project_dir=tmp_path,
        project_name="p",
        env={},
        docker_args=(),
        files=(),
        key="k",
    )
    fake = SimpleNamespace(
        _mounts_json=[
            {"type": "bind", "source": "/home/user/data", "target": "/data"},
            {"type": "volume", "source": "vol", "target": "/vol"},
        ],
        _launch=launch,
    )
    assert DockerSandbox.trust_host_mount_sources(fake) == [
        "/home/user/data",
        "/srv/task-data",
        "/srv/dev",
    ]


# -- a restore reaches its frozen parent lineage (m1 A M2) --


def test_restore_finds_parent_lineage_when_private_state_is_relocated(
    tmp_path, monkeypatch
):
    from benchflow.sandbox.protocol import SandboxImage

    shared_env = tmp_path / "env-shared"
    shared_env.mkdir()
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(shared_env / "state"))
    monkeypatch.setattr(trust.tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    parent = _sandbox(tmp_path, "parent")
    parent.environment_dir = shared_env
    trust.begin_fresh(parent, BINDING)
    trust.record_manifest(parent, "main", {"m": 1}, checklist_passed=True)
    image = trust.stamp_snapshot(parent, SandboxImage(provider="docker", ref="img-1"))
    child = _sandbox(tmp_path, "child")
    child.environment_dir = shared_env
    trust.restore_from(child, SandboxImage.from_record(image.to_record()), strict=True)
    book = trust.ledger(child)
    assert not book.lineage_missing
    assert book.services["main"].manifest_sha256 is not None
    trust.require_sealed(child, "main")


def test_restore_finds_parent_lineage_in_the_folder_the_snapshot_named(
    tmp_path, monkeypatch
):
    from benchflow.sandbox.protocol import SandboxImage

    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(tmp_path / "first"))
    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    trust.record_manifest(parent, "main", {"m": 1}, checklist_passed=True)
    image = trust.stamp_snapshot(parent, SandboxImage(provider="docker", ref="img-1"))
    # A later resume whose private state is configured elsewhere.
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(tmp_path / "second"))
    child = _sandbox(tmp_path, "child")
    trust.restore_from(child, image, strict=True)
    trust.require_sealed(child, "main")


def test_restore_refuses_lineage_bytes_other_than_the_snapshot_recorded(
    private_root, tmp_path
):
    from benchflow.sandbox.protocol import SandboxImage

    parent = _sandbox(tmp_path, "parent")
    trust.begin_fresh(parent, BINDING)
    image = trust.stamp_snapshot(parent, SandboxImage(provider="docker", ref="img-1"))
    path = Path(image.meta[trust.LINEAGE_ROOT_META_KEY]) / (
        image.meta[trust.LINEAGE_META_KEY] + ".json"
    )
    record = json.loads(path.read_text())
    record["services"]["main"]["checklist_passed"] = True
    path.write_text(json.dumps(record))
    child = _sandbox(tmp_path, "child")
    trust.restore_from(child, image)
    assert trust.ledger(child).lineage_missing
    with pytest.raises(LineageMissingError):
        trust.restore_from(_sandbox(tmp_path, "strict"), image, strict=True)


# -- a failed authoritative write keeps admission blocked (m1 A M3) --


def test_failed_ledger_write_keeps_admission_blocked_until_a_write_succeeds(
    private_root, tmp_path, monkeypatch
):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    book = trust.ledger(box)
    real_write = trust._atomic_write

    def failing(path, data, mode):
        if path == book.path:
            raise OSError("disk full")
        real_write(path, data, mode)

    monkeypatch.setattr(trust, "_atomic_write", failing)
    for _ in range(2):
        with pytest.raises(OSError):
            trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
        on_disk = json.loads(book.path.read_text())["services"]["main"]["state"]
        assert on_disk == "pristine"
        # Memory holds what the disk holds, and nothing is admitted on it.
        assert book.services["main"].state is TrustState.PRISTINE
        with pytest.raises(TrustStateError):
            trust.require_pristine(box, "main")
        assert not trust.sealed(box, "main")
    monkeypatch.setattr(trust, "_atomic_write", real_write)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    assert json.loads(book.path.read_text())["services"]["main"]["state"] == "tainted"
    assert book.services["main"].state is TrustState.TAINTED


def test_failed_ledger_write_after_taint_still_blocks_admission(
    private_root, tmp_path, monkeypatch
):
    box = _sandbox(tmp_path)
    trust.begin_fresh(box, BINDING)
    trust.admit(box, "main", IngressKind.AGENT_LAUNCH)
    book = trust.ledger(box)

    def failing(path, data, mode):
        raise OSError("disk full")

    real_write = trust._atomic_write
    monkeypatch.setattr(trust, "_atomic_write", failing)
    with pytest.raises(OSError):
        trust.begin_fresh(box, BINDING)
    # The failed fresh generation is not kept in memory.
    assert book.services["main"].state is TrustState.TAINTED
    with pytest.raises(OSError):
        trust.admit(box, "main", IngressKind.SDK_EXEC)
    monkeypatch.setattr(trust, "_atomic_write", real_write)
    trust.admit(box, "main", IngressKind.SDK_EXEC)
    with pytest.raises(TrustStateError):
        trust.require_pristine(box, "main")
