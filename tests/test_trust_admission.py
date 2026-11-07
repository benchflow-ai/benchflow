"""Every transfer primitive requires the Admission only ``trust.admit()``
makes (TRUSTED-RUNTIME v2 §5; m1 review A M4)."""

from __future__ import annotations

import asyncio
import functools
import importlib
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchflow.sandbox
from benchflow.sandbox._base import BaseSandbox
from benchflow.trust import state as trust
from benchflow.trust.state import (
    TRUSTED_TRANSFER,
    Admission,
    AdmissionRequired,
    Binding,
    IngressKind,
    admit,
    guard_transfer,
)


@pytest.fixture(autouse=True)
def private_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "private"
    monkeypatch.setenv("BENCHFLOW_PRIVATE_STATE", str(root))
    return root


class Box:
    """A backend whose primitives are guarded as BaseSandbox guards them."""

    def __init__(self, tmp_path: Path, name: str = "box") -> None:
        rollout = tmp_path / f"trial-{name}"
        rollout.mkdir()
        self.session_id = name
        self.rollout_paths = SimpleNamespace(rollout_dir=rollout)
        self.environment_dir = tmp_path / f"env-{name}"
        self.moved: list[tuple[str, str]] = []

    async def _upload_file(self, src, dst, *, mode=None):
        self.moved.append(("file", dst))

    async def _upload_dir(self, src, dst, service="main"):
        # A backend's primitive that calls its own primitive: one transfer.
        self.moved.append(("dir", dst))
        await self.upload_file(src, f"{dst}/x")

    async def _live_process(self, *, agent=None):
        self.moved.append(("live", agent or ""))

    upload_file = guard_transfer("upload_file", _upload_file)
    upload_dir = guard_transfer("upload_dir", _upload_dir)
    live_process = guard_transfer("live_process", _live_process)


def _run(coro):
    return asyncio.run(coro)


def _started(tmp_path: Path, name: str = "box", services=()) -> Box:
    box = Box(tmp_path, name)
    trust.begin_fresh(box, Binding(), services=services)
    return box


def _sandbox_classes() -> list[type]:
    for info in pkgutil.iter_modules(benchflow.sandbox.__path__):
        try:
            importlib.import_module(f"benchflow.sandbox.{info.name}")
        except ImportError:
            continue
    found: list[type] = []
    todo = list(BaseSandbox.__subclasses__())
    while todo:
        cls = todo.pop()
        found.append(cls)
        todo.extend(cls.__subclasses__())
    return [cls for cls in found if cls.__module__.startswith("benchflow.")]


def test_every_backend_primitive_requires_an_admission():
    classes = _sandbox_classes()
    assert {c.__name__ for c in classes} >= {
        "DockerSandbox",
        "RemoteDockerSandbox",
        "DaytonaSandbox",
        "ModalSandbox",
        "AgentCoreSandbox",
        "AppleContainerSandbox",
    }
    unguarded = []
    for cls in classes:
        for name in trust.TRANSFER_PRIMITIVES:
            method = getattr(cls, name)
            if not getattr(method, "__benchflow_guarded__", False):
                unguarded.append(f"{cls.__module__}.{cls.__name__}.{name}")
    assert not unguarded


@pytest.mark.parametrize("primitive", ["upload_file", "upload_dir", "live_process"])
def test_a_backend_primitive_without_an_admission_moves_nothing(tmp_path, primitive):
    from benchflow.sandbox.docker import DockerSandbox

    box = _started(tmp_path)
    method = getattr(DockerSandbox, primitive)
    args = () if primitive == "live_process" else (tmp_path, "/x")
    for admission in (None, "token", object()):
        with pytest.raises(AdmissionRequired):
            _run(method(box, *args, admission=admission))
    with pytest.raises(AdmissionRequired):
        _run(method(box, *args))


def test_only_admit_makes_an_admission(tmp_path):
    box = _started(tmp_path)
    with pytest.raises(TypeError):
        Admission(
            object(), trust.ledger(box), 1, frozenset({"main"}), TRUSTED_TRANSFER, "x"
        )
    forged = object.__new__(Admission)
    with pytest.raises(AdmissionRequired):
        _run(box.upload_file(tmp_path, "/x", admission=forged))
    assert box.moved == []


def test_an_admitted_transfer_runs(tmp_path):
    box = _started(tmp_path)
    admission = admit(box, "main", IngressKind.SDK_UPLOAD)
    _run(box.upload_file(tmp_path, "/x", admission=admission))
    _run(box.upload_dir(tmp_path, "/d", admission=admission))
    _run(box.live_process(agent="a", admission=admission))
    assert box.moved == [("file", "/x"), ("dir", "/d"), ("file", "/d/x"), ("live", "a")]
    assert trust.ledger(box).services["main"].state is trust.TrustState.TAINTED


def test_a_trusted_transfer_runs_without_taint_and_needs_a_reason(tmp_path):
    box = _started(tmp_path)
    with pytest.raises(TypeError):
        admit(box, "main", TRUSTED_TRANSFER)
    admission = admit(box, "main", TRUSTED_TRANSFER, reason="the task's tests")
    _run(box.upload_file(tmp_path, "/tests/x", admission=admission))
    assert trust.ledger(box).services["main"].state is trust.TrustState.PRISTINE


def test_an_admission_is_for_one_sandbox(tmp_path):
    first = _started(tmp_path, "first")
    second = _started(tmp_path, "second")
    admission = admit(first, "main", IngressKind.SDK_UPLOAD)
    with pytest.raises(AdmissionRequired):
        _run(second.upload_file(tmp_path, "/x", admission=admission))
    assert second.moved == []


def test_an_admission_is_for_one_generation(tmp_path):
    box = _started(tmp_path)
    admission = admit(box, "main", IngressKind.SDK_UPLOAD)
    trust.begin_fresh(box, Binding())
    with pytest.raises(AdmissionRequired):
        _run(box.upload_file(tmp_path, "/x", admission=admission))
    assert box.moved == []


def test_an_admission_reaches_only_its_services(tmp_path):
    box = _started(tmp_path)
    trusted = admit(box, "main", TRUSTED_TRANSFER, reason="the task's tests")
    with pytest.raises(AdmissionRequired):
        _run(box.upload_dir(tmp_path, "/tests", service="db", admission=trusted))
    with pytest.raises(AdmissionRequired):
        _run(box.upload_dir(tmp_path, "/tests", "db", admission=trusted))
    assert box.moved == []


def test_conditional_partial_and_aliased_transfers_without_an_admission_refuse(
    tmp_path,
):
    box = _started(tmp_path)
    admission = None
    if False:  # an admission on one path only
        admission = admit(box, "main", IngressKind.SDK_UPLOAD)
    with pytest.raises(AdmissionRequired):
        _run(box.upload_file(tmp_path, "/x", admission=admission))
    with pytest.raises(AdmissionRequired):
        _run(functools.partial(box.upload_file, tmp_path, "/x")())
    launch = box.live_process
    with pytest.raises(AdmissionRequired):
        _run(launch())
    assert box.moved == []


def test_a_session_factory_agents_transfers_are_admitted(tmp_path):
    """The in-process agent gets the sandbox through SessionFactorySandbox,
    which passes other methods through: its transfers are admitted as its
    ingress, so they run and the sandbox is TAINTED."""
    from benchflow.rollout.session_factory_runtime import SessionFactorySandbox

    box = _started(tmp_path, services=("db",))
    handle = SessionFactorySandbox(box, {}, None, None)
    _run(handle.upload_file(tmp_path, "/x"))
    _run(handle.upload_dir(tmp_path, "/d", service="db"))
    _run(handle.live_process(agent="a"))
    assert ("file", "/x") in box.moved and ("live", "a") in box.moved
    assert trust.ledger(box).services["main"].admitted.get("session-factory") == 3
    assert trust.ledger(box).services["db"].state is trust.TrustState.TAINTED


# --- skills from a learner store (TR-P2 r3, review M3) -------------------------


class _SkillsBox(Box):
    """A Box whose execs (the skills upload's staging steps) succeed."""

    async def exec(self, command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.parametrize(
    ("agent_derived", "state"),
    [(True, trust.TrustState.TAINTED), (False, trust.TrustState.PRISTINE)],
    ids=["learner-store", "configured"],
)
def test_learner_store_skills_are_admitted_as_agent_derived(
    tmp_path, agent_derived, state
):
    """Skills earlier agents wrote are an agent-derived ingress; the task's
    and the operator's configured skills stay BenchFlow's own transfer."""
    from benchflow.agents.install import _upload_skills

    box = _SkillsBox(tmp_path)
    trust.begin_fresh(box, Binding())
    skills = tmp_path / "skills" / "s"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# s\n")
    _run(
        _upload_skills(
            box, tmp_path / "skills", "/skills", "agent", agent_derived=agent_derived
        )
    )
    assert box.moved and box.moved[0][0] == "dir"
    assert trust.ledger(box).services["main"].state is state


def test_an_image_with_learner_skills_baked_in_starts_tainted(tmp_path):
    """A build that bakes a learner store's skills into the image: the
    sandbox is admitted as agent-derived as soon as it exists."""
    from benchflow.rollout._setup import _start_env_and_upload

    for baked, state in ((True, trust.TrustState.TAINTED), (False, None)):
        box = _SkillsBox(tmp_path, f"baked-{baked}")
        trust.begin_fresh(box, Binding())
        _run(
            _start_env_and_upload(
                box,
                tmp_path,
                {},
                skip_start=True,
                upload_task_files=False,
                image_holds_agent_bytes=baked,
            )
        )
        got = trust.ledger(box).services["main"].state
        assert got is (state or trust.TrustState.PRISTINE)
