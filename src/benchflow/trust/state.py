"""The host trust state of a sandbox (TRUSTED-RUNTIME v2 §3 I2, §5).

Each service of each sandbox generation is in one of three states:

- ``PRISTINE``: the container holds only what the task's image, setup,
  uploads and installs put there. ``bfx scan`` may run only here.
- ``MANIFEST``: a scan's facts and the checklist's verdict are recorded.
- ``TAINTED``: an ingress (§5) has been admitted. Facts read from now on are
  observations; they are compared with the manifest and never adopted.

The authoritative ledger lives in the controller's private run state, a
folder outside every folder the sandbox can reach, checked again at every
write (:func:`private_state_root`, :func:`_sandbox_mount_sources`). The trial
folder gets a report copy only (``trust-state.json``), which nothing reads
back.

Every ingress goes through :func:`admit`, which records TAINTED for the
service (and every service it reaches) and writes the ledger before it
returns, so the state is TAINTED before any ingress bytes move or any ingress
process starts. A state change stays in memory only once it is on disk, and
after a failed write every admission writes again until one succeeds. ``tests/test_trust_ingress_registry.py`` enumerates the
ingress call sites and requires each one to call :func:`admit`.

A generation is one container lineage:

- :func:`begin_fresh` opens a new generation from the task's image: every
  service starts PRISTINE.
- :func:`begin_restored` opens a generation from a snapshot: it inherits the
  parent's states and manifests before any restored process starts. A
  snapshot without lineage gives a TAINTED generation with no manifest: it
  is never treated as pristine, scans refuse, the sealed route raises
  :class:`LineageMissingError`, and ``strict`` refuses the restore itself.
- A stop and a later start of the same container stay in one generation.

The manifest is bound to the image digest, setup, credentials, ``bfx``
version and lineage (:class:`Binding`). A service is sealed only when its
manifest's ``checklist_passed`` is ``True`` and the binding matches; a missing
value is "not sealed".
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import enum
import hashlib
import json
import os
import secrets
import stat
import tempfile
import threading
import time
import weakref
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MAIN",
    "REPORT_NAME",
    "TRANSFER_PRIMITIVES",
    "TRUSTED_TRANSFER",
    "Admission",
    "AdmissionRequired",
    "Binding",
    "IngressKind",
    "LineageMissingError",
    "NotSealedError",
    "ServiceTrust",
    "TrustLedger",
    "TrustState",
    "TrustStateError",
    "TrustedTransfer",
    "admit",
    "begin_fresh",
    "begin_restored",
    "checklist_passed",
    "guard_transfer",
    "ledger",
    "lineage_record",
    "private_state_root",
    "record_manifest",
    "require_admission",
    "require_pristine",
    "require_sealed",
    "restore_from",
    "sealed",
    "stamp_snapshot",
]

MAIN = "main"
REPORT_NAME = "trust-state.json"
LEDGER_VERSION = 1
# A snapshot's lineage rides on its SandboxImage's meta under this key: the
# frozen record's id. The record itself stays in the private run state.
LINEAGE_META_KEY = "benchflow_trust_lineage"
# Beside it: the sha256 of the frozen record's bytes, and the private folder
# that held it when the snapshot was taken.
LINEAGE_SHA256_META_KEY = "benchflow_trust_lineage_sha256"
LINEAGE_ROOT_META_KEY = "benchflow_trust_lineage_root"


class TrustState(enum.StrEnum):
    PRISTINE = "pristine"
    MANIFEST = "manifest"
    TAINTED = "tainted"


class IngressKind(enum.StrEnum):
    """Every kind of ingress in TRUSTED-RUNTIME v2 §5."""

    AGENT_LAUNCH = "agent-launch"
    ORACLE_LAUNCH = "oracle-launch"
    SESSION_FACTORY = "session-factory"
    SDK_EXEC = "sdk-exec"
    SDK_UPLOAD = "sdk-upload"
    AGENT_DERIVED_UPLOAD = "agent-derived-upload"
    RECOVERY_TRANSFER = "recovery-transfer"
    REGRADE_TRANSFER = "regrade-transfer"
    REVIEW_RESUME_TRANSFER = "review-resume-transfer"
    SEPARATE_VERIFIER_TRANSFER = "separate-verifier-transfer"
    SNAPSHOT_RESTORE = "snapshot-restore"
    BRANCH_CHILD = "branch-child"
    CONTINUATION_REPLAY = "continuation-replay"
    USER_CONTEXT = "user-context"
    SHARED_MOUNT = "shared-mount"


class TrustStateError(RuntimeError):
    """A trust-state transition that the state machine does not allow."""


class LineageMissingError(TrustStateError):
    """A restored or branched sandbox whose parent manifest is unknown."""


class NotSealedError(TrustStateError):
    """The sealed system route is not admitted for a service."""


_LINEAGE_MISSING = (
    "restored sandbox has no trust lineage: it is never treated as pristine "
    "or sealed (legacy resume refuses, or use the separate verifier)"
)


@dataclass(frozen=True)
class Binding:
    """What a manifest holds for. Any field that differs means "not sealed".

    ``setup`` and ``credentials`` are digests the caller computes from the
    task's setup definition and the credential names staged before the scan
    (never their values). ``lineage`` is the chain of ledger record ids this
    generation descends from, oldest first.
    """

    image_digest: str | None = None
    setup: str | None = None
    credentials: str | None = None
    bfx_version: str | None = None
    lineage: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "image_digest": self.image_digest,
            "setup": self.setup,
            "credentials": self.credentials,
            "bfx_version": self.bfx_version,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Binding:
        return cls(
            image_digest=record.get("image_digest"),
            setup=record.get("setup"),
            credentials=record.get("credentials"),
            bfx_version=record.get("bfx_version"),
            lineage=tuple(record.get("lineage") or ()),
        )


@dataclass
class ServiceTrust:
    """One service's state in one generation."""

    state: TrustState = TrustState.PRISTINE
    manifest_sha256: str | None = None
    manifest_binding: Binding | None = None
    checklist_passed: bool | None = None
    reasons: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    admitted: dict[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "manifest_sha256": self.manifest_sha256,
            "manifest_binding": (
                self.manifest_binding.to_record() if self.manifest_binding else None
            ),
            "checklist_passed": self.checklist_passed,
            "reasons": list(self.reasons),
            "history": list(self.history),
            "admitted": dict(self.admitted),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ServiceTrust:
        binding = record.get("manifest_binding")
        passed = record.get("checklist_passed")
        return cls(
            state=TrustState(record["state"]),
            manifest_sha256=record.get("manifest_sha256"),
            manifest_binding=Binding.from_record(binding) if binding else None,
            checklist_passed=passed if isinstance(passed, bool) else None,
            reasons=list(record.get("reasons") or ()),
            history=list(record.get("history") or ()),
            admitted={
                str(key): value
                for key, value in (record.get("admitted") or {}).items()
                if isinstance(value, int)
            },
        )


def private_state_root() -> Path:
    """The controller's private run state: never mounted into a sandbox.

    ``BENCHFLOW_PRIVATE_STATE`` overrides the default
    ``$XDG_STATE_HOME/benchflow/trust`` (``~/.local/state/benchflow/trust``).
    The folder is created mode 0700.
    """
    configured = os.environ.get("BENCHFLOW_PRIVATE_STATE")
    if configured:
        root = Path(configured)
    else:
        base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        root = Path(base) / "benchflow" / "trust"
    root = Path(os.path.realpath(root.absolute()))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(root, 0o700)
    return root


def _sandbox_mount_sources(sandbox: Any) -> list[Path]:
    """Host folders a sandbox can reach, on every backend:

    - its rollout folder (Apple mounts it whole) and its environment folder;
    - every bind source of the operator's ``--mount`` options
      (``_mounts_json``), on any backend that holds them;
    - every host path the backend names in ``trust_host_mount_sources()``
      (Docker: the operator's binds and the frozen launch's binds).

    Raises :class:`TrustStateError` when the backend cannot list its binds.
    """
    values: list[Any] = []
    paths = getattr(sandbox, "rollout_paths", None)
    values.append(getattr(paths, "rollout_dir", None))
    values.append(getattr(sandbox, "environment_dir", None))
    mounts = getattr(sandbox, "_mounts_json", None)
    if isinstance(mounts, list):
        values.extend(
            mount.get("source")
            for mount in mounts
            if isinstance(mount, dict) and mount.get("type") in (None, "", "bind")
        )
    named = getattr(type(sandbox), "trust_host_mount_sources", None)
    if callable(named):
        try:
            listed = named(sandbox)
        except Exception as exc:
            raise TrustStateError(
                f"the sandbox's host mounts cannot be listed ({exc}); the trust "
                "ledger's isolation from them is not established"
            ) from exc
        values.extend(listed or ())
    return [
        Path(value)
        for value in values
        if isinstance(value, (str, os.PathLike)) and str(value)
    ]


def _check_outside_mounts(root: Path, sandbox: Any) -> None:
    """Raise unless ``root`` is outside every folder the sandbox can reach:
    neither a mount source nor beneath one (an ancestor's mount reaches it)."""
    from benchflow._utils.reachable_write import _inside_declared_mount

    if _inside_declared_mount(root):
        raise TrustStateError(
            f"private trust state {root} is inside a folder the harness mounts"
        )
    real = Path(os.path.realpath(root))
    for source in _sandbox_mount_sources(sandbox):
        try:
            src = Path(os.path.realpath(source))
        except OSError:
            src = Path(os.path.abspath(source))
        if real == src or src in real.parents:
            raise TrustStateError(
                f"private trust state {root} is inside sandbox-reachable {source}"
            )


def _fallback_root() -> Path:
    """The private folder used where a sandbox can reach
    :func:`private_state_root`: ``benchflow-trust-<uid>`` in the system
    temporary folder, 0700 and owned by this user. It is the same folder for
    every sandbox of this user, so a restore finds lineage frozen there. A
    folder that is not ours, not 0700 or not a folder gives a new private
    temporary folder instead."""
    base = Path(os.path.realpath(tempfile.gettempdir()))
    root = base / f"benchflow-trust-{os.geteuid()}"
    with contextlib.suppress(FileExistsError):
        root.mkdir(mode=0o700)
    try:
        info = os.lstat(root)
    except OSError:
        info = None
    if (
        info is None
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        return Path(os.path.realpath(tempfile.mkdtemp(prefix="benchflow-trust-")))
    return root


def _private_root_for(sandbox: Any) -> Path:
    """:func:`private_state_root`, or, where a sandbox can reach that folder,
    :func:`_fallback_root`. Raises when the sandbox can reach both."""
    root = private_state_root()
    try:
        _check_outside_mounts(root, sandbox)
    except TrustStateError:
        fallback = _fallback_root()
        _check_outside_mounts(fallback, sandbox)
        return fallback
    return root


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".trust-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class TrustLedger:
    """The trust state of one sandbox object across its generations."""

    def __init__(self, sandbox: Any, root: Path | None = None) -> None:
        self._lock = threading.RLock()
        try:
            self._sandbox_ref: Any = weakref.ref(sandbox)
        except TypeError:
            self._sandbox_ref = lambda: sandbox
        # A root the caller chose is never moved: a sandbox that can reach it
        # refuses the write instead.
        self._fixed_root = root is not None
        if root is None:
            root = _private_root_for(sandbox)
        else:
            _check_outside_mounts(root, sandbox)
        self.root = root
        # Set when an authoritative write failed, cleared by the next write
        # that succeeds. While set, every admission writes (and so raises
        # while writes fail), and no scan or sealed route is admitted.
        self._write_failed = False
        session = str(getattr(sandbox, "session_id", "") or "sandbox")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session)
        self.record_id = f"{safe[:80]}-{secrets.token_hex(8)}"
        self.path = self.root / f"{self.record_id}.json"
        self.generation = 0
        self.lineage_missing = False
        self.binding = Binding()
        self.services: dict[str, ServiceTrust] = {}
        self._report_dir: Path | None = None
        paths = getattr(sandbox, "rollout_paths", None)
        rollout_dir = getattr(paths, "rollout_dir", None)
        if isinstance(rollout_dir, (str, os.PathLike)):
            self._report_dir = Path(rollout_dir)

    # -- generations -----------------------------------------------------

    def begin_fresh(self, binding: Binding, services: Iterable[str]) -> None:
        with self._lock, self._change():
            self.generation += 1
            self.lineage_missing = False
            self.binding = binding
            self.services = {name: ServiceTrust() for name in {MAIN, *services}}
            self._event(None, "fresh")

    def begin_restored(
        self,
        parent: Mapping[str, Any] | None,
        binding: Binding | None = None,
        *,
        strict: bool = False,
    ) -> None:
        """Inherit a parent's states and manifests.

        Without a parent record the new generation is TAINTED for every
        service, with no manifest and ``lineage_missing`` set: a scan is
        refused and the sealed route is refused (:meth:`require_sealed`).
        With ``strict`` the restore itself is refused.
        """
        if not parent or not isinstance(parent.get("services"), Mapping):
            if strict:
                raise LineageMissingError(_LINEAGE_MISSING)
            with self._lock, self._change():
                names = {MAIN, *self.services}
                self.generation += 1
                self.binding = Binding(
                    image_digest=self.binding.image_digest,
                    setup=self.binding.setup,
                    credentials=self.binding.credentials,
                    bfx_version=self.binding.bfx_version,
                )
                self.lineage_missing = True
                self.services = {name: ServiceTrust() for name in names}
                for name in names:
                    self._taint(name, IngressKind.SNAPSHOT_RESTORE)
                self._event(None, "restored", parent=None)
            return
        with self._lock, self._change():
            # A parent that itself lacked lineage passes that on.
            self.lineage_missing = parent.get("lineage_missing") is True
            parent_binding = Binding.from_record(parent.get("binding") or {})
            lineage = (*parent_binding.lineage, str(parent["record_id"]))
            base = binding or parent_binding
            self.generation += 1
            self.binding = Binding(
                image_digest=base.image_digest,
                setup=base.setup,
                credentials=base.credentials,
                bfx_version=base.bfx_version,
                lineage=lineage,
            )
            self.services = {
                name: ServiceTrust.from_record(rec)
                for name, rec in parent["services"].items()
            }
            # A restored service is never more trusted than its parent; the
            # restore itself is an ingress of the snapshot's bytes.
            for name in self.services:
                self._taint(name, IngressKind.SNAPSHOT_RESTORE)
            self._event(None, "restored", parent=str(parent["record_id"]))

    # -- transitions -----------------------------------------------------

    def service(self, name: str) -> ServiceTrust:
        with self._lock:
            entry = self.services.get(name)
            if entry is None:
                # A service first seen after the generation began: it shares
                # main's container lineage only if main is pristine.
                entry = ServiceTrust()
                main = self.services.get(MAIN)
                if main is not None and main.state is TrustState.TAINTED:
                    entry.state = TrustState.TAINTED
                    entry.history.append(
                        {"t": time.time(), "event": "first-seen-after-taint"}
                    )
                self.services[name] = entry
            return entry

    def _taint(self, name: str, kind: IngressKind) -> bool:
        """Mark ``name`` TAINTED; True when that changed its state. Repeated
        ingress is counted per kind, not listed."""
        entry = self.service(name)
        entry.admitted[kind.value] = entry.admitted.get(kind.value, 0) + 1
        if entry.state is TrustState.TAINTED:
            return False
        entry.state = TrustState.TAINTED
        entry.history.append({"t": time.time(), "event": "admit", "kind": kind.value})
        return True

    def admit(self, services: Iterable[str], kind: IngressKind) -> None:
        """Mark each service TAINTED. Whenever a state changes, the ledger on
        disk is written before this returns; an ingress into services that are
        all TAINTED already changes no state and is only counted.

        After a failed write every admission writes again, and raises while
        writes fail: an admission returns only once a ledger holding TAINTED
        for these services is on disk."""
        with self._lock:
            before = self._memory()
            changed = self._write_failed
            if self.generation == 0:
                # Admitted before any start was recorded (a backend that does
                # not report starts): the sandbox is tainted from here on, and
                # no later scan can make it pristine.
                self.generation = 1
                changed = True
            for name in services:
                changed = self._taint(name, kind) or changed
            if changed:
                try:
                    self._save()
                except BaseException:
                    self._restore_memory(before)
                    raise

    def require_pristine(self, name: str) -> None:
        with self._lock:
            if self._write_failed:
                raise TrustStateError(
                    f"scan of service {name!r} refused: the last write of the "
                    "trust ledger failed, so its saved state is not known"
                )
            entry = self.service(name)
            if self.generation == 0 or entry.state is not TrustState.PRISTINE:
                raise TrustStateError(
                    f"scan of service {name!r} refused: its state is "
                    f"{entry.state.value if self.generation else 'unstarted'}, "
                    "and a scan runs only in PRISTINE"
                )

    def record_manifest(
        self,
        name: str,
        manifest: Mapping[str, Any],
        *,
        checklist_passed: bool | None,
        reasons: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        with self._lock:
            self.require_pristine(name)
            self._record_manifest(name, manifest, checklist_passed, reasons)

    def _record_manifest(
        self,
        name: str,
        manifest: Mapping[str, Any],
        checklist_passed: bool | None,
        reasons: Iterable[Mapping[str, Any]],
    ) -> None:
        with self._change():
            entry = self.service(name)
            blob = json.dumps(manifest, sort_keys=True, default=str).encode()
            entry.state = TrustState.MANIFEST
            entry.manifest_sha256 = hashlib.sha256(blob).hexdigest()
            entry.manifest_binding = self.binding
            entry.checklist_passed = (
                checklist_passed if isinstance(checklist_passed, bool) else None
            )
            entry.reasons = [dict(r) for r in reasons]
            entry.history.append(
                {
                    "t": time.time(),
                    "event": "manifest",
                    "checklist_passed": entry.checklist_passed,
                }
            )

    def sealed(self, name: str) -> bool:
        with self._lock:
            if self._write_failed:
                return False
            entry = self.services.get(name)
            if entry is None or entry.manifest_binding is None:
                return False
            if entry.checklist_passed is not True:
                return False
            binding = entry.manifest_binding
            current = self.binding
            return (
                binding.image_digest == current.image_digest
                and binding.setup == current.setup
                and binding.credentials == current.credentials
                and binding.bfx_version == current.bfx_version
                and tuple(current.lineage[: len(binding.lineage)]) == binding.lineage
            )

    # -- storage -----------------------------------------------------------

    def _event(self, name: str | None, event: str, **detail: Any) -> None:
        targets = [self.service(name)] if name else self.services.values()
        for entry in targets:
            entry.history.append({"t": time.time(), "event": event, **detail})

    def to_record(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": LEDGER_VERSION,
                "record_id": self.record_id,
                "generation": self.generation,
                "lineage_missing": self.lineage_missing,
                "binding": self.binding.to_record(),
                "services": {
                    name: entry.to_record()
                    for name, entry in sorted(self.services.items())
                },
            }

    def _memory(self) -> tuple[Any, ...]:
        return copy.deepcopy(
            (self.generation, self.lineage_missing, self.binding, self.services)
        )

    def _restore_memory(self, saved: tuple[Any, ...]) -> None:
        self.generation, self.lineage_missing, self.binding, self.services = saved

    @contextlib.contextmanager
    def _change(self) -> Iterator[None]:
        """One state change: it stays in memory only once the ledger holding
        it is on disk. A failed write puts memory back as it was."""
        before = self._memory()
        try:
            yield
            self._save()
        except BaseException:
            self._restore_memory(before)
            raise

    def _place(self) -> None:
        """Keep the authoritative ledger outside every folder the sandbox can
        reach now. The sandbox's binds can grow after the ledger was created
        (a frozen launch); a reachable root moves to :func:`_fallback_root`,
        and a root the caller chose refuses the write."""
        sandbox = self._sandbox_ref()
        if sandbox is None:
            return
        try:
            _check_outside_mounts(self.root, sandbox)
            return
        except TrustStateError:
            if self._fixed_root:
                raise
        root = _fallback_root()
        _check_outside_mounts(root, sandbox)
        # The file at the old path stays as it was. It is harmless: nothing
        # reads a ledger back (memory is authoritative, and a restore reads
        # only the frozen lineage record, by its digest). It is not removed,
        # because its folder is one the sandbox can reach: an unlink there is
        # a host write through a path the sandbox can replace with a link.
        self.root = root
        self.path = root / f"{self.record_id}.json"

    def _save(self) -> None:
        """Write the authoritative ledger, then the report copy. A failure of
        the authoritative write raises and sets ``_write_failed`` until a
        later write succeeds."""
        data = json.dumps(self.to_record(), indent=2, sort_keys=True).encode()
        try:
            self._place()
            _atomic_write(self.path, data, 0o600)
        except BaseException:
            self._write_failed = True
            raise
        self._write_failed = False
        if self._report_dir is not None and self._report_dir.is_dir():
            # The report copy is for people; a failure to write it never
            # changes the authoritative state.
            with contextlib.suppress(OSError):
                _atomic_write(self._report_dir / REPORT_NAME, data, 0o644)


def load_record(
    record_id: str, root: Path | None = None, *, sha256: str | None = None
) -> dict[str, Any] | None:
    """A ledger record from the private run state, or None. With ``sha256``,
    only a file with exactly those bytes is a record."""
    if not record_id or "/" in record_id or record_id.startswith("."):
        return None
    path = (root or private_state_root()) / f"{record_id}.json"
    try:
        data = path.read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            return None
        record = json.loads(data)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("record_id") != record_id:
        return None
    return record


# -- the sandbox-facing API ------------------------------------------------

_LEDGER_ATTR = "_trust_ledger"
_LEDGER_LOCK = threading.Lock()
_UNATTACHED: dict[int, tuple[Any, TrustLedger]] = {}


def ledger(sandbox: Any) -> TrustLedger:
    """The sandbox's ledger, created on first use."""
    existing = getattr(sandbox, _LEDGER_ATTR, None)
    if isinstance(existing, TrustLedger):
        return existing
    with _LEDGER_LOCK:
        existing = getattr(sandbox, _LEDGER_ATTR, None)
        if isinstance(existing, TrustLedger):
            return existing
        held = _UNATTACHED.get(id(sandbox))
        if held is not None and held[0] is sandbox:
            return held[1]
        created = TrustLedger(sandbox)
        try:
            object.__setattr__(sandbox, _LEDGER_ATTR, created)
        except (AttributeError, TypeError):
            # An object that cannot hold attributes keeps its ledger here,
            # with a reference to it, so its id is never reused meanwhile.
            _UNATTACHED[id(sandbox)] = (sandbox, created)
        return created


def _reached_services(sandbox: Any, service: str, kind: IngressKind) -> list[str]:
    """The services an ingress into ``service`` reaches.

    A service is tainted when any agent-writable volume or socket reaches it.
    ``sandbox.trust_shared_services(service)`` names the services an ingress
    into ``service`` reaches, where the backend knows them exactly. Without
    it, every known service is reached. A shared-mount ingress always reaches
    every known service.
    """
    names = {service}
    shared = getattr(type(sandbox), "trust_shared_services", None)
    reached = (
        shared(sandbox, service)
        if callable(shared) and kind is not IngressKind.SHARED_MOUNT
        else None
    )
    if isinstance(reached, (list, tuple, set, frozenset)):
        names.update(str(name) for name in reached)
    else:
        names.update(ledger(sandbox).services)
    return sorted(names)


class TrustedTransfer(enum.Enum):
    """The one non-ingress kind :func:`admit` takes: bytes that are the
    task's or BenchFlow's own. It changes no state; the reason is required."""

    TRUSTED = "trusted-transfer"


TRUSTED_TRANSFER = TrustedTransfer.TRUSTED

_MINT = object()


class Admission:
    """The proof every transfer primitive (``upload_file``, ``upload_dir``,
    ``live_process``, ``script_runner_upload``) requires: one sandbox's ledger, its generation, the
    services the admission reached, and the kind. Only :func:`admit` makes
    one."""

    __slots__ = ("_book", "_mint", "generation", "kind", "reason", "services")

    def __init__(
        self,
        mint: object,
        book: TrustLedger,
        generation: int,
        services: frozenset[str],
        kind: IngressKind | TrustedTransfer,
        reason: str | None,
    ) -> None:
        if mint is not _MINT:
            raise TypeError("an Admission is made only by benchflow.trust.state.admit")
        self._mint = mint
        self._book = book
        self.generation = generation
        self.services = services
        self.kind = kind
        self.reason = reason

    def __repr__(self) -> str:
        kind = self.kind.value
        return (
            f"Admission({kind}, services={sorted(self.services)}, g{self.generation})"
        )


class AdmissionRequired(TrustStateError):
    """A transfer primitive was called without a valid :class:`Admission`."""


def admit(
    sandbox: Any,
    service: str,
    kind: IngressKind | TrustedTransfer,
    *,
    reason: str | None = None,
) -> Admission:
    """Admit a transfer into ``service`` and return the :class:`Admission`
    the transfer primitive requires.

    For an :class:`IngressKind`, record TAINTED for ``service`` (and every
    service it reaches) before the ingress moves bytes or starts a process.
    The ledger is written before this returns. ``admit`` never raises for a
    tainted sandbox: an ingress into an already tainted service is recorded
    again and allowed. A failure to write the ledger raises, and the caller's
    ingress does not proceed.

    For :data:`TRUSTED_TRANSFER`, ``reason`` says why the bytes are the
    task's or BenchFlow's own; no state changes, and the admission reaches
    ``service`` only.
    """
    book = ledger(sandbox)
    if isinstance(kind, TrustedTransfer):
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError("a trusted transfer is admitted with its reason")
        with book._lock:
            return Admission(
                _MINT, book, book.generation, frozenset({service}), kind, reason
            )
    if not isinstance(kind, IngressKind):
        kind = IngressKind(kind)
    reached = _reached_services(sandbox, service, kind)
    with book._lock:
        book.admit(reached, kind)
        return Admission(_MINT, book, book.generation, frozenset(reached), kind, reason)


def require_admission(
    sandbox: Any, admission: Any, primitive: str, service: str = MAIN
) -> None:
    """Raise :class:`AdmissionRequired` unless ``admission`` was made by
    :func:`admit` for this sandbox, in its current generation, and reached
    ``service``."""
    if (
        not isinstance(admission, Admission)
        or getattr(admission, "_mint", None) is not _MINT
    ):
        raise AdmissionRequired(
            f"{primitive} into {service!r} needs the Admission that "
            "trust.admit() returns"
        )
    book = _existing(sandbox)
    if book is None or admission._book is not book:
        raise AdmissionRequired(f"{primitive}: the admission is for another sandbox")
    if admission.generation != book.generation:
        raise AdmissionRequired(
            f"{primitive}: the admission is for generation {admission.generation}, "
            f"and the sandbox is in generation {book.generation}"
        )
    if service not in admission.services:
        raise AdmissionRequired(
            f"{primitive}: the admission does not reach service {service!r}"
        )


TRANSFER_PRIMITIVES = (
    "upload_file",
    "upload_dir",
    "live_process",
    "script_runner_upload",
)

# The sandboxes whose transfer primitive is running in this context: a
# backend's primitive that calls its own (or its base class's) primitive is
# one transfer, admitted once.
_IN_TRANSFER: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "benchflow_trust_in_transfer", default=frozenset()
)


def guard_transfer(name: str, impl: Any) -> Any:
    """``impl``, a backend's transfer primitive, taking a required
    ``admission`` keyword that :func:`require_admission` checks before
    ``impl`` runs. ``impl`` itself never sees the keyword."""
    import functools

    @functools.wraps(impl)
    async def primitive(self: Any, *args: Any, admission: Any = None, **kwargs: Any):
        inside = _IN_TRANSFER.get()
        if id(self) in inside:
            return await impl(self, *args, **kwargs)
        service = MAIN
        if name == "upload_dir":
            service = kwargs.get("service", args[2] if len(args) > 2 else MAIN)
        require_admission(self, admission, name, str(service))
        token = _IN_TRANSFER.set(inside | {id(self)})
        try:
            return await impl(self, *args, **kwargs)
        finally:
            _IN_TRANSFER.reset(token)

    vars(primitive)["__benchflow_guarded__"] = True
    return primitive


def begin_fresh(
    sandbox: Any, binding: Binding | None = None, services: Iterable[str] = ()
) -> None:
    """A new container from the task's image: every service PRISTINE."""
    ledger(sandbox).begin_fresh(binding or Binding(), services)


def _freeze(sandbox: Any, image_ref: str) -> tuple[str, str, Path]:
    """Write the frozen record; return its id, the sha256 of its bytes and
    the private folder holding it."""
    book = ledger(sandbox)
    with book._lock:
        book._place()
        record = book.to_record()
        frozen_id = f"{book.record_id}-g{book.generation}-{secrets.token_hex(4)}"
        record["record_id"] = frozen_id
        record["frozen_from"] = book.record_id
        record["image_ref"] = image_ref
        data = json.dumps(record, indent=2, sort_keys=True).encode()
        _atomic_write(book.root / f"{frozen_id}.json", data, 0o600)
        return frozen_id, hashlib.sha256(data).hexdigest(), book.root


def lineage_record(sandbox: Any, image_ref: str) -> str:
    """Freeze the sandbox's state now, for the snapshot ``image_ref`` taken
    now, and return the frozen record's id.

    The frozen record is a separate file bound to ``image_ref``: later
    ingress into the source sandbox never changes what a restore of this
    snapshot inherits, and the record admits a restore of that image only.
    """
    return _freeze(sandbox, image_ref)[0]


def stamp_snapshot(sandbox: Any, image: Any) -> Any:
    """``image`` with the sandbox's trust state recorded as its lineage: the
    frozen record's id, the sha256 of its bytes and the private folder that
    holds it. A restore reads that record wherever it is found, and only
    with those bytes."""
    import dataclasses

    record_id, digest, root = _freeze(sandbox, image.ref)
    return dataclasses.replace(
        image,
        meta={
            **image.meta,
            LINEAGE_META_KEY: record_id,
            LINEAGE_SHA256_META_KEY: digest,
            LINEAGE_ROOT_META_KEY: str(root),
        },
    )


def _find_lineage(
    book: TrustLedger,
    sandbox: Any,
    lineage_id: str,
    sha256: str | None,
    root_hint: str | None,
) -> dict[str, Any] | None:
    """The parent record, from any private folder that holds it: the folder
    the snapshot named, this ledger's, :func:`private_state_root` and
    :func:`_fallback_root`. With the record's sha256 any of them may supply
    it; without, only a folder this sandbox cannot reach."""
    candidates: list[Path] = []
    if sha256 is not None and root_hint and os.path.isabs(root_hint):
        candidates.append(Path(root_hint))
    candidates.append(book.root)
    for choose in (private_state_root, _fallback_root):
        with contextlib.suppress(OSError):
            candidates.append(choose())
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if sha256 is None:
            try:
                _check_outside_mounts(root, sandbox)
            except TrustStateError:
                continue
        record = load_record(lineage_id, root, sha256=sha256)
        if record is not None:
            return record
    return None


def begin_restored(
    sandbox: Any,
    lineage_id: str | None,
    binding: Binding | None = None,
    *,
    image_ref: str | None = None,
    strict: bool = False,
    sha256: str | None = None,
    root_hint: str | None = None,
) -> None:
    """A container from a snapshot: inherit the parent's states before any
    restored process starts.

    A parent record that is missing, unreadable, other than the bytes the
    snapshot recorded, or bound to another image is no lineage: the
    generation is TAINTED with no manifest (``strict`` refuses the restore
    instead).
    """
    book = ledger(sandbox)
    parent = (
        _find_lineage(book, sandbox, lineage_id, sha256, root_hint)
        if lineage_id
        else None
    )
    if (
        parent is not None
        and image_ref is not None
        and parent.get("image_ref") != image_ref
    ):
        parent = None
    book.begin_restored(parent, binding, strict=strict)


def restore_from(sandbox: Any, image: Any, *, strict: bool = False) -> None:
    """:func:`begin_restored` from a SandboxImage's lineage meta."""
    meta = getattr(image, "meta", None) or {}
    if not isinstance(meta, Mapping):
        meta = {}

    def text(key: str) -> str | None:
        value = meta.get(key)
        return value if isinstance(value, str) and value else None

    begin_restored(
        sandbox,
        text(LINEAGE_META_KEY),
        image_ref=str(getattr(image, "ref", "")),
        strict=strict,
        sha256=text(LINEAGE_SHA256_META_KEY),
        root_hint=text(LINEAGE_ROOT_META_KEY),
    )


def require_pristine(sandbox: Any, service: str = MAIN) -> None:
    ledger(sandbox).require_pristine(service)


def record_manifest(
    sandbox: Any,
    service: str,
    manifest: Mapping[str, Any],
    *,
    checklist_passed: bool | None,
    reasons: Iterable[Mapping[str, Any]] = (),
) -> None:
    ledger(sandbox).record_manifest(
        service, manifest, checklist_passed=checklist_passed, reasons=reasons
    )


def checklist_passed(manifest: Mapping[str, Any] | None, service: str) -> bool:
    """The sealed route's admission input, read from a manifest.

    Until the I3 policy modules land, the manifest carries one boolean per
    service: ``{"services": {"main": {"checklist_passed": true}}}``. Anything
    other than ``True`` (absent, null, a string) is "not passed".
    """
    if not isinstance(manifest, Mapping):
        return False
    services = manifest.get("services")
    if not isinstance(services, Mapping):
        return False
    entry = services.get(service)
    if not isinstance(entry, Mapping):
        return False
    return entry.get("checklist_passed") is True


def _existing(sandbox: Any) -> TrustLedger | None:
    book = getattr(sandbox, _LEDGER_ATTR, None)
    if isinstance(book, TrustLedger):
        return book
    held = _UNATTACHED.get(id(sandbox))
    return held[1] if held is not None and held[0] is sandbox else None


def sealed(sandbox: Any, service: str = MAIN) -> bool:
    """Whether the sealed system route is admitted for ``service``."""
    book = _existing(sandbox)
    if not isinstance(book, TrustLedger):
        return False
    return book.sealed(service)


def require_sealed(sandbox: Any, service: str = MAIN) -> None:
    """Admit the sealed system route for ``service``, or raise.

    Raises :class:`LineageMissingError` for a restored generation without
    lineage, and :class:`NotSealedError` for every other "not sealed".
    """
    book = _existing(sandbox)
    if book is not None and book.lineage_missing:
        raise LineageMissingError(_LINEAGE_MISSING)
    if not sealed(sandbox, service):
        raise NotSealedError(
            f"service {service!r} is not sealed: no manifest with a passed "
            "system checklist for this image, setup, credentials, bfx and lineage"
        )
