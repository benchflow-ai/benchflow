"""Hugging Face Sandbox backend (``--sandbox hf-sandbox``).

Each rollout gets one *dedicated* ``huggingface_hub.Sandbox`` (one HF Job running
the prebuilt task image, root, open network). Pool sandboxes are not used: they
run as uid 20000 under Landlock, which Terminal-Bench-style tasks can't live with.
Modelled on Harbor's ``environments/hf_sandbox.py``.

Facts this backend relies on (probed Sept 30, 2026): start 4-7 s on cpu-basic;
exec round trip 30-40 ms; ``cpu-basic`` = 2 vCPU / 16 GB, ``cpu-upgrade`` =
8 vCPU / 32 GB; no Docker, no KVM, no enforced offline mode; prebuilt images only
(a task that only ships a Dockerfile must be built and pushed first).

Lifecycle and cost controls:

- ``idle_timeout``: the sandbox shuts itself down after this long without API
  calls or running processes. A foreground command does not count as activity,
  so the default is the task's agent + verifier + build time limits plus 10
  minutes (at least 30 minutes); ``BENCHFLOW_HF_SANDBOX_IDLE_TIMEOUT_SEC`` overrides.
- labels: every sandbox carries ``benchflow-managed=1``, ``benchflow-session``,
  optional ``benchflow-run`` (``BENCHFLOW_HF_SANDBOX_RUN``) and any extra labels
  from ``BENCHFLOW_HF_SANDBOX_LABELS`` (``k=v,k=v``, e.g. ``posttrain=1``).
- cleanup: live sandboxes are killed on ``stop()``, at interpreter exit and on
  SIGTERM/SIGINT (a canceled HF Job gets SIGTERM ~60 s before it is killed);
  :func:`sweep` cancels labelled leftovers (``python -m benchflow.sandbox.hf_sandbox sweep``).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import io
import logging
import os
import shlex
import signal
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from benchflow._paths import iter_safe_tree
from benchflow.sandbox._base import BaseSandbox, ExecResult
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths, SandboxPaths

logger = logging.getLogger("benchflow")

MANAGED_LABEL = "benchflow-managed"
SESSION_LABEL = "benchflow-session"
RUN_LABEL = "benchflow-run"
_ENV_FLAVOR = "BENCHFLOW_HF_SANDBOX_FLAVOR"
_ENV_IDLE = "BENCHFLOW_HF_SANDBOX_IDLE_TIMEOUT_SEC"
_ENV_NAMESPACE = "BENCHFLOW_HF_SANDBOX_NAMESPACE"
_ENV_LABELS = "BENCHFLOW_HF_SANDBOX_LABELS"
_ENV_RUN = "BENCHFLOW_HF_SANDBOX_RUN"
_ENV_START_TIMEOUT = "BENCHFLOW_HF_SANDBOX_START_TIMEOUT_SEC"
_MIN_IDLE_SEC = 1800
_IDLE_MARGIN_SEC = 600
# (flavor, vCPU, memory MB), smallest first. GPU flavors are opt-in via BENCHFLOW_HF_SANDBOX_FLAVOR.
_CPU_FLAVORS = (("cpu-basic", 2, 16 * 1024), ("cpu-upgrade", 8, 32 * 1024))
_TERMINAL_STAGES = {"COMPLETED", "ERROR", "DELETED", "CANCELED"}

# Live sandboxes of this process, for cleanup on exit/cancel: id -> huggingface_hub.Sandbox
_LIVE: dict[str, Any] = {}
_LIVE_LOCK = threading.RLock()  # re-entrant: the signal handler may run while the main thread holds it
_HOOKS_INSTALLED = False


def _label_value(value: str) -> str:
    # HF Job label values: keep to a conservative charset and length.
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in value).strip("-.")
    return cleaned[:63] or "x"


def extra_labels() -> dict[str, str]:
    """Labels from ``BENCHFLOW_HF_SANDBOX_LABELS`` (``k=v,k=v``) and ``BENCHFLOW_HF_SANDBOX_RUN``."""
    labels: dict[str, str] = {}
    for part in os.environ.get(_ENV_LABELS, "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                labels[k.strip()] = _label_value(v.strip())
    run = os.environ.get(_ENV_RUN, "").strip()
    if run:
        labels[RUN_LABEL] = _label_value(run)
    return labels


def pick_flavor(cpus: int, memory_mb: int, gpus: int = 0) -> str:
    """Smallest CPU flavor that fits the task, unless ``BENCHFLOW_HF_SANDBOX_FLAVOR`` is set."""
    override = os.environ.get(_ENV_FLAVOR, "").strip()
    if override:
        return override
    if gpus:
        raise ValueError(
            f"task asks for {gpus} GPU(s); set {_ENV_FLAVOR} to an HF GPU flavor to run it on HF Sandboxes"
        )
    for name, vcpu, mem in _CPU_FLAVORS:
        if cpus <= vcpu and memory_mb <= mem:
            return name
    name, vcpu, mem = _CPU_FLAVORS[-1]
    logger.warning(
        "HF sandbox: task asks for %s vCPU / %s MB, more than the largest CPU flavor (%s: %s vCPU / %s MB); using it anyway",
        cpus,
        memory_mb,
        name,
        vcpu,
        mem,
    )
    return name


def idle_timeout_for(config: SandboxConfig, agent_timeout: float | None, verifier_timeout: float | None) -> int:
    raw = os.environ.get(_ENV_IDLE, "").strip()
    if raw:
        return max(60, int(float(raw)))
    total = float(agent_timeout or 0) + float(verifier_timeout or 0) + float(config.build_timeout_sec or 0)
    return int(max(_MIN_IDLE_SEC, total + _IDLE_MARGIN_SEC))


def _kill_quietly(sandbox_id: str, sandbox: Any) -> None:
    try:
        sandbox.kill()
    except Exception as exc:  # pragma: no cover - best effort at exit
        logger.warning("HF sandbox %s: kill failed: %s", sandbox_id, exc)


def kill_all_live() -> int:
    """Kill every sandbox this process started and has not stopped. Returns how many."""
    with _LIVE_LOCK:
        live = list(_LIVE.items())
        _LIVE.clear()
    threads = [threading.Thread(target=_kill_quietly, args=item, daemon=True) for item in live]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    if live:
        logger.info("HF sandbox: killed %d live sandbox(es) at shutdown", len(live))
    return len(live)


def _install_cleanup_hooks() -> None:
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True
    atexit.register(kill_all_live)
    if threading.current_thread() is not threading.main_thread():
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(signum)
        except (ValueError, OSError):  # pragma: no cover
            continue

        def handler(sig: int, frame: Any, _previous: Any = previous) -> None:
            kill_all_live()
            if callable(_previous):
                _previous(sig, frame)
            elif _previous == signal.SIG_DFL:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)

        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):  # pragma: no cover
            pass


def sweep(
    *,
    namespace: str | None = None,
    run: str | None = None,
    labels: dict[str, str] | None = None,
    older_than_minutes: float = 0.0,
    dry_run: bool = False,
    token: str | None = None,
) -> dict[str, int]:
    """Cancel running BenchFlow HF sandboxes that match ``run``/``labels``.

    Only jobs labelled ``hf-sandbox=1`` *and* ``benchflow-managed=1`` are ever
    touched, so other jobs in a shared namespace are safe.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    want = {MANAGED_LABEL: "1", **(labels or {})}
    if run:
        want[RUN_LABEL] = _label_value(run)
    counts = {"found": 0, "canceled": 0, "failed": 0}
    now = time.time()
    for job in api.list_jobs(namespace=namespace):
        job_labels = getattr(job, "labels", None) or {}
        if job_labels.get("hf-sandbox") != "1":
            continue
        if any(job_labels.get(k) != v for k, v in want.items()):
            continue
        stage = str(getattr(getattr(job, "status", None), "stage", "") or "")
        if stage in _TERMINAL_STAGES:
            continue
        created = getattr(job, "created_at", None)
        if older_than_minutes and created is not None:
            if (now - created.timestamp()) / 60.0 < older_than_minutes:
                continue
        counts["found"] += 1
        if dry_run:
            logger.info("HF sandbox sweep: would cancel %s (%s)", job.id, stage)
            continue
        try:
            owner = getattr(getattr(job, "owner", None), "name", None) or namespace
            api.cancel_job(job_id=job.id, namespace=owner)
            counts["canceled"] += 1
        except Exception as exc:
            counts["failed"] += 1
            logger.warning("HF sandbox sweep: cancel %s failed: %s", job.id, exc)
    return counts


class HFSandbox(BaseSandbox):
    """One dedicated Hugging Face Sandbox per rollout."""

    @classmethod
    def preflight(cls) -> None:
        try:
            from huggingface_hub import Sandbox, get_token  # noqa: F401
        except ImportError:
            raise SystemExit(
                "huggingface_hub>=1.32 with Sandbox support is not installed. "
                "Install it with `uv sync --extra sandbox-hf`."
            ) from None
        if not get_token():
            raise SystemExit("HF Sandboxes need a Hugging Face token: run `hf auth login` or set HF_TOKEN.")

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        agent_timeout_sec: float | None = None,
        verifier_timeout_sec: float | None = None,
        *args: object,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )
        self._sandbox: Any = None
        self._flavor = pick_flavor(
            int(task_env_config.cpus or 1),
            int(task_env_config.memory_mb or 2048),
            int(task_env_config.gpus or 0),
        )
        self._idle_timeout = idle_timeout_for(task_env_config, agent_timeout_sec, verifier_timeout_sec)
        self._namespace = os.environ.get(_ENV_NAMESPACE, "").strip() or None
        if task_env_config.allow_internet is False:
            self.logger.warning(
                "HF Sandboxes cannot block network access; task %s asks for no internet and will run with it",
                environment_name,
            )

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def sandbox_id(self) -> str | None:
        return getattr(self._sandbox, "id", None)

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError(
                "HF Sandboxes run prebuilt images only: set sandbox.docker_image (task.md) or "
                "[environment].docker_image (task.toml). A Dockerfile-only task must be built and pushed first."
            )

    def _labels(self) -> dict[str, str]:
        return {**extra_labels(), MANAGED_LABEL: "1", SESSION_LABEL: _label_value(self.session_id)}

    def _require(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("HF sandbox not started. Call start() first.")
        return self._sandbox

    async def start(self, force_build: bool) -> None:
        from huggingface_hub import Sandbox

        image = self.task_env_config.docker_image
        start_timeout = float(os.environ.get(_ENV_START_TIMEOUT, "") or max(180.0, float(self.task_env_config.build_timeout_sec or 0)))
        labels = self._labels()
        self.logger.info(
            "Creating HF sandbox: image=%s flavor=%s idle_timeout=%ss labels=%s",
            image,
            self._flavor,
            self._idle_timeout,
            sorted(labels),
        )
        last: Exception | None = None
        for attempt in range(2):
            try:
                self._sandbox = await asyncio.to_thread(
                    Sandbox.create,
                    image=image,
                    flavor=self._flavor,
                    idle_timeout=self._idle_timeout,
                    namespace=self._namespace,
                    labels=labels,
                    start_timeout=start_timeout,
                )
                break
            except Exception as exc:
                last = exc
                self.logger.warning("HF sandbox create failed (attempt %d): %s", attempt + 1, exc)
                if "402" in str(exc) or "Payment" in str(exc):
                    break
                await asyncio.sleep(3)
        if self._sandbox is None:
            from benchflow.sandbox.protocol import SandboxStartupError

            raise SandboxStartupError(f"HF sandbox creation failed: {last}", attempts=2)
        with _LIVE_LOCK:
            _LIVE[self._sandbox.id] = self._sandbox
        _install_cleanup_hooks()
        self.logger.info("Sandbox %s (HFSandbox) started for %s", self._sandbox.id, self.session_id)
        try:
            dirs = [str(SandboxPaths.agent_dir), str(SandboxPaths.verifier_dir)]
            if self.task_env_config.workdir:
                dirs.append(self.task_env_config.workdir)
            quoted = " ".join(shlex.quote(d) for d in dirs)
            result = await self.exec(
                f"mkdir -p {quoted} && chmod 777 {shlex.quote(dirs[0])} {shlex.quote(dirs[1])}",
                user="root",
                timeout_sec=60,
            )
            if result.return_code != 0:
                raise RuntimeError(f"HF sandbox setup failed: {result.stderr or result.stdout}")
        except BaseException:
            await self.stop(delete=True)
            raise

    async def stop(self, delete: bool) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        self._sandbox = None
        with _LIVE_LOCK:
            _LIVE.pop(sandbox.id, None)
        if not delete:
            # Keeping it means it lives until its idle timeout; say so.
            self.logger.info(
                "HF sandbox %s kept (delete=False); it stops itself after %ss idle", sandbox.id, self._idle_timeout
            )
            return
        try:
            await asyncio.to_thread(sandbox.kill)
        except Exception as exc:
            self.logger.warning("HF sandbox %s: kill failed: %s", sandbox.id, exc)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        if service != "main":
            raise ValueError(
                f"HF sandbox is single-container and cannot target service {service!r}; "
                "multi-container tasks need the Docker sandbox."
            )
        sandbox = self._require()
        user = self._resolve_user(user)
        env = self._merge_env(env)
        if user is not None and str(user) not in ("root", "0"):
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"
        # argv mode: no extra /bin/sh -c wrapper; env travels in the API payload.
        try:
            result = await asyncio.to_thread(
                sandbox.run,
                ["/bin/bash", "-c", command],
                shell=False,
                cwd=cwd,
                env=env or None,
                timeout=timeout_sec,
                check=False,
            )
        except Exception as exc:
            self.logger.warning(
                "HF sandbox exec failed (%s): cwd=%r env_keys=%s command=%r",
                exc,
                cwd,
                sorted(env or {}),
                command[:300],
            )
            raise
        if getattr(result, "timed_out", False):
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        code = getattr(result, "exit_code", None)
        if code is None:
            sig = getattr(result, "signal", None)
            code = 128 + int(sig) if sig else -1
        return ExecResult(stdout=result.stdout or "", stderr=result.stderr or "", return_code=code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        sandbox = self._require()
        data = Path(source_path).read_bytes()
        await asyncio.to_thread(sandbox.files.write, target_path, data)

    async def upload_dir(self, source_dir: Path | str, target_dir: str, service: str = "main") -> None:
        if service != "main":
            raise ValueError(f"HF sandbox is single-container and cannot target service {service!r}.")
        sandbox = self._require()
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        def pack() -> bytes:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                # followlinks=False and symlinks skipped (#411): never ship host files a link points at.
                for path in iter_safe_tree(source, context=f"hf upload_dir {source}"):
                    tar.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
            return buf.getvalue()

        data = await asyncio.to_thread(pack)
        remote = f"/tmp/.bf-upload-{uuid.uuid4().hex[:12]}.tar.gz"
        await asyncio.to_thread(sandbox.files.write, remote, data)
        t = shlex.quote(target_dir)
        result = await self.exec(
            f"mkdir -p {t} && tar -xzf {shlex.quote(remote)} -C {t} --no-same-owner; rc=$?; rm -f {shlex.quote(remote)}; exit $rc",
            user="root",
            timeout_sec=300,
        )
        if result.return_code != 0:
            raise RuntimeError(f"HF sandbox upload_dir to {target_dir} failed: {result.stderr or result.stdout}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox = self._require()
        data = await asyncio.to_thread(sandbox.files.read, source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str, service: str = "main") -> None:
        if service != "main":
            raise ValueError(f"HF sandbox is single-container and cannot target service {service!r}.")
        sandbox = self._require()
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        remote = f"/tmp/.bf-download-{uuid.uuid4().hex[:12]}.tar.gz"
        s = shlex.quote(source_dir)
        result = await self.exec(
            f"[ -d {s} ] || exit 3; tar -czf {shlex.quote(remote)} -C {s} .",
            user="root",
            timeout_sec=300,
        )
        if result.return_code == 3:
            return  # nothing there, like the other backends' empty listing
        if result.return_code != 0:
            raise RuntimeError(f"HF sandbox download_dir {source_dir} failed: {result.stderr or result.stdout}")
        try:
            data = await asyncio.to_thread(sandbox.files.read, remote)
        finally:
            await self.exec(f"rm -f {shlex.quote(remote)}", user="root", timeout_sec=30)

        def unpack() -> None:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(target, filter="data")

        await asyncio.to_thread(unpack)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchflow.sandbox.hf_sandbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep", help="cancel running BenchFlow HF sandboxes by label")
    sw.add_argument("--namespace", default=os.environ.get(_ENV_NAMESPACE) or None)
    sw.add_argument("--run", default=os.environ.get(_ENV_RUN) or None, help="benchflow-run label value")
    sw.add_argument("--label", action="append", default=[], help="extra k=v label to match (repeatable)")
    sw.add_argument("--older-than-minutes", type=float, default=0.0)
    sw.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    labels = dict(part.split("=", 1) for part in a.label if "=" in part)
    counts = sweep(
        namespace=a.namespace,
        run=a.run,
        labels=labels,
        older_than_minutes=a.older_than_minutes,
        dry_run=a.dry_run,
    )
    print(f"hf sandbox sweep: found {counts['found']} canceled {counts['canceled']} failed {counts['failed']}")
    return 0 if not counts["failed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_main())
