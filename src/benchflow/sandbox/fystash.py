"""FystashSandbox — BenchFlow Sandbox backend over the public Fystash API.

Maps:
  start/stop      → compile → Branch → Plan → approve → apply → stop
  exec            → Run.exec (catalog execProcess)
  upload/download → exec + base64 (no new catalog op)
  snapshot/restore → same-Run /workspace tar (not a Firecracker VM snapshot)

A new Branch pays Drive initialize (~2 min when healthy). Later starts MUST
reuse the same branch_id. Reusing only revision_id and minting a new Branch
is still a cold Drive init.

Residuals (honest, do not paper over):
  - ImageBuilder does not build arbitrary guest Dockerfiles. Tasks must pin
    ``python-agent`` / ``browser-agent`` via ``environment/environment.yaml``
    or ``environment/sandbox.yaml``.
  - Single workspace only — ``service != main`` is rejected.
  - No DinD, no GPU.
  - ``exec(..., user=)`` is ignored (catalog actor is ``agent``).
  - ``expose_ports`` / Preview grants are unwired. ``host`` is ``127.0.0.1``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shlex
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any

from benchflow.sandbox._base import BaseSandbox, ExecResult, wrap_command_with_env_file
from benchflow.sandbox.protocol import (
    ImageConfig,
    ImageRef,
    SandboxImage,
    SandboxSnapshotNotSupported,
    SandboxStartupError,
)
from benchflow.task.paths import SandboxPaths

GUEST_WORKSPACE = "/workspace"
_SNAP_DIR = "/tmp/fystash-benchflow-snaps"
_MANIFEST_NAMES = ("environment.yaml", "sandbox.yaml")


def _load_fystash_sdk() -> tuple[Any, Any, Any]:
    """Import the optional ``fystash`` extra. Call at env-selection time."""

    from fystash import Fystash, Plan, SandboxHandle

    return Fystash, Plan, SandboxHandle


def _output_text(attempt: Any, stream: str) -> str:
    rec = attempt if isinstance(attempt, dict) else {}
    chunks: list[str] = []
    for item in rec.get("output") or []:
        if isinstance(item, dict) and item.get("stream") == stream:
            chunks.append(str(item.get("text") or ""))
    if chunks:
        return "".join(chunks)
    if stream == "stdout":
        return str(rec.get("stdout") or "")
    return str(rec.get("stderr") or "")


def _exit_code(attempt: Any) -> int:
    rec = attempt if isinstance(attempt, dict) else {}
    raw = rec.get("exitCode", rec.get("exit_code"))
    if raw is None:
        state = str(rec.get("state") or rec.get("outcome") or "").lower()
        if state in {"succeeded", "ok", "completed"}:
            return 0
        return 1
    return int(raw)


def _guest_path(dst: str) -> str:
    if dst.startswith("/"):
        return dst
    return f"{GUEST_WORKSPACE.rstrip('/')}/{dst.lstrip('/')}"


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _reject_non_main(service: str) -> None:
    if service and service != "main":
        raise ValueError(
            f"Fystash sandbox is single-workspace and cannot target "
            f"service {service!r}. Multi-container (vulhub-style) tasks "
            "require the Docker sandbox."
        )


def _branch_cache_path() -> Path:
    override = os.environ.get("FYSTASH_BENCHFLOW_BRANCH_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".fystash" / "benchflow-branch.json"


def _load_branch_cache(path: Path) -> dict[str, str]:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _save_branch_cache(path: Path, *, revision_id: str, branch_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"revision_id": revision_id, "branch_id": branch_id}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _branch_usable(fy: Any, branch_id: str) -> bool:
    try:
        rec = fy.client.get_branch(branch_id)
    except Exception:
        return False
    if not isinstance(rec, dict):
        rec = {}
    state = str(rec.get("state") or rec.get("status") or "").lower()
    return state not in {"failed", "deleted", "closed", "error"}


def _has_cli_credentials() -> bool:
    if os.environ.get("FYSTASH_ACCESS_TOKEN") and os.environ.get("FYSTASH_PROJECT_ID"):
        return True
    config_dir = Path(
        os.environ.get("FYSTASH_CONFIG_DIR") or (Path.home() / ".fystash")
    )
    if (config_dir / "credentials.json").is_file():
        return True
    return Path(".fystash/project.json").is_file()


class FystashImageBuilder:
    """Honest residual: Fystash pins python-agent / browser-agent bases."""

    async def build(self, config: ImageConfig) -> ImageRef:
        raise SandboxSnapshotNotSupported(
            "Fystash ImageBuilder does not build guest Dockerfiles; "
            "pin python-agent via environment/environment.yaml or "
            "environment/sandbox.yaml."
        )

    async def cached(self, config: ImageConfig) -> ImageRef | None:
        return None


class FystashSandbox(BaseSandbox):
    @classmethod
    def preflight(cls) -> None:
        if _has_cli_credentials():
            return
        raise SystemExit(
            "Fystash requires FYSTASH_ACCESS_TOKEN and FYSTASH_PROJECT_ID, "
            "or `fystash login` credentials under ~/.fystash/credentials.json. "
            "FYSTASH_API_URL defaults to https://api.fystash.ai/v1."
        )

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_snapshot(self) -> bool:
        """Same-Run workspace rollback, not a Firecracker / Class C VM snapshot."""
        return True

    @property
    def sandbox_id(self) -> str | None:
        if self._handle is None:
            return None
        return str(self._handle.id)

    @property
    def host(self) -> str:
        return "127.0.0.1"

    def _manifest_path(self) -> Path:
        for name in _MANIFEST_NAMES:
            candidate = self.environment_dir / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{self.environment_dir}/environment.yaml (or sandbox.yaml) not found. "
            "Fystash does not build arbitrary guest Dockerfiles; pin python-agent "
            "in a Fystash Environment manifest."
        )

    def _validate_definition(self) -> None:
        self._manifest_path()

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._handle = None
        self._snaps: dict[str, str] = {}
        self._revision_id = os.environ.get("FYSTASH_REVISION_ID") or ""
        self._branch_id = os.environ.get("FYSTASH_BRANCH_ID") or ""
        self._plan_digest = ""
        self._approver = os.environ.get("FYSTASH_BENCHFLOW_APPROVER") or "benchflow"
        self._persist_branch = os.environ.get(
            "FYSTASH_BENCHFLOW_PERSIST_BRANCH", "1"
        ) not in {"0", "false", "no"}
        self._branch_cache = _branch_cache_path()
        self.last_start: dict[str, int | str] = {}
        super().__init__(*args, **kwargs)
        if self._persist_branch and not self._revision_id and not self._branch_id:
            cached = _load_branch_cache(self._branch_cache)
            self._revision_id = str(cached.get("revision_id") or "")
            self._branch_id = str(cached.get("branch_id") or "")

    def _client(self) -> Any:
        Fystash, _, _ = _load_fystash_sdk()
        return Fystash.from_cli_context()

    async def start(self, force_build: bool) -> None:
        if self._handle is not None:
            return
        if force_build:
            self._revision_id = ""
        _Fystash, Plan, SandboxHandle = _load_fystash_sdk()
        fy = self._client()
        timings: dict[str, int | str] = {}
        wall0 = time.monotonic()
        try:
            if not self._revision_id:
                t = time.monotonic()
                rev = await asyncio.to_thread(
                    fy.environments.compile_file, str(self._manifest_path())
                )
                self._revision_id = rev.id
                timings["compileMs"] = int((time.monotonic() - t) * 1000)
            else:
                timings["compileMs"] = 0

            if self._branch_id and not await asyncio.to_thread(
                _branch_usable, fy, self._branch_id
            ):
                self._branch_id = ""
                self._plan_digest = ""
            if not self._branch_id:
                t = time.monotonic()
                name = f"benchflow-{uuid.uuid4().hex[:10]}"
                br = await asyncio.to_thread(
                    fy.branches.create, revision=self._revision_id, name=name
                )
                self._branch_id = br.id
                timings["branchCreateMs"] = int((time.monotonic() - t) * 1000)
                timings["branchReused"] = "false"
            else:
                timings["branchCreateMs"] = 0
                timings["branchReused"] = "true"

            t = time.monotonic()
            plan = await asyncio.to_thread(
                fy.runs.plan, revision=self._revision_id, branch=self._branch_id
            )
            self._plan_digest = plan.digest
            timings["planMs"] = int((time.monotonic() - t) * 1000)

            t = time.monotonic()
            await asyncio.to_thread(
                fy.client.approve_plan, self._plan_digest, self._approver
            )
            timings["approveMs"] = int((time.monotonic() - t) * 1000)

            t = time.monotonic()

            def _apply_ready() -> Any:
                run = Plan(
                    fy,
                    {"digest": self._plan_digest},
                    {
                        "revision_id": self._revision_id,
                        "branch_id": self._branch_id,
                    },
                ).apply()
                run.wait_ready()
                return SandboxHandle(
                    run,
                    revision_id=self._revision_id,
                    branch_id=self._branch_id,
                    plan_digest=self._plan_digest,
                )

            self._handle = await asyncio.to_thread(_apply_ready)
            timings["applyReadyMs"] = int((time.monotonic() - t) * 1000)
        except Exception as exc:
            raise SandboxStartupError(str(exc)) from exc

        timings["revisionId"] = self._revision_id
        timings["branchId"] = self._branch_id
        timings["planDigest"] = self._plan_digest
        timings["wallMs"] = int((time.monotonic() - wall0) * 1000)
        self.last_start = timings
        if self._persist_branch:
            _save_branch_cache(
                self._branch_cache,
                revision_id=self._revision_id,
                branch_id=self._branch_id,
            )
        await self.exec(
            f"mkdir -p {GUEST_WORKSPACE} {_SNAP_DIR} "
            f"{SandboxPaths.agent_dir} {SandboxPaths.verifier_dir} "
            f"{SandboxPaths.verifier_code_dir}",
            timeout_sec=30,
        )

    async def stop(self, delete: bool) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        if delete:
            try:
                await asyncio.to_thread(handle.stop, "cancelled")
            except Exception as exc:
                self.logger.warning("Error stopping Fystash run: %s", exc)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        _reject_non_main(service)
        if self._handle is None:
            raise SandboxStartupError("sandbox is not started")
        # Residual: BenchFlow user= is ignored. Catalog execProcess uses actor=agent.
        _ = user
        timeout = 30 if timeout_sec is None else int(timeout_sec)
        wrapped = command
        if cwd:
            wrapped = f"cd {shlex.quote(cwd)} && {wrapped}"
        merged = self._merge_env(env)
        if merged:
            wrapped = wrap_command_with_env_file(
                merged, wrapped, env_path_prefix="/tmp/fystash-bf-env-"
            )
        quoted = _shell_quote(wrapped)
        script = (
            f"if command -v timeout >/dev/null 2>&1; then "
            f"timeout {timeout} /bin/sh -c {quoted}; "
            f"else /bin/sh -c {quoted}; fi"
        )
        try:
            attempt = await asyncio.to_thread(
                self._handle.exec, ["/bin/sh", "-c", script]
            )
        except Exception as exc:
            return ExecResult(return_code=1, stdout="", stderr=str(exc)[:800])
        return ExecResult(
            return_code=_exit_code(attempt),
            stdout=_output_text(attempt, "stdout"),
            stderr=_output_text(attempt, "stderr"),
        )

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        data = Path(source_path).read_bytes()
        guest = _guest_path(target_path)
        b64 = base64.b64encode(data).decode("ascii")
        parent = str(Path(guest).parent)
        chmod = f" && chmod {mode} {_shell_quote(guest)}" if mode else ""
        script = (
            f"mkdir -p {_shell_quote(parent)} && "
            f"python3 -c {_shell_quote('import base64,sys; open(sys.argv[1],"wb").write(base64.b64decode(sys.argv[2]))')} "
            f"{_shell_quote(guest)} {_shell_quote(b64)}"
            f"{chmod}"
        )
        result = await self.exec(script, timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"upload_file failed rc={result.return_code} {(result.stderr or '')[:240]}"
            )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        _reject_non_main(service)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(Path(source_dir), arcname=".")
        guest = _guest_path(target_dir)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        script = (
            f"mkdir -p {_shell_quote(guest)} && "
            f"python3 -c {_shell_quote('import base64,sys,tarfile,io; tarfile.open(fileobj=io.BytesIO(base64.b64decode(sys.argv[1])), mode="r:gz").extractall(sys.argv[2])')} "
            f"{_shell_quote(b64)} {_shell_quote(guest)}"
        )
        result = await self.exec(script, timeout_sec=180)
        if result.return_code != 0:
            raise RuntimeError(
                f"upload_dir failed rc={result.return_code} {(result.stderr or '')[:240]}"
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        guest = _guest_path(source_path)
        script = (
            f"python3 -c {_shell_quote('import base64,sys; print(base64.b64encode(open(sys.argv[1],"rb").read()).decode())')} "
            f"{_shell_quote(guest)}"
        )
        result = await self.exec(script, timeout_sec=120)
        if result.return_code != 0:
            raise FileNotFoundError(f"{guest}: {(result.stderr or '')[:240]}")
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode((result.stdout or "").strip()))

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        _reject_non_main(service)
        guest = _guest_path(source_dir)
        script = (
            f"python3 -c {_shell_quote('import base64,io,sys,tarfile; buf=io.BytesIO(); t=tarfile.open(fileobj=buf, mode="w:gz"); t.add(sys.argv[1], arcname="."); t.close(); print(base64.b64encode(buf.getvalue()).decode())')} "
            f"{_shell_quote(guest)}"
        )
        result = await self.exec(script, timeout_sec=180)
        if result.return_code != 0:
            raise FileNotFoundError(f"{guest}: {(result.stderr or '')[:240]}")
        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode((result.stdout or "").strip())
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            tar.extractall(out)

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        ref = name or f"snap-{uuid.uuid4().hex[:12]}"
        guest = f"{_SNAP_DIR}/{ref}.tgz"
        result = await self.exec(
            f"mkdir -p {_SNAP_DIR} && "
            f"tar czf {_shell_quote(guest)} -C {GUEST_WORKSPACE} "
            f"--exclude .fystash --exclude .fystash/* .",
            timeout_sec=45,
        )
        if result.return_code != 0:
            raise RuntimeError(f"snapshot failed: {(result.stderr or '')[:240]}")
        self._snaps[ref] = guest
        return SandboxImage(
            provider="fystash",
            ref=ref,
            meta={"run": self.sandbox_id or "", "path": guest},
        )

    async def restore(self, image: SandboxImage) -> None:
        if image.provider != "fystash":
            raise SandboxSnapshotNotSupported(
                f"cannot restore provider={image.provider}"
            )
        guest = self._snaps.get(image.ref) or image.meta.get("path")
        if not guest:
            raise SandboxSnapshotNotSupported(f"unknown snapshot ref {image.ref}")
        result = await self.exec(
            f"find {GUEST_WORKSPACE} -mindepth 1 -maxdepth 1 ! -name '.fystash' "
            f"-exec rm -rf {{}} \\; && "
            f"tar xzf {_shell_quote(guest)} -C {GUEST_WORKSPACE}",
            timeout_sec=45,
        )
        if result.return_code != 0:
            raise RuntimeError(f"restore failed: {(result.stderr or '')[:240]}")
