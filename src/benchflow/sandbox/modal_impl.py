"""Native ModalSandbox, in RL-first terminology.

Uses Modal sandboxes for cloud-based container execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import os
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchflow._bfx import delivery as bfx_delivery
from benchflow._bfx.locate import SANDBOX_DIR as BFX_DIR
from benchflow._bfx.locate import SANDBOX_PATH as BFX_PATH
from benchflow._bfx.locate import BfxUnavailable
from benchflow._paths import iter_safe_tree
from benchflow._runtime_tree import KIT_BIN
from benchflow._utils.modal_gpu import modal_gpu_request
from benchflow.sandbox import _transport
from benchflow.sandbox._base import (
    STARTUP_ENV_ARGV,
    SYSTEM_PATH,
    SYSTEM_SHELL_EMPTY_VARS,
    SYSTEM_SHELL_UNSET_VARS,
    BaseSandbox,
    ExecResult,
    StartupNamesRefused,
    _root_exec_command,
    bash_argv_without_bash_env,
    bash_startup_name,
    capture_pid1_image_env,
    restore_bash_env,
    restore_image_env,
    startup_env_refusal,
    startup_name,
    with_system_path,
)
from benchflow.sandbox._dir_archive import (
    NO_TAR,
    NOT_ROOTS,
    extract_command,
    source_mode,
    write_archive,
)
from benchflow.sandbox._download import private_download
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._transport import (
    fit,
    prepare_stage_root,
    reader_cat,
    stage_folder,
    stage_proof,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths, SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

#: Inert Secret names for the values a Modal command re-exports by name
#:: no loader or shell reads them.
MODAL_CARRY_PREFIX = "_BF_MODAL_ENV_"


def _reexport(name: str) -> str:
    """Builtins: *name* back from its MODAL_CARRY_PREFIX copy, by name."""
    kept = MODAL_CARRY_PREFIX + name
    return f'{name}="${kept}"; export {name}; unset {kept}; '


# Modal refuses a sandbox name over 64 characters or outside [A-Za-z0-9._-].
_MODAL_NAME_MAX = 64
_MODAL_NAME_INVALID = re.compile(r"[^A-Za-z0-9._-]")


def modal_sandbox_name(session_id: str) -> str:
    """``session_id`` as a name Modal accepts for a sandbox.

    Rollout names are ``<task>__<id>`` and a separate verifier's adds
    ``-verifier``, so a long task name passes Modal's 64-character limit and
    the sandbox is refused at create. A shortened or cleaned name ends in a
    hash of the whole one, so two rollouts never share a name.
    """
    name = _MODAL_NAME_INVALID.sub("-", session_id)
    if name == session_id and 0 < len(name) <= _MODAL_NAME_MAX:
        return name
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:10]
    return f"{name[: _MODAL_NAME_MAX - len(digest) - 1]}-{digest}"


#: The most bytes Modal's ``Sandbox.filesystem`` moves in one file (its file
#: tool refuses more: "file is N bytes, which exceeds the 5368709120 byte
#: limit"). A workspace capture or transfer over it fails with that reason.
MODAL_FILE_TRANSFER_LIMIT = 5 * 1024**3


class ModalFileTooLargeError(RuntimeError):
    """A file is over :data:`MODAL_FILE_TRANSFER_LIMIT`: nothing of it was
    copied, and the caller's step (a capture, a transfer) fails with this."""


def modal_exec_timeout(timeout_sec: float | None) -> int | None:
    """``Sandbox.exec``'s ``timeout``: whole seconds, rounded up.

    Modal sends it as an integer field: a float such as a task's
    ``timeout_sec: 600.0`` fails with "'float' object cannot be interpreted
    as an integer".
    """
    if timeout_sec is None:
        return None
    return max(1, math.ceil(timeout_sec))


#: ``Sandbox.filesystem``'s errors and the builtin ones the legacy file API
#: raised, which BenchFlow's callers catch (FileNotFoundError, ...).
_FILESYSTEM_ERRORS: dict[str, type[OSError]] = {
    "SandboxFilesystemNotFoundError": FileNotFoundError,
    "SandboxFilesystemIsADirectoryError": IsADirectoryError,
    "SandboxFilesystemNotADirectoryError": NotADirectoryError,
    "SandboxFilesystemPermissionError": PermissionError,
}


@contextlib.contextmanager
def _builtin_filesystem_errors(path: str, direction: str) -> Iterator[None]:
    """Raise ``Sandbox.filesystem``'s path errors as the builtin ones, and its
    size refusal as :class:`ModalFileTooLargeError`, naming the limit."""
    try:
        yield
    except Exception as exc:
        if type(exc).__name__ == "SandboxFilesystemFileTooLargeError":
            raise ModalFileTooLargeError(
                f"{path} cannot be copied {direction}: it is over Modal's "
                f"per-file transfer limit of {MODAL_FILE_TRANSFER_LIMIT} bytes "
                f"(5 GiB), and no part of it was copied ({exc})"
            ) from exc
        builtin = _FILESYSTEM_ERRORS.get(type(exc).__name__)
        if builtin is None:
            raise
        raise builtin(str(exc)) from exc


class ModalSandbox(BaseSandbox):
    # a root agent here runs under the provider's own metadata
    # policy (the block's record says provider-managed).
    cloud_metadata_provider_managed = True
    # a Modal sandbox has no route to any cloud metadata endpoint.
    # Measured on 2026-10-26 from a public-network sandbox (gVisor), as root
    # and as uid 1000: every IPv4 endpoint the block covers timed out on
    # ports 80 and 443, every IPv6 one had no route, the metadata host names
    # did not resolve, and 1.1.1.1:443 connected. So no rule is needed (none
    # can go in: root has no NET_ADMIN); each trial proves it again with the
    # same connects and records not_reachable, or stops on any answer.
    cloud_metadata_unreachable = True

    agent_exec_shell = "/bin/bash"

    # exec passes the env as a Modal Secret, never in the command (exec below),
    # so another verifier user gets it privately (b173 PY8-MAIN-40).
    exec_env_is_private = True

    @classmethod
    def preflight(cls) -> None:
        modal_config = Path.home() / ".modal.toml"
        has_env_token = os.environ.get("MODAL_TOKEN_ID") and os.environ.get(
            "MODAL_TOKEN_SECRET"
        )
        if not modal_config.exists() and not has_env_token:
            raise SystemExit(
                "Modal requires authentication. Run 'modal token new' to set up "
                "credentials, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "environment variables."
            )

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image:
            return
        if not self._environment_definition_path.exists():
            raise FileNotFoundError(
                f"{self._environment_definition_path} not found. Please ensure the "
                "file exists."
            )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        secrets: list[str] | None = None,
        registry_secret: str | None = None,
        volumes: dict[str, str] | None = None,
        sandbox_timeout_secs: int = 60 * 60 * 24,
        sandbox_idle_timeout_secs: int | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            rollout_paths=rollout_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        # Lazy imports — Modal is optional
        self._image = None
        self._app = None
        self._sandbox = None
        self._secrets = secrets or []
        self._registry_secret = registry_secret
        self._volumes = volumes or {}
        self._sandbox_timeout = sandbox_timeout_secs
        self._sandbox_idle_timeout = sandbox_idle_timeout_secs

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(
        self,
        gpu_config: str | None,
        secrets_config: Sequence[object],
        volumes_config: Mapping[str, object],
    ) -> object:
        from modal import Sandbox

        return await Sandbox.create.aio(
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=modal_sandbox_name(self.session_id),
            cpu=self.task_env_config.cpus,
            memory=self.task_env_config.memory_mb,
            gpu=gpu_config,
            block_network=not self.task_env_config.allow_internet,
            secrets=secrets_config,
            volumes=volumes_config,
        )

    async def start(self, force_build: bool) -> None:
        gpu_config = modal_gpu_request(
            self.task_env_config.gpus, self.task_env_config.gpu_types
        )
        from modal import App, Image, Secret, Volume

        from benchflow.trust import state as trust_state

        trust_state.begin_fresh(self)

        docker_image = self.task_env_config.docker_image

        if docker_image:
            registry_secret = (
                Secret.from_name(self._registry_secret)
                if self._registry_secret
                else None
            )
            if ".dkr.ecr." in docker_image:
                self._image = Image.from_aws_ecr(
                    docker_image,
                    secret=registry_secret,
                )
            else:
                self._image = Image.from_registry(
                    docker_image,
                    secret=registry_secret,
                )
        else:
            self._image = Image.from_dockerfile(
                self._environment_definition_path,
                context_dir=self.environment_dir,
            )

        self._app = await App.lookup.aio(
            name="__benchflow__",
            create_if_missing=True,
        )

        secrets_config = [Secret.from_name(secret) for secret in self._secrets]
        volumes_config = {
            mount_path: Volume.from_name(volume_name)
            for mount_path, volume_name in self._volumes.items()
        }

        # bfx at /.bfx: a read-only Volume, set when the sandbox is created
        # (TRUSTED-RUNTIME.md §6). Until it passes a live run it is opt-in
        # (bfx_delivery.provider_enabled); without it the kit path stays.
        self.bfx_delivery = None
        bfx_volume = None
        if BFX_DIR not in volumes_config and bfx_delivery.provider_enabled("modal"):
            import modal

            try:
                bfx_volume = await asyncio.to_thread(bfx_delivery.modal_volume, modal)
            except BfxUnavailable as exc:
                self.logger.debug("bfx: not delivered (%s)", exc)
        if bfx_volume is not None:
            volumes_config[BFX_DIR] = bfx_volume

        self._sandbox = await self._create_sandbox(
            gpu_config=gpu_config,
            secrets_config=secrets_config,
            volumes_config=volumes_config,
        )
        await self._refuse_startup_names()

        await self._make_log_dirs()

        await exec_raw(
            self.exec, f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}"
        )
        if bfx_volume is not None:
            await self._check_bfx()

    async def _make_log_dirs(self) -> None:
        """The agent's and verifier's log folders, with their parents.

        ``Sandbox.filesystem`` (modal 1.6.2): Modal's servers refuse the
        legacy ``Sandbox.mkdir`` / ``open`` / ``ls`` with ``UnimplementedError``
        ("The legacy Sandbox filesystem API is no longer supported"), so every
        Modal rollout failed right after its sandbox started.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        for path in (SandboxPaths.agent_dir, SandboxPaths.verifier_dir):
            await self._sandbox.filesystem.make_directory.aio(str(path))

    async def _refuse_startup_names(self) -> None:
        """Refuse an image whose ENV sets bash's start-up names
        (``startup_name``): Modal starts every process with the image's
        environment, and the routes' ``/bin/bash`` would import them as root
        before its first command. Read by ``cat`` (argv, no shell) before
        any shell runs; the sandbox is ended on a refusal."""
        _transport.sent("Modal", list(STARTUP_ENV_ARGV))
        process = await self._sandbox.exec.aio(*STARTUP_ENV_ARGV, timeout=60)
        out = await process.stdout.read.aio()
        refusal = startup_env_refusal(out, await process.wait.aio(), "the Modal image")
        if refusal is not None:
            self.logger.error(refusal)
            try:
                await self._terminate_sandbox()
            except Exception:
                self.logger.warning("Could not end the refused Modal sandbox")
            else:
                self._sandbox = None
            raise StartupNamesRefused(refusal)

    async def _check_bfx(self) -> None:
        """The mounted bfx reports the pinned hash (argv exec, no shell,
        no environment); then it is this sandbox's helper."""
        try:
            _transport.sent("Modal", [BFX_PATH, "version"])
            process = await self._sandbox.exec.aio(BFX_PATH, "version", timeout=60)
            out = await process.stdout.read.aio()
            code = await process.wait.aio()
            if code != 0:
                raise BfxUnavailable(f"bfx version exited {code}")
            bfx_delivery.check_version(bfx_delivery.version_answer(out), "x86_64")
        except BfxUnavailable as exc:
            self.logger.warning("bfx: the mounted volume is not usable (%s)", exc)
            return
        self.bfx_delivery = bfx_delivery.VOLUME

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _terminate_sandbox(self) -> None:
        if self._sandbox:
            await self._sandbox.terminate.aio()

    async def stop(self, delete: bool) -> None:
        if not self._sandbox:
            return

        try:
            await self._terminate_sandbox()
            await self._sandbox.wait.aio(raise_on_termination=False)
        except Exception as e:
            self.logger.warning(f"Error terminating Modal sandbox: {e}")
        finally:
            self._sandbox = None
            self._app = None
            self._image = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # A file over Modal's transfer limit is refused again on a retry.
        retry=retry_if_not_exception_type(ModalFileTooLargeError),
        reraise=True,
    )
    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Streams the file in, as root, making missing parent folders
        # (Sandbox.filesystem: the legacy Sandbox.open is refused, see
        # _make_log_dirs).
        with _builtin_filesystem_errors(target_path, "into the Modal sandbox"):
            await self._sandbox.filesystem.copy_from_local.aio(
                str(source_path), target_path
            )
        await self._apply_upload_mode(target_path, mode)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        await exec_raw(
            self.exec, f"mkdir -p {shlex.quote(str(target_dir))}", user="root"
        )
        # One archive, whatever the file count (DAYTONA-GAPS rule 1): one
        # upload into a fresh folder of root's staging folder and one root
        # exec that proves its route root's alone and unpacks it, with the
        # source's permission bits, owned by root. Walk with
        # followlinks=False and skip symlinks (#411): a task or workspace
        # symlink under source_path must not exfiltrate host files into the
        # remote Modal sandbox.
        archive, count = await asyncio.to_thread(
            write_archive,
            source_path,
            mode=source_mode,
            context=f"modal upload_dir {source_path}",
        )
        try:
            if not count:
                return
            staged = f"{stage_folder()}/dir.tar"
            folder = shlex.quote(staged.rsplit("/", 1)[0])
            root = await self.exec_route("root", system_shell=True)
            preparation = prepare_stage_root(
                shell=root.shell, path=root.pin or SYSTEM_PATH
            )
            made = await exec_raw(
                self.exec,
                f"umask 077; {preparation} && mkdir {folder}",
                user="root",
                system_shell=True,
                timeout_sec=60,
            )
            if made.return_code != 0:
                raise RuntimeError("Could not make Modal's directory upload folder")
            try:
                await self.upload_file(archive, staged)
                result = await exec_raw(
                    self.exec,
                    extract_command(
                        staged,
                        str(target_dir),
                        stage_proof(
                            [staged], shell=root.shell, path=root.pin or SYSTEM_PATH
                        ),
                    ),
                    user="root",
                    system_shell=True,
                    timeout_sec=600,
                )
            except Exception:
                # The unpack removes its folder; an upload or exec that
                # raised did not get there (DAYTONA-GAPS review A L4).
                with contextlib.suppress(Exception):
                    await exec_raw(
                        self.exec,
                        f"rm -rf -- {folder}",
                        user="root",
                        system_shell=True,
                        timeout_sec=60,
                    )
                raise
        finally:
            archive.unlink(missing_ok=True)
        if result.return_code == NO_TAR:
            # A sandbox without tar: file by file, inside this primitive (the
            # guard admitted this call).
            for file_path in iter_safe_tree(
                source_path, context=f"modal upload_dir {source_path}"
            ):
                relative_path = file_path.relative_to(source_path).as_posix()
                target_file_path = str(PurePosixPath(target_dir) / relative_path)

                target_file_parent = str(PurePosixPath(target_file_path).parent)
                if target_file_parent != target_dir:
                    await exec_raw(
                        self.exec,
                        f"mkdir -p {shlex.quote(str(target_file_parent))}",
                        user="root",
                    )

                await self.upload_file(
                    file_path,
                    target_file_path,
                    mode=f"{file_path.stat().st_mode & 0o777:o}",
                )
        elif result.return_code != 0:
            raise RuntimeError(
                "Could not unpack the Modal directory upload"
                + (
                    ": its staging folder is not root's alone"
                    if result.return_code == NOT_ROOTS
                    else ""
                )
            )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        # A file over Modal's transfer limit is refused again on a retry.
        retry=retry_if_not_exception_type(ModalFileTooLargeError),
        reraise=True,
    )
    @private_download(directory=False)
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Streams into a temporary file beside the target (private_download's
        # staging folder), then renames it (Sandbox.filesystem: the legacy
        # Sandbox.open is refused, see _make_log_dirs).
        with _builtin_filesystem_errors(source_path, "out of the Modal sandbox"):
            await self._sandbox.filesystem.copy_to_local.aio(source_path, target_path)

    @private_download(directory=True)
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # NUL-framed names, carried as a receipt: a name may hold a newline.
        result = await receipt_exec(
            self, f"find {shlex.quote(source_dir)} -type f -print0"
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to list files in {source_dir}: {result.stderr}")

        file_paths = [p for p in (result.stdout or "").split("\0") if p]
        if not file_paths:
            return

        from benchflow._utils.archive_paths import ArchivePaths

        inventory = ArchivePaths()
        relative_paths = {
            remote: inventory.add(
                str(PurePosixPath(remote).relative_to(PurePosixPath(source_dir))),
                directory=False,
            )
            for remote in file_paths
        }

        sem = asyncio.Semaphore(5)

        async def _download_one(remote_path: str) -> None:
            async with sem:
                rel = relative_paths[remote_path]
                local_path = target_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self.download_file(remote_path, local_path)

        async with asyncio.TaskGroup() as tg:
            for p in file_paths:
                tg.create_task(_download_one(p))

    # is_dir / is_file: BaseSandbox's ``test -d`` / ``test -f`` through exec,
    # which honours ``user`` and refuses a service other than main; the
    # legacy Sandbox.ls they used is refused (see _make_log_dirs).

    @property
    def supports_system_shell(self) -> bool:
        return True

    @property
    def supports_runtime_kit(self) -> bool:
        return True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        system_shell: bool = False,
    ) -> ExecResult:
        """Run ``command`` in the sandbox.

        ``system_shell``: the loader variables the caller does not set are
        passed empty with the exec (Modal hands env over as the process's
        environment), and PATH is the system folders' before anything else
        runs (see :data:`benchflow.sandbox._base.SYSTEM_SHELL`).

        Every command, K1's agent bootstrap and its root staging included,
        is built on PT's route (C8) in :meth:`_dispatch_exec`; an agent
        command then also gets K1's barrier.
        """
        from benchflow.sandbox.agent_exec import AgentExecCommand

        agent_bootstrap = isinstance(command, AgentExecCommand)
        if service != "main":
            raise ValueError(
                f"Modal sandbox is single-container and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )

        requested_user = user
        user = self._resolve_user(user)
        route = await self.exec_route(requested_user, system_shell=system_shell)
        env = self._merge_env(
            env,
            requested_user,
            plumbing=route.pin is not None,
            explicit_root=route.explicit_root,
        )

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        if agent_bootstrap and env:
            from benchflow.sandbox.native_agent_env import uploaded_agent_env

            root_route = await self.exec_route("root", system_shell=True)

            async def root_exec(body: str) -> ExecResult:
                # Do not merge caller/persistent env into staging or cleanup:
                # root's plumbing route, no env (K1).
                return await self._dispatch_exec(
                    body,
                    route=root_route,
                    user=self._resolve_user("root"),
                    timeout_sec=30,
                    agent_bootstrap=True,
                )

            async with uploaded_agent_env(
                command, env, root_exec=root_exec, upload=self._exec_env_upload()
            ) as (_restored, opening):
                return await self._dispatch_exec(
                    command,
                    route=route,
                    cwd=cwd,
                    user=user,
                    timeout_sec=timeout_sec,
                    agent_bootstrap=True,
                    agent_env_open=opening,
                    agent_restore=True,
                )
        return await self._dispatch_exec(
            command,
            route=route,
            cwd=cwd,
            env=env,
            user=user,
            timeout_sec=timeout_sec,
            agent_bootstrap=agent_bootstrap,
        )

    async def _dispatch_exec(
        self,
        command: str,
        *,
        route: Any,
        agent_bootstrap: bool,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        agent_env_open: str = "",
        agent_restore: bool = False,
    ) -> ExecResult:
        """Start *command* on *route* and wait for it. *agent_restore*:
        *command* is a K1 agent command whose uploaded env the native
        decoder restores (``restore_command``) before it runs."""
        from modal import Secret

        from benchflow.sandbox.agent_exec import native_agent_argv

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # The programs this wrapping names by path: the protected kit's for
        # root once the sandbox has one (C8), else the image's system ones.
        su = f"{KIT_BIN}/su" if route.kit else "/bin/su"
        bash = route.bash if route.kit and route.bash.endswith("/bash") else "/bin/bash"
        # C8: Modal hands env over as the process's environment, so the
        # loader and start-up variables (the merged env's and the image's)
        # never go there: the process starts with the loader ones empty, and
        # the values the command should have are exported in its own shell
        # instead. A plumbing exec (route.pin) gets only the caller's values,
        # after its shell pinned PATH. Every other command gets them back
        # just before it runs: the image's BASH_ENV carried past the wrapping
        # bash (bash_argv_without_bash_env), its loader values copied from
        # PID 1 by the root shell that starts it, the merged env's over both.
        # bash's start-up names (startup_name) never reach the process env
        # either: the wrapping /bin/bash would import them as root before
        # its first command, and no shell can export a function's name back.
        startup = {
            k: v
            for k, v in (env or {}).items()
            if k in SYSTEM_SHELL_UNSET_VARS or startup_name(k) or k.startswith("LD_")
        }
        # No value is ever written into the command
        # text, which Modal runs as argv. The values the command's own shells
        # export (the start-up names, PATH, the image PATH) travel in the
        # Secret under inert MODAL_CARRY_PREFIX names, and the text re-exports
        # them by name.
        carried = {
            k: v
            for k, v in (env or {}).items()
            if (k in startup or k in ("PATH", "_BENCHFLOW_IMAGE_PATH"))
            and k.isidentifier()
            and not bash_startup_name(k)
        }
        exports = "".join(_reexport(k) for k in startup if k in carried)
        process_env = {k: v for k, v in (env or {}).items() if k not in startup}
        process_env.update({MODAL_CARRY_PREFIX + k: v for k, v in carried.items()})
        # The names a pinned shell carries across its env -i: the process
        # env before its empty start-up names are added below.
        keep = dict(process_env)

        def argv_for(command: str) -> list[str]:
            if agent_restore:
                from benchflow.sandbox.native_agent_env import restore_command

                command = restore_command(command)
            if route.pin is None:
                command = (
                    restore_bash_env(route)
                    + restore_image_env(
                        env or (),
                        tools=route.tools if route.kit and route.root else None,
                    )
                    + exports
                    + command
                )
            else:
                command = exports + command
            if user is not None:
                from benchflow.sandbox._su import su_command

                # K1's agent bootstrap starts POSIX sh with absolute tools (a
                # minimal image may have no bash), the route's su either way.
                command = su_command(
                    command,
                    user,
                    shell="/bin/sh" if agent_bootstrap else bash,
                    su=su,
                    trusted_tools=agent_bootstrap,
                )

            # Modal supplies env directly rather than through the env-file
            # wrapper. Keep the verifier's explicit validated PATH after
            # trusted bootstrap.
            if "PATH" in carried:
                command = _reexport("PATH") + command
            # Likewise the task's PATH a root command carries as data
            # (_merge_env), which the bootstrap's capture would otherwise
            # replace.
            if "_BENCHFLOW_IMAGE_PATH" in carried:
                command = _reexport("_BENCHFLOW_IMAGE_PATH") + command
            if route.pin is not None:
                command = with_system_path(command, route.pin, keep=keep)
                argv = [route.shell, "-c", command]
            else:
                command = capture_pid1_image_env(route.tools) + _root_exec_command(
                    command, route.explicit_root, route.root_path
                )
                argv = bash_argv_without_bash_env(command, bash)
            if agent_bootstrap:
                # K1's barrier around the routed command: no_new_privs before
                # the first shell; the private payload is opened and unlinked
                # inside it, before su; only the selected account's fixed
                # decoder reads its descriptor.
                argv = native_agent_argv(agent_env_open + command)
            return argv

        if route.pin is not None:
            process_env.update({name: "" for name in SYSTEM_SHELL_EMPTY_VARS})
        else:
            # The image's BASH_ENV stays in the process env for the wrapping
            # bash to carry (it reads no file in POSIX mode).
            process_env.update(
                {name: "" for name in SYSTEM_SHELL_EMPTY_VARS if name != "BASH_ENV"}
            )
        # The transport contract (benchflow.sandbox._transport): a command
        # over the bound is the process's stdin and the reader runs it, in
        # the same shell, as the same user. A K1 agent command's payload is
        # the agent's own command, its barrier rebuilt around the reader.
        agent = command if hasattr(command, "agent_target") else None
        fitted = fit(argv_for, command, cat=reader_cat(route), agent=agent)
        _transport.sent("Modal", fitted.sent)
        env = process_env
        process = await self._sandbox.exec.aio(
            *fitted.sent,
            workdir=cwd,
            secrets=[Secret.from_dict(env)] if env else [],
            timeout=modal_exec_timeout(timeout_sec),
        )
        if fitted.payload is not None:
            process.stdin.write(fitted.payload)
            process.stdin.write_eof()
            await process.stdin.drain.aio()

        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()

        return_code = await process.wait.aio()

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        """The agent's stdio over ``Sandbox.exec`` streams (:class:`ModalProcess`):
        root's launch bootstrap as on Docker, the agent's environment as the
        first stdin line."""
        from benchflow.sandbox.process import ModalProcess

        return ModalProcess.from_sandbox_env(self)

    async def attach(self) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        os.execvp(
            "modal",
            [
                "modal",
                "shell",
                self._sandbox.object_id,
            ],
        )
