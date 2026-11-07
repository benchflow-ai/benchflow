"""Daytona implementation strategies — base interface and direct single-container.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam: the strategy
base class ``_DaytonaStrategy`` and the single-container ``_DaytonaDirect``
strategy. The Docker-in-Docker compose strategy ``_DaytonaDinD`` lives in the
sibling ``daytona_dind`` module (it subclasses ``_DaytonaStrategy`` from here).
All three names are re-exported from ``benchflow.sandbox.daytona`` so existing
imports such as ``from benchflow.sandbox.daytona import _DaytonaDirect`` keep
working unchanged.

The optional Daytona SDK handles (``Resources``, ``Image``,
``CreateSandboxFromImageParams`` …) and the ``DaytonaClientManager`` singleton
live on ``benchflow.sandbox.daytona`` — they are materialized there lazily by
``_load_daytona_sdk`` (#358). The ``start`` and ``restore`` paths reach them
through a local import of that module so the lazy-load / monkeypatch contract
is unchanged.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import json
import os
import shlex
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.sandbox import daytona_content_snapshots as content_snapshots
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    ExecResult,
    ExecRoute,
    system_shell_route,
)
from benchflow.sandbox._exec_merge import start_with_identity_probe
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._snapshot_credentials import (
    AsUserResult,
    StashedCredential,
    put_back_credentials,
    scrub_credentials,
)
from benchflow.sandbox.daytona_delete import delete_sandbox
from benchflow.sandbox.daytona_layers import (
    InstallLayerPlan,
    LayeredRecipe,
    build_record_outcome,
    claim_layer_build,
    disk_gib,
    layer_build_failed,
    layer_build_finished,
    layered_recipe,
)
from benchflow.sandbox.daytona_pty import (
    _exec_failure_output,
    _reject_non_main_service,
    stamp_transient_transport,
)
from benchflow.sandbox.daytona_reaper import (
    _benchflow_owned_labels,
    benchflow_snapshot_name,
)
from benchflow.sandbox.protocol import (
    ExecTimeoutError,
    SandboxImage,
    SandboxSnapshotNotSupported,
)
from benchflow.task.paths import SandboxPaths

if TYPE_CHECKING:
    from benchflow.sandbox.daytona import DaytonaSandbox

# ``_SandboxParams`` mirrors the façade alias: the SDK types are loaded lazily
# (issue #358), so concrete sandbox params are typed as ``Any`` here — callers
# build them inside methods that have already called ``_load_daytona_sdk()``.
_SandboxParams = Any

# Waits (seconds) before each retry of a deferred snapshot delete at stop():
# sandbox deletion is asynchronous on the provider, so the snapshot can stay
# "in use" for a few seconds after the sandbox created from it was deleted.
_DELETE_RETRY_DELAYS = (3, 10)

# Waits (seconds) before each new try of an upload's destination mkdir that
# timed out: a loaded sandbox can answer it late, which failed the launch
# before the agent ran. The mkdir is idempotent.
_UPLOAD_PREP_RETRY_DELAYS = (15, 15)


async def _delete_provider_snapshot(ref: str) -> None:
    """Delete the Daytona snapshot named ``ref``; a missing one is fine."""
    from benchflow.sandbox import daytona as _sdk

    _sdk._load_daytona_sdk()
    manager = await _sdk.DaytonaClientManager.get_instance()
    client = await manager.get_client()
    try:
        snapshot = await client.snapshot.get(ref)
    except Exception as exc:
        if (
            isinstance(exc, _sdk.DaytonaNotFoundError)
            or "not found" in str(exc).lower()
        ):
            return
        raise
    await client.snapshot.delete(snapshot)


#: Daytona's toolbox daemon (root) writes its log here, readable by every
#: user, with its routes and the client's address in it; a model agent read
#: it while looking for a way to root (Rust 57213637, decision G-18). Before
#: the agent starts, root makes it root's and readable by root only, then
#: checks that it is; the daemon keeps writing through the file it has open.
#: Mode 0600 alone protects the file's owner, who need not be root, so the
#: owner is set and checked too. A link, or anything but a
#: regular file, is refused: chmod and chown would follow a link, and its
#: target may be readable. A compose task's daemon log is on the DinD VM, out
#: of the agent's reach.
DAYTONA_DAEMON_LOG = "/tmp/daytona-daemon.log"


def daemon_log_root_only_cmd(path: str = DAYTONA_DAEMON_LOG) -> str:
    """Make ``path`` root's and readable by root only, and check that it is.

    Prints ``BF daemon-log root-only`` and exits 0 when the file is now root's
    with mode 0600 (no ACL), ``BF daemon-log absent`` and exits 0 when there is
    no file; otherwise prints ``BF daemon-log <why>`` (``not-root``, ``link``,
    ``not-a-file``, ``unprotected``) and exits 1. Reads the result with ``ls``
    (busybox has it; ``stat -c`` is not on every image).
    """
    return (
        f"f={shlex.quote(path)}; "
        'if [ "$(id -u)" != 0 ]; then echo "BF daemon-log not-root"; exit 1; fi; '
        'if [ -L "$f" ]; then echo "BF daemon-log link"; exit 1; fi; '
        'if [ ! -e "$f" ]; then echo "BF daemon-log absent"; exit 0; fi; '
        'if [ ! -f "$f" ]; then echo "BF daemon-log not-a-file"; exit 1; fi; '
        'chown 0:0 "$f" 2>/dev/null && chmod 600 "$f" 2>/dev/null || '
        '{ echo "BF daemon-log unprotected"; exit 1; }; '
        'set -- $(ls -ln "$f" 2>/dev/null); '
        'case "$1 $3" in "-rw------- 0"|"-rw-------. 0") '
        'echo "BF daemon-log root-only"; exit 0 ;; esac; '
        'echo "BF daemon-log unprotected"; exit 1'
    )


DAEMON_LOG_ROOT_ONLY_CMD = daemon_log_root_only_cmd()

_DAEMON_LOG_OK = ("BF daemon-log root-only", "BF daemon-log absent")


def check_daemon_log_result(result: Any, path: str = DAYTONA_DAEMON_LOG) -> None:
    """Raise unless the daemon-log step exited 0 and printed an accepted line.

    A missing exit status, a missing line or a refusal means the agent might
    read the log, so the sandbox is not used.
    """
    code = getattr(result, "return_code", None)
    stdout = getattr(result, "stdout", None)
    lines = stdout.split("\n") if isinstance(stdout, str) else []
    marks = [ln.strip() for ln in lines if ln.startswith("BF daemon-log ")]
    if (
        isinstance(code, int)
        and not isinstance(code, bool)
        and code == 0
        and len(marks) == 1
        and marks[0] in _DAEMON_LOG_OK
    ):
        return
    why = marks[-1].removeprefix("BF daemon-log ") if marks else "no result"
    raise RuntimeError(
        f"Daytona's daemon log {path} could not be made root-only ({why}, "
        f"exit status {code!r}); the agent could read it, so the sandbox is "
        "not used."
    )


class _DaytonaStrategy:
    """Base for Daytona implementation strategies."""

    # Strategies declare whether they can satisfy container-level snapshots;
    # the DaytonaSandbox wrapper forwards the question to its strategy so the
    # capability tracks the active mode (direct vs. DinD/compose) — see #384.
    supports_snapshot: bool = False

    def __init__(self, env: DaytonaSandbox) -> None:
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult: ...

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult:
        """Run a child-free command, cleaning its provider session if supported."""
        return await exec_raw(
            self.exec,
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            service=service,
            route=route,
        )

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None: ...

    @abstractmethod
    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None: ...

    @abstractmethod
    async def is_dir(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool: ...

    @abstractmethod
    async def is_file(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool: ...

    @abstractmethod
    async def services(self) -> list[str]: ...

    @abstractmethod
    async def attach(self) -> None: ...

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Capture a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots."
        )

    async def restore(self, image: SandboxImage) -> None:
        """Restore from a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level restore."
        )

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        """Delete a provider-level snapshot. Default: not supported."""
        raise SandboxSnapshotNotSupported(
            f"{type(self).__name__} does not support container-level snapshots."
        )


async def _deliver_bfx(env: Any) -> None:
    """bfx by upload (root's 0555 /.bfx), as root on the system shell,
    before the agent's first byte (TRUSTED-RUNTIME.md §6). A sandbox whose
    snapshot already has /.bfx gets none and keeps the kit path."""
    from benchflow._bfx import delivery as bfx_delivery

    if not bfx_delivery.provider_enabled("daytona"):
        env.bfx_delivery = None
        return
    route = await system_shell_route(env)

    async def run_root(command: str) -> tuple[int, str]:
        result = await env._sandbox_exec(
            command, user="root", timeout_sec=60, route=route
        )
        out = result.stdout or ""
        if result.return_code != 0:
            out += result.stderr or ""
        return result.return_code, out

    await bfx_delivery.deliver_upload(env, "daytona", env._sdk_upload_file, run_root)


class _DaytonaDirect(_DaytonaStrategy):
    """Direct sandbox strategy — single-container behavior."""

    # Daytona ships a native sandbox-snapshot API on AsyncSandbox; the direct
    # strategy uses it for the container layer of Branch (#384).
    supports_snapshot: bool = True

    def __init__(self, env: DaytonaSandbox) -> None:
        super().__init__(env)
        # Snapshots still in use when their fork finished; stop() deletes them.
        self._deferred_snapshot_refs: set[str] = set()
        # Credential files scrubbed out of each snapshot, held in host memory
        # so restore() can put them back (see _snapshot_credentials).
        self._snapshot_credentials: dict[str, list[StashedCredential]] = {}

    def _credential_ops(self) -> _DaytonaCredentialOps:
        return _DaytonaCredentialOps(self._env)

    async def start(self, force_build: bool) -> None:
        from benchflow.sandbox import daytona as _sdk
        from benchflow.sandbox.setup import _DAYTONA_MAX_STORAGE_MB

        env = self._env
        plan: InstallLayerPlan | None = getattr(env, "_install_layers", None)
        footprint_mb = plan.footprint_mb if plan is not None else 0
        # #1169: the disk holds the task's storage_mb plus BenchFlow's own
        # proxy and agent installs (daytona_layers.disk_gib).
        disk = disk_gib(
            env.task_env_config.storage_mb, footprint_mb, _DAYTONA_MAX_STORAGE_MB
        )
        resources = _sdk.Resources(
            cpu=env.task_env_config.cpus,
            memory=env.task_env_config.memory_mb // 1024,
            disk=disk,
        )
        env._install_layer_keys = {}
        env._install_layers_record = (
            None
            if plan is None
            else {
                "backend": "daytona",
                "mode": "fresh" if plan.fresh else "layers" if plan.layers else "none",
                "storage_mb": env.task_env_config.storage_mb,
                "footprint_mb": footprint_mb,
                "disk_gib": disk,
                "layers": {},
                "snapshot": "off",
                "snapshot_reason": "BENCHFLOW_DAYTONA_SNAPSHOTS is not content",
                "snapshot_name": None,
            }
        )

        env._client_manager = await _sdk.DaytonaClientManager.get_instance()
        daytona = await env._client_manager.get_client()

        snapshot_name: str | None = None
        snapshot_exists = False

        if env._snapshot_template_name:
            snapshot_name = env._snapshot_template_name.format(
                name=env.environment_name
            )
            try:
                snapshot = await daytona.snapshot.get(snapshot_name)
                if snapshot.state == _sdk.SnapshotState.ACTIVE:
                    snapshot_exists = True
            except Exception:
                snapshot_exists = False

        if snapshot_exists and force_build:
            env.logger.warning(
                "Snapshot template specified but force_build is True. "
                "Snapshot will be used instead of building from scratch."
            )

        params: _SandboxParams
        # Set when the recipe carries install layers: the task image alone,
        # for the one retry after a layer that does not build.
        plain_params: _SandboxParams | None = None
        layered = LayeredRecipe()
        build_allowance_sec = 0
        # This trial builds the layers for the run's later trials (M4).
        layer_claim: str | None = None
        claim_keys: dict[str, str] = {}

        start_ref = getattr(env, "_start_snapshot_ref", None)
        if (start_ref or snapshot_exists) and env._install_layers_record is not None:
            # A snapshot already holds what was installed before it was taken.
            env._install_layers_record["mode"] = "snapshot"
            env._install_layers_record["snapshot_reason"] = (
                "starts from a branch checkpoint"
                if start_ref
                else "starts from an explicit snapshot template"
            )
        # Set when the sandbox is to come from a content snapshot (opt-in):
        # the layered image's own params, for one retry if that create fails.
        snapshot_image_params: _SandboxParams | None = None
        if start_ref:
            # A branch child: the checkpoint, never the task image.
            env.logger.debug(f"Starting from branch snapshot: {start_ref}")
            params = _sdk.CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=start_ref,
                network_block_all=env._network_block_all,
                labels=_benchflow_owned_labels(),
            )
        elif snapshot_exists and snapshot_name:
            env.logger.debug(f"Using snapshot: {snapshot_name}")
            params = _sdk.CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=snapshot_name,
                network_block_all=env._network_block_all,
                labels=_benchflow_owned_labels(),
            )
        else:
            # Both non-snapshot paths build identical CreateSandboxFromImageParams
            # (including the benchflow.managed ownership label); only the image
            # source and log line differ between build-from-Dockerfile and
            # prebuilt-image.
            from_dockerfile = force_build or not env.task_env_config.docker_image
            if from_dockerfile:
                env.logger.debug(f"Building environment from {env._dockerfile_path}")
            else:
                env.logger.debug(
                    f"Using prebuilt image: {env.task_env_config.docker_image}"
                )

            def task_image() -> Any:
                if from_dockerfile:
                    return _sdk.Image.from_dockerfile(env._dockerfile_path)
                return _sdk.Image.base(env.task_env_config.docker_image)

            def image_params(image: Any) -> _SandboxParams:
                return _sdk.CreateSandboxFromImageParams(
                    image=image,
                    auto_delete_interval=env._auto_delete_interval,
                    auto_stop_interval=env._auto_stop_interval,
                    resources=resources,
                    network_block_all=env._network_block_all,
                    labels=_benchflow_owned_labels(),
                )

            image = task_image()
            base_recipe = ""
            if plan is not None and plan.layers:
                base_recipe = image.dockerfile()
                layered = layered_recipe(plan, base_recipe)
                failed_before = layer_build_failed(layered.keys)
                if (
                    layered.lines
                    and not failed_before
                    and plan.only_if_reused
                    and not content_snapshots.enabled()
                ):
                    claim_keys = dict(layered.keys)
                    layer_claim = claim_layer_build(claim_keys)
                    if layer_claim is None:
                        # The small guard install is cached for single trials
                        # too; only the proxy and agent need planned reuse.
                        plan = replace(plan, proxy=None, agent=None)
                        layered = layered_recipe(plan, base_recipe)
                        if env._install_layers_record is not None:
                            env._install_layers_record["mode"] = (
                                "layers" if layered.lines else "fresh"
                            )
                            env._install_layers_record["fresh_reason"] = (
                                "not built in this run, and the run starts no "
                                "other trial of this image"
                            )
                if failed_before or layer_build_failed(layered.keys):
                    layered = LayeredRecipe()
                    if env._install_layers_record is not None:
                        env._install_layers_record["mode"] = "fresh"
                        env._install_layers_record["fallback"] = (
                            "layer build failed earlier in this process"
                        )
            if layered.lines:
                # The proxy and agent installs as cached layers on top of the
                # task image (daytona_layers); Daytona reuses an unchanged
                # recipe's build for 24 h on the runner that built it.
                plain_params = image_params(task_image())
                lines = layered.lines
                if base_recipe and not base_recipe.endswith("\n"):
                    lines = ["", *lines]
                image = image.dockerfile_commands(lines)
                assert plan is not None
                build_allowance_sec = plan.build_allowance_sec
            params = image_params(image)
            if env._install_layers_record is not None:
                content = await self._content_snapshot(
                    daytona,
                    image,
                    resources,
                    layered,
                    build_allowance_sec=build_allowance_sec,
                    force_build=force_build,
                    fresh=plan is not None and plan.fresh,
                )
                if content is not None:
                    snapshot_image_params = params
                    params = _sdk.CreateSandboxFromSnapshotParams(
                        auto_delete_interval=env._auto_delete_interval,
                        auto_stop_interval=env._auto_stop_interval,
                        snapshot=content,
                        network_block_all=env._network_block_all,
                        labels=_benchflow_owned_labels(),
                    )

        env._create_attempts = 0
        env._layer_build_allowance_sec = build_allowance_sec
        env._layer_build_falls_back = plain_params is not None
        try:
            if snapshot_image_params is None:
                if getattr(env, "_start_snapshot_ref", None):
                    await env._create_sandbox_without_toolbox_bash_env(
                        params, startup_env=env._start_snapshot_env
                    )
                else:
                    await env._create_sandbox_without_toolbox_bash_env(params)
            else:
                try:
                    await env._create_sandbox_without_toolbox_bash_env(params)
                except Exception as e:
                    if isinstance(e, _sdk.SandboxStartupError):
                        raise
                    # The snapshot exists but a sandbox could not come from
                    # it: build from the recipe, as without snapshots.
                    env.logger.warning(
                        "Could not start from content snapshot "
                        f"{env._install_layers_record.get('snapshot_name')}; "  # type: ignore[union-attr]
                        f"building the image instead ({e})"
                    )
                    if env._install_layers_record is not None:
                        env._install_layers_record["snapshot"] = "fallback"
                        env._install_layers_record["snapshot_reason"] = (
                            f"sandbox from snapshot failed: {e}"[:500]
                        )
                    await env._create_sandbox_without_toolbox_bash_env(
                        snapshot_image_params
                    )
        except Exception as e:
            env._layer_build_falls_back = False
            if layer_claim is not None:
                layer_build_finished(claim_keys, built=False)
                layer_claim = None
            if isinstance(e, _sdk.SandboxStartupError):
                raise
            timed_out = (
                isinstance(e, (TimeoutError, _sdk.DaytonaTimeoutError))
                and not env._create_outcome_unknown
            )
            if plain_params is None or not (
                "build_failed" in str(e).lower() or timed_out
            ):
                raise env._startup_failure(e) from e
            if env._layer_build_failure_confirmed:
                layer_build_finished(layered.keys, built=False, failed=True)
            # A layer that does not build on this image (no bash or base64, a
            # non-root image user) must not cost the trial: build the task
            # image alone and install in the sandbox, as without layers.
            env.logger.warning(
                "Daytona could not build the cached install layers on this "
                f"image; installing in the sandbox instead ({e})"
            )
            layered = LayeredRecipe()
            env._layer_build_allowance_sec = 0
            if env._install_layers_record is not None:
                env._install_layers_record["fallback"] = "layer build failed"
            try:
                await env._create_sandbox_without_toolbox_bash_env(plain_params)
            except Exception as retry_error:
                raise env._startup_failure(retry_error) from retry_error
        env._layer_build_falls_back = False
        if layer_claim is not None:
            layer_build_finished(claim_keys, built=True)

        if env._install_layers_record is not None:
            env._install_layer_keys = dict(layered.keys)
            # A content snapshot holds the same layers as the recipe (its
            # hash covers their lines and keys); the trial still checks each
            # layer's key file in the sandbox before it skips an install.
            source = (
                "snapshot"
                if env._install_layers_record.get("snapshot")
                in ("created", "reused", "waited")
                else "image"
            )
            env._install_layers_record["layers"] = {
                name: {"key": key, "source": source}
                for name, key in layered.keys.items()
            }
            if layered.keys:
                env._install_layers_record.update(build_record_outcome(env._sandbox))
            env._on_sandbox_created()

        # As the default user, in one exec with that user's ``pwd`` and
        # ``id -u``, which the sandbox keeps until it stops or is restored
        # (PY8 U12). ``_sandbox_exec`` adds no persistent environment. On
        # the system shell (C8): a sandbox started from a snapshot can hold
        # what an agent left on the image's PATH.
        await start_with_identity_probe(
            env,
            f"mkdir -p {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir} && "
            f"chmod 777 {SandboxPaths.agent_dir} {SandboxPaths.verifier_dir}",
            functools.partial(
                env._sandbox_exec,
                route=await system_shell_route(env),
            ),
            user=None,
            env=None,
        )
        # As root, bounded, and checked: the image's default user may not be
        # root, and the agent must never start while it can read the log.
        result = await env._sandbox_exec(
            DAEMON_LOG_ROOT_ONLY_CMD,
            user="root",
            timeout_sec=60,
            route=await system_shell_route(env),
        )
        check_daemon_log_result(result)
        await _deliver_bfx(env)

    async def _content_snapshot(
        self,
        daytona: Any,
        image: Any,
        resources: Any,
        layered: LayeredRecipe,
        *,
        build_allowance_sec: int,
        force_build: bool,
        fresh: bool,
    ) -> str | None:
        """The content snapshot to start from (opt-in), or None.

        Records ``snapshot`` (``created``, ``reused``, ``waited``,
        ``fallback`` or ``off``), its reason and ``snapshot_name`` in
        ``install_layers`` (daytona_content_snapshots). Cancellation
        propagates; every other failure falls back to the image recipe.
        """
        env = self._env
        record = env._install_layers_record
        assert record is not None

        def off(reason: str) -> None:
            record["snapshot"] = "off"
            record["snapshot_reason"] = reason

        if not content_snapshots.enabled():
            return None
        prefix = content_snapshots.owner_prefix()
        if prefix is None:
            off("no BENCHFLOW_DAYTONA_OWNER: a snapshot must name its owner")
            return None
        if fresh:
            off("--fresh-install")
            return None
        if force_build:
            off("force_build")
            return None
        context_stage = getattr(env, "_daytona_contexts", None)
        if context_stage is not None:
            await context_stage.complete()
        # A content snapshot builds this same image recipe. Preserve the
        # planned installs' budgets, just as the ordinary image create does;
        # a recipe without install layers needs only the task build budget.
        timeout = round(env.task_env_config.build_timeout_sec) + build_allowance_sec
        try:
            async with content_snapshots.captured_image_async(image) as upload_image:
                identity = await content_snapshots.blocking_work(
                    content_snapshots.content_identity,
                    recipe=upload_image.dockerfile(),
                    context=[
                        (str(c.source_path), str(c.archive_path))
                        for c in getattr(upload_image, "_context_list", None) or []
                    ],
                    layer_lines=layered.lines,
                    layer_keys=layered.keys,
                    cpu=getattr(resources, "cpu", None),
                    memory=getattr(resources, "memory", None),
                    disk=getattr(resources, "disk", None),
                    api_target=content_snapshots.api_target(),
                )
                if identity.content_hash is None:
                    off(identity.reason or "inputs the hash cannot see")
                    return None
                name = content_snapshots.snapshot_name(prefix, identity.content_hash)
                record["snapshot_name"] = name
                create_params = importlib.import_module("daytona").CreateSnapshotParams

                def make_params() -> Any:
                    return create_params(
                        name=name, image=upload_image, resources=resources
                    )

                result = await content_snapshots.ensure_snapshot(
                    daytona, name, make_params, timeout_sec=timeout
                )
        except content_snapshots.SnapshotFallback as e:
            if not record.get("snapshot_name"):
                off(str(e))
                return None
            env.logger.warning(f"No content snapshot; building the image ({e})")
            record["snapshot"] = "fallback"
            record["snapshot_reason"] = str(e)[:500]
            return None
        record["snapshot"] = result.outcome
        record["snapshot_reason"] = None
        record["snapshot_events"] = {
            "create_attempts": result.create_attempts,
            "conflicts": result.conflicts,
            "replaced": result.replaced,
        }
        return name

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        try:
            sandbox = env._sandbox
            if not sandbox:
                env._log_no_sandbox_to_stop()
            else:
                try:
                    await env._stop_sandbox(sandbox)
                except BaseException as e:
                    # Cancelled too (a caller's stop bound): the delete may be
                    # in flight, so the ledger confirms it.
                    env._log_sandbox_left_behind(
                        e, sandbox=sandbox, release_channels=True
                    )
                    if not isinstance(e, Exception):
                        raise
                finally:
                    if env._sandbox is sandbox:
                        env._sandbox = None
            self._snapshot_credentials.clear()
            await self._delete_deferred_snapshots()
        finally:
            if env._sandbox is None:
                env._client_manager = None

    async def _delete_deferred_snapshots(self) -> None:
        """Delete snapshots that were in use when their fork finished."""
        last_error: Exception | None = None
        for delay in (0, *_DELETE_RETRY_DELAYS):
            if not self._deferred_snapshot_refs:
                return
            await asyncio.sleep(delay)
            for ref in sorted(self._deferred_snapshot_refs):
                try:
                    await _delete_provider_snapshot(ref)
                except Exception as exc:
                    last_error = exc
                    continue
                self._deferred_snapshot_refs.discard(ref)
        for ref in sorted(self._deferred_snapshot_refs):
            self._env.logger.warning(
                f"Daytona snapshot {ref} could not be deleted ({last_error}); "
                "`bench sandbox cleanup` removes it once it is stale"
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult:
        # route.task_root needs nothing here: the sandbox starts every shell
        # with the loader variables empty and the image's values aside
        # (DaytonaSandbox._create_sandbox_without_toolbox_bash_env).
        _reject_non_main_service(service)
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            route=route,
        )

    async def exec_transient(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
        *,
        route: ExecRoute,
    ) -> ExecResult:
        _reject_non_main_service(service)
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            cleanup_session=True,
            route=route,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        for delay in (*_UPLOAD_PREP_RETRY_DELAYS, None):
            try:
                prep_result = await self._env._sandbox_exec(
                    f"mkdir -p {shlex.quote(target_dir)}",
                    timeout_sec=30,
                    user="root",
                    route=await system_shell_route(self._env),
                )
            except ExecTimeoutError:
                if delay is None:
                    raise
            else:
                # The in-sandbox `timeout` ends a late mkdir with status 124.
                if prep_result.return_code != 124 or delay is None:
                    break
            await asyncio.sleep(delay)
        if prep_result.return_code != 0:
            raise RuntimeError(
                "Daytona direct upload_dir destination prep failed: "
                f"{_exec_failure_output(prep_result)}"
            )
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        _reject_non_main_service(service)
        await self._env._sdk_download_dir(source_dir, target_dir)

    @stamp_transient_transport
    async def is_dir(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool:
        # The toolbox's file API, no shell: *route* has nothing to pick.
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return file_info.is_dir

    @stamp_transient_transport
    async def is_file(
        self, path: str, user: str | int | None = None, *, route: ExecRoute
    ) -> bool:
        sandbox = self._env._require_sandbox()
        file_info = await sandbox.fs.get_file_info(path)
        return not file_info.is_dir

    async def services(self) -> list[str]:
        raise NotImplementedError(
            "Direct (non-compose) Daytona sandbox is single-container and has "
            "no compose topology. services() requires a multi-container "
            "docker-compose task (#248)."
        )

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        from benchflow.sandbox._root_launch import (
            protect_root_terminal_home,
            require_protected_root_terminal,
            typed_root_prelude,
        )
        from benchflow.sandbox.docker import interactive_root_shell

        # The provider's login shell reads root's HOME and /etc's profile
        # files before the line below: give it a protected HOME, or refuse
        # (PT-C8-R9-7 astra), as for the agent's SSH launch.
        await protect_root_terminal_home(env, "Daytona's attach")
        await require_protected_root_terminal(env, "Daytona's attach")
        # The provider starts the login shell; its first line keeps the
        # task's values, pins the route's PATH and starts root's
        # interactive shell by absolute path with no profile or rc file (the
        # protected kit's once the sandbox has one, C8).
        route = await env.exec_route("root", system_shell=True)
        remote = (
            f"{typed_root_prelude(route.pin or SYSTEM_PATH)}; "
            f"exec {shlex.join(interactive_root_shell(route))}"
        )
        ssh_access = await env._sandbox.create_ssh_access()
        os.execvp(
            "ssh",
            ["ssh", "-t", f"{ssh_access.token}@ssh.app.daytona.io", remote],
        )

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        """Create a Daytona snapshot from the current sandbox state.

        Wraps ``AsyncSandbox._experimental_create_snapshot``; the snapshot
        name is the ``ref`` other Daytona sandboxes can be created from via
        :class:`CreateSandboxFromSnapshotParams`.
        """
        env = self._env
        if not env._sandbox:
            raise SandboxSnapshotNotSupported(
                "DaytonaSandbox.snapshot requires a started sandbox; call "
                "start() before snapshot()."
            )
        from benchflow.sandbox.daytona import _TOOLBOX_STARTUP_META

        receipt = getattr(env, "_toolbox_startup_receipt", None)
        if receipt is None:
            raise RuntimeError(
                "Daytona startup state is unavailable; cannot create checkpoint"
            )
        # Default names carry the owner scope so `bench sandbox cleanup` can
        # find a leaked snapshot (Daytona snapshots have no labels).
        snap_name = name or benchflow_snapshot_name(env.environment_name)
        # Daytona names: lowercase, dash-separated, ascii — sanitize defensively.
        snap_name = snap_name.lower().replace("_", "-")
        # Agent credential files stay out of the provider snapshot.
        ops = self._credential_ops()
        stash = await scrub_credentials(ops)
        try:
            await env._sandbox._experimental_create_snapshot(snap_name)
        finally:
            if stash:
                await put_back_credentials(ops, stash)
        if stash:
            self._snapshot_credentials[snap_name] = stash
        env.logger.info(
            f"Snapshot created: {snap_name}; "
            f"{len(stash)} credential file(s) kept out of it"
        )
        return SandboxImage(
            provider="daytona",
            ref=snap_name,
            meta={
                "sandbox_id": getattr(env._sandbox, "id", "") or "",
                _TOOLBOX_STARTUP_META: json.dumps(receipt),
            },
        )

    async def adopt_snapshot(self, image: SandboxImage) -> None:
        """Remember the live sandbox's credential files for ``image``, a
        snapshot of this state taken elsewhere (a kept checkpoint), so a
        restore from it puts them back. The live files are left as they are."""
        ops = self._credential_ops()
        stash = await scrub_credentials(ops)
        if stash:
            await put_back_credentials(ops, stash)
            self._snapshot_credentials[image.ref] = stash

    async def restore(self, image: SandboxImage) -> None:
        """Replace the current Daytona sandbox with one from ``image``.

        Daytona snapshots are immutable — restore is implemented by deleting
        the current sandbox and creating a fresh one from the snapshot,
        matching the provider's native semantics.
        """
        from benchflow.sandbox import daytona as _sdk

        env = self._env
        if image.provider != "daytona":
            raise SandboxSnapshotNotSupported(
                f"DaytonaSandbox.restore cannot consume a {image.provider!r} "
                f"snapshot (got ref={image.ref!r}); snapshots are not portable "
                "across providers."
            )
        # Validate before deleting the live sandbox. Legacy checkpoints have
        # no pre-agent startup evidence and must not be probed as root.
        startup_env = _sdk._checkpoint_startup_env(image)
        sandbox = env._sandbox
        if sandbox is not None:
            try:
                await delete_sandbox(sandbox)
            except BaseException as e:
                env._log_sandbox_left_behind(e, when="before restore", sandbox=sandbox)
                raise
            env._release_command_channels(sandbox)
            if env._sandbox is sandbox:
                env._sandbox = None

        params = _sdk.CreateSandboxFromSnapshotParams(
            auto_delete_interval=env._auto_delete_interval,
            auto_stop_interval=env._auto_stop_interval,
            snapshot=image.ref,
            network_block_all=env._network_block_all,
            labels=_benchflow_owned_labels(),
        )
        await env._create_sandbox_without_toolbox_bash_env(
            params, startup_env=startup_env
        )
        # The new sandbox's daemon starts again, and the snapshot may predate
        # the step (a kept checkpoint), so protect its log as start() does,
        # before the agent's credentials go back and before any agent runs.
        result = await env._sandbox_exec(
            DAEMON_LOG_ROOT_ONLY_CMD,
            user="root",
            timeout_sec=60,
            route=await system_shell_route(env),
        )
        check_daemon_log_result(result)
        await _deliver_bfx(env)
        stash = self._snapshot_credentials.get(image.ref)
        if stash:
            await put_back_credentials(self._credential_ops(), stash)
        env.logger.info(f"Snapshot restored: {image.ref}")

    async def delete_snapshot(self, image: SandboxImage) -> bool:
        """Delete the Daytona snapshot ``image``; defer it while still in use.

        Returns True when the snapshot is gone (including already gone). When
        the provider refuses (for example while a sandbox created from it is
        being replaced), the snapshot is deleted by :meth:`stop` instead.
        """
        if image.provider != "daytona":
            raise SandboxSnapshotNotSupported(
                f"DaytonaSandbox.delete_snapshot cannot delete a "
                f"{image.provider!r} snapshot (got ref={image.ref!r})"
            )
        self._snapshot_credentials.pop(image.ref, None)
        try:
            await _delete_provider_snapshot(image.ref)
        except Exception as exc:
            self._deferred_snapshot_refs.add(image.ref)
            self._env.logger.info(
                f"Daytona snapshot {image.ref} not deleted yet ({exc}); "
                "retrying when the sandbox stops"
            )
            return False
        self._deferred_snapshot_refs.discard(image.ref)
        self._env.logger.info(f"Snapshot deleted: {image.ref}")
        return True


class _DaytonaCredentialOps:
    """Access to the live Daytona sandbox for the credential scrub.

    File contents never pass through session commands: the Daytona daemon keeps
    each session command's output on disk inside the sandbox, where the
    snapshot would capture it. :meth:`run_as` moves stdin and stdout through a
    fresh root-only staging directory with the file API; the root shell opens
    those files for the unprivileged command (``su`` to the credential home's
    owner), so that command never reaches the staging directory by path.
    """

    def __init__(self, env: DaytonaSandbox) -> None:
        self._env = env

    async def run(self, command: str) -> ExecResult:
        return await self._env._sandbox_exec(
            command,
            user="root",
            timeout_sec=120,
            cleanup_session=True,
            route=await system_shell_route(self._env),
        )

    async def run_as(
        self, uid: int, gid: int, script: str, *, stdin: bytes | None = None
    ) -> AsUserResult:
        made = await self.run('umask 077 && d="$(mktemp -d)" && printf %s "$d"')
        stage = (made.stdout or "").strip()
        if made.return_code != 0 or not stage.startswith("/"):
            return AsUserResult(1, b"", "could not create a staging directory")
        quote = shlex.quote
        try:
            source = "/dev/null"
            if stdin is not None:
                await self._env._require_sandbox().fs.upload_file(stdin, f"{stage}/in")
                source = f"{stage}/in"
            redirect = f"< {quote(source)} > {quote(stage + '/out')}"
            if int(uid) == 0:
                command = f"sh -c {quote(script)} {redirect}"
            else:
                # su needs a name; a uid without one is refused, never run as root.
                command = (
                    f"u=$(awk -F: -v id={int(uid)} '$3 == id {{ print $1; exit }}' "
                    "/etc/passwd); "
                    f'[ -n "$u" ] || {{ echo "refused: no user with uid {int(uid)}" '
                    ">&2; exit 97; }; "
                    f'su "$u" -s /bin/sh -c {quote(script)} {redirect}'
                )
            result = await self.run(command)
            stdout = b""
            if result.return_code == 0:
                content = await self._env._require_sandbox().fs.download_file(
                    f"{stage}/out"
                )
                stdout = bytes(content or b"")
            return AsUserResult(result.return_code, stdout, result.stderr or "")
        finally:
            await self.run(f"rm -rf -- {quote(stage)}")
