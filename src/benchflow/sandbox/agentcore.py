"""Amazon Bedrock AgentCore Runtime sandbox backend.

AgentCore runs each session in an isolated Firecracker-class microVM. Unlike
Docker/Modal/Daytona it does not take an image per run: an image must first be
pushed to ECR and registered as an *agent runtime* (a control-plane resource
with its own ARN), after which sessions are invoked against that ARN.

**A rollout is a session, not a runtime.** One registered runtime hosts many
concurrent sessions, each an isolated microVM with its own filesystem, and the
account quotas are lopsided in exactly that direction: 5000 concurrent
sessions against 100 total runtimes, with ``CreateAgentRuntime`` limited to
5/s. So the image and the runtime are built **once per distinct task image**
— keyed by a digest of the build context, and shared by every trial and skill
arm that resolves to it — while sessions are what scale out. That sharing is
what makes this backend usable for a large parallel matrix; see
``agentcore_provisioning`` for the single-flight machinery.

Runtimes therefore outlive the rollout that first needed them, like a built
Docker image, and are reclaimed by age via ``bench sandbox cleanup``.

Two facts about the platform shape this module, both established by direct
experiment against the live service rather than from the docs:

1. **The container must answer the Runtime HTTP contract.** A microVM whose
   image merely sleeps reaches ``READY`` but every
   ``InvokeAgentRuntimeCommand`` against it fails with a 500 from the runtime.
   Serving ``GET /ping`` on port 8080 fixes it. Task images know nothing about
   AgentCore, so BenchFlow appends a tiny stdlib-only responder to the image
   and makes it the entrypoint (see ``agentcore_image.PING_SHIM``).
2. **Command execution and the interactive shell share one session.** A file
   written by ``exec()`` is visible to the agent running under
   ``open_shell()`` when both use the same ``runtimeSessionId``, which is what
   lets the kernel stage files and the verifier read results around an agent
   that lives on the WebSocket.

Not supported, and gated rather than faked: container snapshots (no platform
primitive — the ``BaseSandbox`` default raises), multi-service compose
topologies (single container), and ``network_mode = "no-network"`` — the
``networkConfiguration.networkMode`` enum offers only ``PUBLIC`` and ``VPC``,
so isolation is declared unsupported in the provider registry instead of being
silently ignored.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import shlex
import tarfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from benchflow._paths import iter_safe_tree
from benchflow.sandbox import agentcore_builder as builders
from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox._base import BaseSandbox, ExecResult
from benchflow.sandbox._compose import compose_definition_path
from benchflow.sandbox.agentcore_image import AgentCoreImagePublisher
from benchflow.sandbox.agentcore_sealed import SealedChannel
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

if TYPE_CHECKING:
    from benchflow.sandbox.process import LiveProcess

_DEFAULT_REGION = "us-west-2"
_DEFAULT_ECR_REPOSITORY = "benchflow-agentcore"
_RUNTIME_READY_TIMEOUT_SEC = 600
_RUNTIME_POLL_INTERVAL_SEC = 5
# A cold image can take a while to pull into the microVM on the first command.
_SESSION_WARMUP_ATTEMPTS = 8
_SESSION_WARMUP_BACKOFF_SEC = 4.0
_SESSION_WARMUP_MAX_BACKOFF_SEC = 30.0
_LIFECYCLE_MIN_SEC = 60
_LIFECYCLE_MAX_SEC = 28800
# The service caps a single command payload at 64 KB. Base64 inflates by 4/3,
# and the wrapper adds shell scaffolding, so keep a wide margin.
_MAX_INLINE_UPLOAD_BYTES = 24 * 1024
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
# ``PING_SHIM`` implements the plain HTTP contract (GET /ping,
# POST /invocations), so register the runtime as HTTP rather than relying on
# whatever the service currently defaults to.
_PROTOCOL_CONFIGURATION = {"serverProtocol": "HTTP"}
#: AgentCore offers PUBLIC or VPC only; see the registry's enforces_no_network.
_NETWORK_CONFIGURATION = {"networkMode": "PUBLIC"}

# Environment overrides.
_ENV_REGION = "BENCHFLOW_AGENTCORE_REGION"
_ENV_ROLE_ARN = "BENCHFLOW_AGENTCORE_ROLE_ARN"
_ENV_ECR_REPOSITORY = "BENCHFLOW_AGENTCORE_ECR_REPOSITORY"
_ENV_IDLE_TIMEOUT = "BENCHFLOW_AGENTCORE_IDLE_TIMEOUT_SEC"
_ENV_MAX_LIFETIME = "BENCHFLOW_AGENTCORE_MAX_LIFETIME_SEC"


class AgentCoreSandbox(BaseSandbox):
    """Sandbox backend for Bedrock AgentCore Runtime microVMs."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        rollout_paths: RolloutPaths | None,
        task_env_config: SandboxConfig,
        *args: Any,
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
        self.region = os.environ.get(_ENV_REGION) or _DEFAULT_REGION
        ecr_repository = os.environ.get(_ENV_ECR_REPOSITORY) or _DEFAULT_ECR_REPOSITORY
        self._role_arn = os.environ.get(_ENV_ROLE_ARN)
        self._agent_timeout_sec = 0
        self.runtime_arn: str | None = None
        self.runtime_session_id: str | None = None
        self._runtime_id: str | None = None
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()
        self._sealed = SealedChannel(self._exec_raw, self.logger)
        self._images = AgentCoreImagePublisher(
            environment_dir=environment_dir,
            environment_name=environment_name,
            task_env_config=task_env_config,
            region=self.region,
            ecr_repository=ecr_repository,
            client_factory=self._client,
            logger=self.logger,
        )

    # ---------------------------------------------------------------- config

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def sandbox_id(self) -> str | None:
        return self.runtime_arn

    @classmethod
    def preflight(cls) -> None:
        try:
            import bedrock_agentcore  # noqa: F401
            import boto3
        except ImportError as exc:
            raise SystemExit(
                "AgentCore requires the 'sandbox-agentcore' extra. Install it "
                "with `uv sync --extra sandbox-agentcore` or "
                "`pip install 'benchflow[sandbox-agentcore]'`."
            ) from exc

        from botocore.exceptions import BotoCoreError, ClientError

        try:
            boto3.Session().client(
                "sts", region_name=os.environ.get(_ENV_REGION) or _DEFAULT_REGION
            ).get_caller_identity()
        except (BotoCoreError, ClientError) as exc:
            raise SystemExit(
                "AgentCore requires working AWS credentials. Configure them with "
                "`aws configure`, an SSO profile, or AWS_* environment "
                f"variables. Underlying error: {exc}"
            ) from exc

        # Docker is not required: without a local daemon, images are built on
        # AWS CodeBuild instead. Only flag the case where neither route is
        # usable, so the run fails now rather than at the first build.
        if not builders.docker_available() and not os.environ.get(
            builders.ENV_CODEBUILD_ROLE
        ):
            raise SystemExit(
                "AgentCore needs a way to build task images. Either start a "
                "local Docker daemon, or set "
                f"{builders.ENV_CODEBUILD_ROLE} to a CodeBuild service role so "
                "images can be built in AWS without Docker."
            )

    def _validate_definition(self) -> None:
        dockerfile = self.environment_dir / "Dockerfile"
        if not self.task_env_config.docker_image and (
            not os.path.lexists(dockerfile)
            or dockerfile.is_symlink()
            or not dockerfile.is_file()
        ):
            raise ValueError(
                f"{dockerfile} must be a regular, non-symlink Dockerfile "
                "when no docker_image is specified."
            )
        for reserved in (
            provisioning.GENERATED_DOCKERFILE,
            provisioning.GENERATED_SHIM,
        ):
            if os.path.lexists(self.environment_dir / reserved):
                # Staging owns these names and the canonical walk excludes
                # them. Accepting a task file here would silently replace it
                # with backend scaffolding and cache-hit as if it did not exist.
                raise ValueError(
                    f"{reserved} already exists in {self.environment_dir}. "
                    "That path is reserved by the AgentCore backend. Rename "
                    "the task file."
                )
        compose = compose_definition_path(self.environment_dir)
        if compose is not None:
            # AgentCore builds and runs exactly one container. Accepting a
            # compose task would launch the agent's container without its side
            # services, and the resulting failure would be scored against the
            # agent rather than reported as an unsupported environment.
            raise ValueError(
                f"{compose.name} found in {self.environment_dir}: AgentCore is a "
                "single-container backend and cannot start compose side "
                "services. Run multi-service tasks on the docker or daytona "
                "sandbox."
            )

    # ------------------------------------------------------------- lifecycle

    def _client(self, service: str) -> Any:
        """Return a cached boto3 client for *service*.

        Building a client parses the service's JSON model and re-resolves
        credentials, which is far too expensive to repeat per ``exec()`` — and
        it floods the run log with botocore's credential-discovery line. The
        cache is guarded because ``exec`` dispatches through
        ``asyncio.to_thread``, so concurrent calls can arrive on different
        threads. botocore clients are safe to share for API calls.
        """
        cached = self._clients.get(service)
        if cached is not None:
            return cached
        with self._client_lock:
            cached = self._clients.get(service)
            if cached is None:
                import boto3

                cached = boto3.Session(region_name=self.region).client(service)
                self._clients[service] = cached
        return cached

    async def start(self, force_build: bool) -> None:
        try:
            image_uri = await self._images.publish(force_build=force_build)
            self.runtime_arn = await self._ensure_runtime(image_uri)
            # AgentCore requires a session id of at least 33 characters.
            self.runtime_session_id = f"{uuid.uuid4()}-{uuid.uuid4().hex[:8]}"
            # Provisioning is memoized, so only the first rollout of an image
            # runs the creation path. Every rollout refreshes the lease (rate
            # limited) or a long matrix outlives the lease it inherited.
            await self._renew_lease()
            await self._warm_session()
            self.logger.info(
                "AgentCore session ready (runtime=%s, session=%s)",
                self.runtime_arn,
                self.runtime_session_id,
            )
        except SandboxStartupError:
            await self._safe_teardown()
            raise
        except Exception as exc:
            await self._safe_teardown()
            raise SandboxStartupError(
                f"AgentCore sandbox failed to start: {exc}",
                sandbox_id=self.runtime_arn,
                build_timeout_sec=self.task_env_config.build_timeout_sec,
            ) from exc

    async def _ensure_runtime(self, image_uri: str) -> str:
        """Resolve the shared agent runtime for *image_uri*, creating it once.

        The runtime is named after the image's content digest, so every rollout
        of that image — each trial, and both skill arms when their images match
        — resolves to the same runtime and simply opens another session against
        it. Keying on the task name instead meant concurrent trials raced to
        create one runtime and the first to finish deleted it while the others
        were still running.
        """
        digest, _tag = self._images.identity()
        name = provisioning.runtime_name(self.environment_name, digest)

        async def _create() -> tuple[str, str]:
            return await self._create_or_adopt_runtime(name, image_uri)

        # Keyed on the image as well as the name: with only the name, a
        # force-rebuild that pushes new bytes under the same context identity
        # hits the memo and silently skips compare/update/verify.
        arn, runtime_id = await provisioning.once(
            f"runtime:{name}:{image_uri}", _create
        )
        self._runtime_id = runtime_id
        return arn

    async def _create_or_adopt_runtime(
        self, name: str, image_uri: str
    ) -> tuple[str, str]:
        """Create the runtime, or adopt an equivalent one that already exists.

        Attempting the create first and falling back on conflict keeps the
        common path off ``ListAgentRuntimes``, whose 5/s quota would otherwise
        throttle a large matrix run before it started.
        """
        from botocore.exceptions import ClientError

        control = self._client("bedrock-agentcore-control")
        try:
            created = await asyncio.to_thread(
                control.create_agent_runtime,
                agentRuntimeName=name,
                agentRuntimeArtifact={
                    "containerConfiguration": {"containerUri": image_uri}
                },
                roleArn=self._require_role_arn(),
                networkConfiguration=_NETWORK_CONFIGURATION,
                protocolConfiguration=_PROTOCOL_CONFIGURATION,
                lifecycleConfiguration=self._lifecycle_configuration(),
                description=f"BenchFlow sandbox for {self.environment_name}",
                tags={
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    "benchflow-task": self.environment_name[:120],
                },
            )
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code not in {"ConflictException", "ValidationException"}:
                raise
            existing = await asyncio.to_thread(
                provisioning.find_runtime_by_name, control, name
            )
            if existing is None:
                raise
            arn, runtime_id, current_image = existing
            await asyncio.to_thread(self._write_lease, control, arn)
            provisioning.mark_lease_renewed(arn, time.monotonic())
            if current_image != image_uri:
                # Adopting a runtime bound to different bytes would silently run
                # the wrong environment. The name encodes the build-context
                # digest, but that is not the same as the pushed image: a
                # repository change or a force rebuild can move what the name
                # resolves to. Re-point the runtime and wait for it to settle.
                self.logger.warning(
                    "AgentCore runtime %s points at %s, not %s — updating",
                    arn,
                    current_image,
                    image_uri,
                )
                await asyncio.to_thread(
                    control.update_agent_runtime,
                    agentRuntimeId=runtime_id,
                    agentRuntimeArtifact={
                        "containerConfiguration": {"containerUri": image_uri}
                    },
                    roleArn=self._require_role_arn(),
                    networkConfiguration=_NETWORK_CONFIGURATION,
                    protocolConfiguration=_PROTOCOL_CONFIGURATION,
                    # Omitting this resets idle/max-lifetime to the service
                    # defaults, silently discarding the caller's configured
                    # window and reclaiming long sessions early.
                    lifecycleConfiguration=self._lifecycle_configuration(),
                )
            await asyncio.to_thread(self._wait_ready, control, runtime_id)
            await asyncio.to_thread(
                self._verify_adopted_runtime,
                control,
                runtime_id,
                image_uri,
                self._lifecycle_configuration(),
                self._require_role_arn(),
                _NETWORK_CONFIGURATION,
                _PROTOCOL_CONFIGURATION,
            )
            self.logger.info("Adopted existing AgentCore runtime %s", arn)
            return arn, runtime_id

        runtime_id = created["agentRuntimeId"]
        await asyncio.to_thread(self._write_lease, control, created["agentRuntimeArn"])
        provisioning.mark_lease_renewed(created["agentRuntimeArn"], time.monotonic())
        await asyncio.to_thread(self._wait_ready, control, runtime_id)
        self.logger.info(
            "Registered AgentCore runtime %s for %s",
            created["agentRuntimeArn"],
            image_uri,
        )
        return created["agentRuntimeArn"], runtime_id

    async def _renew_lease(self) -> None:
        """Refresh this runtime's lease if it is due, before the session runs."""
        if not self.runtime_arn:
            return
        lifecycle = self._lifecycle_configuration()
        window = float(
            max(lifecycle["maxLifetime"], lifecycle["idleRuntimeSessionTimeout"])
        )
        runtime_arn = self.runtime_arn

        async def _write() -> None:
            await asyncio.to_thread(
                self._write_lease,
                self._client("bedrock-agentcore-control"),
                runtime_arn,
            )

        await provisioning.renew_lease(
            runtime_arn,
            window,
            _write,
            monotonic=time.monotonic,
        )

    def _write_lease(self, control: Any, runtime_arn: str) -> None:
        """Mark the runtime as possibly-in-use until the session window closes.

        There is no API that enumerates a runtime's active sessions
        (``ListSessions`` is Memory-scoped), and session traffic does not move
        the runtime's ``lastUpdatedAt`` — so control-plane age alone cannot
        distinguish an idle runtime from one serving a matrix right now.

        The lease is the explicit contract that closes that gap. It extends
        beyond the longest session window by one renewal interval: a rollout
        admitted just before the throttle boundary may still live for the full
        configured window, so a lease of only ``window`` seconds would expire
        while that session was still legal.
        """
        lifecycle = self._lifecycle_configuration()
        window = max(lifecycle["maxLifetime"], lifecycle["idleRuntimeSessionTimeout"])
        duration = window + provisioning.lease_renewal_interval(float(window))
        until = datetime.now(UTC) + timedelta(seconds=duration)
        try:
            control.tag_resource(
                resourceArn=runtime_arn,
                tags={provisioning.LEASE_TAG: until.isoformat()},
            )
        except Exception as exc:
            # Swallowing this would leave the runtime unleased while sessions
            # run against it, which is precisely the state cleanup is allowed
            # to delete. Fail the launch instead of starting work that another
            # process may reap out from under us.
            raise SandboxStartupError(
                f"Could not write the AgentCore lease for {runtime_arn}: {exc}. "
                "Refusing to start a session that cleanup could delete.",
                sandbox_id=runtime_arn,
            ) from exc

    @staticmethod
    def _verify_adopted_runtime(
        control: Any,
        runtime_id: str,
        image_uri: str,
        lifecycle: dict[str, int],
        role_arn: str,
        network: dict[str, Any],
        protocol: dict[str, Any],
    ) -> None:
        """Fail closed unless the adopted runtime matches what this run needs.

        Adoption is the one path where a runtime can predate this process, so
        it is the one path where a mismatch could otherwise go unnoticed — an
        agent run against the wrong image, or against a shorter session window
        than the caller configured.
        """
        detail = control.get_agent_runtime(agentRuntimeId=runtime_id)
        artifact = detail.get("agentRuntimeArtifact") or {}
        bound = (artifact.get("containerConfiguration") or {}).get("containerUri")
        if bound != image_uri:
            raise SandboxStartupError(
                f"AgentCore runtime {runtime_id} is bound to {bound!r} but this "
                f"task needs {image_uri!r}. Refusing to run against the wrong "
                "image.",
                sandbox_id=runtime_id,
            )
        actual = detail.get("lifecycleConfiguration") or {}
        drifted: dict[str, tuple[Any, Any]] = {
            key: (actual.get(key), value)
            for key, value in lifecycle.items()
            if actual.get(key) != value
        }
        # Everything the create request pins must also hold on adoption. A
        # runtime with the right image but the wrong execution role, network
        # mode, or protocol is a different sandbox contract than this run asked
        # for — wrong permissions, wrong egress, or a shell that never answers.
        for label, expected, found in (
            ("roleArn", role_arn, detail.get("roleArn")),
            (
                "networkConfiguration",
                dict(network),
                dict(detail.get("networkConfiguration") or {}),
            ),
            (
                "protocolConfiguration",
                dict(protocol),
                dict(detail.get("protocolConfiguration") or {}),
            ),
        ):
            if found != expected:
                drifted[label] = (found, expected)
        if drifted:
            raise SandboxStartupError(
                f"AgentCore runtime {runtime_id} does not match this run's "
                f"configuration (found, expected): {drifted}. Refusing to adopt "
                "a runtime with a different contract.",
                sandbox_id=runtime_id,
            )

    def _lifecycle_configuration(self) -> dict[str, int]:
        """Session idle/lifetime caps.

        The service defaults (900 s idle, 8 h lifetime) are short enough that a
        long agent turn can have its microVM reclaimed mid-run, which surfaces
        as a dead transport rather than as an infrastructure error. Default the
        idle window to the task's own agent timeout when that is larger.
        """

        def _configured(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw in {None, ""}:
                value = default
            else:
                try:
                    value = int(raw)
                except ValueError as exc:
                    raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
            if not _LIFECYCLE_MIN_SEC <= value <= _LIFECYCLE_MAX_SEC:
                raise ValueError(
                    f"{name} must be between {_LIFECYCLE_MIN_SEC} and "
                    f"{_LIFECYCLE_MAX_SEC} seconds for AgentCore, got {value}."
                )
            return value

        requested_idle = max(900, self._agent_timeout_sec)
        idle = _configured(_ENV_IDLE_TIMEOUT, requested_idle)
        lifetime = _configured(_ENV_MAX_LIFETIME, 28800)
        return {
            "idleRuntimeSessionTimeout": idle,
            "maxLifetime": lifetime,
        }

    def configure_agent_timeout(self, timeout_sec: int) -> None:
        """Keep the remote session alive for the effective rollout budget."""
        self._agent_timeout_sec = max(0, int(timeout_sec))

    def _require_role_arn(self) -> str:
        if self._role_arn:
            return self._role_arn
        raise RuntimeError(
            "AgentCore needs an execution role for the runtime. Set "
            f"{_ENV_ROLE_ARN} to a role that bedrock-agentcore.amazonaws.com "
            "can assume and that can pull from ECR and write CloudWatch logs."
        )

    @staticmethod
    def _wait_ready(control: Any, runtime_id: str) -> None:
        import time

        deadline = time.monotonic() + _RUNTIME_READY_TIMEOUT_SEC
        status = "CREATING"
        while time.monotonic() < deadline:
            status = control.get_agent_runtime(agentRuntimeId=runtime_id)["status"]
            if status == "READY":
                return
            if status in {"CREATE_FAILED", "UPDATE_FAILED"}:
                raise SandboxStartupError(
                    f"AgentCore runtime entered {status}",
                    sandbox_id=runtime_id,
                    sandbox_state=status,
                )
            time.sleep(_RUNTIME_POLL_INTERVAL_SEC)
        raise SandboxStartupError(
            f"AgentCore runtime not READY within {_RUNTIME_READY_TIMEOUT_SEC}s "
            f"(last status {status})",
            sandbox_id=runtime_id,
            sandbox_state=status,
        )

    async def _warm_session(self) -> None:
        """Boot the microVM and block until the command plane answers.

        ``READY`` describes the *runtime definition*, not a running session.
        The first command on a new session is what actually pulls the image
        and starts the container, and until the container is serving on 8080
        the service answers with a 500 ``RuntimeClientError``. Observed
        directly: a cold, never-pulled image 500s on the first command and
        succeeds moments later, while an image already warm in the region
        succeeds immediately.

        Retrying here is what keeps that cold-start race from surfacing as a
        task failure — the alternative is a rollout that scores 0 because the
        sandbox was still booting.
        """
        delay = _SESSION_WARMUP_BACKOFF_SEC
        last: Exception | None = None
        for attempt in range(1, _SESSION_WARMUP_ATTEMPTS + 1):
            try:
                result = await self.exec("true", timeout_sec=60)
                if result.return_code == 0:
                    if attempt > 1:
                        self.logger.info(
                            "AgentCore session warm after %d attempts", attempt
                        )
                    return
                last = RuntimeError(
                    (result.stderr or result.stdout or "no output").strip()[:500]
                )
            except SandboxStartupError:
                # Throttling/quota — genuinely infrastructure, and retrying
                # here would just burn the caller's time.
                raise
            except Exception as exc:
                last = exc
            if attempt < _SESSION_WARMUP_ATTEMPTS:
                self.logger.debug(
                    "AgentCore session not warm yet (attempt %d/%d): %s",
                    attempt,
                    _SESSION_WARMUP_ATTEMPTS,
                    last,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _SESSION_WARMUP_MAX_BACKOFF_SEC)

        raise SandboxStartupError(
            f"AgentCore session did not accept commands after "
            f"{_SESSION_WARMUP_ATTEMPTS} attempts: {last}",
            sandbox_id=self.runtime_arn,
        )

    async def stop(self, delete: bool) -> None:
        """End this rollout's session. The runtime is shared and outlives it.

        A rollout owns a *session*, not a runtime: concurrent trials of the
        same task run as separate sessions against one registered runtime, so
        deleting the runtime here would tear down sandboxes still in use — and
        would also burn the 5/s ``DeleteAgentRuntime``/``CreateAgentRuntime``
        quotas re-registering it for the next rollout.

        Runtimes are therefore left in place, like a built Docker image, and
        reclaimed by age with ``bench sandbox cleanup --sandbox agentcore``.
        """
        await self._safe_teardown()

    async def _safe_teardown(self) -> None:
        if self.runtime_arn and self.runtime_session_id:
            try:
                await asyncio.to_thread(
                    self._client("bedrock-agentcore").stop_runtime_session,
                    runtimeSessionId=self.runtime_session_id,
                    agentRuntimeArn=self.runtime_arn,
                    qualifier="DEFAULT",
                )
            except Exception as exc:
                self.logger.warning("Failed to stop AgentCore session: %s", exc)
            self.runtime_session_id = None

    # ------------------------------------------------------------------ exec

    async def live_process(self, *, agent: str | None = None) -> LiveProcess:
        from benchflow.sandbox.process import AgentCoreProcess

        return AgentCoreProcess.from_sandbox_env(self)

    def _reject_non_main(self, service: str) -> None:
        if service != "main":
            raise ValueError(
                f"AgentCore is a single-container backend and cannot target "
                f"service {service!r}. Multi-container (vulhub-style) tasks "
                "require the Docker sandbox (#248)."
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        service: str = "main",
    ) -> ExecResult:
        self._reject_non_main(service)
        if not self.runtime_arn or not self.runtime_session_id:
            raise RuntimeError("AgentCore sandbox not started. Call start() first.")

        wrapped = command
        merged_env = self._merge_env(env)
        if merged_env:
            # NEVER use the shared env-file command wrapper here: it embeds
            # the whole environment as reversible base64 in the command, and
            # this platform records command bodies permanently in the
            # runtime's CloudWatch log group. Stage the environment through
            # the sealed (ciphertext-only) channel and source it instead, so
            # the logged text holds only a file path.
            env_path = await self._sealed.stage_env(merged_env)
            wrapped = (
                f"set -a; . {shlex.quote(env_path)}; set +a; "
                f"rm -f {shlex.quote(env_path)}; {wrapped}"
            )
        if cwd:
            wrapped = f"cd {shlex.quote(cwd)} && {wrapped}"

        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            wrapped = f"su {user_arg} -s /bin/bash -c {shlex.quote(wrapped)}"

        payload = f"/bin/bash -c {shlex.quote(wrapped)}"
        # The service enforces 1..3600 for the command timeout.
        timeout = max(1, min(int(timeout_sec or 300), 3600))
        return await asyncio.to_thread(self._invoke_command, payload, timeout)

    def _invoke_command(self, payload: str, timeout: int) -> ExecResult:
        client = self._client("bedrock-agentcore")
        response = client.invoke_agent_runtime_command(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=self.runtime_session_id,
            qualifier="DEFAULT",
            contentType="application/json",
            accept="application/vnd.amazon.eventstream",
            body={"command": payload, "timeout": timeout},
        )

        stdout: list[str] = []
        stderr: list[str] = []
        exit_code: int | None = None
        status: str | None = None

        for event in response.get("stream", []):
            chunk = event.get("chunk")
            if chunk is None:
                self._raise_for_stream_error(event)
                continue
            delta = chunk.get("contentDelta")
            if delta:
                stdout.append(delta.get("stdout") or "")
                stderr.append(delta.get("stderr") or "")
            stop = chunk.get("contentStop")
            if stop:
                exit_code = stop.get("exitCode")
                status = stop.get("status")

        if status == "TIMED_OUT":
            raise TimeoutError(
                f"AgentCore command timed out after {timeout}s: {payload[:200]}"
            )
        return ExecResult(
            stdout="".join(stdout),
            stderr="".join(stderr),
            # A stream that stops without contentStop has no exit code to
            # report; surfacing 1 keeps callers' `return_code != 0` checks
            # honest rather than inventing a success.
            return_code=exit_code if exit_code is not None else 1,
        )

    @staticmethod
    def _raise_for_stream_error(event: dict[str, Any]) -> None:
        """Translate a typed stream error into the right BenchFlow failure.

        Throttling and quota exhaustion are infrastructure, not agent
        behaviour; raising ``SandboxStartupError`` keeps them attributable in
        ``result.json`` instead of being recorded as a failed command.
        """
        infra = {
            "throttlingException",
            "serviceQuotaExceededException",
            "internalServerException",
        }
        for key, value in event.items():
            message = (
                (value or {}).get("message", "") if isinstance(value, dict) else ""
            )
            if key in infra:
                raise SandboxStartupError(
                    f"AgentCore {key}: {message}", sandbox_state=key
                )
            if key.endswith("Exception") or key == "runtimeClientError":
                raise RuntimeError(f"AgentCore {key}: {message}")

    # -------------------------------------------------------- file transfer

    async def _exec_raw(
        self,
        command: str,
        *,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Run *command* with no environment injection at all.

        The sealed transport and the env-staging path must use this: routing
        them through :meth:`exec` would re-enter environment wrapping (and,
        for env staging, recurse). Nothing secret may appear in *command* —
        every caller passes ciphertext or plain paths.
        """

        wrapped = command
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            wrapped = f"su {user_arg} -s /bin/bash -c {shlex.quote(wrapped)}"
        payload = f"/bin/bash -c {shlex.quote(wrapped)}"
        timeout = max(1, min(int(timeout_sec or 300), 3600))
        return await asyncio.to_thread(self._invoke_command, payload, timeout)

    async def write_text_file(
        self, remote_path: str, body: str, *, mode: str = "600"
    ) -> bool:
        """Write *body* to *remote_path* inside the session, base64-encoded.

        Used by the ACP transport to stage the agent env file without typing
        secrets into the PTY, where they would be echoed into the agent log.
        """
        try:
            await self._sealed.upload(
                body.encode(), target=remote_path, mode=mode, timeout_sec=120
            )
        except RuntimeError:
            return False
        return True

    async def upload_file(
        self, source_path: Path | str, target_path: str, *, mode: str | None = None
    ) -> None:
        source = Path(source_path)
        await self._sealed.upload(
            source.read_bytes(), target=target_path, mode=mode, timeout_sec=300
        )

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str, service: str = "main"
    ) -> None:
        self._reject_non_main(service)
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory {source} does not exist")
        # Skip symlinks (#411): a task-controlled link must not exfiltrate host
        # files into the remote microVM.
        members = {
            path: PurePosixPath(target_dir) / path.relative_to(source).as_posix()
            for path in iter_safe_tree(source, context=f"agentcore upload_dir {source}")
        }
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        if members:
            await self._upload_via_tar(members)

    async def _upload_via_tar(self, members: dict[Path, PurePosixPath]) -> None:
        """Ship files as a single base64 tar stream, chunked under the cap.

        One archive per directory keeps this O(1) commands for the common case
        instead of one round trip per file, and the chunk loop keeps every
        individual command well inside the service's 64 KB payload limit.
        """
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for local, remote in members.items():
                tar.add(local, arcname=str(remote).lstrip("/"))
        await self._sealed.upload(
            buffer.getvalue(), target=None, extract_tar=True, timeout_sec=600
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        result = await self.exec(f"base64 {shlex.quote(source_path)}", timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                f"AgentCore download_file failed: {(result.stderr or '')[:500]}"
            )
        target.write_bytes(
            self._decode_download_payload(result.stdout or "", kind="file")
        )

    @staticmethod
    def _decode_download_payload(raw: str, *, kind: str) -> bytes:
        """Decode one bounded, validated base64 command response."""
        compact = "".join(raw.split())
        max_encoded = 4 * ((_MAX_DOWNLOAD_BYTES + 2) // 3)
        if len(compact) > max_encoded:
            raise RuntimeError(
                f"AgentCore download_{kind} encoded payload exceeds the "
                f"{_MAX_DOWNLOAD_BYTES} byte cap; narrow the {kind}."
            )
        try:
            blob = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                f"AgentCore download_{kind} returned invalid base64"
            ) from exc
        if len(blob) > _MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                f"AgentCore download_{kind} payload {len(blob)} bytes exceeds "
                f"the {_MAX_DOWNLOAD_BYTES} byte cap; narrow the {kind}."
            )
        return blob

    async def download_dir(
        self, source_dir: str, target_dir: Path | str, service: str = "main"
    ) -> None:
        self._reject_non_main(service)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        result = await self.exec(
            f"tar -czf - -C {shlex.quote(source_dir)} . | base64 -w0",
            timeout_sec=600,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"AgentCore download_dir failed: {(result.stderr or '')[:500]}"
            )
        raw = (result.stdout or "").strip()
        if not raw:
            return
        blob = self._decode_download_payload(raw, kind="directory")
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(target, filter="data")
