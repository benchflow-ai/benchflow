"""How a task image gets built and pushed to ECR for the AgentCore backend.

AgentCore only accepts an image that already exists in ECR, so something has
to build one. Two strategies:

* :class:`LocalDockerBuilder` shells out to the local Docker daemon. Fast when
  the host is arm64 (a native build), and the obvious choice on a developer
  machine.
* :class:`CodeBuildBuilder` ships the build context to S3 and builds it in AWS
  CodeBuild on a Graviton worker, pushing straight to ECR. Nothing is required
  locally — no Docker, no arm64 host, no qemu — which is what makes the
  backend usable from a laptop without a container runtime, from CI, and from
  Windows.

The default is ``auto``: use Docker when a working daemon is present, and fall
back to CodeBuild when it is not. Selection is explicit in the logs, because
"why did this build take four minutes" should never be a mystery.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchflow.sandbox import agentcore_provisioning as provisioning

logger = logging.getLogger("benchflow").getChild("agentcore-builder")

ENV_BUILDER = "BENCHFLOW_AGENTCORE_BUILDER"
ENV_CODEBUILD_ROLE = "BENCHFLOW_AGENTCORE_CODEBUILD_ROLE_ARN"
ENV_BUILD_BUCKET = "BENCHFLOW_AGENTCORE_BUILD_BUCKET"

CODEBUILD_PROJECT = "benchflow-agentcore-builder"
# Graviton worker: builds linux/arm64 natively, so no qemu emulation.
CODEBUILD_IMAGE = "aws/codebuild/amazonlinux-aarch64-standard:3.0"
CODEBUILD_COMPUTE = "BUILD_GENERAL1_LARGE"
_CODEBUILD_POLL_SEC = 10
_CODEBUILD_TIMEOUT_MIN = 60


@dataclass(frozen=True)
class BuildRequest:
    """Everything needed to produce one image in ECR."""

    context_dir: Path
    dockerfile_text: str
    shim_text: str
    image_uri: str
    registry: str
    region: str
    force_build: bool
    timeout_sec: float | None


class Builder(Protocol):
    name: str

    async def build_and_push(self, request: BuildRequest) -> None: ...


@contextmanager
def materialized(request: BuildRequest) -> Iterator[Path]:
    """Yield a generated Dockerfile inside an isolated staged build context.

    Generated files never belong in the caller's task tree. Fixed filenames
    there were vulnerable to dangling symlinks, partial-write cleanup, and
    cross-process builds overwriting or unlinking each other's inputs. Each
    build now receives a unique context containing exactly the canonical,
    already-filtered entries plus BenchFlow's two generated files.
    """
    with tempfile.TemporaryDirectory(prefix="benchflow-agentcore-context-") as raw:
        context = Path(raw)
        directory_modes: list[tuple[Path, int]] = []
        # Do not copy ignore-control files into the filtered staging tree: the
        # local daemon/CodeBuild docker invocation would apply them a second
        # time and could exclude BenchFlow's generated shim.
        ignore_controls = {
            ".dockerignore",
            f"{provisioning.GENERATED_DOCKERFILE}.dockerignore",
        }
        for source, relative in provisioning.iter_context_entries(request.context_dir):
            if relative in ignore_controls:
                continue
            target = context / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                # Apply restrictive directory modes only after descendants are
                # copied; chmod(0555) here would make a valid read-only source
                # directory impossible to stage.
                directory_modes.append((target, source.stat().st_mode & 0o7777))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=False)

        shim = context / provisioning.GENERATED_SHIM
        dockerfile = context / provisioning.GENERATED_DOCKERFILE
        shim.write_text(request.shim_text)
        dockerfile.write_text(request.dockerfile_text)
        # Generated modes are part of the image but not caller-controlled
        # identity inputs, so make them deterministic across host umasks.
        shim.chmod(0o644)
        dockerfile.chmod(0o644)
        for directory, mode in reversed(directory_modes):
            directory.chmod(mode)
        yield dockerfile


def _run(*args: str, timeout: float | None = None, input_text: str | None = None):
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, input=input_text
    )


def docker_available() -> bool:
    """True when a Docker daemon is actually reachable.

    Checks the daemon, not just the CLI: an installed ``docker`` binary with a
    stopped daemon is the common laptop case, and discovering that only at
    build time would waste the whole image push.
    """
    import shutil

    if not shutil.which("docker"):
        return False
    try:
        return (
            _run(
                "docker", "info", "--format", "{{.ServerVersion}}", timeout=15
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


class LocalDockerBuilder:
    """Build with the local Docker daemon."""

    name = "docker"

    def __init__(self, client_factory: Any) -> None:
        self._client = client_factory

    async def build_and_push(self, request: BuildRequest) -> None:
        with materialized(request) as dockerfile:
            context = dockerfile.parent
            args = [
                "docker",
                "build",
                "--platform",
                "linux/arm64",
                "-f",
                str(dockerfile),
                "-t",
                request.image_uri,
            ]
            if request.force_build:
                args.append("--no-cache")
            args.append(str(context))
            result = await asyncio.to_thread(_run, *args, timeout=request.timeout_sec)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker build failed (exit {result.returncode}):\n"
                f"{(result.stderr or result.stdout or '').strip()[:4000]}"
            )

        await self._reject_oversized(request.image_uri)
        await asyncio.to_thread(self._login, request.registry)
        push = await asyncio.to_thread(
            _run, "docker", "push", request.image_uri, timeout=1800
        )
        if push.returncode != 0:
            raise RuntimeError(
                f"docker push to ECR failed (exit {push.returncode}):\n"
                f"{(push.stderr or push.stdout or '').strip()[:4000]}"
            )
        logger.info("Pushed AgentCore image %s (local docker)", request.image_uri)

    async def _reject_oversized(self, image_uri: str) -> None:
        inspect = await asyncio.to_thread(
            _run, "docker", "image", "inspect", "-f", "{{.Size}}", image_uri
        )
        from benchflow.sandbox.protocol import SandboxStartupError

        raw = (inspect.stdout or "").strip()
        if inspect.returncode != 0 or not raw.isdigit():
            # Failing open here pushes an image whose size is unknown; if it is
            # over the hard 2 GB cap the failure resurfaces later as an opaque
            # runtime error that reads as a task failure.
            raise SandboxStartupError(
                f"Could not measure the size of {image_uri} "
                f"(docker image inspect exited {inspect.returncode}). Refusing "
                "to push an image that cannot be checked against AgentCore's "
                f"{provisioning.MAX_IMAGE_MB} MB limit.",
                sandbox_id=image_uri,
            )
        message = provisioning.image_size_error(int(raw), image_uri)
        if message:
            raise SandboxStartupError(message, sandbox_id=image_uri)

    def _login(self, registry: str) -> None:
        import base64

        token = self._client("ecr").get_authorization_token()
        blob = token["authorizationData"][0]["authorizationToken"]
        _user, password = base64.b64decode(blob).decode().split(":", 1)
        proc = _run(
            "docker",
            "login",
            "--username",
            "AWS",
            "--password-stdin",
            registry,
            timeout=120,
            input_text=password,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ECR docker login failed: {(proc.stderr or '')[:1000]}")


# Runs on the CodeBuild worker. The size gate lives here too so an oversized
# image is rejected before the push rather than surfacing later as an opaque
# AgentCore runtime error.
_BUILDSPEC = {
    "version": "0.2",
    "phases": {
        "pre_build": {
            "commands": [
                "aws ecr get-login-password --region $AWS_REGION "
                "| docker login --username AWS --password-stdin $BF_REGISTRY",
            ]
        },
        "build": {
            "commands": [
                f"docker build --platform linux/arm64 "
                f"-f {provisioning.GENERATED_DOCKERFILE} -t $BF_IMAGE_URI .",
                'SIZE=$(docker image inspect -f "{{.Size}}" $BF_IMAGE_URI)',
                'echo "benchflow: image size $((SIZE / 1048576)) MB"',
                # Compare raw bytes, not floored megabytes: an image of
                # exactly the cap plus one byte floors to the cap and would
                # slip past a megabyte comparison that image_size_error()
                # correctly rejects.
                f'if [ "$SIZE" -gt {provisioning.MAX_IMAGE_MB * 1024 * 1024} ]; '
                'then echo "BENCHFLOW_IMAGE_TOO_LARGE $SIZE"; exit 1; fi',
                "docker push $BF_IMAGE_URI",
            ]
        },
    },
}


class CodeBuildBuilder:
    """Build on AWS CodeBuild (Graviton), requiring nothing locally."""

    name = "codebuild"

    def __init__(self, client_factory: Any, account_id: str, region: str) -> None:
        self._client = client_factory
        self._account_id = account_id
        self._region = region

    @property
    def _bucket(self) -> str:
        return (
            os.environ.get(ENV_BUILD_BUCKET)
            or f"benchflow-agentcore-build-{self._account_id}-{self._region}"
        )

    def _role_arn(self) -> str:
        role = os.environ.get(ENV_CODEBUILD_ROLE)
        if role:
            return role
        raise RuntimeError(
            "Remote image builds need a CodeBuild service role. Set "
            f"{ENV_CODEBUILD_ROLE} to a role assumable by codebuild.amazonaws.com "
            "that can push to ECR, read the build bucket, and write CloudWatch "
            "logs. (Alternatively install Docker locally and set "
            f"{ENV_BUILDER}=docker.)"
        )

    async def build_and_push(self, request: BuildRequest) -> None:
        key = f"contexts/{uuid.uuid4().hex}.zip"
        archive = await asyncio.to_thread(self._package, request)
        await asyncio.to_thread(self._ensure_bucket)
        try:
            # The upload belongs inside the cleanup scope. S3 can commit the
            # object and lose the response; deleting the key even when
            # put_object raises is the only safe handling of that ambiguity.
            upload = asyncio.create_task(
                asyncio.to_thread(
                    self._client("s3").put_object,
                    Bucket=self._bucket,
                    Key=key,
                    Body=archive,
                )
            )
            try:
                await asyncio.shield(upload)
            except asyncio.CancelledError:
                # to_thread cancellation does not stop the underlying upload.
                # Wait for it to settle before the finally block deletes the
                # key, or a late successful upload can recreate the object
                # after cleanup.
                with suppress(Exception):
                    await upload
                raise
            await asyncio.to_thread(self._ensure_project)
            await self._run_build(request, key)
        finally:
            try:
                await asyncio.to_thread(
                    self._client("s3").delete_object, Bucket=self._bucket, Key=key
                )
            except Exception as exc:
                # The archive can carry task source and credentials. Bucket
                # hardening and the one-day lifecycle bound the exposure, so
                # this must not replace a successful build result — but at
                # DEBUG a retained context is invisible for that whole day.
                logger.warning(
                    "Could not delete uploaded build context s3://%s/%s: %s. "
                    "It will expire with the bucket lifecycle policy.",
                    self._bucket,
                    key,
                    exc,
                )

    def _package(self, request: BuildRequest) -> bytes:
        """Zip the build context with the generated Dockerfile and shim inside.

        ZIP, not tar.gz: CodeBuild's S3 source type only unpacks ZIP archives.
        A tarball is downloaded verbatim, so the build directory ends up
        holding the archive itself and the build fails with a misleading
        ``Dockerfile ... no such file or directory``.
        """
        buffer = io.BytesIO()
        with (
            materialized(request) as dockerfile,
            zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive,
        ):
            context = dockerfile.parent
            for path, relative in provisioning.iter_context_entries(context):
                archive.write(
                    path,
                    arcname=relative + "/" if path.is_dir() else relative,
                )
            # The generated scaffolding is excluded from the canonical walk
            # (it must not affect image identity), so add it explicitly —
            # CodeBuild only ever sees this archive.
            for generated in (
                provisioning.GENERATED_DOCKERFILE,
                provisioning.GENERATED_SHIM,
            ):
                archive.write(context / generated, arcname=generated)
        return buffer.getvalue()

    def _ensure_bucket(self) -> None:
        """Create the build bucket if needed, and always assert its controls.

        Hardening runs on every call, not only on creation. An existing bucket
        — one a previous run created before these controls existed, or one the
        operator pointed us at — would otherwise keep uploaded build contexts
        public-by-default and un-expiring forever. Build contexts can contain
        task source and credentials, so failing to establish those controls is
        a stop condition, not a warning.
        """
        from botocore.exceptions import ClientError

        s3 = self._client("s3")
        try:
            s3.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = str(exc.response["Error"].get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                # 403 and throttling are not "absent" — creating over them
                # would fail confusingly and mask the real problem.
                raise
            kwargs: dict[str, Any] = {"Bucket": self._bucket}
            if self._region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            try:
                s3.create_bucket(**kwargs)
            except ClientError as create_exc:
                if create_exc.response["Error"]["Code"] not in {
                    "BucketAlreadyOwnedByYou",
                    "BucketAlreadyExists",
                }:
                    raise

        s3.put_public_access_block(
            Bucket=self._bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        lifecycle_rule = {
            "ID": "benchflow-expire-build-contexts",
            "Status": "Enabled",
            "Filter": {"Prefix": "contexts/"},
            "Expiration": {"Days": 1},
        }
        try:
            existing = s3.get_bucket_lifecycle_configuration(Bucket=self._bucket)
            rules = [
                rule
                for rule in existing.get("Rules", [])
                if rule.get("ID") != lifecycle_rule["ID"]
            ]
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
                raise
            rules = []
        # Preserve operator-owned lifecycle rules on a custom/shared bucket.
        # put_bucket_lifecycle_configuration replaces the entire document.
        rules.append(lifecycle_rule)
        s3.put_bucket_lifecycle_configuration(
            Bucket=self._bucket,
            LifecycleConfiguration={"Rules": rules},
        )

    def _ensure_project(self) -> None:
        from botocore.exceptions import ClientError

        codebuild = self._client("codebuild")
        config = {
            "source": {"type": "S3", "location": f"{self._bucket}/placeholder"},
            "artifacts": {"type": "NO_ARTIFACTS"},
            "environment": {
                "type": "ARM_CONTAINER",
                "image": CODEBUILD_IMAGE,
                "computeType": CODEBUILD_COMPUTE,
                # Required to run a Docker daemon inside the build container.
                "privilegedMode": True,
            },
            "serviceRole": self._role_arn(),
        }
        try:
            codebuild.create_project(name=CODEBUILD_PROJECT, **config)
            logger.info("Created CodeBuild project %s", CODEBUILD_PROJECT)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
            # This project is shared across runs. Reconcile its mutable
            # execution contract so an old project (or an operator edit)
            # cannot silently switch us to x86, disable Docker, or retain a
            # stale service role.
            codebuild.update_project(name=CODEBUILD_PROJECT, **config)
            logger.info("Updated CodeBuild project %s", CODEBUILD_PROJECT)

    async def _run_build(self, request: BuildRequest, key: str) -> None:
        codebuild = self._client("codebuild")
        start = asyncio.create_task(
            asyncio.to_thread(
                codebuild.start_build,
                projectName=CODEBUILD_PROJECT,
                sourceTypeOverride="S3",
                sourceLocationOverride=f"{self._bucket}/{key}",
                buildspecOverride=json.dumps(_BUILDSPEC),
                environmentVariablesOverride=[
                    {"name": "BF_IMAGE_URI", "value": request.image_uri},
                    {"name": "BF_REGISTRY", "value": request.registry},
                ],
                timeoutInMinutesOverride=_CODEBUILD_TIMEOUT_MIN,
            )
        )
        cancelled_during_start: asyncio.CancelledError | None = None
        try:
            started = await asyncio.shield(start)
        except asyncio.CancelledError as exc:
            # The API may have accepted the build even though our task was
            # cancelled while waiting for its response. Recover the id before
            # honoring cancellation so the remote build can be stopped.
            cancelled_during_start = exc
            started = await start
        build_id = started["build"]["id"]
        logger.info(
            "CodeBuild %s building %s (arm64, no local docker)",
            build_id,
            request.image_uri,
        )

        terminal = False

        async def _stop_unfinished() -> None:
            try:
                await asyncio.to_thread(codebuild.stop_build, id=build_id)
            except Exception as exc:
                # Preserve the timeout/cancellation/provider error that caused
                # the abort, while making a potentially still-running paid
                # build visible to the operator.
                logger.warning(
                    "Could not stop unfinished CodeBuild %s: %s", build_id, exc
                )

        if cancelled_during_start is not None:
            await asyncio.shield(_stop_unfinished())
            raise cancelled_during_start

        try:
            deadline = time.monotonic() + (
                request.timeout_sec or _CODEBUILD_TIMEOUT_MIN * 60
            )
            while time.monotonic() < deadline:
                await asyncio.sleep(_CODEBUILD_POLL_SEC)
                builds = await asyncio.to_thread(
                    codebuild.batch_get_builds, ids=[build_id]
                )
                build = builds["builds"][0]
                if build["buildStatus"] == "IN_PROGRESS":
                    continue
                terminal = True
                if build["buildStatus"] == "SUCCEEDED":
                    logger.info(
                        "Pushed AgentCore image %s (codebuild)", request.image_uri
                    )
                    return
                raise await asyncio.to_thread(
                    self._build_failure, build, request.image_uri
                )
            raise TimeoutError(
                f"CodeBuild {build_id} did not finish within the build timeout"
            )
        except asyncio.CancelledError:
            if not terminal:
                await asyncio.shield(_stop_unfinished())
            raise
        except Exception:
            if not terminal:
                await _stop_unfinished()
            raise

    def _build_failure(self, build: dict[str, Any], image_uri: str) -> Exception:
        """Turn a failed build into the most specific error we can offer."""
        from benchflow.sandbox.protocol import SandboxStartupError

        phases = build.get("phases") or []
        detail = ""
        for phase in phases:
            for context in phase.get("contexts") or []:
                message = context.get("message")
                if message:
                    detail = message
        tail = self._log_tail(build)
        if "BENCHFLOW_IMAGE_TOO_LARGE" in tail:
            return SandboxStartupError(
                provisioning.image_size_error(
                    (provisioning.MAX_IMAGE_MB + 1) * 1024 * 1024, image_uri
                )
                or "image too large",
                sandbox_id=image_uri,
            )
        return RuntimeError(
            f"CodeBuild {build['id']} failed ({build['buildStatus']}): "
            f"{detail or 'see CloudWatch logs'}\n{tail[-2000:]}"
        )

    def _log_tail(self, build: dict[str, Any]) -> str:
        logs = build.get("logs") or {}
        group, stream = logs.get("groupName"), logs.get("streamName")
        if not group or not stream:
            return ""
        try:
            events = self._client("logs").get_log_events(
                logGroupName=group, logStreamName=stream, limit=100
            )
            return "".join(e.get("message", "") for e in events.get("events", []))
        except Exception:
            return ""


def select_builder(
    client_factory: Any,
    *,
    account_id: str,
    region: str,
    preference: str | None = None,
) -> Builder:
    """Choose a build strategy.

    ``auto`` (the default) prefers a working local Docker daemon and otherwise
    builds remotely, so the backend works on a machine with no container
    runtime without the caller configuring anything extra.
    """
    choice = (preference or os.environ.get(ENV_BUILDER) or "auto").strip().lower()
    if choice not in {"auto", "docker", "codebuild"}:
        raise ValueError(
            f"Invalid {ENV_BUILDER}={choice!r}; use auto, docker, or codebuild."
        )

    if choice == "docker":
        return LocalDockerBuilder(client_factory)
    if choice == "codebuild":
        return CodeBuildBuilder(client_factory, account_id, region)

    if docker_available():
        return LocalDockerBuilder(client_factory)
    logger.info("No local Docker daemon; building images on AWS CodeBuild")
    return CodeBuildBuilder(client_factory, account_id, region)
