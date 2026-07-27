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
import subprocess
import time
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
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
    """Write the generated Dockerfile and shim, yield the Dockerfile, clean up.

    The build context is the task's own environment directory, so these files
    must not survive the build — otherwise every run would litter the caller's
    task tree.
    """
    context = request.context_dir
    shim = context / provisioning.GENERATED_SHIM
    dockerfile = context / provisioning.GENERATED_DOCKERFILE
    shim.write_text(request.shim_text)
    dockerfile.write_text(request.dockerfile_text)
    try:
        yield dockerfile
    finally:
        for path in (dockerfile, shim):
            path.unlink(missing_ok=True)


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
            args.append(str(request.context_dir))
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
        raw = (inspect.stdout or "").strip()
        if inspect.returncode != 0 or not raw.isdigit():
            logger.debug("Could not measure image size for %s", image_uri)
            return
        message = provisioning.image_size_error(int(raw), image_uri)
        if message:
            from benchflow.sandbox.protocol import SandboxStartupError

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
                "MB=$((SIZE / 1048576))",
                'echo "benchflow: image size ${MB} MB"',
                f'if [ "$MB" -gt {provisioning.MAX_IMAGE_MB} ]; then '
                f'echo "BENCHFLOW_IMAGE_TOO_LARGE ${{MB}}"; exit 1; fi',
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
        await asyncio.to_thread(
            self._client("s3").put_object,
            Bucket=self._bucket,
            Key=key,
            Body=archive,
        )
        try:
            await asyncio.to_thread(self._ensure_project)
            await self._run_build(request, key)
        finally:
            try:
                await asyncio.to_thread(
                    self._client("s3").delete_object, Bucket=self._bucket, Key=key
                )
            except Exception as exc:
                logger.debug("Could not delete build context %s: %s", key, exc)

    def _package(self, request: BuildRequest) -> bytes:
        """Zip the build context with the generated Dockerfile and shim inside.

        ZIP, not tar.gz: CodeBuild's S3 source type only unpacks ZIP archives.
        A tarball is downloaded verbatim, so the build directory ends up
        holding the archive itself and the build fails with a misleading
        ``Dockerfile ... no such file or directory``.
        """
        buffer = io.BytesIO()
        with (
            materialized(request),
            zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive,
        ):
            for path in sorted(request.context_dir.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    # Symlinks are skipped for the same reason as on upload
                    # (#411): they must not pull host files into the image.
                    continue
                archive.write(
                    path, arcname=path.relative_to(request.context_dir).as_posix()
                )
        return buffer.getvalue()

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        s3 = self._client("s3")
        try:
            s3.head_bucket(Bucket=self._bucket)
            return
        except ClientError:
            pass
        kwargs: dict[str, Any] = {"Bucket": self._bucket}
        if self._region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
        try:
            s3.create_bucket(**kwargs)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                raise
        for call, kw in (
            (
                s3.put_public_access_block,
                {
                    "Bucket": self._bucket,
                    "PublicAccessBlockConfiguration": {
                        "BlockPublicAcls": True,
                        "IgnorePublicAcls": True,
                        "BlockPublicPolicy": True,
                        "RestrictPublicBuckets": True,
                    },
                },
            ),
            (
                s3.put_bucket_lifecycle_configuration,
                {
                    "Bucket": self._bucket,
                    # Build contexts are consumed within minutes; expiring them
                    # keeps a benchmark run from accruing storage cost forever.
                    "LifecycleConfiguration": {
                        "Rules": [
                            {
                                "ID": "benchflow-expire-build-contexts",
                                "Status": "Enabled",
                                "Filter": {"Prefix": "contexts/"},
                                "Expiration": {"Days": 1},
                            }
                        ]
                    },
                },
            ),
        ):
            try:
                call(**kw)
            except Exception as exc:
                logger.debug("Bucket hardening step skipped: %s", exc)

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

    async def _run_build(self, request: BuildRequest, key: str) -> None:
        codebuild = self._client("codebuild")
        started = await asyncio.to_thread(
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
        build_id = started["build"]["id"]
        logger.info(
            "CodeBuild %s building %s (arm64, no local docker)",
            build_id,
            request.image_uri,
        )

        deadline = time.monotonic() + (
            request.timeout_sec or _CODEBUILD_TIMEOUT_MIN * 60
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(_CODEBUILD_POLL_SEC)
            builds = await asyncio.to_thread(codebuild.batch_get_builds, ids=[build_id])
            build = builds["builds"][0]
            if build["buildStatus"] == "IN_PROGRESS":
                continue
            if build["buildStatus"] == "SUCCEEDED":
                logger.info("Pushed AgentCore image %s (codebuild)", request.image_uri)
                return
            raise self._build_failure(build, request.image_uri)
        raise TimeoutError(
            f"CodeBuild {build_id} did not finish within the build timeout"
        )

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
