"""Image identity, build, and publication for the AgentCore backend.

AgentCore sessions run only images already pushed to ECR. This module owns that
entire image lifecycle so the sandbox itself can focus on runtime/session and
data-plane behavior.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchflow.sandbox import agentcore_builder as builders
from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox._base import (
    BASH_STARTUP_PREFIX,
    BASH_STARTUP_VARS,
    IMAGE_ENV_PREFIX,
    SYSTEM_SHELL_EMPTY_VARS,
    restore_image_env,
    startup_refusal,
)
from benchflow.task.config import SandboxConfig

# Stdlib-only responder for the AgentCore Runtime HTTP contract. Kept
# dependency-free so it runs on any task base image that has a ``python3``.
PING_SHIM = '''\
"""BenchFlow shim: satisfies the AgentCore Runtime HTTP contract.

AgentCore refuses to service InvokeAgentRuntimeCommand for a session whose
container does not answer this contract, so BenchFlow injects this responder
as the image entrypoint. It does no work beyond staying alive and replying;
the agent itself is launched later over the shell WebSocket.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/ping", ""):
            self._reply(200, {"status": "Healthy"})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply(200, {"result": "benchflow-sandbox"})

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
'''


# AgentCore starts every command's first shell itself, as root, with the
# image's ENV (C8): an image whose LD_PRELOAD or BASH_ENV names a file the
# agent can replace would run the agent's code there before any text
# BenchFlow sends. The image BenchFlow builds keeps those values aside under
# IMAGE_ENV_PREFIX names and sets them empty; restore_image_env gives them
# back to every command but BenchFlow's plumbing. One ENV instruction:
# Docker expands each value from the environment before it.
_STARTUP_ENV_ASIDE = (
    "# --- BenchFlow: loader and start-up values aside (C8) ---\n"
    + "ENV "
    + " ".join(
        f"{IMAGE_ENV_PREFIX}{name}=${{{name}}} {name}="
        for name in SYSTEM_SHELL_EMPTY_VARS
    )
    + "\n"
)


# bash's start-up names (an exported function, SHELLOPTS, BASHOPTS) cannot
# be set aside the same way (no ENV line can name an unknown ``BASH_FUNC_*%%``)
# or given back to a task's command, so an image that sets one to a value is
# refused at build (``startup_name``). Exec form and the
# shim's python3, so no shell of the image runs the check.
_STARTUP_NAMES_CHECK = (
    "# --- BenchFlow: no bash start-up names in the image's ENV (C8) ---\n"
    + "RUN "
    + json.dumps(
        [
            "python3",
            "-c",
            "import os, sys\n"
            "bad = sorted(n for n, v in os.environ.items() if v and "
            f"(n.startswith({BASH_STARTUP_PREFIX!r}) or n in {BASH_STARTUP_VARS!r}))\n"
            "if bad:\n"
            f"    sys.exit({startup_refusal(['{names}'], 'the AgentCore image')!r}"
            ".replace('{names}', ', '.join(bad)))\n",
        ]
    )
    + "\n"
)


# The ping shim is the image's own ``python3``, started when the session
# starts, before any agent turn: it gets the image's loader values back (a
# Python that finds libpython only through the image's LD_LIBRARY_PATH) in a
# subshell, so only the shim has them, not PID 1: the ``/bin/sh`` that
# passes a stop on to the shim. Every command the runtime starts later still
# begins with them empty (the ENV line above).
_SHIM_PATH = "/opt/benchflow_agentcore_shim.py"
_SHIM_START = (
    f"({restore_image_env()}exec python3 {_SHIM_PATH}) & "
    'pid=$!; trap \'kill "$pid"\' TERM INT; wait "$pid"'
)
_SHIM_CMD = "CMD " + json.dumps(["/bin/sh", "-c", _SHIM_START]) + "\n"


class AgentCoreImagePublisher:
    """Publish one task context to ECR and return its immutable image URI."""

    def __init__(
        self,
        *,
        environment_dir: Path,
        environment_name: str,
        task_env_config: SandboxConfig,
        region: str,
        ecr_repository: str,
        client_factory: Callable[[str], Any],
        logger: Any,
    ) -> None:
        self._environment_dir = environment_dir
        self._environment_name = environment_name
        self._task_env_config = task_env_config
        self._region = region
        self._ecr_repository = ecr_repository
        self._client = client_factory
        self._logger = logger
        self._account_id: str | None = None

    def _resolve_account_id(self) -> str:
        if self._account_id is None:
            self._account_id = self._client("sts").get_caller_identity()["Account"]
        return self._account_id

    def _registry(self) -> str:
        return f"{self._resolve_account_id()}.dkr.ecr.{self._region}.amazonaws.com"

    def identity(self) -> tuple[str, str]:
        """Return ``(context digest, ECR tag)`` for this task image."""
        digest = provisioning.build_context_digest(
            self._environment_dir,
            self.generated_dockerfile_text(),
            PING_SHIM,
        )
        return digest, provisioning.image_tag(self._environment_name, digest)

    async def publish(self, *, force_build: bool) -> str:
        """Build/push the image once and return an immutable digest URI."""
        _digest, tag = await asyncio.to_thread(self.identity)
        registry = await asyncio.to_thread(self._registry)
        tagged_uri = f"{registry}/{self._ecr_repository}:{tag}"
        cache_key = f"image:{tagged_uri}:{force_build}"

        async def _publish() -> str:
            await asyncio.to_thread(self._ensure_ecr_repository)
            if force_build or not await asyncio.to_thread(self._image_exists, tag):
                await self._build_and_push(tagged_uri, force_build=force_build)
            else:
                self._logger.info("Reusing published AgentCore image %s", tagged_uri)
            return await asyncio.to_thread(self._resolve_image_digest, tag)

        return await provisioning.once(cache_key, _publish)

    def _resolve_image_digest(self, tag: str) -> str:
        response = self._client("ecr").describe_images(
            repositoryName=self._ecr_repository, imageIds=[{"imageTag": tag}]
        )
        details = response.get("imageDetails") or []
        if not details or not details[0].get("imageDigest"):
            raise RuntimeError(
                f"ECR did not report a digest for {self._ecr_repository}:{tag}; "
                "cannot bind an AgentCore runtime to a verifiable image."
            )
        return f"{self._registry()}/{self._ecr_repository}@{details[0]['imageDigest']}"

    async def _build_and_push(self, image_uri: str, *, force_build: bool) -> None:
        builder = builders.select_builder(
            self._client,
            account_id=self._resolve_account_id(),
            region=self._region,
        )
        self._logger.info("Building %s via %s", image_uri, builder.name)
        await builder.build_and_push(
            builders.BuildRequest(
                context_dir=self._environment_dir,
                dockerfile_text=self.generated_dockerfile_text(),
                shim_text=PING_SHIM,
                image_uri=image_uri,
                registry=self._registry(),
                region=self._region,
                force_build=force_build,
                timeout_sec=self._task_env_config.build_timeout_sec,
            )
        )

    def generated_dockerfile_text(self) -> str:
        """Return the Dockerfile BenchFlow builds without touching the task."""
        if self._task_env_config.docker_image:
            base = f"FROM {self._task_env_config.docker_image}\n"
        else:
            base = provisioning.read_regular_text(self._environment_dir / "Dockerfile")
        return (
            base
            + "\n"
            + "# Sealed uploads decrypt with openssl inside the runtime;\n"
            + "# install it when the base image lacks it and a package\n"
            + "# manager exists (fails loudly at runtime otherwise).\n"
            + "RUN command -v openssl >/dev/null 2>&1 || "
            + "(apt-get update -qq && apt-get install -y -qq openssl) || "
            + "dnf -y install openssl || apk add --no-cache openssl || true\n"
            + _STARTUP_NAMES_CHECK
            + _STARTUP_ENV_ASIDE
            + "# --- BenchFlow AgentCore runtime contract ---\n"
            + "# AgentCore refuses command execution for a session whose\n"
            + "# container does not answer GET /ping on :8080.\n"
            + f"COPY {provisioning.GENERATED_SHIM} {_SHIM_PATH}\n"
            + "EXPOSE 8080\n"
            + "ENTRYPOINT []\n"
            + _SHIM_CMD
        )

    def _ensure_ecr_repository(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self._client("ecr").create_repository(repositoryName=self._ecr_repository)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "RepositoryAlreadyExistsException":
                raise

    def _image_exists(self, tag: str) -> bool:
        """Return false only for a genuine ECR image/repository miss."""
        from botocore.exceptions import ClientError

        try:
            self._client("ecr").describe_images(
                repositoryName=self._ecr_repository,
                imageIds=[{"imageTag": tag}],
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {
                "ImageNotFoundException",
                "RepositoryNotFoundException",
            }:
                return False
            raise
