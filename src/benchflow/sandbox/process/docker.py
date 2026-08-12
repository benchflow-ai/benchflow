"""Live stdio to a container managed by local Docker Compose."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from typing import Any

from benchflow.sandbox.process._base import _BUFFER_LIMIT, SubprocessLiveProcess

logger = logging.getLogger(__name__)


class DockerProcess(SubprocessLiveProcess):
    """Live stdin/stdout via `docker compose exec -i`."""

    def __init__(
        self,
        project_name: str,
        project_dir: str,
        compose_files: list[str],
        service: str = "main",
    ):
        self._project_name = project_name
        self._project_dir = project_dir
        self._compose_files = compose_files
        self._service = service

    @classmethod
    def from_sandbox_env(cls, env: Any, service: str = "main") -> DockerProcess:
        """Create from a sandbox environment (DockerSandbox).

        ``service`` selects which compose service the agent process runs
        in. Defaults to ``"main"``. Multi-container (vulhub-style) tasks
        may run the agent in a dedicated attacker container — e.g. a Kali
        service — while keeping target containers separate (#248).
        """
        project_name = env.session_id.lower().replace(".", "-")
        project_dir = str(env.environment_dir.resolve().absolute())
        compose_files = [str(p.resolve().absolute()) for p in env._docker_compose_paths]
        return cls(
            project_name=project_name,
            project_dir=project_dir,
            compose_files=compose_files,
            service=service,
        )

    def _compose_cmd(self) -> list[str]:
        """Base docker compose command with project/file flags."""
        cmd = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._project_dir,
        ]
        for f in self._compose_files:
            cmd.extend(["-f", f])
        return cmd

    def _host_env(self) -> dict[str, str]:
        """Host process env with DOCKER_HOST resolved if needed."""
        proc_env = os.environ.copy()
        if not proc_env.get("DOCKER_HOST"):
            import subprocess

            try:
                r = subprocess.run(
                    [
                        "docker",
                        "context",
                        "inspect",
                        "--format",
                        "{{.Endpoints.docker.Host}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip():
                    proc_env["DOCKER_HOST"] = r.stdout.strip()
            except Exception:
                logger.debug("Could not inspect docker context", exc_info=True)
        return proc_env

    _ENV_PATH = "/tmp/.benchflow_env"

    async def _write_env_to_container(
        self,
        env: dict[str, str],
        proc_env: dict[str, str],
    ) -> None:
        """Write env vars to a file inside the container (not visible in ps aux)."""
        lines = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
        write_cmd = self._compose_cmd()
        write_cmd.extend(
            [
                "exec",
                "-T",
                self._service,
                "bash",
                "-c",
                f"cat > {self._ENV_PATH} && chmod 600 {self._ENV_PATH}",
            ]
        )
        proc = await asyncio.create_subprocess_exec(
            *write_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(lines.encode()), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to write env file in container (rc={proc.returncode}): "
                f"{stderr.decode()[:500]}"
            )
        logger.debug("Env file written inside container (%d vars)", len(env))

    async def _remove_env_from_container(self, proc_env: dict[str, str]) -> None:
        """Remove a staged env file when the live process cannot start."""
        cleanup_cmd = self._compose_cmd()
        cleanup_cmd.extend(["exec", "-T", self._service, "rm", "-f", self._ENV_PATH])
        proc = await asyncio.create_subprocess_exec(
            *cleanup_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to remove staged Docker env file "
                f"(rc={proc.returncode}): {stderr.decode()[:500]}"
            )

    async def start(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        proc_env = await asyncio.to_thread(self._host_env)

        # Write env vars to a file inside the container, then source it
        # in the main command. This keeps secrets off `ps aux` on the host
        # and avoids `--env-file` (not supported in all Compose versions).
        if env:
            await self._write_env_to_container(env, proc_env)
            command = f"source {self._ENV_PATH} && rm -f {self._ENV_PATH} && {command}"

        cmd = self._compose_cmd()
        cmd.extend(["exec", "-i", "-T"])
        if cwd:
            cmd.extend(["-w", cwd])
        cmd.extend([self._service, "bash", "-c", command])

        logger.debug(f"DockerProcess: {' '.join(cmd[:10])}...")
        try:
            self._set_process(
                await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=proc_env,
                    limit=_BUFFER_LIMIT,
                )
            )
        except BaseException:
            if env:
                try:
                    await asyncio.shield(self._remove_env_from_container(proc_env))
                except Exception:
                    logger.warning(
                        "Could not remove staged Docker env after launch failure",
                        exc_info=True,
                    )
            raise
        logger.info(
            f"Docker process started (pid={self._process.pid}, "
            f"project={self._project_name})"
        )
