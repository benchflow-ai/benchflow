"""Resolve an optional verifier cache build without changing ordinary builds.

Resolution either supplies a certified plain plugin, all its inputs and their
identity, or refuses caching. The original Docker launcher always owns ordinary
and fallback builds. Values stay in memory, never in logs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandbox.env_layers import LayerError
from benchflow.sandbox.verifier_layers import compose_builder_identity

_INFO_FORMAT = (
    '{"ID":{{json .ID}},"OSType":{{json .OSType}},'
    '"Architecture":{{json .Architecture}},"ServerVersion":{{json .ServerVersion}},'
    '"ClientInfo":{{json .ClientInfo}},"ServerErrors":{{json .ServerErrors}},'
    '"ClientErrors":{{json .ClientErrors}}}'
)

# Native Linux amd64 release artifacts, not a filename, shebang or --version
# heuristic: even an ELF launcher can be a wrapper that changes build inputs.
# Unknown distributor builds and releases still work through ordinary Docker.
# Docker: https://download.docker.com/linux/static/stable/x86_64/docker-29.1.3.tgz
# Compose/Buildx: their v2.35.1/v0.30.1 GitHub release checksums, respectively.
_CERTIFIED_TOOLS = {
    "docker": {
        "226408f543344f0d2bfc84c7df4243c5364baccf509e8984d04e1e62c74efac0": "29.1.3",
    },
    "compose": {
        "7bdb2ce2916e5dd0354e5d129892bf96fdcdb1a9ab8eed69b9173e131db4c230": "2.35.1",
    },
    "buildx": {
        "c37114fcd034025ec68e224657c8a5a850df472ded3ddcbca75ad3a7ebb9710d": "0.30.1",
    },
}


@dataclass(frozen=True)
class ComposeBuild:
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = field(repr=False)
    cwd: str
    inputs: str = field(default="", repr=False)
    uncertainty: str | None = "unresolved Compose build invocation"

    def identity(self) -> str:
        if self.uncertainty is not None:
            raise LayerError(self.uncertainty)
        material = json.dumps(
            [self.argv, self.environment, self.cwd, self.inputs],
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode()).hexdigest()


def _executable(path: str) -> tuple[str, str]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("unavailable executable")
    return str(resolved), hashlib.sha256(resolved.read_bytes()).hexdigest()


def _certified_tool(name: str, digest: str, version: str | None = None) -> str:
    certified = _CERTIFIED_TOOLS[name].get(digest)
    if certified is None or (
        version is not None and version.removeprefix("v") != certified
    ):
        raise ValueError("unresolved or wrapped Docker toolchain")
    return certified


async def _output(argv: list[str], env: dict[str, str], cwd: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise ValueError("build resolution timed out") from None
    if process.returncode:
        raise ValueError("build resolution failed")
    return stdout.decode()


async def _json(argv: list[str], env: dict[str, str], cwd: str) -> Any:
    return json.loads(await _output(argv, env, cwd))


async def resolve_compose_build(
    argv: list[str],
    env: dict[str, str],
    files: list[Path],
    project_dir: Path,
) -> ComposeBuild:
    """Freeze a cache build only when no launcher/plugin wrapper can intervene.

    Docker invokes a plugin with its complete original argument list and
    DOCKER_CLI_PLUGIN_ORIGINAL_CLI_COMMAND. Use that same protocol with the
    resolved absolute plugin path, avoiding a second plugin discovery at use.
    Only the certified release binaries and explicit internal BuildKit frontend
    qualify. Every failure leaves the original argv/environment untouched.
    """
    cwd = str(Path.cwd().resolve())
    env = dict(env)
    ordinary = ComposeBuild(tuple(argv), tuple(sorted(env.items())), cwd)
    try:
        index = argv.index("compose")
        docker = shutil.which(argv[0], path=env.get("PATH", os.defpath))
        if docker is None:
            raise ValueError("Docker executable unavailable")
        docker_path, docker_hash = _executable(docker)
        docker_version = _certified_tool("docker", docker_hash)
        prefix = [docker_path, *argv[1:index]]
        info = await _json([*prefix, "info", "--format", _INFO_FORMAT], env, cwd)
        if (
            not isinstance(info, dict)
            or info.get("ClientErrors")
            or info.get("ServerErrors")
        ):
            raise ValueError("invalid Docker info")
        plugins = info["ClientInfo"]["Plugins"]
        if not isinstance(plugins, list) or any(
            not isinstance(p, dict) for p in plugins
        ):
            raise ValueError("invalid Docker plugins")
        tools = {}
        for name in ("compose", "buildx"):
            matches = [p for p in plugins if p.get("Name") == name]
            if len(matches) != 1 or matches[0].get("Err"):
                raise ValueError("ambiguous Docker plugin")
            plugin = matches[0]
            path, digest = _executable(plugin["Path"])
            if not isinstance(plugin["Version"], str) or not plugin["Version"]:
                raise ValueError("unknown Docker plugin version")
            if name == "compose" and plugin["Version"].removeprefix("v") != "2.35.1":
                raise LayerError(
                    "verifier context builder is unresolved for this Compose version"
                )
            _certified_tool(name, digest, plugin["Version"])
            tools[name] = [path, digest, plugin["Version"]]
        plugin_env = {**env, "DOCKER_CLI_PLUGIN_ORIGINAL_CLI_COMMAND": docker_path}
        # Pin exactly the plugin Docker selected with these flags/environment.
        command = [tools["compose"][0], *argv[1:]]
        builder = compose_builder_identity(env)
        # Only daemon/context selectors are understood global options. Anything
        # else (including every --config spelling) remains an ordinary build.
        # Never read config.json: it may hold credentials and build proxies.
        flags = iter(argv[1:index])
        for flag in flags:
            if flag in {"--host", "-H", "--context", "-c"}:
                value = next(flags, None)
                if value is None:
                    raise ValueError("incomplete Docker global option")
                if flag in {"--host", "-H"} and not value.startswith("unix://"):
                    raise ValueError("unresolved Docker host transport")
            elif flag.startswith("--host="):
                if not flag.removeprefix("--host=").startswith("unix://"):
                    raise ValueError("unresolved Docker host transport")
            elif not flag.startswith(("--host=", "--context=")):
                raise ValueError("unresolved Docker global option")
        root = env.get("DOCKER_CONFIG")
        if not root:
            if not env.get("HOME"):
                raise ValueError("unresolved Docker configuration directory")
            root = str(Path(env["HOME"]) / ".docker")
        config = Path(root) / "config.json"
        if config.exists() or config.is_symlink():
            raise ValueError("unresolved saved Docker build settings")
        if (
            info.get("OSType") != "linux"
            or info.get("Architecture") not in {"x86_64", "amd64"}
            or not info.get("ID")
            or not info.get("ServerVersion")
        ):
            raise ValueError("unknown Docker daemon identity/platform")
        if (
            env.get("DOCKER_DEFAULT_PLATFORM")
            or (env.get("DOCKER_HOST") and not env["DOCKER_HOST"].startswith("unix://"))
            or env.get("BUILDX_BUILDER", "") not in {"", "default"}
            or env.get("BUILDKIT_HOST")
            or "COMPOSE_ENV_FILES" in env
            or "EXPERIMENTAL_BUILDKIT_SOURCE_POLICY" in env
            or any(
                name in env
                for name in (
                    "BUILDX_CONFIG",
                    "DOCKER_CERT_PATH",
                    "DOCKER_TLS_VERIFY",
                    "LD_PRELOAD",
                    "LD_LIBRARY_PATH",
                )
            )
        ):
            raise ValueError("unresolved builder inputs")
        for directory in {Path(cwd), project_dir}:
            dotenv = directory / ".env"
            if dotenv.exists() or dotenv.is_symlink():
                raise ValueError("unresolved Compose dotenv")
        inspection = await _output(
            [
                tools["buildx"][0],
                *argv[1:index],
                "buildx",
                "inspect",
            ],
            plugin_env,
            cwd,
        )
        header, separator, nodes = inspection.partition("\nNodes:")
        if (
            not separator
            or re.findall(r"^Name:[ \t]+([^\r\n]+)$", header, re.M) != ["default"]
            or re.findall(r"^Driver:[ \t]+([^\r\n]+)$", header, re.M) != ["docker"]
            or len(re.findall(r"^Endpoint:[ \t]+\S+", nodes, re.M)) != 1
            or len(re.findall(r"^BuildKit(?: version)?:[ \t]+\S+", nodes, re.M)) != 1
            or re.findall(r"^Status:[ \t]+([^\r\n]+)$", nodes, re.M) != ["running"]
        ):
            raise ValueError("unresolved default builder")
        # Last Activity is build telemetry, not an input. Retain all node
        # configuration, including endpoint, BuildKit version and worker labels.
        selected = ["default", "docker", nodes]
        contexts = await _json([*prefix, "context", "inspect"], env, cwd)
        if not isinstance(contexts, list) or len(contexts) != 1:
            raise ValueError("unresolved Docker context")
        context = contexts[0]
        if (
            not isinstance(context, dict)
            or not isinstance(context.get("Name"), str)
            or not context["Name"]
            or not isinstance(context.get("Endpoints"), dict)
            or not isinstance(context["Endpoints"].get("docker"), dict)
            or not isinstance(context["Endpoints"]["docker"].get("Host"), str)
            or not context["Endpoints"]["docker"]["Host"]
            or not context["Endpoints"]["docker"]["Host"].startswith("unix://")
            or context.get("TLSMaterial")
            or context["Endpoints"]["docker"].get("SkipTLSVerify")
        ):
            raise ValueError("unresolved Docker context endpoint")
        material = {
            "docker": [docker_path, docker_hash, info["ClientInfo"].get("Version")],
            "plugins": tools,
            "daemon": [info["ID"], info["ServerVersion"]],
            "builder": [builder, selected],
            "contexts": contexts,
            "compose_files": [
                [
                    str(p.resolve(strict=True)),
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                ]
                for p in files
            ],
        }
        if material["docker"][2] != docker_version:
            raise ValueError("unknown Docker client version")
        return ComposeBuild(
            tuple(command),
            tuple(sorted(plugin_env.items())),
            cwd,
            json.dumps(material, sort_keys=True, separators=(",", ":")),
            None,
        )
    except LayerError as exc:
        reason = str(exc)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        reason = "verifier context build invocation could not be resolved; use the ordinary build path"
    return ComposeBuild(ordinary.argv, ordinary.environment, cwd, uncertainty=reason)
