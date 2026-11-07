"""What a task's Compose files take from the machine ``docker compose`` runs on.

Docker Compose fills ``${VAR}`` in every file it reads (the task's
``environment/docker-compose.yaml``, the files it ``include``s or
``extends``, the project's ``.env`` and every ``env_file``) from the
environment of the ``docker compose`` process; an ``environment:`` entry with
no value copies the variable itself into the container; and a bind mount,
``env_file``, build context or secret file may name any path on the machine.
BenchFlow ran Compose with the operator's whole environment, so a task's
Compose file could hand ``ANTHROPIC_API_KEY`` to the agent's container or
mount ``~/.config/gcloud``.

A task's Compose project is checked once and launched as checked, on the backends whose Compose runs on this machine
(:data:`COMPOSE_ON_HOST`):

* The sandbox works on a private, no-follow copy of the task's environment
  folder (:mod:`benchflow.task.staging`), never on the task's own files.
* The copy's Compose files, and every file they make Compose read before it
  resolves anything (includes, extends, env and label files, ``.env``
  files), are read strictly first (:mod:`benchflow.task.compose_load`):
  each must be in the copy, keys must be text, tags core YAML, and keys ones
  BenchFlow checks.
* ``docker compose config`` then resolves the project once, with the run's
  files, project name and values, in one hermetic environment
  (:func:`compose_process_env`: PATH and HOME of the operator's, the
  variables named with ``--allow-task-env``, the task's values, BenchFlow's
  own) and a docker client configuration without the operator's proxies
  (:func:`hermetic_docker_config`). Where the daemon is goes on the command
  line (:func:`docker_connection_args`).
* The resolved project, its ``$$`` escaping undone, is checked for what
  reaches this machine (:class:`_ProjectCheck`): bind and volume sources,
  bind-backed and reused volumes, build contexts and Dockerfiles, secret and
  config files, and the service settings that hand a container this machine.
  No writable mount may contain another launch source's directory entry or
  an ancestor, including through aliases and bind-backed volume subpaths:
  an agent must not replace those entries before a restart or restore.
* The checked project is written to a private file, Compose is asked to read
  that file back and must resolve it to the same project, and every Compose
  call of the sandbox uses only that file and the same environment
  (:func:`freeze_compose`): nothing a task changes after the check reaches
  the launch. The operator's opt-in :data:`ALLOW_HOST_ACCESS_VAR` clears the
  host-reach refusals.

On remote-docker every bind mount, bind-backed volume and file secret or
config is a path on the remote host, never the task's files, so each is
refused. Daytona's DinD Compose runs in the sandbox's VM: the same strict
reading and, where ``docker compose`` is installed here, the same resolved
check run before the sandbox is created; :func:`check_compose_in_vm` checks
the project Compose resolves in the VM before the task's pre-compose hook
runs and after it.

A task's value for one of BenchFlow's own Compose variables
(:data:`BENCHFLOW_COMPOSE_VARIABLES`) is always refused on docker and
remote-docker.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandbox._compose import (
    DIND_ENVIRONMENT_DIR,
    DIND_LOGS_DIR,
    DIND_TASK_DIR,
)
from benchflow.task.env import TaskEnvRequest, allowed_task_env

COMPOSE_FILE = "docker-compose.yaml"
PRE_COMPOSE_HOOK = "benchflow-pre-compose.sh"

#: Backends whose ``docker compose`` runs on this machine, with its
#: environment; Daytona's runs inside the sandbox and sees none of it.
COMPOSE_ON_HOST = ("docker", "remote-docker")

#: Backends whose Compose project the launch gate checks: the ones above,
#: and Daytona, whose DinD Compose reads only the uploaded environment
#: folder (its resolved project is checked in the VM).
COMPOSE_CHECKED = (*COMPOSE_ON_HOST, "daytona")

#: The operator's opt-in for a task that reaches outside its folder on
#: this machine: a Compose path outside it, or a pre-compose hook.
ALLOW_HOST_ACCESS_VAR = "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"
_HOST_REMEDY = (
    "A task's Compose files may reach only files in the task's folder (no "
    "remote files, privileged containers, host namespaces, host devices or "
    "added capabilities), and its benchflow-pre-compose.sh runs on this "
    f"machine as you: edit the task, or set {ALLOW_HOST_ACCESS_VAR}=1 if you "
    "trust it with this machine"
)

_BENCHFLOW_REMEDY = (
    "BenchFlow sets its own Compose variables (the image it builds, the host "
    "folders it mounts, CPUS and MEMORY) and a task's value for one is not "
    "used: take it out of the task's [environment] env"
)

_HOST_PROGRAM_REMEDY = (
    "Compose runs credential helpers, git and ssh on this machine as you, "
    "and these variables steer them, so a task's value for one is not "
    "given to Compose: write the value in the Compose file itself, or "
    "give the variable another name"
)

_CONFIG_REMEDY = (
    "BenchFlow checks a task's Compose files with `docker compose config`, "
    "run as the task's own run is: with PATH, HOME and the docker connection "
    "variables of yours, the variables you named, and the task's values. "
    "Fix the files, or name a variable they need with --allow-task-env"
)

#: What BenchFlow puts in Compose's environment itself (``DockerSandboxEnvVars``
#: and the process token of the leftover sweep).
BENCHFLOW_COMPOSE_VARIABLES = frozenset(
    {
        "MAIN_IMAGE_NAME",
        "CONTEXT_DIR",
        "HOST_VERIFIER_LOGS_PATH",
        "HOST_AGENT_LOGS_PATH",
        "HOST_ARTIFACTS_PATH",
        "ENV_VERIFIER_LOGS_PATH",
        "ENV_AGENT_LOGS_PATH",
        "ENV_ARTIFACTS_PATH",
        "PREBUILT_IMAGE_NAME",
        "CPUS",
        "MEMORY",
        "BENCHFLOW_PROCESS",
    }
)

#: All ``docker compose`` keeps of the operator's environment: what the
#: docker CLI needs to start and find its plugins and config. No credential
#: (a registry login stays in the docker config's credential store;
#: ``DOCKER_AUTH_CONFIG``, which holds one, is left out) and no connection
#: setting: where the daemon is goes on the command line
#: (:func:`docker_connection_args`), so no Compose file can read it.
_OPERATOR_NAMES = (
    "PATH",
    "HOME",
    "DOCKER_BUILDKIT",
    "DOCKER_DEFAULT_PLATFORM",
)
#: The docker CLI's connection settings, given to it as flags instead.
_CONNECTION_NAMES = frozenset(
    {
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_API_VERSION",
    }
)
#: Windows: the CLI cannot start or find its config without these.
_WINDOWS_NAMES = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
)

#: Compose's switches for fetching git and OCI includes, which it does by
#: default (Compose 2.40.3 and v5.6.0 both fetch with neither set; ``false``
#: turns each off, and the process environment beats a task's ``.env``).
_REMOTE_INCLUDE_SWITCHES = (
    "COMPOSE_EXPERIMENTAL_GIT_REMOTE",
    "COMPOSE_EXPERIMENTAL_OCI_REMOTE",
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


# Variables that steer the programs Compose runs on this machine as the
# operator (the docker CLI, credential helpers such as docker-credential-pass
# and -gcloud, the dynamic loader, the git and ssh it calls, Go's runtime): a
# task's value for one never reaches Compose unless the operator names it.
_HOST_PROGRAM_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "PWD",
        "LANG",
        "BASH_ENV",
        "ENV",
        "IFS",
        "PS4",
        "SHELLOPTS",
        "BASHOPTS",
        "PROMPT_COMMAND",
        "GCONV_PATH",
        "HOSTALIASES",
        "LOCALDOMAIN",
        "RES_OPTIONS",
        "KUBECONFIG",
        "DBUS_SESSION_BUS_ADDRESS",
        "NETRC",
        "CURL_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "GODEBUG",
        "GOTRACEBACK",
        "GOMAXPROCS",
        "GOMEMLIMIT",
        "GOGC",
        *_WINDOWS_NAMES,
    }
)
_HOST_PROGRAM_PREFIXES = (
    "LD_",
    "DYLD_",
    "DOCKER_",
    "COMPOSE_",
    "BUILDKIT_",
    "BUILDX_",
    "SSH_",
    "XDG_",
    "SSL_",
    "OPENSSL_",
    "KRB5",
    "PASSWORD_STORE_",
    "GPG",
    "GNUPG",
    "CLOUDSDK_",
    "GCLOUD_",
    "GOOGLE_",
    "AWS_",
    "AZURE_",
    "GIT_",
    "PYTHON",
    "NODE_",
    "PERL",
    "RUBY",
    "BASH_FUNC_",
)


def host_program_variable(name: str) -> bool:
    """A variable that steers a program Compose runs on this machine."""
    return name in _HOST_PROGRAM_NAMES or name.startswith(_HOST_PROGRAM_PREFIXES)


def host_access_allowed() -> bool:
    """Whether the operator set :data:`ALLOW_HOST_ACCESS_VAR`."""
    return os.environ.get(ALLOW_HOST_ACCESS_VAR, "").strip().lower() in _TRUTHY


def compose_process_env(
    task_values: Mapping[str, str] | None = None,
    benchflow_values: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    allowed: Iterable[str] | None = None,
    host_access: bool | None = None,
) -> dict[str, str]:
    """The whole environment of ``docker compose`` (and a pre-compose hook).

    Of the operator's environment (*environ*): PATH and HOME (and the
    BuildKit and platform settings), and the variables named with
    ``--allow-task-env`` (*allowed*), never a connection setting, which goes
    on the command line (:func:`docker_connection_args`). Then the task's values (*task_values*), except one that
    steers a program Compose runs here (:func:`host_program_variable`) and
    was not named; then BenchFlow's (*benchflow_values*), which a task cannot
    replace. Without the host-access opt-in, Compose's git and OCI include
    fetches are turned off.
    """
    source = os.environ if environ is None else environ
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    if host_access is None:
        host_access = host_access_allowed()
    kept = _OPERATOR_NAMES + (_WINDOWS_NAMES if os.name == "nt" else ())
    env = {name: source[name] for name in kept if name in source}
    env.update(
        (name, source[name])
        for name in sorted(names - _CONNECTION_NAMES)
        if name in source
    )
    env.update(
        (key, value)
        for key, value in (task_values or {}).items()
        if key not in _CONNECTION_NAMES
        and (key in names or not host_program_variable(key))
    )
    env.update(benchflow_values or {})
    # Values are opaque, even when they start with / (S1-CONF-1). A named
    # value can supply both a host path and secret/config/container content.
    # Only resolved path consumers in freeze_compose may use its realpath.
    if not host_access:
        env.update(dict.fromkeys(_REMOTE_INCLUDE_SWITCHES, "false"))
    return env


def docker_tls_args(
    cert_path: str | None, verify: bool, tls: bool = False
) -> list[str]:
    """The docker CLI's TLS flags for a certificate folder (``ca.pem``,
    ``cert.pem``, ``key.pem``), as ``DOCKER_CERT_PATH`` and
    ``DOCKER_TLS_VERIFY`` would set them."""
    if not (verify or tls):
        return []
    args = ["--tlsverify"] if verify else ["--tls"]
    if cert_path:
        args += [
            "--tlscacert",
            os.path.join(cert_path, "ca.pem"),
            "--tlscert",
            os.path.join(cert_path, "cert.pem"),
            "--tlskey",
            os.path.join(cert_path, "key.pem"),
        ]
    return args


def docker_connection_args(environ: Mapping[str, str] | None = None) -> list[str]:
    """The docker CLI's global flags for the daemon the operator's
    environment names (``docker <flags> compose ...``): ``DOCKER_CONFIG``,
    then ``DOCKER_HOST`` with its TLS settings, else ``DOCKER_CONTEXT``.
    Compose 2.40.3 and v5.6.0 take them from the command line, so the
    connection is never in the environment it fills ``${VAR}`` from."""
    source = os.environ if environ is None else environ
    args: list[str] = []
    if source.get("DOCKER_CONFIG"):
        args += ["--config", source["DOCKER_CONFIG"]]
    if source.get("DOCKER_HOST"):
        verify = source.get("DOCKER_TLS_VERIFY", "").strip() not in ("", "0")
        args += ["-H", source["DOCKER_HOST"]]
        args += docker_tls_args(
            source.get("DOCKER_CERT_PATH") or None,
            verify,
            source.get("DOCKER_TLS", "").strip() not in ("", "0"),
        )
    elif source.get("DOCKER_CONTEXT"):
        args += ["--context", source["DOCKER_CONTEXT"]]
    return args


def daemon_filesystem_local(environ: Mapping[str, str], args: Iterable[str]) -> bool:
    """Only the implicit native daemon can use this process's filesystem proof.

    Endpoints and contexts (including the saved context) can name another
    filesystem. Do not infer locality from a URL spelling or a context name.
    Unknown global flags also fail closed (S1-CONF-8 at 6dfad9e5).
    """
    if sys.platform != "linux" or any(
        environ.get(key) for key in ("DOCKER_HOST", "DOCKER_CONTEXT")
    ):
        return False
    config = operator_docker_config_dir(environ)
    flags = iter(args)
    for flag in flags:
        if flag != "--config":
            return False
        folder = next(flags, None)
        if folder is None:
            return False
        config = Path(folder)
    try:
        data = json.loads((config / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and not data.get("currentContext")


#: The keys of the operator's docker ``config.json`` that the client
#: configuration of a task's Compose calls keeps: registry logins and the
#: credential helpers that hold them, the current context, plugin settings.
#: Everything else is left out, ``proxies`` above all: Compose copies them
#: into every container's environment and every build's arguments, so a
#: proxy password would reach the task.
_CLIENT_CONFIG_KEYS = (
    "auths",
    "credsStore",
    "credHelpers",
    "currentContext",
    "plugins",
    "cliPluginsExtraDirs",
    "aliases",
    "features",
    "HttpHeaders",
    "experimental",
)

_client_configs: dict[str, str] = {}
_client_configs_lock = threading.Lock()


def _remove_client_configs() -> None:
    with _client_configs_lock:
        folders = list(_client_configs.values())
        _client_configs.clear()
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


atexit.register(_remove_client_configs)


def operator_docker_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The docker client's configuration folder the operator uses:
    ``DOCKER_CONFIG``, else ``~/.docker``."""
    source = os.environ if environ is None else environ
    if source.get("DOCKER_CONFIG"):
        return Path(source["DOCKER_CONFIG"])
    home = source.get("HOME") or os.path.expanduser("~")
    return Path(home) / ".docker"


def hermetic_docker_config(environ: Mapping[str, str] | None = None) -> str:
    """The docker client configuration folder every Compose call of a task
    uses (``docker --config <folder> compose ...``), at the check and at the
    launch alike: the operator's ``config.json`` with only
    :data:`_CLIENT_CONFIG_KEYS` (no proxies), their plugins found where they
    are (``cliPluginsExtraDirs``), and their contexts and buildx builders
    reached through links. A ``0700`` folder of this process's, one per
    distinct configuration, removed at exit."""
    folder = operator_docker_config_dir(environ)
    try:
        data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    kept = {key: data[key] for key in _CLIENT_CONFIG_KEYS if key in data}
    extra = kept.get("cliPluginsExtraDirs")
    extra = [str(d) for d in extra] if isinstance(extra, list) else []
    kept["cliPluginsExtraDirs"] = [*extra, str(folder / "cli-plugins")]
    text = json.dumps(kept, sort_keys=True)
    key = hashlib.sha256(f"{folder}\0{text}".encode()).hexdigest()
    with _client_configs_lock:
        made = _client_configs.get(key)
        if made is not None and os.path.isdir(made):
            return made
        made = tempfile.mkdtemp(prefix="benchflow-docker-config-")
        target = os.path.join(made, "config.json")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(text)
        for name in ("contexts", "buildx"):
            if (folder / name).is_dir():
                os.symlink(folder / name, os.path.join(made, name))
        _client_configs[key] = made
        return made


def manifest_env_keys(manifest: Any) -> set[str]:
    """The variables an environment manifest puts in Compose's environment."""
    if manifest is None:
        return set()
    keys = set(manifest.forward_env.keys)
    selection = manifest.task_selection
    if selection.mechanism == "env_var" and selection.inject_into == "entrypoint":
        keys.add(selection.key)
    return keys


def task_compose_values(
    sandbox_env: Mapping[str, str] | None,
    manifest: Any = None,
    allowed: Iterable[str] | None = None,
    *,
    task_id: str = "",
) -> dict[str, str]:
    """What the task and the manifest put in Compose's environment, by name,
    resolved by the run's own resolvers (``resolve_env_vars``, the
    manifest's ``resolve_manifest_runtime_env``), so the check sees the
    values the run gives Compose: ``NM = "${S1_MODE}"`` with
    ``S1_MODE`` named is ``NM``'s value of ``S1_MODE``, from the environment
    or BenchFlow's ``.env``. A table the run refuses gives nothing here (the
    env-table check refuses it)."""
    from benchflow.task.env import resolve_env_vars

    values: dict[str, str] = {}
    with contextlib.suppress(ValueError):
        values.update(resolve_env_vars(sandbox_env or {}, allowed=allowed))
    if manifest is not None:
        from benchflow.environment.manifest import resolve_manifest_runtime_env

        values.update(resolve_manifest_runtime_env(manifest, task_id=task_id))
    return values


def compose_path(environment_dir: Path) -> Path:
    return environment_dir / COMPOSE_FILE


# Reading Compose's own YAML output (a project printed as YAML)


class _YamlDate(str):
    """An unquoted date or time: text to BenchFlow, a time to go-yaml."""


def _go_int(loader: Any, node: Any) -> Any:
    """An int as go-yaml reads it: ``0o17`` and ``0777`` are octal."""
    text = loader.construct_scalar(node)
    plain = text.replace("_", "")
    try:
        return int(plain, 0)
    except ValueError:
        try:
            return int(plain, 8)  # a leading-zero octal, 0777
        except ValueError:
            return text


def _load_all(text: str) -> list[Any] | None:
    """Each YAML document of *text*, plain scalars resolved as go-yaml v3
    resolves them (:func:`benchflow.task.compose_load.loader_class`);
    Compose's ``!override``/``!reset`` tags kept as their plain values.
    ``None`` when it does not parse."""
    import yaml

    from benchflow.task.compose_load import loader_class

    Loader = loader_class()

    def any_tag(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> Any:
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)  # type: ignore[arg-type]

    def date(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        return _YamlDate(loader.construct_scalar(node))  # type: ignore[arg-type]

    Loader.add_multi_constructor("!", any_tag)
    Loader.add_constructor("tag:yaml.org,2002:timestamp", date)
    Loader.add_constructor("tag:yaml.org,2002:int", _go_int)
    try:
        return list(yaml.load_all(text, Loader=Loader))
    except yaml.YAMLError:
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _remote(value: str) -> bool:
    return "://" in value or value.startswith(("git@", "github.com/"))


def _host_refusal(field_name: str, what: str) -> TaskEnvRequest:
    return TaskEnvRequest(
        field_name, "", None, f"{field_name} {what}", remedy=_HOST_REMEDY
    )


# Running `docker compose config`


#: How long one ``docker compose config`` may take.
_CONFIG_TIMEOUT_SEC = 60


@dataclass
class _Resolved:
    """What ``docker compose config`` gave for the task."""

    #: The project as Compose printed it (its ``$`` doubled, so that Compose
    #: reads it back to the same project).
    project: dict | None = None
    #: How it was printed (``json``, or ``yaml`` for what JSON cannot hold).
    form: str = "json"
    #: The variables the Compose files read (``config --variables``).
    variables: dict[str, dict] = field(default_factory=dict)
    #: The names Compose interpolates from (``config --environment``: its
    #: process environment and the project's ``.env``).
    environment: frozenset[str] = frozenset()
    error: str | None = None


def _error_line(text: str) -> str:
    """The last line of Compose's output that is not a warning, shortened."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and "level=warning" not in line
    ]
    line = lines[-1] if lines else "no output"
    return line if len(line) <= 300 else line[:297] + "..."


def _parse_project(text: str) -> dict | None:
    try:
        loaded = json.loads(text)
    except ValueError:
        documents = _load_all(text)
        loaded = documents[0] if documents else None
    return loaded if isinstance(loaded, dict) else None


def _run(command: list[str], env: Mapping[str, str]) -> tuple[int, str, str]:
    found = subprocess.run(
        command,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_CONFIG_TIMEOUT_SEC,
        check=False,
    )
    return found.returncode, found.stdout, found.stderr


def _run_config(
    command: list[str], env: Mapping[str, str], *, project_only: bool = False
) -> _Resolved:
    """Run ``docker compose ... config`` the ways the check needs, at once.

    The project is resolved whole: env files are read into it, as the run
    reads them (their paths were checked before, by the strict reading).
    """
    runs = {"project": ["config", "--format", "json"]}
    if not project_only:
        runs["variables"] = ["config", "--variables", "--format", "json"]
        runs["environment"] = ["config", "--environment"]
    processes: dict[str, subprocess.Popen[str]] = {}
    out: dict[str, tuple[int, str, str]] = {}
    try:
        for key, args in runs.items():
            processes[key] = subprocess.Popen(
                [*command, *args],
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        for key, process in processes.items():
            stdout, stderr = process.communicate(timeout=_CONFIG_TIMEOUT_SEC)
            out[key] = (process.returncode, stdout, stderr)
        code, stdout, stderr = out["project"]
        form = "json"
        if code and "MarshalJSON" in stderr:
            # Compose 2.40.3 and v5.6.0 cannot print a build.ssh key path as
            # JSON (types.SSHKey); their YAML has it.
            code, stdout, stderr = _run([*command, "config", "--format", "yaml"], env)
            form = "yaml"
    except (OSError, subprocess.TimeoutExpired) as exc:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.communicate()
        _note_transient(type(exc).__name__)
        if isinstance(exc, OSError):
            return _Resolved(error=f"could not run docker compose ({exc.strerror})")
        return _Resolved(
            error=f"docker compose config took over {_CONFIG_TIMEOUT_SEC} s"
        )
    resolved = _Resolved(form=form)
    if code:
        resolved.error = _error_line(stderr or stdout)
        return resolved
    resolved.project = _parse_project(stdout)
    if resolved.project is None:
        resolved.error = "docker compose config printed no project"
        return resolved
    if "variables" in out:
        code, stdout, _ = out["variables"]
        if not code:
            with contextlib.suppress(ValueError):
                found = json.loads(stdout)
                if isinstance(found, dict):
                    resolved.variables = {
                        k: v for k, v in found.items() if isinstance(v, dict)
                    }
    if "environment" in out:
        code, stdout, _ = out["environment"]
        if not code:
            resolved.environment = frozenset(
                line.split("=", 1)[0] for line in stdout.splitlines() if "=" in line
            )
    return resolved


class _Undecodable(ValueError):
    """A printed value with a ``$`` Compose would not have printed."""


def decode_dollars(value: Any) -> Any:
    """A project as Compose printed it, as its run holds it: Compose 2.40.3
     and v5.6.0 print every ``$`` of a value as ``$$`` (so that reading the
     file back gives the same value), and a check of the printed text would
     look at ``leak$$dir`` where the run mounts ``leak$dir``
    . Keys are not interpolated and are left as printed. A
     lone ``$`` raises :class:`_Undecodable`: this Compose did not escape."""
    if isinstance(value, dict):
        return {key: decode_dollars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_dollars(item) for item in value]
    if isinstance(value, str) and "$" in value:
        if "$" in value.replace("$$", ""):
            raise _Undecodable(value)
        return value.replace("$$", "$")
    return value


# Checking the resolved project


#: The strings Compose casts to false for a boolean setting (compose-go's
#: toBoolean; YAML 1.1's ``n`` included).
_COMPOSE_FALSE = frozenset({"n", "no", "false", "off"})


def _compose_on(value: Any) -> bool:
    """Whether Compose may read a boolean setting as on. Fails closed:
    anything but false, unset, empty or one of Compose's false strings
    counts as on."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        return bool(text) and text not in _COMPOSE_FALSE
    return True


#: Capabilities a task's service may add: Docker's default set, and ones that
#: act only inside the container's own namespaces. Any other (``SYS_ADMIN``,
#: ``DAC_READ_SEARCH``, ``SYS_MODULE``, ``ALL`` ...) reaches the host.
_SAFE_CAPABILITIES = frozenset(
    {
        "AUDIT_WRITE",
        "CHOWN",
        "DAC_OVERRIDE",
        "FOWNER",
        "FSETID",
        "IPC_LOCK",
        "KILL",
        "MKNOD",
        "NET_ADMIN",
        "NET_BIND_SERVICE",
        "NET_BROADCAST",
        "NET_RAW",
        "SETFCAP",
        "SETGID",
        "SETPCAP",
        "SETUID",
        "SYS_CHROOT",
        "SYS_NICE",
        "SYS_PTRACE",
        "SYS_RESOURCE",
    }
)
#: Service keys whose ``host`` (or ``container:<name>``) value shares one of
#: this machine's namespaces, or another container's, with the service.
_NAMESPACE_KEYS = ("pid", "ipc", "network_mode", "userns_mode", "uts", "cgroup")


def _host_settings(document: dict, project: str | None = None) -> Iterator[str]:
    """What a Compose project's services set that hands a container this
    machine: root in a privileged container, or one with the host's PID
    namespace, reads the bench process's environment and mounts the host's
    disk; the host network reaches this machine's local services. With
    *project* (the run's project name), also a network the project does not
    own: an external one or one it names itself, which the daemon may
    already have for other containers."""
    services = document.get("services")
    for name, service in services.items() if isinstance(services, dict) else ():
        if not isinstance(service, dict):
            continue
        where = f"service {name}"
        if _compose_on(service.get("privileged")):
            yield f"runs {where} privileged"
        if _compose_on(service.get("use_api_socket")):
            # Compose mounts the engine's socket (root on this machine) and
            # writes every registry login of your docker config into it.
            yield f"gives {where} the Docker API socket and your registry logins"
        provider = service.get("provider")
        if provider is not None:
            # Compose runs the provider's program on this machine, as you.
            # Compose 2.38's JSON names the program "driver"; YAML says "type".
            shown = (
                provider.get("type") or provider.get("driver")
                if isinstance(provider, dict)
                else provider
            )
            yield f"runs the program {shown} on this machine for {where} (provider)"
        if service.get("models"):
            yield f"gives {where} models, which this machine's model runner serves"
        for key in _NAMESPACE_KEYS:
            value = service.get(key)
            if isinstance(value, str) and (
                value.strip().lower() == "host"
                or value.strip().startswith("container:")
            ):
                yield f"gives {where} {key}: {value.strip()}"
        for capability in _as_list(service.get("cap_add")):
            plain = str(capability).strip().upper().removeprefix("CAP_")
            if plain not in _SAFE_CAPABILITIES:
                yield f"gives {where} the capability {capability}"
        for option in _as_list(service.get("security_opt")):
            text = str(option).lower()
            if "unconfined" in text or "disable" in text:
                yield f"runs {where} with security_opt {option}"
        for device in _as_list(service.get("devices")):
            shown = device.get("source") if isinstance(device, dict) else device
            yield f"gives {where} the host device {shown}"
        if service.get("device_cgroup_rules"):
            yield f"gives {where} device_cgroup_rules"
        for source in _as_list(service.get("volumes_from")):
            if isinstance(source, str) and source.strip().startswith("container:"):
                yield f"gives {where} the volumes of {source.strip()}"
        for hook_key in ("post_start", "pre_stop"):
            for hook in _as_list(service.get(hook_key)):
                if isinstance(hook, dict) and _compose_on(hook.get("privileged")):
                    yield f"runs a {hook_key} command of {where} privileged"
        credential_spec = service.get("credential_spec")
        if isinstance(credential_spec, dict) and (
            credential_spec.get("file") or credential_spec.get("registry")
        ):
            yield f"gives {where} a credential_spec read from this machine"
        build = service.get("build")
        if isinstance(build, dict):
            if _compose_on(build.get("privileged")):
                yield f"builds {where} privileged"
            if str(build.get("network", "")).strip().lower() == "host":
                yield f"builds {where} on the host network"
            for entitlement in _as_list(build.get("entitlements")):
                yield f"builds {where} with the entitlement {entitlement}"
    networks = document.get("networks")
    for key, network in networks.items() if isinstance(networks, dict) else ():
        if not isinstance(network, dict):
            continue
        external = network.get("external")
        name = network.get("name") or key
        if isinstance(external, dict):
            name = external.get("name") or name
            external = True
        if str(network.get("driver", "")).strip().lower() == "host" or (
            _compose_on(external) and str(name).strip().lower() == "host"
        ):
            yield f"puts network {key} on the host network"
        elif project is not None and _compose_on(external):
            yield (
                f"joins the existing network {name} (external), which other "
                "containers of this machine may use"
            )
        elif project is not None and name != f"{project}_{key}":
            yield (
                f"names its network {key} {name}, which joins a network of that "
                "name if this machine has one"
            )


#: Local-driver mount types whose ``device`` is not a path on this machine.
_NETWORK_MOUNT_TYPES = frozenset(
    {"nfs", "nfs4", "cifs", "smb", "smb3", "ceph", "glusterfs"}
)
#: Local-driver mount types with no backing device at all.
_MEMORY_MOUNT_TYPES = frozenset({"tmpfs", "ramfs"})
#: Mount options that make the kernel bind the device, whatever the type.
_BIND_OPTIONS = frozenset({"bind", "rbind"})
#: Additional build contexts that are not a folder here.
_NOT_A_FOLDER = ("docker-image://", "service:", "target:")
#: Where BenchFlow mounts its pinned helper, bfx (TRUSTED-RUNTIME.md §6).
_BFX_DIR = "/.bfx"


def _ssh_entries(value: Any) -> Iterator[tuple[str, str | None]]:
    """``build.ssh`` as (id, key path or ``None`` for the SSH agent); Compose
    prints an entry as ``id``, ``id=path`` or (YAML) ``id: path``."""
    if isinstance(value, dict):
        for key, path in value.items():
            yield str(key), path if isinstance(path, str) and path else None
        return
    for item in _as_list(value):
        text = str(item)
        for separator in ("=", ": "):
            if separator in text:
                key, _, path = text.partition(separator)
                yield key.strip(), path.strip() or None
                break
        else:
            yield text.strip(), None


def _clean_path(path: str) -> str:
    """A path cleaned as Go's path.Clean cleans it, the one spelling rule
    for every comparison here (S1-CONF-3): Docker cleans a bind source so
    before it mounts, and Compose prints absolute values as written. No
    empty or `.` part, no inner `..`, one leading slash."""
    cleaned = posixpath.normpath(path)
    return "/" + cleaned.lstrip("/") if cleaned.startswith("/") else cleaned


def _local_realpath(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return _clean_path(path)


def _local_identity(path: str) -> tuple[int, int] | None:
    """The folder or file *path* names, links followed: one value for every
    spelling of it (S1-CONF-2). ``None``: nothing is there."""
    try:
        found = os.stat(path)
    except (OSError, ValueError):
        return None
    return (found.st_dev, found.st_ino)


_SOURCE_DIRECTORY_LIMIT = 65536


def _local_tree_identities(path: str) -> set[Any] | None:
    """Prove the complete directory population reachable through a writer.

    Follow directory mounts, including bind aliases, but never symlinks:
    a symlink inside a container is resolved in that container's namespace.
    Missing facts, inaccessible subtrees and oversized trees are unknown,
    not evidence of separation (S1-CONF-4/5 at 8484e852).
    """
    found: set[Any] = set()
    pending = [path]
    try:
        while pending:
            current = pending.pop()
            identity = _local_identity(current)
            if identity is None:
                return None
            if identity in found:
                # The same directory reached at another mount location may
                # have different descendants. A partial proof is no proof.
                return None
            found.add(identity)
            if len(found) > _SOURCE_DIRECTORY_LIMIT:
                return None
            if not stat.S_ISDIR(os.stat(current).st_mode):
                continue
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        if len(found) + len(pending) >= _SOURCE_DIRECTORY_LIMIT:
                            return None
                        pending.append(entry.path)
    except (OSError, ValueError):
        return None
    return found


def _mountpoints(text: str) -> set[str]:
    """Decode the kernel's mountinfo paths, refusing malformed records."""
    points = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or " - " not in line:
            raise ValueError("incomplete mountinfo")
        points.add(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), parts[4]))
    if not points:
        raise ValueError("empty mountinfo")
    return points


def _local_mountpoints() -> set[str] | None:
    try:
        return _mountpoints(Path("/proc/self/mountinfo").read_text())
    except (OSError, ValueError):
        return None


def _same_path(path: str) -> str:
    return path


@dataclass
class _ProjectCheck:
    """Checks a resolved Compose project for what reaches the machine its
    daemon runs on: this one (docker), the remote host (remote-docker) or
    the sandbox's VM (Daytona DinD)."""

    project_dir: str
    #: The folders a path may be in: the task's, BenchFlow's own trial
    #: folders and those the operator named.
    folders: tuple[str, ...]
    #: Whether Compose has an SSH agent the operator did not name
    #: (remote-docker over ssh).
    agent: bool = False
    #: remote-docker: a bind mount, a bind-backed volume and a file secret
    #: or config are paths on the remote host, never the task's files.
    remote_binds: bool = False
    #: BenchFlow's own read-only binds, as (source, target) pairs: a bind of
    #: exactly that source at exactly that target, read-only, is admitted
    #: (bfx at /.bfx). The same source anywhere else, or writable, is not.
    read_only_binds: tuple[tuple[str, str], ...] = ()
    #: BenchFlow's own external volumes, as (name, target) pairs: admitted
    #: when every mount of the volume is read-only, never copied from the
    #: image, at that target (bfx's harness volume at /.bfx).
    read_only_volumes: tuple[tuple[str, str], ...] = ()
    #: Where a path lands once symlinks are followed (local realpath; the
    #: VM's for DinD).
    resolve: Callable[[str], str] = _local_realpath
    #: The file identity of a path (:func:`_local_identity`; the VM's for
    #: DinD). ``None`` reads this machine's at call time.
    identity: Callable[[str], Any] | None = None
    #: False when the identities could not be read (the VM's batch failed):
    #: a check with a writable mount then refuses rather than guesses.
    identities_known: bool = True
    #: Complete directory identities reachable through a writable mount.
    #: None uses the local filesystem; DinD supplies the VM's inspection.
    tree_identities: Callable[[str], set[Any] | None] | None = None
    #: The disposable DinD preflight has no VM filesystem yet. It checks
    #: the graph; the mandatory VM check must prove identities before up.
    defer_filesystem: bool = False
    mountpoints: Callable[[], set[str] | None] = _local_mountpoints
    #: The run's project name: a volume or network it does not own (an
    #: external one, or one named other than ``<project>_<key>``) is
    #: refused. ``None`` (the VM's check): not checked.
    project: str | None = None
    machine: str = "this machine"
    field_name: str = f"environment/{COMPOSE_FILE}"
    refused: list[TaskEnvRequest] = field(default_factory=list)
    read_paths: set[str] = field(default_factory=set, init=False)

    def place(self, raw: str) -> str:
        return _clean_path(posixpath.join(self.project_dir, raw))

    def ident(self, path: str) -> Any:
        return (self.identity or _local_identity)(path)

    def within(self, raw: str, folders: Iterable[str]) -> bool:
        path = self.place(raw)
        self.read_paths.add(path)
        resolved = self.resolve(path)
        return any(
            resolved == folder or resolved.startswith(folder.rstrip("/") + "/")
            for folder in folders
        )

    def outside(self, raw: str) -> bool:
        return not self.within(raw, self.folders)

    def refuse(self, what: str) -> None:
        self.refused.append(_host_refusal(self.field_name, what))

    def path(self, where: str, what: str, raw: Any) -> None:
        if isinstance(raw, str) and raw and self.outside(raw):
            self.refuse(f"{what} {raw} for {where}, outside the task's folder")

    def host_path(self, where: str, what: str, raw: Any) -> None:
        """A path the daemon's machine reads (a bind source, a file secret)."""
        if not isinstance(raw, str) or not raw:
            return
        if self.remote_binds:
            self.refuse(
                f"{what} {raw} for {where}, a path on {self.machine}; "
                "remove task host mounts, or unset DOCKER_HOST and "
                "DOCKER_CONTEXT, remove the saved context and remote flags, "
                "and use the native local daemon so its filesystem can be proved"
            )
        else:
            self.path(where, what, raw)

    def check(self, project: dict) -> list[TaskEnvRequest]:
        services = project.get("services")
        services = services if isinstance(services, dict) else {}
        for name, service in services.items():
            if isinstance(service, dict):
                self.service(f"service {name}", service)
        for top in ("secrets", "configs"):
            items = project.get(top)
            for key, item in items.items() if isinstance(items, dict) else ():
                if isinstance(item, dict):
                    # Compose bind-mounts a file secret or config.
                    self.host_path(
                        f"{top[:-1]} {key}", "reads the file", item.get("file")
                    )
        volumes = project.get("volumes")
        for key, item in volumes.items() if isinstance(volumes, dict) else ():
            if self.harness_volume(key, item, services):
                continue
            self.volume(key, item)
        for service in services.values():
            if not isinstance(service, dict):
                continue
            for mount in _as_list(service.get("volumes")):
                if isinstance(mount, dict) and mount.get("type") == "volume":
                    self.host_path(
                        "volume mount", "mounts", self.mount_source(mount, volumes)
                    )
        self.refused += [
            _host_refusal(self.field_name, f"{setting}, which reaches {self.machine}")
            for setting in _host_settings(project, self.project)
        ]
        self.mutable_sources(project)
        return self.refused

    def mount_source(self, mount: dict, volumes: Any) -> str | None:
        """The host path of a bind or a bind-backed volume consumer."""
        source = mount.get("source")
        if mount.get("type") == "volume":
            volume = volumes.get(source) if isinstance(volumes, dict) else None
            if not isinstance(volume, dict):
                return None
            options = volume.get("driver_opts") or {}
            if not isinstance(options, dict):
                return None
            kind = str(options.get("type") or "").strip().lower()
            flags = {
                part.strip().lower().partition("=")[0]
                for part in str(options.get("o") or "").split(",")
            }
            if not flags & _BIND_OPTIONS and kind not in ("", "none", "bind"):
                return None
            # Non-path devices (network mounts, tmpfs) are not host trees.
            source = options.get("device")
            if not isinstance(source, str) or not source.startswith("/"):
                return None
            subpath = (mount.get("volume") or {}).get("subpath")
            if isinstance(subpath, str):
                source = posixpath.join(source, subpath)
        elif mount.get("type") != "bind":
            return None
        return self.place(source) if isinstance(source, str) and source else None

    def writable_sources(self, project: dict) -> set[str]:
        """Every effective host-backed writer, including inherited mounts."""
        services = project.get("services")
        services = services if isinstance(services, dict) else {}
        volumes = project.get("volumes")
        volumes = volumes if isinstance(volumes, dict) else {}

        def sources(name: str, seen: frozenset[str], inherited: bool = False):
            if name in seen:
                return
            service = services.get(name)
            if not isinstance(service, dict):
                return
            for mount in _as_list(service.get("volumes")):
                if not isinstance(mount, dict):
                    continue
                if not inherited and _compose_on(mount.get("read_only")):
                    continue
                source = self.mount_source(mount, volumes)
                if source is not None:
                    yield source
            for reference in _as_list(service.get("volumes_from")):
                inherited_name, _, mode = str(reference).partition(":")
                if mode != "ro" or inherited:
                    # A rw/default inherited mount can expose a source that
                    # its declaring service mounted read-only. Follow chains.
                    yield from sources(inherited_name, seen | {name}, True)

        return {source for name in services for source in sources(name, frozenset())}

    def mutable_sources(self, project: dict) -> None:
        """Keep launch path entries outside every container-writable tree.

        A private snapshot is still mutable if a container mounts its parent
        writable. An agent can replace a child with a symlink after startup;
        Compose restart or Docker snapshot restore then resolves that child
        on the daemon's host (sol at 62c80723). Refuse that
        mount graph before startup, so later launches need no racy recheck.
        The mount root itself cannot be replaced through that mount.
        """
        writable = self.writable_sources(project)
        if not writable:
            return
        # A nonrecursive bind hides descendant mounts and reveals the tree
        # underneath them, which the visible-tree scan cannot inspect. Check
        # the volume device before subpath selection, not just the subpath.
        if not self.defer_filesystem:
            volumes = project.get("volumes") or {}
            for volume in volumes.values() if isinstance(volumes, dict) else ():
                options = (
                    volume.get("driver_opts") if isinstance(volume, dict) else None
                )
                if not isinstance(options, dict):
                    continue
                device = options.get("device")
                flags = {
                    p.strip().lower().partition("=")[0]
                    for p in str(options.get("o") or "").split(",")
                }
                if not isinstance(device, str) or not device.startswith("/"):
                    continue
                if "rbind" in flags and "bind" not in flags:
                    continue
                points = self.mountpoints()
                root = self.resolve(self.place(device)).rstrip("/") + "/"
                if points is None or any(p.startswith(root) for p in points):
                    self.refuse(
                        f"cannot prove the tree exposed by nonrecursive bind "
                        f"volume {device}; use o=rbind, remove descendant mounts, "
                        "or use read-only mounts"
                    )
        # A symlink can traverse a writable tree and leave it again inside
        # its target, without that tree appearing in any written ancestor
        # or the final realpath. Require canonical source spellings beside
        # writers instead of attempting to reconstruct hidden link chains.
        # This also holds for the VM resolver, which returns only endpoints.
        if writable:
            for path in sorted(self.read_paths):
                if self.resolve(path) != path:
                    self.refuse(
                        f"uses noncanonical launch source {path} beside a "
                        "writable mount; use canonical source paths so no "
                        "symlink can hide a writable ancestor"
                    )
        # Resolve every ancestor as well as the leaf: an alias can enter a
        # writable tree and another symlink can leave it before the leaf.
        # Looking only at the final realpath would miss that writable entry.
        ancestors: dict[str, set[str]] = {}
        for path in self.read_paths:
            parents: set[str] = set()
            parent = posixpath.dirname(path)
            while True:
                parents.update((parent, self.resolve(parent)))
                if parent == posixpath.dirname(parent):
                    break
                parent = posixpath.dirname(parent)
            ancestors[path] = parents
        # Spellings can also differ without a link: a case- or Unicode-
        # insensitive filesystem, or a bind mount of the host, names one
        # folder two ways (S1-CONF-2). An agent cannot choose the spelling,
        # but it makes the swap; compare what each ancestor is, too.
        if not self.defer_filesystem and not self.identities_known:
            self.refuse(
                f"has writable mounts, and {self.machine} did not say which "
                "folders its launch sources are in; make the sources readable "
                "and retry, or use read-only mounts"
            )
        found = {parent: self.ident(parent) for p in ancestors.values() for parent in p}
        if not self.defer_filesystem:
            for parent, identity in sorted(found.items()):
                if identity is None:
                    self.refuse(
                        f"cannot establish the identity of launch-source ancestor "
                        f"{parent} on {self.machine}; create the source folders "
                        "and make them readable before retrying, or use "
                        "read-only mounts"
                    )
        for source in sorted(writable):
            roots = {source, self.resolve(source)}
            same = {self.ident(root) for root in roots} - {None}
            if not self.defer_filesystem:
                if any(self.ident(root) is None for root in roots):
                    self.refuse(
                        f"cannot establish the identity of writable mount "
                        f"{source} on {self.machine}; create it and make it "
                        "readable before retrying, or use read-only mounts"
                    )
                tree = (self.tree_identities or _local_tree_identities)(source)
                if tree is None:
                    self.refuse(
                        f"cannot inspect every writable directory beneath "
                        f"{source} on {self.machine}; make the complete tree "
                        "readable and small enough to inspect (at most 65536 "
                        "directories) with no repeated directory identities, "
                        "or use read-only mounts"
                    )
                else:
                    same.update(tree)
            for path, parents in ancestors.items():
                if any(
                    found[parent] in same
                    or any(
                        parent == root or parent.startswith(root.rstrip("/") + "/")
                        for root in roots
                    )
                    for parent in parents
                ):
                    self.refuse(
                        f"mounts {source} writable over the launch source {path}; "
                        "use read-only mounts or separate source folders so an "
                        "agent cannot replace a source before restart or restore"
                    )

    def harness_volume(self, key: str, item: Any, services: dict) -> bool:
        targets = {name: target for name, target in self.read_only_volumes}
        if key not in targets or not isinstance(item, dict):
            return False
        if not _compose_on(item.get("external")) or item.get("name") not in (None, key):
            return False
        for name, service in services.items():
            for mount in (
                _as_list(service.get("volumes")) if isinstance(service, dict) else ()
            ):
                if not isinstance(mount, dict) or mount.get("source") != key:
                    continue
                nocopy = (mount.get("volume") or {}).get("nocopy")
                if not (
                    mount.get("type") == "volume"
                    and _compose_on(mount.get("read_only"))
                    and _compose_on(nocopy)
                    and _clean_path(str(mount.get("target") or "")) == targets[key]
                ):
                    self.refuse(
                        f"mounts BenchFlow's volume {key} for service {name} "
                        f"other than read-only and uncopied at {targets[key]}"
                    )
        return True

    def harness_read_only_bind(self, volume: dict) -> bool:
        source, target = volume.get("source"), volume.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            return False
        if self.remote_binds or not _compose_on(volume.get("read_only")):
            return False
        resolved = self.resolve(self.place(source))
        return any(
            resolved == want_source and _clean_path(target) == want_target
            for want_source, want_target in self.read_only_binds
        )

    def harness_volume_mount(self, volume: dict) -> bool:
        """The mount names BenchFlow's own helper volume at its target
        (:meth:`harness_volume` checks how it is mounted)."""
        return volume.get("type") == "volume" and any(
            volume.get("source") == name
            and _clean_path(str(volume.get("target") or "")) == target
            for name, target in self.read_only_volumes
        )

    def bfx_target(self, where: str, target: Any, kind: Any) -> None:
        """Nothing but BenchFlow's own read-only helper mount sits at or
        beneath /.bfx: a task's or operator's mount there would cover the
        program BenchFlow runs as root."""
        if not isinstance(target, str) or not target.startswith("/"):
            return
        target = _clean_path(target)
        if target == _BFX_DIR or target.startswith(_BFX_DIR + "/"):
            self.refuse(
                f"mounts a {kind} mount at {target} for {where}; {_BFX_DIR} is "
                "BenchFlow's own read-only folder, so nothing may be mounted at "
                "or beneath it"
            )

    def service(self, where: str, service: dict) -> None:
        for volume in _as_list(service.get("volumes")):
            if not isinstance(volume, dict):
                self.refuse(
                    f"mounts {volume} for {where}, which Compose left unresolved"
                )
                continue
            kind = volume.get("type")
            if kind == "bind" and self.harness_read_only_bind(volume):
                continue
            if self.harness_volume_mount(volume):
                continue
            self.bfx_target(where, volume.get("target"), kind)
            if kind == "bind":
                self.host_path(where, "mounts", volume.get("source"))
            elif kind == "npipe":
                self.refuse(f"mounts the named pipe {volume.get('source')} for {where}")
            elif kind not in ("volume", "tmpfs", "image", None):
                self.refuse(f"mounts a {kind} mount for {where}")
        # The short form `tmpfs: [PATH[:OPTIONS]]`.
        for entry in _as_list(service.get("tmpfs")):
            self.bfx_target(where, str(entry).partition(":")[0], "tmpfs")
        # A Compose that keeps env and label files in the resolved project
        # reads them again at the run: they must be the task's.
        for key, what in (
            ("env_file", "reads the env file"),
            ("label_file", "reads the label file"),
        ):
            for entry in _as_list(service.get(key)):
                path = entry.get("path") if isinstance(entry, dict) else entry
                self.path(where, what, path)
        build = service.get("build")
        if isinstance(build, dict):
            self.build(where, build)

    def build(self, where: str, build: dict) -> None:
        context = build.get("context")
        if isinstance(context, str) and context and not _remote(context):
            inline = build.get("dockerfile_inline") is not None
            dockerfile = posixpath.join(
                self.place(context), str(build.get("dockerfile") or "Dockerfile")
            )
            if not self.outside(context):
                if not inline and self.outside(dockerfile):
                    self.refuse(
                        f"builds {where} with the Dockerfile {dockerfile}, "
                        "outside the task's folder"
                    )
            elif inline or not self.outside(dockerfile):
                # The build copies from this machine's folder with a
                # Dockerfile the task wrote.
                self.refuse(
                    f"builds {where} from {context}, outside the task's "
                    "folder, with a Dockerfile of the task's"
                )
            elif not self.within(dockerfile, (self.resolve(self.place(context)),)):
                self.refuse(
                    f"builds {where} from {context}, outside the task's "
                    f"folder, with the Dockerfile {dockerfile} from elsewhere"
                )
        contexts = build.get("additional_contexts")
        if isinstance(contexts, dict):
            values = list(contexts.values())
        else:
            values = [str(v).partition("=")[2] for v in _as_list(contexts)]
        for value in values:
            if not isinstance(value, str) or value.startswith(_NOT_A_FOLDER):
                continue
            if value.startswith("oci-layout://"):
                value = value[len("oci-layout://") :]
            elif _remote(value):
                continue
            self.path(where, "builds with the context", value)
        for key in ("cache_from", "cache_to"):
            for entry in _as_list(build.get(key)):
                # type=local,src=PATH / dest=PATH: a folder of this machine.
                for part in str(entry).split(","):
                    name, _, value = part.partition("=")
                    if name.strip() in ("src", "dest") and "type=local" in str(entry):
                        self.path(where, f"builds with the {key} folder", value.strip())
        for key, path in _ssh_entries(build.get("ssh")):
            if path is not None:
                self.path(where, "gives the build the SSH key", path)
            elif self.agent:
                self.refuse(
                    f"gives the build of {where} your SSH agent (build.ssh: {key})"
                )

    def volume(self, key: str, item: Any) -> None:
        if not isinstance(item, dict):
            return
        where = f"volume {key}"
        if self.project is not None:
            # A volume the project does not own is one the daemon already
            # has: the operator's data, or a local-driver volume backed by
            # any folder of the machine.
            if _compose_on(item.get("external")):
                self.refuse(
                    f"uses the existing volume {item.get('name') or key} "
                    f"(external) for {where}, which may hold data of "
                    f"{self.machine}"
                )
                return
            if item.get("name") not in (None, f"{self.project}_{key}"):
                self.refuse(
                    f"names {where} {item.get('name')}, which reuses a volume of "
                    f"that name if {self.machine} has one"
                )
                return
        driver = str(item.get("driver") or "local").strip().lower()
        options = item.get("driver_opts")
        if driver != "local":
            # A volume plugin runs on the daemon's machine and mounts what
            # its options say.
            self.refuse(f"uses the volume driver {driver} for {where}")
            return
        if not isinstance(options, dict) or not options:
            return
        device = options.get("device")
        kind = str(options.get("type") or "").strip().lower()
        mount_options = {
            part.strip().lower().partition("=")[0]
            for part in str(options.get("o") or "").split(",")
        }
        # `o: bind` makes the kernel bind `device`, whatever `type` says
        # it is a host path.
        if mount_options & _BIND_OPTIONS or kind in ("", "none", "bind"):
            if isinstance(device, str) and device:
                if self.remote_binds:
                    self.host_path(where, "binds", device)
                elif not posixpath.isabs(device):
                    self.refuse(
                        f"binds {where} from relative device {device}, which the "
                        "daemon resolves from its own working directory"
                    )
                else:
                    self.host_path(where, "binds", device)
            elif mount_options & _BIND_OPTIONS:
                self.refuse(f"binds {where} with no device")
            return
        if kind in _MEMORY_MOUNT_TYPES or kind in _NETWORK_MOUNT_TYPES:
            return
        # Any other filesystem (ext4 on a block device, overlay of host
        # folders ...) mounts something of the daemon's machine.
        self.refuse(f"mounts {where} as {kind} from {device}")

    def paths(self, project: dict) -> set[str]:
        """Every path :meth:`check` resolves, for a resolver that needs them
        all at once (the VM's)."""
        seen: set[str] = set()
        recorder = _ProjectCheck(
            self.project_dir,
            self.folders,
            resolve=lambda path: seen.add(path) or path,
            identity=lambda path: None,
            defer_filesystem=True,
        )
        recorder.check(project)
        return seen


#: The path fields of a resolved project, by where they are: what
#: :func:`_map_project_paths` rewrites.
def _map_project_paths(project: dict, mapping: Callable[[str], str]) -> dict:
    """Rewrite each consumer's path into the checked coordinate space.

    Docker keeps task paths in its private snapshot; DinD preflight maps
    its disposable local resolver's paths into the VM. Only absolute paths
    are mapped; Compose prints them absolute.
    """

    def one(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("/"):
            return mapping(value)
        return value

    out = json.loads(json.dumps(project))
    services = out.get("services")
    for service in services.values() if isinstance(services, dict) else ():
        if not isinstance(service, dict):
            continue
        for volume in _as_list(service.get("volumes")):
            if isinstance(volume, dict) and volume.get("type") == "bind":
                volume["source"] = one(volume.get("source"))
        for key in ("env_file", "label_file"):
            entries = service.get(key)
            if isinstance(entries, list):
                service[key] = [
                    {**entry, "path": one(entry.get("path"))}
                    if isinstance(entry, dict)
                    else one(entry)
                    for entry in entries
                ]
        build = service.get("build")
        if isinstance(build, dict):
            context = build.get("context")
            build["context"] = one(context)
            if isinstance(build.get("dockerfile"), str):
                dockerfile = build["dockerfile"]
                if isinstance(context, str) and build["context"] != context:
                    dockerfile = _clean_path(posixpath.join(context, dockerfile))
                build["dockerfile"] = one(dockerfile)
            contexts = build.get("additional_contexts")
            if isinstance(contexts, dict):
                for name, value in contexts.items():
                    contexts[name] = (
                        "oci-layout://" + one(value[len("oci-layout://") :])
                        if isinstance(value, str) and value.startswith("oci-layout://")
                        else one(value)
                    )
            for key in ("cache_from", "cache_to"):
                if isinstance(build.get(key), list):
                    build[key] = [
                        ",".join(
                            part.partition("=")[0] + "=" + one(part.partition("=")[2])
                            if part.partition("=")[0].strip() in ("src", "dest")
                            else part
                            for part in entry.split(",")
                        )
                        if isinstance(entry, str) and "type=local" in entry
                        else entry
                        for entry in build[key]
                    ]
            ssh = build.get("ssh")
            if isinstance(ssh, dict):
                build["ssh"] = {name: one(path) for name, path in ssh.items()}
            elif isinstance(ssh, list):
                build["ssh"] = [
                    entry.partition("=")[0] + "=" + one(entry.partition("=")[2])
                    if isinstance(entry, str) and "=" in entry
                    else entry
                    for entry in ssh
                ]
    for top in ("secrets", "configs"):
        items = out.get(top)
        for item in items.values() if isinstance(items, dict) else ():
            if isinstance(item, dict) and "file" in item:
                item["file"] = one(item["file"])
    volumes = out.get("volumes")
    for item in volumes.values() if isinstance(volumes, dict) else ():
        options = item.get("driver_opts") if isinstance(item, dict) else None
        if not isinstance(options, dict):
            continue
        kind = str(options.get("type") or "").strip().lower()
        flags = {
            part.strip().lower().partition("=")[0]
            for part in str(options.get("o") or "").split(",")
        }
        if (
            str(item.get("driver") or "local").strip().lower() == "local"
            and (kind in ("", "none", "bind") or flags & _BIND_OPTIONS)
            and "device" in options
        ):
            options["device"] = one(options["device"])
    return out


def _operator_project_paths(
    project: dict, names: Iterable[str], environ: Mapping[str, str]
) -> dict:
    """Resolve only named host roots in the fixed set of path consumers.

    Compose has already interpolated values but prints absolute ones as
    written, so both sides are cleaned by the one rule (:func:`_clean_path`,
    S1-CONF-3) before they are compared. Replace the operator's root,
    leaving task-supplied descendants alone: a symlink below that root
    must still face the mutable-source check. A consumer that is not the
    root or below it after cleaning is left as written. Printed paths have
    escaped dollars, like every Compose project value.
    """
    aliases = sorted(
        (
            (
                _clean_path(value).replace("$", "$$"),
                _local_realpath(value).replace("$", "$$"),
            )
            for name in set(names) - _CONNECTION_NAMES
            if (value := environ.get(name, "")).startswith("/")
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    def resolve(path: str) -> str:
        cleaned = _clean_path(path)
        for source, target in aliases:
            if cleaned == source:
                return target
            if cleaned.startswith(source.rstrip("/") + "/"):
                return target.rstrip("/") + "/" + cleaned[len(source) :].lstrip("/")
        return path

    return _map_project_paths(project, resolve)


def _launchable(project: dict) -> dict:
    """The project as the launch file holds it: every service Compose
    printed is one it started with the task's profiles on, so their
    ``profiles`` go (the launch has no ``.env`` to turn them on)."""
    out = json.loads(json.dumps(project))
    services = out.get("services")
    for service in services.values() if isinstance(services, dict) else ():
        if isinstance(service, dict):
            service.pop("profiles", None)
    return out


def project_bind_sources(project: Mapping[str, Any]) -> list[str]:
    """Every host path a resolved project binds into a container: each
    service's bind mounts, and each local-driver volume whose options bind a
    ``device`` (``o: bind`` or a bind ``type``)."""
    sources: list[str] = []
    services = project.get("services")
    for service in services.values() if isinstance(services, Mapping) else ():
        if not isinstance(service, Mapping):
            continue
        for volume in _as_list(service.get("volumes")):
            if (
                isinstance(volume, Mapping)
                and volume.get("type") == "bind"
                and isinstance(volume.get("source"), str)
                and volume["source"]
            ):
                sources.append(volume["source"])
    volumes = project.get("volumes")
    for item in volumes.values() if isinstance(volumes, Mapping) else ():
        options = item.get("driver_opts") if isinstance(item, Mapping) else None
        if not isinstance(options, Mapping):
            continue
        kind = str(options.get("type") or "").strip().lower()
        flags = {
            part.strip().lower().partition("=")[0]
            for part in str(options.get("o") or "").split(",")
        }
        device = options.get("device")
        if (
            str(item.get("driver") or "local").strip().lower() == "local"
            and (kind in ("", "none", "bind") or flags & _BIND_OPTIONS)
            and isinstance(device, str)
            and device
        ):
            sources.append(device)
    return sources


# Freezing the launch


@dataclass(frozen=True)
class ComposeLaunch:
    """The checked project a sandbox launches: every Compose call
    of the sandbox is ``docker <docker_args> compose -p <project_name>
    --project-directory <project_dir> -f <path>`` with *env*, nothing else."""

    path: Path
    project_dir: Path
    project_name: str
    env: Mapping[str, str]
    docker_args: tuple[str, ...]
    #: The run's ``-f`` files the checked project was resolved from.
    files: tuple[Path, ...]
    #: What the launch was made from (files, environment, flags, name):
    #: another start with other inputs makes and checks another.
    key: str

    def bind_sources(self) -> list[str]:
        """Every host path this launch binds into a container
        (:func:`project_bind_sources` of the frozen project). Raises
        ``OSError`` or ``ValueError`` when the frozen file cannot be read."""
        project = json.loads(self.path.read_bytes())
        if not isinstance(project, dict):
            raise ValueError(f"frozen launch {self.path} is not a project")
        return project_bind_sources(project)


def launch_key(
    files: Iterable[Path],
    env: Mapping[str, str],
    docker_args: Iterable[str],
    project_name: str,
    project_dir: Path,
) -> str:
    text = json.dumps(
        [
            [str(f) for f in files],
            sorted(env.items()),
            list(docker_args),
            project_name,
            str(project_dir),
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class ComposeCheck:
    """What :func:`check_compose` (or :func:`freeze_compose`) found."""

    #: What the task's Compose setup asks of this machine that nobody allowed.
    refused: list[TaskEnvRequest] = field(default_factory=list)
    #: Variables the Compose files read that Compose does not have (no
    #: default, or one): they get their default or nothing, never the
    #: operator's value. For :func:`benchflow.task.env.defaulted_task_env`.
    defaulted: list[TaskEnvRequest] = field(default_factory=list)
    #: The frozen launch, when the check passed and one was asked for.
    launch: ComposeLaunch | None = None


def _config_refusal(error: str | None, where: str = "") -> TaskEnvRequest:
    field_name = f"environment/{COMPOSE_FILE}"
    return TaskEnvRequest(
        field_name,
        "",
        None,
        f"{field_name} does not resolve with docker compose config{where}: {error}",
        remedy=_CONFIG_REMEDY,
    )


def _create_host_path_spelled_out(printed: dict, again: Any) -> dict | None:
    """*printed* with ``create_host_path: true`` written out on each bind
    whose ``bind`` mapping Compose printed without the key and read back with
    it false, when those are the only differences between *printed* and its
    read back *again*; otherwise None.

    Compose 5.0 prints a bind whose create_host_path is true (what a
    short-syntax bind implies) as ``"bind": {}``, and reads a ``bind`` that
    leaves the key out as false, so its own print does not read back the
    same. Compose 2.40 prints true and leaves false out, and reads its print
    back unchanged. Any other difference is still a refusal.
    """
    if again == printed:
        return None
    expected = json.loads(json.dumps(printed))
    explicit = json.loads(json.dumps(printed))
    found = False
    services = printed.get("services")
    for name, service in services.items() if isinstance(services, dict) else ():
        volumes = service.get("volumes") if isinstance(service, dict) else None
        for index, volume in enumerate(volumes if isinstance(volumes, list) else ()):
            if not isinstance(volume, dict) or volume.get("type") != "bind":
                continue
            bind = volume.get("bind")
            if not isinstance(bind, dict) or "create_host_path" in bind:
                continue
            try:
                read = again["services"][name]["volumes"][index]["bind"]
            except (KeyError, IndexError, TypeError):
                continue
            if read == {**bind, "create_host_path": False}:
                expected["services"][name]["volumes"][index]["bind"] = read
                explicit["services"][name]["volumes"][index]["bind"] = {
                    **bind,
                    "create_host_path": True,
                }
                found = True
    return explicit if found and again == expected else None


def _first_difference(a: Any, b: Any, path: str = "") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b or a[key] != b[key]:
                return _first_difference(a.get(key), b.get(key), f"{path}.{key}")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
    return path.lstrip(".") or "the project"


def _environment_sources(
    printed: dict,
    command: list[str],
    env: Mapping[str, str],
    staged: Any,
    *,
    frozen: dict[str, str] | None,
) -> str | None:
    """Freeze late-bound secret/config contents, never host-program env.

    Compose serializes their variable names but reads their contents only
    at creation. Ask the same loader to interpolate just
    those names into a disposable service: JSON preserves multiline values
    and dollars, unlike parsing ``config --environment`` line by line.

    Keep environment delivery: Compose copies the bytes through the daemon
    API with each consumer's uid/gid/mode. A file substitute would become
    a client-local bind and lose those semantics. Use
    fresh names in a fixed, inert namespace, so a dotenv source named after
    a host-program setting never activates that setting at launch.
    """
    sources = [
        (top, key, item)
        for top in ("secrets", "configs")
        for key, item in (printed.get(top) or {}).items()
        if isinstance(item, dict) and isinstance(item.get("environment"), str)
    ]
    if not sources:
        return None
    if any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item["environment"])
        for _, _, item in sources
    ):
        return "environment-backed secrets/configs need plain variable names"
    folder = staged.private_folder("environment-sources")
    service = "benchflow-source-" + os.urandom(8).hex()
    probe = folder / f"{service}.json"
    variables = {
        f"V{i}": "${" + item["environment"] + "?required by a secret or config}"
        for i, (top, key, item) in enumerate(sources)
    }
    with probe.open("w", encoding="utf-8") as out:
        json.dump(
            {"services": {service: {"image": "busybox", "environment": variables}}}, out
        )
    resolved = _run_config([*command, "-f", str(probe)], env, project_only=True)
    if resolved.error is not None or resolved.project is None:
        return resolved.error or "could not resolve environment-backed secrets/configs"
    try:
        values = decode_dollars(resolved.project)["services"][service]["environment"]
        for i, (top, key, item) in enumerate(sources):
            value = values[f"V{i}"]
            if not isinstance(value, str):
                return f"environment variable {item['environment']} required by {top} {key} is not set"
            if frozen is not None:
                name = f"BENCHFLOW_COMPOSE_SOURCE_{os.urandom(16).hex()}_{i}"
                frozen[name] = value
                item["environment"] = name
    except (KeyError, TypeError, _Undecodable):
        return "could not read environment-backed secrets/configs from Compose"
    finally:
        probe.unlink()
    return None


def freeze_compose(
    *,
    staged: Any,
    files: list[Path],
    project_name: str,
    env: Mapping[str, str],
    docker_args: Iterable[str],
    check: _ProjectCheck | None,
    host_access: bool,
    provided: Mapping[str, str] | None = None,
    names: frozenset[str] = frozenset(),
    transport: Mapping[str, str] | None = None,
    operator_environ: Mapping[str, str] | None = None,
    freeze: bool = True,
    map_paths: Callable[[str], str] | None = None,
    read_alias: tuple[str, Path] | None = None,
    agent_security: bool = False,
) -> ComposeCheck:
    """Read, resolve, check and freeze a staged task's Compose project once
    .

     *staged* is the task's :class:`~benchflow.task.staging.StagedEnvironment`;
     *files* the run's ``-f`` files (BenchFlow's and the copy's
     ``docker-compose.yaml``); *env*, *docker_args* and *project_name* are
     the run's own. *check* checks the resolved project (``None``: host
     access). With *freeze*, the checked project is written to a private file
     of *staged*, Compose must read that file back to the same project, and
     the result's ``launch`` is what every later Compose call uses. With
     *agent_security*, every service of the frozen project gets
     no-new-privileges once, in place of any spelling the task gave it
     (:func:`~benchflow.sandbox._compose_runtime.with_no_new_privileges`):
     the launch of every start, recovery included, creates each service with
     it, and no file stacked before the task's sets it.
    """
    from benchflow.task.compose_load import read_compose

    provided = provided or {}
    transport = transport or {}
    result = ComposeCheck()
    environment_dir = Path(staged.environment_dir)
    compose_file = compose_path(environment_dir)
    field_name = f"environment/{COMPOSE_FILE}"
    if compose_file.exists():
        loaded = read_compose(
            environment_dir,
            compose_file,
            host_access=host_access,
            host_remedy=_HOST_REMEDY,
            outside=lambda path: staged.stage_file(Path(staged.original(str(path)))),
            original=staged.original,
            path_alias=read_alias,
        )
        if loaded.refused:
            # Compose would read files nobody may check: it is not run.
            result.refused += loaded.refused
            return result
    docker_args = tuple(docker_args)
    command = [
        "docker",
        *docker_args,
        "compose",
        "--project-name",
        project_name,
        "--project-directory",
        str(environment_dir),
    ]
    for path in files:
        command += ["-f", str(path)]
    resolved = _run_config(command, env)
    if resolved.error is not None or resolved.project is None:
        result.refused.append(_config_refusal(resolved.error))
        return result
    reads = sorted(resolved.variables)
    # A task value Compose is not given (it steers a program Compose runs
    # here) that the files read: they get nothing. Say so up front.
    result.refused += [
        TaskEnvRequest(
            "sandbox.env",
            "",
            None,
            f"sandbox.env sets {key}, which programs Compose runs on this "
            "machine read, so it is not passed to Compose",
            remedy=_HOST_PROGRAM_REMEDY,
        )
        for key in reads
        if key in provided and key not in names and host_program_variable(key)
    ]
    # The ssh transport's agent is in Compose's environment (remote-docker
    # over ssh): a file reading it would hand the task the operator's way
    # into the remote host.
    result.refused += [
        TaskEnvRequest(
            field_name,
            name,
            None,
            f"{field_name} reads ${{{name}}}, which remote-docker reaches its host with",
        )
        for name in reads
        if name in transport and name not in names
    ]
    given = resolved.environment | set(env)
    for name in reads:
        if name in given:
            continue
        default = resolved.variables[name].get("DefaultValue")
        result.defaulted.append(
            TaskEnvRequest(
                field_name,
                name,
                default if isinstance(default, str) else "",
                f"{field_name} reads ${{{name}}}",
            )
        )
    # Preserve all content; canonicalize only identified host path fields,
    # before putting original task paths into the private snapshot.
    operator_project = _operator_project_paths(
        resolved.project, names, operator_environ or {}
    )
    printed = _launchable(
        _map_project_paths(operator_project, map_paths or staged.snapshot_path)
    )
    try:
        project = decode_dollars(printed)
    except _Undecodable as exc:
        result.refused.append(
            _config_refusal(
                f"it printed {str(exc)[:80]!r} with a single $, so BenchFlow "
                "cannot tell what the run gets"
            )
        )
        return result
    if check is not None:
        result.refused += check.check(project)
    if result.refused:
        return result
    frozen_sources: dict[str, str] = {}
    error = _environment_sources(
        printed, command, env, staged, frozen=frozen_sources if freeze else None
    )
    if error is not None:
        result.refused.append(_config_refusal(error))
    if result.refused or not freeze:
        return result
    if agent_security:
        from benchflow.sandbox._compose_runtime import with_no_new_privileges

        services = printed.get("services")
        for service in services.values() if isinstance(services, dict) else ():
            if isinstance(service, dict):
                service["security_opt"] = with_no_new_privileges(
                    service.get("security_opt")
                )
    launch_env = {**env, **frozen_sources}
    launch_dir = staged.private_folder(f"compose-{project_name}")
    launch_file = launch_dir / "compose.json"

    def write_and_read_back(document: dict) -> _Resolved:
        fd = os.open(launch_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(document, out, indent=1)
        return _run_config(
            [
                "docker",
                *docker_args,
                "compose",
                "--project-name",
                project_name,
                "--project-directory",
                str(launch_dir),
                "-f",
                str(launch_file),
            ],
            launch_env,
            project_only=True,
        )

    reread = write_and_read_back(printed)
    if reread.error is not None or reread.project is None:
        result.refused.append(_config_refusal(reread.error, " once checked"))
        return result
    again = _launchable(reread.project)
    spelled_out = _create_host_path_spelled_out(printed, again)
    if spelled_out is not None:
        # Compose 5 leaves a bind's create_host_path: true out of what it
        # prints and reads a bind without the key as false. The launch file
        # spells the value out, which it reads as true and prints as it did:
        # the launch creates what the checked project creates, and still
        # reads back to exactly that project.
        reread = write_and_read_back(spelled_out)
        if reread.error is not None or reread.project is None:
            result.refused.append(_config_refusal(reread.error, " once checked"))
            return result
        again = _launchable(reread.project)
    if again != printed:
        result.refused.append(
            _config_refusal(
                "Compose reads the checked project back differently, at "
                f"{_first_difference(printed, again)}"
            )
        )
        return result
    result.launch = ComposeLaunch(
        path=launch_file,
        project_dir=launch_dir,
        project_name=project_name,
        env=launch_env,
        docker_args=docker_args,
        files=tuple(files),
        key=launch_key(files, env, docker_args, project_name, environment_dir),
    )
    return result


# The gate


def _gate_files(
    environment_dir: Path, sandbox_config: Any, *, remote: bool
) -> list[Path]:
    from benchflow.sandbox._compose import (
        COMPOSE_BASE_PATH,
        COMPOSE_REMOTE_BASE_PATH,
        compose_file_paths,
    )

    paths = compose_file_paths(
        environment_dir,
        sandbox_config,
        base=COMPOSE_REMOTE_BASE_PATH if remote else COMPOSE_BASE_PATH,
        prebuilt=bool(sandbox_config.docker_image),
        no_network=False,
    )
    return [path.resolve().absolute() for path in paths]


def benchflow_compose_values(
    environment_dir: Path,
    sandbox_config: Any,
    folders: Mapping[str, Path],
    *,
    image_name: str,
) -> dict[str, str]:
    """BenchFlow's Compose variables as a run sets them."""
    from benchflow.sandbox._docker_sweep import PROCESS_ENV, process_token
    from benchflow.task.paths import SandboxPaths

    values = {
        "MAIN_IMAGE_NAME": image_name,
        "CONTEXT_DIR": str(environment_dir.resolve().absolute()),
        "ENV_VERIFIER_LOGS_PATH": str(SandboxPaths.verifier_dir),
        "ENV_AGENT_LOGS_PATH": str(SandboxPaths.agent_dir),
        "ENV_ARTIFACTS_PATH": str(SandboxPaths.artifacts_dir),
        "CPUS": str(sandbox_config.cpus),
        "MEMORY": f"{sandbox_config.memory_mb}M",
        PROCESS_ENV: process_token(),
    }
    values.update((name, _local_realpath(str(path))) for name, path in folders.items())
    if sandbox_config.docker_image:
        values["PREBUILT_IMAGE_NAME"] = sandbox_config.docker_image
    return values


def _image_name(task_name: str) -> str:
    """The image name a run of the task builds (DockerSandbox's)."""
    from benchflow.sandbox.docker import _sanitize_docker_image_name

    return _sanitize_docker_image_name(f"bf__{task_name}")


def _named_folders(names: Iterable[str], environ: Mapping[str, str]) -> list[str]:
    """The absolute paths the operator chose by naming a variable that holds
    one (robohub's ``${ROBOUSE_GPU_REMOTE}``, a worker-endpoints file): a
    task may reach that path, or below it, and nothing else through it."""
    found: list[str] = []
    for name in names:
        value = environ.get(name, "")
        if name not in _CONNECTION_NAMES and value.startswith("/"):
            found.append(_local_realpath(value))
    return found


def project_check(
    staged: Any,
    folders: Iterable[Path | str],
    *,
    names: frozenset[str],
    environ: Mapping[str, str],
    remote: bool,
    agent: bool,
    project: str | None,
    read_only_binds: Iterable[tuple[Path | str, str]] = (),
    read_only_volumes: Iterable[tuple[str, str]] = (),
) -> _ProjectCheck:
    """The check of a staged task's resolved project on docker or
    remote-docker: paths may be in the private task copy, *folders* (BenchFlow's
    trial folders, the operator's mounts) and the named variables' paths;
    *read_only_binds* and *read_only_volumes* are BenchFlow's own (source
    or volume name, target) mounts, admitted only read-only at that target."""
    return _ProjectCheck(
        read_only_volumes=tuple(
            (str(name), _clean_path(target)) for name, target in read_only_volumes
        ),
        read_only_binds=tuple(
            (_local_realpath(str(source)), _clean_path(target))
            for source, target in read_only_binds
        ),
        project_dir=str(staged.environment_dir),
        folders=(
            _local_realpath(str(staged.task_dir)),
            *(_local_realpath(str(f)) for f in folders),
            *_named_folders(names, environ),
        ),
        agent=agent,
        remote_binds=remote,
        machine="the remote Docker host" if remote else "this machine",
        project=project,
    )


def gate_connection(
    sandbox_type: str, environ: Mapping[str, str]
) -> tuple[list[str], dict[str, str]]:
    """The docker CLI's global flags and the transport's environment the
    run of *sandbox_type* uses: the operator's connection with the hermetic
    client configuration (:func:`hermetic_docker_config`), or the remote
    host's."""
    config = {**environ, "DOCKER_CONFIG": hermetic_docker_config(environ)}
    if sandbox_type != "remote-docker":
        return docker_connection_args(config), {}
    from benchflow.sandbox.remote_docker import (
        RemoteDockerConfigError,
        resolve_remote_docker_host,
    )

    try:
        host = resolve_remote_docker_host()
    except RemoteDockerConfigError:
        # A host that is not set up: the sandbox refuses it when made.
        return docker_connection_args(config), {}
    return host.compose_args(config), host.compose_transport_env(environ)


# One check per set of inputs in this process (PERF-D). A
# job's admission, the trial factory's launch check and the trial's own
# admission ask the same question of the same task with the same values: the
# answer is kept under a key over the inputs the check reads, so a later call
# with any of them changed checks again. The inputs: the whole task folder
# staging copies and refuses on (every entry by its raw name, file contents
# included, and where each link lands), the task's values and the names allowed
# through, the sandbox and its config, this process's environment (it carries
# the Docker connection), the operator's docker config.json and current
# context, and the Docker CLI and Compose plugin files. Off when host access is
# allowed (the check may then read files outside the folder), when a named
# variable holds a folder (the check reads below it, which the key does not
# cover), and for a task too large to hash cheaply (found by a stat-only pass,
# before any file is read). An answer is not kept when Compose could not run
# (a timeout, an OSError), when staging found the folder changing, or when the
# folder changed while it was checked.
_CHECK_MEMO_SIZE = 32
# A miss hashes the folder twice (before and after the check) and every hit
# once; hashing reads at about 1.5 GiB/s against a fresh check's staging copy
# and Compose run, so past ~100 MiB three calls cost more kept than checked
# (PERF-D r3, reviews A-L2 and B-R2-L5). 64 MiB keeps the memo a win.
_CHECK_MEMO_MAX_BYTES = 64 * 1024 * 1024
_CHECK_MEMO_MAX_ENTRIES = 20_000
_CHECK_MEMO: dict[str, ComposeCheck] = {}
_CHECK_MEMO_LOCK = threading.Lock()
#: What made the current check's answer depend on more than its inputs (a
#: Compose run that failed for a reason other than the task, staging that
#: found the folder changing): such an answer is not kept.
_TRANSIENT_FAILURES: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "benchflow_compose_transient", default=None
)


def _note_transient(reason: str) -> None:
    """The current check's answer depends on more than its inputs: not kept."""
    failures = _TRANSIENT_FAILURES.get()
    if failures is not None:
        failures.append(reason)


#: Where the Docker CLI looks for its Compose plugin besides the config
#: folder's cli-plugins and cliPluginsExtraDirs.
_SYSTEM_CLI_PLUGIN_DIRS = (
    "/usr/local/lib/docker/cli-plugins",
    "/usr/local/libexec/docker/cli-plugins",
    "/usr/lib/docker/cli-plugins",
    "/usr/libexec/docker/cli-plugins",
)


def _folder_digest(root: Path) -> str | None:
    """Every entry under *root* (type, raw name, mode; a file's sha256, a
    link's target and where it lands), without following links; None when it
    cannot be read or is too large to hash cheaply. Sizes and entry counts are
    taken first, from lstat alone, so a folder over the caps reads no file."""
    entries: list[tuple[str, str, os.stat_result]] = []
    budget = {"bytes": _CHECK_MEMO_MAX_BYTES, "entries": _CHECK_MEMO_MAX_ENTRIES}

    def walk(folder: str, rel: str) -> bool:
        with os.scandir(folder) as found:
            listed = sorted(found, key=lambda e: os.fsencode(e.name))
        for entry in listed:
            budget["entries"] -= 1
            if budget["entries"] < 0:
                return False
            path = f"{rel}/{entry.name}"
            info = entry.stat(follow_symlinks=False)
            entries.append((path, entry.path, info))
            if stat.S_ISDIR(info.st_mode):
                if not walk(entry.path, path):
                    return False
            elif stat.S_ISREG(info.st_mode):
                budget["bytes"] -= info.st_size
                if budget["bytes"] < 0:
                    return False
        return True

    digest = hashlib.sha256()

    def line(text: str) -> None:
        # Names are bytes on POSIX: a name that is not UTF-8 is kept as its
        # own bytes (surrogateescape), never refused or merged with another.
        digest.update(text.encode("utf-8", "surrogateescape") + b"\n")

    try:
        info = os.lstat(root)
        line(f"root {os.path.realpath(root)} {info.st_mode:o}")
        if stat.S_ISDIR(info.st_mode) and not walk(str(root), "."):
            return None
        for path, full, info in entries:
            mode = f"{info.st_mode:o}"
            if stat.S_ISLNK(info.st_mode):
                # Staging decides by where the link lands, links outside the
                # task followed: key that too, not only the link's text.
                line(f"l {path} {mode} {os.readlink(full)} {os.path.realpath(full)}")
            elif stat.S_ISDIR(info.st_mode):
                line(f"d {path} {mode}")
            elif stat.S_ISREG(info.st_mode):
                content = hashlib.sha256()
                with open(full, "rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        content.update(block)
                line(f"f {path} {mode} {content.hexdigest()}")
            else:
                line(f"o {path} {mode}")
    except OSError:
        return None
    return digest.hexdigest()


def _file_identity(path: str | Path) -> str:
    """A file's bytes (sha256), or why it could not be read."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        return f"unreadable {exc.errno}"


def _binary_identity(path: str | Path) -> list[Any]:
    """Where a binary lands and which file it is (no read: Compose is large)."""
    try:
        real = os.path.realpath(path)
        info = os.stat(real)
    except OSError:
        return [str(path), None]
    return [str(path), real, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def _client_inputs(environ: Mapping[str, str]) -> list[Any]:
    """The operator's Docker client state the check reads besides the process
    environment: config.json (its current context decides the filesystem
    proof; :func:`hermetic_docker_config` copies it), the current context's
    meta.json, and the Docker CLI and Compose plugin files that run."""
    folder = operator_docker_config_dir(environ)
    config_file = folder / "config.json"
    inputs: list[Any] = [str(folder), _file_identity(config_file)]
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    context = environ.get("DOCKER_CONTEXT") or data.get("currentContext")
    if isinstance(context, str) and context:
        name = hashlib.sha256(context.encode()).hexdigest()
        inputs.append(_file_identity(folder / "contexts" / "meta" / name / "meta.json"))
    # Not shutil.which: callers stand in for it with one that takes a name only.
    docker = next(
        (
            path
            for folder in (environ.get("PATH") or os.defpath).split(os.pathsep)
            if os.access(path := os.path.join(folder or ".", "docker"), os.X_OK)
            and not os.path.isdir(path)
        ),
        None,
    )
    inputs.append(_binary_identity(docker) if docker else None)
    extra = data.get("cliPluginsExtraDirs")
    plugin_dirs = [
        *(str(d) for d in (extra if isinstance(extra, list) else ())),
        str(folder / "cli-plugins"),
        *_SYSTEM_CLI_PLUGIN_DIRS,
    ]
    inputs.append(
        [
            _binary_identity(p)
            for d in plugin_dirs
            if os.path.lexists(p := os.path.join(d, "docker-compose"))
        ]
    )
    return inputs


def _check_key(
    environment_dir: Path,
    provided: Mapping[str, str] | Iterable[str],
    allowed: Iterable[str] | None,
    sandbox_type: str,
    sandbox_config: Any,
) -> str | None:
    if host_access_allowed():
        return None
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    environ = os.environ
    if _named_folders(names, environ):
        # The check resolves and walks what a task binds below a named
        # folder (links, modes, writable trees): not keyed, so not kept.
        return None
    # stage_environment copies the folder holding the environment folder's
    # real path, the whole task, and refuses on any entry of it.
    environment = os.path.realpath(environment_dir)
    folder = _folder_digest(Path(os.path.dirname(environment)))
    if folder is None:
        return None
    if isinstance(provided, Mapping):
        given: Any = sorted((str(k), repr(v)) for k, v in provided.items())
    else:
        given = sorted(str(k) for k in provided)
    dump = getattr(sandbox_config, "model_dump_json", None)
    config = dump() if callable(dump) else repr(sandbox_config)
    text = json.dumps(
        [
            str(environment_dir),
            environment,
            folder,
            isinstance(provided, Mapping),
            given,
            None if allowed is None else sorted(allowed),
            sorted(allowed_task_env()) if allowed is None else None,
            sandbox_type,
            config,
            sorted(environ.items()),
            _client_inputs(environ),
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()


def check_compose(
    environment_dir: Path,
    provided: Mapping[str, str] | Iterable[str] = (),
    allowed: Iterable[str] | None = None,
    *,
    sandbox_type: str = "docker",
    sandbox_config: Any = None,
) -> ComposeCheck:
    """:func:`_check_compose`, once per set of inputs in this process (see
    ``_CHECK_MEMO``); each caller gets its own copy of the answer."""
    import copy

    if not isinstance(provided, Mapping):
        provided = tuple(provided)
    if allowed is not None:
        allowed = tuple(allowed)
    key = _check_key(
        Path(environment_dir), provided, allowed, sandbox_type, sandbox_config
    )
    if key is not None:
        with _CHECK_MEMO_LOCK:
            kept = _CHECK_MEMO.get(key)
        if kept is not None:
            return copy.deepcopy(kept)
    failures: list[str] = []
    token = _TRANSIENT_FAILURES.set(failures)
    try:
        result = _check_compose(
            environment_dir,
            provided,
            allowed,
            sandbox_type=sandbox_type,
            sandbox_config=sandbox_config,
        )
    finally:
        _TRANSIENT_FAILURES.reset(token)
    if (
        key is not None
        and result.launch is None
        and not failures
        # The folder may have changed between the key and the check's own
        # staging: keep the answer only under the state it was read in.
        and _check_key(
            Path(environment_dir), provided, allowed, sandbox_type, sandbox_config
        )
        == key
    ):
        with _CHECK_MEMO_LOCK:
            _CHECK_MEMO[key] = copy.deepcopy(result)
            while len(_CHECK_MEMO) > _CHECK_MEMO_SIZE:
                _CHECK_MEMO.pop(next(iter(_CHECK_MEMO)))
    return result


def _check_compose(
    environment_dir: Path,
    provided: Mapping[str, str] | Iterable[str] = (),
    allowed: Iterable[str] | None = None,
    *,
    sandbox_type: str = "docker",
    sandbox_config: Any = None,
) -> ComposeCheck:
    """Check the task's Compose setup as its run would, before any sandbox
    starts, on a private copy of the task's environment
    folder that is removed after: the same strict reading, resolution and
    check as :func:`freeze_compose`, with BenchFlow's trial folders replaced
    by stand-ins and nothing frozen. The run itself freezes its own copy.

    *provided* is what the task and the manifest put in Compose's
    environment (:func:`task_compose_values`); *sandbox_config* the task's
    sandbox config (its image, network and resources pick BenchFlow's own
    Compose files and values, as the run's).
    """
    from benchflow.task.config import SandboxConfig
    from benchflow.task.staging import StagingRefused, stage_environment

    environment_dir = Path(environment_dir)
    names = allowed_task_env() if allowed is None else frozenset(allowed)
    values: dict[str, str] = (
        {str(k): v for k, v in provided.items() if isinstance(v, str)}
        if isinstance(provided, Mapping)
        else {}
    )
    if sandbox_config is None:
        sandbox_config = SandboxConfig()
    host_access = host_access_allowed()
    result = ComposeCheck()
    if sandbox_type in COMPOSE_ON_HOST:
        # BenchFlow's own Compose variables name the image it builds and the
        # host folders it mounts (docker-compose-base.yaml); a task's value
        # for one would mount a host folder of the task's choosing.
        # compose_process_env puts BenchFlow's values last, so the task's is
        # not used: say so.
        result.refused += [
            TaskEnvRequest(
                "sandbox.env",
                "",
                None,
                f"sandbox.env sets {key}, which BenchFlow sets for Compose itself",
                remedy=_BENCHFLOW_REMEDY,
            )
            for key in sorted(BENCHFLOW_COMPOSE_VARIABLES & set(provided))
        ]
    hook = (environment_dir / PRE_COMPOSE_HOOK).is_file()
    if not compose_path(environment_dir).exists():
        if hook and not host_access and sandbox_type in COMPOSE_ON_HOST:
            result.refused.append(_hook_refusal())
        return result
    try:
        staged = stage_environment(environment_dir, allow_outside=host_access)
    except StagingRefused as exc:
        if exc.transient:
            _note_transient("staging")
        result.refused += exc.refused
        return result
    try:
        if sandbox_type == "daytona":
            result.refused += _dind_preflight(
                staged, values, sandbox_config, host_access
            ).refused
            return result
        folders = {
            name: staged.private_folder(name.lower())
            for name in (
                "HOST_VERIFIER_LOGS_PATH",
                "HOST_AGENT_LOGS_PATH",
                "HOST_ARTIFACTS_PATH",
            )
        }
        environ = os.environ
        docker_args, transport = gate_connection(sandbox_type, environ)
        env = {
            **compose_process_env(
                values,
                benchflow_compose_values(
                    staged.environment_dir,
                    sandbox_config,
                    folders,
                    image_name=_image_name(environment_dir.parent.name),
                ),
                environ=environ,
                allowed=names,
                host_access=host_access,
            ),
            **transport,
        }
        remote = sandbox_type == "remote-docker"
        local_proof = daemon_filesystem_local(environ, docker_args)
        project = f"benchflow-check-{os.getpid()}-{os.urandom(4).hex()}"
        checked = freeze_compose(
            staged=staged,
            files=_gate_files(staged.environment_dir, sandbox_config, remote=remote),
            project_name=project,
            env=env,
            docker_args=docker_args,
            check=None
            if host_access
            else project_check(
                staged,
                folders.values(),
                names=names,
                environ=environ,
                remote=remote or not local_proof,
                agent="SSH_AUTH_SOCK" in transport and "SSH_AUTH_SOCK" not in names,
                project=project,
            ),
            host_access=host_access,
            provided=values,
            names=names,
            transport=transport,
            operator_environ=environ,
            freeze=False,
        )
        result.refused += checked.refused
        result.defaulted += checked.defaulted
    finally:
        staged.remove()
    if hook and not host_access and sandbox_type in COMPOSE_ON_HOST:
        result.refused.append(_hook_refusal())
    return result


def _hook_refusal() -> TaskEnvRequest:
    return TaskEnvRequest(
        f"environment/{PRE_COMPOSE_HOOK}",
        "",
        None,
        f"environment/{PRE_COMPOSE_HOOK} runs on this machine before Compose starts",
        remedy=_HOST_REMEDY,
    )


def refused_compose_env(
    environment_dir: Path,
    provided: Mapping[str, str] | Iterable[str] = (),
    allowed: Iterable[str] | None = None,
    *,
    sandbox_type: str = "docker",
    sandbox_config: Any = None,
) -> list[TaskEnvRequest]:
    """:func:`check_compose`'s refusals."""
    return check_compose(
        environment_dir,
        provided,
        allowed,
        sandbox_type=sandbox_type,
        sandbox_config=sandbox_config,
    ).refused


# Daytona DinD: Compose runs in the sandbox's VM


def _compose_cli_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        found = subprocess.run(
            ["docker", "compose", "version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_CONFIG_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _note_transient(type(exc).__name__)
        return False
    return found.returncode == 0


def _dind_preflight(
    staged: Any, values: Mapping[str, str], sandbox_config: Any, host_access: bool
) -> ComposeCheck:
    """The check of a Daytona DinD task before its sandbox is created
    , on the staged copy Daytona then uploads: the strict
     reading, bounded by the environment folder (all Daytona uploads), and,
     where ``docker compose`` is installed here (it reads no daemon for
     ``config``), the project resolved and checked as the VM's Compose would
     resolve it: DinD's own values with the task's winning, paths allowed
     only in the uploaded task folder and BenchFlow's log folders. Without
     Compose here, a service setting written through a variable is refused,
     since only the VM could resolve it; the VM check runs either way."""
    from types import SimpleNamespace

    from benchflow.task.compose_load import read_compose
    from benchflow.task.staging import copy_tree

    result = ComposeCheck()
    environment_dir = Path(staged.environment_dir)
    loaded = read_compose(
        environment_dir,
        compose_path(environment_dir),
        host_access=host_access,
        host_remedy=_HOST_REMEDY,
        original=staged.original,
        path_alias=(DIND_ENVIRONMENT_DIR, environment_dir),
    )
    if loaded.refused or host_access:
        result.refused += loaded.refused
        return result
    folders = {
        name: staged.private_folder(f"dind-{name.lower()}")
        for name in (
            "HOST_VERIFIER_LOGS_PATH",
            "HOST_AGENT_LOGS_PATH",
            "HOST_ARTIFACTS_PATH",
        )
    }
    if not _compose_cli_available():
        if len(loaded.files) > 1:
            result.refused.append(
                _host_refusal(
                    f"environment/{COMPOSE_FILE}",
                    "uses include or extends files, which need docker compose "
                    "to resolve before a Daytona sandbox is created: install "
                    "docker compose here",
                )
            )
        result.refused += _static_dind_check(environment_dir, values)
        return result
    # Resolve in a disposable copy: absolute VM dependency paths must name
    # local staged files for config's reads, while the uploaded source keeps
    # its actual VM coordinates.
    resolver_dir = staged.private_folder("dind-resolver") / "environment"
    copy_tree(environment_dir, resolver_dir, label="environment")
    _dind_resolver_files(environment_dir, resolver_dir, loaded.files)
    resolver = SimpleNamespace(
        environment_dir=resolver_dir,
        original=lambda path: path,
        snapshot_path=lambda path: path,
        private_folder=staged.private_folder,
        stage_file=lambda path: None,
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        **dict.fromkeys(_REMOTE_INCLUDE_SWITCHES, "false"),
        **benchflow_compose_values(
            environment_dir,
            sandbox_config,
            folders,
            image_name=_image_name(staged.original_task_dir.name),
        ),
        "CONTEXT_DIR": DIND_ENVIRONMENT_DIR,
        "HOST_VERIFIER_LOGS_PATH": f"{DIND_LOGS_DIR}/verifier",
        "HOST_AGENT_LOGS_PATH": f"{DIND_LOGS_DIR}/agent",
        "HOST_ARTIFACTS_PATH": f"{DIND_LOGS_DIR}/artifacts",
        # DinD lets a task's value win over BenchFlow's (daytona_dind.py).
        **values,
    }
    docker_args, _ = gate_connection("docker", os.environ)
    check = _ProjectCheck(
        project_dir=DIND_ENVIRONMENT_DIR,
        folders=(
            DIND_TASK_DIR,
            DIND_ENVIRONMENT_DIR,
            f"{DIND_LOGS_DIR}/verifier",
            f"{DIND_LOGS_DIR}/agent",
            f"{DIND_LOGS_DIR}/artifacts",
        ),
        resolve=lambda path: _dind_local_path(path, environment_dir),
        identity=lambda path: _dind_local_identity(path, environment_dir),
        defer_filesystem=True,
        machine="the sandbox's VM",
    )
    checked = freeze_compose(
        staged=resolver,
        files=_gate_files(resolver_dir, sandbox_config, remote=False),
        project_name=f"benchflow-check-{os.getpid()}-{os.urandom(4).hex()}",
        env=env,
        docker_args=docker_args,
        check=check,
        host_access=False,
        freeze=False,
        map_paths=lambda path: _dind_vm_path(path, resolver_dir),
        read_alias=(DIND_ENVIRONMENT_DIR, resolver_dir),
    )
    result.refused += checked.refused
    return result


def _dind_resolver_files(source: Path, target: Path, files: Iterable[Path]) -> None:
    """Translate only a private resolver's VM path coordinates, preserving
    YAML node tags, merges and aliases for Compose to interpret itself."""
    import yaml

    from benchflow.task.compose_load import compose_nodes

    for path in files:
        copied = target / path.relative_to(source)
        documents = compose_nodes(copied.read_text(encoding="utf-8"))
        if documents is None:
            continue  # the original strict reading already refused this
        stack = list(documents)
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, yaml.ScalarNode):
                root = DIND_ENVIRONMENT_DIR
                if node.value == root or node.value.startswith(root + "/"):
                    node.value = str(target) + node.value[len(root) :]
            elif isinstance(node, yaml.MappingNode):
                stack += [value for _, value in node.value]
            elif isinstance(node, yaml.SequenceNode):
                stack += node.value
        copied.write_text(yaml.serialize_all(documents), encoding="utf-8")


def _dind_local_path(path: str, environment_dir: Path) -> str:
    """Check VM paths against the copy that Daytona uploads, including links."""
    root = DIND_TASK_DIR
    task_dir = environment_dir.parent
    if path == root or path.startswith(root + "/"):
        local = _local_realpath(str(task_dir) + path[len(root) :])
        copied = str(task_dir)
        if local == copied or local.startswith(copied + "/"):
            return root + local[len(copied) :]
        return local
    return _clean_path(path)


def _dind_local_identity(path: str, environment_dir: Path) -> Any:
    """What a VM path is, read on the copy Daytona uploads. BenchFlow's log
    folders are not on this machine: the VM's check reads them."""
    root = DIND_TASK_DIR
    if path == root or path.startswith(root + "/"):
        return _local_identity(str(environment_dir.parent) + path[len(root) :])
    return None


def _dind_vm_path(path: str, environment_dir: Path) -> str:
    copied = str(environment_dir.parent)
    if path == copied or path.startswith(copied + "/"):
        return DIND_TASK_DIR + path[len(copied) :]
    return path


#: The service keys whose value, written through a variable, only Compose
#: can resolve (:func:`_static_dind_check`).
_SETTING_KEYS = (
    "privileged",
    "use_api_socket",
    "cap_add",
    "security_opt",
    "devices",
    "device_cgroup_rules",
    "volumes_from",
    *_NAMESPACE_KEYS,
)


def _static_dind_check(
    environment_dir: Path, values: Mapping[str, str] | None = None
) -> list[TaskEnvRequest]:
    """Without Compose here: the literal service settings of the task's own
    Compose file that hand a container the VM, and those written through a
    variable (only Compose could resolve them)."""
    field_name = f"environment/{COMPOSE_FILE}"
    documents = _load_all(
        compose_path(environment_dir).read_text(encoding="utf-8", errors="replace")
    )
    refused: list[TaskEnvRequest] = []
    unresolved_paths = any(
        key in (values or {})
        for key in (
            "CONTEXT_DIR",
            "HOST_VERIFIER_LOGS_PATH",
            "HOST_AGENT_LOGS_PATH",
            "HOST_ARTIFACTS_PATH",
        )
    )
    if unresolved_paths or documents is None:
        refused.append(
            _host_refusal(
                field_name,
                "needs docker compose to resolve its runtime paths before "
                "a Daytona sandbox is created: install docker compose here",
            )
        )
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        services = document.get("services")
        if any(key in document for key in ("volumes", "secrets", "configs")) or any(
            isinstance(service, dict)
            and any(
                key in service for key in ("volumes", "build", "secrets", "configs")
            )
            for service in (services.values() if isinstance(services, dict) else ())
        ):
            refused.append(
                _host_refusal(
                    field_name,
                    "uses paths that need docker compose to resolve before "
                    "a Daytona sandbox is created: install docker compose here",
                )
            )
        if "include" in document or any(
            isinstance(service, dict) and "extends" in service
            for service in (services.values() if isinstance(services, dict) else ())
        ):
            refused.append(
                _host_refusal(
                    field_name,
                    "uses include or extends, which need docker compose "
                    "to resolve before a Daytona sandbox is created: install "
                    "docker compose here",
                )
            )
        refused += [
            _host_refusal(field_name, f"{setting}, which reaches the sandbox's VM")
            for setting in _host_settings(document)
        ]
        for name, service in services.items() if isinstance(services, dict) else ():
            if not isinstance(service, dict):
                continue
            build = (
                service.get("build") if isinstance(service.get("build"), dict) else {}
            )
            for key, value in [
                *((k, service.get(k)) for k in _SETTING_KEYS),
                *((f"build.{k}", build.get(k)) for k in ("privileged", "network")),
            ]:
                if "$" in json.dumps(value, default=str):
                    refused.append(
                        _host_refusal(
                            field_name,
                            f"sets {key} of service {name} through a variable, "
                            "which only docker compose can resolve: install "
                            "docker compose here, or write the value",
                        )
                    )
    return refused


_VM_PROBE = (
    'd=$(mktemp -d) && printf "services:\\n  p:\\n    image: busybox\\n'
    '    environment: {P: \\"x\\$\\$y\\"}\\n" > "$d/docker-compose.yaml" && '
    'docker compose -p benchflow-probe --project-directory "$d" config '
    '--format json; s=$?; rm -rf "$d"; exit $s'
)

# Prints where each argument lands with links followed: the longest part
# that exists, resolved, then the rest as written.
_VM_REALPATH = (
    'for p in "$@"; do q=$p; rest=; '
    'while [ ! -e "$q" ] && [ ! -L "$q" ] && [ "$q" != / ] && [ "$q" != . ]; do '
    'rest="/$(basename "$q")$rest"; q=$(dirname "$q"); done; '
    'r=$(readlink -f -- "$q" 2>/dev/null) || exit 1; '
    'q=${r%/}$rest; printf "%s\\n" "${q:-/}"; done'
)

# Prints what each argument is, links followed (device:inode, the same
# value for every spelling of one folder), or - when nothing is there.
_VM_IDENTITY = 'for p in "$@"; do stat -L -c %d:%i -- "$p" 2>/dev/null || echo -; done'

# A transported mount inventory must prove that no records were dropped.
_VM_MOUNTINFO = (
    'awk \'{print NR " " $0} END {print "complete " NR}\' /proc/self/mountinfo'
)


def _vm_mountpoints(output: str) -> set[str] | None:
    lines = output.splitlines()
    records = [line.partition(" ") for line in lines[:-1]]
    if (
        not records
        or not lines
        or lines[-1] != f"complete {len(records)}"
        or any(
            number != str(index) or not separator
            for index, (number, separator, _) in enumerate(records, 1)
        )
    ):
        return None
    try:
        return _mountpoints("\n".join(line for _, _, line in records))
    except ValueError:
        return None


# Number every record and finish with the count: even a VM transport that
# truncates output cannot turn a partial population into a complete proof.
# Bound output at its producer. No path names enter stdout.
_VM_TREE = (
    '(if [ -d "$1" ]; then '
    'find "$1" -type d -exec stat -L -c %d:%i -- {} +; '
    'else stat -L -c %d:%i -- "$1"; fi && '
    'printf "%s\\n" complete) | ('
    "n=0; last=; while IFS= read -r line; do n=$((n + 1)); "
    '[ "$n" -le "$2" ] || exit 1; last=$line; '
    'if [ "$line" = complete ]; then printf "complete %s\\n" "$((n - 1))"; '
    'else printf "%s %s\\n" "$n" "$line"; fi || exit 1; '
    'done; [ "$last" = complete ])'
)


def _vm_identity_line(line: str) -> bool:
    return re.fullmatch(r"[0-9]+:[0-9]+", line) is not None


async def check_compose_in_vm(
    run: Callable[[str], Awaitable[Any]],
    compose: list[str],
    project_dir: str,
    folders: Iterable[str],
) -> list[TaskEnvRequest]:
    """Check what Compose resolves inside a Daytona DinD sandbox's VM, before
     the task's pre-compose hook runs there and again before Compose builds
    . *run* runs a shell command in the VM with Compose's
     environment (BenchFlow's variables and the task's; nothing of the
     operator's) and returns its result; *compose* is the run's ``docker
     compose`` command up to its subcommand; *folders* are where a path may
     be (the uploaded environment folder, BenchFlow's log folders). Refused,
     unless the operator allowed host access: a path outside them (links
     followed in the VM) and the service settings that hand a container the
     VM. The VM's Compose prints a ``$`` doubled or not (asked with a probe);
     a doubled one is undone before the check.
    """
    if host_access_allowed():
        return []
    probe = await run(_VM_PROBE)
    escapes = probe.return_code == 0 and "x$$y" in (probe.stdout or "")
    found = await run(shlex.join([*compose, "config", "--format", "json"]))
    if found.return_code and "MarshalJSON" in (found.stderr or ""):
        found = await run(shlex.join([*compose, "config", "--format", "yaml"]))
    printed = _parse_project(found.stdout or "") if not found.return_code else None
    if printed is None:
        error = _error_line((found.stderr or "") + "\n" + (found.stdout or ""))
        return [_config_refusal(error, " in the sandbox")]
    try:
        project = decode_dollars(printed) if escapes else printed
    except _Undecodable as exc:
        return [_config_refusal(f"it printed {str(exc)[:80]!r}", " in the sandbox")]
    check = _ProjectCheck(
        project_dir,
        tuple(_clean_path(f) for f in folders),
        machine="the sandbox's VM",
    )
    paths = sorted(p for p in check.paths(project) if "\n" not in p)
    landed: dict[str, str] = {}
    resolved_known = not paths
    if paths:
        resolved = await run(shlex.join(["sh", "-c", _VM_REALPATH, "realpath", *paths]))
        # Only LF is a record separator; every other whitespace character
        # can be part of an honest host path.
        lines = (resolved.stdout or "").split("\n")
        if lines[-1] == "":
            lines.pop()
        if not resolved.return_code and len(lines) == len(paths):
            landed = dict(zip(paths, lines, strict=True))
            resolved_known = True
    # A path the VM could not resolve is taken as written, except one with a
    # line break, which lands nowhere the check knows: outside.
    check.resolve = lambda path: landed.get(path, path if "\n" not in path else "/")
    # The same identity rule as this machine's check (S1-CONF-2). Without
    # the VM's answer, a project with a writable mount is refused.
    identities: dict[str, str] = {}
    known = resolved_known
    if paths:
        stats = await run(shlex.join(["sh", "-c", _VM_IDENTITY, "identity", *paths]))
        lines = (stats.stdout or "").splitlines()
        if not stats.return_code and len(lines) == len(paths):
            identities = {
                p: line
                for p, line in zip(paths, lines, strict=True)
                if _vm_identity_line(line)
            }
        else:
            known = False
    check.identity = identities.get
    check.identities_known = known
    mounts = await run(_VM_MOUNTINFO)
    points = _vm_mountpoints(mounts.stdout or "") if not mounts.return_code else None
    check.mountpoints = lambda: points
    trees: dict[str, set[Any] | None] = {}
    for source in sorted(check.writable_sources(project)):
        tree = await run(
            shlex.join(
                ["sh", "-c", _VM_TREE, "tree", source, str(_SOURCE_DIRECTORY_LIMIT + 2)]
            )
        )
        lines = (tree.stdout or "").splitlines()
        records = [line.partition(" ") for line in lines[:-1]]
        trees[source] = (
            {identity for _, _, identity in records}
            if not tree.return_code
            and lines
            and lines[-1] == f"complete {len(records)}"
            and 1 <= len(records) <= _SOURCE_DIRECTORY_LIMIT
            and len({identity for _, _, identity in records}) == len(records)
            and all(
                number == str(index) and _vm_identity_line(identity)
                for index, (number, _, identity) in enumerate(records, 1)
            )
            else None
        )
    check.tree_identities = trees.get
    return check.check(project)
