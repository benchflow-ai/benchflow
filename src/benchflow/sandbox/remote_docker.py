"""Sandboxes on a Docker host the user controls (``--sandbox remote-docker``).

The local Docker provider (:class:`~benchflow.sandbox.docker.DockerSandbox`)
with the daemon somewhere else, reached over ``ssh://user@host[:port]`` or
``tcp://host:port`` with TLS. Compose files, sandbox user, verifier
hardening, network modes and snapshots are the local provider's. What
differs:

- Compose gets the address as ``-H`` (TLS files as flags) on its own
  command line (``compose_args``, which the launch gate's ``compose config`` uses
  too); every other
  ``docker`` call gets it from a provider-owned ``docker`` shim first on
  ``PATH``. The address is never in the environment, which Compose
  interpolates a task's files from (``${DOCKER_HOST}`` is empty); the
  caller's ``DOCKER_HOST``, ``DOCKER_CONTEXT``, TLS variables,
  ``DOCKER_CONFIG`` (given as ``--config``), SSH agent (only the provider's
  ssh gets it) and login names are not in it, nor is the ``docker`` shim;
  its ``PATH`` starts with the provider's ssh folder. Builds go to buildx's ``default`` builder, the docker driver
  of that endpoint, whatever ``BUILDX_BUILDER`` or ``buildx use`` says.
- ssh runs through a shim that adds ``-o StrictHostKeyChecking=yes`` and
  ``-F`` a provider file setting it before it includes the operator's
  configuration; ssh hands ``-F`` to each ProxyJump hop's ssh, so a changed or
  unknown host key fails on every hop, whatever the configuration says. A
  hop through a ProxyCommand is refused.
- No host path is bind-mounted: the remote daemon cannot see the rollout
  folders and would create empty ones on its own disk. Logs, verifier output
  and artifacts are copied back (``is_mounted`` is False). A task's own bind
  mounts, ``file:`` secrets and configs, host-path volumes and ``external``
  volumes are refused, and so are a volume that already exists on the host
  without the trial's label and a build that asks for the SSH agent
  (``build.ssh``) (``_check_task_project``, after the pre-compose hook and
  before anything is built). Not checked here: service settings that hand a
  container the host (``privileged``, ``devices``, host namespaces, added
  capabilities, ``volumes_from``). A task's pre-compose hook is refused
  unless the operator sets ``BENCHFLOW_ALLOW_TASK_HOST_ACCESS=1`` (the launch
  gate's switch): it would run as the operator and can read whatever they
  can, the provider's shim folder (the address with its ssh user, the TLS
  file paths) and ~/.ssh included. Allowed, it runs with Compose's
  environment (no connection setting or shim).
- The host is checked (``docker info``) before anything starts: an
  unreachable host or one with fewer CPUs or less memory than the task asks
  for is a :class:`~benchflow.sandbox.protocol.SandboxStartupError`.
- Teardown kills the containers, runs ``down --remove-orphans``, removes
  what still carries the rollout's compose project label (its named volumes
  included, and the volumes BenchFlow names for what Docker would make
  anonymous, ``_trial_volumes``) and, when the
  sandbox is deleted, the images the project built. Never ``down --volumes`` or
  ``--rmi all``, which also take pre-existing volumes and pulled images.
- Messages, records and logs name the host by an opaque label
  (``remote-docker[ssh]``); ``BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS=1`` shows
  the address instead. The ssh user (often an access token) is redacted in
  each form docker or ssh prints it: as written, percent-decoded, and every
  login ssh's configuration picks for the host and its ProxyJump hops
  (``ssh -G``); so are the host names, in any letter case, and both names
  in a TLS name mismatch. An ssh host whose every hop ``ssh -G`` cannot
  resolve, or which some hop reaches through a ProxyCommand, is refused
  before any call. Output read
  as data (``compose config``, inspect and list calls) is never redacted;
  only the messages made from it are. An address docker would echo whole in an error (a user with an
  unencoded ``/``, ``?`` or ``#``, a bad ``%`` escape, a space or control
  character, a password, a broken ``[`` ``]`` host) is refused before any
  call, quoting no part of it.

What assumes the daemon is on this machine, and what each does here:

- Host disk: ``bench doctor`` reads free space with ``df`` only for a local
  daemon; the remote row reports version, CPUs and memory from ``docker
  info``, and a full remote disk is named when a build or start hits it.
- Build contexts and ``docker cp``: the CLI streams both over the
  connection; nothing is read from the daemon's disk.
- Bind mounts: none (see above); the mount probe never runs.
- Host ports: BenchFlow publishes none and connects to none; the model
  proxy runs in the sandbox.
- The image and build cache: images are built and kept on the remote host.
  The leftover sweep runs only for ``--sandbox docker``. ``bench sandbox
  cleanup`` acts on whatever daemon the caller's docker CLI reaches, which
  is this host when it came from ``DOCKER_HOST``; it removes every
  ``bf-snap-*`` image older than ``--max-age`` there, whoever made it.
- Firewall and lockdown rules live in each container, so they hold wherever
  it runs.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from benchflow.errors import UserError
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._compose import COMPOSE_REMOTE_BASE_PATH
from benchflow.sandbox._compose_teardown import compose_down_command
from benchflow.sandbox._docker_subnets import DockerDaemon
from benchflow.sandbox._exec_output import (
    diagnostic_result,
    exec_raw,
    raw_output_requested,
)
from benchflow.sandbox.docker import (
    DockerSandbox,
    _is_retryable_docker_build_error,
)
from benchflow.sandbox.env_layers import DockerLayers
from benchflow.sandbox.protocol import SandboxImage, SandboxStartupError
from benchflow.task.paths import SandboxPaths

REMOTE_DOCKER_HOST_ENV = "BENCHFLOW_REMOTE_DOCKER_HOST"
# Set to 1 to name the host's address (without its ssh user) in messages,
# records and logs instead of the opaque remote-docker[ssh] label.
REMOTE_DOCKER_SHOW_ADDRESS_ENV = "BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS"
_PROBE_TIMEOUT_SEC = 20
_REASON_LIMIT = 300
_TLS_FILES = ("ca.pem", "cert.pem", "key.pem")
_DISK_FULL = re.compile(r"no space left on device", re.IGNORECASE)
_LIST_TIMEOUT_SEC = 30
# A user this long is treated as a token: every occurrence is hidden from
# docker's and ssh's own messages. A shorter one is a login name ("ubuntu",
# "docker") that also names images and commands ("ubuntu:24.04"), so it is
# hidden where it stands as the user (``user@``, ``-l user``, ``'user'``),
# which is also the rule for a command's own output.
_TOKEN_USER_LEN = 8
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
# What Go's url.Parse (docker's parser) accepts in the user part. Anything
# else, or a bad %-escape, makes docker fail with an error that quotes the
# whole address, user included.
_USERINFO_OK = re.compile(r"^[A-Za-z0-9\-._:~!$&'()*+,;=%@]*$")
_BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
# A TCP host must survive Go's URL parse/string round trip unchanged. DNS
# names use ASCII (IDNA for international names); IPv6 keeps its brackets.
# Scoped IPv6 is refused: Docker parses the URL twice, so even a %25 zone
# separator followed by ASCII can be decoded again before HTTP errors.
# In particular, percent-encoded DNS names are not an alternative to IDNA.
_TCP_AUTHORITY = re.compile(r"(?:[A-Za-z0-9_.-]+|\[[A-Fa-f0-9:.]+\])(?::[0-9]*)?")
_TLS_DIAGNOSTIC = re.compile(r"\b(?:tls|x509):", re.IGNORECASE)
_TLS_CERTIFICATE_PREFIX = "tls: failed to verify certificate: "
_TRANSPORT_CERTIFICATE = (
    _TLS_CERTIFICATE_PREFIX
    + "<remote certificate error>; check the remote Docker TLS and proxy configuration"
)
_TRANSPORT_UNKNOWN = (
    "cannot connect to the Docker daemon; "
    "check the remote Docker host, TLS and proxy configuration"
)
_TRANSPORT_DISK_FULL = (
    "remote Docker host is out of disk space: no space left on device; "
    "free space on the remote Docker host and retry"
)
_SSH_TRANSPORT_MESSAGES = {
    "permission denied": "Permission denied (publickey); check SSH authentication and remote Docker access",
    "host key verification failed": "Host key verification failed; verify the remote SSH host key and known_hosts",
    "connection refused": "Connection refused; check the remote SSH host and port",
}
# Python's urlsplit drops tab, CR and LF, but docker keeps them and quotes the
# whole address with them escaped (``\t``), which no form of the user matches.
_SPACE_OR_CONTROL = re.compile(r"[\x00-\x20\x7f]")
_ENCODE_HINT = (
    "percent-encode its ssh user (/ as %2F, ? as %3F, # as %23, @ as %40, "
    "a space as %20)"
)


class RemoteDockerConfigError(ValueError, UserError):
    """The remote Docker host is not configured, or configured unsafely."""


@dataclass(frozen=True)
class RemoteDockerHost:
    """One remote daemon endpoint and the client settings that reach it."""

    url: str
    #: The user part of the URL as written (percent-encoded), or None.
    user: str | None = None
    cert_path: str | None = None
    #: ``BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS=1``: messages name the address
    #: (still without its ssh user) instead of :attr:`label`.
    show_address: bool = False

    @property
    def scheme(self) -> str:
        return self.url.split("://", 1)[0].lower()

    @property
    def label(self) -> str:
        """The opaque name every message, record and log line uses for the host."""
        return f"remote-docker[{self.scheme}]"

    @property
    def display(self) -> str:
        """How messages name the host: :attr:`label`, or the address without
        its ssh user when the operator asked to see it."""
        return _safe(self.url) if self.show_address else self.label

    def ssh_view(self) -> _SshView:
        """What ssh's configuration makes of this host, every ProxyJump hop
        included (``ssh -G``, which connects to nothing); empty for tcp://."""
        if self.scheme != "ssh":
            return _SshView((), (), True, False)
        return _ssh_view(self.url, _real_program("ssh"), _ssh_config_file())

    @property
    def _user_forms(self) -> tuple[str, ...]:
        """The user as written, percent-decoded (as ssh gets it from docker's
        ``-l``) and without a ``:password``, and every login ssh's own
        configuration picks for the host and each ProxyJump hop (``ssh -G``;
        a ``User`` line or the local user name when none is written);
        longest first."""
        users = [self.user] if self.user else []
        users += self.ssh_view().users
        forms: set[str] = set()
        for user in users:
            name = user.split(":", 1)[0]
            forms |= {f for f in (user, unquote(user), name, unquote(name)) if f}
        return tuple(sorted(forms, key=lambda form: -len(form)))

    @property
    def _host_forms(self) -> tuple[str, ...]:
        """The host as written and every name ssh's configuration resolves it
        and its ProxyJump hops to; longest first. Empty when the address is
        shown."""
        if self.show_address:
            return ()
        try:
            hosts = {urlsplit(self.url).hostname or ""}
        except ValueError:
            hosts = set()
        hosts |= set(self.ssh_view().hosts)
        hosts.discard("")
        return tuple(sorted(hosts, key=lambda form: -len(form)))

    @functools.cached_property
    def _tcp_authority(self) -> str:
        """The explicit host:port used by both Docker and its error matcher.

        Docker drops TCP userinfo and fills an omitted or empty port with
        2375, even with --tlsverify. Supply that authority ourselves so no
        downstream default can diverge from the redaction context. Keep
        IPv6 brackets and explicit port digits, including leading zeros.
        """
        parts = urlsplit(self.url)
        authority = parts.netloc.rsplit("@", 1)[-1]
        if not _TCP_AUTHORITY.fullmatch(authority):
            raise RemoteDockerConfigError(
                "the remote Docker TCP host must use an ASCII hostname (use its "
                "IDNA spelling), IPv4 or unscoped bracketed IPv6; "
                "zone identifiers and percent-encoded hostnames are not supported"
            )
        if parts.port is None:
            authority = authority.removesuffix(":") + ":2375"
        return authority

    @functools.cached_property
    def _daemon_transport_context(self) -> re.Pattern[str] | None:
        """Recognize the daemon context without parsing transport details.

        Ordinary requests name this daemon's HTTPS authority. Attached
        exec dials outside HTTP and uses a fixed daemon-connection wrapper
        with no URL. Neither the wrappers after that context nor their
        error details are safe to copy, even if we do not recognize them.
        """
        if self.scheme == "ssh":
            # Docker's SSH helper uses this dummy HTTP authority. Its
            # command envelope includes arbitrary, possibly multiline
            # helper stderr (%s), not just SSH's own host diagnostics.
            # Reserve the whole envelope; never parse its inner command,
            # remote context, proxy wrappers or certificate spellings.
            return re.compile(
                r'(?:"http://docker\.example\.com(?=[/:?#"])'
                r"|\bcommand \["
                r"|\bfailed to open the raw stream connection:"
                r"|\bcannot connect to the Docker daemon\. "
                r"Is 'docker daemon' running on this host\?: )"
                r"(?P<detail>[^\r\n]*)",
                re.IGNORECASE,
            )
        if self.scheme != "tcp":
            return None
        authority = re.escape(self._tcp_authority)
        return re.compile(
            rf'(?:"https://{authority}(?=[/?#"])'
            r"|\bcannot connect to the Docker daemon\. "
            r"Is 'docker daemon' running on this host\?: )"
            r"(?P<detail>[^\r\n]*)",
            re.IGNORECASE,
        )

    def _transport_messages(self, text: str, *, command_output: bool) -> str:
        """Map daemon diagnostics to fixed messages; unknown means opaque.

        Only the context decides whether mixed exec output belongs to the
        daemon. The whole diagnostic line, including its prefix and URL,
        is discarded. Recognized details select a fixed message, never a
        substring to retain. New proxy wrappers and error spellings fall
        back to the same generic message without needing another matcher.
        Outside mixed task output, even context-free TLS errors are opaque.
        """
        context = self._daemon_transport_context
        if context is None:
            return text
        fixed = {
            f"{self.label}: {message}"
            for message in (
                _TRANSPORT_CERTIFICATE,
                _TRANSPORT_UNKNOWN,
                _TRANSPORT_DISK_FULL,
                *_SSH_TRANSPORT_MESSAGES.values(),
            )
        }
        # Keep line endings exact, including standalone CR and EOF. Live
        # stderr supplies whole lines before retention or tail truncation.
        lines = re.split(r"(\r\n|\r|\n)", text)
        opaque = False
        for index in range(0, len(lines), 2):
            line = lines[index]
            if not line:
                continue
            if line in fixed:
                continue
            match = context.search(line)
            if opaque or (match is not None and self.scheme == "ssh"):
                # Helper stderr has no trustworthy end delimiter. Once
                # its envelope starts, all following lines are unknown.
                opaque = True
                message = _TRANSPORT_UNKNOWN
                if _DISK_FULL.search(line):
                    message = _TRANSPORT_DISK_FULL
                else:
                    for marker, known in _SSH_TRANSPORT_MESSAGES.items():
                        if marker in line.lower():
                            message = known
                            break
            elif match is not None:
                # An ordinary request's remainder starts with its URL
                # suffix and closing quote. Parse that only to select a
                # more useful fixed message. A malformed or future form
                # still reaches the generic message, never raw output.
                detail = re.sub(r'^(?:[/?#][^"]*)?": ', "", match["detail"], count=1)
                if detail.lower().startswith(_TLS_CERTIFICATE_PREFIX):
                    message = _TRANSPORT_CERTIFICATE
                elif _DISK_FULL.search(detail):
                    message = _TRANSPORT_DISK_FULL
                else:
                    message = _TRANSPORT_UNKNOWN
            elif not command_output and (
                _TLS_DIAGNOSTIC.search(line) or _TLS_NAME_MISMATCH.search(line)
            ):
                message = _TRANSPORT_UNKNOWN
            else:
                continue
            lines[index] = f"{self.label}: {message}"
        return "".join(lines)

    def stream_redactor(self) -> Callable[..., str]:
        """One stderr stream's redactor, retaining SSH envelope taint.

        Live readers deliver complete lines, but helper stderr can span
        lines. Keep this state per process, never on the shared host.
        """
        from benchflow.sandbox.process.docker import _STDERR_LINE_TOO_LONG

        opaque = False

        def redact(text: str, **kwargs: Any) -> str:
            nonlocal opaque
            fixed = {
                f"{self.label}: {message}"
                for message in (
                    _TRANSPORT_UNKNOWN,
                    _TRANSPORT_DISK_FULL,
                    *_SSH_TRANSPORT_MESSAGES.values(),
                )
            }
            if text.rstrip("\r\n") in fixed:
                return self.redact(text, **kwargs)
            if text and not self.show_address and kwargs.get("transport_errors", True):
                if self.scheme == "ssh" and text == _STDERR_LINE_TOO_LONG:
                    opaque = True
                if opaque:
                    ending = re.search(r"(?:\r\n|\r|\n)$", text)
                    text = f"{self.label}: {_TRANSPORT_UNKNOWN}" + (
                        ending[0] if ending else ""
                    )
                elif self.scheme == "ssh":
                    context = self._daemon_transport_context
                    assert context is not None
                    opaque = context.search(text) is not None
            return self.redact(text, **kwargs)

        return redact

    def redact(
        self, text: str, *, command_output: bool = False, transport_errors: bool = True
    ) -> str:
        """``text`` with the address shown as :attr:`display`, the ssh users
        replaced (see ``_TOKEN_USER_LEN``) and, unless the address is shown,
        the host names too, in any letter case. ``command_output`` is a
        command's own output, where a user is replaced only where it stands
        as one and a host only where it stands as an address (``@host``,
        ``//host``, ``host host``, ``host:port``). Elsewhere a dotted
        name or an address is replaced wherever it stands as a word, and a
        one-label name (``docker``, the docker:dind service) where it stands
        as an address or ssh or docker names a host with it (``hostname
        gpu``, ``lookup docker``, ``-- gpu``), so "Docker daemon" and
        BenchFlow's own wording stay as they are; ssh's ``login@host:``
        diagnostic form is masked whoever it names, and so are both operands
        of a TLS name mismatch (``_TLS_NAME_MISMATCH``) unless the address
        is shown. Daemon transport diagnostics map to fixed messages in
        Docker's request or attached-connection failure contexts; a
        successful exec sets ``transport_errors`` false, and unrelated
        task TLS diagnostics keep their names."""
        if not text:
            return text
        if not self.show_address:
            if transport_errors:
                text = self._transport_messages(text, command_output=command_output)
            if not command_output:
                text = _TLS_NAME_MISMATCH.sub(_tls_mismatch_mask, text)
        display = self.display
        text = re.sub(
            re.escape(self.url), lambda _m: display, text, flags=re.IGNORECASE
        )
        for form in self._user_forms:
            if not command_output and len(form) >= _TOKEN_USER_LEN:
                text = text.replace(form, "***")
                continue
            user = re.escape(form)
            text = re.sub(rf"(?<![\w.%-]){user}(?=@)", "***", text)
            text = re.sub(rf"(-l\s*){user}(?![\w.%-])", r"\1***", text)
            text = re.sub(rf"(['\"]){user}(?=\1)", r"\1***", text)
        for form in self._host_forms:
            host = re.escape(form)
            if command_output:
                pattern = (
                    rf"(?:(?<=@)|(?<=//)|(?<=host )|(?<=\[)){host}(?![\w-])"
                    rf"|(?<![\w.-]){host}(?=\]?:\d)"
                )
            elif re.search(r"[.:]", form):
                pattern = rf"(?<![\w.-]){host}(?![\w-])"
            else:
                pattern = (
                    rf"(?:{_HOST_CONTEXT}){host}(?![\w-]|\.\w)"
                    rf"|(?<![\w.-]){host}(?=\]?:\d)"
                )
            text = re.sub(pattern, _HOST_MASK, text, flags=re.IGNORECASE)
        if not command_output:
            # Whoever it names: a login the configuration did not show (an
            # unresolved hop) is still never printed; the host only when
            # the operator asked to see addresses.
            host = r"\2" if self.show_address else _HOST_MASK
            text = _SSH_LOGIN_AT.sub(rf"***@{host}", text)
        return text

    def apply(self, env: dict[str, str]) -> dict[str, str]:
        """Point a docker CLI environment at this host (in place; returned).

        For raw ``docker`` calls (never Compose's: :meth:`compose_args`).
        ``docker`` resolves to a provider-owned shim first on ``PATH`` that
        passes the address as ``-H`` (and the TLS files as flags);
        ``ssh`` resolves to one that holds every hop to
        ``StrictHostKeyChecking=yes``. The caller's ``DOCKER_HOST``,
        ``DOCKER_CONTEXT`` and TLS variables are removed, and builds go to the
        ``default`` builder, which is the docker driver of the endpoint the
        CLI talks to.
        """
        for name in _CONNECTION_ENV:
            env.pop(name, None)
        env["BUILDX_BUILDER"] = _BUILDER
        env["PATH"] = _on_path(_transport_dirs(self, env).client, env)
        return env

    def client_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """``base`` (the caller's environment by default), pointed at this host."""
        return self.apply(dict(os.environ if base is None else base))

    def docker_args(self) -> list[str]:
        """The global ``docker`` flags that reach this host."""
        url = self.url
        if self.scheme == "tcp":
            url = urlsplit(url)._replace(netloc=self._tcp_authority).geturl()
        args = ["-H", url]
        if self.cert_path is not None:
            cert = Path(self.cert_path)
            args += [
                "--tlsverify",
                "--tlscacert",
                str(cert / "ca.pem"),
                "--tlscert",
                str(cert / "cert.pem"),
                "--tlskey",
                str(cert / "key.pem"),
            ]
        return args

    def compose_args(self, environ: Mapping[str, str] | None = None) -> list[str]:
        """The docker CLI's global flags that point ``docker compose`` at this
        host (``docker <flags> compose ...``): the one place Compose gets the
        address from. The URL and its TLS files go on the command line,
        never into the environment Compose fills a task's ``${VAR}`` from;
        the operator's ``DOCKER_CONFIG`` (their registry logins) comes along
        as ``--config``. The launch gate's ``compose config`` uses the same."""
        source = os.environ if environ is None else environ
        config = source.get("DOCKER_CONFIG")
        return [*(["--config", config] if config else []), *self.docker_args()]

    def compose_transport_env(
        self, environ: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """What ``docker compose``'s own process needs beyond its hermetic
        environment to reach this host: buildx's ``default`` builder and,
        for ``ssh://``, a ``PATH`` whose ``ssh`` is the provider's (every
        hop held to ``StrictHostKeyChecking=yes``; no ``docker`` shim there,
        so the address is given once). The operator's SSH agent goes to that
        ssh alone, never into Compose's environment: Compose fills a task's
        ``${...}`` from it, and BuildKit would forward an agent it finds there
        to a build that asks (``build.ssh``). Never given to a pre-compose
        hook. The launch gate's ``compose config`` uses the same."""
        source = os.environ if environ is None else environ
        env = {"BUILDX_BUILDER": _BUILDER}
        if self.scheme == "ssh":
            env["PATH"] = _on_path(_transport_dirs(self, source).ssh, source)
        return env

    def compose_env(self, env: dict[str, str]) -> dict[str, str]:
        """Compose's own environment, the one it interpolates a task's files
        from and the pre-compose hook gets (in place; returned): no
        connection setting, registry configuration, SSH agent or login name,
        and no shim on ``PATH``; buildx's ``default`` builder."""
        for name in (*_CONNECTION_ENV, *_COMPOSE_ONLY_ON_THE_COMMAND_LINE):
            env.pop(name, None)
        env["BUILDX_BUILDER"] = _BUILDER
        env["PATH"] = _on_path(None, env)
        return env


# Removed from every docker and compose call's environment; the shim (raw
# docker calls) or compose_args (Compose) carries the address instead.
_CONNECTION_ENV = (
    REMOTE_DOCKER_HOST_ENV,
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_TLS",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
# Also kept out of Compose's environment (RemoteDockerHost.compose_env): the
# registry configuration goes on its command line, the agent only to the
# provider's ssh (_transport_dirs), and the login names identify the operator.
_COMPOSE_ONLY_ON_THE_COMMAND_LINE = (
    "DOCKER_CONFIG",
    "SSH_AUTH_SOCK",
    "USER",
    "LOGNAME",
)
# buildx's name for the docker driver of the CLI's current endpoint; it is
# reserved ("default is a reserved name"), so no other builder can take it.
_BUILDER = "default"
# OpenSSH takes the first value it gets for an option, and the command line
# comes before every configuration file, so this holds whatever the
# operator's ~/.ssh/config says; their keys, agent and other settings stay.
_SSH_POLICY = ("-o", "StrictHostKeyChecking=yes")
# The ssh configuration the provider's ssh reads instead of ~/.ssh/config.
REMOTE_DOCKER_SSH_CONFIG_ENV = "BENCHFLOW_REMOTE_DOCKER_SSH_CONFIG"
_HOST_MASK = "<remote host>"
# Where ssh's and docker's messages name a host by a one-label name: after
# ``user@``, ``//``, ``[``, "host " ("connect to host gpu port 22"),
# "hostname " ("Could not resolve hostname gpu"), "lookup " (Go's resolver),
# "-- " (docker's ssh command line) and "key for "/"known for " (ssh's host
# key messages).
_HOST_CONTEXT = "|".join(
    rf"(?<={re.escape(before)})"
    for before in (
        "@",
        "//",
        "[",
        "host ",
        "hostname ",
        "lookup ",
        "-- ",
        "key for ",
        "known for ",
    )
)
# Go's TLS name-mismatch forms (crypto/x509 HostnameError): "certificate
# is valid for <names>, not <host>" and "certificate is not valid for any
# names, but wanted to match <host>". The host is the one dialed, in any
# spelling (a one-label name, an IP), and the names are the certificate's,
# often other internal hosts: both operands are masked. An IP dialed against
# a certificate with no IP SANs is named by "cannot validate certificate for".
_TLS_NAME_MISMATCH = re.compile(
    r"(certificate is valid for )[^\r\n]*?(, not )(?:<remote host>|[^\s'\x22,;)]+)"
    r"|(but wanted to match )(?:<remote host>|[^\s'\x22,;)]+)"
    r"|(cannot validate certificate for )(?:<remote host>|[^\s'\x22,;)]+)"
)
# ssh's "<login>@<host>: Permission denied" form, whichever hop it names.
_SSH_LOGIN_AT = re.compile(
    r"(?<![\w.%+-])([\w.%+-]+)@((?:\[[^\]\s]+\]|[\w.-]+)(?::\d+)?)(?=:\s)"
)
# Lines of ssh's own output that say why the login failed.
_SSH_VERDICTS = (
    "host key verification failed",
    "permission denied",
    "connection refused",
    "could not resolve hostname",
    "no route to host",
    "connection timed out",
    "operation timed out",
)
_MAX_SSH_HOPS = 8
_SHIM_DIRS: dict[tuple[str, ...], _ShimDirs] = {}


@dataclass(frozen=True)
class _SshView:
    """What ``ssh -G`` says about a host and its ProxyJump hops."""

    users: tuple[str, ...]
    hosts: tuple[str, ...]
    #: Every hop was resolved, so every login and host ssh can print is known.
    complete: bool
    #: Some hop connects through a ProxyCommand, whose own connections no
    #: ssh option of BenchFlow's reaches.
    proxy_command: bool


@dataclass(frozen=True)
class _ShimDirs:
    #: ``docker`` and ``ssh`` shims, for raw docker calls.
    client: str
    #: The ``ssh`` shim alone, for Compose (which gets the address as flags).
    ssh: str

    def __iter__(self) -> Iterator[str]:
        return iter((self.client, self.ssh))


def _shim_dir_set() -> set[str]:
    return {d for dirs in _SHIM_DIRS.values() for d in dirs}


def _on_path(folder: str | None, env: Mapping[str, str]) -> str:
    """``env``'s PATH without any shim folder, with ``folder`` first."""
    shims = _shim_dir_set()
    entries = [
        p for p in env.get("PATH", os.defpath).split(os.pathsep) if p and p not in shims
    ]
    return os.pathsep.join([folder, *entries] if folder else entries)


def _real_program(name: str, path: str | None = None) -> str | None:
    """``name`` on ``path`` (this process's by default), never a shim."""
    search = os.environ.get("PATH", os.defpath) if path is None else path
    shims = _shim_dir_set()
    dirs = [p for p in search.split(os.pathsep) if p and p not in shims]
    return shutil.which(name, path=os.pathsep.join(dirs))


def _ssh_config_file() -> str:
    """The ssh configuration the provider's ssh reads (``ssh -F``): see
    :func:`_ssh_config_for`."""
    return _ssh_config_for(
        os.environ.get(REMOTE_DOCKER_SSH_CONFIG_ENV) or "~/.ssh/config"
    )


@functools.lru_cache(maxsize=8)
def _ssh_config_for(operator: str) -> str:
    """The ssh configuration the provider's ssh reads (``ssh -F``).

    ``StrictHostKeyChecking yes`` comes first, at the top level, so it holds
    for every host whatever the files it then includes say (the first value
    of an option wins); ssh hands ``-F`` on to the ssh it starts for each
    ProxyJump hop, so the hops read it too. Then the operator's own file
    (``~/.ssh/config``, or ``BENCHFLOW_REMOTE_DOCKER_SSH_CONFIG``) and the
    system's, as plain ssh would read them, keys and agent included.
    """
    fd, path = tempfile.mkstemp(prefix="benchflow-remote-docker-ssh-", text=True)
    atexit.register(_remove_quietly, path)
    with os.fdopen(fd, "w") as fh:
        fh.write(
            "# BenchFlow --sandbox remote-docker (sandbox/remote_docker.py):\n"
            "# every ssh hop, ProxyJump hops included, checks host keys.\n"
            "StrictHostKeyChecking yes\n"
            f"Include {_ssh_quote(operator)}\n"
            "Include /etc/ssh/ssh_config\n"
        )
    return path


def _tls_mismatch_mask(match: re.Match[str]) -> str:
    if match.group(1):
        return f"{match.group(1)}<certificate names>{match.group(2)}{_HOST_MASK}"
    return f"{match.group(3) or match.group(4)}{_HOST_MASK}"


def _ssh_quote(value: str) -> str:
    return '"' + value.replace('"', "") + '"' if re.search(r"\s", value) else value


def _remove_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _ssh_settings(
    ssh: str, config: str, user: str | None, host: str, port: int | None
) -> dict[str, str] | None:
    """``ssh -G`` for one hop: its resolved settings, or None if ssh cannot say."""
    args = [ssh, "-G", "-F", config, *_SSH_POLICY]
    if user:
        args += ["-l", user]
    if port:
        args += ["-p", str(port)]
    args += ["--", host]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    found: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        key, _, value = line.strip().partition(" ")
        if value.strip():
            found.setdefault(key.lower(), value.strip())
    return found


def _jump_hop(spec: str) -> tuple[str | None, str, int | None] | None:
    """One ProxyJump entry (``[user@]host[:port]`` or ``ssh://...``)."""
    spec = spec.strip()
    try:
        parts = urlsplit(spec if spec.startswith("ssh://") else f"ssh://{spec}")
        hostname, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not hostname:
        return None
    return (unquote(parts.username) if parts.username else None, hostname, port)


@functools.lru_cache(maxsize=32)
def _ssh_view(url: str, ssh: str | None, config: str) -> _SshView:
    """The logins and host names ssh would use for ``url`` and each of its
    ProxyJump hops, recursively (``ssh -G`` per hop, connecting to nothing).

    Incomplete when ssh is missing, cannot read a hop, or the chain is
    longer than ``_MAX_SSH_HOPS``.
    """
    # ``url`` can be the user's own DOCKER_HOST, unvalidated (bench doctor's
    # docker line): a bad port or bracket is docker's to report, not a crash.
    first = _jump_hop(url) if url.lower().startswith("ssh://") else None
    if ssh is None or first is None:
        return _SshView((), (), False, False)
    users: list[str] = []
    hosts: list[str] = []
    complete, proxy_command = True, False
    pending = [first]
    seen: set[tuple[str | None, str, int | None]] = set()
    while pending:
        hop = pending.pop(0)
        if hop in seen:
            continue
        if len(seen) >= _MAX_SSH_HOPS:
            complete = False
            break
        seen.add(hop)
        user, host, port = hop
        hosts.append(host)
        if user:
            users.append(user)
        settings = _ssh_settings(ssh, config, user, host, port)
        if settings is None:
            complete = False
            continue
        if settings.get("user"):
            users.append(settings["user"])
        if settings.get("hostname"):
            hosts.append(settings["hostname"])
        if settings.get("proxycommand", "none").lower() != "none":
            proxy_command = True
        jump = settings.get("proxyjump", "none")
        if jump.lower() != "none":
            for spec in jump.split(","):
                parsed = _jump_hop(spec)
                if parsed is None:
                    complete = False
                else:
                    pending.append(parsed)
    return _SshView(
        tuple(dict.fromkeys(users)),
        tuple(dict.fromkeys(hosts)),
        complete,
        proxy_command,
    )


def _transport_dirs(host: RemoteDockerHost, env: Mapping[str, str]) -> _ShimDirs:
    """The provider's private shim folders for ``host`` (mode 0700, removed
    when the process exits): ``client`` holds ``docker`` (the address as
    flags) and ``ssh``; ``ssh`` holds ``ssh`` alone, for Compose, which gets
    the address on its own command line. A program missing on ``env``'s
    PATH gets no shim, so running it fails as it would without one.
    """
    if os.name == "nt":
        raise RemoteDockerConfigError(
            "remote-docker runs on Linux and macOS: it passes the host to "
            "docker through a shell shim, never through DOCKER_HOST"
        )
    path = env.get("PATH", os.defpath)
    docker = _real_program("docker", path)
    ssh = _real_program("ssh", path) if host.scheme == "ssh" else None
    config = _ssh_config_file() if ssh else ""
    # The provider's ssh is handed the operator's agent itself, so Compose's
    # environment never holds it (compose_transport_env).
    agent = (env.get("SSH_AUTH_SOCK") or "") if ssh else ""
    key = (host.url, host.cert_path or "", docker or "", ssh or "", config, agent)
    dirs = _SHIM_DIRS.get(key)
    if dirs is not None and all(Path(d).is_dir() for d in dirs):
        return dirs
    root = Path(tempfile.mkdtemp(prefix="benchflow-remote-docker-"))
    atexit.register(shutil.rmtree, str(root), True)
    dirs = _ShimDirs(client=str(root / "client"), ssh=str(root / "ssh"))
    for folder in dirs:
        Path(folder).mkdir(mode=0o700)
    if docker is not None:
        flags = " ".join(shlex.quote(a) for a in host.docker_args())
        _write_shim(Path(dirs.client) / "docker", f"{shlex.quote(docker)} {flags}")
    if ssh is not None:
        policy = " ".join([shlex.quote("-F"), shlex.quote(config), *_SSH_POLICY])
        setup = (
            f"SSH_AUTH_SOCK={shlex.quote(agent)}; export SSH_AUTH_SOCK\n"
            if agent
            else ""
        )
        for folder in dirs:
            _write_shim(Path(folder) / "ssh", f"{shlex.quote(ssh)} {policy}", setup)
    _SHIM_DIRS[key] = dirs
    return dirs


def _write_shim(path: Path, command: str, setup: str = "") -> None:
    path.write_text(
        "#!/bin/sh\n"
        "# BenchFlow --sandbox remote-docker (sandbox/remote_docker.py).\n"
        f"{setup}"
        f'exec {command} "$@"\n'
    )
    path.chmod(0o700)


def resolve_remote_docker_host(
    environ: Mapping[str, str] | None = None, *, home: Path | None = None
) -> RemoteDockerHost:
    """Read and validate the host from ``BENCHFLOW_REMOTE_DOCKER_HOST`` or ``DOCKER_HOST``.

    No message quotes the user part of the address.
    """
    env = os.environ if environ is None else environ
    source = (
        REMOTE_DOCKER_HOST_ENV if env.get(REMOTE_DOCKER_HOST_ENV) else "DOCKER_HOST"
    )
    url = (env.get(REMOTE_DOCKER_HOST_ENV) or env.get("DOCKER_HOST") or "").strip()
    if not url:
        raise RemoteDockerConfigError(
            f"remote-docker needs {REMOTE_DOCKER_HOST_ENV} or DOCKER_HOST: "
            "ssh://user@host[:port] or tcp://host:2376 with DOCKER_TLS_VERIFY=1"
        )
    if _SPACE_OR_CONTROL.search(url):
        raise RemoteDockerConfigError(
            f"the Docker host in {source} holds a space, tab, line break or other "
            f"control character, which docker would quote with the address in its "
            f"error: remove it, or {_ENCODE_HINT}"
        )
    try:
        parts = urlsplit(url)
    except ValueError:
        # "Invalid IPv6 URL" (a missing ]), a name in [ ], or a netloc that
        # NFKC normalisation changes; the message can quote the netloc, user
        # included, so it is not passed on.
        raise RemoteDockerConfigError(
            f"the Docker host in {source} is not a URL docker can read (an IPv6 "
            "address goes in [ ], a name never does): use ssh://user@host[:port] "
            "or tcp://host:port with TLS"
        ) from None
    scheme = parts.scheme.lower()
    show_address = env.get(REMOTE_DOCKER_SHOW_ADDRESS_ENV, "").strip() == "1"
    # The address the operator wrote is named only when they asked for it.
    shown = repr(_safe(url)) if show_address else f"in {source}"
    if scheme in ("unix", "npipe", "fd"):
        raise RemoteDockerConfigError(
            f"the Docker host {shown} is a local socket; use --sandbox docker for a local "
            "daemon, or set an ssh:// or tcp:// host for remote-docker"
        )
    if scheme not in ("ssh", "tcp") or not _SCHEME.match(url):
        raise RemoteDockerConfigError(
            f"unsupported remote Docker host {shown}: use "
            "ssh://user@host[:port] or tcp://host:port with TLS"
        )
    if "@" in url and "@" not in parts.netloc:
        # Go's url.Parse ends the user part at the first /, ? or #, so docker
        # would read the rest as a path or query and quote it in its error.
        raise RemoteDockerConfigError(
            f"the {scheme}:// host in {source} has an @ after a /, ? or #: "
            f"{_ENCODE_HINT}"
        )
    user = parts.netloc.rpartition("@")[0] or None
    if user is not None and (not _USERINFO_OK.match(user) or _BAD_ESCAPE.search(user)):
        raise RemoteDockerConfigError(
            f"the ssh user in {source} has a character docker cannot read in a "
            f"URL: {_ENCODE_HINT}"
        )
    try:
        port_ok = bool(parts.hostname) and (parts.port is None or parts.port > 0)
    except ValueError:
        port_ok = False
    if not port_ok:
        raise RemoteDockerConfigError(
            f"unsupported remote Docker host {shown}: use "
            "ssh://user@host[:port] or tcp://host:port with TLS"
        )
    if scheme == "ssh":
        if parts.password is not None:
            raise RemoteDockerConfigError(
                f"the ssh:// host in {source} holds a password, which docker "
                "does not accept: log in with an ssh key or agent instead"
            )
        if parts.path or parts.query or parts.fragment:
            raise RemoteDockerConfigError(
                f"the ssh:// host {shown} takes no path, query or fragment"
            )
        return RemoteDockerHost(url=url, user=user, show_address=show_address)
    verify = env.get("DOCKER_TLS_VERIFY", "").strip()
    if not verify or verify == "0":
        raise RemoteDockerConfigError(
            f"the tcp:// host {shown} needs TLS: set DOCKER_TLS_VERIFY=1 and DOCKER_CERT_PATH "
            "(ca.pem, cert.pem, key.pem); a plain tcp daemon gives root on the host "
            "to anyone who can reach the port, so it is refused"
        )
    cert_path = env.get("DOCKER_CERT_PATH") or str((home or Path.home()) / ".docker")
    missing = [n for n in _TLS_FILES if not (Path(cert_path) / n).is_file()]
    if missing:
        raise RemoteDockerConfigError(
            f"the tcp:// host {shown} with TLS: {', '.join(missing)} not found in {cert_path} "
            "(DOCKER_CERT_PATH)"
        )
    host = RemoteDockerHost(
        url=url, user=user, cert_path=cert_path, show_address=show_address
    )
    # Validate the same authority every launch and TLS diagnostic consumes,
    # before a configured host can reach any subprocess or retained output.
    _ = host._tcp_authority
    return host


def _safe(value: str) -> str:
    """``value`` with everything up to its last ``@`` masked.

    Whatever a docker host address holds there can be an ssh user, which is
    often a token; this holds for values that are not URLs at all
    (``user@host``, ``ssh:user@host``) and for a user written with an
    unencoded ``/``.
    """
    if "@" not in value:
        return value
    match = _SCHEME.match(value)
    prefix = match.group(0) if match else ""
    return f"{prefix}***@{value.rsplit('@', 1)[1]}"


def redact_docker_host(text: str, url: str) -> str:
    """``text`` without the user part of any docker host value ``url``.

    For ``DOCKER_HOST`` as the local provider and ``bench doctor`` see it,
    unvalidated: everything between the scheme and the last ``@`` counts as
    the user. When docker cannot read that user in a URL (an unencoded ``/``,
    ``?`` or ``#``, a space or control character, a bad ``%`` escape), docker
    splits it or quotes it escaped, so no form of it can be found in its
    text: the text is replaced whole by a message saying how to fix the
    address.
    """
    if not text or "@" not in url:
        return text
    match = _SCHEME.match(url)
    rest = url[match.end() :] if match else url
    user = rest.rpartition("@")[0]
    scheme = match.group(0)[:-3].lower() if match else ""
    if (
        user
        and scheme not in ("unix", "npipe", "fd")
        and (not _USERINFO_OK.match(user) or _BAD_ESCAPE.search(user))
    ):
        return (
            "the ssh user in DOCKER_HOST has a character docker cannot read in a "
            f"URL (docker's own message would show part of it): {_ENCODE_HINT}"
        )
    # The local provider's own daemon: its address stays in the message.
    return RemoteDockerHost(url=url, user=user or None, show_address=True).redact(text)


@dataclass(frozen=True)
class HostCapacity:
    """What ``docker info`` reports for the remote host.

    On a daemon inside a VM or container, these are the numbers the daemon
    sees, which can be the physical machine's rather than a quota's, so the
    capacity check is a lower bound.
    """

    cpus: int
    memory_mb: int
    server_version: str
    #: The daemon's architecture as ``docker info`` names it (bfx's pin).
    architecture: str = ""


def unreachable_reason(text: str) -> str:
    """The line of docker/ssh output that says why the host could not be reached."""
    marker = text.find("stderr=")
    if marker >= 0:
        # ssh's verdict, not the first line of a banner: a changed host key
        # starts with lines of @ and ends with "Host key verification failed."
        reason = [
            ln.strip()
            for ln in text[marker + len("stderr=") :].splitlines()
            if ln.strip().strip("@ ")
        ]
        line = next(
            (ln for ln in reason if any(key in ln.lower() for key in _SSH_VERDICTS)),
            reason[0] if reason else "",
        )
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        line = next(
            (
                ln
                for ln in lines
                if any(message in ln for message in _SSH_TRANSPORT_MESSAGES.values())
            ),
            "",
        )
        line = line or next(
            (
                ln
                for ln in lines
                if any(
                    key in ln.lower()
                    for key in (
                        "cannot connect",
                        "error during connect",
                        "tls",
                        "unable to resolve docker endpoint",
                        "timed out",
                    )
                )
            ),
            lines[-1] if lines else "no output",
        )
    if len(line) > _REASON_LIMIT:
        line = line[: _REASON_LIMIT - 3] + "..."
    return line


def probe_remote_docker(
    host: RemoteDockerHost,
    *,
    timeout_sec: float = _PROBE_TIMEOUT_SEC,
    environ: Mapping[str, str] | None = None,
) -> HostCapacity:
    """Run ``docker info`` on ``host``; raise :class:`SandboxStartupError` if it cannot.

    ``environ`` is the environment the docker CLI starts from (this
    process's by default); the host's settings are applied on top.

    An ssh host whose configuration connects some hop through a
    ProxyCommand is refused before any call: no option of BenchFlow's
    reaches that command's own connections, so host keys could go
    unchecked there (ProxyJump hops are held to the policy). So is one
    whose every hop BenchFlow could not read (``ssh -G`` failed for one, an
    entry it cannot parse, more than ``_MAX_SSH_HOPS``, no ssh): an unread
    hop can hide a ProxyCommand, and its login and host name would be
    unknown to every later message's redaction. Every docker and Compose
    call of a sandbox comes after this check.
    """
    view = host.ssh_view()
    if host.scheme == "ssh" and not view.complete:
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: BenchFlow could "
            "not read its ssh configuration for every hop (`ssh -G` failed for "
            f"one, ssh is missing, or the chain has more than {_MAX_SSH_HOPS} "
            "hops), so it cannot hold each to StrictHostKeyChecking or keep its "
            "login out of messages; fix the configuration or shorten the chain"
        )
    if view.proxy_command:
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: its ssh "
            "configuration reaches it through a ProxyCommand, whose connection "
            "BenchFlow cannot hold to StrictHostKeyChecking; use ProxyJump instead"
        )
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=host.client_env(environ),
        )
    except FileNotFoundError as exc:
        raise SandboxStartupError(
            "Remote Docker host unreachable: docker CLI not found on PATH "
            "(remote-docker runs the local docker CLI against the remote daemon)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: no answer to "
            f"`docker info` within {timeout_sec:g}s"
        ) from exc
    output = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    info: dict[str, Any] = {}
    if proc.returncode == 0:
        try:
            info = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            info = {}
    if proc.returncode != 0 or not info.get("ServerVersion"):
        # Redacted before unreachable_reason cuts the line, so a user cut in
        # half cannot leave its first part behind.
        reason = unreachable_reason(host.redact(output))
        raise SandboxStartupError(
            f"Remote Docker host unreachable: {host.display}: {reason}"
        )
    return HostCapacity(
        cpus=int(info.get("NCPU") or 0),
        memory_mb=int(info.get("MemTotal") or 0) // (1024 * 1024),
        server_version=str(info.get("ServerVersion")),
        architecture=str(info.get("Architecture") or ""),
    )


def check_capacity(
    capacity: HostCapacity, *, cpus: int, memory_mb: int, host_display: str
) -> None:
    """Refuse a task that asks for more CPUs or memory than the host has."""
    problems = []
    if capacity.cpus and cpus > capacity.cpus:
        problems.append(f"task needs {cpus} CPUs; {host_display} has {capacity.cpus}")
    if capacity.memory_mb and memory_mb > capacity.memory_mb:
        problems.append(
            f"task needs {memory_mb} MB of memory; {host_display} has "
            f"{capacity.memory_mb} MB"
        )
    if problems:
        raise SandboxStartupError(
            "Remote Docker host lacks resources: " + "; ".join(problems)
        )


# Starts every refusal of a task's Compose project; a retry would only repeat
# it (_utils/scoring.py's unrecoverable startup errors).
_REFUSED = "Remote Docker host refuses this task: "
# The operator's opt-in for a task that reaches outside its folder on this
# machine, a pre-compose hook among them: the launch gate's switch
# (task/compose_env.py's ALLOW_HOST_ACCESS_VAR on py8/task-env), one name so
# one setting governs both.
_ALLOW_HOST_ACCESS_VAR = "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _host_access_allowed() -> bool:
    return os.environ.get(_ALLOW_HOST_ACCESS_VAR, "").strip().lower() in _TRUTHY


def _is_bfx_volume(
    project: Mapping[str, Any],
    key: str,
    spec: Mapping[str, Any],
    bfx_volumes: Collection[str],
) -> bool:
    """BenchFlow's own bfx volume (``benchflow._bfx.delivery``): exactly the
    volume this start delivered (*bfx_volumes*), mounted only read-only at
    /.bfx without a copy from the image. Another volume of that name form
    (an older pin's, or one a task names) is external like any other."""
    from benchflow._bfx.locate import SANDBOX_DIR

    name = spec.get("name") or key
    if name != key or key not in bfx_volumes:
        return False
    for service in (project.get("services") or {}).values():
        for mount in (service or {}).get("volumes") or []:
            if (
                isinstance(mount, dict)
                and mount.get("source") == key
                and not (
                    mount.get("type") == "volume"
                    and mount.get("target") == SANDBOX_DIR
                    and mount.get("read_only") is True
                    and (mount.get("volume") or {}).get("nocopy") is True
                )
            ):
                return False
    return True


def _host_reach(
    project: Mapping[str, Any], bfx_volumes: Collection[str] = ()
) -> list[str]:
    """What in a resolved Compose project names a path on the Docker host, or
    a volume there that the trial does not make (``external``), or forwards
    this machine's SSH agent to a build (``build.ssh``). Service
    settings that hand a container the host (``privileged``, ``devices``,
    host namespaces, added capabilities, ``volumes_from``) are not looked at
    here."""
    problems = []
    for service, spec in (project.get("services") or {}).items():
        build = (spec or {}).get("build")
        if isinstance(build, dict) and build.get("ssh"):
            # BuildKit would forward this machine's SSH agent (or a key file
            # here) to the build on the remote host, which could then log in
            # wherever those keys can. Compose 2.40 cannot print a keyed
            # entry (``id=path``) as JSON at all, so that form fails the
            # ``config`` call before anything is built.
            problems.append(
                f"service {service!r} builds with `ssh:`, which forwards your "
                "SSH agent or keys to the build on the remote host; fetch what "
                "the build needs another way"
            )
        for mount in (spec or {}).get("volumes") or []:
            if isinstance(mount, dict) and mount.get("type") in ("bind", "npipe"):
                problems.append(
                    f"service {service!r} bind-mounts {mount.get('source')!r}, a "
                    "path on the remote host (remote-docker copies no task folder "
                    "there; COPY files in the Dockerfile instead)"
                )
    for kind in ("secrets", "configs"):
        for key, spec in (project.get(kind) or {}).items():
            if isinstance(spec, dict) and spec.get("file"):
                problems.append(
                    f"{kind[:-1]} {key!r} comes from a file, which Compose "
                    "bind-mounts from the remote host"
                )
    for key, spec in (project.get("volumes") or {}).items():
        spec = spec or {}
        if spec.get("external") and _is_bfx_volume(project, key, spec, bfx_volumes):
            continue
        if spec.get("external"):
            problems.append(
                f"volume {key!r} is external: it names "
                f"{spec.get('name') or key!r}, a volume on the host that the "
                "trial did not make (another workload's data); drop `external:` "
                "so Compose makes one for the trial"
            )
            continue
        opts = spec.get("driver_opts") or {}
        bind = "bind" in str(opts.get("o") or "").split(",")
        if bind or opts.get("device"):
            problems.append(
                f"volume {key!r} binds {opts.get('device')!r}, a path on the "
                "remote host"
            )
    return problems


def _with_message(exc: RuntimeError, message: str) -> RuntimeError:
    """``exc``'s own class (``SandboxRestoreHostConfigUnavailable``, a
    startup error) carrying ``message``, so callers that sort errors by
    class still can; a plain RuntimeError when the class cannot be rebuilt
    from one message."""
    try:
        redacted = copy.copy(exc)
        redacted.args = (message,)
    except Exception:
        return RuntimeError(message)
    return redacted if str(redacted) == message else RuntimeError(message)


def _owned_volume_names(project: Mapping[str, Any], project_name: str) -> list[str]:
    """The names of the project's volumes that are not ``external``: the ones
    Compose creates (and teardown removes) as this trial's."""
    names = []
    for key, spec in (project.get("volumes") or {}).items():
        spec = spec or {}
        if not spec.get("external"):
            names.append(str(spec.get("name") or f"{project_name}_{key}"))
    return sorted(set(names))


class RemoteDockerSandbox(DockerSandbox):
    """:class:`DockerSandbox` on a remote daemon. See the module docstring."""

    _DOCKER_COMPOSE_BASE_PATH = COMPOSE_REMOTE_BASE_PATH
    # Its bind sources are paths on the remote host, never this controller's.
    _HOST_MOUNTS_ARE_LOCAL = False

    def __init__(
        self,
        *args: Any,
        docker_host: RemoteDockerHost | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("mounts_json"):
            raise RemoteDockerConfigError(
                "remote-docker cannot use host mounts: the paths are on this "
                "machine, not on the remote Docker host"
            )
        self.docker_host = docker_host or resolve_remote_docker_host()
        super().__init__(*args, **kwargs)
        self._logs_are_mounted = False
        self._host_checked = False

    @property
    def _project_name(self) -> str:
        return self.compose_project_name

    # --- where docker calls go ------------------------------------------------

    def _docker_client_env(self) -> dict[str, str] | None:
        return self.docker_host.client_env()

    #: Every bind is a path on the remote host.
    _remote_daemon = True

    def _docker_layers(self) -> DockerLayers:
        return DockerLayers(daemon=self._docker_subnet_key())

    def _docker_daemon_display(self) -> str:
        return self.docker_host.display

    def _docker_subnet_key(self) -> DockerDaemon:
        if not hasattr(self, "_daemon_connection"):
            self._daemon_connection = DockerDaemon(
                self._docker_client_env() or {},
                self.docker_host.url,
                self.docker_host.redact,
            )
        return self._daemon_connection

    def _docker_compose_env(self) -> dict[str, str]:
        # What Compose interpolates a task's files from and the pre-compose
        # hook gets: no connection setting and no shim (compose_env).
        return self.docker_host.compose_env(super()._docker_compose_env())

    def _docker_compose_global_args(self) -> list[str]:
        # The one place Compose gets the address from (the shim on the raw
        # docker calls' PATH never reaches Compose's process).
        from benchflow.task.compose_env import hermetic_docker_config

        return self.docker_host.compose_args(
            {**os.environ, "DOCKER_CONFIG": hermetic_docker_config()}
        )

    def _compose_transport_env(self) -> dict[str, str]:
        return self.docker_host.compose_transport_env()

    def _compose_client(self) -> tuple[list[str], dict[str, str]]:
        """The docker CLI flags and the environment of every Compose call
        (``docker <flags> compose``): the ordinary ones, attach and the live
        process (``DockerProcess.from_sandbox_env``). Never the raw docker
        calls' environment, which keeps the operator's SSH agent, registry
        configuration and login names for the shim, and from which Compose
        would fill a task's ``${VAR}`` differently."""
        return self._docker_compose_global_args(), {
            **self._docker_compose_env(),
            **self._compose_transport_env(),
        }

    def _compose_base(self) -> list[str]:
        """``docker <flags> compose`` with this project's name, folder and
        files, as _run_docker_compose_command gives them. The folder is the
        frozen launch's (``compose_project_dir``, py8/task-env) where the
        sandbox has one, else the environment folder."""
        global_args, _ = self._compose_client()
        frozen = getattr(type(self), "compose_project_dir", None)
        folder = (
            frozen.__get__(self)
            if isinstance(frozen, property)
            else self.environment_dir.resolve().absolute()
        )
        command = [
            "docker",
            *global_args,
            "compose",
            "--project-name",
            self.compose_project_name,
            "--project-directory",
            str(folder),
        ]
        for path in self._docker_compose_paths:
            command.extend(["-f", str(path.resolve().absolute())])
        return command

    def _remote_error(
        self, exc: Exception, *, command_output: bool = False
    ) -> RuntimeError:
        message = self.docker_host.redact(str(exc), command_output=command_output)
        if _DISK_FULL.search(message):
            message = f"remote Docker host is out of disk space ({self.docker_host.display}): {message}"
        if not isinstance(exc, RuntimeError):
            return RuntimeError(message)
        if message == str(exc):
            return exc
        return _with_message(exc, message)

    @contextlib.contextmanager
    def _redacted_errors(self, *, command_output: bool = False) -> Iterator[None]:
        """Re-raise a docker failure with the ssh user out of its message."""
        try:
            yield
        except Exception as exc:
            redacted = self._remote_error(exc, command_output=command_output)
            if redacted is exc:
                raise
            raise redacted from None

    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: float | None = None,
        *,
        merge_stderr: bool = True,
        raw_output: bool = False,
        stdin_data: bytes | None = None,
        unset_env: Sequence[str] = (),
    ) -> ExecResult:
        raw_output = raw_output or raw_output_requested()
        with self._redacted_errors(command_output=command[:1] == ["exec"]):
            result = await super()._run_docker_compose_command(
                command,
                check=check,
                timeout_sec=timeout_sec,
                merge_stderr=merge_stderr,
                raw_output=raw_output,
                stdin_data=stdin_data,
                unset_env=unset_env,
            )
        if raw_output:
            result._diagnostic_result = diagnostic_result(
                self._redact_result(result, command_output=command[:1] == ["exec"])
            )
            return result
        return self._redact_result(result, command_output=command[:1] == ["exec"])

    async def _run_docker_container_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        *,
        raw_output: bool = False,
        stdin_data: bytes | None = None,
    ) -> ExecResult:
        # docker exec / cp on the main container: redacted as their compose
        # forms are (_run_docker_compose_command), exec output as command output.
        raw_output = raw_output or raw_output_requested()
        with self._redacted_errors(command_output=command[:1] == ["exec"]):
            result = await super()._run_docker_container_command(
                command,
                check=check,
                timeout_sec=timeout_sec,
                raw_output=raw_output,
                stdin_data=stdin_data,
            )
        if raw_output:
            result._diagnostic_result = diagnostic_result(
                self._redact_result(result, command_output=command[:1] == ["exec"])
            )
            return result
        return self._redact_result(result, command_output=command[:1] == ["exec"])

    async def _docker_cli(
        self, args: list[str], check: bool = True, *, raw_output: bool = False
    ) -> ExecResult:
        raw_output = raw_output or raw_output_requested()
        with self._redacted_errors(command_output=args[:1] == ["exec"]):
            result = await super()._docker_cli(args, check=check)
        if raw_output:
            result._diagnostic_result = diagnostic_result(
                self._redact_result(result, command_output=args[:1] == ["exec"])
            )
            return result
        return self._redact_result(result, command_output=args[:1] == ["exec"])

    async def _run_pre_compose_hook(self) -> None:
        # The task's hook would run as the operator, and can read whatever
        # they can: the provider's shim folder (the address, the ssh agent
        # socket, the TLS key's path) included, and copy it into the image
        # it builds. No environment it is given can stop that, so it runs
        # only when the operator opted in with the launch gate's switch
        # (_ALLOW_HOST_ACCESS_VAR), and then with Compose's environment
        # (compose_env: no connection setting, SSH agent or shim). Then,
        # before anything is built or created, the project the hook leaves
        # behind is checked.
        if self._pre_compose_hook_path.is_file() and not _host_access_allowed():
            raise SandboxStartupError(
                f"{_REFUSED}environment/benchflow-pre-compose.sh would run on "
                "this machine as you, where it can read how BenchFlow reaches "
                f"the remote host; set {_ALLOW_HOST_ACCESS_VAR}=1 if you trust "
                "the task with this machine"
            )
        with self._redacted_errors():
            await super()._run_pre_compose_hook()
        await self._check_task_project()

    async def _check_task_project(self) -> None:
        """Refuse a task whose Compose project reaches the remote host's files
        or another workload's volume.

        Reads the project as Compose resolves it (``config --format json``):

        - a bind mount (or a ``file:`` secret or config, which Compose binds)
          names a path on the remote host, which this machine never copies
          anything to: refused;
        - a volume the local driver binds to a host path: refused;
        - an ``external`` volume names one on the host that the trial did not
          make: refused;
        - a volume that is not ``external`` and already exists on the host
          without this trial's project label belongs to someone else, and
          teardown would delete it with the project's volumes: refused;
        - a build with ``ssh:`` would get this machine's SSH agent or keys:
          refused.

        Not checked here: service settings that hand a container the host
        (``privileged``, ``devices``, host namespaces, added capabilities,
        ``volumes_from``); the pre-compose hook, which runs before this as the
        operator when they allowed it (_run_pre_compose_hook).
        """
        # Compose writes its warnings (an obsolete `version:` key, a variable
        # that is not set) to stderr: only stdout is the project.
        config = await self._run_docker_compose_command(
            ["config", "--format", "json"], timeout_sec=60, raw_output=True
        )
        try:
            project = json.loads(config.stdout or "")
        except json.JSONDecodeError as exc:
            raise SandboxStartupError(
                f"{_REFUSED}could not read the Compose project "
                f"(`docker compose config`): {exc}"
            ) from None
        problems = _host_reach(
            project, bfx_volumes={v for v, _ in self._bfx_read_only_volumes()}
        )
        names = _owned_volume_names(project, self._project_name)
        if names:
            problems += await self._foreign_volumes(names)
        if problems:
            # The names come from exact (unredacted) data; the message is a
            # diagnostic, so it is redacted like any other, all but its
            # fixed start, which scoring reads to know a retry only repeats it.
            raise SandboxStartupError(
                _REFUSED + self.docker_host.redact("; ".join(problems))
            )

    async def _foreign_volumes(self, names: list[str]) -> list[str]:
        """Those of ``names`` that exist on the host without this project's label."""
        result = await self._docker_cli(
            ["volume", "inspect", *names], check=False, raw_output=True
        )
        try:
            found = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            found = None
        if not isinstance(found, list) or (
            result.return_code != 0
            and not found
            and "no such volume" not in str(result.stderr or "").lower()
        ):
            # Redacted before the cut, so no half of a login is left behind.
            detail = self.docker_host.redact(result.stderr or result.stdout or "")
            return [
                "could not check whether its volumes already exist on the host "
                f"({detail.strip()[-200:]})"
            ]
        problems = []
        for volume in found:
            labels = (volume or {}).get("Labels") or {}
            if labels.get("com.docker.compose.project") != self._project_name:
                problems.append(
                    f"volume {volume.get('Name')!r} already exists on the host and "
                    "is not this trial's (the trial would write into another "
                    "workload's data); drop its `name:`"
                )
        return problems

    async def snapshot(self, name: str | None = None) -> SandboxImage:
        # `docker commit` runs outside _docker_cli; its error quotes stderr.
        with self._redacted_errors():
            return await super().snapshot(name)

    async def restore(self, image: SandboxImage) -> None:
        # Its errors quote the container id and inspect's output (base code).
        with self._redacted_errors():
            await super().restore(image)

    async def adopt_snapshot(self, image: SandboxImage) -> None:
        with self._redacted_errors():
            await super().adopt_snapshot(image)

    def _diagnostic_text(self, text: str) -> str:
        return self.docker_host.redact(text)

    def _redact_result(
        self, result: ExecResult, *, command_output: bool = False
    ) -> ExecResult:
        redact = self.docker_host.redact
        return ExecResult(
            stdout=redact(
                result.stdout or "",
                command_output=command_output,
                transport_errors=result.return_code != 0,
            )
            or result.stdout,
            stderr=redact(
                result.stderr or "",
                command_output=command_output,
                transport_errors=result.return_code != 0,
            )
            or result.stderr,
            return_code=result.return_code,
        )

    def _is_retryable_build_error(self, message: str) -> bool:
        # A full remote disk stays full across a retry seconds later.
        if _DISK_FULL.search(message):
            return False
        return _is_retryable_docker_build_error(message)

    # --- lifecycle --------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        capacity = await asyncio.to_thread(probe_remote_docker, self.docker_host)
        check_capacity(
            capacity,
            cpus=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
            host_display=self.docker_host.display,
        )
        self._host_checked = True
        self._daemon_architecture = capacity.architecture
        self.logger.info(
            "Remote Docker host %s: Docker %s, %d CPUs, %d MB",
            self.docker_host.display,
            capacity.server_version,
            capacity.cpus,
            capacity.memory_mb,
        )
        await super().start(force_build)

    async def _prepare_log_dirs_after_up(self) -> None:
        """Create the log folders in the container; nothing is mounted."""
        dirs = " ".join(
            str(p)
            for p in (
                SandboxPaths.agent_dir,
                SandboxPaths.verifier_dir,
                SandboxPaths.artifacts_dir,
            )
        )
        result = await exec_raw(
            self.exec, f"mkdir -p {dirs} && chmod 777 {dirs}", user="root"
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"could not create the log folders in the remote container: "
                f"{diagnostic_result(result).stdout.render() or diagnostic_result(result).stderr.render()}"
            )
        self._logs_are_mounted = False

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False, service: str = "main"
    ) -> None:
        # ``docker cp`` writes copied files as the caller; owner ids from this
        # machine mean nothing inside a container on another host.
        return None

    async def attach(self) -> None:
        """A shell in ``main`` (``compose exec -it``), then ``down``.

        Both run as every other Compose call does (_compose_client): the
        same address flags, project, files and environment, so a task's
        files interpolate the same values. The terminal keeps the shell;
        docker's own messages (its stderr) and ``down``'s output are redacted
        before they are shown.
        """
        from benchflow.sandbox.docker import interactive_root_shell, route_exec_argv
        from benchflow.sandbox.process.docker import redacted_stderr_lines

        _, env = self._compose_client()
        # Root's interactive shell comes from the route (C8), as on Docker.
        route = await self.exec_route("root", system_shell=True)
        process = await asyncio.create_subprocess_exec(
            *self._compose_base(),
            *route_exec_argv(
                route,
                "",
                container="main",
                user="0",
                stdin=True,
                tty=True,
                shell_argv=interactive_root_shell(route),
            ),
            env=env,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stderr is not None
        redact = functools.partial(
            self.docker_host.stream_redactor(), command_output=True
        )
        async for line in redacted_stderr_lines(process.stderr, redact):
            sys.stderr.write(line.decode())
            sys.stderr.flush()
        await process.wait()
        result = await self._run_docker_compose_command(
            self._down_command(False), check=False
        )
        if result.return_code != 0:
            sys.stderr.write(
                f"docker compose down failed: {result.stdout or result.stderr}\n"
            )

    def _down_command(self, delete: bool) -> list[str]:
        # No --volumes and no --rmi (compose_down_command): `down --volumes`
        # removes every volume the project names, a pre-existing one
        # included, and `--rmi all` every image a service uses, pulled ones
        # included. The base teardown kills the containers first; the
        # project's volumes (the trial's, named at creation) go
        # by label (_remove_project_volumes, _remove_leftovers) and its
        # images by what it built (_remove_built_images, after the leftovers).
        return compose_down_command()

    def _removes_volumes(self, delete: bool) -> bool:
        # Remote leftovers are unreachable from the client, but reused foreign
        # volumes are not ours: use the same ownership filter as local Docker
        # (a recovery sandbox's volumes are its own, as there).
        return True

    def _removes_built_images(self, delete: bool) -> bool:
        # stop() removes them itself, once the leftovers are gone too, so no
        # container of the project still uses one.
        return False

    async def stop(self, delete: bool) -> None:
        if not self._host_checked:
            # The host check failed or start() never ran: nothing exists.
            self._snapshot_credentials.clear()
            return
        # The base teardown SIGKILLs every container at once, then removes
        # them with their anonymous volumes (only this project uses those).
        await super().stop(delete)
        if self._keep_containers:
            return
        leftovers = await self._project_leftovers()
        if leftovers and any(leftovers.values()):
            await self._remove_leftovers(leftovers)
            leftovers = await self._project_leftovers()
        if super()._removes_built_images(delete) and leftovers is not None:
            await self._remove_built_images()
        if leftovers is None:
            self.logger.warning(
                "Could not confirm cleanup of compose project %s on %s (host "
                "unreachable). Remove what carries its label when the host is "
                "back: docker ps -a, docker network ls and docker volume ls "
                "with --filter label=com.docker.compose.project=%s",
                self._project_name,
                self.docker_host.display,
                self._project_name,
            )
        elif any(leftovers.values()):
            # The names are exact data (the removal used them); the warning
            # is a diagnostic, redacted like any other (a volume can be named
            # like the ssh login).
            self.logger.warning(
                "Compose project %s left on %s after teardown: %s",
                self._project_name,
                self.docker_host.display,
                self.docker_host.redact(
                    ", ".join(
                        f"{kind} {' '.join(ids)}"
                        for kind, ids in leftovers.items()
                        if ids
                    )
                ),
            )

    async def _force_kill_project(self) -> None:
        leftovers = await self._project_leftovers()
        if leftovers:
            await self._remove_leftovers(leftovers)

    async def _project_leftovers(self) -> dict[str, list[str]] | None:
        """Containers, networks and volumes with this project's label; None if unreachable."""
        label = f"label=com.docker.compose.project={self._project_name}"
        found: dict[str, list[str]] = {}
        for kind, args in (
            ("containers", ["ps", "-aq", "--filter", label]),
            ("networks", ["network", "ls", "-q", "--filter", label]),
            ("volumes", ["volume", "ls", "-q", "--filter", label]),
        ):
            try:
                result = await asyncio.wait_for(
                    self._docker_cli(args, check=False, raw_output=True),
                    timeout=_LIST_TIMEOUT_SEC,
                )
            except (TimeoutError, OSError):
                return None
            if result.return_code != 0:
                return None
            found[kind] = str(result.stdout or "").split()
        return found

    async def _remove_leftovers(self, leftovers: dict[str, list[str]]) -> None:
        for kind, args in (
            # Without -v: the trial's volumes carry the project label.
            ("containers", ["rm", "-f"]),
            ("networks", ["network", "rm"]),
            ("volumes", ["volume", "rm", "-f"]),
        ):
            ids = leftovers.get(kind) or []
            if not ids:
                continue
            try:
                await asyncio.wait_for(
                    self._docker_cli([*args, *ids], check=False),
                    timeout=_LIST_TIMEOUT_SEC,
                )
            except (TimeoutError, OSError) as exc:
                self.logger.warning(
                    "Removing %s of compose project %s failed: %s",
                    kind,
                    self._project_name,
                    self.docker_host.redact(str(exc)),
                )
