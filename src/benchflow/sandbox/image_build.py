"""A task image that does not build: the task's problem, named at its line.

Docker Compose's ``build`` fails with BuildKit's whole log in one
``RuntimeError`` (the compose command line, every step's output, then the
error). When the failure is one another attempt would only repeat (a ``RUN``
step that exits nonzero, a base image that is not there or not public, a
``COPY`` source that is missing, a Dockerfile that does not parse),
:func:`image_build_error` turns it into an :class:`ImageBuildError`: one
line that names the Dockerfile line, the step and BuildKit's reason, plus the
failing step's last output lines. A network or registry hiccup is left as it
was (``sandbox_setup``, retried), and so is a RUN step whose own output
names a network or registry failure (:class:`ImageBuildNetworkError`). A
registry that refused this machine's login is the user's setup
(:class:`RegistryAuthError`, ``registry_auth``).

The error is a :class:`~benchflow.errors.UserError` with the ``task`` fault:
the rollout logs it without a traceback, the summary files it under
``image_build`` (:mod:`benchflow.failures`), and it is never retried.
"""

from __future__ import annotations

import re

from benchflow._utils.scoring import IMAGE_BUILD_HEADER, REGISTRY_AUTH_HEADER
from benchflow.errors import UserError

# BuildKit reasons another attempt cannot change.
_DETERMINISTIC = (
    re.compile(r"did not complete successfully: exit code: \d+"),
    re.compile(r"\bnot found\b"),
    re.compile(r"\b403 forbidden\b"),
    re.compile(r"dockerfile parse error"),
    re.compile(r"unknown instruction"),
)
# A registry that refused this machine's login (an expired `docker login`,
# a token service that said no): the user's setup, not the task's line
# (review A-1). GHCR's anonymous-token 403 for a missing or private image
# names no login and stays the task's "not found or not public".
_REGISTRY_AUTH = (
    re.compile(r"\b401\b"),
    re.compile(r"\bunauthorized\b"),
    re.compile(r"authentication required"),
    re.compile(r"\bpull access denied\b"),
    re.compile(r"\binsufficient_scope\b"),
    re.compile(r"authorization failed"),
    re.compile(r"\bdenied: "),
)
# What a RUN step prints when the network, a mirror or an index failed under
# it (apt, pip, npm, curl, git). BuildKit's own reason for such a step is only
# "exit code: N", so the step's output decides (review A-1): the build is
# the sandbox's, retried.
_STEP_NETWORK = (
    re.compile(r"temporary failure (?:resolving|in name resolution)"),
    re.compile(r"could not resolve(?: host)?"),
    re.compile(r"name or service not known|no such host|getaddrinfo"),
    re.compile(r"hash sum mismatch"),
    re.compile(r"read timed out|connection timed out|timed out while|i/o timeout"),
    re.compile(r"network is unreachable|no route to host"),
    re.compile(r"connection (?:refused|reset)|remote end closed connection"),
    re.compile(r"tls handshake timeout|eof occurred in violation of protocol"),
    re.compile(r"max retries exceeded|failed to connect|unable to connect"),
    re.compile(r"\b(?:http|error|status)\b[^\n]{0,20}\b5\d\d\b"),
    re.compile(
        r"\b5\d\d (?:server error|internal server error|bad gateway|"
        r"service unavailable|gateway time-?out)"
    ),
    re.compile(r"\b429\b|too many requests|toomanyrequests|rate limit"),
    re.compile(r"\betimedout\b|\beconnreset\b|\beai_again\b"),
)
# Reasons that are the network's, the registry's or the host's: the build is
# left to the retry the sandbox_setup category gives it.
_TRANSIENT = (
    re.compile(r"failed to do request"),
    re.compile(r"deadlineexceeded|deadline exceeded"),
    re.compile(r"i/o timeout|tls handshake timeout|timed out|timeout"),
    re.compile(r"connection (?:refused|reset)"),
    re.compile(r"toomanyrequests|too many requests|\b429\b"),
    re.compile(r"\b5\d\d (?:internal server error|bad gateway|service unavailable)"),
    re.compile(r"no space left on device"),
    re.compile(r"temporary failure in name resolution|no such host"),
)
_SOLVE_RE = re.compile(r"failed to solve: (.+)")
# BuildKit's "Dockerfile:5" line (after "Stderr: " in the compose error).
_DOCKERFILE_LINE_RE = re.compile(
    r"(?m)(?:^|\s)([\w.\-]*[Dd]ockerfile[\w.\-]*):(\d+)\s*$"
)
_MARKED_STEP_RE = re.compile(r"(?m)^\s*\d+\s*\|\s*>>>\s?(.*?)\s*$")
_FAILED_STEP_RE = re.compile(r"(?m)^#(\d+) ERROR: ")
_EXIT_CODE_RE = re.compile(r"did not complete successfully: exit code: (\d+)")
_STEP_TAIL_LINES = 2
_ERROR_LINE_RE = re.compile(r"^(?:e:|error|fatal|err!)", re.IGNORECASE)
_DETAIL_LIMIT = 400


class ImageBuildError(RuntimeError, UserError):
    """The task's image did not build, for a reason another attempt repeats."""

    fault = "task"

    def __init__(self, detail: str, *, log: str = "") -> None:
        super().__init__(f"{IMAGE_BUILD_HEADER}: {detail}")
        self.detail = detail
        # Docker's whole output: logged at DEBUG, never printed by default.
        self.log = log
        self.hint = (
            "fix that line in the task's environment/Dockerfile; "
            "`docker build <task>/environment` prints the whole build log"
        )


class ImageBuildNetworkError(RuntimeError):
    """A RUN step failed because the network or a registry did, not the task.

    It opens with "Sandbox startup failed: ", so classify_error files it as
    ``sandbox_setup`` (infrastructure) and the trial is retried.
    """

    def __init__(self, detail: str, *, log: str = "") -> None:
        super().__init__(
            "Sandbox startup failed: network or registry failure while the image "
            f"built: {detail}"
        )
        self.log = log


class RegistryAuthError(RuntimeError, UserError):
    """The registry refused this machine's login for the task's base image."""

    fault = "setup"

    def __init__(self, detail: str, *, log: str = "") -> None:
        super().__init__(f"{REGISTRY_AUTH_HEADER}: {detail}")
        self.log = log
        self.hint = (
            "check `docker login <registry>` on this machine, and that the FROM "
            "line names an image that login can read"
        )


def _step_lines(text: str) -> list[str]:
    """Every output line of the failing step (``#6 0.32 E: ...``), oldest first."""
    failed = _FAILED_STEP_RE.findall(text)
    if not failed:
        return []
    number = failed[-1]
    lines = []
    for raw in text.splitlines():
        match = re.match(rf"#{number} \d+(?:\.\d+)? (.*)$", raw)
        if match and match.group(1).strip():
            lines.append(match.group(1).strip())
    return lines


def _step_output(text: str) -> list[str]:
    """The failing step's last error lines, oldest first."""
    lines = _step_lines(text)
    # The step's own error lines (apt's "E: ...") say it best; else its last.
    errors = [line for line in lines if _ERROR_LINE_RE.match(line)]
    return (errors or lines)[-_STEP_TAIL_LINES:]


def _network_line(text: str) -> str | None:
    """The failing step's first line that names a network or registry failure."""
    for line in _step_lines(text):
        if any(p.search(line.lower()) for p in _STEP_NETWORK):
            return line
    return None


def image_build_error(
    message: str,
) -> ImageBuildError | ImageBuildNetworkError | RegistryAuthError | None:
    """The typed error a failed ``compose build`` stands for, or None.

    - :class:`ImageBuildNetworkError` (infrastructure, retried): a RUN step
      whose own output names a network, mirror or index failure;
    - :class:`RegistryAuthError` (setup): the registry refused the login;
    - :class:`ImageBuildError` (the task's, not retried): anything else
      another attempt repeats.

    None when the output is not BuildKit's, or when BuildKit's reason is one
    a retry can change (the network, a rate limit, the host's disk).
    """
    lower = message.lower()
    solved = _SOLVE_RE.findall(message)
    if not solved:
        return None
    reason = solved[-1].strip().rstrip(".").strip()
    reason_lower = reason.lower()
    if any(p.search(reason_lower) for p in _TRANSIENT):
        return None
    if "no space left on device" in lower:
        return None
    auth = "anonymous token" not in reason_lower and any(
        p.search(reason_lower) for p in _REGISTRY_AUTH
    )
    if not auth and not any(p.search(reason_lower) for p in _DETERMINISTIC):
        return None
    where = ""
    located = _DOCKERFILE_LINE_RE.findall(message)
    if located:
        name, line = located[-1]
        where = f"environment/{name}:{line}"
    marked = _MARKED_STEP_RE.findall(message)
    step = marked[-1] if marked else ""
    exit_code = _EXIT_CODE_RE.search(reason)
    if auth:
        said = reason.rsplit(": ", 1)[-1]
        parts = [p for p in (where, f"`{step}`" if step else "", f"({said})") if p]
        return RegistryAuthError(_cut(" ".join(parts)), log=message)
    if exit_code is not None and (network := _network_line(message)) is not None:
        parts = [p for p in (where, f"`{step}`" if step else "", f"({network})") if p]
        return ImageBuildNetworkError(_cut(" ".join(parts)), log=message)
    if exit_code is not None:
        what = f"exited with code {exit_code.group(1)}"
        output = _step_output(message)
        if output:
            what += " (" + "; ".join(output) + ")"
    elif step.upper().startswith("FROM ") and (
        "not found" in reason_lower or "403" in reason_lower or "denied" in reason_lower
    ):
        image = step.split()[1] if len(step.split()) > 1 else "the base image"
        said = reason.rsplit(": ", 1)[-1]
        what = f"base image {image} is not found or not public ({said})"
    else:
        what = reason
    parts = [p for p in (where, f"`{step}`" if step else "", what) if p]
    return ImageBuildError(_cut(" ".join(parts)), log=message)


def _cut(detail: str) -> str:
    if len(detail) > _DETAIL_LIMIT:
        return detail[: _DETAIL_LIMIT - 1].rstrip() + "…"
    return detail
