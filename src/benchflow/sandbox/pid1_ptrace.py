"""PID 1 as the sandbox user, on a host that lets any of its processes trace it.

On Docker PID 1 runs as the image's ``USER``. When that is the sandbox user's
account (Terminal-Bench 4.1's ``rs-archive-clone`` ends with ``USER agent``),
PID 1 is the agent's by uid, and no kill from inside the sandbox can end it.
The kill before grading leaves it alone only when nothing of the agent's could
have traced it: Yama's ``kernel.yama.ptrace_scope`` at 1 to 3
(``lockdown._KILL_PRELUDE``'s ``pid1_inert``). At 0 (Debian's default) or on a
kernel without Yama, any process of the agent's can attach to PID 1 and write
its memory, so the kill counts PID 1, cannot end it, and every trial errors in
the verifier's hardening after the agent's whole turn.

The sandbox is refused when it starts instead, before anything of the agent's
runs, by the same test the kill applies to PID 1: one of its real, effective or
saved uids is the sandbox user's, and the scope is 0 or there is no Yama. The
task's image and the host decide it, so another attempt meets it again
(``scoring._UNRECOVERABLE_STARTUP_RES``). ``bench run --dry-run`` refuses the
same task with the same text where it can read the scope of the kernel the
sandbox will run on; where it cannot (a remote or virtualised daemon, a
provider's sandbox) it says so.

Every trial records the scope it read: result.json's ``host_ptrace_scope``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: The Yama setting, as the sandbox and the host read it.
PTRACE_SCOPE_FILE = "/proc/sys/kernel/yama/ptrace_scope"

#: result.json's ``host_ptrace_scope`` on a kernel without Yama, which leaves
#: ptrace to the classic same-uid rule, as scope 0 does.
NO_YAMA = "absent"

#: The record's key in result.json.
RECORD_KEY = "host_ptrace_scope"

_RESULT = re.compile(
    r"^BF pid1-ptrace scope=(\S+) pid1=(\S+) (\S+) (\S+) user=(\d+|-)$", re.M
)


class Pid1TraceableError(RuntimeError):
    """PID 1 is the sandbox user's on a host where its processes can trace it.

    The message starts with "sandbox startup refused", so the trial is a
    sandbox-setup failure of the task's and the host's, never retried.
    ``scope`` is the scope that was read, for the refused trial's record.
    """

    scope: int | str | None = None


def refusal(sandbox_user: str, scope: int | str) -> Pid1TraceableError:
    """The refusal the run raises at sandbox start and its dry run prints."""
    host = (
        "this host's kernel has no Yama (no kernel.yama.ptrace_scope)"
        if scope == NO_YAMA
        else f"this host's kernel.yama.ptrace_scope is {scope}"
    )
    error = Pid1TraceableError(
        f"sandbox startup refused: PID 1 of the task's image runs as the "
        f"sandbox user {sandbox_user!r}, and {host}, so any process of the "
        f"sandbox user's could trace PID 1 and the kill before grading could "
        f"not vouch for it (every trial would error at the verifier's "
        f"hardening after the agent's turn). Set kernel.yama.ptrace_scope=1 on "
        f"the host (sysctl -w kernel.yama.ptrace_scope=1, kept in "
        f"/etc/sysctl.d/), or end the task's image on another USER than "
        f"{sandbox_user!r}"
    )
    error.scope = scope
    return error


def unprotected(scope: int | str | None) -> bool:
    """A scope under which the sandbox user's processes may trace PID 1."""
    return scope == 0 or scope == NO_YAMA


def parse_scope(text: str) -> int | str | None:
    """The value of :data:`PTRACE_SCOPE_FILE`: 0 to 3, :data:`NO_YAMA`, or
    None when it could not be read."""
    text = text.strip()
    if text == NO_YAMA:
        return NO_YAMA
    if text in ("0", "1", "2", "3"):
        return int(text)
    return None


def probe_command(sandbox_user: str) -> str:
    """Root's read of the scope, PID 1's real, effective and saved uids and
    the sandbox user's uid (``-`` while the image has no such account:
    BenchFlow then creates it with a uid nothing in the image uses). Shell
    builtins and ``id`` only, as the kill: busybox and slim images."""
    user = shlex.quote(sandbox_user)
    return (
        # A sysctl file ends at any offset past 0, so the shell's byte-wise
        # read gets the digit and then a failed status: the value decides.
        f"s={NO_YAMA}; if [ -e {PTRACE_SCOPE_FILE} ]; then s=; "
        f"{{ read -r s < {PTRACE_SCOPE_FILE}; }} 2>/dev/null; "
        'case "$s" in 0|1|2|3) ;; *) s=unreadable ;; esac; fi; '
        "ids='- - -'; "
        '{ while read -r k a b c rest || [ -n "$k" ]; do '
        'if [ "$k" = Uid: ]; then ids="$a $b $c"; break; fi; k=; '
        "done < /proc/1/status; } 2>/dev/null; "
        f"u=$(id -u {user} 2>/dev/null) || u=-; "
        'case "$u" in ""|*[!0-9]*) u=- ;; esac; '
        'echo "BF pid1-ptrace scope=$s pid1=$ids user=$u"'
    )


def decide(stdout: Any, sandbox_user: str | None) -> tuple[int | str | None, bool]:
    """(The scope read, whether the sandbox is refused) from the probe's
    output. An answer that cannot be read refuses nothing: the kill before
    grading still fails closed on a PID 1 it cannot vouch for."""
    found = _RESULT.search(stdout) if isinstance(stdout, str) else None
    if found is None:
        return None, False
    scope = parse_scope(found[1])
    uid = found[5]
    pid1 = {found[2], found[3], found[4]}
    refused = bool(sandbox_user) and uid != "-" and uid in pid1 and unprotected(scope)
    return scope, refused


async def check_at_start(env: Any, sandbox_user: str | None) -> int | str | None:
    """Read the scope and refuse the sandbox it covers (see the module), right
    after the sandbox starts. Returns the scope for the trial's record; None
    when the probe gave no answer. The refusal carries it as ``scope``."""
    from benchflow.sandbox._exec_output import exec_raw

    try:
        result = await exec_raw(
            env.exec,
            probe_command(sandbox_user or "root"),
            user="root",
            timeout_sec=30,
        )
    except Exception:
        return None
    scope, refused = decide(getattr(result, "stdout", None), sandbox_user)
    if refused and sandbox_user and scope is not None:
        raise refusal(sandbox_user, scope)
    return scope


# --- the dry run: the same decision, from the task's files and this host ---

#: Notes of the checks a dry run could not make, by task ('s
#: ``unchecked``); set while ``bench run --dry-run`` runs its gate.
_DRY_RUN_NOTES: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("benchflow_pid1_ptrace_notes", default=None)
)


@contextlib.contextmanager
def collecting_dry_run_notes() -> Iterator[list[dict[str, str]]]:
    """The notes :func:`dry_run_check` leaves while the block runs."""
    notes: list[dict[str, str]] = []
    token = _DRY_RUN_NOTES.set(notes)
    try:
        yield notes
    finally:
        _DRY_RUN_NOTES.reset(token)


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append((current + line).strip())
        current = ""
    if current.strip():
        lines.append(current.strip())
    return lines


_OPTION_VALUES = {
    "-u",
    "--uid",
    "-g",
    "--gid",
    "-G",
    "--groups",
    "-d",
    "--home",
    "--home-dir",
    "-s",
    "--shell",
    "-c",
    "--comment",
    "-h",
    "-k",
    "--skel",
    "-K",
    "--key",
    "-e",
    "--expiredate",
    "-f",
    "--inactive",
    "-p",
    "--password",
}


def _accounts(command: str) -> dict[str, str | None]:
    """Each account a ``useradd`` or ``adduser`` in *command* creates, with
    the uid it names (None when it names none)."""
    found: dict[str, str | None] = {}
    for match in re.finditer(r"\b(?:useradd|adduser)\b([^;&|]*)", command):
        words = match.group(1).split()
        uid = None
        name = None
        skip = False
        for index, word in enumerate(words):
            if skip:
                skip = False
                continue
            if word in ("-u", "--uid") and index + 1 < len(words):
                uid = words[index + 1]
            if word.startswith("--uid="):
                uid = word.split("=", 1)[1]
            if word.startswith("-"):
                skip = word in _OPTION_VALUES
                continue
            name = word
        if name:
            found[name.strip("\"'")] = uid.strip("\"'") if uid else None
    return found


@functools.lru_cache(maxsize=256)
def _local_image_user(ref: str) -> str | None:
    """Config.User of *ref* in this host's Docker image store; None when it is
    not there (a dry run pulls nothing)."""
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.User}}", ref],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        user = json.loads(done.stdout.strip() or "null")
    except ValueError:
        return None
    return user if isinstance(user, str) else None


def _base_user(ref: str) -> str | None:
    """The USER an external base image gives: the local image store's, else
    root for a Docker official image (they all start as root); None when it
    cannot be known without pulling."""
    local = _local_image_user(ref)
    if local is not None:
        return local
    head, _, last = ref.split("@", 1)[0].rpartition("/")
    last = last.split(":", 1)[0]
    name = f"{head}/{last}" if head else last
    if "/" not in name or name.startswith(("library/", "docker.io/library/")):
        return ""
    return None


def dockerfile_user(text: str) -> tuple[str | None, dict[str, str | None]]:
    """The final stage's USER (through earlier stages and its base image) and
    the accounts the Dockerfile creates; the user is None when it depends on
    a base image this host does not hold."""
    args: dict[str, str] = {}
    stages: list[dict[str, Any]] = []
    accounts: dict[str, str | None] = {}

    def expand(value: str) -> str:
        return re.sub(
            r"\$\{?(\w+)(?::?-([^}]*))?\}?",
            lambda m: args.get(m.group(1), m.group(2) or m.group(0)),
            value,
        )

    for line in _logical_lines(text):
        keyword, _, rest = line.partition(" ")
        keyword = keyword.upper()
        rest = rest.strip()
        if keyword == "ARG":
            key, _, value = rest.partition("=")
            args.setdefault(key.strip(), value.strip().strip("\"'"))
        elif keyword == "FROM":
            words = [w for w in rest.split() if not w.startswith("--")]
            name = (
                words[2].lower()
                if len(words) >= 3 and words[1].lower() == "as"
                else None
            )
            stages.append(
                {"base": expand(words[0]) if words else "", "name": name, "user": None}
            )
        elif keyword == "USER" and stages:
            stages[-1]["user"] = expand(rest.strip("\"'"))
        elif keyword == "RUN":
            accounts.update(
                {
                    expand(k): (expand(v) if v else None)
                    for k, v in _accounts(rest).items()
                }
            )

    def effective(index: int) -> str | None:
        stage = stages[index]
        if stage["user"] is not None:
            return stage["user"]
        base = stage["base"].lower()
        for earlier in range(index):
            if stages[earlier]["name"] == base:
                return effective(earlier)
        if base == "scratch":
            return ""
        return _base_user(stage["base"])

    if not stages:
        return None, accounts
    return effective(len(stages) - 1), accounts


def task_image_user(task: Any) -> tuple[str | None, dict[str, str | None]]:
    """Main's Config.User as the run's sandbox will have it, from the task's
    files: the compose file's ``user:`` for main, else the Dockerfile's, else
    a prebuilt image's from this host's image store."""
    import yaml

    environment = Path(task.paths.environment_dir)
    accounts: dict[str, str | None] = {}
    user: str | None = None
    compose = environment / "docker-compose.yaml"
    dockerfile = environment / "Dockerfile"
    if compose.is_file():
        try:
            spec = yaml.safe_load(compose.read_text()) or {}
            main = (spec.get("services") or {}).get("main") or {}
        except (OSError, ValueError, yaml.YAMLError, AttributeError):
            main = {}
        if isinstance(main, dict):
            if main.get("user") is not None:
                user = str(main["user"])
            build = main.get("build")
            if isinstance(build, dict):
                context = str(build.get("context", "."))
                folder = environment if "${" in context else environment / context
                dockerfile = folder / str(build.get("dockerfile", "Dockerfile"))
    image_user: str | None = None
    if dockerfile.is_file():
        try:
            image_user, accounts = dockerfile_user(dockerfile.read_text())
        except OSError:
            image_user = None
    else:
        image = getattr(task.config.sandbox, "docker_image", None)
        if image:
            image_user = _local_image_user(str(image))
    return (user if user is not None else image_user), accounts


def runs_as(
    user: str | None, accounts: dict[str, str | None], sandbox_user: str
) -> bool | None:
    """Whether a Config.User of *user* is the sandbox user's account: by name,
    or by the uid the Dockerfile gives that account. None when the task's
    files cannot tell: the user is unknown (a base image this host does not
    hold), or a uid the Dockerfile's account gets only when the image builds
    (``useradd`` without ``-u``) or from a base image."""
    if user is None:
        return None
    name = user.split(":", 1)[0]
    if name in ("", "root", "0"):
        return False
    if name == sandbox_user:
        return True
    if not name.isdigit():
        return False
    if sandbox_user in accounts:
        uid = accounts[sandbox_user]
        return None if uid is None else uid == name
    if any(uid == name for uid in accounts.values()):
        # The Dockerfile gives that uid to another account.
        return False
    return None


def host_scope(
    sandbox: str, environ: dict[str, str] | None = None
) -> tuple[int | str | None, str | None]:
    """(The scope of the kernel the run's sandbox will use, why it cannot be
    read here): only a local Docker daemon on this Linux host shares its
    kernel."""
    env = os.environ if environ is None else environ
    if sandbox != "docker":
        return None, f"the {sandbox} sandbox runs on another machine's kernel"
    if not sys.platform.startswith("linux"):
        return None, "Docker on this host runs its containers in a virtual machine"
    host = (env.get("DOCKER_HOST") or "").strip()
    if host and not host.startswith("unix://"):
        return None, "DOCKER_HOST names a remote Docker daemon"
    if ".docker/desktop" in host:
        return None, "Docker Desktop runs its containers in a virtual machine"
    context = (env.get("DOCKER_CONTEXT") or "").strip()
    if context and context != "default":
        return None, f"the Docker context {context!r} may be another machine's"
    if not host and not context:
        config = (
            Path(env.get("DOCKER_CONFIG") or Path.home() / ".docker") / "config.json"
        )
        with contextlib.suppress(OSError, ValueError, AttributeError):
            current = json.loads(config.read_text()).get("currentContext")
            if current and current != "default":
                return None, f"the Docker context {current!r} may be another machine's"
    path = Path(PTRACE_SCOPE_FILE)
    if not path.exists():
        return NO_YAMA, None
    try:
        scope = parse_scope(path.read_text())
    except OSError:
        scope = None
    if scope is None:
        return None, f"{PTRACE_SCOPE_FILE} could not be read"
    return scope, None


def dry_run_check(task: Any, cfg: Any) -> None:
    """The run's start refusal, decided before any sandbox where the task's
    files and this host can decide it; otherwise a note that it was not
    checked (``bench run --dry-run``'s ``unchecked``)."""
    from benchflow._utils.config import normalize_sandbox_user

    sandbox_user = normalize_sandbox_user(getattr(cfg, "sandbox_user", None))
    if not sandbox_user:
        return
    user, accounts = task_image_user(task)
    matched = runs_as(user, accounts, sandbox_user)
    if matched is False:
        return
    scope, why = host_scope(str(getattr(cfg, "environment", "docker")))
    known = why is None and scope is not None
    if known and not unprotected(scope):
        # Yama 1 to 3: the run refuses no image, whatever its user.
        return
    if known and matched and scope is not None:
        raise refusal(sandbox_user, scope)
    # Not decided here, and the run may refuse: say so (its unchecked).
    unknown = []
    if matched is None:
        unknown.append(
            "the image's user cannot be told from the task's files ("
            + (
                "its USER comes from a base image this host does not hold"
                if user is None
                else f"USER {user} is a uid the image gives its accounts when it builds"
            )
            + ")"
        )
    if not known:
        unknown.append(f"the host's scope cannot be read here ({why})")
    notes = _DRY_RUN_NOTES.get()
    if notes is not None:
        notes.append(
            {
                "target": Path(task.paths.task_dir).name,
                "reason": (
                    f"the run refuses the task when PID 1 of its image runs as "
                    f"the sandbox user {sandbox_user!r} and the host's "
                    f"kernel.yama.ptrace_scope is 0 or the kernel has no Yama; "
                    f"this dry run cannot check it: " + "; ".join(unknown)
                ),
            }
        )
