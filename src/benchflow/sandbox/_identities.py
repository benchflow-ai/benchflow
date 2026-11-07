"""Pre-agent account snapshot for script supervision and service retention."""

import re
import shlex
from dataclasses import dataclass
from typing import Any

from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import (
    new_receipt_nonce,
    read_receipt,
    receipt_command,
)

#: Prints the UID and GID of the process that runs it, on one tagged line,
#: whatever a launcher prints around it (su's warnings share the stream).
ID_PROBE = 'printf \'benchflow-id:%s:%s\\n\' "$(id -u)" "$(id -g)"'
_ID_LINE = re.compile(r"^benchflow-id:([0-9]+):([0-9]+)$", re.MULTILINE)


def protected_exec_argv(
    config: dict[str, Any],
    command: str,
    *,
    service: str = "main",
    shell: tuple[str, ...] | None = None,
    route: Any = None,
    cwd: str | None = None,
) -> list[str]:
    """Clear all container startup inputs before even env or the shell loads."""
    entries = config.get("Env")
    if entries is None:
        entries = []
    if not isinstance(entries, list) or any(
        not isinstance(entry, str) or "\0" in entry or not entry.partition("=")[0]
        for entry in entries
    ):
        raise RuntimeError("script: invalid Docker container environment")
    from benchflow._runtime_tree import KIT_BIN, ROOT_HOME
    from benchflow.sandbox._base import SYSTEM_PATH, ExecRoute, loader_variable
    from benchflow.sandbox.docker import route_exec_argv

    if route is None:
        route = ExecRoute(pin=SYSTEM_PATH, system_shell=True, task_root=False)
    shell = shell or (route.shell, "-c")
    executable = f"{KIT_BIN}/env" if route.kit else "/usr/bin/env"
    shell_argv = [
        executable,
        "-i",
        f"PATH={route.pin}",
        f"HOME={ROOT_HOME if route.kit else '/root'}",
        *shell,
        command,
    ]
    argv = route_exec_argv(
        route,
        command,
        container=service,
        user="root",
        cwd=cwd,
        shell_argv=shell_argv,
    )
    # The route's own loader flags (-e LD_PRELOAD=, ...) become removals, so
    # no loader name reaches the exec with any value, an empty one included.
    end = len(argv) - len(shell_argv) - 1
    for i in range(1, end):
        flag = argv[i]
        if argv[i - 1] == "-e" and flag.endswith("=") and loader_variable(flag[:-1]):
            argv[i] = flag[:-1]
    argv.insert(1, "-T")
    cleared = []
    # A bare Config.Env name (S1's inert bootstrap clears loader names that
    # way) is not in Docker's exec environment, so it stays off the list:
    # "-e LD_SHOW_AUXV=" revived it as present and glibc printed its aux
    # vector over the runner's markers (SCRIPT-AGENT-DOCKER-F1). A loader
    # name with a value is removed, any other name set empty, all before
    # even env itself loads.
    for name in sorted({e.partition("=")[0] for e in entries if "=" in e}):
        cleared.extend(["-e", name if loader_variable(name) else f"{name}="])
    argv[2:2] = cleared
    return argv


def removed_env_names(argv: list[str]) -> list[str]:
    """The names *argv* removes with a bare ``-e NAME``. Docker and Compose
    fill a bare name in from their own environment, so a transport runs the
    exec with these names unset there."""
    return sorted(
        {
            argv[i + 1]
            for i in range(len(argv) - 1)
            if argv[i] == "-e" and "=" not in argv[i + 1]
        }
    )


@dataclass(frozen=True)
class Identity:
    uid: int
    gid: int


def oracle_command(run_as: str, command: str) -> str:
    """How the oracle runs a command as the task's ``[agent].user``.

    su selects the account by name only: BusyBox and util-linux su both look
    up ``"1000"`` as a name, never as UID 1000 (DT2-R11-1). solve.sh and
    the identity probe that decides its service retention use this one form.
    """
    return f"su -s /bin/bash {shlex.quote(run_as)} -c {shlex.quote(command)}"


def probed_identity(result: Any, what: str) -> Identity:
    """The identity :data:`ID_PROBE` printed, or a RuntimeError naming ``what``."""
    stdout = getattr(result, "stdout", None)
    found = set(_ID_LINE.findall(stdout if isinstance(stdout, str) else ""))
    if getattr(result, "return_code", 1) != 0 or len(found) != 1:
        raise RuntimeError(
            f"sandbox identity: cannot resolve the configured user ({what}); "
            "use an account that exists in the image and that su (or the "
            "backend's exec) can switch to before the agent starts"
        )
    uid, gid = (int(n) for n in found.pop())
    return Identity(uid, gid)


async def launched_identity(
    env: Any, launch: str, *, key: tuple[Any, ...], what: str, **exec_kwargs: Any
) -> Identity:
    """Run :data:`ID_PROBE` through the launcher that will run the account
    (``launch`` wraps it), once per sandbox, before any agent code runs."""
    cache = vars(env).setdefault("_benchflow_launched_identities", {})
    if key not in cache:
        try:
            result = await exec_raw(env.exec, launch, timeout_sec=30, **exec_kwargs)
        except ValueError:
            # A launcher that refuses the selector itself (su_command takes
            # only a plain name or a decimal UID) selects no account.
            result = None
        cache[key] = probed_identity(result, what)
    return cache[key]


class Identities:
    def __init__(self, passwd: str, group: str, default_user: str = ""):
        self.users = [line.split(":") for line in passwd.splitlines() if line]
        self.groups = [line.split(":") for line in group.splitlines() if line]
        self.default_user = default_user or "0:0"
        self.cache: dict[str, Identity] = {}
        #: Whether exec selects accounts by Docker's rule (``docker exec -u``:
        #: a decimal is a UID, anything else a name), which :meth:`resolve`
        #: implements; other backends switch with su, by name (DT2-R11-1).
        self.docker_rule = False

    @staticmethod
    def number(value: str) -> int | None:
        digits = value.removeprefix("+").removeprefix("-")
        if digits and digits.isascii() and digits.isdecimal():
            number = int(value)
            if 0 <= number < 2**32 - 1:
                return number
        return None

    def resolve(self, user: str | int | None) -> Identity:
        if user is None or user == "":
            user = self.default_user
        if type(user) is int:
            user = str(user)
        if not isinstance(user, str) or any(c.isspace() for c in user):
            raise ValueError("invalid account selector")
        if user in self.cache:
            return self.cache[user]
        name, colon, group = user.partition(":")
        uid = self.number(name) if name else 0
        record = next(
            (
                r
                for r in self.users
                if len(r) == 7
                and (r[0] == name if uid is None else self.number(r[2]) == uid)
            ),
            None,
        )
        if uid is None:
            if record is None:
                raise ValueError("account is absent from /etc/passwd")
            uid = self.number(record[2])
        gid = self.number(record[3]) if record else 0
        if colon and group:
            gid = self.number(group)
            if gid is None:
                entry = next(
                    (r for r in self.groups if len(r) == 4 and r[0] == group), None
                )
                gid = self.number(entry[2]) if entry else None
        if uid is None or gid is None:
            raise ValueError("UID or GID is absent or invalid")
        identity = Identity(uid, gid)
        self.cache[user] = identity
        return identity


async def snapshot(
    env: Any, *, service: str = "main", root_exec: Any = None, config: Any = None
) -> Identities:
    own = vars(env)
    tables = own.setdefault("_benchflow_service_identity_tables", {})
    cached = (
        own.get("_benchflow_identities") if service == "main" else tables.get(service)
    )
    if isinstance(cached, Identities):
        return cached
    if root_exec is None:
        from benchflow.sandbox._base import BaseSandbox

        if isinstance(env, BaseSandbox):
            try:
                config = await env.identity_container_config(service=service)
            except RuntimeError as exc:
                raise RuntimeError(
                    "sandbox identity: cannot read selected service configuration; "
                    "check verifier.service and start its container before agent setup"
                ) from exc

            async def root_exec(command: str, **kwargs: Any) -> Any:
                return await env.identity_exec(command, service=service, **kwargs)

        else:

            async def root_exec(command: str, **kwargs: Any) -> Any:
                return await exec_raw(
                    env.exec, command, user="root", service=service, **kwargs
                )

    # The tables cross as a receipt (hex, checked): a session command's
    # output on Daytona loses every NUL, so the separator never arrived raw.
    nonce = new_receipt_nonce()
    result = read_receipt(
        await root_exec(
            receipt_command("cat /etc/passwd && printf '\\0' && cat /etc/group", nonce),
            timeout_sec=30,
        ),
        nonce,
    )
    passwd, separator, group = (result.stdout or "").partition("\0")
    if result.return_code != 0 or not separator:
        raise RuntimeError(
            "sandbox identity: cannot read trusted /etc/passwd and /etc/group; "
            "provide /bin/sh, /usr/bin/env, cat and readable account files in "
            "the image before running the agent"
        )
    default = config.get("User", "") if isinstance(config, dict) else ""
    if config is None and root_exec is not None:
        # Non-Docker backends expose their default exec identity directly.
        nonce = new_receipt_nonce()
        default_result = read_receipt(
            await exec_raw(
                env.exec,
                receipt_command("id -u; id -g", nonce),
                service=service,
                timeout_sec=30,
            ),
            nonce,
        )
        parts = (default_result.stdout or "").splitlines()
        if default_result.return_code != 0 or len(parts) != 2:
            raise RuntimeError(
                "sandbox identity: cannot resolve the default user; set verifier.user explicitly"
            )
        default = ":".join(parts)
    table = Identities(passwd, group, default)
    table.docker_rule = isinstance(config, dict)
    tables[service] = table
    if service == "main":
        # Process cleanup always acts in main, regardless of verifier.service.
        own["_benchflow_identities"] = table
    return table


def resolve(table: Identities, user: str | int | None) -> Identity:
    try:
        return table.resolve(user)
    except (ValueError, TypeError, OverflowError) as exc:
        raise RuntimeError(
            "sandbox identity: cannot resolve the configured user; use an existing "
            "image account or a decimal UID[:GID] without whitespace, and define "
            "named groups in /etc/group"
        ) from exc


def kill_prelude(env: Any, script: str, user: str) -> str:
    """Use the pre-agent numbers for every kill/freeze/release ownership test."""
    table = vars(env).get("_benchflow_identities")
    if not isinstance(table, Identities):
        return script
    identity = resolve(table, user)
    start = script.index("uid=$(id -u ")
    end = script.index("me=\n", start)
    return (
        script[:start]
        + (
            f"uid={identity.uid}\ngid={identity.gid}\n"
            "[ \"$uid\" != 0 ] || { echo 'Invalid sandbox UID' >&2; exit 2; }\n"
            "[ -r \"$P/self/status\" ] || { echo 'Process table is unavailable' >&2; exit 2; }\n"
        )
        + script[end:]
    )
