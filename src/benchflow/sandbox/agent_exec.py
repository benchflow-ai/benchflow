"""Agent exec barriers.

Docker establishes no_new_privs at container creation. Its execs keep the
provider's user and shell. Native providers establish it at session entry,
before their own account selection or environment-file wrappers. The inner
bootstrap also checks the selected identity (including its real primary group)
without ever upgrading a generic exec to root just to drop it again.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any

from benchflow.sandbox._su import BASH_READONLY_ENV, bash_startup_metadata

_ROOT = (None, "root", 0, "0")


class AgentExecCommand(str):
    """A provider must enclose its entire dispatch in native_agent_argv.

    File sourcing and helpers prepended by the provider also need the barrier;
    putting it only inside this string is too late (K1-R16-1).
    """

    agent_command: str
    agent_shell: str
    agent_target: Any

    def rebuilt(self, command: str) -> AgentExecCommand:
        """The same barrier for the same account around *command* instead:
        how a backend sends an agent command over the transport bound as
        data (``benchflow.sandbox._transport.fit``)."""
        return _agent_command(command, self.agent_target, self.agent_shell)


def native_agent_argv(command: str) -> list[str]:
    """Establish the kernel barrier before any provider shell or helper.

    Providers must keep all caller environment out of this first exec's
    environment. File sourcing, timeout, account selection and the command
    bootstrap are *inside* this invocation, never prefixes outside it.
    The provider's image environment is trusted, like its executables.
    """
    return ["/usr/bin/setpriv", "--no-new-privs", "--", "/bin/sh", "-c", command]


def agent_exec_call(
    command: str,
    user: Any,
    *,
    sandbox_user: Any = None,
    identity_switch: Sequence[str] = (),
    sandbox: Any = None,
) -> tuple[str, Any]:
    """Keep the backend's identity selection and command-shell contract.

    The fallback starts as the requested account, so neither the provider's
    first shell nor the bootstrap can run agent files as root. Exported
    environment travels through a private descriptor across a clean bootstrap,
    restored only after the kernel barrier, before the command's own shell.
    Caller environment data never becomes an oversized single argv entry.
    ``identity_switch`` no longer exempts a command from the barrier: such a
    task is refused before start where the container lacks no_new_privs, and
    a command that still got here could restart an account's processes after
    that account's cleanup (K1-R15-2). The keyword stays for callers.
    """
    del identity_switch
    if sandbox is not None:
        from benchflow.trust.state import IngressKind, admit

        admit(sandbox, "main", IngressKind.SDK_EXEC)
    target = user if user is not None else sandbox_user
    if getattr(sandbox, "agent_container_no_new_privileges", False) is True:
        return command, target
    if target in _ROOT:
        return command, user
    shell = getattr(sandbox, "agent_exec_shell", "/bin/sh")
    if shell not in ("/bin/sh", "/bin/bash"):
        raise ValueError("Agent exec requires an absolute system sh or bash")
    return _agent_command(command, target, shell), target


def _agent_command(command: str, target: Any, shell: str) -> AgentExecCommand:
    """*command* for the account *target* inside K1's barrier (see
    :func:`agent_exec_call`)."""
    from benchflow.sandbox._base import SYSTEM_SHELL_UNSET_VARS

    # /bin/sh may be Dash, BusyBox, or Bash. A fresh Bash owns readonly
    # variables (e.g. UID/SHELLOPTS); never source assignments to those names.
    # Their original startup metadata goes separately to the final
    # env invocation. All other values stay on a descriptor, avoiding ARG_MAX
    # and any temporary variable that could collide with caller exports.
    # Bash invoked as sh adds posix/inherit_errexit to its own options: read
    # those intrinsic fields from the initial exec environment, not its dump.
    restored = (
        ". /proc/self/fd/3 || exit 97; "
        f'set -- {shell} -c "$1"; '
        ". /proc/self/fd/4 || exit 97; "
        'exec 3<&- 4<&-; exec /usr/bin/env "$@"'
    )
    bootstrap = (
        "[ -x /usr/bin/setpriv ] || { "
        "echo 'Agent exec requires /usr/bin/setpriv (util-linux) in the image' >&2; exit 97; }; "
        'gid=$(/usr/bin/id -g "$1") || exit 97; '
        'uid=$(/usr/bin/id -u "$1") || exit 97; '
        '[ "$(/usr/bin/id -u)" = "$uid" ] || exit 97; '
        'exec /usr/bin/setpriv --no-new-privs --regid="$gid" --keep-groups '
        f'-- /bin/sh -c {shlex.quote(restored)} agent "$2"'
    )
    wrapped = (
        "unset -f builtin command printf read; "
        "exec 3<<_BF_AGENT_ENV 4<<_BF_AGENT_SHELL_METADATA\n"
        '$(case "$(PATH=/usr/bin:/bin command -v builtin)" in\n'
        f"builtin) builtin export -n {' '.join(BASH_READONLY_ENV)}; "
        "builtin export -p ;;\n"
        "*) export -p ;;\nesac)\n_BF_AGENT_ENV\n"
        f"{bash_startup_metadata()}\n_BF_AGENT_SHELL_METADATA\n"
        f"unset {' '.join(SYSTEM_SHELL_UNSET_VARS)}; "
        "exec /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin "
        f"/bin/sh -c {shlex.quote(bootstrap)} agent {shlex.quote(str(target))} "
        f"{shlex.quote(command)}\n"
    )
    result = AgentExecCommand(wrapped)
    result.agent_command = command
    result.agent_shell = shell
    result.agent_target = target
    return result
