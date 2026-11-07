"""Fail-closed account selection for providers whose exec API needs su."""

from __future__ import annotations

import re
import shlex

BASH_READONLY_ENV = ("BASHOPTS", "SHELLOPTS", "UID", "EUID", "PPID", "BASH_VERSINFO")


def bash_startup_metadata() -> str:
    """Quote only Bash's intrinsic fields from the initial exec environment.

    A Bash implementation of sh changes its own option variables. Keep their
    original values across every wrapping shell. The substitution writes data
    on a descriptor; temporary variables never enter an executable's env.
    """
    patterns = "|".join(name + "=*" for name in BASH_READONLY_ENV)
    return (
        '$(case "$(PATH=/usr/bin:/bin command -v builtin)" in\n'
        f"builtin) builtin printf 'builtin export -n {' '.join(BASH_READONLY_ENV)}\\n'; "
        "while IFS= builtin read -r -d '' _bf_entry; do\n"
        f'  case "$_bf_entry" in {patterns}) '
        'builtin printf \'set -- %q "$@"\\n\' "$_bf_entry" ;; esac;\n'
        "done </proc/self/environ ;;\nesac)"
    )


def su_command(
    command: str,
    user: str | int,
    *,
    shell: str = "/bin/bash",
    su: str = "su",
    getent: str = "getent",
    trusted_tools: bool = False,
) -> str:
    """Select exactly one account, or refuse before the command can run.

    su defaults to root when an unquoted failed lookup disappears. Numeric
    strings and integers share a checked lookup; named accounts are one
    literal argument and cannot be options. Unsupported group syntax fails
    explicitly instead of being interpreted differently by each provider.
    *su* and *getent* name the programs (by absolute path from the route,
    the protected kit's once the sandbox has one, C8).
    """
    value = str(user)
    if isinstance(user, bool) or not re.fullmatch(
        r"[0-9]+|[A-Za-z_][A-Za-z0-9_.-]*[$]?", value
    ):
        raise ValueError("Sandbox user must be a nonempty account name or numeric UID")
    prefix = ""
    account = shlex.quote(value)
    if trusted_tools:
        # K1's agent bootstrap: never a bare name off PATH. The route's kit
        # paths (C8), when the caller passes them, stay.
        getent = "/usr/bin/getent" if getent == "getent" else getent
        su = "/bin/su" if su == "su" else su
    if value.isascii() and value.isdigit():
        uid = int(value)
        prefix = (
            f"_bf_account=$({getent} passwd {uid}) || exit 97; "
            "_bf_account=${_bf_account%%:*}; "
            'case "$_bf_account" in ""|-*|*[!A-Za-z0-9_.$-]*) exit 97;; esac; '
        )
        account = '"$_bf_account"'
    selected = f"{su} {account} -s {shlex.quote(shell)} -c {shlex.quote(command)}"
    if not trusted_tools:
        return prefix + selected
    # Modal/AgentCore add an outer system sh before su. If it is Bash, keep
    # that shell's original options too, before su starts the user's shell.
    # Only fixed intrinsic metadata enters argv; arbitrary exports stay in env.
    return (
        "unset -f builtin command printf read set exec; "
        + prefix
        + f"set -- {selected}; exec 5<<_BF_BACKEND_SHELL_METADATA\n"
        + bash_startup_metadata()
        + "\n_BF_BACKEND_SHELL_METADATA\n"
        + '. /proc/self/fd/5 || exit 97; exec 5<&-; exec /usr/bin/env "$@"'
    )
