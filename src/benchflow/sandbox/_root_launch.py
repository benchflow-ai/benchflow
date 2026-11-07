"""The agent launch's root bootstrap (C8).

Every live-process backend starts the agent as the sandbox's root user and
drops to the sandbox user from there. Before the drop nothing may come from
the task's PATH or start-up environment: the image's PATH can lead into a
folder the agent wrote in an earlier scene (``ENV PATH=/app/.venv/bin:$PATH``
with ``/app`` the agent's), so a bare ``bash``, ``setpriv``, ``grep``, ``su``,
``rm`` or ``python3`` there would run the agent's program as root.

The bootstrap therefore:

1. keeps the task's values of PATH and the loader and start-up variables
   (:data:`TASK_ENV_NAMES`) in ``_BENCHFLOW_KEEP_<name>``, with builtins only;
2. unsets them and pins PATH to :data:`~benchflow.sandbox._base.SYSTEM_PATH`;
3. sources the staged env file (a builtin), moves what it set aside the same
   way, and removes the file with the system ``rm``;
4. drops privileges with the system ``setpriv`` (or ``su``) and ``/bin/bash``
   started with no profile or rc file;
5. only then, as the sandbox user, puts the task's values back and runs the
   agent (through the ACP line filter, if the backend uses it).

Where the backend starts the first shell itself (``docker exec``, ``container
exec``), the loader variables are also set empty at that exec
(``system_shell_env_flags``), so the image's ``LD_PRELOAD`` or
``LD_LIBRARY_PATH`` never reaches root's ``/bin/bash``; the task's values come
back from PID 1's environment (``/proc/1/environ``, read as root by the
system ``tr``), the same environment the exec started with.
"""

from __future__ import annotations

import inspect
import shlex
from typing import Any

from benchflow._runtime_tree import KIT_BIN, KIT_RECORD
from benchflow.sandbox._base import (
    IMAGE_ENV_PREFIX,
    SYSTEM_PATH,
    SYSTEM_SHELL_EMPTY_VARS,
    SYSTEM_SHELL_UNSET_VARS,
)
from benchflow.sandbox._exec_output import exec_raw

#: The shell root's launch runs, by absolute path. Every live-process backend
#: already required ``bash`` (``bash -c`` / ``bash -lc``).
LAUNCH_SHELL = "/bin/bash"
#: ``/bin/bash`` reading no profile, rc or ``BASH_ENV`` file.
_ROOT_BASH = f"{LAUNCH_SHELL} --noprofile --norc"

#: What the agent gets back after the drop, unchanged from the task's.
TASK_ENV_NAMES = ("PATH", *SYSTEM_SHELL_UNSET_VARS)
_NAMES = " ".join(TASK_ENV_NAMES)
_KEEP = "_BENCHFLOW_KEEP_"
_SAVED = "_BENCHFLOW_TASK_ENV_SAVED"
#: The launch env's own loader and start-up values travel under this prefix
#: (inert names) and become the real ones only after the drop (PT-C8-3).
_SET = "_BENCHFLOW_SET_"
_STARTUP_NAMES = " ".join(SYSTEM_SHELL_UNSET_VARS)
_EMPTY_NAMES = " ".join(SYSTEM_SHELL_EMPTY_VARS)


def inert_task_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """*env* (the agent's launch env) with each loader and start-up variable
    under an inert ``_BENCHFLOW_SET_<name>``.

    A launcher stages this env for root's shell (an env file, ``--env NAME``
    of compose, the provider's process env): under its own name a value such
    as ``LD_PRELOAD=/app/lib/profile.so``, a library the agent may have
    replaced since, would load into root's first shell and every system tool
    before the drop. :func:`restore_task_env` makes them real as the sandbox
    user, over the task's own values.
    """
    if not env:
        return env
    return {
        (f"{_SET}{k}" if k in SYSTEM_SHELL_UNSET_VARS else k): v for k, v in env.items()
    }


def _keep_each_set(names: str) -> str:
    # eval sees only the fixed names above; values are expanded, never parsed.
    return (
        f"for _bf_n in {names}; do eval "
        f'"if [ \\"\\${{$_bf_n+x}}\\" = x ]; then {_KEEP}$_bf_n=\\${{$_bf_n}}; '
        f'export {_KEEP}$_bf_n; fi"; done'
    )


def save_task_env() -> str:
    """Builtins only: keep the task's values, once per launch."""
    return (
        f'if [ "${{{_SAVED}-}}" != 1 ]; then {_keep_each_set(_NAMES)}; '
        f"{_SAVED}=1; export {_SAVED}; fi"
    )


def sanitize_root_env(path: str = SYSTEM_PATH) -> str:
    """Builtins only: no loader or start-up variable, PATH *path* (the
    route's: the protected kit's once the sandbox has one, C8)."""
    return (
        f"unset {' '.join(SYSTEM_SHELL_UNSET_VARS)}; "
        f"PATH={shlex.quote(path)}; export PATH"
    )


#: The PATH an :class:`AgentLaunch` pins for its own root steps.
AGENT_LAUNCH_PATH = f"{KIT_BIN}:{SYSTEM_PATH}"


def _route_rm(path: str) -> str:
    """The ``rm`` a launch's traps run by absolute path: the kit's when
    *path* starts with the kit's bin, else the system's."""
    first = path.split(":", 1)[0]
    return f"{first}/rm" if first == KIT_BIN else "/bin/rm"


def restore_task_env() -> str:
    """As the sandbox user: the task's values back, the kept copies gone.

    In order: what root's shell started with (kept by :func:`save_task_env`,
    or read from PID 1); the image's values a provider kept aside
    (``_BF_IMAGE_<name>``, Daytona and AgentCore) where that is empty; then
    the launch env's own (``_BENCHFLOW_SET_<name>``, :func:`inert_task_env`).
    """
    return (
        f"for _bf_n in {_NAMES}; do eval "
        f'"if [ \\"\\${{{_KEEP}$_bf_n+x}}\\" = x ]; then $_bf_n=\\${{{_KEEP}$_bf_n}}; '
        f'export $_bf_n; unset {_KEEP}$_bf_n; fi"; done; '
        f"for _bf_n in {_EMPTY_NAMES}; do eval "
        f'"if [ -z \\"\\${{$_bf_n-}}\\" ] && [ -n \\"\\${{{IMAGE_ENV_PREFIX}$_bf_n-}}\\" ]; '
        f"then $_bf_n=\\${{{IMAGE_ENV_PREFIX}$_bf_n}}; export $_bf_n; fi; "
        f'unset {IMAGE_ENV_PREFIX}$_bf_n"; done; '
        f"for _bf_n in {_STARTUP_NAMES}; do eval "
        f'"if [ \\"\\${{{_SET}$_bf_n+x}}\\" = x ]; then $_bf_n=\\${{{_SET}$_bf_n}}; '
        f'export $_bf_n; unset {_SET}$_bf_n; fi"; done; unset {_SAVED} _bf_n'
    )


def _keep_from_pid1() -> str:
    """The image's loader values, for names the exec set empty (see module doc)."""
    arms = "".join(
        f'{n}=*) [ "${{{_KEEP}{n}+x}}" = x ] || '
        f"{{ {_KEEP}{n}=${{_bf_l#{n}=}}; export {_KEEP}{n}; }};; "
        for n in SYSTEM_SHELL_EMPTY_VARS
    )
    empty = " ".join(SYSTEM_SHELL_EMPTY_VARS)
    return (
        # An empty kept value is the exec's, not the task's.
        f"for _bf_n in {empty}; do eval "
        f'"[ -n \\"\\${{{_KEEP}$_bf_n-}}\\" ] || unset {_KEEP}$_bf_n"; done\n'
        "if [ -r /proc/1/environ ]; then\n"
        f"while IFS= read -r _bf_l; do case $_bf_l in {arms}esac; done <<_BF_PID1_ENV\n"
        "$(tr '\\000' '\\n' < /proc/1/environ 2>/dev/null)\n"
        "_BF_PID1_ENV\n"
        "fi"
    )


class AgentLaunch(str):
    """An agent command to run as ``sandbox_user`` after root's bootstrap.

    Its string value is a complete POSIX command (keep, pin, drop, restore)
    for an ``exec`` as root; a live-process backend renders it with
    :meth:`render` inside :func:`root_launch_script` instead, with its own
    shell options.
    """

    launch: str
    sandbox_user: str
    root_tools: bool
    protected_tools: bool

    def __new__(
        cls,
        launch: str,
        sandbox_user: str,
        *,
        root_tools: bool = False,
        protected_tools: bool = False,
    ) -> AgentLaunch:
        # The protected kit's bin first (root's alone, or absent: nobody
        # else may create it), then the system folders.
        text = (
            f"{save_task_env()}; {sanitize_root_env(AGENT_LAUNCH_PATH)}; "
            + _priv_drop(
                launch,
                sandbox_user,
                login=False,
                filter_lines=False,
                root_tools=root_tools,
                protected_tools=protected_tools,
            )
        )
        obj = super().__new__(cls, text)
        obj.launch = launch
        obj.sandbox_user = sandbox_user
        obj.root_tools = root_tools
        obj.protected_tools = protected_tools
        return obj

    def render(self, *, login: bool, filter_lines: bool) -> str:
        return _priv_drop(
            self.launch,
            self.sandbox_user,
            login=login,
            filter_lines=filter_lines,
            root_tools=self.root_tools,
            protected_tools=self.protected_tools,
        )


def _task_command(
    command: str, *, login: bool, filter_lines: bool, privileged: bool = False
) -> str:
    if filter_lines:
        from benchflow.sandbox.process._acp_lines import filter_command

        command = filter_command(command)
    shell = f"{LAUNCH_SHELL} -l" if login else LAUNCH_SHELL
    if privileged:
        # The credential-bearing launch's own shell reads no startup file
        # (integ CONF-1, cli-other-blockers); the task's BASH_ENV stays in its
        # environment for the agent's own shells.
        shell += " -p"
    return f"{restore_task_env()}; exec {shell} -c {shlex.quote(command)}"


def _priv_drop(
    launch: str,
    user: str,
    *,
    login: bool,
    filter_lines: bool,
    root_tools: bool = False,
    protected_tools: bool = False,
) -> str:
    # Imported here: _base imports benchflow.task, which imports lockdown.
    from benchflow.sandbox.lockdown import (
        _SU_EXEC,
        _SU_NAMES_ENV,
        _SU_PATH_ENV,
        _su_launch,
    )

    # Both setpriv (util-linux) and the environment-preserving su (Alpine,
    # BusyBox, PY8 SU) keep the environment, so the kept values reach the
    # inner shell on either path.
    inner = f"export HOME=/home/{user}; " + _task_command(
        launch, login=login, filter_lines=filter_lines
    )
    protected_tools = protected_tools or root_tools

    # Root audits supply S2's custody resolver (``tool``); every root-side
    # executable then uses its checked absolute path, the su bookkeeping
    # included. Otherwise the pinned route PATH (the kit's, then the system's).
    def tool(name: str) -> str:
        if root_tools and name in {"setpriv", "grep", "bash", "su"}:
            return f'"$bf_{name}"'
        return f'"$(tool {name})"' if protected_tools else name

    su = _su_launch(user, inner, su_command=tool("su"))
    su_exec = _SU_EXEC
    if protected_tools:
        names = f"{tool('env')} | {tool('sed')} -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)=.*/\\1/p'"
        su_exec = (
            f"bf_su_shell=$(tool bash) || bf_su_shell=$(tool sh) || exit 7; "
            f"{_SU_NAMES_ENV}=\" $({names} | {tool('tr')} '\\n' ' ')\" "
            f'{_SU_PATH_ENV}="$PATH" exec'
        )
    # S2-CONF-9 (integ): no_new_privs and the firewall capabilities dropped
    # before any agent code or login profile runs, on both paths; a setpriv
    # that cannot remove the bounding capabilities refuses the launch.
    restrictions = (
        "--no-new-privs --bounding-set=-net_admin,-net_raw"
        " --inh-caps=-all --ambient-caps=-all"
    )
    setpriv, grep = tool("setpriv"), tool("grep")
    if root_tools:
        bash = tool("bash")
    elif protected_tools:
        bash = f"{tool('bash')} --noprofile --norc"
    else:
        bash = _ROOT_BASH
    require_su = (
        ' [ -n "$bf_su" ] || { printf "BF fwpower-error tool=su\\n"; exit 126; };'
        if root_tools
        else ""
    )
    return (
        f"if {setpriv} --help 2>&1 | {grep} -q reuid; then"
        f" exec {setpriv} --reuid={user} --regid={user} --init-groups {restrictions}"
        f" -- {bash} -c {shlex.quote(inner)};"
        f" elif {setpriv} --help 2>&1 | {grep} -q bounding-set; then"
        f"{require_su} {su_exec} {setpriv} {restrictions} -- {su};"
        " else echo 'Agent launch requires util-linux setpriv with"
        " no_new_privs and capability bounding-set support' >&2; exit 1;"
        " fi"
    )


def root_launch_script(
    command: str,
    *,
    env_path: str | None = None,
    env_stdin: bool = False,
    cwd: str | None = None,
    login: bool = False,
    filter_lines: bool = False,
    loader_cleared: bool = False,
    path: str = SYSTEM_PATH,
) -> str:
    """The text root's first shell runs to start ``command``.

    ``command`` is an :class:`AgentLaunch` (dropped to its user) or a plain
    command for an agent that runs as root (no sandbox user; the task's
    environment is put back before it starts). ``env_path`` is the staged
    env file, sourced and removed here. ``env_stdin``: the env is instead the
    first line of the exec's stdin (integ's live transport: no container
    pathname and no argv holds a value), applied the same way.
    ``loader_cleared``: the backend's exec set
    :data:`SYSTEM_SHELL_EMPTY_VARS` empty (the text then has a here-document,
    so it must not be typed into a terminal). *path* is the
    launch route's PATH (:attr:`ExecRoute.pin`: the protected kit's once the
    sandbox has one); root's bootstrap pins it and its traps' ``rm`` comes
    from it.
    """
    parts = [save_task_env(), sanitize_root_env(path)]
    if loader_cleared:
        parts.append(_keep_from_pid1())
    if env_path:
        q = shlex.quote(env_path)
        rm = _route_rm(path)
        parts += [
            # A shell that treats a failed ``.`` as fatal still removes it.
            f"trap {shlex.quote(f'{rm} -f {q}')} EXIT",
            # PATH is unset while the file is sourced (only builtins run), so
            # a PATH the file sets is seen as set even when it equals the
            # pinned one (PT-C8-5).
            "unset PATH",
            f"if . {q}; then :; else {rm} -f {q}; exit 1; fi",
            # What the file set is the task's.
            f'if [ "${{PATH+x}}" = x ]; then {_KEEP}PATH=$PATH; export {_KEEP}PATH; fi',
            _keep_each_set(" ".join(SYSTEM_SHELL_UNSET_VARS)),
            sanitize_root_env(path),
            f"rm -f {q}",
            "trap - EXIT",
        ]
    elif env_stdin:
        parts += [
            # Exactly one line, read before anything else reads stdin; later
            # bytes (ACP) are left intact. integ's spelling;
            # the decoder is the kit's once its RECORD
            # exists, never the image's, else the system one by absolute path
            # (review A).
            "IFS= read -r __bf_live_env || exit 125",
            f"if [ -f {KIT_RECORD} ]; then __bf_b64={KIT_BIN}/base64; "
            "else __bf_b64=/usr/bin/base64; fi",
            '__bf_live_env=$(printf \'%s\' "$__bf_live_env" | "$__bf_b64" -d)'
            " || exit 125",
            # PATH is unset while the values are applied (only builtins run),
            # as for a staged file (PT-C8-5).
            "unset PATH",
            'eval "unset __bf_live_env __bf_b64; $__bf_live_env" || exit 125',
            f'if [ "${{PATH+x}}" = x ]; then {_KEEP}PATH=$PATH; export {_KEEP}PATH; fi',
            _keep_each_set(" ".join(SYSTEM_SHELL_UNSET_VARS)),
            sanitize_root_env(path),
        ]
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)} || exit 1")
    if isinstance(command, AgentLaunch):
        parts.append(command.render(login=login, filter_lines=filter_lines))
    else:
        parts.append(
            _task_command(
                command,
                login=login,
                filter_lines=filter_lines,
                privileged=env_stdin,
            )
        )
    # One line unless the here-document needs more: a terminal backend types
    # it (a newline there would print a continuation prompt).
    return ("\n" if loader_cleared else "; ").join(parts)


def launch_bash(route: Any = None) -> str:
    """The bash root's launch starts by absolute path: the route's
    (:attr:`ExecRoute.bash`, the protected kit's once the sandbox has one)
    when it is a bash, else :data:`LAUNCH_SHELL`."""
    bash = getattr(route, "bash", None)
    return bash if isinstance(bash, str) and bash.endswith("/bash") else LAUNCH_SHELL


def launch_path(route: Any = None) -> str:
    """The PATH root's launch bootstrap pins: the route's, or
    :data:`SYSTEM_PATH` without one."""
    pin = getattr(route, "pin", None)
    return pin if isinstance(pin, str) and pin else SYSTEM_PATH


def root_launch_argv(script: str, bash: str = LAUNCH_SHELL) -> list[str]:
    """argv for an exec that starts root's shell itself (*bash*: the
    route's, :func:`launch_bash`)."""
    return [bash, "--noprofile", "--norc", "-c", script]


def typed_root_prelude(path: str = SYSTEM_PATH) -> str:
    """One line for a terminal root's shell already runs: keep and pin first."""
    return f"{save_task_env()}; {sanitize_root_env(path)}"


async def launch_route(sandbox: Any, service: str = "main") -> Any:
    """Root's route for a live launch in *sandbox* (C8): the protected kit's
    shell, bash and PATH once the sandbox has one. None for a stand-in
    without a real ``exec_route`` (the launch then keeps the system's)."""
    exec_route = getattr(sandbox, "exec_route", None)
    if not inspect.iscoroutinefunction(exec_route):
        return None
    return await exec_route("root", system_shell=True, service=service)


#: The values an interactive root terminal is created with on the protected
#: kit: HOME the kit's empty one and every startup hook empty, so the shell a
#: provider starts reads no file and runs no command of the image's or the
#: agent's before BenchFlow's first line (PY8-PT-R8-8).
def terminal_env(route: Any = None) -> dict[str, str]:
    from benchflow._runtime_tree import ROOT_HOME

    if not getattr(route, "kit", False):
        return {}
    return {
        **dict.fromkeys(SYSTEM_SHELL_EMPTY_VARS, ""),
        "HOME": ROOT_HOME,
        "PROMPT_COMMAND": "",
        "PS0": "",
        "PS1": "$ ",
        "PS2": "> ",
        "PS4": "+ ",
        "INPUTRC": "/dev/null",
        "HISTFILE": "/dev/null",
    }


_TERMINAL_STARTUP_FILES = (
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".profile",
    ".inputrc",
    ".bash_logout",
    ".ssh/rc",
    ".ssh/environment",
)


async def protect_root_terminal_home(sandbox: Any, what: str) -> None:
    """Give a provider-made root terminal (Daytona's SSH session) a HOME
    that is root's alone before it opens (PT-C8-R9-3).

    sshd and the shell it starts read root's HOME from the account
    database, not from BenchFlow's route. When that HOME is not root's
    alone (a task whose workspace is ``/root``), root's entry is pointed at
    the kit's empty, protected HOME (:data:`~benchflow._runtime_tree.ROOT_HOME`),
    so the session reads no startup file the agent could write. The account
    file must itself be root's alone, or the terminal is refused. Runs only
    on the protected kit; a root agent's terminal is its own.
    """
    from benchflow._root_only import ROOT_CHAIN_FUNCS
    from benchflow._runtime_tree import ROOT_HOME

    exec_route = getattr(sandbox, "exec_route", None)
    if not inspect.iscoroutinefunction(exec_route):
        return
    route = await exec_route("root")
    if not getattr(route, "kit", False):
        return
    home = shlex.quote(ROOT_HOME)
    script = (
        f"{ROOT_CHAIN_FUNCS}"
        'h=$(getent passwd root | cut -d: -f6) && [ -n "$h" ] || exit 1; '
        'bfc_chain "$h" 0 && exit 0; '
        "bfc_chain /etc/passwd 0 || { echo /etc/passwd; exit 1; }; "
        f"bfc_chain {home} 0 || {{ echo {home}; exit 1; }}; "
        f"sed -i -e 's|^\\(root:[^:]*:0:[^:]*:[^:]*:\\)[^:]*:|\\1'{home}':|' "
        "/etc/passwd || exit 1; "
        '[ "$(getent passwd root | cut -d: -f6)" = '
        f"{home} ] || exit 1; echo moved"
    )
    result = await exec_raw(sandbox.exec, script, user="root", timeout_sec=60)
    if getattr(result, "return_code", 1) != 0:
        found = (getattr(result, "stdout", "") or "").strip() or "root's HOME"
        raise RuntimeError(
            f"{what} starts a root shell from root's HOME, which another user "
            f"can change, and BenchFlow could not give it a protected one "
            f"({found} is not root's alone), so it will not start the agent "
            "through it"
        )


async def require_protected_root_terminal(sandbox: Any, what: str) -> None:
    """Refuse a root terminal the provider creates with settings BenchFlow
    cannot choose (AgentCore's shell, Daytona's SSH session) unless what it
    reads at start is root's alone (PY8-PT-R8-8): root's HOME and each
    startup file in it, /etc's profile files, and no prompt hook in the
    image's environment. Runs only on the protected kit (a sandbox user
    exists); a root agent's own terminal is its own.
    """
    from benchflow._root_only import ROOT_CHAIN_FUNCS
    from benchflow.sandbox._task_route import as_task_command

    exec_route = getattr(sandbox, "exec_route", None)
    if not inspect.iscoroutinefunction(exec_route):
        return
    route = await exec_route("root")
    if not getattr(route, "kit", False):
        return
    files = " ".join(_TERMINAL_STARTUP_FILES)
    check = (
        f"{ROOT_CHAIN_FUNCS}"
        'h=$(getent passwd root | cut -d: -f6) && [ -n "$h" ] || exit 1; '
        'bf_ok() { bfc_chain "$1" 0; s=$?; [ "$s" = 0 ] || [ "$s" = 2 ]; }; '
        'bfc_chain "$h" 0 || { echo "root\'s HOME $h"; exit 1; }; '
        f'for f in {files}; do bf_ok "$h/$f" || {{ echo "$h/$f"; exit 1; }}; done; '
        "for f in /etc/profile /etc/profile.d /etc/bash.bashrc /etc/bashrc "
        '/etc/environment /etc/inputrc; do bf_ok "$f" || { echo "$f"; exit 1; }; done; '
        'for f in /etc/profile.d/*; do [ -e "$f" ] || continue; '
        'bf_ok "$f" || { echo "$f"; exit 1; }; done'
    )
    result = await exec_raw(sandbox.exec, check, user="root", timeout_sec=60)
    hooks = None
    with as_task_command(sandbox):
        hooks = await exec_raw(
            sandbox.exec,
            'printf "%s|%s|%s|%s" "${PROMPT_COMMAND-}" "${PS0-}" "${PS1-}" "${PS4-}"',
            user="root",
            timeout_sec=30,
        )
    hook_text = getattr(hooks, "stdout", "") or ""
    bad_hooks = (
        hook_text.split("|", 1)[0].strip() or "$(" in hook_text or "`" in hook_text
    )
    if getattr(result, "return_code", 1) != 0 or bad_hooks:
        found = (getattr(result, "stdout", "") or "").strip() or "a prompt hook"
        raise RuntimeError(
            f"{what} starts a root shell that reads {found} before BenchFlow's "
            "first line, and another user can change it, so BenchFlow will not "
            "start the agent through it; use the Docker or Daytona PTY transport, "
            "or keep root's HOME and /etc profile files out of the workspace"
        )
