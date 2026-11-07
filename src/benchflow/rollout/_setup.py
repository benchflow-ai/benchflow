"""Environment, agent, oracle, verify and prompt setup helpers for the rollout.

These free functions cover the *input* side of the 5-phase lifecycle: reading
the task instruction, applying the no-web policy, resolving the agent cwd,
initialising the rollout directory tree, resolving prompts, starting the
sandbox + uploading task files, running oracle mode, publishing the trajectory
for the verifier, and running the verifier itself.

Split out of ``rollout.py`` for cohesion; every name is re-exported from
:mod:`benchflow.rollout` so existing imports — ``_init_rollout``,
``_resolve_prompts``, ``_run_oracle``, ``_start_env_and_upload``,
``_verify_rollout`` (``sdk.py``), ``_resolve_agent_cwd`` /
``_start_env_and_upload`` / ``_verify_rollout`` (``task/acceptance_live.py``) —
keep resolving unchanged.

Note on patching: callers that patch ``benchflow.rollout.default_rollout_planes``
always thread an explicit ``planes`` into these helpers (via ``self._planes``),
so the ``planes or default_rollout_planes()`` fallback here is never the patched
seam — it only fires for direct, unpatched calls.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import inspect
import json
import logging
import os
import re
import shlex
import stat
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from benchflow._utils import kill_record as kill_record_module
from benchflow._utils import timing_spans
from benchflow._utils.reachable_write import ReachablePath, destination
from benchflow._utils.scoring import NO_USABLE_REWARD
from benchflow._utils.text import describe_exception
from benchflow._utils.trial_warnings import report_trial_warning
from benchflow._utils.trusted_path import (
    VerifierPathEvidenceError,
    recording_trusted_path,
    refused_programs,
    refused_programs_cmd,
)
from benchflow._verifier_env import VERIFIER_ATTEMPT_ENV
from benchflow.contracts import RolloutPlanes, default_rollout_planes
from benchflow.diagnostics import (
    Diagnostic,
    RewardOutOfRangeDiagnostic,
    VerifierTimeoutDiagnostic,
)
from benchflow.environment.manifest import EnvironmentManifest, resolve_manifest_image
from benchflow.rewards.validation import (
    REWARD_OUT_OF_RANGE,
    RewardOutOfRange,
    declared_reward_range,
    reward_lenient_from_env,
    validate_reward_map,
)
from benchflow.rollout._results import _DIAG_TRUNCATE
from benchflow.sandbox import _inert_bootstrap
from benchflow.sandbox._compose import docker_daemon_unreachable_reason
from benchflow.sandbox._exec_merge import kept_default_identity
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._hooks import sandbox_hook
from benchflow.sandbox._in_order import in_order
from benchflow.sandbox._locked_path_validation import _validate_locked_path
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.egress_denylist import (
    EgressDenylist,
    agent_network_sandbox_config,
    egress_denylist_for,
    task_service_policy,
)
from benchflow.sandbox.oracle_egress import (
    arm_oracle_egress,
    disarm_oracle_egress,
    join_group_prefix,
)
from benchflow.sandbox.protocol import (
    EnvironmentStartTimeoutError,
    ExecTimeoutError,
    SandboxStartupError,
)
from benchflow.task.verifier_errors import (
    PluginGuardLoadError,
    PluginGuardMarkersUnreadableError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    RewardFileParseError,
    VerifierExecutionError,
    VerifierInfrastructureError,
    VerifierOutputParseError,
    VerifierSetupError,
    VerifierStartUnconfirmedError,
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_text,
)
from benchflow.trust.state import TRUSTED_TRANSFER, IngressKind, admit

logger = logging.getLogger(__name__)

_DISALLOW_WEB_TOOLS_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"


def _task_disallows_internet(task: Any) -> bool:
    """Return True when task config requests no internet for the agent task.

    That is an offline sandbox, or an online sandbox whose ``[agent]`` network
    is ``no-network`` (task.md's ``[agent] network = "none"``): the agent's
    phase runs offline, its model calls going through the sandbox-local proxy
    and its uid's other traffic refused, while installs and the verifier keep
    the sandbox's network. Before, the agent's ``no-network`` was accepted and
    nothing enforced it, so the agent had the sandbox's full network.
    """
    env_config = getattr(getattr(task, "config", None), "sandbox", None)
    if getattr(env_config, "allow_internet", True) is False:
        return True
    return _task_agent_offline(task)


def _task_agent_offline(task: Any) -> bool:
    """True when the sandbox is online and the task's ``[agent]`` network is
    ``no-network``.

    Only the agent user's uid firewall enforces it there, so an agent running
    as root (no sandbox user) cannot be held to it.
    """
    config = getattr(task, "config", None)
    if getattr(getattr(config, "sandbox", None), "allow_internet", True) is False:
        return False
    agent_mode = getattr(getattr(config, "agent", None), "network_mode", None)
    return str(getattr(agent_mode, "value", agent_mode)) == "no-network"


def _task_egress_denylist(task: Any) -> EgressDenylist | None:
    """Return the egress policy (denylist or allowlist) the task declares, if any."""
    config = getattr(task, "config", None)
    if config is None:
        return None
    return egress_denylist_for(agent_network_sandbox_config(config))


def _read_task_instruction(task_path: Path) -> str:
    """Read the agent-facing instruction from legacy files or ``task.md``."""
    document_path = task_path / "task.md"
    if document_path.exists():
        from benchflow.task.document import TaskDocument

        return TaskDocument.from_path(document_path).instruction.strip()

    instruction_path = task_path / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text().strip()
    from benchflow.task import Task

    # Task names the task formats when neither prompt file exists, and loads a
    # task.toml multi-step task (no root prompt) so the launch gate refuses its
    # ``steps`` instead of this function failing on a missing file.
    return Task(task_path).instruction.strip()


def _environment_uses_prebuilt_image(
    env_config: object | None, environment_manifest: EnvironmentManifest | None
) -> bool:
    """Return True when sandbox startup will skip the task Dockerfile build."""
    if env_config is not None and getattr(env_config, "docker_image", None):
        return True
    if environment_manifest is None:
        return False
    return bool(resolve_manifest_image(environment_manifest))


def _apply_web_policy(agent_env: dict[str, str], *, disallow: bool) -> dict[str, str]:
    """Inject BenchFlow's no-web policy marker into agent env when requested."""
    if not disallow:
        return agent_env
    return {**agent_env, _DISALLOW_WEB_TOOLS_ENV: "1"}


def _agent_launch_with_web_policy(
    agent: str, *, disallow: bool, planes: RolloutPlanes | None = None
) -> str:
    """Return launch command, appending the agent's no-web launch knob if any."""
    return (planes or default_rollout_planes()).agent_launch(
        agent, disallow_web_tools=disallow
    )


# Package runners whose *next* token is a subcommand (``uv run <agent>``,
# ``npx <agent>``), not the agent binary — skip both.
_PACKAGE_RUNNERS = frozenset({"uv", "uvx", "npx", "npm", "pnpm", "yarn", "pipx"})
# Interpreters that launch an agent but never *identify* it: keying a
# ``pkill -f`` pattern on one of these reaps every interpreter process in the
# sandbox, not just the agent. (e.g. ``pythonX.Y`` is matched by the
# ``python`` prefix in :func:`_is_generic_interpreter`.)
_GENERIC_INTERPRETERS = (
    frozenset(
        {
            "python",
            "pypy",
            "node",
            "nodejs",
            "deno",
            "bun",
            "ruby",
            "perl",
            "sh",
            "bash",
            "dash",
            "env",
        }
    )
    | _PACKAGE_RUNNERS
)
# Subcommands consumed by a package runner before the agent binary appears.
_RUNNER_SUBCOMMANDS = frozenset({"run", "tool", "exec", "x"})
# A leading ``FOO=bar`` environment-variable assignment (e.g. harvey-lab's
# ``HARVEY_LABS_ROOT=/opt/harvey-labs ... python <shim>``).
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
# Shell operators that separate commands; the agent invocation is the final
# command in a ``setup && … && <agent>`` launch (openhands).
_SHELL_SEP_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


def _is_generic_interpreter(basename: str) -> bool:
    return basename in _GENERIC_INTERPRETERS or basename.startswith("python")


def _agent_process_kill_pattern(agent_launch: str) -> str | None:
    """Return a ``pkill -f`` pattern that targets the *agent* process only.

    The pattern must identify the agent — never the interpreter that launches
    it. Several agents launch as ``<venv>/bin/python <shim>`` (deepagents,
    harvey-lab); naively keying on the first token then yields ``python``, and
    ``pkill -f python`` reaps **every** Python process in the sandbox. That
    includes Environment-plane services — mock APIs run as console scripts via
    their ``#!/usr/bin/python`` shebang, so their argv is
    ``/usr/bin/python /usr/local/bin/<svc> …`` — and Python verifiers. The
    service is then dead when the verifier reads its live state, scoring 0.0
    on an otherwise-correct rollout (BF-10).

    Resolution:

    1. Reduce to the **last shell-command segment** — for a
       ``export … && mkdir … && <agent>`` launch (openhands) the agent is the
       final command, not the leading ``export`` builtin.
    2. Walk that segment and key on the agent's own binary/shim token, skipping
       ``FOO=bar`` env assignments, generic interpreters, package-runner
       subcommands (``uv run <agent>``), and flags.

    Returns ``None`` when nothing specific enough is found — better to skip the
    cleanup pkill than to fire a sandbox-wide one.
    """
    segments = [s for s in _SHELL_SEP_RE.split(agent_launch.strip()) if s.strip()]
    if not segments:
        return None

    after_runner = False
    for token in segments[-1].split():
        if _ENV_ASSIGN_RE.match(token):  # FOO=bar prefix
            continue
        if token.startswith("-"):  # a flag (e.g. python -m, gemini --acp)
            continue
        basename = PurePosixPath(token).name
        if not basename:
            continue
        if _is_generic_interpreter(basename):  # too broad to pkill on
            after_runner = basename in _PACKAGE_RUNNERS
            continue
        if after_runner and basename in _RUNNER_SUBCOMMANDS:  # `uv run` etc.
            after_runner = False
            continue
        return rf"(^|[ /]){re.escape(basename)}( |$)"
    return None


def _configured_task_workdir(task: Any) -> str | None:
    """Return the task-declared sandbox workdir, if any."""

    env_config = getattr(getattr(task, "config", None), "sandbox", None)
    value = getattr(env_config, "workdir", None)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _validate_agent_workdir(workdir: str) -> None:
    path = PurePosixPath(workdir)
    if not path.is_absolute() or path == PurePosixPath("/"):
        raise ValueError("sandbox.workdir must be an absolute non-root path")


def trust_workspace_repos_cmd(workspace: str) -> str:
    """Shell that adds the workspace and every git repository in it, and
    nothing else, to the running user's ``safe.directory`` list.

    The sandbox-user grant chowns the workspace to the agent user, so a
    repository the image built as root (TB2's ``git-leak-recovery`` keeps one
    at ``/app/repo``) is no longer owned by the root oracle that runs in it,
    and git refuses with "detected dubious ownership" (DF-p3-eval-runner-2).
    The owner changed only because BenchFlow switched users, so BenchFlow
    trusts exactly those paths again; the image and its system git config
    are unchanged. ``safe.directory`` does not cover subfolders, and ``/*``
    needs git 2.46, so each repository (a ``.git`` folder or file, a bare
    ``*.git`` folder, or a bare folder of any name, known by its ``HEAD``
    file beside ``objects/`` and ``refs/``) is listed. Links are not
    followed. Paths outside the resolved workspace are never added, and a
    workspace of ``/`` adds only itself. The verifier
    reads no global git config but its own (``GIT_CONFIG_GLOBAL`` is
    ``/dev/null``, or for a root verifier an empty file made after the agent
    stopped) and runs after the workspace's folders are chowned back to
    root, so it is unaffected. A task's collect hooks get no such trust: they
    run as the agent's account (``_collect_hooks.agent_account``).
    """
    q = shlex.quote(workspace)
    return (
        "command -v git >/dev/null 2>&1 || exit 0; "
        f"[ -d {q} ] || exit 0; "
        f"ws=$(cd -P {q} && pwd) || exit 0; "
        "trust() { git config --global --get-all safe.directory 2>/dev/null "
        '| grep -qxF -- "$1" || git config --global --add safe.directory "$1"; }; '
        f"trust {q}; "
        '[ "$ws" = / ] && exit 0; '
        'trust "$ws"; '
        'find "$ws" -xdev \\( -name .git -o -type d -name "*.git" \\) -prune '
        "-print -o -type f -name HEAD -print 2>/dev/null "
        "| while IFS= read -r d; do "
        'case "$d" in */.git) d=${d%/.git} ;; '
        "*/HEAD) d=${d%/HEAD}; "
        '[ -d "$d/objects" ] && [ -d "$d/refs" ] || continue ;; '
        "esac; "
        'case "$d" in "$ws"/*) trust "$d" ;; esac; '
        "done; exit 0"
    )


async def trust_workspace_repos(env: Any, workspace: str, task: Any) -> None:
    """Run :func:`trust_workspace_repos_cmd` for every user solve.sh may run
    as: root, and the task's string ``[agent].user`` when it names another.

    The trust rule (DX-G), on the task route (C8): the global config
    written is the one solve.sh's git reads. Root's plumbing route starts
    with BenchFlow's own empty HOME, so trust written there never reached
    the oracle (PT-integ round 3: TB2 fix-git scored 0 on PT's route, 1 on
    integ).
    """
    users = ["root"]
    declared = getattr(
        getattr(getattr(task, "config", None), "agent", None), "user", None
    )
    if isinstance(declared, str) and declared not in ("", "root"):
        users.append(declared)
    for user in users:
        try:
            with as_task_command(env):
                await exec_raw(
                    env.exec,
                    trust_workspace_repos_cmd(workspace),
                    user=user,
                    timeout_sec=60,
                )
        except Exception:
            # A huge workspace or a failed exec: the oracle still runs, as it
            # did when this step ignored its errors.
            logger.warning(
                "Listing git repositories in %s for %s failed; git may refuse "
                "them as 'dubious ownership'",
                workspace,
                user,
                exc_info=True,
            )


async def _resolve_agent_cwd(env: Any, task: Any) -> str:
    """Resolve and materialize the workspace path used by agents and verifiers."""

    configured = _configured_task_workdir(task)
    if configured is None:
        # The start probe already ran this ``pwd`` as the default user in
        # this sandbox generation (PY8 U12); otherwise run it.
        kept = kept_default_identity(env)
        if kept is not None and kept.pwd is not None:
            probed = kept.pwd.strip()
        else:
            cwd_result = await exec_raw(env.exec, "pwd", timeout_sec=10)
            probed = (cwd_result.stdout or "").strip()
        if probed and probed != "/":
            return probed
        fallback = "/root"
        result = await exec_raw(
            env.exec,
            f"mkdir -p {fallback} && cd {fallback} && pwd",
            user="root",
            timeout_sec=10,
        )
        return (getattr(result, "stdout", "") or "").strip() or fallback

    _validate_agent_workdir(configured)
    quoted = shlex.quote(configured)
    result = await exec_raw(
        env.exec,
        f"mkdir -p {quoted} && cd {quoted} && pwd",
        user="root",
        timeout_sec=10,
    )
    return_code = getattr(result, "return_code", getattr(result, "exit_code", 0))
    if isinstance(return_code, int) and return_code != 0:
        stderr = (diagnostic_result(result).stderr.render() or "").strip()
        raise RuntimeError(
            f"failed to prepare sandbox.workdir {configured!r}: {stderr}"
        )
    return (getattr(result, "stdout", "") or "").strip() or configured


async def _ensure_sandbox_dir(
    env: Any, path: str | Path, sandbox_user: str | None = None
) -> None:
    """Create a sandbox directory and optionally make it writable by the agent."""
    q_path = shlex.quote(str(path))
    cmd = f"mkdir -p {q_path}"
    if sandbox_user:
        q_user = shlex.quote(sandbox_user)
        cmd += f" && chown -R {q_user}:{q_user} {q_path}"
    result = await exec_raw(env.exec, cmd, timeout_sec=10)
    if result.return_code != 0:
        raise RuntimeError(
            f"Failed to create sandbox directory {path}: "
            f"{diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render()}"
        )


# A temp folder entry is one plain name: never a path, never "." or "..".
_TEMP_DIR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _agent_temp_dirs_cmd(
    names: list[str], sandbox_user: str, *, locked_paths: list[str] | None = None
) -> str:
    """The root shell step behind :func:`_prepare_agent_temp_dirs`.

    Exit 3 names a link or non-folder at a temp folder's path, exit 4 a
    folder it could not make or re-own, exit 5 a folder that holds what
    another user could have put there. A missing temp folder is made only
    where no other user can swap the path under root.

    In a scene, earlier roles have already run as the sandbox user, and their
    background processes can still be alive between roles. So root
    re-owns a folder only when no other user could have written it or any
    folder in it, nor any folder on the path to it: everything inside was
    put there by root (the install), and nothing can be renamed or swapped
    for a link during the walk. The walk is depth-first and never follows a
    link, so a folder becomes the agent's only after everything in it.
    """
    for name in names:
        if not _TEMP_DIR_NAME.fullmatch(name):
            raise ValueError(f"temp_dirs entry {name!r} is not a plain folder name")
    user = shlex.quote(sandbox_user)
    owner = shlex.quote(f"{sandbox_user}:{sandbox_user}")
    # Root's alone: owned by root, and no group or other write bit unless
    # sticky (then others cannot rename or remove root's entries).
    root_only = r"-user 0 \( -perm -1000 -o ! -perm -020 ! -perm -002 \)"
    steps = [
        # GNU readlink supports missing components with -m; BusyBox has -f.
        # Resolve the nearest existing ancestor there and append only the
        # missing suffix, never guessing through an unresolved symlink.
        r"""bf_temp_canonical() {
readlink -m -- "$1" 2>/dev/null && return 0
bf_target=$1; bf_suffix=
while [ ! -e "$bf_target" ] && [ ! -L "$bf_target" ]; do
bf_suffix="/$(basename "$bf_target")$bf_suffix"
bf_target=$(dirname "$bf_target")
done
bf_resolved=$(readlink -f -- "$bf_target") || return 1
printf '%s%s\n' "${bf_resolved%/}" "$bf_suffix"
}""",
        # Match policy components before expanding existing names: a missing
        # temp directory must not be created inside a locked glob. Unlike a
        # whole-string case pattern, '*' here never consumes a slash.
        r"""bf_temp_policy_overlap() {
bf_a=$1; bf_b=$2
while [ "${bf_a#/}" != "$bf_a" ]; do bf_a=${bf_a#/}; done
while [ "${bf_b#/}" != "$bf_b" ]; do bf_b=${bf_b#/}; done
if [ -z "$bf_a" ] || [ -z "$bf_b" ]; then return 0; fi
while :; do
bf_ac=${bf_a%%/*}; bf_bc=${bf_b%%/*}
case "$bf_ac" in $bf_bc) ;; *) return 1;; esac
if [ "$bf_a" = "$bf_ac" ] || [ "$bf_b" = "$bf_bc" ]; then return 0; fi
bf_a=${bf_a#*/}; bf_b=${bf_b#*/}
done
}""",
        # Bind mounts keep different canonical names for the same objects.
        # Compare each path's backing device/root from the kernel mount table,
        # including mounts below either tree. This also catches an alias of a
        # protected ancestor or descendant without walking the protected tree.
        # Unreadable mount metadata is unknown, never permission to chown.
        r"""bf_temp_mount_overlap() {
BF_TEMP_A="$1" BF_TEMP_B="$2" awk '
function unescape(p) {
    gsub(/\\040/, " ", p); gsub(/\\011/, "\t", p)
    gsub(/\\012/, "\n", p); gsub(/\\134/, "\\", p)
    return p
}
function within(p, root) {
    return root == "/" || p == root || index(p, root "/") == 1
}
function overlap(a, b, aa, bb, na, nb, k, c, j, pattern) {
    if (a == "/" || b == "/") return 1
    na = split(a, aa, "/"); nb = split(b, bb, "/")
    for (k = 2; k <= na && k <= nb; k++) {
        pattern = ""
        for (j = 1; j <= length(bb[k]); j++) {
            c = substr(bb[k], j, 1)
            if (c == "*") pattern = pattern ".*"
            else if (c == "?") pattern = pattern "."
            else pattern = pattern (index(".\\+[](){}^$|", c) ? "\\" : "") c
        }
        if (aa[k] !~ ("^" pattern "$")) return 0
    }
    return 1
}
function backing(p, i, suffix, result) {
    suffix = substr(p, length(mp[i]) + 1)
    if (mp[i] == "/") suffix = p
    result = (base[i] == "/" ? "" : base[i]) suffix
    return result == "" ? "/" : result
}
NF >= 6 {
    n++; dev[n] = $3; base[n] = unescape($4); mp[n] = unescape($5)
}
END {
    a = ENVIRON["BF_TEMP_A"]; b = ENVIRON["BF_TEMP_B"]
    for (i = 1; i <= n; i++) {
        if (within(a, mp[i]) && (!ai || length(mp[i]) >= length(mp[ai]))) ai = i
        if (within(b, mp[i]) && (!bi || length(mp[i]) >= length(mp[bi]))) bi = i
    }
    if (!ai || !bi) exit 2
    ad[1] = dev[ai]; ap[1] = backing(a, ai); an = 1
    bd[1] = dev[bi]; bp[1] = backing(b, bi); bn = 1
    for (i = 1; i <= n; i++) {
        if (within(mp[i], a)) { an++; ad[an] = dev[i]; ap[an] = base[i] }
        if (within(mp[i], b)) { bn++; bd[bn] = dev[i]; bp[bn] = base[i] }
    }
    for (i = 1; i <= an; i++) for (j = 1; j <= bn; j++)
        if (ad[i] == bd[j] && overlap(ap[i], bp[j])) exit 0
    exit 1
}' < /proc/self/mountinfo
}""",
        # The temp folder the agent's tool uses (os.tmpdir() reads TMPDIR).
        't="${TMPDIR:-/tmp}"; case "$t" in /*) ;; *) t=/tmp ;; esac',
        't=$(cd "$t" 2>/dev/null && pwd -P) || exit 0',
        # s=1 when every folder from / to $t is root's alone.
        's=1; p="$t"; while :; do '
        f'[ -n "$(find "$p" -prune {root_only} -print)" ] || {{ s=0; break; }}; '
        '[ "$p" = / ] && break; p=$(dirname "$p"); done',
    ]
    for name in names:
        # Check both lexical and canonical paths, before any mutation. A
        # protected parent or child is an overlap too; relocking afterwards
        # leaves a read window for an earlier role's background processes.
        policy_checks = []
        for locked in locked_paths or []:
            _validate_locked_path(locked)
            policy_checks.append(
                f'if bf_temp_policy_overlap "$bf_d_canonical" {shlex.quote(locked)}; '
                'then echo "$d"; exit 6; fi; '
                # Same validated glob expansion as lockdown_paths.
                f"for l in {locked}; do "
                "for pair in lexical canonical; do "
                'if [ "$pair" = canonical ]; then '
                'a=$(bf_temp_canonical "$d") && b=$(bf_temp_canonical "$l") || exit 6; '
                'else a="$d"; b="$l"; fi; a=${a%/}; b=${b%/}; '
                'case "$a/" in "$b/"*) echo "$d"; exit 6;; esac; '
                'case "$b/" in "$a/"*) echo "$d"; exit 6;; esac; done; '
                # File identity is authoritative even across mount names.
                'if [ -e "$d" ] && [ -e "$l" ]; then '
                'di=$(stat -Lc "%d:%i" -- "$d") && '
                'li=$(stat -Lc "%d:%i" -- "$l") || exit 6; '
                '[ "$di" != "$li" ] || { echo "$d"; exit 6; }; fi; '
                'if bf_temp_mount_overlap "$a" "$b"; then echo "$d"; exit 6; '
                'else rc=$?; [ "$rc" = 1 ] || exit 6; fi; done'
            )
        steps += [
            f'd="$t"/{shlex.quote(name)}',
            *(
                ['bf_d_canonical=$(bf_temp_canonical "$d") || exit 6']
                if policy_checks
                else []
            ),
            *policy_checks,
            'if [ -L "$d" ] || { [ -e "$d" ] && [ ! -d "$d" ]; }; then '
            'echo "$d"; exit 3; fi',
            # Missing: a fresh folder for the agent. Under a path another
            # user could swap, root makes nothing; the tool makes its own.
            'if [ ! -d "$d" ]; then '
            f'if [ "$s" = 1 ]; then mkdir "$d" && chown -h {owner} "$d" || exit 4; fi; '
            # Anchor the existing directory as cwd before inspecting it.
            # stat without -L compares the entry itself with that directory;
            # every later walk uses '.', never a re-resolved mutable name.
            'else (cd "$d" || exit 3; '
            '[ "$(stat -c "%d:%i:%F" -- "$d")" = '
            '"$(stat -c "%d:%i:%F" .)" ] || exit 5; '
            # Already the agent's (an earlier role's step, or the agent made
            # it): left as it is, so nothing renamed into it changes owner.
            f'if [ -n "$(find . -prune -user {user} -print)" ]; then :; '
            # Root's alone throughout: folders and plain single-link files
            # only, on this file system (a link, a hard link, a device or a
            # mount inside stays as it was), each folder made u+rwx (a 555
            # one is the agent's but still unwritable) and then the agent's
            # after its contents.
            'elif [ "$s" = 1 ]; then '
            "bad=$(find . -xdev -type d "
            r"\( ! -user 0 -o -perm -020 -o -perm -002 \) -print) || exit 5; "
            '[ -z "$bad" ] || { echo "$d"; exit 5; }; '
            r"find . -xdev -depth \( -type d -exec chmod u+rwx {} \; "
            rf"-exec chown -h {owner} {{}} \; \) -o \( -type f -links 1 "
            rf"-exec chown -h {owner} {{}} \; \) || exit 4; "
            'else echo "$d"; exit 5; fi; '
            '[ -n "$(find . -prune -type d '
            f'-user {user} -print)" ] || exit 4) || exit $?; fi',
            'echo "$d"',
        ]
    return "; ".join(steps)


async def _prepare_agent_temp_dirs(
    env: Any,
    agent_cfg: Any,
    sandbox_user: str | None,
    *,
    locked_paths: list[str] | None = None,
) -> None:
    """Make the agent tool's temp folders the sandbox user's before launch.

    The agent's install runs as root and can leave its tool's temp folder
    root-owned (OpenCode's postinstall creates /tmp/opencode), where the tool
    then cannot write as the sandbox user: every trial ended in "OpenCode
    service failure" at the agent budget (#1183). Only the folders the agent
    declares (``AgentConfig.temp_dirs``) are made or re-owned. A link or a
    non-folder at one of them, or a folder another user could have filled,
    refuses the trial as a setup error rather than letting it time out. A
    folder that is already the sandbox user's is left as it is.
    """
    names = getattr(agent_cfg, "temp_dirs", None)
    if not sandbox_user or not isinstance(names, list) or not names:
        return
    result = await exec_raw(
        env.exec,
        _agent_temp_dirs_cmd(names, sandbox_user, locked_paths=locked_paths),
        user="root",
        timeout_sec=60,
    )
    code = getattr(result, "return_code", None)
    if code == 0:
        return
    out = (getattr(result, "stdout", "") or "").strip().splitlines()
    path = out[-1] if out else "its temp folder"
    agent = getattr(agent_cfg, "name", "the agent")
    if code == 3:
        reason = "is a link or not a folder"
    elif code == 5:
        reason = "could hold files another user put there"
    elif code == 6:
        reason = "overlaps a protected path"
    else:
        reason = "could not be made or given to the agent user"
    detail = (diagnostic_result(result).stderr.render() or "").strip()[:300]
    raise RuntimeError(
        f"sandbox startup refused: {agent}'s temp folder {path} {reason}, so "
        f"{agent} could not write it as {sandbox_user} (#1183)"
        + (f": {detail}" if detail else "")
    )


def _init_rollout(
    task_path: Path,
    job_name: str | None,
    rollout_name: str | None,
    jobs_dir: str | Path,
    *,
    rollout_paths: Any = None,
) -> tuple[Any, Path, Any, datetime, str, str]:
    """Set up trial directory tree and return core trial objects."""
    from uuid import uuid4

    from benchflow.task import RolloutPaths, Task

    task = Task(task_path)
    if rollout_paths is None:
        job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        rollout_name = rollout_name or f"{task_path.name}__{uuid4().hex[:8]}"
        rollout_dir = Path(jobs_dir) / job_name / rollout_name
        rollout_paths = RolloutPaths(rollout_dir=rollout_dir)
    else:
        rollout_dir = rollout_paths.rollout_dir
        job_name = job_name or rollout_dir.parent.name
        rollout_name = rollout_name or rollout_dir.name
    started_at = datetime.now()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("agent", "verifier", "artifacts", "trajectory"):
        (rollout_dir / subdir).mkdir(exist_ok=True)
    # Pin the rollout's own root before any agent runs: late publication
    # (egress logs, TASKMD-R14 P-H1) selects it, never a fresh root.
    rollout_paths.mkdir()
    return task, rollout_dir, rollout_paths, started_at, job_name, rollout_name


def _resolve_prompts(
    task_path: Path,
    prompts: list[str | None] | None,
) -> list[str]:
    """Read the task instruction and resolve prompt list."""
    instruction = _read_task_instruction(task_path)
    if prompts is None:
        return [instruction]
    return [p if p is not None else instruction for p in prompts]


def _apply_prompt_prefix(prompts: list[str], prompt_prefix: str | None) -> list[str]:
    """Prepend a recorded, task-agnostic harness policy to resolved prompts."""
    if prompt_prefix is None:
        return prompts
    return [f"{prompt_prefix}\n\n{prompt}" for prompt in prompts]


async def _start_env_and_upload(
    env: Any,
    task_path: Path,
    timing: dict,
    *,
    skip_start: bool = False,
    on_started: Callable[[], Awaitable[None] | None] | None = None,
    uploads: dict[str, str] | None = None,
    upload_task_files: bool = True,
    upload_solution: bool = False,
    image_holds_agent_bytes: bool = False,
) -> None:
    """Start environment and upload task files.

    ``image_holds_agent_bytes``: the image was built with agent-derived bytes
    in it (a learner store's skills baked in), so the sandbox is admitted as
    agent-derived as soon as it exists, before anything else runs in it.

    ``upload_task_files=False`` only starts the sandbox: one created from a
    branch snapshot already holds the instruction, the solution (an oracle
    run's) and uploads.

    ``upload_solution``: upload the task's ``oracle/`` (or ``solution/``) too.
    Only an oracle run does: the lockdown makes it root-only, which keeps it
    from a non-root agent but not from an agent run as root, so no other
    run has a copy in the sandbox at all (GAP-ORACLE-READABLE).
    Verifiers never read it; only an oracle run needs the solution.

    ``skip_start=True`` is used when the sandbox was created and started
    by the caller (Runtime with a live Environment, #388) — we still
    upload task files but must not re-run ``start()`` since most sandbox
    backends (e.g. daytona) are not idempotent.

    ``on_started`` runs once the sandbox exists but *before* any upload
    (#554/#563). Persisting the sandbox id here — not after upload —
    closes the failure window where an upload error or interrupt after
    Daytona creation would otherwise leave no ``sandbox.json`` to audit
    or clean up.
    """
    if skip_start:
        logger.info(f"Reusing caller-owned environment: {task_path.name}")
        timing["environment_setup"] = 0.0
    else:
        logger.info(f"Starting environment: {task_path.name}")
        t0 = datetime.now()
        from benchflow.task.config import SandboxConfig

        sandbox_config = getattr(env, "task_env_config", None)
        # Only a real task sandbox config carries the bound (a test's mock
        # environment, or a backend without one, starts as before).
        deadline = (
            sandbox_config.start_deadline_sec
            if isinstance(sandbox_config, SandboxConfig)
            else None
        )
        try:
            if deadline is None:
                await env.start(force_build=False)
            else:
                # A YAML job config's build phase: the whole start is
                # bounded by one deadline, including image preparation and
                # sandbox startup.
                try:
                    await asyncio.wait_for(env.start(force_build=False), deadline)
                except TimeoutError as exc:
                    raise EnvironmentStartTimeoutError(
                        f"the environment did not start within {deadline:g}s, the "
                        "environment build budget (environment.build_timeout_sec "
                        "times the job's build timeout multiplier)"
                    ) from exc
        except SandboxStartupError:
            raise
        except RuntimeError as exc:
            reason = docker_daemon_unreachable_reason(str(exc))
            if reason is None:
                raise
            raise SandboxStartupError(
                f"Docker daemon unreachable: {reason}. Start Docker and re-run; "
                "`bench doctor` shows the fix."
            ) from exc
        timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    if on_started is not None:
        started = on_started()
        if inspect.isawaitable(started):
            await started
    if image_holds_agent_bytes:
        admit(env, "main", IngressKind.AGENT_DERIVED_UPLOAD)
    if not upload_task_files:
        return
    task_files = admit(env, "main", TRUSTED_TRANSFER, reason="the task's own files")
    instruction_path = task_path / "instruction.md"
    if instruction_path.exists() and not (task_path / "task.md").exists():
        await env.upload_file(instruction_path, "/instruction.md", admission=task_files)
    else:
        instruction = _read_task_instruction(task_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(instruction)
            f.write("\n")
            temp_instruction = Path(f.name)
        try:
            await env.upload_file(
                temp_instruction, "/instruction.md", admission=task_files
            )
        finally:
            temp_instruction.unlink(missing_ok=True)
    from benchflow.task.paths import SandboxPaths, TaskPaths

    paths = TaskPaths(task_path)
    if upload_solution and paths.solution_dir.is_dir():
        sandbox_paths = SandboxPaths()
        target_dir = (
            sandbox_paths.oracle_dir
            if paths.uses_native_oracle_dir
            else sandbox_paths.solution_dir
        )
        await env.upload_dir(paths.solution_dir, str(target_dir), admission=task_files)

    from benchflow._paths import is_safe_regular_dir, is_safe_regular_file

    configured = None
    if uploads:
        # Configured uploads are the caller's bytes (reviewer evidence, trial
        # copies), never the task's own files.
        configured = admit(env, "main", IngressKind.AGENT_DERIVED_UPLOAD)
    for host_path, sandbox_path in (uploads or {}).items():
        source = Path(host_path)
        if is_safe_regular_dir(source):
            await env.upload_dir(source, sandbox_path, admission=configured)
        elif is_safe_regular_file(source):
            await env.upload_file(source, sandbox_path, admission=configured)
        else:
            raise FileNotFoundError(
                "configured upload source must be an existing regular file or "
                f"directory, not a symlink: {host_path}"
            )


def oracle_run_user(task: Any) -> str | None:
    """The user name the oracle runs solve.sh as when no sandbox user is
    passed (the rollout passes none): the task's ``[agent].user`` when it is
    a name, else None, and then solve.sh runs as the sandbox's default exec
    user (root unless the image says otherwise; a numeric ``[agent].user``
    is left to the image user, as su takes a name)."""
    declared = task.config.agent.user
    return declared if isinstance(declared, str) and declared else None


#: The first words of solve.sh's command (see _run_oracle): the exec names
#: root, which starts on root's fixed PATH, so the image's comes back.
ORACLE_IMAGE_PATH = 'export PATH="${_BENCHFLOW_IMAGE_PATH-$PATH}"; '


def oracle_user_command(command: str, user: str | None) -> str:
    """Run an oracle command under the same identity as solve.sh."""
    if user:
        from benchflow.sandbox._identities import oracle_command

        return oracle_command(user, command)
    return command


class OracleTimeoutError(TimeoutError):
    """solve.sh ran past the task's agent timeout, the oracle's budget."""

    def __init__(self, label: str, timeout_sec: float) -> None:
        super().__init__(
            f"{label} timed out after {timeout_sec:g}s, the task's agent timeout"
        )


async def _run_oracle(
    env: Any,
    task_path: Path,
    timeout: int,
    sandbox_user: str | None = None,
    *,
    rollout_dir: Path | None = None,
    task: Any = None,
) -> tuple[list[dict], str]:
    """Run oracle mode (oracle/solve.sh or legacy solution/solve.sh).

    A task network policy (``allowlist`` or ``denylist``) covers the
    oracle's processes while solve.sh runs, and only then (REPORT B10;
    :mod:`benchflow.sandbox.oracle_egress`). Its block log
    goes to ``rollout_dir`` when given.
    """
    from benchflow.task import Task, resolve_env_vars
    from benchflow.task.paths import SandboxPaths

    logger.info("Oracle mode: running oracle solve.sh")
    # Callers with validated overrides supply the effective task. Reloading
    # it here would silently restore the on-disk network/solution settings.
    task = task if task is not None else Task(task_path)
    if not task.paths.solve_path.exists():
        raise FileNotFoundError(
            f"Oracle requires oracle/solve.sh or legacy solution/solve.sh: {task_path}"
        )
    sandbox_paths = SandboxPaths()
    oracle_dir = (
        sandbox_paths.oracle_dir
        if task.paths.uses_native_oracle_dir
        else sandbox_paths.solution_dir
    )
    oracle_command_label = (
        "oracle/solve.sh" if task.paths.uses_native_oracle_dir else "solution/solve.sh"
    )
    oracle_script = shlex.quote(str(oracle_dir / "solve.sh"))
    # Run solve.sh as the task's [agent].user in its
    # [environment].workdir; without either the oracle keeps running as root
    # in the image's default directory.
    declared_user = task.config.agent.user
    # su takes a user name, digits included ("1000" is an account name, not
    # UID 1000); an integer [agent].user is left to the image user.
    if not sandbox_user:
        sandbox_user = oracle_run_user(task)
    from benchflow.task import keep_services

    # The account solve.sh runs as below, root when neither names one
    # (keep_services.run_identity, DT2-R9-2).
    await keep_services.prepare_identities(env, task, sandbox_user, inherit=True)
    workdir = _configured_task_workdir(task)
    if sandbox_user and sandbox_user == declared_user:
        # The oracle dir is uploaded root-owned; let the declared user run it.
        await exec_raw(
            env.exec,
            f"chmod -R a+rX {shlex.quote(str(oracle_dir))}",
            user="root",
            timeout_sec=30,
        )
    oracle_env: dict[str, str] = {"DEBIAN_FRONTEND": "noninteractive"}
    if task.config.solution.env:
        oracle_env.update(resolve_env_vars(task.config.solution.env))
    exec_kwargs: dict[str, Any] = {"env": oracle_env, "timeout_sec": timeout}
    if workdir is not None:
        _validate_agent_workdir(workdir)
        exec_kwargs["cwd"] = workdir
    policy = await task_service_policy(env, _task_egress_denylist(task))
    if policy is not None:
        policy = replace(
            policy,
            diagnostic_secrets=tuple(
                resolve_env_vars(task.config.solution.env).values()
            ),
        )
    armed = None
    if policy is not None:
        with timing_spans.span(timing_spans.ORACLE_NETWORK_POLICY):
            armed = await arm_oracle_egress(env, policy, run_as=sandbox_user)
        # The proxy and CA variables, over any the task's solution env sets:
        # clients that honour them take the policy's path, the rules refuse
        # the rest.
        oracle_env.update(armed.env)
    # A root oracle joins the policy's group just before solve.sh; one the
    # task runs as a non-root user is matched by its uid.
    join = (
        join_group_prefix(armed.gid)
        if armed is not None and armed.gid is not None
        else ""
    )
    oracle_cmd = f"{join}bash {oracle_script}"
    if sandbox_user:
        oracle_cmd = f"DEBIAN_FRONTEND=noninteractive {oracle_cmd}"
    if "PATH" not in oracle_env:
        # The exec below names root, and a command that names root starts on
        # root's fixed PATH with the image's kept aside (_root_exec_command):
        # solve.sh gets the image's back, as it had when the exec named no
        # user. A conda image's only python is in /opt/conda/bin.
        oracle_cmd = ORACLE_IMAGE_PATH + oracle_cmd
    # The same su that prepare_identities measured the account with.
    cmd = oracle_user_command(oracle_cmd, sandbox_user)
    # From root, whatever the image's USER: su switches account from root,
    # and an oracle with no account of its own runs as root. Named no user,
    # the exec ran as the image's USER, and an image ending with `USER
    # nobody` (Terminal-Bench 4.1's risk-scorer-replay) could neither open
    # /logs/agent/oracle.txt nor write the workspace, both the sandbox
    # user's after setup: solve.sh never ran (rc 2, no output).
    exec_kwargs["user"] = "root"
    admit(env, "main", IngressKind.ORACLE_LAUNCH)
    try:
        if keep_services.kept_by_identity(task) is not None:
            # A service solve.sh starts without a redirect of its own inherits
            # the output file, and the release kills a kept process that holds
            # a file in /logs (K2p-R1-A-2). So the output goes to a new
            # root-only file outside /logs and is copied into
            # /logs/agent/oracle.txt when solve.sh returns (a pipe would never
            # close while the service holds it). /logs/agent is the agent
            # user's, so root opens /logs/agent/oracle.txt before solve.sh runs
            # and writes the copy through that descriptor, which solve.sh does
            # not get: opened after, it would follow a link solve.sh or its
            # service put there to a file of root's (K2p-R2-B-2). With no
            # mktemp the output goes to /logs/agent/oracle.txt as before.
            cmd = (
                "o=$(mktemp /tmp/runtime-oracle.XXXXXX 2>/dev/null) || o=; "
                'if [ -n "$o" ]; then '
                "exec 3>/logs/agent/oracle.txt; "
                f'{cmd} 3>&- > "$o" 2>&1; rc=$?; '
                'cat "$o" >&3; exec 3>&-; rm -f "$o"; '
                f"else {cmd} > /logs/agent/oracle.txt 2>&1; rc=$?; fi; "
                "exit $rc"
            )
            with as_task_command(env):
                result = await exec_raw(env.exec, cmd, **exec_kwargs)
        else:
            with as_task_command(env):
                result = await exec_raw(
                    env.exec, f"{cmd} > /logs/agent/oracle.txt 2>&1", **exec_kwargs
                )
    except ExecTimeoutError:
        # solve.sh ran out of the agent budget it was given: the agent
        # timeout, never retried, not a sandbox failure (DF-p5-cli-craft-1).
        raise OracleTimeoutError(oracle_command_label, timeout) from None
    finally:
        if armed is not None:
            with timing_spans.span(timing_spans.ORACLE_NETWORK_POLICY):
                await disarm_oracle_egress(env, armed, rollout_dir)
    if result.return_code != 0:
        logger.warning(f"Oracle solve.sh exited with rc={result.return_code}")
    preview = await exec_raw(
        env.exec,
        f"tail -c {shlex.quote(str(_DIAG_TRUNCATE))} /logs/agent/oracle.txt 2>/dev/null || true",
        user="root",
        timeout_sec=10,
    )
    trajectory = [
        {
            "type": "oracle",
            "command": oracle_command_label,
            "return_code": result.return_code,
            "stdout": (diagnostic_result(preview).stdout.render() or "")[
                :_DIAG_TRUNCATE
            ],
        }
    ]
    return trajectory, "oracle"


_SCRIPT_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SCRIPT_ENV_PREFIX = "BENCHFLOW_SCRIPT_"
_SCRIPT_IMAGE_LOCALE = ("LANG", "LC_ALL")
# GNU timeout exits 124 (137 after -k); BusyBox's dies by the signal (143).
_WATCHDOG_CODES = (124, 137, 143)


async def _run_script(
    env: Any,
    script: bytes,
    label: str,
    timeout: int,
    sandbox_user: str,
    workdir: str,
    agent_env: dict[str, str],
    *,
    planes: RolloutPlanes | None = None,
    task: Any = None,
) -> tuple[list[dict], str]:
    """Run an operator's file as the agent, with an in-container watchdog.

    Docker's exec timeout only stops the host waiter and raises RuntimeError.
    The inner timeout and the uid kill must finish before verification starts.
    On Docker the script gets the container's configured environment (what
    an exec as the agent sees, as in Rust), the agent's HOME/USER/LOGNAME and
    the policy's proxy vars. Backends without a trusted root transport
    refuse the script before execution. BenchFlow's own host credentials, exec-time
    variables and the root solution's environment are not added; anything
    the task puts in the container environment is shared, as for any agent.
    A runner that never started the script, or never recorded its exit, raises instead of
    scoring an unattempted workspace.
    """
    from benchflow.agents.scripts import validate_script_user
    from benchflow.sandbox._base import LOADER_FLAG_VARS, BaseSandbox
    from benchflow.sandbox._identities import resolve, snapshot
    from benchflow.task import keep_services

    validate_script_user(sandbox_user)
    # The script is the agent: it is launched in this sandbox below.
    launch = admit(env, "main", IngressKind.AGENT_LAUNCH)
    bindings = planes or default_rollout_planes()
    _validate_agent_workdir(workdir)
    # Only / is an ancestor. A root-owned leaf beneath an agent-owned
    # workspace can be renamed, even if the leaf itself is unwritable.
    directory = f"/.runtime-script-{uuid4().hex}"
    target = f"{directory}/attempt.sh"

    async def root_exec(command: str, **kwargs: Any) -> Any:
        return await env.script_runner_exec(command, **kwargs)

    async def upload(source: str, target: str) -> None:
        await env.script_runner_upload(source, target, admission=launch)

    class RootTransport:
        # The process-stop binding must use the same trusted entry point as
        # the receipts and capture reads, including on timeout/cancellation.
        async def exec(self, command: str, *, user: str, **kwargs: Any) -> Any:
            if user != "root":
                raise RuntimeError("script: unexpected process-stop user")
            return await root_exec(command, **kwargs)

    # Check every lexical ancestor and symlink hop before the script can
    # plant replacements. A protected final inode is not sufficient.
    protect = r"""
declare -A protected=()
canonical=
protect() {
    local p=$1 depth=${2:-0} parent leaf owner mode link
    (( depth < 80 )) || return 1
    [[ $p == /* ]] || return 1
    while [[ $p != / && $p == */ ]]; do p=${p%/}; done
    if [[ $p != / ]]; then
        leaf=${p##*/}
        parent=${p%/*}; parent=${parent:-/}
        protect "$parent" "$((depth+1))" || return 1
        case $leaf in
            .) return 0 ;;
            ..) canonical=${canonical%/*}; canonical=${canonical:-/}; return 0 ;;
        esac
        p=${canonical%/}/$leaf
    fi
    canonical=$p
    [[ ${protected[$p]+checked} ]] && { canonical=${protected[$p]}; return 0; }
    # A missing entry beneath an already protected parent cannot be planted
    # by the agent. Trusted images can contain dormant, dangling aliases
    # (BusyBox's pinentry); still check every ancestor before accepting one.
    if [[ ! -e $p && ! -L $p ]]; then
        protected[$p]=$p
        return 0
    fi
    read -r owner mode < <(stat -c '%u %a' -- "$p")
    [[ $owner == 0 && $mode =~ ^[0-7]+$ ]] || return 1
    if [[ -L $p ]]; then
        # A filename may end in any number of newlines. The sentinel keeps
        # command substitution from stripping them; remove only readlink's
        # delimiter and the sentinel, preserving the exact target bytes.
        link=$(readlink -- "$p" && printf '.') || return 1
        [[ $link == *$'\n.' ]] || return 1
        link=${link%$'\n.'}
        [[ $link == /* ]] || link="${p%/*}/$link"
        protect "$link" "$((depth+1))" || return 1
    else
        (( (8#$mode & 8#22) == 0 )) || return 1
    fi
    protected[$p]=$canonical
}
for tool in / /etc/passwd /etc/group /bin/bash /usr/bin/env id mkdir chown chmod stat readlink su timeout cat wc tail base64 sleep; do
    [[ $tool == /* ]] || tool=$(command -v "$tool") || exit 1
    protect "$tool" || exit 1
done
# The shared process-stop command sets its own PATH. Check its entire command
# namespace, including non-executable files the agent could chmod later and
# hidden names. With protected directories no new name can be planted, and
# with protected entries/links no existing name can be replaced. This also
# covers helpers added to either cleanup implementation in a future merge.
# The cleanup PATH and this runner's fixed PATH are kept equal by a regression.
shopt -s nullglob dotglob
for directory in __CLEANUP_DIRECTORIES__; do
    p=$directory
    while [[ ! -e $p && ! -L $p ]]; do p=${p%/*}; p=${p:-/}; done
    protect "$p" || exit 1
    for tool in "$directory"/*; do protect "$tool" || exit 1; done
done
""".replace(
        "__CLEANUP_DIRECTORIES__",
        " ".join(shlex.quote(path) for path in _SCRIPT_DEFAULT_PATH.split(":")),
    )
    prepared = await root_exec(
        protect + f"mkdir -m 755 {directory}",
        timeout_sec=30,
    )
    if prepared.return_code != 0:
        raise RuntimeError(
            "script: could not prepare a non-root runner before the script started; "
            "make system tools, symlink targets and their ancestors root-owned "
            "and unwritable by the agent in the image"
        )
    configured = (
        await env.script_container_config() if isinstance(env, BaseSandbox) else None
    )
    identities = await snapshot(env, root_exec=root_exec, config=configured or {})
    agent_identity = resolve(identities, sandbox_user)
    if agent_identity.uid == 0:
        raise RuntimeError("script: choose a non-root sandbox user in the image")
    await keep_services.prepare_identities(env, task, sandbox_user)
    root_transport = RootTransport()
    vars(root_transport)["_benchflow_identities"] = identities
    with tempfile.NamedTemporaryFile() as source:
        source.write(script)
        source.flush()
        await upload(source.name, target)
    staged = await root_exec(
        f"chown root:root {target} && chmod 755 {target}", timeout_sec=30
    )
    if staged.return_code != 0:
        raise RuntimeError("script: could not stage the root-owned script")
    clean_env = {
        "HOME": f"/home/{sandbox_user}",
        "USER": sandbox_user,
        "LOGNAME": sandbox_user,
    }
    # Config.Env is what Docker exec inherits, including image ENV and task
    # compose environment, but not BenchFlow's exec-time additions. Unlike
    # /proc/1/environ it works with a non-root PID 1 and ignores entrypoint
    # exports (which Docker exec also ignores). Upload NUL-separated values
    # as data, never as shell code or arguments to the host Docker CLI.
    # Keep other backends' existing PID-1/PATH-and-locale capture unchanged.
    image_env_file = f"{directory}/image.env"
    capture_image_env = (
        f"{{ cat /proc/1/environ > {image_env_file} 2>/dev/null || "
        f": > {image_env_file}; }} && "
    )
    if configured is not None:
        entries = configured.get("Env")
        if entries is None:
            entries = []
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or "\0" in entry for entry in entries
        ):
            raise RuntimeError("script: invalid Docker container environment")
        # Docker's Linux exec environment starts with PATH and HOSTNAME, then
        # Config.Env replaces/adds entries by name. Imported images can have
        # no Env or a non-empty Env without PATH. HOME is supplied below.
        image_env = {"PATH": _SCRIPT_DEFAULT_PATH}
        hostname = configured.get("Hostname")
        if isinstance(hostname, str) and hostname:
            image_env["HOSTNAME"] = hostname
        for entry in entries:
            name, separator, value = entry.partition("=")
            if separator:
                image_env[name] = value
        # S1's bootstrap created main with the loader and start-up names
        # cleared (bare keys, not in Docker's exec environment). The image's
        # values come back as they do for ACP agents and the oracle
        # (restored_env): LD_LIBRARY_PATH for the image's own tools,
        # PYTHONPATH, NODE_PATH, ... The loader's switches never do
        # (review A M-1, pre-existing since 6ad32f11d).
        restored = _inert_bootstrap.restored_env(getattr(env, "_strategy", None) or env)
        for name, value in restored.items():
            if name not in LOADER_FLAG_VARS:
                image_env.setdefault(name, value)
        with tempfile.NamedTemporaryFile() as source:
            for name, value in image_env.items():
                source.write(f"{name}={value}".encode() + b"\0")
            source.flush()
            await upload(source.name, image_env_file)
        capture_image_env = ""
    fallback = " ".join(
        [f'PATH="${{{_SCRIPT_ENV_PREFIX}PATH:-{_SCRIPT_DEFAULT_PATH}}}"']
        + [
            f'${{{_SCRIPT_ENV_PREFIX}{name}:+"{name}=${_SCRIPT_ENV_PREFIX}{name}"}}'
            for name in _SCRIPT_IMAGE_LOCALE
        ]
    )
    load_image_env = (
        "image=(); while IFS= read -r -d '' entry; do "
        'if [[ $entry =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then image+=("$entry"); fi; '
        f"done < {image_env_file} || exit 125; "
        f'[ "${{#image[@]}}" -gt 0 ] || image=({fallback}); '
    )
    streams = (f"{directory}/script.stdout", f"{directory}/script.stderr")
    # Bash's protected startup file stamps the receipt inside a protected
    # interpreter, after exec and all agent-side setup succeeded.
    # This bootstrap gets only a fixed environment: task settings such as
    # POSIXLY_CORRECT or SHELLOPTS can suppress BASH_ENV or its commands.
    # Carry the task environment as arguments until the receipt fd is closed,
    # then always hand off to native Bash with the same held script inode.
    startup_file = f"{directory}/started.bash"
    with tempfile.NamedTemporaryFile() as source:
        source.write(b"""printf 1 >&3 || exit 125
exec 3>&-
exec /usr/bin/env -i "$@" /bin/bash --noprofile --norc /dev/fd/6
""")
        source.flush()
        await upload(source.name, startup_file)
    # Hold the script's inode and open both output files before the final
    # interpreter starts. No setup/exec failure can leave a start receipt.
    inner = (
        load_image_env
        + f"exec 4> {streams[0]} 5> {streams[1]} 6< {target} || exit 125; "
        + "exec /usr/bin/env -i PATH=/usr/bin:/bin "
        + f"BASH_ENV={startup_file} /bin/bash --noprofile --norc /dev/fd/6 "
        + '"${image[@]}" '
        + " ".join(shlex.quote(f"{k}={v}") for k, v in clean_env.items())
        + " "
        + " ".join(shlex.quote(f"{k}={v}") for k, v in agent_env.items())
    )
    inner += " 1>&4 2>&5 4>&- 5>&-"
    # Capture only the script, not su's watchdog diagnostics. Preserve the two
    # streams so the stored output uses Rust's stdout.rstrip + stderr.rstrip.
    ready = await root_exec(
        f": > {streams[0]} && : > {streams[1]} && chmod 666 {' '.join(streams)} && "
        f"{capture_image_env}chown root:root {image_env_file} {startup_file} && "
        f"chmod 644 {image_env_file} {startup_file}",
        timeout_sec=10,
    )
    if ready.return_code != 0:
        raise RuntimeError("script: could not prepare output capture")
    # A root-owned completion marker distinguishes a script that exits 124/137
    # itself from the watchdog. The agent cannot forge it in this directory.
    supervised = (
        f"su -s /bin/bash {shlex.quote(sandbox_user)} -c {shlex.quote(inner)} "
        f"3> {directory}/started; "
        f'rc=$?; printf \'%s\' "$rc" > {directory}/exit-code; exit "$rc"'
    )
    # No task or run limit means no limit, as for the oracle and in Rust.
    limit = timeout if timeout > 0 else None
    # Short options only: BusyBox timeout rejects GNU's long ones.
    watchdog = f"timeout -s TERM -k 1 {limit} " if limit else ""
    from benchflow.sandbox._base import system_shell_route

    root_route = await system_shell_route(env)
    command = (
        f"{watchdog}{shlex.quote(root_route.bash)} --noprofile --norc -c {shlex.quote(supervised)} "
        f"> {directory}/launch.txt 2>&1"
    )
    timed_out = False
    try:
        result = await root_exec(
            command,
            cwd=workdir,
            timeout_sec=limit + 15 if limit else None,
        )
        code = result.return_code
        # The two files' contents come back on one line tagged with a nonce,
        # read wherever it is in the output: anything else a program prints
        # on the way (glibc's LD_SHOW_AUXV dump, SCRIPT-AGENT-DOCKER-F1) can
        # neither hide nor forge the receipt, and is reported, not dropped.
        tag = f"benchflow-script-marks-{uuid4().hex}"
        markers = await root_exec(
            f'printf "\\n{tag}:%s:%s\\n" "$(cat {directory}/started 2>/dev/null)" '
            f'"$(cat {directory}/exit-code 2>/dev/null)"',
            timeout_sec=10,
        )
        stdout = markers.stdout if isinstance(markers.stdout, str) else ""
        found = re.findall(rf"^{tag}:([^:\n]*):([^\n]*)$", stdout, re.MULTILINE)
        started_mark, recorded_text = (
            found[0] if markers.return_code == 0 and len(found) == 1 else ("", "")
        )
        stray = re.sub(rf"^{tag}:[^\n]*$", "", stdout, flags=re.MULTILINE).strip()
        if stray:
            logger.warning(
                "script: the root runner printed %d unexpected characters around "
                "its markers (an image loader or libc setting?)",
                len(stray),
            )
        recorded = recorded_text or None
        if started_mark != "1":
            # su never ran the agent-side shell (missing, PAM or dropped
            # SETUID/SETGID): its own exit code is not the script's, and the
            # workspace was never attempted, so it must not be scored.
            await bindings.quiesce_agent(root_transport, sandbox_user)
            raise RuntimeError(
                f"script: the runner exited {code} before the script started; "
                "the image needs /bin/bash, su and timeout, and root must be "
                f"able to su to {sandbox_user} "
                f"(see {directory}/launch.txt)"
            )
        if recorded != str(code):
            if not limit or code not in _WATCHDOG_CODES:
                # The supervising shell never recorded the script's exit, so
                # the runner itself failed (missing bash, an unusable timeout):
                # the workspace was never attempted and must not be scored.
                await bindings.quiesce_agent(root_transport, sandbox_user)
                raise RuntimeError(
                    f"script: the runner exited {code} before the script "
                    "finished; the image needs /bin/bash, su and timeout "
                    f"(see {directory}/launch.txt)"
                )
            timed_out = True
    except (TimeoutError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "Command timed out" not in str(exc):
            raise
        timed_out = True
        code = 124
    except BaseException:
        await bindings.quiesce_agent(root_transport, sandbox_user)
        raise
    if timed_out:
        from benchflow.task import keep_services

        # A timeout ends an attempted turn and still reaches grading. Use the
        # same task policy and freeze/record path as other agent turns; runner
        # failures and cancellation above still end every agent process.
        keep = keep_services.trial_keep(task, sandbox_user)
        if keep is None:
            await bindings.quiesce_agent(root_transport, sandbox_user)
        else:
            outcome = await bindings.quiesce_agent(
                root_transport, sandbox_user, keep_ports=keep.ports
            )
            keep_services.mark_waited(task)
            keep_services.note_turn_end(task, outcome)
        code = 124
    output = []
    for stream in streams:
        captured = await root_exec(
            f"wc -c < {stream}; tail -c 16777216 {stream} | base64 -w0",
            timeout_sec=30,
        )
        if captured.return_code != 0:
            raise RuntimeError("script: could not read output capture")
        size, _, encoded = (captured.stdout or "").partition("\n")
        data = base64.b64decode(encoded)
        dropped = max(0, int(size.strip()) - len(data))
        if dropped:
            while data and data[0] & 0xC0 == 0x80:
                data = data[1:]
                dropped += 1
        text = data.decode("utf-8", errors="replace")
        if dropped:
            text = f"[benchflow: {dropped} earlier bytes of output were dropped; the last {len(data)} are kept]\n{text}"
        output.append(text.rstrip())
    combined = "\n".join(part for part in output if part)
    with tempfile.NamedTemporaryFile() as log:
        log.write(combined.encode("utf-8"))
        log.flush()
        await upload(log.name, f"{directory}/combined.txt")
    await root_exec(
        f"chown root:root {directory}/combined.txt && chmod 644 {directory}/combined.txt",
        timeout_sec=10,
    )
    # /logs/agent belongs to the agent. Publish with its privileges, never a
    # root upload that could follow an agent-created destination symlink.
    # Best effort: the trajectory keeps the output, and an agent-made FIFO or
    # other trap at the destination must not turn the trial into an error.
    publish = f"rm -f /logs/agent/script.txt && cat {directory}/combined.txt > /logs/agent/script.txt"
    try:
        published = await exec_raw(
            env.exec,
            f"timeout -s KILL 5 /bin/sh -c {shlex.quote(publish)}",
            user=sandbox_user,
            timeout_sec=10,
        )
        failed = published.return_code != 0
    except (TimeoutError, RuntimeError) as exc:
        logger.warning(f"Publishing script.txt failed: {exc}")
        failed = True
    if failed:
        logger.warning(
            "Could not publish script.txt as the agent; trajectory retains its output"
        )
    return [
        {
            "type": "script",
            "command": label,
            "return_code": code,
            "stdout": combined[-4000:],
            "timed_out": timed_out,
        }
    ], label


# The file a verifier that grades the agent's reply reads (#1182). Registry
# tasks read /logs/agent/output.txt (MMMLU's test.sh tries oracle.txt,
# gemini-cli.txt, codex.txt, then output.txt, then every *.txt there). The
# agent's own stderr log is <agent>.txt in the same folder on Docker, so the
# reply never takes that name.
FINAL_REPLY_FILE = "output.txt"
TRAJECTORY_FILE = "acp_trajectory.jsonl"


class VerifierPublicationError(RuntimeError):
    """Required agent evidence could not be placed; no grade is valid."""


class VerifierPublicationInfrastructureError(
    VerifierPublicationError, VerifierInfrastructureError
):
    """Publishing failed in the controller or sandbox API."""


def final_agent_reply(trajectory: list[dict]) -> str | None:
    """The agent's final reply: its last top-level message of the last turn.

    The last ``agent_message`` with text after the last ``user_message`` (a
    turn that ends on a tool call, or on an empty chunk after one, keeps the
    text before it), joined with the messages just before it that only
    thoughts or notices split off. A sub-agent's
    text (``parent_tool_call_id``) is not the reply. ``None`` when the last
    prompt got no text.
    """

    def top_level(event: Any, kind: str) -> bool:
        return (
            isinstance(event, dict)
            and event.get("type") == kind
            and "parent_tool_call_id" not in event
        )

    end = None
    for index in range(len(trajectory) - 1, -1, -1):
        event = trajectory[index]
        if top_level(event, "user_message"):
            break
        # The session records empty chunks too: a blank one after a tool
        # call is not the reply.
        text = event.get("text") if top_level(event, "agent_message") else None
        if isinstance(text, str) and text.strip():
            end = index
            break
    if end is None:
        return None
    parts: list[str] = []
    for event in reversed(trajectory[: end + 1]):
        if top_level(event, "agent_message"):
            parts.append(str(event.get("text") or ""))
        elif isinstance(event, dict) and (
            "parent_tool_call_id" in event
            or event.get("type") in ("agent_thought", "notice")
        ):
            continue
        else:
            break
    reply = "".join(reversed(parts))
    return reply if reply.strip() else None


def _write_host_file(path: ReachablePath, text: str) -> None:
    """Encode first, then replace: never truncate an agent-provided inode.

    Escaped lone surrogates are legal ACP JSON but not UTF-8 scalars. Keep
    them visible as backslash escapes in raw text, as in the original JSONL.
    A new inode also prevents hard-linked reply/trajectory files from
    truncating one another; it is written privately and placed
    through the held agent folder (reachable_write, R4-1).
    """
    payload = text.encode("utf-8", errors="backslashreplace")
    path.write(payload)


async def _publish_trajectory_for_verifier(
    env,
    trajectory: list[dict],
    agent_dir: ReachablePath,
    *,
    sandbox_user: str | None = None,
    publish_reply: bool = True,
) -> None:
    """Make the captured ACP trajectory and the agent's final reply available
    inside /logs for verifiers.

    ``acp_trajectory.jsonl`` holds the whole trajectory; ``output.txt`` the
    agent's final reply (:func:`final_agent_reply`, redacted like the
    trajectory), where verifiers that grade the reply read it (#1182).

    Also writes the same payloads to the host rollout ``agent/`` dir so the
    artifact set matches across backends: Docker bind-mounts ``/logs/agent``
    to the host agent dir, while remote sandboxes (Daytona, Modal) never
    mirror the published files back.

    /logs/agent belongs to ``sandbox_user``, and an agent process can still
    be running (the kill comes with hardening, after this). So root never
    writes there by name: each file is staged in a fresh root-owned folder
    under /logs and copied into place as ``sandbox_user``, so a link the
    agent left at the name reaches only what the agent could write itself.
    Without a sandbox user the agent ran as root, or no agent ran here.
    """
    files = {TRAJECTORY_FILE: redact_acp_trajectory_jsonl(trajectory) + "\n"}
    # Scripted agents own their files directly and have no ACP reply. Their
    # trusted rollout configuration selects this policy, never event text.
    if publish_reply:
        reply = final_agent_reply(trajectory)
        if reply is None:
            # No text is still the current turn's answer. Replacing the old
            # inode also clears an earlier round's reply on mounted backends.
            files[FINAL_REPLY_FILE] = ""
        else:
            reply = redact_trajectory_text(reply)
            files[FINAL_REPLY_FILE] = reply if reply.endswith("\n") else reply + "\n"
    agent_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        try:
            _write_host_file(agent_dir / name, payload)
        except OSError as exc:
            if verifier_infrastructure_failure(exc):
                raise VerifierPublicationInfrastructureError(
                    f"Host publication of {name} failed: "
                    + verifier_failure_reason(exc)
                ) from exc
            logger.warning(f"Could not write the host copy of {name}: {exc}")
    # Only defensive: mounted backends already have /logs/agent from the bind
    # mount, and a failed upload below still fails loudly. Publishing runs
    # after the agent finished, so a slow or failing exec here must not
    # discard an otherwise complete rollout. That mkdir rides in the stage's
    # root exec (PY8 U12): its failure is ignored as before, and the stage's
    # mkdir still decides the step.
    stage = f"/logs/.runtime-publish-{uuid4().hex}"
    # The agent's log folder, as setup leaves it (_log_dir_cmds), in case the
    # mkdir before it just made it root's.
    owner = (
        f" && {{ [ -L /logs/agent ] || chown {shlex.quote(sandbox_user)} /logs/agent; }}"
        if sandbox_user
        else ""
    )
    try:
        # A backend can lose the acknowledgement after mkdir completed.
        # Allocate a new private stage for each attempt rather than retrying
        # a non-idempotent mkdir or trusting an existing path. Keep every
        # attempted name for cleanup, including an unacknowledged success.
        stages = [stage]
        failures: list[Exception] = []
        for attempt in range(2):
            if attempt:
                stage = f"/logs/.runtime-publish-{uuid4().hex}"
                stages.append(stage)
            # The defensive mkdir rides in front of each attempt: it is
            # idempotent, its status is dropped by the ";", and /logs exists
            # before the stage mkdir needs it.
            if await _publish_step(
                env,
                f"mkdir -p /logs/agent; mkdir -m 755 {stage}{owner}",
                "root",
                failures=failures,
            ):
                break
        else:
            if failures and not isinstance(
                failures[-1], (TimeoutError, ExecTimeoutError)
            ):
                raise VerifierPublicationInfrastructureError(
                    "verifier evidence publication failed after 2 attempts: "
                    + verifier_failure_reason(failures[-1])
                ) from failures[-1]
            raise VerifierPublicationError(
                "verifier evidence publication failed after 2 attempts"
            )
        # The files go up at once where the sandbox allows it: each to its
        # own name in the fresh stage, so neither upload depends on the other
        # (sandbox._in_order, PERF-G). A failed one fails the stage as before.
        tmp_paths: dict[str, str] = {}
        admission = admit(env, "main", IngressKind.AGENT_DERIVED_UPLOAD)
        try:
            for name, payload in files.items():
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                    f.write(payload.encode("utf-8", errors="backslashreplace"))
                    tmp_paths[name] = f.name
            await in_order(
                env,
                *(
                    functools.partial(
                        env.upload_file,
                        tmp_path,
                        f"{stage}/{name}",
                        admission=admission,
                    )
                    for name, tmp_path in tmp_paths.items()
                ),
            )
        finally:
            for tmp_path in tmp_paths.values():
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp_path)
        if await _required_publish_step(env, f"chmod 644 {stage}/*", "root"):
            # Each file on its own: one that cannot be placed (the agent made
            # a folder of its name) does not keep the other from the verifier.
            copy = "; ".join(
                f"{{ rm -f /logs/agent/{name} && "
                f"cat {stage}/{name} > /logs/agent/{name}; }} || rc=1"
                for name in files
            )
            await _required_publish_step(
                env, f"rc=0; {copy}; exit $rc", sandbox_user or "root"
            )
    except VerifierPublicationError:
        raise
    except (TimeoutError, ExecTimeoutError) as exc:
        raise VerifierPublicationError(
            "verifier evidence publication timed out"
        ) from exc
    except Exception as exc:
        raise VerifierPublicationInfrastructureError(
            "verifier evidence publication upload failed: "
            + verifier_failure_reason(exc)
        ) from exc
    finally:
        for attempted in stages:
            await _publish_step(env, f"rm -rf {attempted}", "root")


async def _required_publish_step(env: Any, command: str, user: str) -> bool:
    # Bounded retries preserve completed solver work across a transient exec
    # failure. Persistent failure belongs to verification, never reward zero.
    failures: list[Exception] = []
    for _ in range(2):
        if await _publish_step(env, command, user, failures=failures):
            return True
    if failures and not isinstance(failures[-1], (TimeoutError, ExecTimeoutError)):
        raise VerifierPublicationInfrastructureError(
            "verifier evidence publication failed after 2 attempts: "
            + verifier_failure_reason(failures[-1])
        ) from failures[-1]
    raise VerifierPublicationError(
        "verifier evidence publication failed after 2 attempts"
    )


async def _publish_step(
    env: Any, command: str, user: str, *, failures: list[Exception] | None = None
) -> bool:
    """Run one publishing command; a failure is logged, never raised.

    Publishing runs after the agent finished: a slow or failing exec must not
    discard an otherwise complete rollout (#948).
    """
    if failures is not None:
        failures.clear()
    try:
        result = await exec_raw(env.exec, command, user=user, timeout_sec=30)
    except Exception as exc:
        if failures is not None:
            failures.append(exc)
        logger.warning(f"Publishing for the verifier: {command!r} failed: {exc}")
        return False
    code = getattr(result, "return_code", 0)
    if code:
        detail = (
            diagnostic_result(result).stderr.render()
            or diagnostic_result(result).stdout.render()
            or ""
        )
        logger.warning(
            f"Publishing for the verifier: {command!r} exited {code}: "
            f"{detail.strip()[:300]}"
        )
        return False
    return True


_VERIFIER_START_GRACE_SEC = 30.0


async def _verifier_started(env: Any, verifier: Any) -> bool | None:
    """Whether the command wrote its start receipt; ``None`` when unknown.

    A missing receipt is marked abandoned in the same shell command, and the
    test command checks for that mark right after writing its receipt: a
    command that starts after this probe gave up on it exits without running
    the tests, so it cannot race the retry (or recovery) on the verifier's
    outputs.
    """
    receipt = getattr(verifier, "execution_receipt", None)
    if not isinstance(receipt, tuple) or len(receipt) != 2:
        return None
    path, service = receipt
    quoted = shlex.quote(path)
    abandoned = shlex.quote(f"{path}.abandoned")
    try:
        probe = await asyncio.wait_for(
            exec_raw(
                env.exec,
                f"if [ -e {quoted} ]; then cat {quoted}; else : > {abandoned}; fi",
                service=service,
                user="root",
                timeout_sec=10,
            ),
            timeout=15,
        )
        return probe.return_code == 0 and probe.stdout.strip() == "started"
    except Exception as exc:
        # A failed probe says nothing about the verifier. Keep waiting and
        # probe again rather than cancelling a verifier that may be running.
        logger.warning(f"Verifier start-receipt probe failed; retrying: {exc!r}")
        return None


class _NeverStarted(Exception):
    """The test command's start receipt was missing: it never ran (#1136)."""


class _UnprobedTimeout(TimeoutError):
    """The budget ran out without a start receipt to ask (not a script
    verifier, or its receipt directory could not be made)."""


# How often an attempt looks for the start receipt the verifier names when it
# issues the test command; the grace runs from that moment, not from the
# verifier's start, so a slow tests upload is never mistaken for a wedge.
_VERIFIER_RECEIPT_POLL_SEC = 1.0


class _TestEndUnknown(TimeoutError):
    """The budget ran out and the sandbox did not answer whether the test
    command had ended: an unknown outcome, which is never scored."""


# A test command that ended behind a lost exec gets this long, after the
# verifier's budget, to bring in its outputs and be scored.
_VERIFIER_FINISH_BUDGET_SEC = 300.0
_TEST_END_PROBE_TIMEOUT_SEC = 30


async def _verifier_finished(env: Any, verifier: Any) -> int | None:
    """The exit status the issued test command recorded when its tests ended
    (``<receipt>.exit``, :meth:`Verifier.finish_test_command`), or None while
    they run, or for a verifier that records none.

    Raises :class:`_TestEndUnknown` when the sandbox does not answer twice:
    nothing then tells tests still running at the deadline from tests that
    ended behind a lost exec.
    """
    receipt = getattr(verifier, "execution_receipt", None)
    if (
        not isinstance(receipt, tuple)
        or len(receipt) != 2
        or getattr(verifier, "test_command_finishable", False) is not True
    ):
        return None
    path, service = receipt
    ended = shlex.quote(f"{path}.exit")
    command = (
        f"if [ -f {ended} ]; then printf 'ended '; cat {ended}; else printf running; fi"
    )
    failure: BaseException | None = None
    for _ in range(2):
        try:
            probe = await asyncio.wait_for(
                exec_raw(
                    env.exec,
                    command,
                    service=service,
                    user="root",
                    timeout_sec=_TEST_END_PROBE_TIMEOUT_SEC,
                ),
                timeout=_TEST_END_PROBE_TIMEOUT_SEC + 10,
            )
        except Exception as exc:
            failure = exc
            continue
        if probe.return_code != 0:
            failure = RuntimeError(f"the probe exited with rc={probe.return_code}")
            continue
        response = (probe.stdout or "").strip()
        if response == "running":
            return None
        recorded = re.fullmatch(r"ended (\d{1,3})", response)
        if recorded and int(recorded.group(1)) <= 255:
            return int(recorded.group(1))
        failure = RuntimeError("the verifier receipt response is incomplete or invalid")
    assert failure is not None
    raise _TestEndUnknown(describe_exception(failure)) from failure


async def _verifier_attempt(env: Any, verifier: Any, timeout: float) -> Any:
    """Run ``verifier.verify()`` once (:func:`_verify_until_deadline`).

    A test command whose tests ended although its exec never returned, or
    returned only past the budget (a lost Daytona poll, a hung transport), is
    scored from its outputs as if the exec had returned:
    :func:`_verifier_finished` reads the exit status it recorded, and
    ``finish_test_command`` gets ``_VERIFIER_FINISH_BUDGET_SEC`` to bring in
    and score them. Tests still running at the deadline keep the timeout.
    """
    try:
        return await _verify_until_deadline(env, verifier, timeout)
    except _UnprobedTimeout:
        raise
    except (TimeoutError, ExecTimeoutError):
        status = await _verifier_finished(env, verifier)
        if status is None:
            raise
    logger.warning(
        f"Verifier test command ended (exit status {status}) but its exec did "
        "not return within the budget; scoring its outputs"
    )
    try:
        return await asyncio.wait_for(
            verifier.finish_test_command(status),
            timeout=_VERIFIER_FINISH_BUDGET_SEC,
        )
    except (TimeoutError, ExecTimeoutError) as exc:
        raise VerifierExecutionError(
            "the test command ended, but bringing in its outputs did not "
            f"finish within {_VERIFIER_FINISH_BUDGET_SEC:g}s: "
            f"{describe_exception(exc)}"
        ) from exc


async def _verify_until_deadline(env: Any, verifier: Any, timeout: float) -> Any:
    """Run ``verifier.verify()`` once, bounding command startup separately.

    Raises :class:`_NeverStarted` when the start receipt is still missing
    ``_VERIFIER_START_GRACE_SEC`` after the test command was issued (or at the
    deadline, when that comes first), :class:`_UnprobedTimeout` when the
    budget ends with no receipt to ask, and ``TimeoutError`` when a verifier
    that did start (or whose start could not be probed) runs out of budget.
    """
    from benchflow.rollout._deadline import _swallow_abandoned_outcome

    running = asyncio.create_task(verifier.verify())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    issued_at: float | None = None
    try:
        while True:
            now = loop.time()
            if issued_at is None and getattr(verifier, "execution_receipt", None):
                issued_at = now
            if issued_at is None:
                wait = min(deadline - now, _VERIFIER_RECEIPT_POLL_SEC)
            else:
                wait = min(deadline, issued_at + _VERIFIER_START_GRACE_SEC) - now
            done, _ = await asyncio.wait({running}, timeout=max(0.0, wait))
            if done:
                return running.result()
            now = loop.time()
            if issued_at is None:
                if now >= deadline:
                    raise _UnprobedTimeout
                continue
            if now < deadline and now < issued_at + _VERIFIER_START_GRACE_SEC:
                continue
            started = await _verifier_started(env, verifier)
            if started is False:
                raise _NeverStarted
            if started is True:
                done, _ = await asyncio.wait(
                    {running}, timeout=max(0.0, deadline - loop.time())
                )
                if done:
                    return running.result()
                raise TimeoutError
            if loop.time() >= deadline:
                raise TimeoutError
            issued_at = loop.time()  # unknown: probe again one grace later
    finally:
        if not running.done():
            running.cancel()
            done, _ = await asyncio.wait({running}, timeout=5)
            if done:
                _swallow_abandoned_outcome(running)
            else:
                running.add_done_callback(_swallow_abandoned_outcome)


async def _await_verifier(
    env: Any, verifier: Any, timeout: float, *, retry_in_place: bool = True
) -> Any:
    """Run the verifier; retry once where its start cannot be confirmed.

    A missing receipt can mean the exec never started, or the verifier user
    removed it. One retry preserves the existing recovery attempt without
    letting that writable signal discard the trial. A second missing receipt
    raises a named evidence error that the scoring boundary converts to 0.
    Recovery-eligible tasks skip the in-place retry. Without a receipt probe,
    a timeout with no output still gets the existing single retry.
    """
    try:
        return await _verifier_attempt(env, verifier, timeout)
    except _NeverStarted:
        if not retry_in_place:
            raise VerifierStartUnconfirmedError(
                "verifier start receipt is missing: the command may not have "
                "started, or test.sh's user may have removed the receipt"
            ) from None
        why = f"no start receipt after {_VERIFIER_START_GRACE_SEC:g}s"
    except _UnprobedTimeout:
        if not retry_in_place or not await _verifier_wedged_without_output(env):
            raise TimeoutError from None
        why = "timed out with no output"
    logger.warning(
        f"Verifier test command start is unconfirmed ({why}); retrying it once"
    )
    await _kill_orphan_verifier(env, verifier)
    # The first attempt's receipt cannot confirm its start. The retry's
    # command names a fresh receipt when it is issued.
    with contextlib.suppress(AttributeError):
        verifier.execution_receipt = None
    try:
        return await _verifier_attempt(env, verifier, timeout)
    except _NeverStarted:
        raise VerifierStartUnconfirmedError(
            "verifier start receipt is missing after two attempts: the command "
            "may not have started, or test.sh's user may have removed the receipt"
        ) from None
    except _UnprobedTimeout:
        raise TimeoutError from None


async def _release_kept_services(
    planes: RolloutPlanes,
    env: Any,
    sandbox_user: str,
    pids: tuple[int, ...],
    record: dict[str, Any],
    record_dir: str | Path,
    verifier_code_dir: str,
) -> None:
    """Close the verifier's code folder, then let the kept services go on.

    A kept process the release killed instead (it still held a way into
    /logs/verifier) is added to the kill's record with why. At the first
    release of the trial, a kept pid that had exited is added under
    ``kept_services_lost``; a later release (a verifier retried in place)
    finds them running by then. The release no longer reports kept pids
    that are not stopped (``kept_services_unfrozen``): the frozen cgroup,
    not SIGSTOP, holds them (K1-CONFIRM-1).
    """
    first = "kept_services_released" not in record
    outcome = await planes.release_kept_services(
        env, sandbox_user, pids, (verifier_code_dir,)
    )
    released = getattr(outcome, "released", outcome)
    logger.info(f"Kept services resumed for grading: {released} of {len(pids)}")
    changed = False
    if first:
        record["kept_services_released"] = released
        lost = list(getattr(outcome, "lost", ()))
        unfrozen = list(getattr(outcome, "unfrozen", ()))
        if lost:
            record["kept_services_lost"] = lost
        if unfrozen:
            record["kept_services_unfrozen"] = [
                {"pid": pid, "state": state} for pid, state in unfrozen
            ]
        on_terminal = list(getattr(outcome, "on_terminal", ()))
        if on_terminal:
            record["kept_services_on_terminal"] = [
                {"pid": pid, "terminal": terminal} for pid, terminal in on_terminal
            ]
        opened = list(getattr(outcome, "opened", ()))
        if opened:
            record["kept_services_closed_folders"] = [
                {"path": path, "owner": owner, "mode": mode}
                for path, owner, mode in opened
            ]
        changed = True
    dropped = getattr(outcome, "dropped", ())
    if dropped:
        rows = record.setdefault("kept_services_dropped", [])
        known = {row.get("pid") for row in rows}
        rows.extend(
            {"pid": pid, "why": why} for pid, why in dropped if pid not in known
        )
        changed = True
    if changed:
        _write_kill_record(record_dir, record)


async def _end_kept_services(
    planes: RolloutPlanes,
    env: Any,
    sandbox_user: str,
    record: dict[str, Any],
    record_dir: str | Path,
) -> None:
    """After the verifier: kill every process of the agent user, the kept
    services included, then reopen the log folders the release closed
    further (K2-R3-A-1). Records ``kept_services_ended`` (how many were
    killed). A failure is logged: grading is over, and the folders stay
    closed."""
    opened = tuple(
        (row["path"], row["owner"], row["mode"])
        for row in record.get("kept_services_closed_folders") or ()
    )
    try:
        killed = await planes.end_kept_services(env, sandbox_user, opened)
    except Exception as exc:
        logger.warning(f"Ending the kept services after grading failed: {exc}")
        record["kept_services_ended"] = f"failed: {exc}"
    else:
        record["kept_services_ended"] = killed
    _write_kill_record(record_dir, record)


def _write_kill_record(record_dir: str | Path, record: dict[str, Any]) -> None:
    """Write the record of the kill before grading into the trial folder."""
    try:
        kill_record_module.write_kill_record(record_dir, record)
    except Exception:
        logger.debug("kill-before-grading record not written", exc_info=True)


async def _verify_rollout(
    env: Any,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    planes: RolloutPlanes,
    sandbox_user: str | None = None,
    workspace: str | None = None,
    *,
    recovery_eligible: bool = False,
    agent_paths: tuple[str, ...] | None = None,
    kill_record: dict[str, Any] | None = None,
    record_dir: str | Path | None = None,
) -> tuple[dict | None, str | None, Diagnostic | None]:
    """Run verifier with pre-verification hardening (:func:`_verify_hardened`).

    From the end of hardening on, root's commands find their programs on the
    verifier's safe PATH, never on the image's (V3-CONFIRM-1): the Verifier's
    upload, ``chmod +x`` of test.sh, receipt and output folders, the kill of
    a timed-out test command and the backends' own helpers would otherwise
    run a program the agent planted in an image PATH folder it could write,
    as root.
    """
    with contextlib.ExitStack() as root_path:
        return await _verify_hardened(
            env,
            task,
            rollout_paths,
            timing,
            planes,
            sandbox_user,
            workspace,
            recovery_eligible=recovery_eligible,
            agent_paths=agent_paths,
            kill_record=kill_record,
            record_dir=record_dir,
            root_path=root_path,
        )


async def _verify_hardened(
    env: Any,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    planes: RolloutPlanes,
    sandbox_user: str | None,
    workspace: str | None,
    *,
    recovery_eligible: bool,
    agent_paths: tuple[str, ...] | None,
    kill_record: dict[str, Any] | None = None,
    record_dir: str | Path | None = None,
    root_path: contextlib.ExitStack,
) -> tuple[dict | None, str | None, Diagnostic | None]:
    """Run verifier with pre-verification hardening.

    Returns ``(rewards, verifier_error, verifier_diagnostic)``. The
    diagnostic is a :class:`VerifierTimeoutDiagnostic` when the verifier
    exceeded its timeout budget, a :class:`RewardOutOfRangeDiagnostic` when it
    wrote a reward outside the task's range (the score is 0 and the raw number
    is kept), and ``None`` otherwise — the agent-error channel
    is unused (issue #503).

    Every script verifier's command writes a start receipt (#1136): a command
    whose start cannot be confirmed is retried once in place, then scores 0
    with a warning (:func:`_await_verifier`). Recovery-eligible tasks skip the
    in-place retry. The verifier user can remove or forge the receipt.

    ``agent_paths`` marks a separate verifier sandbox and lists the paths its
    transfer wrote; hardening distrusts only those there.

    ``kill_record`` is what an earlier kill in this verify (the quiesce
    before a workspace capture) recorded; the hardening kill adds to it.
    ``record_dir`` is the folder that record is written to (default
    ``rollout_paths.rollout_dir``). A branch child's host writers follow the
    child's own folder while ``rollout_paths`` stays on the parent's root, so
    ``Rollout.verify`` passes the folder it writes to.
    """
    rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    verifier_error = None
    verifier_diagnostic: Diagnostic | None = None
    timeout_budget = task.config.verifier.timeout_sec
    verifier = None
    path_reports: list[Any] = []
    try:
        harden: dict[str, Any] = {"workspace": workspace}
        if agent_paths is not None:
            harden["agent_paths"] = agent_paths
        # Each hardening step records its sub-key; total is the whole call.
        # The kill before grading records whom it stopped and how many it
        # killed (K1), added to what the quiesce before a capture killed; a
        # trial with no sandbox user runs no kill and gets no record.
        kill_counts: dict[str, Any] = dict(kill_record or {})
        try:
            with (
                timing_spans.recording_into(timing),
                recording_trusted_path() as path_reports,
                kill_record_module.recording_into(kill_counts),
                timing_spans.span(
                    timing_spans.HARDEN_BEFORE_VERIFY, "total", into=timing
                ),
            ):
                # No CPU lift here (PERF-SBX reviews A M1 and B M2): processes
                # the task started (a root agent's leftovers, a root
                # solution's kept service) may still run in this window, and
                # they must see the task's own CPU limit. The lift stays
                # only around install_agent, before any agent process exists.
                await planes.harden_before_verify(env, task, sandbox_user, **harden)
        finally:
            if kill_counts:
                if record_dir is None:
                    record_dir = rollout_paths.rollout_dir
                _write_kill_record(record_dir, kill_counts)
        _warn_verifier_dropped_image_env(env, task)
        # The sandbox's own writes so far go to disk while the tests run, not
        # in its teardown (PERF-SBX); a backend without it does nothing.
        flush = sandbox_hook(env, "flush_writes_in_background")
        if flush is not None:
            flush()
        logger.info("Running verifier...")
        verifier = planes.verifier(
            task=task,
            rollout_paths=rollout_paths,
            sandbox=env,
            execution_receipt=True,
        )
        # task.md's omitted workdir means the workspace resolved at startup,
        # including an image cwd or the runtime's fallback directory.
        verifier.taskmd_workdir = workspace
        kept_pids = tuple(
            int(row["pid"]) for row in kill_counts.get("kept_services") or ()
        )
        if kept_pids and sandbox_user:
            assert record_dir is not None
            # The services the task keeps were left frozen by the kill; they go
            # on just before the test command, with the tests closed to them.
            verifier.before_test = functools.partial(
                _release_kept_services,
                planes,
                env,
                sandbox_user,
                kept_pids,
                kill_counts,
                record_dir,
            )
        # While kept services run beside root's commands, exec adds no
        # persistent env and copy-out changes no owner in the sandbox
        # .
        services_kept = getattr(env, "services_kept", None)
        with (
            services_kept()
            if kept_pids and sandbox_user and callable(services_kept)
            else contextlib.nullcontext()
        ):
            try:
                verifier_result = await _await_verifier(
                    env, verifier, timeout_budget, retry_in_place=not recovery_eligible
                )
            finally:
                if sandbox_user and "kept_services_released" in kill_counts:
                    # Nothing kept lives past grading (K2-R3-A-1).
                    assert record_dir is not None
                    await _end_kept_services(
                        planes, env, sandbox_user, kill_counts, record_dir
                    )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        try:
            rewards = _ensure_canonical_rewards(verifier_result.rewards, task=task)
        except RewardOutOfRange:
            raise
        except ValueError as e:
            raise RewardFileParseError(f"Invalid verifier rewards: {e}") from e
        label = getattr(task, "step_label", None)
        logger.info(
            f"Rewards of the final verify ({label}): {rewards}"
            if label
            else f"Rewards: {rewards}"
        )
    except (TimeoutError, ExecTimeoutError) as exc:
        elapsed = (datetime.now() - t0).total_seconds()
        timing["verifier"] = elapsed
        verifier_error = f"verifier timed out after {timeout_budget}s"
        # Tests still running at the deadline score 0: a timeout test.sh's
        # code can cause never discards the assessment. When the sandbox did
        # not answer whether they had ended, the outcome is unknown and stays
        # unscored, a verifier timeout that is never retried.
        scored = not isinstance(exc, _TestEndUnknown) and (
            getattr(env, "_benchflow_test_command_issued", False) is True
            or (verifier is not None and getattr(verifier, "test_command_issued", True))
        )
        verifier_diagnostic = VerifierTimeoutDiagnostic(
            timeout_budget_sec=timeout_budget,
            elapsed_sec=round(elapsed, 1),
            task_name=task.name,
            scored=scored,
        )
        if scored:
            rewards = _verifier_zero(verifier_error)
            verifier_error = None
            await _keep_timed_out_output(
                env, task, verifier, rollout_paths, verifier_diagnostic
            )
        else:
            if isinstance(exc, _TestEndUnknown):
                verifier_error += (
                    "; the sandbox did not answer whether the test command had "
                    f"ended ({exc})"
                )
            rewards = None
            logger.error(verifier_error)
    except RewardOutOfRange as e:
        # A named problem, not a crash: the verifier ran and gave a number
        # the task does not allow. Never clamped, never retried.
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_diagnostic = RewardOutOfRangeDiagnostic.from_error(e)
        rewards = _verifier_zero(f"{REWARD_OUT_OF_RANGE} {e}")
    except PluginGuardMarkersUnreadableError as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        # BenchFlow's own read failed, or its output says the markers are
        # incomplete: the warning names which, not the guard verdicts the
        # markers would have held, none of which was read.
        rewards = _verifier_zero(
            f"pytest plugin guard not confirmed: {e}; test.sh's reward is not "
            "used. See verifier/test-stdout.txt"
        )
    except PluginGuardLoadError as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = _verifier_zero(
            "pytest plugin guard not confirmed: it may have failed to load, "
            "stopped or refused a plugin; code run by test.sh can also print "
            "the diagnostic or change the markers. See verifier/test-stdout.txt "
            "and verifier/_runtime_guard_*" + e.guard_copy_diagnostic
        )
    except VerifierSetupError as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        if getattr(
            env, "_benchflow_test_command_issued", False
        ) is True and not verifier_infrastructure_failure(e):
            rewards = _verifier_zero(f"setup after an earlier test command failed: {e}")
        else:
            verifier_error = f"verifier crashed: {verifier_failure_reason(e)}"
            rewards = None
            logger.error(verifier_error)
    except VerifierInfrastructureError as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier crashed: {verifier_failure_reason(e)}"
        rewards = None
        logger.error(verifier_error)
    except (
        VerifierPathEvidenceError,
        RewardFileEmptyError,
        RewardFileNotFoundError,
        VerifierOutputParseError,
        VerifierStartUnconfirmedError,
    ) as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        if verifier_infrastructure_failure(e):
            verifier_error = f"verifier crashed: {verifier_failure_reason(e)}"
            rewards = None
            logger.error(verifier_error)
        elif isinstance(
            e, (RewardFileEmptyError, RewardFileNotFoundError, RewardFileParseError)
        ):
            # No reward BenchFlow can use: the verifier's error, not a 0 that
            # reads as a wrong answer. Never retried, and still a failure in
            # every score (scoring.NO_USABLE_REWARD).
            verifier_error = f"{NO_USABLE_REWARD}{e}"
            rewards = None
            logger.error(verifier_error)
        else:
            rewards = _verifier_zero(str(e))
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        # describe_exception, not str(e), for the reason the agent-side funnel
        # already documents: an exception raised with no args stringifies to
        # nothing, so the recorded error names neither the failure nor the fact
        # that it had no detail. A teardown ProcessLookupError landing here is
        # exactly that shape, and it arrives in place of the timeout it
        # displaced (#1065).
        if not verifier_infrastructure_failure(e) and (
            getattr(env, "_benchflow_test_command_issued", False) is True
            or (
                verifier is not None
                and getattr(verifier, "test_command_issued", False) is True
            )
        ):
            rewards = _verifier_zero(
                f"verification after a test command failed: {describe_exception(e)}"
            )
        else:
            verifier_error = f"verifier crashed: {verifier_failure_reason(e)}"
            rewards = None
            logger.error(verifier_error)
    report = path_reports[-1] if path_reports else None
    if report is not None and (
        report.dropped or report.kept_other or report.omitted_tools
    ):
        refused = await _refused_programs_run(env, report)
        missed = _verifier_missed_a_program(verifier, rollout_paths)
        _write_trusted_path_record(rollout_paths, report, missed, refused)
        # Only the trusted task author can declare a probe optional. A failed
        # negative check can award full marks, and an independent check can
        # award partial marks; neither score proves dependency (RECONFIRM-3).
        optional = getattr(task.config.verifier, "optional_tools", ())
        optional = optional if isinstance(optional, list | tuple) else ()
        required = [
            (name, folder)
            for name, folder in refused
            if name not in optional or (name, folder) not in report.stubs
        ]
        if required:
            # A refusal stub ran: the verifier needed a program the image has
            # only in a folder left out, whatever test.sh printed or exited
            # with.
            warning = (
                "verifier PATH refusal: a refusal marker names "
                + ", ".join(f"{name} (in {folder})" for name, folder in required)
                + ", which only folders the verifier's PATH leaves out have: "
                + report.dropped_text()
            )
            if verifier_error is None:
                rewards = _verifier_zero(warning)
            else:
                logger.warning(warning)
                report_trial_warning(warning)
        elif missed and report.dropped and verifier_error is not None:
            # Output that looks like a missing program is no evidence on its
            # own (an honest test can print "command not found"); it only
            # names the folders in an error the verifier already has.
            verifier_error += (
                "; the verifier's PATH left out these folders of the image's "
                f"PATH: {report.dropped_text()}"
            )
    return rewards, verifier_error, verifier_diagnostic


def _verifier_zero(reason: str) -> dict[str, float]:
    """Verifier-controlled evidence may lower a score, never remove a trial."""
    warning = f"Verifier scored 0: {reason}"
    logger.warning(warning)
    report_trial_warning(warning)
    return {"reward": 0.0}


async def _refused_programs_run(env: Any, report: Any) -> list[tuple[str, str]]:
    """The refusal stubs that ran, with their folders (``trusted_path``).

    A stubs folder that cannot be read is a refusal too: the hits are the
    only evidence, so not reading them must not leave the score standing.
    """
    command = refused_programs_cmd(report)
    if command is None:
        return []
    try:
        # In the service test.sh ran in, where the stubs are (V3-CONFIRM-5).
        service = getattr(report, "service", "main")
        where = {} if service == "main" else {"service": service}
        result = await receipt_exec(env, command, user="root", timeout_sec=30, **where)
        code = getattr(result, "return_code", None)
        if code is None:
            code = getattr(result, "exit_code", 0)
        if code != 0:
            raise RuntimeError(f"rc={code}")
        return refused_programs(report, getattr(result, "stdout", "") or "")
    except Exception as exc:
        logger.warning("could not read the verifier's refusal stubs: %s", exc)
        return [("(refusal stubs unreadable)", report.refusals)]


# How a shell says it could not find a program: bash's "command not found",
# dash's and busybox's "<name>: not found".
_PROGRAM_NOT_FOUND_RE = re.compile(r"command not found|: not found\s*$", re.MULTILINE)
_TEST_STDOUT_SCAN_BYTES = 4 * 1024 * 1024


def _verifier_missed_a_program(verifier: Any, rollout_paths: Any) -> bool:
    """Whether the verifier exited 127 or its output says a program was missing.

    The verifier folder is the sandbox's ``/logs/verifier``, which code the
    tests ran could write: ``test-stdout.txt`` may be a link (to a host file,
    to ``/dev/zero``), a FIFO or a device by now. Only a regular file is
    read, without following a link, and only its last
    ``_TEST_STDOUT_SCAN_BYTES``.
    """
    if getattr(verifier, "test_return_code", None) == 127:
        return True
    path = getattr(rollout_paths, "test_stdout_path", None)
    if path is None:
        return False
    try:
        # The verifier's outputs are held capabilities after integration;
        # opening through the capability also protects every parent and rejects
        # links/special files before reading. Keep the bounded tail scan.
        with destination(path).open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                return False
            stream.seek(max(0, opened.st_size - _TEST_STDOUT_SCAN_BYTES))
            chunks: list[bytes] = []
            left = _TEST_STDOUT_SCAN_BYTES
            while left > 0:
                chunk = stream.read(min(left, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                left -= len(chunk)
    except OSError:
        return False
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return _PROGRAM_NOT_FOUND_RE.search(text) is not None


def verifier_dropped_image_env(env: Any, task: Any) -> list[str]:
    """The names the verifier's service set itself (image ENV or the task's
    Compose environment) that test.sh gets cleared: V3's trusted verifier
    environment empties LD_LIBRARY_PATH, LD_PRELOAD, PYTHONSTARTUP, ...,
    whatever the image said, because a library or module path the agent
    can reach would run its code as root. Names only."""
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    final = getattr(verifier, "env", None)
    if not isinstance(final, dict):
        return []
    service = getattr(verifier, "service", None)
    service = service if isinstance(service, str) and service else "main"
    own = _inert_bootstrap.image_env(getattr(env, "_strategy", None) or env, service)
    return sorted(name for name in own if final.get(name) == "")


def _warn_verifier_dropped_image_env(env: Any, task: Any) -> None:
    """Record, in the trial's result, each image name test.sh ran without.
    A warning never fails the verify."""
    try:
        dropped = verifier_dropped_image_env(env, task)
    except Exception:
        logger.debug("verifier image-env names not read", exc_info=True)
        return
    if not dropped:
        return
    text = (
        f"test.sh ran without the image's {', '.join(dropped)}: the root "
        "verifier clears these names, so a test that runs the image's own "
        "tools with them fails; set the value the test needs in test.sh or "
        "[verifier.env]"
    )
    logger.warning(text)
    report_trial_warning(text)


def _write_trusted_path_record(
    rollout_paths: Any,
    report: Any,
    missed: bool,
    refused: list[tuple[str, str]] | None = None,
) -> None:
    """``verifier/trusted-path.json``: the folders of the image's PATH the
    verifier's PATH left out and why, the ones it kept that root does not
    own, whether the verifier's output looked like a missing program, and
    the refused programs it ran.

    Use the held verifier-folder capability: staging stays outside every
    sandbox-reachable folder, and publication replaces the leaf through the
    held directory. A test's symlink, hard link or renamed ancestor must not
    redirect the host write (V3-R1-B-1, V3-R2-A-1).
    """
    record = {
        **report.to_record(),
        "verifier_missed_a_program": missed,
        "refused_programs_run": [
            {"program": name, "folder": folder} for name, folder in refused or ()
        ],
    }
    try:
        destination(rollout_paths.verifier_dir / "trusted-path.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
    except OSError:
        logger.warning("could not write verifier/trusted-path.json", exc_info=True)


# POSIX sh that sets ``$held`` to the processes around a running test
# command: those whose command line names the capture file
# /logs/verifier/test-stdout.txt, which every test command redirects into.
# It reads /proc because task images often lack procps. The ``[r]`` keeps the
# pattern from matching this script's own command line, and stderr is dropped
# because the Docker sandbox merges it into the stdout parsed here.
# Sets ``$held`` (when it is empty) to the processes that have test.sh itself
# as a whole argument: the roots where no process names the capture file.
_TEST_SH_ROOTS_SH = r"""
t='/verifie[r]/test[.]sh'; u='/test[s]/test[.]sh'
[ -n "$held" ] || for f in $(grep -l -e "$t" -e "$u" /proc/[0-9]*/cmdline); do
  tr '\0' '\n' < "$f" | grep -qx -e "$t" -e "$u" || continue
  f=${f%/cmdline}
  held="$held ${f#/proc/}"
done
"""

_VERIFIER_ROOTS_SH = (
    r"""
exec 2>/dev/null
held=
for f in $(grep -l '/logs/verifie[r]/test-stdout[.]txt' /proc/[0-9]*/cmdline); do
  f=${f%/cmdline}
  held="$held ${f#/proc/}"
done
# A shell that runs test.sh in place of itself (busybox sh -c with a single
# redirected command) leaves no process naming the capture file, so fall
# back to a process that has test.sh itself as a whole argument.
"""
    + _TEST_SH_ROOTS_SH
)

_VERIFIER_PROBE_SH = (
    _VERIFIER_ROOTS_SH
    + r"""
if [ -n "$held" ]; then echo running
else cd /logs/verifier && wc -c < test-stdout.txt || echo 0
fi
"""
)

# Stop the test command's whole process tree, then kill it. ``grow`` adds
# every live descendant of a held PID (test.sh, pytest and whatever those
# started) in one awk pass over /proc; ``cat`` feeds it because awk gives up
# on the first file that vanished. Parents come from ``status``, not
# ``stat``, which prints the command name raw: a name holding a newline
# could pass off a fake record. Roots are never matched again, so a retry
# that starts meanwhile is left alone. Each round stops what is held and
# looks again until no new child appears, so a process forked mid-scan
# cannot escape, and every stopped PID stays held, so a child reparented
# when its parent exits is killed rather than left stopped. The last line
# is how many processes were killed.
_KILL_TREE_SH = r"""
grow() {
  grown=$(cat /proc/[0-9]*/status | awk -v held="$held" '
    $1 == "Pid:" { p = $2 }
    $1 == "PPid:" { parent[p] = $2 }
    END {
      out = held
      n = split(held, h, " ")
      for (i = 1; i <= n; i++) keep[h[i]] = 1
      do {
        more = 0
        for (p in parent)
          if (!(p in keep) && (parent[p] in keep)) {
            keep[p] = 1; out = out " " p; more = 1
          }
      } while (more)
      print out
    }')
  [ -z "$grown" ] || held=$grown
}
grow
stopped=
while [ "$held" != "$stopped" ]; do
  stopped=$held
  kill -s STOP $held
  grow
done
[ -z "$held" ] || kill -s KILL $held
set -- $held
echo "killed $#"
"""

_KILL_VERIFIER_SH = _VERIFIER_ROOTS_SH + _KILL_TREE_SH

# The name of a test command's start receipt (Verifier._receipt_after).
_RECEIPT_NAME_RE = re.compile(r"verifier-started-[0-9a-f]{32}")


def _kill_verifier_command(receipt: Any) -> str:
    """:data:`_KILL_VERIFIER_SH`, with the roots narrowed to this attempt's
    processes when it has a start receipt: those whose command line names
    the receipt, and those whose environment holds it under
    ``VERIFIER_ATTEMPT_ENV``, which the test command exports. The name is
    unique to the attempt, so no other test command (another attempt's,
    another sandbox's on a shared host) is ever taken. A command too long for
    a command line is sent as data and names the receipt on none; its tests
    still carry it in their environment. Root reads another user's
    environment only with CAP_SYS_PTRACE, which Docker's default set lacks,
    so tests that run as another user and were sent as data are found by
    neither and run on."""
    path = receipt[0] if isinstance(receipt, tuple) and len(receipt) == 2 else None
    name = PurePosixPath(path).name if isinstance(path, str) else ""
    if not _RECEIPT_NAME_RE.fullmatch(name):
        return _KILL_VERIFIER_SH
    # The bracket keeps each pattern from matching this script's own line.
    pattern = f"{name[:-1]}[{name[-1]}]"
    marker = f"{VERIFIER_ATTEMPT_ENV}={pattern}"
    return (
        "exec 2>/dev/null\nheld=\n"
        f"for f in $(grep -l '{pattern}' /proc/[0-9]*/cmdline); do\n"
        '  f=${f%/cmdline}\n  held="$held ${f#/proc/}"\ndone\n'
        f"for f in $(grep -l '{marker}' /proc/[0-9]*/environ); do\n"
        f"  tr '\\0' '\\n' < \"$f\" | grep -qx '{marker}' || continue\n"
        '  f=${f%/environ}\n  held="$held ${f#/proc/}"\ndone\n' + _KILL_TREE_SH
    )


def _verifier_service(task: Any) -> str:
    """The service test.sh runs in (``[verifier].service``, ``main`` unless set)."""
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    service = getattr(verifier, "service", None)
    return service if isinstance(service, str) and service else "main"


async def _verifier_wedged_without_output(env: Any) -> bool:
    """True when the timed-out verifier attempt never got test.sh running.

    A wedged exec layer leaves no verifier process behind and zero bytes in
    ``/logs/verifier/test-stdout.txt``. Empty output alone does not show
    that: a test script that sends pytest's output to its own log and copies
    it to stdout only at the end has empty output while it runs, but its
    processes are alive (:data:`_VERIFIER_PROBE_SH`). When even this probe
    fails, the control plane is still sick and a retry is the best remaining
    move, so failures count as wedged.
    """
    try:
        r = await asyncio.wait_for(
            exec_raw(
                env.exec,
                _VERIFIER_PROBE_SH,
                # As root, so it is on the trusted PATH (V3-CONFIRM-1) and
                # sees processes where /proc is mounted with hidepid.
                user="root",
                timeout_sec=10,
            ),
            timeout=30,
        )
        lines = (r.stdout or "").strip().splitlines()
        out = lines[-1].strip() if lines else ""
        return out != "running" and int(out or 0) == 0
    except Exception:
        return True


async def _kill_verifier_tree(
    env: Any, service: str = "main", receipt: Any = None
) -> int | None:
    """Kill the test command's whole process tree (:data:`_KILL_VERIFIER_SH`).

    A host-side timeout cancels only our await: test.sh, pytest and their
    children run on in the sandbox. *receipt* is the attempt's start receipt
    (``Verifier.execution_receipt``), which narrows the roots to its command.
    Returns how many processes were killed, or None when the sandbox did not
    say. Needs no procps, only ``sh``, ``grep``, ``tr``, ``cat`` and ``awk``;
    without ``awk`` only the roots die.
    """
    where = {} if service == "main" else {"service": service}
    try:
        r = await asyncio.wait_for(
            exec_raw(
                env.exec,
                _kill_verifier_command(receipt),
                # As root: the tests may run as another user.
                user="root",
                timeout_sec=10,
                **where,
            ),
            timeout=30,
        )
    except Exception as exc:
        logger.warning(f"Killing the verifier's processes failed: {exc!r}")
        return None
    stdout = getattr(r, "stdout", None)
    lines = stdout.strip().splitlines() if isinstance(stdout, str) else []
    killed = re.fullmatch(r"killed (\d+)", lines[-1].strip()) if lines else None
    return int(killed.group(1)) if killed else None


async def _kill_orphan_verifier(env: Any, verifier: Any = None) -> None:
    """Kill a possibly orphaned first verifier attempt before the retry.

    If test.sh did start late it would race the retry's run, and killing it
    alone would leave pytest and its children running beside the retry, so
    the whole tree goes.
    """
    receipt = getattr(verifier, "execution_receipt", None)
    service = receipt[1] if isinstance(receipt, tuple) and len(receipt) == 2 else None
    await _kill_verifier_tree(
        env, service if isinstance(service, str) else "main", receipt
    )


# A scored timeout keeps what the tests wrote before their deadline: the
# kill, the copy of /logs/verifier and the renames get this long in all.
_TIMED_OUT_OUTPUT_BUDGET_SEC = 180.0


async def _keep_timed_out_output(
    env: Any,
    task: Any,
    verifier: Any,
    rollout_paths: Any,
    diagnostic: VerifierTimeoutDiagnostic,
) -> None:
    """Tests ran past their deadline: stop them and keep what they wrote.

    The tests' process tree goes first, so nothing writes while their folder
    is copied and no reward lands after the score. Then ``/logs/verifier``
    comes into ``verifier/`` as a finished test command's does, its reward
    files renamed ``timeout-<name>`` (:meth:`Verifier.save_timed_out_outputs`).
    *diagnostic* records the processes killed and the files and bytes kept,
    or why none were. The score stays 0 whatever happens here.
    """
    save = getattr(verifier, "save_timed_out_outputs", None)
    savable = getattr(verifier, "test_command_savable", False) is True
    if not callable(save) or not savable:
        return

    async def keep() -> None:
        diagnostic.processes_killed = await _kill_verifier_tree(
            env,
            _verifier_service(task),
            getattr(verifier, "execution_receipt", None),
        )
        await save()

    try:
        await asyncio.wait_for(keep(), timeout=_TIMED_OUT_OUTPUT_BUDGET_SEC)
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            why = f"not done within {_TIMED_OUT_OUTPUT_BUDGET_SEC:g}s"
        else:
            why = describe_exception(exc)
        diagnostic.partial_output_error = why[:_DIAG_TRUNCATE]
        logger.warning(f"Keeping the timed-out verifier's output failed: {why}")
    files, size = _verifier_output_size(rollout_paths)
    diagnostic.partial_output_files = files
    diagnostic.partial_output_bytes = size


_OUTPUT_COUNT_LIMIT = 100_000


def _verifier_output_size(rollout_paths: Any) -> tuple[int, int]:
    """The regular files under the rollout's ``verifier/`` and their bytes.

    Read without following links, at most ``_OUTPUT_COUNT_LIMIT`` entries.
    """
    root = getattr(rollout_paths, "verifier_dir", None)
    if root is None:
        return 0, 0
    files = size = seen = 0
    folders = [root]
    while folders and seen < _OUTPUT_COUNT_LIMIT:
        folder = folders.pop()
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            try:
                st = entry.lstat()
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                folders.append(entry)
            elif stat.S_ISREG(st.st_mode):
                files += 1
                size += st.st_size
    return files, size


def _ensure_canonical_rewards(rewards: dict | None, *, task: Any = None) -> dict:
    # Honour the same BENCHFLOW_REWARD_LENIENT toggle and task-declared
    # ``[verifier] reward_range`` (BF-8) as the reward.json parse path so the
    # final canonicalization gate stays consistent with how the verifier
    # accepted the map (no-op unless the operator/task opts in).
    return validate_reward_map(
        rewards,
        source="verifier",
        lenient=reward_lenient_from_env(),
        reward_range=declared_reward_range(task),
    )


def _install_docker_compat(planes: RolloutPlanes | None = None) -> None:
    """Activate the Docker DinD compatibility shim.

    Called from ``Rollout.__init__`` so importing ``benchflow.rollout`` has
    no side effects on the Docker sandbox. The underlying patch is
    idempotent — safe to call once per rollout construction.
    """
    (planes or default_rollout_planes()).install_docker_compat()
