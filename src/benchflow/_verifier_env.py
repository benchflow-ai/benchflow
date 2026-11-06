"""The environment every verifier command in a sandbox runs with.

Hardening (``lockdown._build_verifier_env``) assembles the verifier's full
environment before a scored run. Some verifier paths never harden: soft
verification between user rounds, an SDK caller's own ``Verifier``. So the
invariants that need no probe of the sandbox are applied twice: once when
hardening assembles the environment (:func:`pin_verifier_env`), and again in
the verifier's own command (:func:`pinned_shell_prefix`), at every place
``Verifier`` runs a command in the sandbox (the test script and a verifier
strategy's runner). Hard, soft, retried and separate-sandbox verification on
Docker and Daytona all reach the sandbox only through those places, so none of
them can run the verifier without these values, whatever the task's
``[verifier] env`` says.

The command carries them as shell builtins (``export NAME=value``), never in
the exec's ``env``: a non-empty ``env`` makes a backend decode it through a
file under ``/tmp`` with a ``base64`` found on the image's PATH (b127
PY8-MAIN-25), and the values are no secret. The prefix runs after whatever
environment the exec sets up, so it also wins over the task's.

``PIP_CONFIG_FILE=/dev/null``: pip loads no configuration file at all, not a
per-user file under an agent-written HOME, not ``/etc/pip.conf``, not a venv's
site file, not a ``python =`` redirect. A task that needs a private index or
an offline wheelhouse passes pip options explicitly in its own verifier
script.

``BASH_ENV`` and ``ENV``, always: the file a shell would run before test.sh
is the value a trust check validated (hardening's on the scored path, the same
policy on the soft path, none for a ``Verifier`` nobody checked), never the
task's raw ``[verifier.env]`` value (b127 PY8-MAIN-26, b143 PY8-MAIN-28).

``COVERAGE_PROCESS_START`` and ``COVERAGE_PROCESS_CONFIG``, unset: coverage.py's
start-up hook (``coverage.process_startup()``, which a test package, a
``sitecustomize`` or a ``.pth`` file may call) starts measuring whenever either
is defined, even empty, and an empty start variable only skips the config
file. Unset, the hook returns before it reads anything, measures nothing and
writes no data file into the working folder. Hardening's
exec env still names the start variable empty, for commands without this
prefix.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from typing import Any

# Set last, over the task's own values; nothing replaces them.
VERIFIER_PINNED_ENV: dict[str, str] = {"PIP_CONFIG_FILE": os.devnull}

# A test command with a start receipt exports the receipt's name under this
# name, so each process of that attempt carries it in /proc/<pid>/environ,
# also when the command was sent as data and names the receipt on no command
# line: the kill of a timed-out attempt finds them by it
# (rollout._setup._kill_verifier_command), where root may read their
# environment (its own processes', or any with CAP_SYS_PTRACE). No secret:
# the command's own text names the receipt.
VERIFIER_ATTEMPT_ENV = "BENCHFLOW_VERIFIER_ATTEMPT"

# Removed from the verifier's environment; never set, not even empty.
VERIFIER_UNSET_ENV: tuple[str, ...] = (
    "COVERAGE_PROCESS_START",
    "COVERAGE_PROCESS_CONFIG",
)

# What a task that needs pip configuration in its verifier does instead.
PIP_CONFIG_FIX = (
    "A task that needs a private index or an offline wheelhouse should pass "
    "pip options explicitly in its verifier script (for example "
    "`pip install --no-index --find-links /opt/wheels ...`)."
)


def pin_verifier_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """*env* with :data:`VERIFIER_PINNED_ENV` set over it."""
    return {**(env or {}), **VERIFIER_PINNED_ENV}


def sandbox_verifier_env(task: Any) -> dict[str, str] | None:
    """The exec env for a verifier command: the task's own, as before (or None).

    The pins ride in the command (:func:`pinned_shell_prefix`), not here.
    """
    from benchflow.task.env import resolve_env_vars

    configured = task.config.verifier.env
    return resolve_env_vars(configured) if configured else None


def pinned_shell_prefix(env: Mapping[str, str] | None, *, bash_env: str = "") -> str:
    """Shell builtins that set the pins first in a verifier command.

    ``BASH_ENV`` is always *bash_env*, the value a trust check validated
    (hardening's, or soft verification's with hardening's policy), never
    what the exec env names: a task's ``[verifier.env] BASH_ENV`` is raw
    configuration and its file may be agent-written (b143 PY8-MAIN-28).
    ``ENV`` is always empty, as hardening sets it. :data:`VERIFIER_UNSET_ENV`
    is unset. *env* is accepted for the callers that pass the exec env along.
    """
    del env
    pins = dict(VERIFIER_PINNED_ENV)
    pins["BASH_ENV"] = bash_env
    pins["ENV"] = ""
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in pins.items())
    return exports + f"unset {' '.join(VERIFIER_UNSET_ENV)}; "
