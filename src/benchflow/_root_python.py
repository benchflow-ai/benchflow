"""Root's Python helpers, through one entry point (C8, PY8-PT-R8-5 / PT-C8-R8-2).

Every stdlib-only helper BenchFlow runs as root (or verifier-side, in a
sandbox the agent's files reach) starts here, so none imports a module the
agent wrote:

- the interpreter is the protected kit's (:data:`~benchflow._runtime_tree.KIT_PYTHON`)
  once the sandbox has a kit; a kit without a Python refuses the step
  (status 127, a ``benchflow:`` line, which callers with a Python-free
  fallback read as a missing ``python3``). A sandbox with no kit has no
  sandbox user (nobody but root ever ran there) and runs the system
  ``python3`` on root's route;
- always with ``-I -S -B``: no script or current folder on ``sys.path``, no
  ``site`` (so no ``.pth`` or ``sitecustomize``), no ``PYTHON*`` value
  read, no bytecode written;
- the helper's own code starts after :data:`PRELUDE`, which moves to ``/``
  (outside any workspace) and drops every ``PYTHON*`` name from the
  environment the helper could hand on.

:func:`root_python_command` (async, knows the sandbox's kit) gives an argv
string; :func:`root_python_sh` (a static recipe, decides in the sandbox)
gives shell text. A source scan (``tests/test_c8_r10_root_python.py``) holds
every root Python launch in BenchFlow to these two.
"""

from __future__ import annotations

import inspect
import shlex
from typing import Any

from benchflow._runtime_tree import KIT_PYTHON, KIT_RECORD, RuntimeKit

__all__ = [
    "FLAGS",
    "NO_KIT_PYTHON",
    "PRELUDE",
    "root_python_argv",
    "root_python_command",
    "root_python_pick_sh",
    "root_python_sh",
]

FLAGS = ("-I", "-S", "-B")

PRELUDE = (
    "import os as _bf_os\n"
    "_bf_os.chdir('/')\n"
    "[_bf_os.environ.pop(_bf_k) for _bf_k in list(_bf_os.environ)"
    " if _bf_k.startswith('PYTHON')]\n"
    "del _bf_os\n"
)

# What a root Python step prints when the sandbox's kit holds no Python.
NO_KIT_PYTHON = (
    "echo 'benchflow: python3 not found in the protected runtime, so this "
    "root step does not run' >&2; exit 127"
)


def _script(script: str) -> str:
    """*script* with :data:`PRELUDE` first: after its ``__future__``
    imports when it has any (they must come first; a module docstring
    before them stays first too)."""
    lines = script.splitlines(keepends=True)
    after = 0
    for index, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            after = index + 1
    if after:
        return "".join(lines[:after]) + PRELUDE + "".join(lines[after:])
    return PRELUDE + script


def root_python_argv(kit: RuntimeKit | None) -> list[str] | None:
    """The interpreter and flags for a sandbox holding *kit*: the kit's
    Python, ``python3`` without a kit, None (refuse) for a kit without one."""
    if kit is None:
        return ["python3", *FLAGS]
    if not kit.python:
        return None
    return [KIT_PYTHON, *FLAGS]


async def _sandbox_kit(env: Any) -> RuntimeKit | None:
    lookup = getattr(env, "runtime_kit_for_route", None)
    if inspect.iscoroutinefunction(lookup):
        kit = await lookup()
    else:
        kit = getattr(env, "runtime_kit", None)
    return kit if isinstance(kit, RuntimeKit) else None


async def root_python_command(env: Any, script: str, *args: str) -> str:
    """The command that runs *script* (``-c`` text) with *args* as root in
    *env*, on the protected interpreter, or the refusal."""
    argv = root_python_argv(await _sandbox_kit(env))
    if argv is None:
        return NO_KIT_PYTHON
    return shlex.join([*argv, "-c", _script(script), *args])


def root_python_sh(
    script: str, *args: str, raw_args: str = "", fallback: str = "python3"
) -> str:
    """Shell text for a static recipe (it cannot ask the sandbox first): the
    same rule decided in the sandbox, from the kit's RECORD. A subshell, so
    it can stand anywhere a command can.

    *raw_args*: shell words appended unquoted (``"$1"``), for a recipe that
    passes its own positional values. *fallback*: the interpreter without a
    kit, as a shell word (a pre-agent copy of the image's ``python3``, for a
    check that must not start the live one).
    """
    run = shlex.join(["-c", _script(script), *args])
    if raw_args:
        run = f"{run} {raw_args}"
    flags = " ".join(FLAGS)
    return f'( {root_python_pick_sh(fallback=fallback)}; exec "$bf_py" {flags} {run} )'


def root_python_pick_sh(
    var: str = "bf_py", *, fallback: str = "python3", refuse: str = NO_KIT_PYTHON
) -> str:
    """Shell text that sets *var* to root's interpreter, decided in the
    sandbox as :func:`root_python_sh` decides: the kit's Python once the
    kit's RECORD exists (*refuse*, a command that exits, when the kit holds
    none), else *fallback* (a sandbox without a kit: before the kit is
    built, or one only root ever ran in). For a step that starts its own
    script file rather than ``-c`` text (PT-C8-CONF-2: the egress proxy)."""
    return (
        f"if [ -f {KIT_RECORD} ]; then [ -x {KIT_PYTHON} ] || "
        f"{{ {refuse}; }}; {var}={KIT_PYTHON}; else {var}={fallback}; fi"
    )
