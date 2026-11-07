"""The Python interpreter the workspace freeze and the cleanup run as root.

The freeze's symlink and startup-hook purges and the restore's copy fallback
are Python snippets that root runs between the
agent's last action and the verifier. They run the interpreter V3's judge
accepted before root's first Python step: safe folders first, then the
image ``PATH``, each hop root's, closed to the agent and outside every
excluded tree (``lockdown._judge_root_python``). These lines
are the seam PT's merge replaces with its kit (``KIT_PYTHON -I -S -B``,
``root_python_sh``).

The snippet runs from ``/`` with ``-I -S -B``, so nothing on the working
folder, in ``site`` or in ``PYTHON*`` values reaches it. (The injected-hook
cleanup is U5a's helper in lockdown, run on the image's search path,
U5a-CONF-5.) Its own exit status is the command's, so a snippet that fails
fails its step (V8-R1-B-1, round 1 at b449a20b); with no judged interpreter
the caller's *missing* shell runs, which by default does nothing.
The lookup is POSIX sh (dash, bash, busybox).
"""

import shlex

from benchflow._root_python import FLAGS
from benchflow._runtime_tree import KIT_PYTHON, KIT_RECORD

# The verifier's trusted PATH (lockdown.VERIFIER_ENV["PATH"] is this value).
TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Directories the agent could write or that hold what it left; lockdown blocks
# the same ones (and the workspace) from the verifier's PATH.
RUNTIME_PATH_PREFIXES = ("/tmp", "/var/tmp", "/logs", "/testbed")


def hardening_python_cmd(
    script: str,
    args: str,
    workspace: str | None,
    *,
    missing: str = ":",
) -> str:
    """Render ``<python3> -I -S -B -c <script> <args>`` as POSIX sh (PT's
    flags, :data:`benchflow._root_python.FLAGS`).

    *args* is already shell-quoted. *missing* is the shell that runs when the
    image has no ``python3``. *workspace* is unused here; it stays in the
    signature for the PT seam, whose kit check names it.
    """
    del workspace
    flags = " ".join(FLAGS)
    run = f'(cd / && "$__bf_hp_py" {flags} -c {shlex.quote(script)} {args})'
    return "\n".join(
        [
            *_interpreter_lookup(),
            f'if [ -n "$__bf_hp_py" ]; then {run}; else {missing}; fi',
        ]
    )


def _interpreter_lookup() -> list[str]:
    """The shell lines that choose root's interpreter: they set
    ``$__bf_hp_py`` to the protected kit's Python when the kit's RECORD is
    present (none when that kit holds no Python), else to the interpreter
    the host judged before root's first Python step (``$__bf_root_python``,
    which the root route sets: lockdown._prep_on_path), or leave it empty
    when none passed. No ``PATH`` lookup: a ``python3`` found by name had not
    been judged (the kit first, ahead of the judged one).

    The one place V8's snippets get their Python. On the protected runtime
    (PT, PT.md "contract for V2, V8 and FF-install") these lines become the
    kit's: ``KIT_PYTHON`` with ``-I -S -B`` when the kit's RECORD is present
    (``root_python_sh``), and a kit without a Python is
    ``runtime_kit_python(env) is None`` (*missing* runs).
    """
    # PT's kit fills the seam: the kit's
    # Python when the kit's RECORD is present, as root_python_sh decides; a
    # kit without its Python is none (*missing* runs); no kit, V3's judged
    # interpreter (the preparation route sets it).
    return [
        f"if [ -f {KIT_RECORD} ]; then "
        f"if [ -x {KIT_PYTHON} ]; then __bf_hp_py={KIT_PYTHON}; else __bf_hp_py=; fi; "
        # Every preparation step runs on the route that sets it
        # (lockdown._prep_on_path); empty: none was judged, nothing runs.
        "else __bf_hp_py=${__bf_root_python-}; fi"
    ]
