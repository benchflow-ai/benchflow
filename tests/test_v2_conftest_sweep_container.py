"""V2's runtime copy of the image's site folders survives the conftest sweep,
in a real container (merge wave batch 32).

V2 copies the image's site folders into its protected runtime before the
agent's turn, a package's own ``tests/conftest.py`` included (``jeepney`` on
TB2's fix-code-vulnerability image). The verifier's conftest sweep then
removed it and V2's custody check refused an honest trial (TB2 oracle 12/14).
The sweep now prunes that tree, which is root's alone and checked file by
file; a conftest.py anywhere else still goes. Skipped without a Docker daemon
(it runs on a Docker VM).
"""

from __future__ import annotations

from benchflow._runtime_tree import VERIFIER_DIR
from benchflow.sandbox import lockdown
from tests.test_c8_protected_runtime_container import _dx, box  # noqa: F401


def test_the_sweep_spares_the_runtime_copy_and_nothing_else(box):  # noqa: F811
    kept = f"{VERIFIER_DIR}/image-sites/0/jeepney/io/tests/conftest.py"
    planted = "/srv/conftest.py"
    made = _dx(
        box,
        f"mkdir -p {kept.rsplit('/', 1)[0]} && echo honest > {kept} && "
        f"echo planted > {planted}",
    )
    assert made.returncode == 0, made.stderr
    (sweep, *_) = lockdown._build_cleanup_cmd({"cleanup_conftests": True}).split("; ")
    assert "-name conftest.py" in sweep
    ran = _dx(box, sweep)
    assert ran.returncode == 0, ran.stderr
    assert _dx(box, f"cat {kept}").stdout == "honest\n"
    assert _dx(box, f"test -e {planted}").returncode == 1
