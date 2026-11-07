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
from benchflow.sandbox import _hook_cleanup, lockdown
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
    sweep = lockdown._conftest_sweep_cmd()
    assert sweep in lockdown._build_cleanup_cmd({"cleanup_conftests": True})
    assert "-name conftest.py" in sweep
    ran = _dx(box, sweep)
    assert ran.returncode == 0, ran.stderr
    assert _dx(box, f"cat {kept}").stdout == "honest\n"
    assert _dx(box, f"test -e {planted}").returncode == 1


def test_the_sweep_keeps_an_installed_packages_own_file(box):  # noqa: F811
    """Beartype's ``_conf/conftest.py`` in an agent's install tree,
    listed by its RECORD with its digest, stays; a conftest.py no RECORD lists
    and one in the workspace go. On the preparation route the judged
    interpreter is ``$__bf_root_python``."""
    site = "/opt/runtime/uv-tools/openhands/lib/python3.12/site-packages"
    own = f"{site}/beartype/_conf/conftest.py"
    record = f"{site}/beartype-0.21.0.dist-info/RECORD"
    digest = _hook_cleanup.record_digest(b"honest\n")
    unlisted = "/usr/local/lib/python3.12/site-packages/conftest.py"
    planted = "/srv/conftest.py"
    made = _dx(
        box,
        f"mkdir -p {site}/beartype/_conf {site}/beartype-0.21.0.dist-info && "
        f"printf 'honest\\n' > {own} && "
        f"printf 'beartype/_conf/conftest.py,{digest},7\\n' > {record} && "
        f"echo planted > {unlisted} && echo planted > {planted}",
    )
    assert made.returncode == 0, made.stderr
    ran = _dx(
        box,
        "__bf_root_python=/usr/local/bin/python3; "
        + lockdown._conftest_sweep_cmd(workspace="/srv"),
    )
    assert ran.returncode == 0, ran.stderr
    assert _dx(box, f"cat {own}").stdout == "honest\n"
    assert _dx(box, f"test -e {unlisted}").returncode == 1
    assert _dx(box, f"test -e {planted}").returncode == 1


def test_the_sweep_keeps_the_repositorys_unchanged_conftest(box):  # noqa: F811
    """#1171, #1177: the workspace repository's own conftest.py stays, as a
    fresh root-owned file, when it is byte for byte the root-owned pre-agent
    copy's; an edited one, a planted one, and one reached through a link in
    the copy go. Without a non-root agent nothing in the workspace stays."""
    setup = (
        "mkdir -p /testbed/tests /testbed/pkg /testbed/new /testbed/real "
        "/testbed_verify/tests /testbed_verify/pkg && "
        "printf 'repo\\n' > /testbed/tests/conftest.py && "
        "printf 'repo\\n' > /testbed_verify/tests/conftest.py && "
        "printf 'edited\\n' > /testbed/pkg/conftest.py && "
        "printf 'repo\\n' > /testbed_verify/pkg/conftest.py && "
        "printf 'planted\\n' > /testbed/new/conftest.py && "
        "printf 'planted\\n' > /testbed/real/conftest.py && "
        "ln -s /testbed/real /testbed_verify/real && "
        "chown -R 1000:1000 /testbed && chmod 700 /testbed_verify"
    )
    made = _dx(box, setup)
    assert made.returncode == 0, made.stderr
    before = _dx(box, "stat -c %i /testbed/tests/conftest.py").stdout
    sweep = lockdown._conftest_sweep_cmd(workspace="/testbed", agent_user=True)
    ran = _dx(box, "__bf_root_python=/usr/local/bin/python3; " + sweep)
    assert ran.returncode == 0, ran.stderr
    kept = _dx(
        box,
        "cat /testbed/tests/conftest.py; stat -c '%u %a %i' /testbed/tests/conftest.py",
    )
    assert kept.stdout.splitlines()[0] == "repo"
    owner, mode, inode = kept.stdout.splitlines()[1].split()
    assert (owner, mode) == ("0", "644") and inode != before.strip()
    for gone in ("/testbed/pkg", "/testbed/new", "/testbed/real"):
        assert _dx(box, f"test -e {gone}/conftest.py").returncode == 1, gone

    again = _dx(box, "printf 'repo\\n' > /testbed/tests/conftest.py")
    assert again.returncode == 0, again.stderr
    as_root = lockdown._conftest_sweep_cmd(workspace="/testbed", agent_user=False)
    ran = _dx(box, "__bf_root_python=/usr/local/bin/python3; " + as_root)
    assert ran.returncode == 0, ran.stderr
    assert _dx(box, "test -e /testbed/tests/conftest.py").returncode == 1
