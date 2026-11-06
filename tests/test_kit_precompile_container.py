"""The kit's Python carries compiled modules for root's helpers (PERF-G), in a
real container: the kit's build writes checked-hash .pyc files for
KIT_PRECOMPILED_MODULES before the seal, the seal's MANIFEST lists them,
the kit's Python loads them, and a source that no longer matches its .pyc
is what runs. Skipped without a Docker daemon.
"""

from __future__ import annotations

import shutil

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready, _dx, _fresh

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

KIT_PY = "/opt/benchflow/rt/py/python3"
FLAGS = (
    "import importlib.util, json, sys; "
    "c = importlib.util.cache_from_source(json.__file__); "
    "print(json.__file__.startswith(sys.prefix + '/lib/'), "
    "int.from_bytes(open(c, 'rb').read(8)[4:8], 'little'))"
)


def _built(name: str) -> None:
    from benchflow._runtime_tree import kit_prepare_command

    built = _dx(name, kit_prepare_command("build"))
    assert built.returncode == 0, built.stderr
    sealed = _dx(name, kit_prepare_command("seal"))
    assert sealed.returncode == 0, sealed.stderr


def test_the_kit_build_compiles_the_helpers_modules_before_the_seal():
    if not _docker_ready():
        pytest.skip("docker daemon unavailable")
    with _fresh("useradd -m -u 1000 agent") as name:
        _built(name)
        # json's .pyc in the kit is checked-hash (flags 3: hash-based, checked).
        flags = _dx(name, f'cd / && {KIT_PY} -c "{FLAGS}"')
        assert flags.returncode == 0, flags.stderr
        assert flags.stdout.split() == ["True", "3"], flags.stdout
        for module in ("json/__pycache__/__init__", "importlib/metadata/__pycache__"):
            listed = _dx(name, f"grep -c {module} /opt/benchflow/rt/MANIFEST")
            assert listed.returncode == 0 and int(listed.stdout) >= 1, module
        # The kit's Python loads the compiled file (-v names its source).
        verbose = _dx(name, f"cd / && {KIT_PY} -v -c 'import json' 2>&1")
        assert "json/__pycache__/__init__" in verbose.stdout
        # Every file the build made is root's alone, as before.
        owners = _dx(name, "find /opt/benchflow/rt ! -user 0 | head -n 1")
        assert owners.stdout == ""


def test_a_source_that_no_longer_matches_its_pyc_is_what_runs():
    """Checked-hash: Python compares the source's hash before it uses the
    .pyc, so the code that runs is always the source's (a throwaway
    container; the change is made by the test as root, never by BenchFlow)."""
    if not _docker_ready():
        pytest.skip("docker daemon unavailable")
    with _fresh() as name:
        _built(name)
        _dx(
            name,
            "f=$(ls -d /opt/benchflow/rt/lib/python3*/json/__init__.py) && "
            "printf '\\nBF_CHANGED = 1\\n' >> \"$f\"",
        )
        ran = _dx(name, f"cd / && {KIT_PY} -c 'import json; print(json.BF_CHANGED)'")
        assert ran.returncode == 0 and ran.stdout.strip() == "1", ran.stderr
