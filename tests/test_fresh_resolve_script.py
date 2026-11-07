"""tools/ci/fresh_resolve.sh and its list of known failures.

The script itself runs the whole suite against a freshly resolved wheel (run
by hand before a release; no CI job runs it yet); these checks keep it honest
without that cost: the script parses, every listed failure names a test that
exists and says why it is allowed to fail, and the script's own steps run
against a stand-in uv.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ci" / "fresh_resolve.sh"
KNOWN = ROOT / "tools" / "ci" / "fresh_resolve_known_failures.txt"


def _entries() -> list[tuple[str, str]]:
    entries = []
    for line in KNOWN.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        test_id, sep, reason = line.partition("#")
        entries.append((test_id.strip(), reason.strip() if sep else ""))
    return entries


def test_script_is_executable_bash_that_parses():
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    bash = shutil.which("bash")
    assert bash, "bash is needed to check the script"
    done = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("test_id", "reason"), _entries(), ids=[i.split("::")[-1] for i, _ in _entries()]
)
def test_each_known_failure_names_a_real_test_and_a_reason(test_id, reason):
    assert reason, f"{test_id}: give the reason after a '#'"
    path, _, rest = test_id.partition("::")
    function = re.sub(r"\[.*\]$", "", rest.split("::")[-1])
    source = ROOT / path
    assert source.is_file(), f"{test_id}: no file {path}"
    assert re.search(
        rf"^\s*(async )?def {re.escape(function)}\(",
        source.read_text(encoding="utf-8"),
        re.MULTILINE,
    ), f"{test_id}: no test {function} in {path}"


def test_known_failures_are_listed_once():
    ids = [test_id for test_id, _ in _entries()]
    assert len(ids) == len(set(ids))


# A stand-in for uv, so the script's own steps run in seconds: `build` makes
# an empty wheel, `venv` makes a venv whose python is the suite's interpreter
# with the venv's site folder on its path and an importable `benchflow`
# there, `pip install` does nothing and `pip freeze` prints two pins. The
# venv's python answers `-m pytest` by writing $FAKE_JUNIT as the report.
FAKE_UV = r"""#!/bin/sh
set -eu
no_config=0
if [ "$1" = --no-config ]; then no_config=1; shift; fi
cmd=$1
shift
case "$cmd" in
build)
    while [ $# -gt 0 ]; do
        case "$1" in --out-dir) out=$2; shift ;; esac
        shift
    done
    mkdir -p "$out"
    : > "$out/benchflow-0.0.0-py3-none-any.whl"
    ;;
venv)
    for venv in "$@"; do :; done
    "$FAKE_REAL_PYTHON" -m venv --without-pip "$venv/runtime"
    site=$("$venv/runtime/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    mkdir -p "$venv/bin" "$site/benchflow"
    : > "$site/benchflow/__init__.py"
    cat > "$venv/bin/python" <<PY
#!/bin/sh
if [ "\$1" = -m ] && [ "\$2" = pytest ]; then
    for a in "\$@"; do
        case "\$a" in --junitxml=*) cp "\$FAKE_JUNIT" "\${a#--junitxml=}" ;; esac
    done
    exit 0
fi
exec "$venv/runtime/bin/python" "\$@"
PY
    chmod +x "$venv/bin/python"
    ;;
pip)
    if [ "$1" = install ] && [ "${FAKE_REQUIRE_NO_CONFIG:-0}" = 1 ] && [ "$no_config" = 0 ]; then
        echo 'host uv configuration affected the fresh resolve' >&2
        exit 44
    fi
    if [ "$1" = freeze ]; then printf 'click==8.4.2\ntyper==0.24.1\n'; fi
    ;;
esac
"""

NO_FAILURES = "<testsuites><testsuite></testsuite></testsuites>\n"


def _run_with_fake_uv(tmp_path, *, work, known, junit=NO_FAILURES, extra_env=None):
    """Run a copy of the script in a made-up checkout with the stand-in uv."""
    root = tmp_path / "checkout"
    (root / "tools" / "ci").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "tools" / "ci" / SCRIPT.name)
    (root / "tools" / "ci" / KNOWN.name).write_text(known, encoding="utf-8")
    (root / "tests").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV, encoding="utf-8")
    (bin_dir / "uv").chmod(0o755)
    report = tmp_path / "report.xml"
    report.write_text(junit, encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FRESH_RESOLVE_DIR": str(work),
        "FAKE_REAL_PYTHON": sys.executable,
        "FAKE_JUNIT": str(report),
    }
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    bash = shutil.which("bash")
    assert bash, "bash is needed to run the script"
    return subprocess.run(
        [bash, str(root / "tools" / "ci" / SCRIPT.name)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_a_work_folder_under_a_symlink_still_finds_benchflow_in_the_venv(tmp_path):
    """macOS's default TMPDIR is under /var, a symlink to /private/var: the
    import check compares the venv with benchflow's resolved __file__, so the
    venv's path has to be the physical one too (U7-R1-A-1)."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    done = _run_with_fake_uv(tmp_path, work=link / "work", known="# none\n")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "not from the venv" not in done.stderr
    assert "fresh_resolve: OK" in done.stdout


def test_the_script_reads_known_failure_ids_as_the_file_check_does(tmp_path):
    """A line with no space before its "#" or with leading spaces passes
    test_each_known_failure_names_a_real_test_and_a_reason, so the script
    must take the same id from it (U7-R1-A-3)."""
    junit = (
        "<testsuites><testsuite>"
        '<testcase classname="tests.test_x" name="test_y" file="tests/test_x.py">'
        "<failure/></testcase>"
        '<testcase classname="tests.test_x" name="test_z" file="tests/test_x.py">'
        "<failure/></testcase>"
        "</testsuite></testsuites>\n"
    )
    known = (
        "tests/test_x.py::test_y# no space before the reason\n"
        "  tests/test_x.py::test_z  # leading spaces\n"
    )
    done = _run_with_fake_uv(tmp_path, work=tmp_path / "work", known=known, junit=junit)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 failed, 0 outside" in done.stdout


def test_fresh_install_disables_host_uv_configuration(tmp_path):
    """Guards fix ee072494: host lowest/date settings cannot leak in."""
    done = _run_with_fake_uv(
        tmp_path,
        work=tmp_path / "work",
        known="# none\n",
        extra_env={"FAKE_REQUIRE_NO_CONFIG": "1"},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "fresh_resolve: OK" in done.stdout


def test_wheel_probe_ignores_source_pythonpath(tmp_path):
    """Guards fix ee072494: a developer PYTHONPATH is not the wheel."""
    source = tmp_path / "source"
    (source / "benchflow").mkdir(parents=True)
    (source / "benchflow/__init__.py").write_text("# source, not the installed wheel\n")
    done = _run_with_fake_uv(
        tmp_path,
        work=tmp_path / "work",
        known="# none\n",
        extra_env={"PYTHONPATH": str(source)},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "fresh_resolve: OK" in done.stdout


@pytest.mark.parametrize("inside", ["", ".fresh-resolve", "nested/work"])
def test_work_directory_cannot_contain_its_own_source_copy(tmp_path, inside):
    """Guards fix ee072494 before building or recursively archiving."""
    done = _run_with_fake_uv(
        tmp_path,
        work=tmp_path / "checkout" / inside,
        known="# none\n",
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert "outside the checkout" in done.stderr
    assert "built " not in done.stdout


@pytest.mark.parametrize("inside", ["", ".fresh-resolve", "nested/work"])
def test_case_alias_of_checkout_is_rejected_before_building(tmp_path, inside):
    """Guards fix 3286fba9 against ee072494's case-alias bypass (U7-CONFIRM-1)."""
    root = tmp_path / "checkout"
    root.mkdir()
    alias = tmp_path / "CHECKOUT"
    extra_env = {}
    if not alias.exists():
        # Linux's case-sensitive test filesystem needs an alias and a pwd
        # stand-in to model macOS: -P can preserve a different case spelling
        # of the same directory. Directory identity checks still use the
        # real filesystem. On a case-insensitive filesystem use the native
        # alias and pwd instead.
        alias.symlink_to(root, target_is_directory=True)
        startup = tmp_path / "case-preserving-pwd.sh"
        startup.write_text('pwd() { printf "%s\\n" "$PWD"; }\n')
        extra_env["BASH_ENV"] = str(startup)
    assert alias.samefile(root)
    done = _run_with_fake_uv(
        tmp_path,
        work=alias / inside,
        known="# none\n",
        extra_env=extra_env,
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert "outside the checkout" in done.stderr
    assert "built " not in done.stdout


def test_distinct_case_spelling_outside_checkout_is_allowed(tmp_path):
    """Guards fix 3286fba9 (U7-CONFIRM-1): distinct case spellings stay valid."""
    root = tmp_path / "checkout"
    root.mkdir()
    work = tmp_path / "CHECKOUT" / "work"
    if work.parent.exists():
        # A case-insensitive filesystem cannot supply this distinct sibling.
        work = tmp_path / "checkout-other" / "work"
    work.mkdir(parents=True)
    assert not work.parent.samefile(root)
    done = _run_with_fake_uv(tmp_path, work=work, known="# none\n")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "fresh_resolve: OK" in done.stdout
