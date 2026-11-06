#!/usr/bin/env bash
# Install BenchFlow the way a user does and run the unit suite against it.
#
# CI tests the locked environment (`uv sync --locked`), but `uv tool install
# benchflow` and `pip install benchflow` never read uv.lock: they resolve the
# newest versions pyproject.toml allows on the day. This script catches what
# that resolve breaks (typer 0.27 broke 9 tests while the lock held 0.24) and
# any test that only passes next to a source checkout:
#
#   1. builds the wheel from this checkout (`uv build --wheel`);
#   2. installs it, with the dev, sandbox-daytona and sandbox-modal extras, in
#      a fresh venv by a fresh resolve (`uv pip install`, no lock, no
#      constraints);
#   3. copies the checkout without src/ (tests, docs, examples and the rest
#      stay, so tests that read them still can) and checks that `benchflow`
#      then imports from the venv, not from a source tree;
#   4. runs pytest there and compares the failures with
#      tools/ci/fresh_resolve_known_failures.txt: a failure outside that list
#      fails the script; a listed test that passes is reported, so the list
#      only shrinks.
#
# Usage, from anywhere, on a machine with uv on PATH (CI or a build VM: the
# suite starts Docker containers when a daemon is there):
#
#   tools/ci/fresh_resolve.sh [pytest arguments]
#
# Unless the arguments carry their own `-m`, pytest runs the default marker
# set without the deterministic tier (that tier tests the sandbox, not the
# resolve; pass `-m 'not live and not integration and not e2e'` to include
# it). Settings:
#
#   FRESH_RESOLVE_PYTHON  the Python uv makes the venv with (default 3.12)
#   FRESH_RESOLVE_DIR     work folder (default: a new temporary one, kept)
#   UV_EXCLUDE_NEWER etc. any uv setting passes through, e.g. to replay a
#                         resolve as of a date
#
# The resolved versions are in $FRESH_RESOLVE_DIR/freeze.txt; the script
# prints the CLI packages' lines (typer, click) so a log shows what ran.
set -euo pipefail
# sort and comm below compare test ids byte by byte.
export LC_ALL=C

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
KNOWN="$ROOT/tools/ci/fresh_resolve_known_failures.txt"
PYTHON_REQUEST=${FRESH_RESOLVE_PYTHON:-3.12}
WORK=${FRESH_RESOLVE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/benchflow-fresh-resolve.XXXXXX")}
mkdir -p "$WORK"
# The physical path: the import check below compares the venv with
# benchflow's resolved __file__, and a work folder under a symlink (macOS's
# default TMPDIR, /var -> /private/var) would never match its logical path.
WORK=$(cd "$WORK" && pwd -P)
# Physical paths can still spell the same directory differently on a
# case-insensitive filesystem. Compare each ancestor by identity instead
# of folding case (which would reject distinct case-sensitive siblings).
ANCESTOR="$WORK"
while :; do
    if [ "$ANCESTOR" -ef "$ROOT" ]; then
        echo "fresh_resolve: FRESH_RESOLVE_DIR must be outside the checkout (would copy itself)" >&2
        exit 2
    fi
    [ "$ANCESTOR" = / ] && break
    ANCESTOR=$(dirname "$ANCESTOR")
done
DIST="$WORK/dist"
VENV="$WORK/venv"
TREE="$WORK/tree"

command -v uv >/dev/null || {
    echo "fresh_resolve: uv is not on PATH" >&2
    exit 2
}
if [ -e "$VENV" ] || [ -e "$TREE" ] || [ -e "$DIST" ]; then
    echo "fresh_resolve: $WORK already holds a venv, tree or dist; name a new FRESH_RESOLVE_DIR" >&2
    exit 2
fi
echo "fresh_resolve: work folder $WORK"

# 1. The wheel, built in an isolated build environment.
uv --no-config build --wheel --quiet --out-dir "$DIST" "$ROOT"
WHEELS=("$DIST"/benchflow-*.whl)
if [ "${#WHEELS[@]}" -ne 1 ] || [ ! -f "${WHEELS[0]}" ]; then
    echo "fresh_resolve: expected one wheel in $DIST" >&2
    exit 2
fi
WHEEL=${WHEELS[0]}
echo "fresh_resolve: built $(basename "$WHEEL")"

# 2. A fresh venv and a fresh resolve. Run from the work folder so no
#    project's pyproject.toml, uv.toml or uv.lock is near. Disable persistent
#    user/system configuration too; explicit replay environment settings stay.
uv --no-config venv --quiet --python "$PYTHON_REQUEST" "$VENV"
PY="$VENV/bin/python"
(cd "$WORK" && uv --no-config pip install --quiet --python "$PY" "$WHEEL[dev,sandbox-daytona,sandbox-modal]")
(cd "$WORK" && uv --no-config pip freeze --python "$PY") > "$WORK/freeze.txt"
echo "fresh_resolve: $("$PY" -I -c 'import platform; print("Python", platform.python_version())'), $(grep -c '==' "$WORK/freeze.txt") packages resolved"
grep -i -E '^(typer|click)==' "$WORK/freeze.txt" | sed 's/^/fresh_resolve: resolved /'

# 3. The checkout without src/. tar keeps symlinks and modes as they are.
mkdir -p "$TREE"
tar -C "$ROOT" \
    --exclude=./src --exclude=./.git --exclude=./.venv --exclude=./dist \
    --exclude=./build --exclude='__pycache__' --exclude=.pytest_cache \
    --exclude=.ruff_cache -cf - . | tar -C "$TREE" -xf -
if [ -e "$TREE/src" ]; then
    echo "fresh_resolve: $TREE/src exists after the copy" >&2
    exit 2
fi
LOCATION=$(cd "$TREE" && "$PY" -I -c 'import benchflow, pathlib; print(pathlib.Path(benchflow.__file__).resolve())')
case "$LOCATION" in
    "$VENV"/*) echo "fresh_resolve: benchflow imports from $LOCATION" ;;
    *)
        echo "fresh_resolve: benchflow imports from $LOCATION, not from the venv" >&2
        exit 2
        ;;
esac

# 4. The suite, against the installed wheel.
MARKERS_GIVEN=0
for arg in "$@"; do
    case "$arg" in -m | -m*) MARKERS_GIVEN=1 ;; esac
done
if [ "$MARKERS_GIVEN" -eq 0 ]; then
    set -- -m 'not live and not integration and not e2e and not deterministic' "$@"
fi
JUNIT="$WORK/junit.xml"
set +e
(cd "$TREE" && env -u PYTHONPATH "$PY" -m pytest -p no:cacheprovider -o junit_family=xunit1 --junitxml="$JUNIT" "$@")
STATUS=$?
set -e
# pytest: 0 all passed, 1 some failed; anything else (interrupted, usage
# error, no tests collected) is not a result to compare.
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 1 ]; then
    echo "fresh_resolve: pytest exited $STATUS" >&2
    exit "$STATUS"
fi

# Failed and errored test ids from the JUnit report, as `path::name` (the
# form `pytest -rf` prints and the known-failures file uses).
"$PY" -I - "$JUNIT" > "$WORK/failures.txt" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET

ids = set()
for case in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    if case.find("failure") is None and case.find("error") is None:
        continue
    # xunit1 records the file; classname is its dotted module path, then
    # any test classes.
    file = case.get("file", "")
    module = file[: -len(".py")].replace("/", ".")
    classname = case.get("classname", "")
    classes = classname[len(module) + 1 :].split(".") if classname.startswith(module + ".") else []
    ids.add("::".join([file, *classes, case.get("name", "")]))
for test_id in sorted(ids):
    print(test_id)
PYEOF

# The id is the line up to its first "#", without surrounding spaces (the
# rule tests/test_fresh_resolve_script.py checks the file by).
{ grep -v -E '^[[:space:]]*(#|$)' "$KNOWN" || true; } | sed -E 's/#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//' | sort -u > "$WORK/known.txt"
sort -u "$WORK/failures.txt" -o "$WORK/failures.txt"
comm -23 "$WORK/failures.txt" "$WORK/known.txt" > "$WORK/unexpected.txt"
comm -13 "$WORK/failures.txt" "$WORK/known.txt" > "$WORK/now-passing.txt"

echo "fresh_resolve: $(wc -l < "$WORK/failures.txt" | tr -d ' ') failed, $(wc -l < "$WORK/unexpected.txt" | tr -d ' ') outside $(basename "$KNOWN")"
if [ -s "$WORK/now-passing.txt" ]; then
    echo "fresh_resolve: listed as known but did not fail (passed, or not run with these arguments):"
    sed 's/^/  /' "$WORK/now-passing.txt"
fi
if [ -s "$WORK/unexpected.txt" ]; then
    echo "fresh_resolve: FAILED outside the known list:" >&2
    sed 's/^/  /' "$WORK/unexpected.txt" >&2
    exit 1
fi
echo "fresh_resolve: OK"
