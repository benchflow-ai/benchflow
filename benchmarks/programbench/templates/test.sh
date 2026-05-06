#!/bin/bash
# BenchFlow verifier for ProgramBench-adapted tasks.
#
# Per-task data:
#   BF_INSTANCE_ID         — set in the Dockerfile (e.g. abishekvashok__cmatrix.5c082c6)
#   /tests/tests.json      — sidecar in the BenchFlow tests/ dir, copied by the
#                            runner alongside this script. Contains the upstream
#                            branches/tests metadata.
#
# Pipeline (mirrors ProgramBench's eval inside one BenchFlow sandbox):
#   1. Snapshot /workspace's pristine state (binary + docs from cleanroom image).
#   2. Stage agent's submission from /app into /workspace.
#   3. Seed a git repo if absent, then run compile.sh -> /workspace/executable.
#   4. Stash the built executable and record its sha256.
#   5. For each test branch in BF_TESTS_JSON_B64:
#        a. Restore /workspace from snapshot, restore executable, verify hash.
#        b. Stream the branch's test tarball in.
#        c. Run eval/run.sh, parse results.xml -> passed / total.
#   6. reward = total_passed / total_expected, written to /logs/verifier/reward.txt.

set -u
set -o pipefail

VERIFIER_DIR=/logs/verifier
mkdir -p "$VERIFIER_DIR"
log() { echo "[verifier] $*" | tee -a "$VERIFIER_DIR/verifier.log" >&2; }
write_reward() { printf '%s\n' "$1" > "$VERIFIER_DIR/reward.txt"; }

# Default 0 so any early bail produces a graded failure rather than blank output.
write_reward 0

: "${BF_INSTANCE_ID:?BF_INSTANCE_ID env var must be set by the task Dockerfile}"

TESTS_JSON_PATH=${BF_TESTS_JSON_PATH:-/tests/tests.json}
if [ ! -f "$TESTS_JSON_PATH" ]; then
    log "FAIL: tests.json sidecar not found at $TESTS_JSON_PATH"
    exit 0
fi

WORKSPACE=/workspace
APP=/app
SNAPSHOT=/tmp/benchflow_workspace_snapshot
BLOB_CACHE=${BF_BLOB_CACHE:-/tmp/benchflow_blobs}
STASH=/tmp/benchflow_stashed_executable

# 1. Snapshot pristine /workspace before the agent's submission overwrites it.
if [ ! -d "$SNAPSHOT" ]; then
    mkdir -p "$SNAPSHOT"
    if [ -d "$WORKSPACE" ] && [ "$(ls -A "$WORKSPACE" 2>/dev/null)" ]; then
        cp -a "$WORKSPACE/." "$SNAPSHOT/" 2>/dev/null || true
    fi
fi

# 2. Stage submission. Wipe /workspace and copy /app/* in.
mkdir -p "$WORKSPACE"
rm -rf "$WORKSPACE"/* "$WORKSPACE"/.[!.]* 2>/dev/null || true
if [ -d "$APP" ] && [ "$(ls -A "$APP" 2>/dev/null)" ]; then
    cp -a "$APP/." "$WORKSPACE/"
else
    log "FAIL: /app is empty — agent produced no submission"
    exit 0
fi

cd "$WORKSPACE" || { log "FAIL: cannot cd to $WORKSPACE"; exit 0; }

if [ ! -f compile.sh ]; then
    log "FAIL: /app/compile.sh not found"
    exit 0
fi

# 3. Seed a deterministic git repo if compile depends on one.
if [ ! -d .git ]; then
    GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
    GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
    git -c init.defaultBranch=gold init -q 2>/dev/null || true
    git -c user.email=gold@local -c user.name=gold \
        -c commit.gpgsign=false add -A 2>/dev/null || true
    GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
    GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
    git -c user.email=gold@local -c user.name=gold \
        -c commit.gpgsign=false commit -q --allow-empty -m gold 2>/dev/null || true
fi

log "Running compile.sh..."
chmod +x ./compile.sh
if ! timeout 900 bash ./compile.sh > "$VERIFIER_DIR/compile.log" 2>&1; then
    log "FAIL: compile.sh exited non-zero (see compile.log)"
    exit 0
fi
if [ ! -f ./executable ]; then
    log "FAIL: compile.sh did not produce ./executable"
    exit 0
fi
chmod +x ./executable
EXECUTABLE_HASH=$(sha256sum ./executable | awk '{print $1}')
export EXECUTABLE_HASH
log "executable_hash=$EXECUTABLE_HASH"

cp ./executable "$STASH"

# 4. Pre-fetch test blobs from HuggingFace if not already cached / pre-baked.
mkdir -p "$BLOB_CACHE"
if [ ! -d "$BLOB_CACHE/$BF_INSTANCE_ID" ] || [ -z "$(ls -A "$BLOB_CACHE/$BF_INSTANCE_ID/tests" 2>/dev/null)" ]; then
    log "Fetching test blobs for $BF_INSTANCE_ID from HuggingFace..."
    python3 - <<'PYFETCH' >> "$VERIFIER_DIR/verifier.log" 2>&1 || log "blob fetch failed (non-fatal — branch tests will be marked no_blob)"
import os, shutil, subprocess
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call(["pip3", "install", "-q", "--disable-pip-version-check", "huggingface_hub"])
    from huggingface_hub import snapshot_download
inst = os.environ["BF_INSTANCE_ID"]
cache = Path(os.environ.get("BF_BLOB_CACHE", "/tmp/benchflow_blobs"))
base = snapshot_download(
    "programbench/ProgramBench-Tests",
    repo_type="dataset",
    allow_patterns=f"{inst}/**",
)
src = Path(base) / inst
dst = cache / inst
dst.mkdir(parents=True, exist_ok=True)
for entry in src.rglob("*"):
    rel = entry.relative_to(src)
    target = dst / rel
    if entry.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(entry, target)
PYFETCH
fi

# 5. For each branch, run tests in a freshly-restored /workspace.
export VERIFIER_DIR WORKSPACE SNAPSHOT STASH BLOB_CACHE TESTS_JSON_PATH
python3 - <<'PYSCORE' > "$VERIFIER_DIR/score.json"
import json, os, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

INSTANCE_ID = os.environ["BF_INSTANCE_ID"]
SNAPSHOT = Path(os.environ["SNAPSHOT"])
WORKSPACE = Path(os.environ["WORKSPACE"])
STASH = Path(os.environ["STASH"])
BLOB_CACHE = Path(os.environ["BLOB_CACHE"])
EXECUTABLE_HASH = os.environ["EXECUTABLE_HASH"]
TESTS_JSON_PATH = Path(os.environ["TESTS_JSON_PATH"])

tests_json = json.loads(TESTS_JSON_PATH.read_text())
branches_meta = tests_json.get("branches", {}) or {}
active_branches = [n for n, info in branches_meta.items() if not info.get("ignored")]
ignored_test_names_by_branch = {
    branch: {t["name"] for t in (info.get("ignored_tests") or [])}
    for branch, info in branches_meta.items()
}

def restore_workspace():
    for child in WORKSPACE.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try: child.unlink()
            except FileNotFoundError: pass
    if SNAPSHOT.exists():
        for child in SNAPSHOT.iterdir():
            dest = WORKSPACE / child.name
            if child.is_dir():
                shutil.copytree(child, dest, symlinks=True)
            else:
                shutil.copy2(child, dest, follow_symlinks=False)

def restore_executable():
    target = WORKSPACE / "executable"
    if target.exists(): target.unlink()
    shutil.copy2(STASH, target)
    target.chmod(0o755)
    h = subprocess.check_output(["sha256sum", str(target)]).decode().split()[0]
    if h != EXECUTABLE_HASH:
        raise RuntimeError(f"executable_hash mismatch: {h} != {EXECUTABLE_HASH}")

def parse_results_xml(path: Path):
    if not path.exists(): return []
    try: root = ET.parse(path).getroot()
    except ET.ParseError: return []
    out = []
    for case in root.iter("testcase"):
        cls = case.get("classname") or ""
        name = case.get("name") or ""
        full = f"{cls}.{name}" if cls else name
        status = "passed"
        for child in case:
            if child.tag == "failure": status = "failure"; break
            if child.tag == "error": status = "error"; break
            if child.tag == "skipped": status = "skipped"; break
        out.append((full, status))
    return out

results = []
total_passed = 0
total_records = 0  # mirrors upstream: passed + failed + not_run + error etc.

for branch in active_branches:
    branch_ignored = ignored_test_names_by_branch.get(branch, set())
    expected = [t for t in branches_meta[branch].get("tests", []) if t not in branch_ignored]

    tar = BLOB_CACHE / INSTANCE_ID / "tests" / f"{branch}.tar.gz"
    if not tar.exists():
        # No blob -> all expected tests treated as not_run (denominator only).
        results.append({"branch": branch, "status": "no_blob", "passed": 0, "total": len(expected)})
        total_records += len(expected)
        continue

    try:
        restore_workspace()
        restore_executable()
        subprocess.run(["tar", "-xzf", str(tar), "-C", str(WORKSPACE)], check=True, timeout=300)
        run_sh = WORKSPACE / "eval" / "run.sh"
        if not run_sh.exists():
            results.append({"branch": branch, "status": "no_run_sh", "passed": 0, "total": len(expected)})
            total_records += len(expected)
            continue
        try:
            text = run_sh.read_text()
            run_sh.write_text(text.replace("--timeout-method=thread", "--timeout-method=signal"))
        except Exception:
            pass
        run_sh.chmod(0o755)
        t0 = time.monotonic()
        proc = subprocess.run(
            ["bash", str(run_sh)],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        wall = time.monotonic() - t0
        cases = parse_results_xml(WORKSPACE / "eval" / "results.xml")
        # Upstream score: passed / (junit_cases + missing_from_junit) — both
        # filtered by ignored_test_names. We mirror that exactly.
        junit_names = {n for n, _ in cases}
        records = list(cases)  # (name, status) for everything in JUnit
        # Add not_run entries for expected tests missing from JUnit.
        for tname in expected:
            if tname not in junit_names:
                records.append((tname, "not_run"))
        # Drop any record whose bare last segment is in this branch's ignored list.
        kept = [(n, s) for n, s in records if n.rsplit(".", 1)[-1] not in branch_ignored and n not in branch_ignored]
        passed = sum(1 for _, s in kept if s == "passed")
        total_records += len(kept)
        total_passed += passed
        results.append({
            "branch": branch, "status": "ran",
            "wall_time_sec": round(wall, 2),
            "returncode": proc.returncode,
            "passed": passed, "total": len(kept),
            "junit_cases": len(cases),
            "expected_in_tests_json": len(expected),
        })
    except subprocess.TimeoutExpired:
        results.append({"branch": branch, "status": "timeout", "passed": 0, "total": len(expected)})
        total_records += len(expected)
    except Exception as e:
        results.append({"branch": branch, "status": "error", "error": str(e), "passed": 0, "total": len(expected)})
        total_records += len(expected)

json.dump({"passed": total_passed, "total": total_records, "branches": results}, sys.stdout)
PYSCORE

PASSED=$(python3 -c "import json; print(json.load(open('$VERIFIER_DIR/score.json'))['passed'])" 2>/dev/null || echo 0)
TOTAL=$(python3 -c "import json; print(json.load(open('$VERIFIER_DIR/score.json'))['total'])" 2>/dev/null || echo 0)
log "score: $PASSED / $TOTAL"
if [ "${TOTAL:-0}" -gt 0 ]; then
    REWARD=$(python3 -c "print(${PASSED}/${TOTAL})")
else
    REWARD=0
fi
write_reward "$REWARD"
log "reward=$REWARD"
