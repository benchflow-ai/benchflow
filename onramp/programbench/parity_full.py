"""Full-set parity sweep: BenchFlow onramp vs. upstream `programbench eval`.

For each upstream instance:
  1. Pull both image tags (`programbench/<id>:task` for upstream, `:task_cleanroom`
     for BenchFlow's verifier image).
  2. Pre-fetch the instance's per-branch test blobs from HuggingFace.
  3. Build the BenchFlow task image.
  4. Stage a deterministic stub submission (`templates/stub_compile.sh`) under
     /app/ and run the BenchFlow verifier; record passed / total / score.
  5. Run upstream `programbench eval` against the same submission archive;
     record passed / total / score.
  6. Compare; record `agree = (bf.passed == up.passed and bf.total == up.total)`.
  7. Delete pulled images to free disk before the next instance.

Resumable — skip instances already in the output JSON. Designed for the
disk-tight (~15 GB) sandbox: footprint stays under ~2 GB at any time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).parent / "templates"
STUB_COMPILE_SH = TEMPLATE_DIR / "stub_compile.sh"


def _docker(*args: str, **kw):
    """sudo docker — the sandbox runs docker via sudo."""
    return subprocess.run(["sudo", "docker", *args], **kw)


def cleanroom_image(instance_id: str) -> str:
    return f"programbench/{instance_id.replace('__', '_1776_')}"


def pull_blobs(instance_id: str, blob_cache: Path) -> bool:
    """Fetch this instance's HF tarballs into ``blob_cache/<instance>/tests/``."""
    if (blob_cache / instance_id / "tests").exists() and any((blob_cache / instance_id / "tests").iterdir()):
        return True
    blob_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--with", "huggingface_hub",
        "python", "-c",
        "import shutil, sys, os; from pathlib import Path; from huggingface_hub import snapshot_download; "
        "inst = sys.argv[1]; cache = Path(sys.argv[2]); "
        "base = snapshot_download('programbench/ProgramBench-Tests', repo_type='dataset', allow_patterns=f'{inst}/**'); "
        "src = Path(base) / inst; dst = cache / inst; dst.mkdir(parents=True, exist_ok=True); "
        "[shutil.copy2(p, (dst / p.relative_to(src))) if p.is_file() and not (dst / p.relative_to(src)).exists() else "
        " (dst / p.relative_to(src)).mkdir(parents=True, exist_ok=True) "
        " for p in src.rglob('*')]",
        instance_id, str(blob_cache),
    ]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        logger.error("blob fetch failed for %s: %s", instance_id, r.stderr[-300:])
        return False
    return True


def benchflow_score(instance_id: str, task_dir: Path, blob_cache: Path,
                    stub_dir: Path) -> dict:
    """Build the BF task image, stage the stub, run /tests/test.sh, return score dict."""
    image_tag = f"bfp-{instance_id}".lower().replace("_", "-").replace(".", "-")
    if len(image_tag) > 60:
        image_tag = image_tag[:60]
    container = image_tag + "-run"
    df_dir = task_dir / "environment"
    _docker("rm", "-f", container, capture_output=True)
    try:
        b = _docker("build", "-q", "-t", image_tag, "-f", str(df_dir / "Dockerfile"), str(df_dir),
                    capture_output=True, text=True, timeout=900)
        if b.returncode != 0:
            return {"error": f"docker build failed: {b.stderr[-300:]}"}
        # Mount the local blob cache so test.sh skips its HF fetch entirely.
        r = _docker("run", "-d", "--name", container,
                    "-v", f"{blob_cache}:/tmp/benchflow_blobs:ro",
                    image_tag, "sleep", "3600",
                    capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"error": f"docker run failed: {r.stderr[-300:]}"}
        _docker("cp", f"{stub_dir}/.", f"{container}:/app/", check=True)
        _docker("exec", container, "mkdir", "-p", "/tests", check=True)
        _docker("cp", str(task_dir / "tests") + "/.", f"{container}:/tests/", check=True)
        _docker("exec", container, "chmod", "+x", "/tests/test.sh", check=True)
        _docker("exec", container, "bash", "/tests/test.sh", capture_output=True, timeout=7200)
        out = _docker("exec", container, "cat", "/logs/verifier/score.json",
                      capture_output=True, text=True)
        if out.returncode != 0:
            reward = _docker("exec", container, "cat", "/logs/verifier/reward.txt",
                             capture_output=True, text=True)
            return {"error": "no score.json", "reward_txt": reward.stdout.strip()}
        score = json.loads(out.stdout)
        passed = score.get("passed", 0)
        total = score.get("total", 0)
        return {
            "passed": passed,
            "total": total,
            "score": passed / total if total else 0.0,
            "branches": [{"branch": b["branch"], "status": b["status"], "passed": b.get("passed", 0), "total": b.get("total", 0)} for b in score.get("branches", [])],
        }
    finally:
        _docker("rm", "-f", container, capture_output=True)
        _docker("rmi", "-f", image_tag, capture_output=True)


def upstream_score(instance_id: str, blob_cache: Path, stub_dir: Path,
                   upstream_repo: Path) -> dict:
    """Run `programbench eval` on the same stub. PROGRAMBENCH_BLOB_DIR uses local cache."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        inst_dir = run_dir / instance_id
        inst_dir.mkdir(parents=True)
        sub_tar = inst_dir / "submission.tar.gz"
        subprocess.run(["tar", "-czf", str(sub_tar), "-C", str(stub_dir), "."], check=True)

        # Make a sudo-wrapper for docker since upstream eval invokes it directly.
        # `2>/dev/null` on the sudo invocation is load-bearing on hosts where
        # sudo emits "unable to resolve host" warnings on stderr — upstream
        # concatenates stderr with the captured XML, producing "junk after
        # document element" parse errors that abort whole instances.
        sudo_docker = Path(tmp) / "sudo_docker.sh"
        sudo_docker.write_text(
            '#!/bin/bash\nexec sudo /usr/bin/docker "$@" 2> >(grep -v "unable to resolve host" >&2)\n'
        )
        sudo_docker.chmod(0o755)

        env = {
            **os.environ,
            "PROGRAMBENCH_BLOB_DIR": str(blob_cache),
            "PROGRAMBENCH_DOCKER_EXECUTABLE": str(sudo_docker),
            "PROGRAMBENCH_DOCKER_CPUS": "2",
        }
        try:
            subprocess.run(
                ["uv", "run", "programbench", "eval", str(run_dir),
                 "--workers", "1", "--branch-workers", "1", "--branch-retries", "0"],
                cwd=str(upstream_repo), env=env,
                capture_output=True, text=True, timeout=7200,
            )
        except subprocess.TimeoutExpired:
            return {"error": "upstream eval timeout"}

        eval_json = inst_dir / f"{instance_id}.eval.json"
        if not eval_json.exists():
            return {"error": "upstream produced no eval.json"}
        data = json.loads(eval_json.read_text())

        # Apply `programbench info`'s scoring rule directly on the JSON, with
        # no in-process import of programbench (which leaks state between
        # instances and trips on `importlib.metadata` lookups in the .ref
        # checkout). Drop ignored branches/tests and any test belonging to a
        # branch missing from tests.json's active set; then score.
        return _score_from_eval_json(data, instance_id, upstream_repo)


def _score_from_eval_json(data: dict, instance_id: str, upstream_repo: Path) -> dict:
    upstream_task_yaml = upstream_repo / "src" / "programbench" / "data" / "tasks" / instance_id / "task.yaml"
    upstream_tests_json = upstream_repo / "src" / "programbench" / "data" / "tasks" / instance_id / "tests.json"

    active_branches: set[str] = set()
    ignored_tests: set[str] = set()  # "branch/test_name"
    if upstream_tests_json.exists():
        tj = json.loads(upstream_tests_json.read_text())
        for branch, info in (tj.get("branches") or {}).items():
            if not info.get("ignored"):
                active_branches.add(branch)
            for t in (info.get("ignored_tests") or []):
                ignored_tests.add(f"{branch}/{t['name']}")

    # `for_branches(active)` + `without_ignored(ignored_tests)`:
    kept = []
    for tr in data.get("test_results", []):
        branch = tr.get("branch", "")
        name = tr.get("name", "")
        if active_branches and branch not in active_branches:
            continue
        if f"{branch}/{name}" in ignored_tests:
            continue
        kept.append(tr)
    passed = sum(1 for tr in kept if tr.get("status") == "passed")
    total = len(kept)
    return {
        "passed": passed,
        "total": total,
        "score": passed / total if total else 0.0,
        "error_code": data.get("error_code"),
    }


def cleanup_image(instance_id: str) -> None:
    img = cleanroom_image(instance_id)
    _docker("rmi", "-f", f"{img}:task_cleanroom", capture_output=True)
    _docker("rmi", "-f", f"{img}:task", capture_output=True)


def run_one(instance_id: str, *, blob_cache: Path, tasks_dir: Path,
            stub_dir: Path, upstream_repo: Path) -> dict:
    t0 = time.monotonic()
    task_dir = tasks_dir / instance_id
    if not task_dir.exists():
        return {"instance_id": instance_id, "error": "task dir missing — run benchflow.py first"}

    if not pull_blobs(instance_id, blob_cache):
        return {"instance_id": instance_id, "error": "blob fetch failed"}

    bf = benchflow_score(instance_id, task_dir, blob_cache, stub_dir)
    up = upstream_score(instance_id, blob_cache, stub_dir, upstream_repo)
    cleanup_image(instance_id)

    bf_passed = bf.get("passed")
    up_passed = up.get("passed")
    bf_total = bf.get("total")
    up_total = up.get("total")
    agree = (
        bf_passed is not None and up_passed is not None
        and bf_passed == up_passed and bf_total == up_total
    )
    return {
        "instance_id": instance_id,
        "wall_time_sec": round(time.monotonic() - t0, 1),
        "benchflow": bf,
        "upstream": up,
        "agree": agree,
        "delta": (
            abs((bf.get("score") or 0) - (up.get("score") or 0))
            if (bf.get("score") is not None and up.get("score") is not None)
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-repo", required=True, type=Path)
    p.add_argument("--tasks-dir", required=True, type=Path,
                   help="Where converted BenchFlow tasks live (`benchflow.py` output).")
    p.add_argument("--blob-cache", default=Path("/tmp/benchflow_blobs"), type=Path)
    p.add_argument("--output", default=Path("onramp/programbench/parity_full_results.json"), type=Path)
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="Restrict to these instance IDs (default: all under --tasks-dir).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budget-min", type=float, default=None,
                   help="Stop accepting new instances after this many minutes elapsed.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Materialize the stub submission.
    stub_dir = Path(tempfile.mkdtemp(prefix="bf-parity-stub-"))
    shutil.copy(STUB_COMPILE_SH, stub_dir / "compile.sh")
    (stub_dir / "compile.sh").chmod(0o755)

    if args.task_ids:
        ids = list(args.task_ids)
    else:
        ids = sorted(d.name for d in args.tasks_dir.iterdir() if d.is_dir())
        # Skip the upstream fixture — its image isn't on Docker Hub.
        ids = [i for i in ids if not i.startswith("testorg__")]
    if args.limit is not None:
        ids = ids[: args.limit]

    # Resume: skip already-completed entries.
    existing: list[dict] = []
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text()).get("results", [])
        except Exception:
            existing = []
    done = {r["instance_id"] for r in existing}

    pending = [i for i in ids if i not in done]
    logger.info("Total: %d, already done: %d, pending: %d", len(ids), len(done), len(pending))

    results = list(existing)
    t_start = time.monotonic()
    for i, iid in enumerate(pending, 1):
        if args.budget_min and (time.monotonic() - t_start) / 60 > args.budget_min:
            logger.info("Budget %s min exhausted; stopping after %d done", args.budget_min, i - 1)
            break
        logger.info("[%d/%d] %s", i, len(pending), iid)
        try:
            r = run_one(iid, blob_cache=args.blob_cache, tasks_dir=args.tasks_dir,
                        stub_dir=stub_dir, upstream_repo=args.upstream_repo)
        except Exception as e:
            r = {"instance_id": iid, "error": f"unhandled: {e!r}"}
        results.append(r)
        # Persist after each instance so a crash doesn't lose progress.
        n_agree = sum(1 for x in results if x.get("agree"))
        n_with_score = sum(1 for x in results if x.get("agree") is not None)
        summary = {
            "n_total": len(ids),
            "n_evaluated": len(results),
            "n_agreeing": n_agree,
            "n_with_score": n_with_score,
            "agreement_rate": n_agree / n_with_score if n_with_score else None,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        agree_marker = "✓" if r.get("agree") else "✗" if r.get("agree") is False else "?"
        logger.info("  %s bf=%s/%s up=%s/%s [%ss]",
                    agree_marker,
                    r.get("benchflow", {}).get("passed"), r.get("benchflow", {}).get("total"),
                    r.get("upstream", {}).get("passed"), r.get("upstream", {}).get("total"),
                    r.get("wall_time_sec", "?"))

    print(f"Done. {sum(1 for x in results if x.get('agree'))}/{len(results)} agree. "
          f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
