"""Run the corpus's cells through a set of arms and write one snapshot per arm.

Three steps, all on a build VM (never on a workstation; the cells start
Docker containers):

    python3 -I tools/dt/run.py fetch --repo <git dir> --out <tasks dir>
    python3 -I tools/dt/run.py build --repo <git dir> --arms-dir <dir> --arm base
    python3 -I tools/dt/run.py run --arms base,mut-a --arms-dir <dir> \\
        --tasks-dir <tasks dir> --work <empty dir> --rounds 3 -P 8

``fetch`` unpacks each corpus tree with ``git archive <commit> <path>`` and
checks the git tree hash of what it unpacked against the corpus pin (the
hash is computed here, from the files, not taken from git). ``build`` makes
an arm from ``arms.toml``. ``run`` runs every (task, agent, round) cell in
every arm, each arm's cells of a task one after another in an order that
rotates every round (interleaved), different tasks in parallel. Each cell
runs alone: its own copy of the task in an empty parent folder, an empty
jobs folder, its own HOME, and an ``env -i`` environment that passes only
HOME, PATH, LANG and DOCKER_HOST (as ../engine-bench-1015/scripts/
trial.sh:15-23 did), with ``BENCHFLOW_REWARD_LENIENT`` and
``BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS`` refused in the caller's own
environment. The snapshot line of a cell holds the pinned reader's record
(normalise.py), the raw view, and, for a Python arm, what that arm's own
``bench eval inspect --json`` read. A requested Python cell with no trial
folder whose log names it in a ``Skipping task`` line is ``refused_at_load``.

Python:  bench eval run --tasks-dir <parent> --agent X --sandbox docker
         --concurrency 1 --jobs-dir <empty>
Rust:    bench run <copy> -a X -s docker -j 1 --jobs-dir <empty> --fresh
         --no-task-skills
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Run as a script (python3 -I tools/dt/run.py): make `tools` importable
    # from the repository root, never from the current folder.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dt import normalise

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.toml"
ARMS = HERE / "arms.toml"
REFUSED_ENV = ("BENCHFLOW_REWARD_LENIENT", "BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS")
CELL_PATH = "/usr/local/bin:/usr/bin:/bin"
CELL_TIMEOUT_SEC = 1800


def _toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def parse_task_id(task_id: str) -> tuple[str, str, str]:
    """``p1/examples/tasks/hello@<tree>`` -> (source, path, tree)."""
    spec, sep, tree = task_id.rpartition("@")
    source, slash, path = spec.partition("/")
    if not sep or not slash or not tree or not path:
        raise ValueError(f"corpus id {task_id!r} is not <source>/<path>@<tree hash>")
    return source, path, tree


def corpus_tasks(
    corpus: dict[str, Any], tier: str | None = None
) -> list[dict[str, Any]]:
    tasks = []
    names = set()
    for entry in corpus.get("task", []):
        if tier and tier not in entry.get("tiers", ["smoke"]):
            continue
        source, path, tree = parse_task_id(entry["id"])
        if entry["name"] in names:
            raise ValueError(f"corpus name {entry['name']!r} is used twice")
        names.add(entry["name"])
        commit = corpus["source"][source]["commit"]
        tasks.append(
            {**entry, "source": source, "path": path, "tree": tree, "commit": commit}
        )
    return tasks


# --- tree hashes, computed from the files (git's tree object format) -------


def _git_object(kind: str, body: bytes) -> bytes:
    return hashlib.sha1(f"{kind} {len(body)}\0".encode() + body).digest()


def tree_hash(root: Path) -> str:
    """The git tree hash of a folder, from its files (no git involved)."""

    def walk(folder: Path) -> bytes:
        entries = []
        for child in folder.iterdir():
            if child.is_symlink():
                mode, sha = b"120000", _git_object("blob", os.readlink(child).encode())
            elif child.is_dir():
                mode, sha = b"40000", walk(child)
            else:
                executable = child.stat().st_mode & 0o111
                mode = b"100755" if executable else b"100644"
                sha = _git_object("blob", child.read_bytes())
            # git orders a folder as if its name ended in "/".
            key = child.name.encode() + (b"/" if mode == b"40000" else b"")
            entries.append((key, mode + b" " + child.name.encode() + b"\0" + sha))
        body = b"".join(entry for _, entry in sorted(entries))
        return _git_object("tree", body)

    return walk(root).hex()


def fetch(
    repo: Path, out: Path, *, corpus_path: Path = CORPUS, tier: str | None = "smoke"
) -> list[str]:
    """Unpack every corpus tree under ``out/<name>`` and check its hash."""
    out.mkdir(parents=True, exist_ok=True)
    fetched = []
    for task in corpus_tasks(_toml(corpus_path), tier):
        dest = out / task["name"]
        if dest.exists():
            raise SystemExit(f"fetch: {dest} exists; fetch into an empty folder")
        staging = out / f".{task['name']}.staging"
        staging.mkdir()
        archive = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                task["commit"],
                task["path"],
            ],
            check=True,
            capture_output=True,
            env={**_git_env()},
        ).stdout
        tar_path = staging / "tree.tar"
        tar_path.write_bytes(archive)
        with tarfile.open(tar_path) as tar:
            tar.extractall(staging / "x", filter="tar")
        tar_path.unlink()
        unpacked = staging / "x" / task["path"]
        got = tree_hash(unpacked)
        if got != task["tree"]:
            raise SystemExit(
                f"fetch: {task['name']}: tree {got} != pinned {task['tree']} ({task['id']})"
            )
        unpacked.rename(dest)
        shutil.rmtree(staging)
        fetched.append(f"{task['name']} {task['tree']}")
    (out / "FETCHED.txt").write_text("\n".join(fetched) + "\n")
    return fetched


def _git_env() -> dict[str, str]:
    # No user or system git configuration takes part.
    return {
        "PATH": CELL_PATH,
        "HOME": "/nonexistent",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


# --- arms -------------------------------------------------------------------


def arm_spec(name: str, arms_path: Path = ARMS) -> dict[str, Any]:
    arms = _toml(arms_path).get("arm", {})
    if name not in arms:
        raise SystemExit(f"no arm {name!r} in {arms_path}")
    return {"name": name, **arms[name]}


def bench_path(arms_dir: Path, arm: dict[str, Any]) -> Path:
    base = arms_dir / arm["name"]
    if arm["kind"] == "python":
        return base / "venv" / "bin" / "bench"
    return base / "target" / "release" / "bench"


def build(repo: Path, arms_dir: Path, name: str, *, arms_path: Path = ARMS) -> Path:
    """Make an arm: archive, patch, install (Python) or build (Rust)."""
    arm = arm_spec(name, arms_path)
    base = arms_dir / name
    stamp = {"commit": arm["commit"], "patch": arm.get("patch")}
    if arm.get("patch"):
        stamp["patch_sha256"] = hashlib.sha256(
            (HERE / arm["patch"]).read_bytes()
        ).hexdigest()
    stamp_path = base / "ARM.json"
    if stamp_path.exists():
        if json.loads(stamp_path.read_text()) != stamp:
            raise SystemExit(
                f"build: {base} holds another build; use an empty --arms-dir"
            )
        return bench_path(arms_dir, arm)
    src = base / "src"
    src.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", arm["commit"]],
        check=True,
        capture_output=True,
        env=_git_env(),
    ).stdout
    tar_path = base / "src.tar"
    tar_path.write_bytes(archive)
    with tarfile.open(tar_path) as tar:
        tar.extractall(src, filter="tar")
    tar_path.unlink()
    if arm.get("patch"):
        subprocess.run(
            ["git", "apply", "-p1", "--whitespace=nowarn", str(HERE / arm["patch"])],
            cwd=src,
            check=True,
            env=_git_env(),
        )
    if arm["kind"] == "python":
        env = dict(os.environ)
        env["UV_PROJECT_ENVIRONMENT"] = str(base / "venv")
        subprocess.run(
            ["uv", "sync", "--locked", "--no-dev"], cwd=src, check=True, env=env
        )
    else:
        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(base / "target")
        subprocess.run(
            ["cargo", "build", "--release", "--locked", "-p", "benchflow-cli"],
            cwd=src,
            check=True,
            env=env,
        )
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return bench_path(arms_dir, arm)


# --- cells -------------------------------------------------------------------


def cell_env(home: Path) -> list[str]:
    """``env -i`` with the allowlist; DOCKER_HOST only when the caller set it."""
    argv = ["env", "-i", f"HOME={home}", f"PATH={CELL_PATH}", "LANG=C.UTF-8"]
    if os.environ.get("DOCKER_HOST"):
        argv.append(f"DOCKER_HOST={os.environ['DOCKER_HOST']}")
    return argv


def check_caller_env() -> None:
    present = [name for name in REFUSED_ENV if os.environ.get(name) is not None]
    if present:
        raise SystemExit(
            f"run: {', '.join(present)} is set; the harness compares engines as they "
            "ship, so unset it"
        )


def cell_argv(
    kind: str, bench: Path, task_copy: Path, agent: str, jobs: Path
) -> list[str]:
    if kind == "python":
        return [
            str(bench),
            "eval",
            "run",
            "--tasks-dir",
            str(task_copy.parent),
            "--agent",
            agent,
            "--sandbox",
            "docker",
            "--concurrency",
            "1",
            "--jobs-dir",
            str(jobs),
        ]
    return [
        str(bench),
        "run",
        str(task_copy),
        "-a",
        agent,
        "-s",
        "docker",
        "-j",
        "1",
        "--jobs-dir",
        str(jobs),
        "--fresh",
        "--no-task-skills",
    ]


def containers(name_part: str | None = None) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if name_part is None:
        return out
    needle = name_part.lower().replace("_", "-")
    return [n for n in out if needle in n.lower().replace("_", "-")]


def run_cell(
    *,
    arm: dict[str, Any],
    bench: Path,
    task: dict[str, Any],
    tasks_dir: Path,
    agent: str,
    rnd: int,
    work: Path,
    timeout: int,
) -> dict[str, Any]:
    cell_dir = cell_folder(work, arm["name"], rnd, task["name"], agent)
    # Every arm, agent and round runs the task from the same folder (a task's
    # cells run one after another, never at once): Python 0.8 hashes the
    # source path into a task.md package's task_digest (the materialized
    # package records it), so a folder per cell would change the digest.
    parent = work / "live" / task["name"]
    if parent.exists():
        shutil.rmtree(parent)
    jobs = cell_dir / "jobs"
    home = cell_dir / "home"
    for folder in (parent, jobs, home):
        folder.mkdir(parents=True)
    copy = parent / task["name"]
    shutil.copytree(tasks_dir / task["name"], copy, symlinks=True)
    argv = cell_env(home) + cell_argv(arm["kind"], bench, copy, agent, jobs)
    (cell_dir / "argv.json").write_text(json.dumps(argv) + "\n")
    started = time.monotonic()
    with open(cell_dir / "run.log", "wb") as log:
        try:
            proc = subprocess.run(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            exit_code: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
    wall = time.monotonic() - started
    row: dict[str, Any] = {
        "cell": {
            "name": task["name"],
            "tree": task["tree"],
            "agent": agent,
            "round": rnd,
            "arm": arm["name"],
        },
        "exit_code": exit_code,
        "wall_sec": round(wall, 3),
        "dir": str(cell_dir),
        "record": None,
        "raw": None,
        "reader": None,
        "error": None,
    }
    final = read_cell(row, kind=arm["kind"])
    if final is not None and arm["kind"] == "python":
        row["reader"] = read_with_arm(bench, home, final.parent)
    row["containers_named"] = len(containers(task["name"]))
    return row


def cell_folder(work: Path, arm: str, rnd: int, name: str, agent: str) -> Path:
    return work / arm / f"r{rnd}" / f"{name}__{agent.replace(':', '_')}"


def read_cell(row: dict[str, Any], *, kind: str) -> Path | None:
    """Fill a cell's record, raw view or error from its folder (the pinned
    reader only); return the trial folder read, if any."""
    cell_dir = Path(row["dir"])
    row["record"], row["raw"], row["error"] = None, None, None
    log_text = (cell_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    try:
        trials = normalise.find_trials(cell_dir / "jobs")
        if trials:
            # A retried trial leaves one folder per attempt; the job's
            # summary counts the last, and so does the harness.
            final = normalise.final_attempt(trials)
            record = normalise.normalise_trial(final)
            record["summary"] = normalise.summary_counts(final.parent)
            record["attempts"] = len(trials)
            row["record"] = record
            row["raw"] = normalise.raw_view(final)
            return final
        refused = None
        if kind == "python":
            refused = normalise.refused_at_load_record(
                row["cell"]["name"], row["cell"]["agent"], log_text
            )
        if refused is not None:
            row["record"] = refused
        else:
            row["error"] = (
                "no trial folder and no refusal in the log "
                f"(exit {row.get('exit_code')}); see {cell_dir}/run.log"
            )
    except normalise.NormaliseError as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return None


def renormalise(work: Path) -> int:
    """Read a finished run's cells again with this reader (after a reader
    change), keeping each arm's own reader output; writes <arm>.jsonl in
    place and the old file as <arm>.jsonl.orig."""
    meta = json.loads((work / "run.json").read_text())
    for name, arm in meta["arms"].items():
        path = work / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            cell = row["cell"]
            # The run's folder may have moved since: find the cell under it.
            row["dir"] = str(
                cell_folder(work, name, cell["round"], cell["name"], cell["agent"])
            )
            read_cell(row, kind=arm["kind"])
        if not (work / f"{name}.jsonl.orig").exists():
            path.rename(work / f"{name}.jsonl.orig")
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print(f"{name}: {len(rows)} cells read again")
    return 0


def read_with_arm(bench: Path, home: Path, job_dir: Path) -> dict[str, Any] | None:
    """The arm's own reader: ``bench eval inspect --json <job>``."""
    proc = subprocess.run(
        [*cell_env(home), str(bench), "eval", "inspect", "--json", str(job_dir)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "__error__": {
                "outcome": None,
                "reward": None,
                "error_category": proc.stderr[-500:],
            }
        }
    try:
        return normalise.reader_projection(json.loads(proc.stdout))
    except ValueError:
        return {
            "__error__": {"outcome": None, "reward": None, "error_category": "not JSON"}
        }


def schedule(
    arms: list[str], tasks: list[dict[str, Any]], agents: list[str], rounds: int
):
    """Per round, one group per task: every arm's cells of that task in a
    rotated arm order (the rotation changes every round)."""
    for rnd in range(1, rounds + 1):
        shift = (rnd - 1) % len(arms)
        order = arms[shift:] + arms[:shift]
        groups = []
        # Slow tasks (corpus weight) start first, so they do not finish last.
        for task in sorted(tasks, key=lambda t: -int(t.get("weight", 1))):
            groups.append(
                [(arm, task, agent, rnd) for agent in agents for arm in order]
            )
        yield rnd, groups


def run(args: argparse.Namespace) -> int:
    check_caller_env()
    work: Path = args.work
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"run: {work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    arm_names = [a for a in args.arms.split(",") if a]
    arms = {name: arm_spec(name, args.arms_file) for name in arm_names}
    benches = {name: bench_path(args.arms_dir, arms[name]) for name in arm_names}
    for name, bench in benches.items():
        if not bench.exists():
            raise SystemExit(f"run: arm {name} is not built ({bench})")
    tasks = corpus_tasks(_toml(args.corpus), args.tier)
    if args.names:
        wanted = set(args.names.split(","))
        tasks = [t for t in tasks if t["name"] in wanted]
        missing = wanted - {t["name"] for t in tasks}
        if missing:
            raise SystemExit(f"run: not in the corpus: {sorted(missing)}")
    for task in tasks:
        got = tree_hash(args.tasks_dir / task["name"])
        if got != task["tree"]:
            raise SystemExit(
                f"run: {task['name']} in {args.tasks_dir} is tree {got}, not {task['tree']}"
            )
    agents = [a for a in args.agents.split(",") if a]
    lock = threading.Lock()
    outs = {
        name: open(work / f"{name}.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        for name in arm_names
    }
    meta: dict[str, Any] = {
        "started": datetime.now(UTC).isoformat(),
        "arms": {n: {**arms[n], "bench": str(benches[n])} for n in arm_names},
        "agents": agents,
        "rounds": args.rounds,
        "parallel": args.parallel,
        "tasks": [f"{t['name']} {t['tree']}" for t in tasks],
        "round_wall_sec": {},
        "containers_after_round": {},
    }

    def run_group(group):
        rows = []
        for arm_name, task, agent, rnd in group:
            row = run_cell(
                arm=arms[arm_name],
                bench=benches[arm_name],
                task=task,
                tasks_dir=args.tasks_dir,
                agent=agent,
                rnd=rnd,
                work=work,
                timeout=args.timeout,
            )
            with lock:
                outs[arm_name].write(json.dumps(row, sort_keys=True) + "\n")
                outs[arm_name].flush()
            status = row["error"] or (row["record"] or {}).get("outcome")
            print(
                f"r{rnd} {arm_name} {task['name']}/{agent}: {status} {row['wall_sec']:.1f}s",
                flush=True,
            )
            rows.append(row)
        return rows

    total_start = time.monotonic()
    for rnd, groups in schedule(arm_names, tasks, agents, args.rounds):
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for future in concurrent.futures.as_completed(
                [pool.submit(run_group, g) for g in groups]
            ):
                future.result()
        meta["round_wall_sec"][rnd] = round(time.monotonic() - start, 1)
        meta["containers_after_round"][rnd] = len(containers())
    meta["total_wall_sec"] = round(time.monotonic() - total_start, 1)
    meta["finished"] = datetime.now(UTC).isoformat()
    for handle in outs.values():
        handle.close()
    (work / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: meta[k]
                for k in ("round_wall_sec", "containers_after_round", "total_wall_sec")
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="unpack and check the corpus trees")
    f.add_argument("--repo", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--corpus", type=Path, default=CORPUS)
    f.add_argument("--tier", default="smoke")
    b = sub.add_parser("build", help="make one arm")
    b.add_argument("--repo", type=Path, required=True)
    b.add_argument("--arms-dir", type=Path, required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--arms-file", type=Path, default=ARMS)
    n = sub.add_parser("renormalise", help="read a finished run's cells again")
    n.add_argument("--work", type=Path, required=True)
    r = sub.add_parser("run", help="run cells through arms")
    r.add_argument("--arms", required=True, help="comma-separated arm names")
    r.add_argument("--arms-dir", type=Path, required=True)
    r.add_argument("--arms-file", type=Path, default=ARMS)
    r.add_argument("--tasks-dir", type=Path, required=True)
    r.add_argument("--corpus", type=Path, default=CORPUS)
    r.add_argument("--tier", default="smoke")
    r.add_argument("--names", help="only these corpus names (comma-separated)")
    r.add_argument("--agents", default="oracle,nop")
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("-P", "--parallel", type=int, default=4, help="task groups at once")
    r.add_argument("--timeout", type=int, default=CELL_TIMEOUT_SEC)
    r.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        for line in fetch(args.repo, args.out, corpus_path=args.corpus, tier=args.tier):
            print(line)
        return 0
    if args.command == "build":
        print(build(args.repo, args.arms_dir, args.arm, arms_path=args.arms_file))
        return 0
    if args.command == "renormalise":
        return renormalise(args.work)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
