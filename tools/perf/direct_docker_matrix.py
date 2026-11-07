"""Run the direct Docker path's differential over a task set (PERF-E).

For every case (a TB2 task, or a synthetic single-service task with one Compose
key set), three ``bench eval run --agent nop`` trials go through the
recording proxy, one after another, on one pinned /28
(``BENCHFLOW_DOCKER_SUBNETS``) so the daemon's address choice is the same:

- ``compose-a`` and ``compose-b``: the reference Compose path
  (``BENCHFLOW_DIRECT_DOCKER=0``), twice, to record Compose's own variation;
- ``direct``: the same tip with the direct path on.

Each case is then ``identical`` (both layers, direct vs compose-a),
``fallback`` (the direct trial ran on Compose; its recorded reason is
listed) or ``different`` (the layers that differ are listed). Outcomes,
leftover containers and networks, and the Compose-vs-Compose differences go
in ``<out>/matrix.json``.

Usage (on a Docker VM, never the Mac)::

    python -I tools/perf/direct_docker_matrix.py --out DIR --bench-ref V1 \\
        --bench-tip V2 --tasks-dir TB2 [--tasks a b ...] [--synthetic IMAGE]
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from direct_docker_diff import compare, load  # noqa: E402

RECORDER = HERE.parents[2] / "tests" / "_docker_api_recorder.py"


def free_subnet() -> str:
    """A /28 no daemon network overlaps, at random from 10.251.0.0/16, picked
    once per matrix run (B-R3-L5); PERF_E_SUBNET overrides it."""

    ids = subprocess.run(
        ["docker", "network", "ls", "-q"], capture_output=True, text=True
    ).stdout.split()
    out = (
        subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "--format",
                "{{range .IPAM.Config}}{{.Subnet}} {{end}}",
                *ids,
            ],
            capture_output=True,
            text=True,
        ).stdout
        if ids
        else ""
    )
    used = []
    for word in out.split():
        with contextlib.suppress(ValueError):
            used.append(ipaddress.ip_network(word, strict=False))
    pool = list(ipaddress.ip_network("10.251.0.0/16").subnets(new_prefix=28))
    random.shuffle(pool)
    for candidate in pool:
        if not any(candidate.overlaps(net) for net in used if net.version == 4):
            return str(candidate)
    raise SystemExit("no free /28 in 10.251.0.0/16 on this daemon")


SUBNET: str | None = None

#: One case per Compose service key: (key, value, extra top-level document).
#: Keys the direct path renders must come out identical; every other key
#: must fall back with its reason (or be refused by the launch gate on both
#: paths alike).
DOCKERFILE_VOLUME = "image-volume"
SYNTHETIC: list[tuple[str, object, dict]] = [
    ("annotations", {"io.perf-e.a": "b"}, {}),
    ("attach", False, {}),
    ("blkio_config", {"weight": 300}, {}),
    ("cap_add", ["SYS_PTRACE"], {}),
    ("cap_drop", ["MKNOD"], {}),
    ("cgroup", "private", {}),
    ("cgroup_parent", "perf-e.slice", {}),
    ("command", ["sh", "-c", "exec sleep infinity"], {}),
    (
        "configs",
        ["perf-e-config"],
        {"configs": {"perf-e-config": {"file": "./config.txt"}}},
    ),
    ("container_name", "perf-e-fixed-name", {}),
    ("cpu_count", 1, {}),
    ("cpu_percent", 50, {}),
    ("cpu_period", 100000, {}),
    ("cpu_quota", 50000, {}),
    ("cpu_shares", 512, {}),
    ("cpus", 0.5, {}),
    ("cpuset", "0", {}),
    (
        "deploy",
        {
            "resources": {
                "limits": {"cpus": "0.3", "memory": "512M", "pids": 300},
                "reservations": {"memory": "64M"},
            }
        },
        {},
    ),
    ("device_cgroup_rules", ["c 1:3 mr"], {}),
    ("devices", ["/dev/null:/dev/perf-e-null"], {}),
    ("dns", ["1.1.1.1"], {}),
    ("dns_opt", ["ndots:2"], {}),
    ("dns_search", ["example.com"], {}),
    ("domainname", "example.com", {}),
    ("entrypoint", ["sh", "-c", "exec sleep infinity"], {}),
    ("environment", {"PERF_E": "a$$b c", "EMPTY": ""}, {}),
    ("env_file", ["./perf-e.env"], {}),
    ("expose", ["8080"], {}),
    ("extra_hosts", ["perf-e-host:10.0.0.1"], {}),
    ("group_add", ["audio"], {}),
    ("healthcheck", {"test": ["CMD", "true"], "interval": "1s", "retries": 3}, {}),
    ("hostname", "perf-e-host", {}),
    ("init", True, {}),
    ("ipc", "shareable", {}),
    ("labels", {"io.perf-e.label": "x<y>&z"}, {}),
    ("logging", {"driver": "json-file", "options": {"max-size": "1m"}}, {}),
    ("mac_address", "02:42:ac:11:00:99", {}),
    ("mem_limit", "1g", {}),
    ("mem_reservation", "256m", {}),
    ("mem_swappiness", 10, {}),
    ("memswap_limit", "2g", {}),
    ("networks", {"default": {"aliases": ["perf-e-alias"]}}, {}),
    ("network_mode", "none", {}),
    ("oom_kill_disable", True, {}),
    ("oom_score_adj", 100, {}),
    ("pid", "host", {}),
    ("pids_limit", 300, {}),
    ("platform", "linux/amd64", {}),
    ("ports", ["8080"], {}),
    ("privileged", True, {}),
    ("pull_policy", "never", {}),
    ("read_only", True, {}),
    ("restart", "on-failure:2", {}),
    ("runtime", "runc", {}),
    (
        "secrets",
        ["perf-e-secret"],
        {"secrets": {"perf-e-secret": {"file": "./secret.txt"}}},
    ),
    ("security_opt", ["label=disable"], {}),
    ("shm_size", "128m", {}),
    ("stdin_open", True, {}),
    ("stop_grace_period", "3s", {}),
    ("stop_signal", "SIGINT", {}),
    ("sysctls", {"net.ipv4.tcp_syncookies": 0}, {}),
    ("tmpfs", ["/run/perf-e"], {}),
    ("tty", True, {}),
    ("ulimits", {"nofile": {"soft": 1024, "hard": 2048}}, {}),
    ("user", "0:0", {}),
    ("userns_mode", "host", {}),
    ("uts", "host", {}),
    ("volume_driver", "local", {}),
    ("volumes", ["./data:/perf-e-data:ro"], {}),
    ("volumes", ["perf-e-vol:/perf-e-vol"], {"volumes": {"perf-e-vol": {}}}),
    ("volumes", [{"type": "tmpfs", "target": "/perf-e-tmp"}], {}),
    ("working_dir", "/tmp", {}),
    ("post_start", [{"command": ["true"]}], {}),
    ("pre_stop", [{"command": ["true"]}], {}),
    ("profiles", ["perf-e"], {}),
    ("storage_opt", {"size": "1G"}, {}),
    ("isolation", "default", {}),
    ("gpus", "all", {}),
    # Accepted shapes review B found never diffed (PERF-E-B-L5).
    ("cgroup", "host", {}),
    ("init", False, {}),
    ("security_opt", ["no-new-privileges", "label:disable"], {}),
    ("pull_policy", "if_not_present", {}),
    (
        "labels",
        {"io.perf-e.net": "1"},
        {"networks": {"default": {"labels": {"io.perf-e.net": "x<y>&z"}}}},
    ),
    # Not a Compose key: a task built from a Dockerfile whose image declares a
    # VOLUME, with no compose file (PERF-E-A-1). Compose's recreate carries
    # the prototype's anonymous volume over to main.
    (DOCKERFILE_VOLUME, "FROM ubuntu:24.04\nVOLUME /data\nWORKDIR /app\n", {}),
]


def synthetic_task(
    root: Path, base: Path, image: str, key: str, value: object, top: dict
) -> Path:
    """A copy of *base* (a TB2 task) whose own compose file sets *key* on main."""
    shutil.copytree(base, root)
    env = root / "environment"
    for child in env.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (env / "data").mkdir()
    (env / "config.txt").write_text("perf-e\n")
    (env / "secret.txt").write_text("perf-e\n")
    (env / "perf-e.env").write_text("PERF_E_FILE=1\n")
    if key == DOCKERFILE_VOLUME:
        assert isinstance(value, str)
        (env / "Dockerfile").write_text(value)
        config = root / "task.toml"
        config.write_text(re.sub(r"(?m)^docker_image\s*=.*\n", "", config.read_text()))
    else:
        main: dict = {"image": image, key: value}
        (env / "docker-compose.yaml").write_text(
            json.dumps({"services": {"main": main}, **top})
        )
    tests = root / "tests"
    (tests / "test.sh").write_text("#!/bin/sh\necho 0 > /logs/verifier/reward.txt\n")
    return root


def leftovers(project_hint: str) -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout.split()
    nets = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return [n for n in out + nets if n.startswith(project_hint)]


def run(
    out: Path,
    bench: Path,
    tasks_dir: Path,
    task: str,
    *,
    direct: bool,
    home: Path,
    host_access: bool = False,
) -> dict:
    out.mkdir(parents=True)
    sock = out / "proxy.sock"
    log = out / "api.jsonl"
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(RECORDER),
            str(sock),
            "/var/run/docker.sock",
            str(log),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proxy.stdout is not None
    proxy.stdout.readline()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "BENCHFLOW_TELEMETRY_DISABLED": "1",
        "DO_NOT_TRACK": "1",
        "DOCKER_HOST": f"unix://{sock}",
        "BENCHFLOW_DOCKER_SUBNETS": SUBNET or "",
        "BENCHFLOW_DIRECT_DOCKER": "1" if direct else "0",
    }
    if host_access:
        # DOCKER_HOST (the proxy) makes the launch gate treat the daemon as
        # remote, which refuses every task compose file's binds; the
        # synthetic cases skip that check on all three arms alike.
        env["BENCHFLOW_ALLOW_TASK_HOST_ACCESS"] = "1"
    started = time.time()
    with (out / "bench.log").open("w") as f:
        rc = subprocess.run(
            [str(bench), "eval", "run", "--tasks-dir", str(tasks_dir), "--include", task,
             "--agent", "nop", "--sandbox", "docker", "--jobs-dir", str(out / "jobs"),
             "--retry-attempts", "0", "--concurrency", "1"],
            env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1500,
        ).returncode  # fmt: skip
    proxy.send_signal(signal.SIGTERM)
    try:
        proxy.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proxy.kill()
        proxy.wait()
    record: dict = {"rc": rc, "seconds": round(time.time() - started, 1)}
    for path in (out / "jobs").rglob("config.json"):
        data = json.loads(path.read_text())
        if "direct_docker" in data:
            record["direct_docker"] = data["direct_docker"]
    for path in (out / "jobs").rglob("result.json"):
        data = json.loads(path.read_text())
        if "rewards" in data or "error" in data:
            record["reward"] = (data.get("rewards") or {}).get("reward")
            error = data.get("error") or data.get("exception_info")
            if error:
                record["error"] = str(error)[:300]
    project = None
    if log.exists():
        for r in load(log):
            if r.get("kind") == "request" and "/networks/create" in r["path"]:
                project = r["body"]["Name"].removesuffix("_default")
                break
            if r.get("kind") == "request" and "/containers/create" in r["path"]:
                project = (
                    (r.get("body") or {})
                    .get("Labels", {})
                    .get("com.docker.compose.project")
                )
                break
    record["project"] = project
    record["leftovers"] = leftovers(project) if project else []
    # Volumes count too: every volume a snapshot of the project's containers
    # showed mounted must be gone (DOCKER-ANON-VOLUME-LEAK).
    mounted = {
        m["Name"]
        for r in (load(log) if log.exists() else [])
        if r.get("kind") == "snapshot" and r.get("what") == "container"
        for m in ((r.get("body") or {}).get("Mounts") or [])
        if isinstance(m, dict) and m.get("Type") == "volume" and m.get("Name")
    }
    record["leftover_volumes"] = sorted(
        name
        for name in mounted
        if subprocess.run(
            ["docker", "volume", "inspect", name], capture_output=True
        ).returncode
        == 0
    )
    return record


def case(
    out: Path,
    args: argparse.Namespace,
    name: str,
    tasks_dir: Path,
    task: str,
    *,
    host_access: bool = False,
) -> dict:
    home = out / "home"
    home.mkdir(parents=True, exist_ok=True)
    # A case a stopped matrix left half done starts again from nothing.
    shutil.rmtree(out / name, ignore_errors=True)
    result: dict = {"case": name}
    for label, bench, direct in (
        ("compose-a", args.bench_ref, False),
        ("compose-b", args.bench_ref, False),
        ("direct", args.bench_tip, True),
    ):
        result[label] = run(
            out / name / label,
            bench,
            tasks_dir,
            task,
            direct=direct,
            home=home,
            host_access=host_access,
        )
    judge(out, result)
    print(
        json.dumps(
            {k: result.get(k) for k in ("case", "verdict", "reason", "same_outcome")}
            | {"differences": result.get("differences", [])[:6]}
        ),
        flush=True,
    )
    return result


def judge(out: Path, result: dict) -> None:
    """Compare the case's saved logs (again, with --recompare)."""
    name = result["case"]
    for key in ("verdict", "reason", "differences", "container_starts"):
        result.pop(key, None)
    a, b, d = (
        out / name / k / "api.jsonl" for k in ("compose-a", "compose-b", "direct")
    )
    try:
        result["compose_vs_compose"] = compare(a, b)
    except Exception as exc:  # no container made (refused by the gate)
        result["compose_vs_compose"] = {"error": str(exc)}
    path = (result["direct"].get("direct_docker") or {}).get("path")
    if path == "direct":
        try:
            verdict = compare(a, d)
            result["verdict"] = "identical" if verdict["identical"] else "different"
            result["differences"] = verdict["layer1"] + verdict["layer2"]
            result["container_starts"] = verdict["container_starts"]
        except Exception as exc:
            result["verdict"] = "different"
            result["differences"] = [f"comparison failed: {exc}"]
    elif path == "compose":
        result["verdict"] = "fallback"
        result["reason"] = result["direct"]["direct_docker"].get("reason")
    else:
        result["verdict"] = "no sandbox"
    result["same_outcome"] = all(
        result[k].get("reward") == result["compose-a"].get("reward")
        and bool(result[k].get("error")) == bool(result["compose-a"].get("error"))
        for k in ("compose-b", "direct")
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bench-ref", type=Path, required=True)
    p.add_argument("--bench-tip", type=Path, required=True)
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--synthetic", default=None, help="image for the synthetic cases")
    p.add_argument("--synthetic-base", default="regex-log")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--recompare", action="store_true")
    a = p.parse_args()
    global SUBNET
    SUBNET = os.environ.get("PERF_E_SUBNET") or free_subnet()
    print(json.dumps({"subnet": SUBNET}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    matrix = a.out / "matrix.json"
    if matrix.exists():
        results = json.loads(matrix.read_text())
    if a.recompare:
        for r in results:
            judge(a.out, r)
        matrix.write_text(json.dumps(results, indent=1))
        a.tasks, a.synthetic = [], None
    done = {r["case"] for r in results}
    for task in a.tasks:
        if task in done:
            continue
        results.append(case(a.out, a, task, a.tasks_dir, task))
        matrix.write_text(json.dumps(results, indent=1))
    if a.synthetic:
        synth = a.out / "synthetic-tasks"
        for i, (key, value, top) in enumerate(SYNTHETIC):
            name = f"perf-e-key-{i:02d}-{key.replace('_', '-')}"
            if name in done or (a.only and key not in a.only):
                continue
            if not (synth / name).exists():
                synthetic_task(
                    synth / name,
                    a.tasks_dir / a.synthetic_base,
                    a.synthetic,
                    key,
                    value,
                    top,
                )
            results.append(case(a.out, a, name, synth, name, host_access=True))
            matrix.write_text(json.dumps(results, indent=1))
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
