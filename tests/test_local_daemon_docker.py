"""PERF-F against real Docker: the local runtime changes no result.

Each test runs the real ``bench`` console script on a small task, cold and
through the runtime (attached, and taking a prepared call), and compares the
trial's records with times, names and paths masked. A prepared call must
also leave nothing behind when it is not taken (stop, idle limit), and must
not serve a call whose task changed after it was prepared.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        found = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


pytestmark = [
    pytest.mark.skipif(not _docker_ready(), reason="needs docker compose"),
    pytest.mark.skipif(sys.platform != "linux", reason="the runtime needs Linux"),
]

BENCH = Path(sys.executable).parent / "bench"
STUB = r"""#!/bin/sh
# Minimal ACP agent: answers initialize, session/new and session/prompt.
while IFS= read -r line; do
  id=$(printf '%s' "$line" | grep -o '"id": *[0-9][0-9]*' | head -1 | grep -o '[0-9][0-9]*$')
  m=$(printf '%s' "$line" | grep -o '"method": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$m" in
    initialize) r='{"protocolVersion":1,"agentInfo":{"name":"perf-acp","version":"1"},"agentCapabilities":{"loadSession":false,"promptCapabilities":{"image":false,"audio":false,"embeddedContext":false}},"authMethods":[]}';;
    session/new) r='{"sessionId":"perf-session"}';;
    session/prompt)
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"perf-session","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"done"}}}}\n{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"perf-session","update":{"sessionUpdate":"tool_call","toolCallId":"t1","title":"noop","kind":"execute","status":"completed","rawInput":{}}}}\n'
      r='{"stopReason":"end_turn"}';;
    session/set_config_option) r='{"configOptions":[]}';;
    *) r='{}';;
  esac
  [ -n "$id" ] && printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$r"
done
"""


def _task(root: Path) -> Path:
    task = root / "perf-f-task"
    env = task / "environment"
    env.mkdir(parents=True)
    (env / "acp.sh").write_text(STUB)
    (env / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "COPY acp.sh /opt/perf/acp.sh\n"
        "RUN chmod 755 /opt/perf /opt/perf/acp.sh && mkdir -p /app /logs/verifier /logs/agent\n"
        f"LABEL ai.benchflow.perf={root.name}\n"
        "WORKDIR /app\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\n"
        "if grep -qx ok /app/out.txt 2>/dev/null; then echo 1; else echo 0; fi"
        " > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\necho ok > /app/out.txt\n")
    (task / "instruction.md").write_text("Write ok to /app/out.txt.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 120\n\n[verifier]\ntimeout_sec = 120\n'
    )
    agents = root / "agents" / "perfacp"
    agents.mkdir(parents=True)
    (agents / "manifest.toml").write_text(
        'contract_version = "1.0"\nname = "perfacp"\ndefault_model = "openai/gpt-4.1-mini"\n'
        'install_cmd = "test -x /opt/perf/acp.sh"\nlaunch_cmd = "/opt/perf/acp.sh"\n'
    )
    return task


def _freeze_task_image(task: Path) -> None:
    """Guards 9b0320f: positive preparation controls use immutable images."""
    built = subprocess.run(
        ["docker", "build", "-q", str(task / "environment")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stderr
    image = built.stdout.strip()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image), image
    with (task / "task.toml").open("a") as stream:
        stream.write(f'\n[environment]\ndocker_image = "{image}"\n')
    (task / ".test-image-id").write_text(image)


def _env(root: Path, **extra: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("USER", "LANG", "PATH", "DOCKER_HOST")
    }
    env.update(
        HOME=str(root / "home"),
        TMPDIR=str(root / "tmp"),
        BENCHFLOW_DAEMON_DIR=str(root / "runtime"),
        BENCHFLOW_AGENTS_DIR=str(root / "agents"),
        BENCHFLOW_AGENTS_SOURCE=str(root / "agents"),
    )
    (root / "home").mkdir(exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    env.update(extra)
    return env


def _command(task: Path, jobs: Path, agent: str) -> list[str]:
    cmd = [
        str(BENCH),
        "eval",
        "run",
        "--tasks-dir",
        str(task),
        "--agent",
        agent,
        "--sandbox",
        "docker",
        "--concurrency",
        "1",
        "--jobs-dir",
        str(jobs),
        "--fresh",
        "--retry-attempts",
        "0",
        "--usage-tracking",
        "off",
    ]
    if agent == "perfacp":
        cmd += [
            "--agent-env",
            "OPENAI_API_KEY=perf-f-placeholder",
            "--agent-env",
            "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:9/v1",
        ]
    if agent == "nop":
        cmd += ["--agent-env", "ANTHROPIC_API_KEY=perf-f-placeholder"]
    return cmd


def _call(
    root: Path, task: Path, jobs: Path, agent: str, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    started = time.time()
    done = subprocess.run(
        _command(task, jobs, agent),
        env=env,
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    results = [
        path for path in jobs.glob("*/*/result.json") if path.stat().st_mtime >= started
    ]
    assert len(results) == 1, results
    return done, results[0].parent


def _daemon(root: Path, env: dict[str, str], *args: str) -> None:
    started = subprocess.run(
        [str(BENCH), "daemon", "start", *args],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert started.returncode == 0, started.stderr


def _status(root: Path, env: dict[str, str]) -> dict:
    found = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from benchflow._daemon import request; print(json.dumps(request({'action': 'status'})))",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(found.stdout)


def _stop(root: Path, env: dict[str, str]) -> None:
    for _ in range(600):
        stopped = subprocess.run(
            [str(BENCH), "daemon", "stop"],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if "not running" in stopped.stdout + stopped.stderr or stopped.returncode == 0:
            break
        time.sleep(0.2)
    # The runtime ends once its prepared calls removed what they made.
    for _ in range(1200):
        if not (root / "runtime" / "runtime.lock").exists() or _unlocked(root):
            return
        time.sleep(0.1)


def _unlocked(root: Path) -> bool:
    import fcntl

    fd = os.open(root / "runtime" / "runtime.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _containers(root: Path) -> list[str]:
    found = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=ai.benchflow.perf={root.name}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert found.returncode == 0, found.stderr
    return found.stdout.split()


_MASKED_KEYS = re.compile(
    r"(^|_)(at|timing|seconds|sec|duration|elapsed|rollout_name|job_name|started|finished|time|timestamp|pid|port|id)$"
)
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T_]\d{2}[:-]\d{2}[:-]\d{2}(\.\d+)?")


def _normal(value, masks: list[str]):
    if isinstance(value, dict):
        return {
            key: _normal(item, masks)
            for key, item in sorted(value.items())
            if not _MASKED_KEYS.search(key) and key != "timing"
        }
    if isinstance(value, list):
        return [_normal(item, masks) for item in value]
    if isinstance(value, str):
        for mask in masks:
            value = value.replace(mask, "<path>")
        return _TIME.sub("<time>", value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _record(trial: Path, root: Path) -> dict:
    """The trial's records, with times, names and its own paths masked."""
    masks = [
        str(trial.parent),
        str(trial.parent.parent),
        str(root),
        trial.name,
        trial.parent.name,
    ]
    record: dict = {
        "files": sorted(
            str(path.relative_to(trial)) for path in trial.rglob("*") if path.is_file()
        )
    }
    for name in (
        "result.json",
        "config.json",
        "cloud-metadata.json",
        "artifacts-manifest.json",
    ):
        path = trial / name
        if path.exists():
            record[name] = _normal(json.loads(path.read_text()), masks)
    for name in ("verifier/reward.txt", "rewards.jsonl"):
        path = trial / name
        if path.exists():
            record[name] = _normal(path.read_text(), masks)
    summary = trial.parent / "summary.json"
    record["summary"] = _normal(json.loads(summary.read_text()), masks)
    return record


@pytest.fixture
def root():
    # Short and outside any repo folder: the trial paths are masked by value.
    # Guards PERF-F r7 / 37673d71f: workers must never inspect or mutate
    # another test's containers or build inputs on the shared Docker daemon.
    base = Path(tempfile.mkdtemp(prefix="perf-f-t-", dir="/tmp"))
    yield base
    for record in base.rglob(".test-image-id"):
        subprocess.run(
            ["docker", "image", "rm", record.read_text()],
            capture_output=True,
            timeout=120,
        )
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("agent", ["oracle", "nop", "perfacp"])
def test_attached_run_records_match_a_cold_run(root, agent):
    """Guards PERF-F: a run through the runtime writes the cold run's records."""
    task = _task(root)
    cold_env = _env(root)
    # The first call under this HOME builds the metadata helper image and
    # records it; it starts main before that build is done (keeps NET_ADMIN,
    # never waits), later calls reuse the record. Compare two later calls.
    _call(root, task, root / "jobs-first", agent, cold_env)
    _, cold = _call(root, task, root / "jobs-cold", agent, cold_env)
    warm_env = _env(root, BENCHFLOW_DAEMON="auto")
    _daemon(root, cold_env)
    try:
        before = _status(root, cold_env)["accepted_calls"]
        _, warm = _call(root, task, root / "jobs-warm", agent, warm_env)
        assert _status(root, cold_env)["accepted_calls"] == before + 1
    finally:
        _stop(root, cold_env)
    assert _record(warm, root) == _record(cold, root)
    expected = {"oracle": 1.0, "nop": 0.0, "perfacp": 0.0}[agent]
    assert (
        json.loads((warm / "result.json").read_text())["rewards"]["reward"] == expected
    )
    assert not _containers(root)


@pytest.mark.parametrize("sharing", ["different-home", "same-home", "same-runtime"])
def test_concurrent_cli_calls_keep_their_prepared_sandboxes(root, sharing):
    """Guards PERF-F r7 / 37673d71f: concurrent real CLI jobs with the same
    task basename retain their own images, projects, records and prepared calls.
    One runtime's cleanup must leave the other runtime's prepared call usable.
    """
    lanes = []
    for index in range(2):
        lane = root / f"{root.name}-{index}"
        lane.mkdir()
        task = _task(lane)  # Deliberately the same basename in both checkouts.
        dockerfile = task / "environment" / "Dockerfile"
        with dockerfile.open("a") as handle:
            handle.write(f"RUN echo lane-{index} > /expected.txt\n")
        (task / "solution" / "solve.sh").write_text(
            f"#!/bin/sh\nsleep 3\n"
            f"if grep -qx lane-{index} /expected.txt; then echo ok > /app/out.txt; fi\n"
        )
        _freeze_task_image(task)
        env = _env(lane)
        if sharing != "different-home" and lanes:
            env["HOME"] = lanes[0][2]["HOME"]
        if sharing == "same-runtime" and lanes:
            env["BENCHFLOW_DAEMON_DIR"] = lanes[0][2]["BENCHFLOW_DAEMON_DIR"]
        lanes.append((lane, task, env))
    # A shared runtime keeps the newest call in its single prepared slot.
    # The concurrent pair must take that slot only for its matching job.
    runtimes = lanes[:1] if sharing == "same-runtime" else lanes

    def parked():
        for _ in range(600):
            states = [_status(lane, env) for lane, _, env in runtimes]
            if all(
                state["parked"] == 1 and state["preparing"] == 0 for state in states
            ):
                return states
            time.sleep(0.5)
        pytest.fail(f"Calls did not prepare: {states}")

    def pair(pool, warm):
        futures = [
            pool.submit(
                _call,
                lane,
                task,
                lane / ("jobs-warm" if warm else "jobs-cold"),
                "oracle",
                dict(env, BENCHFLOW_DAEMON="auto") if warm else env,
            )
            for lane, task, env in lanes
        ]
        # Prove overlap on the actual Docker daemon, including the cold pair.
        overlap = False
        while not all(future.done() for future in futures):
            overlap |= all(_containers(lane) for lane, _, _ in lanes)
            time.sleep(0.1)
        trials = [future.result()[1] for future in futures]
        assert overlap, "The CLI jobs did not overlap"
        records = [
            _record(trial, lane)
            for trial, (lane, _, _) in zip(trials, lanes, strict=True)
        ]
        assert all(
            record["result.json"]["rewards"]["reward"] == 1.0 for record in records
        )
        return records

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            cold = pair(pool, False)
            for lane, _, env in runtimes:
                _daemon(
                    lane,
                    env,
                    "--prepare",
                    "1",
                    "--prepare-ttl",
                    "600",
                )
            assert pair(pool, True) == cold
            parked()
            entries = []
            for lane, _, _ in runtimes:
                entries.extend(
                    json.loads(
                        (lane / "runtime" / "prepared.json").read_text()
                    ).values()
                )
            projects = [entry["made"]["project"] for entry in entries]
            assert len(projects) == len(set(projects)) == len(runtimes)
            containers = [cid for lane, _, _ in lanes for cid in _containers(lane)]
            inspected = subprocess.run(
                ["docker", "inspect", *containers],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            boxes = json.loads(inspected.stdout)
            assert {
                box["Config"]["Labels"]["com.docker.compose.project"] for box in boxes
            } == set(projects)
            assert len({box["Config"]["Image"] for box in boxes}) == len(projects)
            assert pair(pool, True) == cold
            states = parked()
            assert [state["prepared_calls"] for state in states] == (
                [1] * len(runtimes)
            )
        if sharing != "same-runtime":
            left, _, left_env = lanes[0]
            right, task, right_env = lanes[1]
            remaining = _containers(right)
            _stop(left, left_env)
            assert not _containers(left)
            assert _containers(right) == remaining
            _, trial = _call(
                right,
                task,
                right / "jobs-warm",
                "oracle",
                dict(right_env, BENCHFLOW_DAEMON="auto"),
            )
            assert _status(right, right_env)["prepared_calls"] == 2
            assert _record(trial, right) == cold[1]
    finally:
        for lane, _, env in runtimes:
            _stop(lane, env)
    assert all(not _containers(lane) for lane, _, _ in lanes)


def test_prepared_run_matches_cold_and_never_serves_a_changed_task(root):
    """Guards PERF-F: a prepared call is the cold run, step for step; a task
    edited after preparation is run fresh; nothing is left after a stop."""
    task = _task(root)
    _freeze_task_image(task)
    env = _env(root)
    warm_env = _env(root, BENCHFLOW_DAEMON="auto")
    jobs = root / "jobs"
    _, cold = _call(root, task, root / "jobs-cold", "oracle", env)
    _daemon(root, env, "--prepare", "1", "--prepare-ttl", "600")
    try:
        _call(root, task, jobs, "oracle", warm_env)
        for _ in range(600):
            if _status(root, env)["parked"] == 1:
                break
            time.sleep(0.5)
        assert _status(root, env)["parked"] == 1
        # While it waits, a plain rerun without --fresh resumes the finished
        # job, cold, as it would without the runtime (A-M-2 / B-L4).
        plain = [word for word in _command(task, jobs, "oracle") if word != "--fresh"]
        resumed = subprocess.run(
            plain,
            env=env,
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert resumed.returncode == 0, resumed.stdout[-1500:] + resumed.stderr[-1500:]
        assert _status(root, env)["parked"] == 1
        _, prepared = _call(root, task, jobs, "oracle", warm_env)
        state = _status(root, env)
        assert state["prepared_calls"] == 1, state
        assert _record(prepared, root) == _record(cold, root)
        timing = json.loads((prepared / "result.json").read_text())["timing"]
        assert timing["warm_pool_idle"] > 0
        for _ in range(600):
            if _status(root, env)["parked"] == 1:
                break
            time.sleep(0.5)
        # A .env appears (it changes a cold call's credentials): the prepared
        # call is not taken; the run is fresh.
        (root / ".env").write_text("PERF_F_NEW_SETTING=1\n")
        _call(root, task, jobs, "oracle", warm_env)
        assert _status(root, env)["prepared_calls"] == 1
        for _ in range(600):
            if _status(root, env)["parked"] == 1:
                break
            time.sleep(0.5)
        # The task changes: the prepared call is not taken, the run is fresh.
        (task / "solution" / "solve.sh").write_text(
            "#!/bin/sh\necho no > /app/out.txt\n"
        )
        _, changed = _call(root, task, jobs, "oracle", warm_env)
        assert _status(root, env)["prepared_calls"] == 1
        assert (
            json.loads((changed / "result.json").read_text())["rewards"]["reward"]
            == 0.0
        )
        jobs_before_stop = sorted(path.name for path in jobs.iterdir() if path.is_dir())
    finally:
        _stop(root, env)
    # The call prepared after the last run is gone, with its job folder.
    assert (
        sorted(path.name for path in jobs.iterdir() if path.is_dir())
        == jobs_before_stop
    )
    assert len(jobs_before_stop) == 4
    assert not _containers(root)
    assert not list((root / "runtime").glob("park-*"))


def test_prepared_call_is_removed_after_its_idle_limit(root):
    """Guards PERF-F: an unused prepared sandbox goes after --prepare-ttl."""
    task = _task(root)
    _freeze_task_image(task)
    env = _env(root)
    jobs = root / "jobs"
    _daemon(root, env, "--prepare", "1", "--prepare-ttl", "5")
    try:
        _call(root, task, jobs, "oracle", _env(root, BENCHFLOW_DAEMON="auto"))
        seen = False
        for _ in range(1200):
            state = _status(root, env)
            seen = seen or state["parked"] == 1
            if seen and state["parked"] == 0 and state["preparing"] == 0:
                break
            time.sleep(0.25)
        assert seen and state["parked"] == 0
        for _ in range(600):
            if not _containers(root):
                break
            time.sleep(0.25)
        assert not _containers(root)
        assert len([path for path in jobs.iterdir() if path.is_dir()]) == 1
    finally:
        _stop(root, env)


def _base(root: Path) -> str:
    # mkdtemp's underscores are valid in a tag, but not after a repository's '-'.
    return f"perf-f-test-base:{root.name}"


def _build_base(root: Path, content: str) -> str:
    """(Re)build the tag a task builds FROM; returns the new image ID."""
    built = subprocess.run(
        [
            "docker",
            "build",
            "-q",
            "--label",
            f"ai.benchflow.perf={root.name}",
            "-t",
            _base(root),
            "-",
        ],
        input=f"FROM ubuntu:24.04\nRUN echo {content} > /base.txt\n",
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stderr
    return built.stdout.strip()


def _side(root: Path) -> str:
    return f"perf-f-test-side:{root.name}"


# How a task's build takes /base.txt from the changing input:
# every one must be caught by the hand-over's rebuild, not by parsing.
_FROM_FORMS = {
    "from": lambda base: f"FROM {base}\n",
    "copy_from": lambda base: (
        f"FROM ubuntu:24.04\nCOPY --from={base} /base.txt /base.txt\n"
    ),
    "mount_from": lambda base: (
        "FROM ubuntu:24.04\n"
        f"RUN --mount=type=bind,from={base},source=/base.txt,target=/mnt/base.txt"
        " cp /mnt/base.txt /base.txt\n"
    ),
    "add_url": lambda base: "FROM ubuntu:24.04\nADD {url}/base.txt /base.txt\n",
}


class _Files(threading.Thread):
    """A local HTTP server for ADD <url> (the Docker daemon on this host
    fetches from it)."""

    def __init__(self, folder: Path) -> None:
        import functools
        import http.server

        super().__init__(daemon=True)
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(folder)
        )
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def run(self) -> None:
        self.server.serve_forever()


def _rest_of_dockerfile(root: Path) -> str:
    return (
        "COPY acp.sh /opt/perf/acp.sh\n"
        "RUN chmod 755 /opt/perf /opt/perf/acp.sh && mkdir -p /app /logs/verifier /logs/agent\n"
        f"LABEL ai.benchflow.perf={root.name}\n"
        "WORKDIR /app\n"
    )


@pytest.mark.parametrize("form", [*_FROM_FORMS, "inline", "include"])
def test_a_changed_build_input_is_never_served(root, form):
    """Guards PERF-F-A-H-1 and 9b0320f: mutable build inputs stay cold.

    An input of the task's build changes between calls (a base image rebuilt under its tag and
    used by FROM, COPY --from or RUN --mount from; the content behind ADD
    <url>; a dockerfile_inline build; an included Compose service's image);
    the next call runs fresh and scores as a cold one."""

    ids = [_build_base(root, "A")]
    task = _task(root)
    environment = task / "environment"
    served_dir = root / "served"
    served_dir.mkdir()
    (served_dir / "base.txt").write_text("A\n")
    files = _Files(served_dir)
    files.start()
    grades = form != "include"
    if form in _FROM_FORMS:
        dockerfile = _FROM_FORMS[form](_base(root)).replace("{url}", files.url)
        (environment / "Dockerfile").write_text(dockerfile + _rest_of_dockerfile(root))
    elif form == "inline":
        # The Dockerfile on disk is not the one built.
        (environment / "docker-compose.yaml").write_text(
            json.dumps(
                {
                    "services": {
                        "main": {
                            "build": {
                                "context": ".",
                                "dockerfile_inline": f"FROM {_base(root)}\n"
                                + _rest_of_dockerfile(root),
                            },
                            "labels": {"ai.benchflow.perf": root.name},
                        }
                    }
                }
            )
        )
    else:
        side_ids = subprocess.run(
            [
                "docker",
                "build",
                "-q",
                "--label",
                f"ai.benchflow.perf={root.name}",
                "-t",
                _side(root),
                "-",
            ],
            input="FROM ubuntu:24.04\nRUN echo one > /side.txt\n",
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert side_ids.returncode == 0, side_ids.stderr
        ids.append(side_ids.stdout.strip())
        (environment / "side.yaml").write_text(
            json.dumps(
                {
                    "services": {
                        "side": {
                            "image": _side(root),
                            "command": ["sleep", "infinity"],
                            "labels": {"ai.benchflow.perf": root.name},
                        }
                    }
                }
            )
        )
        (environment / "docker-compose.yaml").write_text(
            json.dumps({"include": ["side.yaml"]})
        )
    if grades:
        (task / "solution" / "solve.sh").write_text(
            "#!/bin/sh\nif grep -qx A /base.txt; then echo ok > /app/out.txt; fi\n"
        )
    env = _env(root)
    warm_env = _env(root, BENCHFLOW_DAEMON="auto")
    jobs = root / "jobs"
    _daemon(root, env, "--prepare", "1", "--prepare-ttl", "600")
    try:
        _, first = _call(root, task, jobs, "oracle", warm_env)
        assert (
            json.loads((first / "result.json").read_text())["rewards"]["reward"] == 1.0
        )
        for _ in range(120):
            state = _status(root, env)
            if state["preparing"] == 0 and state["workers"] == 0:
                break
            time.sleep(0.5)
        assert state["parked"] == 0 and state["prepared_calls"] == 0, state
        assert not _containers(root)
        first_timing = json.loads((first / "result.json").read_text())["timing"]
        assert "warm_pool_wait" not in first_timing
        if form == "add_url":
            (served_dir / "base.txt").write_text("B\n")
        elif form == "include":
            changed = subprocess.run(
                [
                    "docker",
                    "build",
                    "-q",
                    "--label",
                    f"ai.benchflow.perf={root.name}",
                    "-t",
                    _side(root),
                    "-",
                ],
                input="FROM ubuntu:24.04\nRUN echo two > /side.txt\n",
                capture_output=True,
                text=True,
                timeout=600,
            )
            ids.append(changed.stdout.strip())
        else:
            ids.append(_build_base(root, "B"))
        _, served = _call(root, task, jobs, "oracle", warm_env)
        assert _status(root, env)["prepared_calls"] == 0
        served_timing = json.loads((served / "result.json").read_text())["timing"]
        assert "warm_pool_wait" not in served_timing
        if grades:
            assert (
                json.loads((served / "result.json").read_text())["rewards"]["reward"]
                == 0.0
            )
            _, cold = _call(root, task, root / "jobs-cold", "oracle", env)
            assert (
                json.loads((cold / "result.json").read_text())["rewards"]["reward"]
                == 0.0
            )
    finally:
        _stop(root, env)
        files.server.shutdown()
        for ref in (_base(root), _side(root)):
            subprocess.run(["docker", "rmi", ref], capture_output=True, timeout=120)
        for image in ids:
            subprocess.run(["docker", "rmi", image], capture_output=True, timeout=120)
    assert not _containers(root)


def test_a_taken_call_is_no_longer_the_runtimes_to_reclaim(root):
    """Guards r3 L-A1: once a call takes the prepared sandbox, the runtime
    forgets it, so a runtime killed later never removes that call's sandbox
    (a cold CLI killed the same way would keep it)."""
    task = _task(root)
    _freeze_task_image(task)
    env = _env(root)
    warm_env = _env(root, BENCHFLOW_DAEMON="auto")
    jobs = root / "jobs"
    _daemon(root, env, "--prepare", "1", "--prepare-ttl", "600")
    state_file = root / "runtime" / "prepared.json"
    try:
        _call(root, task, jobs, "oracle", warm_env)
        for _ in range(600):
            state = _status(root, env)
            if state["parked"] == 1:
                break
            time.sleep(0.5)
        taken = str(state["parked_pids"][0])
        project = json.loads(state_file.read_text())[taken]["made"]["project"]
        _call(root, task, jobs, "oracle", warm_env)
        assert _status(root, env)["prepared_calls"] == 1
        assert taken not in json.loads(state_file.read_text())
        runtime_pid = _status(root, env)["pid"]
        os.kill(runtime_pid, 9)
        time.sleep(1)
        _daemon(root, env)
        time.sleep(2)
        assert f"({project})" not in (root / "runtime" / "daemon.log").read_text()
    finally:
        _stop(root, env)


def test_a_killed_prepared_call_leaves_nothing(root):
    """Guards PERF-F-A-L5: a prepared call SIGKILLed while it waits cannot clean
    up; the runtime removes its sandbox, job folder and spool by name."""
    task = _task(root)
    _freeze_task_image(task)
    env = _env(root)
    jobs = root / "jobs"
    _daemon(root, env, "--prepare", "1", "--prepare-ttl", "600")
    try:
        _call(root, task, jobs, "oracle", _env(root, BENCHFLOW_DAEMON="auto"))
        for _ in range(600):
            state = _status(root, env)
            if state["parked"] == 1:
                break
            time.sleep(0.5)
        assert state["parked"] == 1 and _containers(root)
        os.kill(state["parked_pids"][0], 9)
        for _ in range(600):
            if (
                not _containers(root)
                and len([p for p in jobs.iterdir() if p.is_dir()]) == 1
            ):
                break
            time.sleep(0.25)
        assert not _containers(root)
        assert len([p for p in jobs.iterdir() if p.is_dir()]) == 1
        assert not list((root / "runtime").glob("park-*"))
    finally:
        _stop(root, env)
