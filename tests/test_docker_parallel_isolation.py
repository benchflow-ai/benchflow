"""Two Evaluations at once on real Docker: neither loses a container.

Deterministic tier (``tests/integration/README.md``): real sandboxes, no
model (the ``oracle`` agent runs each task's solution). Runs where Docker
answers and skips with a reason otherwise; Docker-only, since the hazard is
one daemon shared by concurrent runs.

Guards the fix for Evaluation's daemon-wide ``docker container prune`` /
``docker network prune`` (now ``sandbox/_docker_sweep.py``). Each scenario
forces the interleaving that lost containers, so it fails deterministically
on the old prune:

- job A's sandbox is between Compose's create and start while job B, in the
  same process, starts and ends (B's start and end sweeps): the hill-climb
  Docker scenario's "container is marked for removal and cannot be started";
- a branch restore has removed the old ``main`` container and not yet made
  its replacement while a sibling job sweeps: the restore then failed on a
  deleted network ("network <project>_default not found"), and with the
  old stop-then-rm order, "removal of container ... is already in progress"
  (test_branch_two_children_in_place on a shared daemon).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
WAIT_SEC = 600

TASK_MD = """---
schema_version: '1.0'
metadata:
  author_name: benchflow
  difficulty: easy
  category: sanity
verifier:
  type: test-script
  timeout_sec: 60.0
agent:
  timeout_sec: 120.0
sandbox:
  build_timeout_sec: 600.0
  cpus: 1
  memory_mb: 512
  allow_internet: true
---

## prompt

Create a file called `hello.txt` in `/app` containing exactly `Hello, world!`.
"""
SOLVE_SH = "#!/bin/bash\nprintf 'Hello, world!\\n' > /app/hello.txt\n"
TEST_SH = """#!/bin/bash
REWARD=0
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then REWARD=1; fi
echo "$REWARD" > /logs/verifier/reward.txt
"""
# ubuntu:24.04 is the deterministic tier's base image, so the tier's hosts
# already hold it.
DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


def _task(root: Path, name: str) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.md").write_text(TASK_MD)
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    for path, text in (
        (task / "solution" / "solve.sh", SOLVE_SH),
        (task / "tests" / "test.sh", TEST_SH),
    ):
        path.write_text(text)
        path.chmod(0o755)
    return task


def _evaluation(root: Path, name: str) -> Evaluation:
    tasks = root / name / "tasks"
    _task(tasks, name)
    return Evaluation(
        tasks_dir=tasks,
        jobs_dir=root / name / "jobs",
        job_name=name,
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=1,
            retry=RetryConfig(max_retries=0),
        ),
    )


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace").strip()


@pytest.mark.usefixtures("require_docker")
async def test_two_evaluations_at_once_keep_their_containers(tmp_path, monkeypatch):
    """Job A stops between ``compose create`` and ``compose start`` until job
    B has run from start to end in the same process; A's container and
    network must still be there, and A must start that same container."""
    job_a = _evaluation(tmp_path, "iso-a")
    job_b = _evaluation(tmp_path, "iso-b")
    original_up = DockerSandbox._run_docker_compose_up
    a_created, b_done = asyncio.Event(), asyncio.Event()
    seen: dict[str, object] = {}

    projects: list[str] = []

    async def up(self: DockerSandbox) -> None:
        projects.append(self.compose_project_name)
        if self.environment_name != "iso-a":
            await original_up(self)
            return
        await self._run_docker_compose_command(["create"])
        listed = await self._run_docker_compose_command(["ps", "-a", "-q", "main"])
        seen["created"] = (listed.stdout or "").strip()
        a_created.set()
        await asyncio.wait_for(b_done.wait(), WAIT_SEC)
        seen["state_after_b"] = await _docker(
            "inspect", "-f", "{{.State.Status}}", str(seen["created"])
        )
        seen["network_after_b"] = (
            await _docker("network", "inspect", f"{self.compose_project_name}_default")
        )[0]
        await self._run_docker_compose_command(["start"])
        listed = await self._run_docker_compose_command(["ps", "-q", "main"])
        seen["started"] = (listed.stdout or "").strip()
        await original_up(self)

    monkeypatch.setattr(DockerSandbox, "_run_docker_compose_up", up)

    async def run_b():
        await asyncio.wait_for(a_created.wait(), WAIT_SEC)
        try:
            return await job_b.run()
        finally:
            b_done.set()

    result_a, result_b = await asyncio.gather(job_a.run(), run_b())

    assert seen["state_after_b"] == (0, "created"), seen
    assert seen["network_after_b"] == 0, seen
    assert seen["started"] == seen["created"], seen
    for result in (result_a, result_b):
        assert (result.total, result.passed, result.errored) == (1, 1, 0), result
    assert len(projects) == 2
    for project in projects:
        code, leftovers = await _docker(
            "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"
        )
        assert (code, leftovers) == (0, ""), f"{project} left containers behind"


@pytest.mark.usefixtures("require_docker")
async def test_a_branch_restore_survives_a_sibling_jobs_sweep(tmp_path, monkeypatch):
    """The sibling job sweeps right after restore removed the old ``main``
    container: the project's network has no container then, and the old
    prune deleted it before ``docker run`` could join it."""
    sibling = _evaluation(tmp_path, "iso-sibling")
    task = _task(tmp_path / "branch", "iso-restore")
    paths = RolloutPaths(rollout_dir=tmp_path / "branch" / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=task / "environment",
        environment_name="iso-restore",
        session_id="n3",  # a branch child's rollout name
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )
    original_cli = sandbox._docker_cli
    swept: list[str] = []

    async def cli(args, check=True):
        result = await original_cli(args, check=check)
        if args[:2] == ["rm", "-f"]:
            await asyncio.to_thread(sibling._prune_docker)
            swept.append(args[2])
        return result

    image = None
    try:
        await sandbox.start(force_build=False)
        await sandbox.exec("echo before > /app/state", user="root")
        image = await sandbox.snapshot()
        await sandbox.exec("echo after > /app/state", user="root")
        monkeypatch.setattr(sandbox, "_docker_cli", cli)
        await sandbox.restore(image)
        monkeypatch.setattr(sandbox, "_docker_cli", original_cli)
        assert swept, "the sibling sweep did not run inside the restore"
        state = await sandbox.exec("cat /app/state", user="root")
        assert state.return_code == 0 and state.stdout.strip() == "before"
    finally:
        monkeypatch.setattr(sandbox, "_docker_cli", original_cli)
        await sandbox.stop(delete=True)
        if image is not None:
            await sandbox.delete_snapshot(image)
    code, leftovers = await _docker(
        "ps",
        "-aq",
        "--filter",
        f"label=com.docker.compose.project={sandbox.compose_project_name}",
    )
    assert (code, leftovers) == (0, "")
