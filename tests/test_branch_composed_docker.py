"""Real Docker + SQLite proof for the selective PR #1046 port.

Uses task-owned containers/network/images, no agents or provider credentials.
Set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 and select -m live. On Colima, set
--basetemp to a Docker-shared workspace path rather than macOS private /tmp.
"""

import asyncio
import os
import uuid

import pytest

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.mark.live
@pytest.mark.parametrize("child_fails", [False, True])
async def test_composed_children_and_parent_roundtrip(tmp_path, child_fails):
    """PR #1046: same-name DBs and files roll back after children/errors."""
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("explicit local Docker proof opt-in required")
    project = "bf-composed-proof-" + uuid.uuid4().hex[:12]
    network = project + "_default"
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment_dir,
        environment_name=project,
        session_id=project,
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )
    snapshots = []
    original_snapshot = sandbox.snapshot

    async def snapshot(name=None):
        image = await original_snapshot(name)
        snapshots.append(image)
        return image

    async def current_container():
        result = await sandbox._docker_cli(
            ["ps", "-q", "--filter", f"label=com.docker.compose.project={project}"]
        )
        return (result.stdout or "").strip() or None

    async def exec_in_container(command, *, timeout_sec=120, **kwargs):
        # Real Docker transport without starting a full compose project.
        return await asyncio.wait_for(
            sandbox._docker_cli(
                ["exec", await current_container(), "sh", "-c", command], check=False
            ),
            timeout=timeout_sec,
        )

    async def checked(command):
        result = await exec_in_container(command)
        assert result.return_code == 0, result.stderr
        return (result.stdout or "").strip()

    sandbox._main_container_id = current_container
    sandbox.exec = exec_in_container
    sandbox.snapshot = snapshot
    try:
        await sandbox._docker_cli(["network", "create", network])
        await sandbox._docker_cli(
            [
                "run",
                "--pull=never",
                "-d",
                "--network",
                network,
                "--label",
                f"com.docker.compose.project={project}",
                "--label",
                "com.docker.compose.service=main",
                "--mount",
                f"type=bind,src={paths.verifier_dir},dst=/proof-output",
                "python:3.12-slim",
                "sleep",
                "infinity",
            ]
        )
        await checked("apt-get update -qq && apt-get install -y -qq sqlite3 curl")
        await checked(
            "mkdir -p /mail /calendar && echo parent > /state && "
            "sqlite3 /mail/state.db 'CREATE TABLE state(v); INSERT INTO state VALUES(11);' && "
            "sqlite3 /calendar/state.db 'CREATE TABLE state(v); INSERT INTO state VALUES(22);'"
        )
        await checked("""cat > /service.py <<'PYTHON'
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        values = []
        for path in ('/mail/state.db', '/calendar/state.db'):
            with sqlite3.connect(path) as db:
                values.append(str(db.execute('SELECT v FROM state').fetchone()[0]))
        body = ('\\n'.join(values)).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
HTTPServer(('127.0.0.1', 8099), Handler).serve_forever()
PYTHON""")
        env = ManifestEnvironment(
            EnvironmentManifest.model_validate(
                {
                    "name": "composed",
                    "image": "local-proof",
                    "owns_lifecycle": False,
                    "services": [
                        {
                            "name": "state-reader",
                            "command": "python3 /service.py",
                            "port": 8099,
                        }
                    ],
                    "readiness": {"timeout_sec": 10},
                    "state": {
                        "kind": "sqlite",
                        "paths": ["/mail/state.db", "/calendar/state.db"],
                    },
                }
            ),
            sandbox=sandbox,
        )
        await env.provision(None)
        assert (await env.readiness()).ready
        rollout = Rollout(
            RolloutConfig(
                task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")]
            )
        )
        rollout._environment, rollout._env = env, sandbox
        parent = rollout._cursor
        seen = []
        query = "cat /state; curl -sf http://localhost:8099/"

        async def child(node):
            seen.append(await checked(query))
            await checked(
                "echo child > /state && sqlite3 /mail/state.db 'UPDATE state SET v=99;' && "
                "sqlite3 /calendar/state.db 'UPDATE state SET v=98;' && echo visible > /proof-output/child"
            )
            assert await checked(query) == "child\n99\n98"  # Negative control.
            if child_fails:
                raise ValueError("intentional child failure")
            return 1.0

        if child_fails:
            with pytest.raises(ValueError, match="intentional child failure"):
                await rollout.branch(
                    2, child, snapshot_layers={"environment", "sandbox"}
                )
        else:
            assert (
                await rollout.branch(
                    2, child, snapshot_layers={"environment", "sandbox"}
                )
                == 1
            )
        assert seen == ["parent\n11\n22"] * (1 if child_fails else 2)
        assert await checked(query) == "parent\n11\n22"
        assert rollout._cursor is parent
        assert (paths.verifier_dir / "child").read_text().strip() == "visible"
        handle = parent.state["snapshot"].environment_ref
        assert len(set(handle.files.values())) == 2
    finally:
        result = await sandbox._docker_cli(
            ["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]
        )
        for container in (result.stdout or "").split():
            await sandbox._docker_cli(["rm", "-f", container])
        for image in snapshots:
            await sandbox._docker_cli(["image", "rm", image.ref])
        await sandbox._docker_cli(["network", "rm", network], check=False)
