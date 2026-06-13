"""Tests for rendering a task-declared healthcheck into the compose service.

A task can declare ``environment.healthcheck`` (parsed into
``SandboxConfig.healthcheck``). Without rendering it into the generated
docker-compose service, ``compose up --wait`` only waits for healthchecks
already present in the compose files, so a container can be treated "up"
before its in-container service is ready (boot race). These unit tests assert
the render maps every field onto the right Compose key and that nothing is
emitted when no healthcheck is declared. No Docker daemon is required.
"""

import json
from pathlib import Path

from benchflow.sandbox._compose import healthcheck_to_compose_block
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import HealthcheckConfig, SandboxConfig
from benchflow.task.paths import RolloutPaths


def _make_sandbox(tmp_path: Path, task_env_config: SandboxConfig) -> DockerSandbox:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.task_env_config = task_env_config
    sandbox.rollout_paths = RolloutPaths(rollout_dir=tmp_path)
    # _docker_compose_paths reads these; the environment dir has no
    # docker-compose.yaml so only BenchFlow's own compose files are listed.
    sandbox.environment_dir = tmp_path
    sandbox._use_prebuilt = False
    sandbox._mounts_json = None
    sandbox._mounts_compose_path = None
    sandbox._healthcheck_compose_path = None
    return sandbox


class TestHealthcheckToComposeBlock:
    def test_maps_all_fields_to_compose_keys(self) -> None:
        hc = HealthcheckConfig(
            command="curl -fsS http://localhost:8080/health",
            interval_sec=2.0,
            timeout_sec=10.0,
            start_period_sec=15.0,
            start_interval_sec=1.5,
            retries=7,
        )

        block = healthcheck_to_compose_block(hc)

        assert block == {
            "test": ["CMD-SHELL", "curl -fsS http://localhost:8080/health"],
            "interval": "2s",
            "timeout": "10s",
            "retries": 7,
            "start_period": "15s",
            "start_interval": "1.5s",
        }

    def test_defaults_render_expected_durations(self) -> None:
        block = healthcheck_to_compose_block(HealthcheckConfig(command="true"))

        # HealthcheckConfig defaults: interval=5, timeout=30, start_period=0,
        # start_interval=5, retries=3.
        assert block == {
            "test": ["CMD-SHELL", "true"],
            "interval": "5s",
            "timeout": "30s",
            "retries": 3,
            "start_period": "0s",
            "start_interval": "5s",
        }


class TestDockerSandboxHealthcheckRender:
    def test_declared_healthcheck_written_to_override_and_path_list(
        self, tmp_path: Path
    ) -> None:
        hc = HealthcheckConfig(
            command="pg_isready -U postgres",
            interval_sec=3.0,
            timeout_sec=4.0,
            retries=5,
        )
        sandbox = _make_sandbox(tmp_path, SandboxConfig(healthcheck=hc))

        path = sandbox._write_healthcheck_compose_file()

        assert path == tmp_path / "docker-compose-healthcheck.json"
        compose = json.loads(path.read_text())
        assert compose == {
            "services": {
                "main": {
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U postgres"],
                        "interval": "3s",
                        "timeout": "4s",
                        "retries": 5,
                        "start_period": "0s",
                        "start_interval": "5s",
                    }
                }
            }
        }

        # Once written, the override participates in the compose file list so
        # `compose up --wait` honors it.
        sandbox._healthcheck_compose_path = path
        assert path in sandbox._docker_compose_paths

    def test_no_healthcheck_emits_no_override(self, tmp_path: Path) -> None:
        sandbox = _make_sandbox(tmp_path, SandboxConfig())

        # No healthcheck declared: nothing is rendered and the override never
        # joins the compose file list (default behavior unchanged).
        assert sandbox.task_env_config.healthcheck is None
        assert sandbox._healthcheck_compose_path is None
        assert not (tmp_path / "docker-compose-healthcheck.json").exists()
        assert sandbox._docker_compose_paths.count(None) == 0
        rendered = [
            p
            for p in sandbox._docker_compose_paths
            if p.name == "docker-compose-healthcheck.json"
        ]
        assert rendered == []
