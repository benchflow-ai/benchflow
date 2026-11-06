"""PERF-D: start() keeps main's prebuilt overlay when its bytes are the same.

start() writes the overlay before and after the cached layers. When the second
write would be the same bytes, the file (and so the launch key and the frozen
project) stays, and freeze_launch does not resolve and check the same project
a second time. Another image writes a new overlay.
"""

from __future__ import annotations

from pathlib import Path


def _prebuilt_sandbox(tmp_path: Path):
    from tests.test_docker_direct_exec import _sandbox

    sandbox = _sandbox(tmp_path)
    sandbox._use_prebuilt = True
    return sandbox


def test_the_same_prebuilt_overlay_is_kept(tmp_path, monkeypatch):
    from benchflow.sandbox.docker import DockerSandbox

    monkeypatch.setattr(DockerSandbox, "_uses_compose", property(lambda self: True))
    sandbox = _prebuilt_sandbox(tmp_path)
    sandbox._env_vars.prebuilt_image_name = "fixture-main:latest"

    sandbox._write_prebuilt_compose_file()
    first = sandbox._runtime_compose_path
    sandbox._write_prebuilt_compose_file()

    assert sandbox._runtime_compose_path == first and first.exists()

    # Another image: a new overlay, and the old one goes.
    sandbox._env_vars.prebuilt_image_name = "fixture-main:other"
    sandbox._write_prebuilt_compose_file()
    assert sandbox._runtime_compose_path != first and not first.exists()
    assert "fixture-main:other" in sandbox._runtime_compose_path.read_text()

    # Not prebuilt any more: no overlay.
    sandbox._use_prebuilt = False
    previous = sandbox._runtime_compose_path
    sandbox._write_prebuilt_compose_file()
    assert sandbox._runtime_compose_path is None and not previous.exists()


def test_the_same_overlay_keeps_the_frozen_launch(tmp_path, monkeypatch):
    """The launch key covers the run's files by path; the same overlay keeps
    its path, so freeze_launch keeps the project it checked."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.compose_env import launch_key

    monkeypatch.setattr(DockerSandbox, "_uses_compose", property(lambda self: True))
    sandbox = _prebuilt_sandbox(tmp_path)
    sandbox._env_vars.prebuilt_image_name = "fixture-main:latest"
    sandbox._write_prebuilt_compose_file()
    before = launch_key([sandbox._runtime_compose_path], {}, (), "p", tmp_path)
    sandbox._write_prebuilt_compose_file()
    after = launch_key([sandbox._runtime_compose_path], {}, (), "p", tmp_path)
    assert before == after


def test_the_factory_check_is_the_one_start_keeps(tmp_path, monkeypatch):
    """The trial factory checks the project start() will launch (main's
    prebuilt overlay included), so start()'s first freeze keeps it instead of
    resolving and checking it again; a forced build still checks again."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task import compose_env
    from benchflow.task.compose_env import ComposeCheck, ComposeLaunch, launch_key
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    checked: list[list[str]] = []

    def freeze(*, staged, files, project_name, env, docker_args, **kwargs):
        checked.append([str(f) for f in files])
        directory = staged.environment_dir
        return ComposeCheck(
            launch=ComposeLaunch(
                path=tmp_path / "compose.json",
                project_dir=tmp_path,
                project_name=project_name,
                env=env,
                docker_args=tuple(docker_args),
                files=tuple(files),
                key=launch_key(files, env, docker_args, project_name, directory),
            )
        )

    monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: fixture-main:latest\n"
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment,
        environment_name="perf-d",
        session_id="perf-d",
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image="fixture-main:latest"),
    )

    sandbox.freeze_launch()  # the trial factory's check
    # What start() does first with force_build False.
    sandbox._use_prebuilt = True
    sandbox._write_prebuilt_compose_file()
    sandbox.freeze_launch(False)
    assert len(checked) == 1
    assert any("benchflow-prebuilt-" in f for f in checked[0])

    # A forced build: no prebuilt overlay, so the project is checked again.
    sandbox._use_prebuilt = False
    sandbox._write_prebuilt_compose_file()
    sandbox.freeze_launch(True)
    assert len(checked) == 2
    assert not any("benchflow-prebuilt-" in f for f in checked[1])


def test_a_forced_build_after_the_factory_check_drops_the_early_overlay(
    tmp_path, monkeypatch
):
    """freeze_launch(force_build=True) without start() (an SDK caller, the
    launch tests) checks the build files, not the factory's prebuilt overlay."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task import compose_env
    from benchflow.task.compose_env import ComposeCheck, ComposeLaunch, launch_key
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    checked: list[list[str]] = []

    def freeze(*, staged, files, project_name, env, docker_args, **kwargs):
        checked.append([str(f) for f in files])
        return ComposeCheck(
            launch=ComposeLaunch(
                path=tmp_path / "compose.json",
                project_dir=tmp_path,
                project_name=project_name,
                env=env,
                docker_args=tuple(docker_args),
                files=tuple(files),
                key=launch_key(
                    files, env, docker_args, project_name, staged.environment_dir
                ),
            )
        )

    monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: fixture-main:latest\n"
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment,
        environment_name="perf-d",
        session_id="perf-d",
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image="fixture-main:latest"),
    )

    sandbox.freeze_launch()
    early = sandbox._runtime_compose_path
    assert early is not None
    sandbox.freeze_launch(force_build=True)

    assert len(checked) == 2
    assert not any("benchflow-prebuilt-" in f for f in checked[1])
    assert sandbox._runtime_compose_path is None and not early.exists()
