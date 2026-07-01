"""Each DockerProcess gets a unique env-file path.

Regression guard for the arena concurrent floor: N agents run in ONE shared
container, so the per-process env file (`source … && rm`) must NOT collide on a
fixed path — one agent's cleanup would delete the file another is about to source
(observed as `bash: /tmp/.benchflow_env: No such file or directory`).
"""

from __future__ import annotations

from benchflow.sandbox.process import DockerProcess


def test_each_docker_process_gets_unique_env_path():
    a = DockerProcess("proj", "/dir", [], "main")
    b = DockerProcess("proj", "/dir", [], "main")
    assert a._env_path != b._env_path
    assert a._env_path.startswith("/tmp/.benchflow_env_")
    assert b._env_path.startswith("/tmp/.benchflow_env_")
