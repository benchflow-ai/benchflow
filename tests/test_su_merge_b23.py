"""Guard the SU b3eb0315 / batch22 a844a8ad Compose fixture boundary."""

import pytest


@pytest.mark.parametrize("value", ["true", "'true'", "'TrUe'", "{name: host}"])
def test_daytona_admission_still_refuses_external_host_networks(
    tmp_path, monkeypatch, value
):
    """Batch 23 constructor doubles cannot waive real task admission."""
    from benchflow.sandbox.setup import _preflight_daytona_compose
    from benchflow.task.config import SandboxConfig
    from benchflow.task.env import TaskEnvNotAllowedError
    from tests.test_su_external_network import _compose

    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(_compose(value))
    with pytest.raises(TaskEnvNotAllowedError, match="host network"):
        _preflight_daytona_compose(environment, "dummy-task", {}, SandboxConfig())


@pytest.mark.parametrize("value", ["false", "'false'", "'FaLsE'"])
def test_daytona_admission_keeps_private_compose_networks(tmp_path, monkeypatch, value):
    """Guard a844a8ad admission alongside SU's normalized false spellings."""
    from benchflow.sandbox.setup import _preflight_daytona_compose
    from benchflow.task.config import SandboxConfig
    from tests.test_su_external_network import _compose

    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(_compose(value))
    _preflight_daytona_compose(environment, "dummy-task", {}, SandboxConfig())
