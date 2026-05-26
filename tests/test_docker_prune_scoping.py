"""Tests for Evaluation._prune_docker scope (closes #418).

Global ``docker container/network prune`` would delete unrelated user
resources on shared developer or CI hosts. Prune must be scoped to
BenchFlow-owned resources via the ``benchflow.owned=true`` label that the
base compose file applies to every container/network we create.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchflow.evaluation import (
    BENCHFLOW_OWNED_LABEL,
    Evaluation,
    EvaluationConfig,
)


@pytest.fixture
def docker_eval(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    cfg = EvaluationConfig(environment="docker")
    return Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        job_name="job-1",
        config=cfg,
    )


@pytest.fixture
def non_docker_eval(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    cfg = EvaluationConfig(environment="daytona")
    return Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        job_name="job-1",
        config=cfg,
    )


class TestPruneScopedToLabel:
    """Verify _prune_docker invokes docker with the BenchFlow label filter.

    These are the security boundary: without the label filter the call
    would also remove unrelated containers/networks on the host.
    """

    def test_label_constant_value(self):
        # If this string ever changes, the compose file and prune call must
        # be updated in lockstep. Lock the value down so the contract is
        # explicit.
        assert BENCHFLOW_OWNED_LABEL == "benchflow.owned=true"

    def test_container_prune_uses_label_filter(self, docker_eval):
        with patch("benchflow.evaluation.subprocess.run") as mock_run:
            docker_eval._prune_docker()

        # Two calls: container prune, then network prune.
        assert mock_run.call_count == 2
        container_cmd = mock_run.call_args_list[0].args[0]
        assert container_cmd[:3] == ["docker", "container", "prune"]
        assert "--filter" in container_cmd
        idx = container_cmd.index("--filter")
        assert container_cmd[idx + 1] == f"label={BENCHFLOW_OWNED_LABEL}"

    def test_network_prune_uses_label_filter(self, docker_eval):
        with patch("benchflow.evaluation.subprocess.run") as mock_run:
            docker_eval._prune_docker()

        network_cmd = mock_run.call_args_list[1].args[0]
        assert network_cmd[:3] == ["docker", "network", "prune"]
        assert "--filter" in network_cmd
        idx = network_cmd.index("--filter")
        assert network_cmd[idx + 1] == f"label={BENCHFLOW_OWNED_LABEL}"

    def test_no_unfiltered_prune_call_anywhere(self, docker_eval):
        """Regression guard: no prune call may omit --filter.

        If someone re-introduces a global ``docker container prune -f`` or
        ``docker network prune -f`` here, this test fails.
        """
        with patch("benchflow.evaluation.subprocess.run") as mock_run:
            docker_eval._prune_docker()

        for call in mock_run.call_args_list:
            cmd = call.args[0]
            # Every prune invocation must carry a --filter argument.
            assert "prune" in cmd, f"Unexpected non-prune call: {cmd}"
            assert "--filter" in cmd, (
                f"Unscoped Docker prune would delete unrelated resources: {cmd}"
            )

    def test_skipped_for_non_docker_environment(self, non_docker_eval):
        """Non-docker environments must not shell out at all."""
        with patch("benchflow.evaluation.subprocess.run") as mock_run:
            non_docker_eval._prune_docker()
        mock_run.assert_not_called()

    def test_subprocess_failure_is_swallowed(self, docker_eval):
        """Prune is best-effort; a subprocess error must not propagate."""
        with patch("benchflow.evaluation.subprocess.run", side_effect=OSError("boom")):
            # Should not raise.
            docker_eval._prune_docker()


class TestComposeBaseLabelsResources:
    """The base compose file must label every container + network we create.

    Without this, the filtered prune above would not find any resources to
    clean up, defeating the fix.
    """

    def _load_compose_base(self) -> dict:
        path = (
            Path(__file__).parent.parent
            / "src"
            / "benchflow"
            / "sandbox"
            / "_compose_files"
            / "docker-compose-base.yaml"
        )
        return yaml.safe_load(path.read_text())

    def test_main_service_carries_benchflow_owned_label(self):
        compose = self._load_compose_base()
        labels = compose["services"]["main"]["labels"]
        # Compose accepts list or dict; either form must include the label.
        if isinstance(labels, dict):
            assert labels.get("benchflow.owned") in ("true", True)
        else:
            assert any(
                "benchflow.owned" in entry and "true" in str(entry) for entry in labels
            )

    def test_default_network_carries_benchflow_owned_label(self):
        compose = self._load_compose_base()
        net_labels = compose["networks"]["default"]["labels"]
        if isinstance(net_labels, dict):
            assert net_labels.get("benchflow.owned") in ("true", True)
        else:
            assert any(
                "benchflow.owned" in entry and "true" in str(entry)
                for entry in net_labels
            )
