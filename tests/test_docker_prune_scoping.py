"""Tests for Evaluation._prune_docker scope (closes #418).

Global ``docker container/network prune`` would delete unrelated user
resources on shared developer or CI hosts. The leftover sweep must be scoped
to BenchFlow-owned resources via the ``benchflow.owned=true`` label that the
base compose file applies to every container/network we create.
"""

from __future__ import annotations

import subprocess
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
    """Verify the leftover sweep lists only BenchFlow-labelled resources.

    These are the security boundary: without the label filter the sweep
    would also consider unrelated containers/networks on the host. Which of
    the listed resources it removes is tested in tests/test_docker_sweep.py.
    """

    SWEEP_RUN = "benchflow.sandbox._docker_sweep.subprocess.run"

    @staticmethod
    def _empty(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "", "")

    def test_label_constant_value(self):
        # If this string ever changes, the compose file and the sweep must
        # be updated in lockstep. Lock the value down so the contract is
        # explicit.
        # "process", not the "true" of BenchFlow before 0.8: an older
        # BenchFlow's daemon-wide prune on the same daemon must not match this
        # version's live containers (tests/test_docker_sweep.py).
        assert BENCHFLOW_OWNED_LABEL == "benchflow.owned=process"

    def test_container_listing_uses_label_filter(self, docker_eval):
        with patch(self.SWEEP_RUN, side_effect=self._empty) as mock_run:
            docker_eval._prune_docker()

        # Two listings (containers, then networks); nothing to remove.
        assert mock_run.call_count == 2
        container_cmd = mock_run.call_args_list[0].args[0]
        assert container_cmd[:3] == ["docker", "ps", "-a"]
        assert f"label={BENCHFLOW_OWNED_LABEL}" in container_cmd

    def test_network_listing_uses_label_filter(self, docker_eval):
        with patch(self.SWEEP_RUN, side_effect=self._empty) as mock_run:
            docker_eval._prune_docker()

        network_cmd = mock_run.call_args_list[1].args[0]
        assert network_cmd[:3] == ["docker", "network", "ls"]
        assert f"label={BENCHFLOW_OWNED_LABEL}" in network_cmd

    def test_no_unfiltered_prune_call_anywhere(self, docker_eval):
        """Regression guard: no daemon-wide prune, and every listing filtered.

        If someone re-introduces a global ``docker container prune -f`` or
        ``docker network prune -f`` here, this test fails.
        """
        with patch(self.SWEEP_RUN, side_effect=self._empty) as mock_run:
            docker_eval._prune_docker()

        for call in mock_run.call_args_list:
            cmd = call.args[0]
            assert "prune" not in cmd, f"Daemon-wide Docker prune: {cmd}"
            assert f"label={BENCHFLOW_OWNED_LABEL}" in cmd, (
                f"Unscoped Docker listing would consider unrelated resources: {cmd}"
            )

    def test_skipped_for_non_docker_environment(self, non_docker_eval):
        """Non-docker environments must not shell out at all."""
        with patch(self.SWEEP_RUN) as mock_run:
            non_docker_eval._prune_docker()
        mock_run.assert_not_called()

    def test_subprocess_failure_is_swallowed(self, docker_eval):
        """The sweep is best-effort; a subprocess error must not propagate."""
        with patch(self.SWEEP_RUN, side_effect=OSError("boom")):
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
            assert labels.get("benchflow.owned") == "process"
        else:
            assert any(
                "benchflow.owned" in entry and "process" in str(entry)
                for entry in labels
            )

    def test_default_network_carries_benchflow_owned_label(self):
        compose = self._load_compose_base()
        net_labels = compose["networks"]["default"]["labels"]
        if isinstance(net_labels, dict):
            assert net_labels.get("benchflow.owned") == "process"
        else:
            assert any(
                "benchflow.owned" in entry and "process" in str(entry)
                for entry in net_labels
            )

    @pytest.mark.parametrize(
        "name", ["docker-compose-base.yaml", "docker-compose-remote-base.yaml"]
    )
    def test_main_and_network_carry_the_creating_process(self, name):
        """The sweep removes a resource only once the process named by
        ``benchflow.process`` is gone (tests/test_docker_sweep.py); the value
        comes from BENCHFLOW_PROCESS, which DockerSandbox always sets."""
        path = Path(__file__).parent.parent / "src/benchflow/sandbox/_compose_files"
        compose = yaml.safe_load((path / name).read_text())
        expected = "${BENCHFLOW_PROCESS:-}"
        assert compose["services"]["main"]["labels"]["benchflow.process"] == expected
        assert compose["networks"]["default"]["labels"]["benchflow.process"] == expected
