"""Tests for Evaluation._prune_docker scope (closes #418).

Global ``docker container/network prune`` would delete unrelated user
resources on shared developer or CI hosts. Prune must be scoped to
BenchFlow-owned resources via the ``benchflow.owned=true`` label that the
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


def _fake_run_factory(image_ls_stdout: str = ""):
    """Return a ``subprocess.run`` stub: image-ls yields *image_ls_stdout*.

    Every other docker call returns an empty-stdout success so the prune
    helper's parsing stays deterministic under mocking.
    """

    def _fake_run(cmd, *args, **kwargs):
        stdout = ""
        if cmd[:3] == ["docker", "image", "ls"]:
            stdout = image_ls_stdout
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return _fake_run


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
        with patch(
            "benchflow.evaluation.subprocess.run", side_effect=_fake_run_factory()
        ) as mock_run:
            docker_eval._prune_docker()

        container_cmd = mock_run.call_args_list[0].args[0]
        assert container_cmd[:3] == ["docker", "container", "prune"]
        assert "--filter" in container_cmd
        idx = container_cmd.index("--filter")
        assert container_cmd[idx + 1] == f"label={BENCHFLOW_OWNED_LABEL}"

    def test_network_prune_uses_label_filter(self, docker_eval):
        with patch(
            "benchflow.evaluation.subprocess.run", side_effect=_fake_run_factory()
        ) as mock_run:
            docker_eval._prune_docker()

        network_cmd = mock_run.call_args_list[1].args[0]
        assert network_cmd[:3] == ["docker", "network", "prune"]
        assert "--filter" in network_cmd
        idx = network_cmd.index("--filter")
        assert network_cmd[idx + 1] == f"label={BENCHFLOW_OWNED_LABEL}"

    def test_image_pass_lists_by_label_then_removes_owned_ids(self, docker_eval):
        """Snapshot-image leak fix: list owned images by label, rm by id.

        ``image prune`` only reaps *dangling* images; tagged ``bf-snap-*``
        images survive. The reaper lists owned image ids via the label filter
        and removes exactly those — never an unscoped ``image rm``/``prune``.
        """
        with patch(
            "benchflow.evaluation.subprocess.run",
            side_effect=_fake_run_factory("img-aaa\nimg-bbb\n"),
        ) as mock_run:
            docker_eval._prune_docker()

        cmds = [c.args[0] for c in mock_run.call_args_list]
        ls_cmd = next(c for c in cmds if c[:3] == ["docker", "image", "ls"])
        assert "--filter" in ls_cmd
        idx = ls_cmd.index("--filter")
        assert ls_cmd[idx + 1] == f"label={BENCHFLOW_OWNED_LABEL}"
        assert "-q" in ls_cmd

        rm_cmd = next(c for c in cmds if c[:3] == ["docker", "image", "rm"])
        # Removes exactly the ids the label-scoped list returned.
        assert "img-aaa" in rm_cmd
        assert "img-bbb" in rm_cmd

    def test_image_rm_skipped_when_no_owned_images(self, docker_eval):
        """No owned images => no ``image rm`` shellout at all."""
        with patch(
            "benchflow.evaluation.subprocess.run", side_effect=_fake_run_factory("")
        ) as mock_run:
            docker_eval._prune_docker()

        cmds = [c.args[0] for c in mock_run.call_args_list]
        assert not any(c[:3] == ["docker", "image", "rm"] for c in cmds)

    def test_no_unscoped_removal_call_anywhere(self, docker_eval):
        """Regression guard: every removal call is scoped to the owned label.

        ``prune``/``rm`` of containers, networks, and images must either carry
        the ``--filter label=...`` argument or operate only on ids that came
        from a label-scoped listing (``image rm`` of ids from ``image ls
        --filter``). A re-introduced global prune fails this test.
        """
        with patch(
            "benchflow.evaluation.subprocess.run",
            side_effect=_fake_run_factory("img-aaa\n"),
        ) as mock_run:
            docker_eval._prune_docker()

        for call in mock_run.call_args_list:
            cmd = call.args[0]
            if cmd[:3] == ["docker", "image", "rm"]:
                # rm operates on ids from the label-scoped image ls above —
                # it must never be a wildcard/prune-style call.
                assert "img-aaa" in cmd
                continue
            assert "--filter" in cmd, (
                f"Unscoped Docker call would touch unrelated resources: {cmd}"
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
