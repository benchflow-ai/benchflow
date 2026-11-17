"""An R401 refusal's next step names a backend that can run the feature.

Guards the fix against the general ``unsupported_task_feature`` row
of benchflow.failures, which told every refused trial to "run the task with
`--sandbox docker`". Terminal-Bench 4.1's fp8-rmsnorm-gemm, jax-speedrun-gpu
and math-eval-grader ask for GPUs; run with ``--sandbox docker`` they were
told to run with ``--sandbox docker``. A refusal of GPUs alone now names
Modal, the one backend that runs them, and no refusal names the backend the
run already used.
"""

from __future__ import annotations

import pytest

from benchflow._utils.text import describe_exception
from benchflow.failures import cause_of
from benchflow.task.config import TaskConfig
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeatureError,
    validate_task_runtime_support,
)


def _refused(sandbox: dict, backend: str) -> dict:
    config = TaskConfig.model_validate({"sandbox": sandbox})
    features = validate_task_runtime_support(config, sandbox=backend)
    assert features, backend
    error = describe_exception(UnsupportedTaskFeatureError(features))
    return {"task_name": "fp8-rmsnorm-gemm", "error": error}


@pytest.mark.parametrize("backend", ["docker", "daytona"])
@pytest.mark.parametrize(
    "sandbox",
    [{"gpus": 1}, {"gpus": 1, "gpu_types": ["H100"]}, {"gpu_types": ["H100"]}],
)
def test_a_gpu_task_is_sent_to_modal(backend, sandbox):
    cause = cause_of(_refused(sandbox, backend))
    assert cause is not None and cause.key == "unsupported_task_feature"
    assert cause.code == "R401"
    assert "`--sandbox modal`" in cause.next_step
    assert f"--sandbox {backend}" not in cause.next_step


def test_another_refusal_on_docker_never_suggests_docker():
    cause = cause_of(_refused({"tpu": {"type": "v5e", "topology": "2x2"}}, "docker"))
    assert cause is not None and cause.key == "unsupported_task_feature"
    assert "--sandbox docker" not in cause.next_step
    assert "`bench explain R401`" in cause.next_step


def test_gpus_with_another_refused_feature_name_no_backend():
    """Modal would still refuse the other feature: no backend is promised."""
    cause = cause_of(
        _refused({"gpus": 1, "tpu": {"type": "v5e", "topology": "2x2"}}, "docker")
    )
    assert cause is not None
    assert "--sandbox modal" not in cause.next_step
    assert "--sandbox docker" not in cause.next_step


def test_an_unreadable_refusal_keeps_the_general_step():
    cause = cause_of(
        {
            "task_name": "t",
            "error": "UnsupportedTaskFeatureError: Task uses parsed runtime "
            "features that BenchFlow cannot execute on the selected sandbox:",
        }
    )
    assert cause is not None and cause.key == "unsupported_task_feature"
    assert "change what it asks for" in cause.next_step


def test_a_gpu_request_modal_refuses_names_the_fix_not_a_backend():
    """review: Modal's own refusal (two GPU types) fell back to the
    general `--sandbox docker` step."""
    cause = cause_of(_refused({"gpus": 1, "gpu_types": ["H100", "A100"]}, "modal"))
    assert cause is not None and cause.key == "unsupported_task_feature"
    assert "sandbox.gpu_types" in cause.next_step
    assert "--sandbox" not in cause.next_step
