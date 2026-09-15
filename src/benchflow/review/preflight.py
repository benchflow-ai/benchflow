"""Check reviewer backend readiness without provisioning a sandbox."""

from benchflow.review.options import ReviewerConfig
from benchflow.sandbox.providers import provider_extra


def validate_reviewer_backend(config: ReviewerConfig) -> None:
    """Delegate to backend-owned installation, daemon, and credential checks.

    Cloud preflights validate the configured authentication mechanism; they do
    not reserve capacity or guarantee that a subsequent VM allocation succeeds.
    No successful result is cached across changed credentials or daemon state.
    """
    backend = config.environment
    try:
        if backend == "docker":
            from benchflow.sandbox.docker import DockerSandbox

            DockerSandbox.preflight()
        elif backend == "daytona":
            from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk

            _load_daytona_sdk()
            DaytonaSandbox.preflight()
        elif backend == "modal":
            from benchflow.sandbox.setup import (
                _create_benchflow_modal_environment_class,
            )

            _create_benchflow_modal_environment_class().preflight()
        elif backend == "apple-container":
            from benchflow.sandbox.apple_container import AppleContainerSandbox

            AppleContainerSandbox.preflight()
        elif backend == "agentcore":
            from benchflow.sandbox.agentcore import AgentCoreSandbox

            AgentCoreSandbox.preflight()
        else:
            raise ValueError(
                f"No reviewer preflight is defined for sandbox {backend!r}"
            )
    except ImportError as exc:
        extra = provider_extra(backend)
        requirement = f"benchflow[{extra}]" if extra else "benchflow"
        raise ValueError(
            f"Reviewer sandbox {backend!r} is missing a dependency. "
            f"Install {requirement}: {exc}"
        ) from exc
    except (SystemExit, RuntimeError, OSError) as exc:
        # Backend command preflights historically exit the CLI. At this shared
        # SDK boundary they must be ordinary validation errors instead.
        raise ValueError(f"Reviewer sandbox {backend!r} is not ready: {exc}") from exc
