"""LiteLLM startup patch for transient upstream 5xx retry policy.

LiteLLM 1.89.0's ``RetryPolicy`` model accepts
``InternalServerErrorRetries``, but the Router helper that reads retry-policy
fields never checks that attribute. BenchFlow still keeps the proxy-wide
``num_retries=0`` fail-fast default; this patch makes the existing
``InternalServerErrorRetries`` policy work for the one transient 5xx class.

This module is copied into the per-run LiteLLM runtime and imported by
``sitecustomize``. Keep it standalone: sandbox proxy processes may not be able
to import the BenchFlow package tree.
"""

from __future__ import annotations

from typing import Any, cast


def _effective_retry_policy(
    *,
    retry_policy: Any,
    model_group: str | None,
    model_group_retry_policy: dict[str, Any] | None,
) -> Any:
    if (
        model_group_retry_policy is not None
        and model_group is not None
        and model_group in model_group_retry_policy
    ):
        return model_group_retry_policy.get(model_group)
    return retry_policy


def _internal_server_error_retries(policy: Any) -> int | None:
    if policy is None:
        return None
    if isinstance(policy, dict):
        value = policy.get("InternalServerErrorRetries")
    else:
        value = getattr(policy, "InternalServerErrorRetries", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _patch_internal_server_error_retry_policy() -> None:
    try:
        import litellm
        import litellm.router as router_mod
        from litellm.router_utils import get_retry_from_policy as policy_mod
    except Exception:
        return

    original = getattr(router_mod, "_get_num_retries_from_retry_policy", None)
    if original is None:
        original = getattr(policy_mod, "get_num_retries_from_retry_policy", None)
    if original is None or getattr(original, "__benchflow_retry_patch__", False):
        return

    internal_server_error = getattr(litellm, "InternalServerError", None)
    if internal_server_error is None:
        return

    def patched_get_num_retries_from_retry_policy(
        exception: Exception,
        retry_policy: Any = None,
        model_group: str | None = None,
        model_group_retry_policy: dict[str, Any] | None = None,
    ) -> int | None:
        retries = original(
            exception=exception,
            retry_policy=retry_policy,
            model_group=model_group,
            model_group_retry_policy=model_group_retry_policy,
        )
        if retries is not None:
            return retries
        if not isinstance(exception, internal_server_error):
            return None
        policy = _effective_retry_policy(
            retry_policy=retry_policy,
            model_group=model_group,
            model_group_retry_policy=model_group_retry_policy,
        )
        return _internal_server_error_retries(policy)

    patched_any = cast(Any, patched_get_num_retries_from_retry_policy)
    patched_any.__benchflow_retry_patch__ = True
    policy_module_any = cast(Any, policy_mod)
    router_module_any = cast(Any, router_mod)
    policy_module_any.get_num_retries_from_retry_policy = (
        patched_get_num_retries_from_retry_policy
    )
    router_module_any._get_num_retries_from_retry_policy = (
        patched_get_num_retries_from_retry_policy
    )


_patch_internal_server_error_retry_policy()
