"""The sandbox-provider registry, under its historical import path.

The registry lives in :mod:`benchflow._utils.sandbox_providers` so the CLI can
build its ``--sandbox`` help without importing the ``benchflow.sandbox``
package (whose ``__init__`` loads the sandbox protocol and services): ``bench
--help`` stays fast and loads no sandbox code. Every name, private ones
included, is the same object as there.
"""

from benchflow._utils.sandbox_providers import (
    _PROVIDERS,
    _SANDBOX_SDK_MODULES,
    ALLOWLIST_SERVICE_CHECK_PROVIDERS,
    ALLOWLIST_UNSUPPORTED_PROVIDERS,
    DENYLIST_UNSUPPORTED_PROVIDERS,
    NO_NETWORK_UNSUPPORTED_PROVIDERS,
    OFF_BOX_MODEL_PROVIDERS,
    OPERATOR_SCRIPT_PROVIDERS,
    OPTIONAL_SANDBOX_EXTRAS,
    PROVIDERS_BY_NAME,
    SANDBOX_MODEL_PROXY_PROVIDERS,
    SANDBOX_PROVIDER_SET,
    SANDBOX_PROVIDERS,
    SEPARATE_VERIFIER_PROVIDERS,
    SINGLE_CONTAINER_PROVIDERS,
    VERIFIER_SERVICE_PROVIDERS,
    ModelProxyLocation,
    SandboxProvider,
    extra_install_hint,
    is_known_provider,
    provider_extra,
    providers_phrase,
    sandbox_sdk_missing,
)

__all__ = [
    "_PROVIDERS",
    "_SANDBOX_SDK_MODULES",
    "ALLOWLIST_SERVICE_CHECK_PROVIDERS",
    "ALLOWLIST_UNSUPPORTED_PROVIDERS",
    "DENYLIST_UNSUPPORTED_PROVIDERS",
    "NO_NETWORK_UNSUPPORTED_PROVIDERS",
    "OFF_BOX_MODEL_PROVIDERS",
    "OPERATOR_SCRIPT_PROVIDERS",
    "OPTIONAL_SANDBOX_EXTRAS",
    "PROVIDERS_BY_NAME",
    "SANDBOX_MODEL_PROXY_PROVIDERS",
    "SANDBOX_PROVIDER_SET",
    "SANDBOX_PROVIDERS",
    "SEPARATE_VERIFIER_PROVIDERS",
    "SINGLE_CONTAINER_PROVIDERS",
    "VERIFIER_SERVICE_PROVIDERS",
    "ModelProxyLocation",
    "SandboxProvider",
    "extra_install_hint",
    "is_known_provider",
    "provider_extra",
    "providers_phrase",
    "sandbox_sdk_missing",
]
