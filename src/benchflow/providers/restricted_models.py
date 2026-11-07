"""Which models a web-restricted run may use.

Three review rounds found new spellings that hid a self-searching model from
the hosted-retrieval catalog (routing variants, presets, router aliases). A
detector cannot be complete, so with web access restricted the configured
model is admitted only when it is positively identified, at launch, before
any proxy starts:

- the provider is split against pinned LiteLLM's provider registry (never
  its get_llm_provider, whose provider hooks can log in over the network),
  then the
  provider's own id grammar finds the base model (OpenAI fine-tunes by their
  base, Bedrock ARNs and inference profiles, Vertex and Gemini resource ids);
- the effective provider, official endpoint and base model must belong to
  the curated first-party allowlist generated from pinned LiteLLM metadata;
- and the hosted-retrieval catalog (litellm_request_guard) must not match,
  as a second check, which no operator entry overrides.

LiteLLM's ``supports_web_search`` flag marks support for the web search
*tool* (gpt-4.1, gpt-5, Claude, Gemini all set it), which the gateway's tool
allowlist already refuses; it does not mean the model searches by itself.

Anything else (gateways, presets, routers, opaque Azure deployments, unknown variants,
operator-run servers, providers LiteLLM does not know) is refused with the
operator entry that admits it: ``BENCHFLOW_RESTRICTED_WEB_MODELS``, a
comma-separated list of configured model ids, or ``<provider>/*`` for an
operator-run provider. The admission, and any override, is recorded in the
trial's config.json (``restricted_web_model``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.providers.restricted_endpoints import (
    OFFICIAL_ENDPOINTS,
    endpoint_forms,
    endpoint_hint,
)

if TYPE_CHECKING:
    from benchflow.providers.litellm_config import LiteLLMRoute

OPERATOR_ENV = "BENCHFLOW_RESTRICTED_WEB_MODELS"
_AZURE = frozenset({"azure", "azure_ai", "azure_text"})
_BEDROCK = frozenset({"bedrock", "bedrock_converse"})
_VERTEX = frozenset({"vertex_ai", "vertex_ai_beta"})
_OPENAI = frozenset({"openai", "text-completion-openai"})
# ARN resource kinds that name one base model; the others (application
# inference profiles, imported, custom and provisioned models, prompt
# routers) are opaque handles chosen in the account.
_BEDROCK_MODEL_ARNS = frozenset({"foundation-model", "inference-profile"})


class RestrictedModelRefused(RuntimeError):
    """The configured model cannot serve a web-restricted run."""


@dataclass(frozen=True)
class RestrictedAdmission:
    model: str
    provider: str
    base: str | None
    metadata_key: str | None
    admitted_by: str  # "metadata" or "operator"
    endpoint: str | None = None

    def record(self) -> dict[str, str | None]:
        return asdict(self)


class _Unidentified(Exception):
    pass


def operator_entries(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    raw = (os.environ if environ is None else environ).get(OPERATOR_ENV, "")
    return frozenset(e.strip() for e in re.split(r"[,\s]+", raw) if e.strip())


def identify(model: str, provider_name: str = "") -> tuple[str, str, list[str]]:
    """(provider, base model, metadata keys to look up) or _Unidentified."""
    from benchflow.providers.litellm_request_guard import split_provider

    found, rest = split_provider(model)
    if found is None:
        # A bare id: its pinned metadata entry names the provider.
        info = pinned_model_metadata().get(model)
        found = info.get("litellm_provider") if isinstance(info, dict) else None
    if not found:
        raise _Unidentified("pinned LiteLLM does not recognise its provider")
    # Metadata names Vertex's model families vertex_ai-<family>.
    provider = "vertex_ai" if str(found).startswith("vertex_ai") else str(found)
    if "@preset/" in rest or rest.startswith("@"):
        raise _Unidentified(
            "an OpenRouter preset chooses its model and tools outside the request"
        )
    if provider in _AZURE or model.split("/", 1)[0] in _AZURE:
        raise _Unidentified("an Azure deployment name does not say which model it runs")
    if provider == "openrouter":
        if rest.split("/", 1)[0] == "openrouter":
            raise _Unidentified(
                "OpenRouter's own routers (auto, fusion and the like) choose models "
                "and tools per request"
            )
        base = _routing_base(rest)
        return provider, base, [f"openrouter/{base}"]
    if provider in _BEDROCK:
        from litellm.llms.bedrock.common_utils import (
            BedrockModelInfo,
            strip_bedrock_routing_prefix,
        )

        resource = strip_bedrock_routing_prefix(rest)
        arn = re.fullmatch(
            r"arn:aws[\w-]*:bedrock:[\w-]*:\d*:(?P<kind>[\w-]+)/(?P<id>[^/]+)", resource
        )
        if arn:
            if arn["kind"] not in _BEDROCK_MODEL_ARNS:
                raise _Unidentified(
                    f"a Bedrock {arn['kind']} ARN does not say which model it runs"
                )
            resource = arn["id"]
        elif "/" in resource or resource.startswith("arn:"):
            raise _Unidentified("not a Bedrock model id or model ARN")
        base = BedrockModelInfo.get_base_model(resource)
        return provider, base, [base, f"bedrock/{base}", resource]
    if provider in _VERTEX:
        match = re.fullmatch(
            r"(?:projects/[^/]+/locations/[^/]+/)?(?:publishers/[^/]+/models/)?"
            r"([A-Za-z][\w.@-]*)",
            rest,
        )
        if not match:
            raise _Unidentified("a Vertex endpoint id does not say which model it runs")
        base = match[1]
        publisher = re.search(r"publishers/([^/]+)/models/", rest)
        if publisher and publisher[1] != (
            "anthropic" if base.startswith("claude-") else "google"
        ):
            raise _Unidentified("the Vertex publisher does not own this model family")
        return provider, base, [f"vertex_ai/{base}", base]
    if provider == "gemini":
        base = rest.removeprefix("models/")
        return provider, base, [f"gemini/{base}"]
    if provider in _OPENAI:
        # ft:<base>:<org>:<name>:<id>; the full id is dispatched unchanged.
        finetune = re.fullmatch(r"ft:([^:]+)(?::[^:]*)+", rest)
        base = finetune[1] if finetune else _routing_base(rest)
        keys = [base, f"openai/{base}"]
    else:
        base = _routing_base(rest)
        keys = [f"{provider}/{base}", base]
    if provider_name:
        keys.append(f"{provider_name}/{base}")
    return provider, base, keys


@cache
def pinned_model_metadata() -> dict[str, Any]:
    """The model map shipped inside the pinned LiteLLM wheel.

    Not litellm.model_cost: unless LITELLM_LOCAL_MODEL_COST_MAP is set, that
    is fetched from the network at import and changes without a pin.
    """
    import litellm

    path = Path(litellm.__file__).with_name(
        "model_prices_and_context_window_backup.json"
    )
    return json.loads(path.read_text())


def _routing_base(rest: str) -> str:
    """The model id without known routing-only ":variants" (PY8-ACP-C26).

    Only variants MODEL_VARIANTS lists as changing routing or pricing alone
    are separated; an unknown or web-enabling variant stays on the id, so it
    finds no metadata and still needs the operator entry. The full id is
    what the gateway dispatches.
    """
    from benchflow.providers.litellm_request_guard import MODEL_VARIANTS

    head, *variants = rest.split(":")
    if variants and all(
        v in MODEL_VARIANTS and MODEL_VARIANTS[v] is None for v in variants
    ):
        return head
    return rest


# Provider aliases are explicit: speaking OpenAI's wire does not make a
# LiteLLM gateway, OpenRouter router or private deployment an OpenAI model.
_PROVIDER_ALIASES = {
    "us-openai": "openai",
    "aws-bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "google-vertex": "vertex_ai",
    "anthropic-vertex": "vertex_ai",
    "vertex_ai_beta": "vertex_ai",
    "zai-coding": "zai",
    "text-completion-openai": "openai",
}
_WIRE_PROVIDERS = {
    "openai": _OPENAI,
    "anthropic": frozenset({"anthropic"}),
    "gemini": frozenset({"gemini"}),
    "vertex_ai": _VERTEX,
    "bedrock": _BEDROCK,
    "deepseek": frozenset({"deepseek"}),
    "zai": frozenset({"openai", "zai"}),
}
# LiteLLM's endpoint fallbacks, when a deployment has no explicit api_base.
# These are read at admission and sealed in the process environment below;
# neither inherited host values nor sandbox image values can choose another
# endpoint after a first-party admission.
ENDPOINT_ENV = {
    "openai": ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"),
    "gemini": ("GEMINI_API_BASE",),
    "vertex_ai": ("VERTEXAI_API_BASE", "VERTEX_API_BASE"),
    "bedrock": ("AWS_BEDROCK_RUNTIME_ENDPOINT",),
}


@cache
def first_party_models() -> dict[tuple[str, str, str], str]:
    """(effective provider, official endpoint, base model) -> metadata key.

    This frozen snapshot is generated by tools/restricted_model_allowlist.py.
    Runtime metadata lookup cannot add a gateway alias to the admission set.
    """
    data = json.loads(
        Path(__file__).with_name("restricted_model_allowlist.json").read_text()
    )
    return {
        (provider, endpoint, base): key
        for provider, base, key in data["entries"]
        for endpoint in set(OFFICIAL_ENDPOINTS[provider].values())
    }


def _catalogued(*models: str | None) -> str | None:
    from benchflow.providers.litellm_request_guard import hosted_retrieval_model

    for model in models:
        why = hosted_retrieval_model(model) if model else None
        if why:
            return why
    return None


def admit_restricted_model(
    model: str,
    *,
    provider_name: str = "",
    api_base: str | None = None,
    environ: Mapping[str, str] | None = None,
    requested: str | None = None,
) -> RestrictedAdmission:
    """Admit ``model`` (the controller-resolved upstream) or refuse it.

    ``requested`` is the id the run was given (``xiaomi/mimo-v2.6-pro``, sent
    upstream as ``openai/mimo-v2.6-pro``): an operator entry may name either,
    and a refusal names the one the user typed."""
    try:
        parsed, base, _ = identify(model)
        effective = provider_name if provider_name not in {"", "native"} else parsed
        provider = _PROVIDER_ALIASES.get(effective, effective)
        endpoints = endpoint_forms(provider, base)
        endpoint = (
            endpoints.get(api_base.rstrip("/"))
            if api_base
            else next(iter(endpoints.values()), None)
        )
        if api_base and not api_base.startswith("https://"):
            endpoint = None
        key = first_party_models().get((provider, endpoint, base)) if endpoint else None
        if parsed not in _WIRE_PROVIDERS.get(provider, ()):
            key = None
        reason = (
            None
            if key
            else (
                "its effective provider, endpoint and base model are not in the "
                "curated first-party model allowlist"
            )
        )
    except _Unidentified as exc:
        provider, base, key, reason = (
            provider_name or model.split("/", 1)[0],
            None,
            None,
            str(exc),
        )
        endpoint = api_base
    why = _catalogued(model, base)
    if why:
        # Catalogued self-searching models: no operator entry admits them.
        raise RestrictedModelRefused(
            f"Refused {model!r} with web access restricted: {why}. "
            "next: choose a model without hosted web search."
        )
    if reason is None:
        return RestrictedAdmission(model, provider, base, key, "metadata", endpoint)
    entries = operator_entries(environ)
    if model in entries or requested in entries or f"{provider}/*" in entries:
        # An unclassified URL can contain credentials; the admission records
        # the provider and override, never an operator's private endpoint.
        return RestrictedAdmission(model, provider, base, key, "operator")
    shown = requested or model
    sent = f" (sent to the provider as {model!r})" if shown != model else ""
    raise RestrictedModelRefused(
        f"Refused {shown!r}{sent} with web access restricted: BenchFlow can't confirm it "
        f"has no hosted web search or retrieval ({reason}).{endpoint_hint(provider, api_base)} "
        f"next: if it runs without web access, set {OPERATOR_ENV}={shown} in the "
        f"controller's environment (comma-separated ids; '{provider}/*' allows an "
        "operator-run provider). The run records the override in config.json "
        "(restricted_web_model)."
    )


def admit_restricted_route(
    route: LiteLLMRoute, agent_env: Mapping[str, str]
) -> tuple[RestrictedAdmission, LiteLLMRoute, dict[str, str]]:
    """Admit and seal the actual deployment, for both proxy placements.

    An operator admission keeps the operator's route. Metadata admission
    pins static API bases and clears cloud endpoint fallbacks explicitly in
    the child environment. Unknown providers cannot borrow wire metadata.
    """
    from dataclasses import replace

    parsed = route.upstream_model.split("/", 1)[0]
    effective = route.provider_name if route.provider_name != "native" else parsed
    provider = _PROVIDER_ALIASES.get(effective, effective)
    process_env = {**os.environ, **agent_env}
    if provider in {"bedrock", "vertex_ai"}:
        # These values become URL path/host components in SDK-default routes.
        # A project/location/region is a component, never an endpoint alias.
        components = (
            {"AWS_REGION": r"[a-z0-9-]+", "AWS_DEFAULT_REGION": r"[a-z0-9-]+"}
            if provider == "bedrock"
            else {
                "GOOGLE_CLOUD_PROJECT": r"[A-Za-z0-9_.:-]+",
                "GOOGLE_CLOUD_LOCATION": r"[a-z0-9-]+",
            }
        )
        bad = [
            n
            for n, pattern in components.items()
            if process_env.get(n) and not re.fullmatch(pattern, process_env[n])
        ]
        if bad:
            raise RestrictedModelRefused(
                f"Refused the {provider} route with web access restricted: "
                f"{' and '.join(bad)} must be a plain name, as it becomes part of "
                "the URL. next: set it to the region, project or location name "
                "itself, not a deployment alias."
            )
    api_base = route.litellm_params.get("api_base")
    if not api_base:
        api_base = next(
            (
                process_env[n]
                for n in ENDPOINT_ENV.get(provider, ())
                if process_env.get(n)
            ),
            None,
        )
    admission = admit_restricted_model(
        route.upstream_model,
        provider_name=route.provider_name,
        api_base=str(api_base) if api_base else None,
        requested=route.requested_model,
    )
    sealed_env: dict[str, str] = {}
    if admission.admitted_by == "metadata":
        params = dict(route.litellm_params)
        if admission.endpoint and admission.endpoint.startswith("https://"):
            params["api_base"] = admission.endpoint
        else:
            sealed_env = dict.fromkeys(ENDPOINT_ENV.get(provider, ()), "")
        route = replace(route, litellm_params=params)
    return admission, route, sealed_env
