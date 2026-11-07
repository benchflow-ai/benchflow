"""Official endpoint forms for restricted admission (FF-ACP-CONF-3).

One finite table supplies both admission and the URL test matrix. Forms are
exact, not host suffixes or path patterns; only terminal slashes normalize.
``{model}`` expands to the identified base model, never a URL component from
the supplied endpoint. Cloud SDK identities deliberately admit no custom URL.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointForms:
    base: str
    forms: tuple[str, ...]
    chat_url: str


# The first row is the default and the safe base suggested on refusal.
# OpenAI: https://developers.openai.com/api/reference/overview
# https://developers.openai.com/api/docs/guides/your-data
# https://help.openai.com/en/articles/10876024-openai-mutual-tls-beta-program
ENDPOINT_TABLE = {
    "openai": tuple(
        EndpointForms(
            f"https://{host}/v1",
            tuple(
                f"https://{host}{path}"
                for path in ("/v1", "", "/v1/chat/completions", "/v1/responses")
            ),
            f"https://{host}/v1/chat/completions",
        )
        for host in (
            "api.openai.com",
            "us.api.openai.com",
            "eu.api.openai.com",
            "au.api.openai.com",
            "ca.api.openai.com",
            "jp.api.openai.com",
            "in.api.openai.com",
            "sg.api.openai.com",
            "kr.api.openai.com",
            "gb.api.openai.com",
            "ae.api.openai.com",
            "mtls.api.openai.com",
            "mtls-eu.api.openai.com",
        )
    ),
    # https://platform.claude.com/docs/en/api/overview (beta uses headers).
    "anthropic": (
        EndpointForms(
            "https://api.anthropic.com",
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1",
                "https://api.anthropic.com/v1/messages",
            ),
            "https://api.anthropic.com/v1/messages",
        ),
    ),
    # https://ai.google.dev/gemini-api/docs/api-versions
    # https://ai.google.dev/api/generate-content
    "gemini": tuple(
        EndpointForms(
            f"https://generativelanguage.googleapis.com/{version}",
            (
                f"https://generativelanguage.googleapis.com/{version}",
                *(
                    ("https://generativelanguage.googleapis.com",)
                    if version == "v1beta"
                    else ()
                ),
                f"https://generativelanguage.googleapis.com/{version}/models/{{model}}:generateContent",
                f"https://generativelanguage.googleapis.com/{version}/models/{{model}}:streamGenerateContent",
            ),
            f"https://generativelanguage.googleapis.com/{version}/models/{{model}}:generateContent",
        )
        for version in ("v1beta", "v1")
    ),
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
    "vertex_ai": (EndpointForms("vertex-sdk-default", ("vertex-sdk-default",), ""),),
    # https://docs.aws.amazon.com/general/latest/gr/bedrock.html
    "bedrock": (EndpointForms("bedrock-sdk-default", ("bedrock-sdk-default",), ""),),
    # https://api-docs.deepseek.com/ and /api/create-chat-completion/
    # https://api-docs.deepseek.com/guides/chat_prefix_completion/
    "deepseek": (
        EndpointForms(
            "https://api.deepseek.com/v1",
            (
                "https://api.deepseek.com/v1",
                "https://api.deepseek.com",
                "https://api.deepseek.com/v1/chat/completions",
            ),
            "https://api.deepseek.com/v1/chat/completions",
        ),
        EndpointForms(
            "https://api.deepseek.com/beta",
            (
                "https://api.deepseek.com/beta",
                "https://api.deepseek.com/beta/chat/completions",
            ),
            "https://api.deepseek.com/beta/chat/completions",
        ),
        # Pinned LiteLLM accepts a full URL. Keep this one as-is to preserve
        # the explicitly selected unversioned /chat path; the existing origin
        # shorthand above continues to supply /v1.
        EndpointForms(
            "https://api.deepseek.com/chat/completions",
            ("https://api.deepseek.com/chat/completions",),
            "https://api.deepseek.com/chat/completions",
        ),
    ),
    # https://docs.z.ai/api-reference/introduction
    # https://docs.z.ai/devpack/quick-start (Chat, not its other wire APIs).
    "zai": (
        EndpointForms(
            "https://api.z.ai/api/paas/v4",
            (
                "https://api.z.ai/api/paas/v4",
                "https://api.z.ai",
                "https://api.z.ai/api/paas/v4/chat/completions",
            ),
            "https://api.z.ai/api/paas/v4/chat/completions",
        ),
        EndpointForms(
            "https://api.z.ai/api/coding/paas/v4",
            (
                "https://api.z.ai/api/coding/paas/v4",
                "https://api.z.ai/api/coding/paas/v4/chat/completions",
            ),
            "https://api.z.ai/api/coding/paas/v4/chat/completions",
        ),
    ),
}

OFFICIAL_ENDPOINTS = {
    provider: {form: row.base for row in rows for form in row.forms}
    for provider, rows in ENDPOINT_TABLE.items()
}


def endpoint_forms(provider: str, model: str) -> dict[str, str]:
    """Expand only the selected model's finite inventory, not a URL grammar."""
    return {
        form.format(model=model): base
        for form, base in OFFICIAL_ENDPOINTS.get(provider, {}).items()
    }


def endpoint_hint(provider: str, supplied: str | None) -> str:
    rows = ENDPOINT_TABLE.get(provider, ())
    if not rows:
        return ""
    if not rows[0].base.startswith("https://"):
        return f" Use the {provider} SDK default; remove custom endpoint settings."
    bases = [row.base for row in rows if row.base != row.chat_url]
    # Only a diagnostic selection: this never grants admission. Prefer the
    # selected official region/version, without echoing a private URL.
    matching = [b for b in bases if supplied and supplied.startswith(b + "/")]
    base = max(matching, key=len) if matching else bases[0]
    return f" Use the supported {provider} base URL {base}."
