"""Retry MiMo's transient "Invalid request parameters" rejections inside the proxy, twice, before the agent sees them.

MiMo's token-plan endpoint (``*.xiaomimimo.com``) now and then answers a valid chat request with HTTP 400
``{"code": "400", "message": "Invalid request parameters", "param": ""}`` and accepts the same request moments later (seen
2026-11-03: two such refusals in about 1,000 calls of Claude Code on MiMo v2.6 Pro; each failing request, replayed from the
trajectory with its tools, effort and streaming, was accepted). The proxy does not retry provider rejections
(``num_retries: 0``, so a deterministic reject never turns into a retry storm), and Claude Code treats a 400 as final, so one
such moment ended the agent's turn and BenchFlow restarted the trial from scratch.

The proxy imports this module from ``sitecustomize``. It wraps LiteLLM's ``OpenAIChatCompletion.make_openai_chat_completion_request``
and, only for a 400 whose message is "Invalid request parameters" from a ``xiaomimimo.com`` endpoint, sends the same request again
after 3 s and then 10 s. Any other error, provider or status passes through at once, and a request that is really invalid fails
13 s later than before.
"""

from __future__ import annotations

import asyncio
from typing import Any

_DELAYS = (3.0, 10.0)
_HOST_SUFFIX = "xiaomimimo.com"
_MESSAGE = "Invalid request parameters"


def _host(client: Any) -> str:
    try:
        return str(getattr(getattr(client, "base_url", None), "host", "") or "").lower()
    except Exception:
        return ""


def _transient_reject(error: BaseException, client: Any) -> bool:
    if getattr(error, "status_code", None) != 400:
        return False
    host = _host(client)
    if not (host == _HOST_SUFFIX or host.endswith("." + _HOST_SUFFIX)):
        return False
    body = getattr(error, "body", None)
    message = body.get("message") if isinstance(body, dict) else None
    return (message or str(error)).strip().endswith(_MESSAGE) or _MESSAGE in str(error)


def _apply_patch() -> None:
    try:
        from litellm.llms.openai.openai import OpenAIChatCompletion
    except Exception:
        return

    original = OpenAIChatCompletion.make_openai_chat_completion_request
    if getattr(original, "__benchflow_provider_retry_patch__", False):
        return

    async def make_openai_chat_completion_request(
        self, openai_aclient, data, timeout, logging_obj
    ):
        for attempt in range(len(_DELAYS) + 1):
            try:
                return await original(
                    self,
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
            except Exception as error:
                if attempt < len(_DELAYS) and _transient_reject(error, openai_aclient):
                    await asyncio.sleep(_DELAYS[attempt])
                    continue
                raise

    setattr(  # noqa: B010 - marker on the monkey-patched vendor function
        make_openai_chat_completion_request,
        "__benchflow_provider_retry_patch__",
        True,
    )
    setattr(  # noqa: B010 - avoids static narrowing on the vendor API
        OpenAIChatCompletion,
        "make_openai_chat_completion_request",
        make_openai_chat_completion_request,
    )


_apply_patch()
