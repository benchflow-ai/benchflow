"""Guards the MiMo transient-400 retry: MiMo's "Invalid request parameters" refusal
of a valid request is retried twice in the proxy; every other error passes through at once.

Seen 2026-11-03: two such refusals in about 1,000 calls of Claude Code on MiMo v2.6 Pro, each ending the agent's turn (Claude
Code treats a 400 as final) and restarting the trial from scratch; replayed from the trajectory, each request was accepted.
"""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

import benchflow.providers.litellm_runtime as runtime_mod


class Reject(Exception):
    def __init__(self, status_code=400, message="Invalid request parameters"):
        super().__init__(
            f"Error code: {status_code} - {{'error': {{'message': '{message}'}}}}"
        )
        self.status_code = status_code
        self.body = {"code": str(status_code), "message": message, "param": ""}


def _client(host):
    return SimpleNamespace(base_url=SimpleNamespace(host=host))


def _wrapped(monkeypatch, outcomes):
    from litellm.llms.openai.openai import OpenAIChatCompletion

    calls = []

    async def fake(self, openai_aclient, data, timeout, logging_obj):
        calls.append(openai_aclient)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        OpenAIChatCompletion, "make_openai_chat_completion_request", fake
    )
    import benchflow.providers.litellm_provider_retry_patch as patch

    patch = importlib.reload(patch)
    monkeypatch.setattr(patch, "_DELAYS", (0.0, 0.0))
    method = OpenAIChatCompletion.make_openai_chat_completion_request
    return method, calls


def _call(method, host):
    return asyncio.run(
        method(None, openai_aclient=_client(host), data={}, timeout=1, logging_obj=None)
    )


def test_mimo_transient_reject_is_retried_then_succeeds(monkeypatch) -> None:
    method, calls = _wrapped(monkeypatch, [Reject(), Reject(), ({}, "ok")])
    assert _call(method, "token-plan-sgp.xiaomimimo.com") == ({}, "ok")
    assert len(calls) == 3


def test_mimo_reject_that_persists_fails_after_two_retries(monkeypatch) -> None:
    method, calls = _wrapped(monkeypatch, [Reject(), Reject(), Reject()])
    with pytest.raises(Reject):
        _call(method, "token-plan-sgp.xiaomimimo.com")
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("host", "error"),
    [
        ("api.openai.com", Reject()),  # another provider
        (
            "token-plan-sgp.xiaomimimo.com",
            Reject(message="max_tokens is too large"),
        ),  # another 400
        ("token-plan-sgp.xiaomimimo.com", Reject(status_code=401)),  # another status
        ("evilxiaomimimo.com", Reject()),  # not the provider's domain
    ],
)
def test_other_errors_pass_through_at_once(monkeypatch, host, error) -> None:
    method, calls = _wrapped(monkeypatch, [error, ({}, "never")])
    with pytest.raises(Reject):
        _call(method, host)
    assert len(calls) == 1


def test_runtime_files_load_provider_retry_patch(tmp_path) -> None:
    runtime_mod._write_runtime_files(tmp_path, config={"model_list": []})
    sitecustomize = (tmp_path / "sitecustomize.py").read_text()
    assert "import benchflow_litellm_provider_retry_patch" in sitecustomize
    assert (tmp_path / "benchflow_litellm_provider_retry_patch.py").is_file()
