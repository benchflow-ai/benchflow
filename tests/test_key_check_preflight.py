"""``bench eval run`` asks the provider about the key before any sandbox.

Before, a rejected ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` was found by
the first model call of every trial, after its image was built, its sandbox
started and its agent installed. The run now lists the provider's models
with the model's key once, before a job exists: a 401 stops it naming the
variable (Rust 0.8.2's R212), a model the list lacks gets a did-you-mean
note, and anything else (a 403 included) checks nothing. No request goes out for a key the
user routes through their own base URL, or with ``BENCHFLOW_KEY_CHECK=0``.
"""

from __future__ import annotations

import httpx
import pytest
import typer

from benchflow.cli.doctor import eval_preflight

KEY_CHECK_ENV = "BENCHFLOW_KEY_CHECK"
FAKE_KEY = "sk-ant-api03-NOTAREALKEYjustatestvalue0123456789"


@pytest.fixture
def preflight_on(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.delenv(KEY_CHECK_ENV, raising=False)
    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)


def _answer(monkeypatch, status: int, models: list[str] | None = None) -> list:
    """Answer every HTTP GET the preflight makes; no request leaves the test."""
    calls: list = []

    def fake_get(url, *args, headers=None, **kwargs):
        calls.append((url, dict(headers or {})))
        body = {"data": [{"id": m} for m in models or []]}
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_a_rejected_key_stops_the_run_before_any_sandbox(
    preflight_on, monkeypatch, capsys
):
    calls = _answer(monkeypatch, 401)

    with pytest.raises(typer.Exit) as exc:
        eval_preflight(
            sandbox="modal", agent="claude-agent-acp", model="claude-sonnet-4-5"
        )

    assert exc.value.exit_code == 1
    assert calls and calls[0][0].startswith("https://api.anthropic.com/v1/models")
    err = capsys.readouterr().err
    assert "Anthropic refused the key in ANTHROPIC_API_KEY (HTTP 401)" in err
    assert "no job was created" in err
    assert "BENCHFLOW_KEY_CHECK=0" in err
    assert FAKE_KEY not in err


def test_an_accepted_key_runs_and_a_missing_model_gets_a_note(
    preflight_on, monkeypatch, capsys
):
    _answer(monkeypatch, 200, ["claude-sonnet-4-5", "claude-opus-4-1"])

    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="claude-sonet-4-5")

    err = capsys.readouterr().err
    assert "model claude-sonet-4-5 is not in Anthropic's model list" in err
    assert "did you mean claude-sonnet-4-5?" in err
    assert FAKE_KEY not in err


def test_a_listed_model_or_an_alias_gets_no_note(preflight_on, monkeypatch, capsys):
    _answer(monkeypatch, 200, ["claude-sonnet-4-5"])

    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="claude-sonnet-4-5")
    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="sonnet")

    assert "Note:" not in capsys.readouterr().err


@pytest.mark.parametrize("status", [429, 500, None])
def test_anything_else_checks_nothing(preflight_on, monkeypatch, capsys, status):
    if status is None:

        def broken(url, *args, **kwargs):
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(httpx, "get", broken)
    else:
        _answer(monkeypatch, status)

    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="claude-sonnet-4-5")

    assert "refused the key" not in capsys.readouterr().err


def test_no_request_for_the_users_own_endpoint_or_when_turned_off(
    preflight_on, monkeypatch
):
    calls = _answer(monkeypatch, 401)

    eval_preflight(
        sandbox="modal",
        agent="claude-agent-acp",
        model="claude-sonnet-4-5",
        agent_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:9/"},
    )
    monkeypatch.setenv(KEY_CHECK_ENV, "0")
    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="claude-sonnet-4-5")

    assert calls == []


def test_the_key_goes_only_in_the_header():
    from benchflow import key_check

    seen: list = []

    def fake_get(url, headers, timeout):
        seen.append((url, dict(headers)))
        return 200, [], ""

    key_check.check_key({"OPENAI_API_KEY": "sk-proj-x"}, "gpt-5", get=fake_get)
    key_check.check_key(
        {"ANTHROPIC_API_KEY": "sk-ant-y"}, "claude-sonnet-4-5", get=fake_get
    )
    assert seen[0] == (
        "https://api.openai.com/v1/models",
        {"Authorization": "Bearer sk-proj-x"},
    )
    assert "sk-ant-y" not in seen[1][0]
    assert seen[1][1]["x-api-key"] == "sk-ant-y"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_the_oracle_and_nop_ask_nothing(preflight_on, monkeypatch, agent):
    """A key in the host's environment is not the oracle's: a refused one must
    not stop a control run."""
    calls = _answer(monkeypatch, 401)

    eval_preflight(sandbox="modal", agent=agent, model=None)

    assert calls == []


def _answer_by_host(monkeypatch, statuses: dict[str, int]) -> list:
    """Answer each provider's models list with its own status."""
    calls: list = []

    def fake_get(url, *args, headers=None, **kwargs):
        calls.append(url)
        status = next(code for host, code in statuses.items() if host in url)
        return httpx.Response(
            status, json={"data": []}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_a_gpt_run_asks_openai_not_a_stale_anthropic_key(
    preflight_on, monkeypatch, capsys
):
    """Guards PT-R1-A-2: the agent's env inherits every key the host holds, so
    the key checked is the one the model uses, not the first one set. Before,
    a codex run on gpt-5 with a stale ``ANTHROPIC_API_KEY`` exported stopped
    with "Anthropic refused the key"."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-NOTAREALKEY0123456789")
    calls = _answer_by_host(
        monkeypatch, {"api.anthropic.com": 401, "api.openai.com": 200}
    )

    eval_preflight(sandbox="modal", agent="codex-acp", model="gpt-5")

    assert calls == ["https://api.openai.com/v1/models"]
    err = capsys.readouterr().err
    assert "refused the key" not in err
    assert "Anthropic" not in err


def test_a_claude_run_asks_anthropic_not_a_stale_openai_key(
    preflight_on, monkeypatch, capsys
):
    """The mirror of PT-R1-A-2: a Claude model checks only the Anthropic key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-NOTAREALKEY0123456789")
    calls = _answer_by_host(
        monkeypatch, {"api.anthropic.com": 200, "api.openai.com": 401}
    )

    eval_preflight(sandbox="modal", agent="claude-agent-acp", model="claude-sonnet-4-5")

    assert calls == ["https://api.anthropic.com/v1/models?limit=1000"]
    assert "refused the key" not in capsys.readouterr().err


def test_a_model_with_no_direct_key_asks_nothing(preflight_on, monkeypatch):
    """A Gemini model uses neither key: nothing is asked."""
    monkeypatch.setenv("GEMINI_API_KEY", "NOTAREALKEY0123456789")
    calls = _answer(monkeypatch, 401)

    eval_preflight(sandbox="modal", agent="gemini", model="gemini-3-pro")

    assert calls == []


def test_a_403_on_the_models_list_does_not_stop_the_run(
    preflight_on, monkeypatch, capsys
):
    """Guards PT-R1-A-4: an OpenAI restricted or project key without Models:
    Read gets 403 on the list and still calls models, so a 403 is no refusal."""
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-NOTAREALKEY0123456789")
    calls = _answer(monkeypatch, 403)

    eval_preflight(sandbox="modal", agent="codex-acp", model="gpt-5")

    assert calls and calls[0][0] == "https://api.openai.com/v1/models"
    assert "refused the key" not in capsys.readouterr().err


def _answer_401(monkeypatch, body: dict) -> list:
    calls: list = []

    def fake_get(url, *args, headers=None, **kwargs):
        calls.append(url)
        return httpx.Response(401, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_a_401_for_a_missing_scope_does_not_stop_the_run(
    preflight_on, monkeypatch, capsys
):
    """Guards PT-R2-B-2: OpenAI answers a restricted key without Models: Read
    with a 401 naming the missing scope, not a 403; that key may still call
    the model, so it checks nothing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-NOTAREALKEY0123456789")
    calls = _answer_401(
        monkeypatch,
        {
            "error": {
                "message": "You have insufficient permissions for this "
                "operation. Missing scopes: api.model.read. Check that you have "
                "the correct role in your organization (Reader, Writer, Owner) "
                "and project (Member, Owner), and if you're using a restricted "
                "API key, that it has the necessary scopes.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )

    eval_preflight(sandbox="modal", agent="codex-acp", model="gpt-5")

    assert calls == ["https://api.openai.com/v1/models"]
    assert "refused the key" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("agent", "model", "key_env", "body", "provider"),
    [
        (
            "codex-acp",
            "gpt-5",
            "OPENAI_API_KEY",
            {
                "error": {
                    "message": "Incorrect API key provided: sk-proj-****6789.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
            "OpenAI",
        ),
        (
            "claude-agent-acp",
            "claude-sonnet-4-5",
            "ANTHROPIC_API_KEY",
            {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key",
                },
            },
            "Anthropic",
        ),
    ],
)
def test_a_401_for_a_bad_key_still_stops_the_run(
    preflight_on, monkeypatch, capsys, agent, model, key_env, body, provider
):
    if key_env == "OPENAI_API_KEY":
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-NOTAREALKEY0123456789")
    _answer_401(monkeypatch, body)

    with pytest.raises(typer.Exit) as exc:
        eval_preflight(sandbox="modal", agent=agent, model=model)

    assert exc.value.exit_code == 1
    assert (
        f"{provider} refused the key in {key_env} (HTTP 401)" in capsys.readouterr().err
    )
