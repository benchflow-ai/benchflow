"""Guards native-auth network preflight at PY8 base 363d815b."""

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
@pytest.mark.parametrize(
    "agent,model,key",
    [
        ("codex", "gpt-5.5", "CODEX_ACCESS_TOKEN"),
        ("muse", "muse-default", "MUSE_AUTH_JSON"),
    ],
)
@pytest.mark.parametrize("policy", ["allowlist", "offline"])
def test_native_login_network_refusals_reach_check(
    tmp_path, monkeypatch, command, agent, model, key, policy
):
    """Guards 363d815b: auth-dependent run refusals precede allocation."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, "dummy-native-login-0123456789")
    config = (
        '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n'
        if policy == "allowlist"
        else "[sandbox]\nallow_internet=false\n"
    )
    task = _task(tmp_path / "task", config)
    result = CliRunner().invoke(
        app, [*command, str(task), "--agent", agent, "--model", model]
    )
    if agent == "codex" and policy == "offline":
        # #38: Codex on a ChatGPT login runs a no-web task through its
        # model-only gate (chatgpt.com Responses requests only).
        assert result.exit_code == 0, result.output
        assert "native subscription" not in result.stderr.lower()
        return
    assert result.exit_code == 1, result.output
    assert "native subscription" in result.stderr.lower()


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("key", ["CODEX_ACCESS_TOKEN", "CODEX_AUTH_JSON"])
def test_explicit_native_env_is_used_by_the_shared_gate(tmp_path, backend, key):
    """Guards the 363d815b SDK route as well as inherited CLI auth."""
    task = _task(
        tmp_path / "task",
        '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n',
    )
    error = RuntimeError if backend == "modal" else ValueError
    reason = (
        "not enforced by modal"
        if backend == "modal"
        else "native subscription endpoint"
    )
    with pytest.raises(error, match=reason):
        validate_task_plan(
            task,
            sandbox=backend,
            agent="codex-acp",
            model="gpt-5.5",
            agent_env={key: "dummy-login"},
        )


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "CODEX_API_KEY"])
@pytest.mark.parametrize("source", ["host", "dotenv"])
def test_api_key_route_wins_over_a_native_login(tmp_path, monkeypatch, key, source):
    """Guards 363d815b's fix: runtime API aliases retain their allowed route."""
    dotenv = tmp_path / "dummy.env"
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "dummy-native-login")
    if source == "dotenv":
        dotenv.write_text(f"{key}=dummy-api-key\n")
    else:
        monkeypatch.setenv(key, "dummy-api-key")
    task = _task(
        tmp_path / "task",
        '[sandbox]\nnetwork_mode="allowlist"\nallowed_hosts=["example.invalid"]\n',
    )
    result = CliRunner().invoke(
        app, ["check", str(task), "--agent", "codex", "--model", "gpt-5.5"]
    )
    assert result.exit_code == 0, result.output
