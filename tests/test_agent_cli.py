import click
from typer.testing import CliRunner

from benchflow.cli.main import app


def test_agent_list_mentions_provider_specific_azure_auth() -> None:
    """Guards PR #422: agent discovery must not hide Azure provider auth."""
    result = CliRunner().invoke(app, ["agent", "list"])

    assert result.exit_code == 0
    output = click.unstyle(result.output)
    assert "OPENAI_API_KEY" in output
    assert "(or login)" in output
    assert "AZURE_API_KEY" in output
    assert "AZURE_API_ENDPOINT" in output


def test_agent_show_mentions_provider_specific_azure_auth() -> None:
    """Guards PR #422: per-agent details explain provider-prefixed auth."""
    result = CliRunner().invoke(app, ["agent", "show", "codex"])

    assert result.exit_code == 0
    output = click.unstyle(result.output)
    assert "Requires:" in output
    assert "OPENAI_API_KEY (or login)" in output
    assert "Provider auth:" in output
    assert "AZURE_API_KEY" in output
    assert "AZURE_API_ENDPOINT" in output


def test_agent_show_names_the_own_model_route(monkeypatch) -> None:
    """An agent with model_route = "agent" says it calls its own
    endpoint, instead of the provider-auth note that does not apply to it."""
    from benchflow.agents.registry import AGENTS, AgentConfig

    monkeypatch.setitem(
        AGENTS,
        "en191-own-route",
        AgentConfig(
            name="en191-own-route",
            install_cmd="true",
            launch_cmd="en191-own-route --acp",
            model_route="agent",
        ),
    )
    result = CliRunner().invoke(app, ["agent", "show", "en191-own-route"])

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output)
    assert "Model route: agent (calls its own endpoint" in output
    assert "Provider auth:" not in output
