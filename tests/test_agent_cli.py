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
