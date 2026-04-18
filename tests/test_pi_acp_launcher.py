"""Tests for pi_acp_launcher.setup_provider — protocol-dependent Pi config."""

import json

import pytest


@pytest.fixture()
def _pi_env(monkeypatch, tmp_path):
    """Redirect Path.home() and clear BENCHFLOW_PROVIDER_* vars."""
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    for key in (
        "BENCHFLOW_PROVIDER_PROTOCOL",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_MODEL",
        "BENCHFLOW_PROVIDER_NAME",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("_pi_env")
class TestSetupProviderOpenAI:
    """OpenAI-completions path: generates ~/.pi/agent/models.json."""

    def test_writes_models_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "test-key")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_MODEL", "Qwen3.5-35B")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_NAME", "vllm")

        from benchflow.agents.pi_acp_launcher import setup_provider

        setup_provider()

        models_path = tmp_path / ".pi" / "agent" / "models.json"
        assert models_path.exists()
        config = json.loads(models_path.read_text())
        provider = config["providers"]["vllm"]
        assert provider["api"] == "openai-completions"
        assert provider["baseUrl"] == "http://localhost:8080/v1"
        assert provider["apiKey"] == "test-key"
        assert provider["models"][0]["id"] == "Qwen3.5-35B"

    def test_merges_with_existing_providers(self, monkeypatch, tmp_path):
        """Manually-added providers survive when a new one is registered."""
        config_dir = tmp_path / ".pi" / "agent"
        config_dir.mkdir(parents=True)
        existing = {
            "providers": {
                "other": {
                    "baseUrl": "http://other:9000/v1",
                    "api": "openai-completions",
                    "apiKey": "k",
                    "models": [{"id": "m1", "name": "m1"}],
                }
            }
        }
        (config_dir / "models.json").write_text(json.dumps(existing))

        monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_MODEL", "new-model")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_NAME", "vllm")

        from benchflow.agents.pi_acp_launcher import setup_provider

        setup_provider()

        config = json.loads((config_dir / "models.json").read_text())
        assert "other" in config["providers"], "pre-existing provider must survive"
        assert "vllm" in config["providers"], "new provider must be added"

    def test_overwrites_corrupt_models_json(self, monkeypatch, tmp_path, capsys):
        config_dir = tmp_path / ".pi" / "agent"
        config_dir.mkdir(parents=True)
        (config_dir / "models.json").write_text("{corrupt json")

        monkeypatch.setenv("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_MODEL", "m")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_NAME", "vllm")

        from benchflow.agents.pi_acp_launcher import setup_provider

        setup_provider()

        config = json.loads((config_dir / "models.json").read_text())
        assert "vllm" in config["providers"]
        assert "Warning" in capsys.readouterr().err


@pytest.mark.usefixtures("_pi_env")
class TestSetupProviderAnthropic:
    """Anthropic path: sets ANTHROPIC_* env vars."""

    def test_sets_anthropic_env_vars(self, monkeypatch):
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "sk-test")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_MODEL", "claude-haiku")
        # No BENCHFLOW_PROVIDER_PROTOCOL → defaults to Anthropic path

        import os

        from benchflow.agents.pi_acp_launcher import setup_provider

        setup_provider()

        assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.example.com"
        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
        assert os.environ["ANTHROPIC_MODEL"] == "claude-haiku"

    def test_setdefault_does_not_overwrite(self, monkeypatch):
        """Pre-existing ANTHROPIC_* values take precedence."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://keep-this.example.com")
        monkeypatch.setenv("BENCHFLOW_PROVIDER_BASE_URL", "https://new.example.com")

        import os

        from benchflow.agents.pi_acp_launcher import setup_provider

        setup_provider()

        assert os.environ["ANTHROPIC_BASE_URL"] == "https://keep-this.example.com"
