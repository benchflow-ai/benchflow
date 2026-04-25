"""Tests for SDK private methods extracted from run().

Step 5 of the sdk-refactor plan: TDD decomposition of run() into
independently testable private methods.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

# ── _resolve_agent_env ──


class TestResolveAgentEnv:
    """Tests for SDK._resolve_agent_env — env var resolution logic."""

    def _resolve(self, agent="claude-agent-acp", model=None, agent_env=None):
        from benchflow._agent_env import resolve_agent_env

        return resolve_agent_env(agent, model, agent_env)

    def _patch_expanduser(self, monkeypatch, tmp_path):
        orig_expanduser = Path.expanduser

        def fake_expanduser(self):
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return orig_expanduser(self)

        monkeypatch.setattr(Path, "expanduser", fake_expanduser)

    def test_env_mapping_applied_after_provider(self, monkeypatch):
        """env_mapping translates BENCHFLOW_PROVIDER_* → agent-native vars."""
        for key in (
            "ZAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)
        result = self._resolve(
            agent="claude-agent-acp",
            model="zai/glm-5",
            agent_env={"ZAI_API_KEY": "zk-test"},
        )
        # claude-agent-acp maps BENCHFLOW_PROVIDER_BASE_URL → ANTHROPIC_BASE_URL
        assert "ANTHROPIC_BASE_URL" in result
        assert "ANTHROPIC_AUTH_TOKEN" in result
        assert result["ANTHROPIC_AUTH_TOKEN"] == "zk-test"

    def test_agent_native_api_key_satisfies_model_check(self, monkeypatch):
        """Agent-native mapped key (LLM_API_KEY) can satisfy provider auth check."""
        for key in ("OPENAI_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        result = self._resolve(
            agent="openhands",
            model="openai/gpt-4.1-mini",
            agent_env={"LLM_API_KEY": "test-llm-key"},
        )
        assert result["LLM_API_KEY"] == "test-llm-key"
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "test-llm-key"

    def test_same_provider_native_alias_satisfies_model_check(self, monkeypatch):
        """Provider-native aliases remain valid for the same auth context."""
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        result = self._resolve(
            agent="gemini",
            model="gemini-2.5-flash",
            agent_env={"GOOGLE_API_KEY": "test-google-key"},
        )
        assert result["GOOGLE_API_KEY"] == "test-google-key"
        assert result["BENCHFLOW_PROVIDER_API_KEY"] == "test-google-key"

    @pytest.mark.parametrize(
        ("agent", "host_key"),
        [
            pytest.param("codex-acp", "OPENAI_API_KEY", id="codex-openai-key"),
            pytest.param(
                "claude-agent-acp",
                "ANTHROPIC_AUTH_TOKEN",
                id="claude-auth-token",
            ),
            pytest.param("gemini", "GOOGLE_API_KEY", id="gemini-google-key"),
        ],
    )
    def test_cross_provider_host_native_key_does_not_bypass_required_key(
        self, monkeypatch, tmp_path, agent, host_key
    ):
        """Host-native keys for another provider must not satisfy zai auth."""
        for key in (
            "ZAI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(host_key, "host-native-key")
        self._patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="ZAI_API_KEY required"):
            self._resolve(agent=agent, model="zai/glm-5", agent_env={})

    def test_auto_inherited_generic_bridge_key_does_not_bypass_required_key(
        self, monkeypatch, tmp_path
    ):
        """Generic agent-native keys must be passed explicitly to bridge auth."""
        for key in ("ZAI_API_KEY", "LLM_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LLM_API_KEY", "host-llm-key")
        self._patch_expanduser(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="ZAI_API_KEY required"):
            self._resolve(agent="openhands", model="zai/glm-5", agent_env={})

    def test_openhands_gemini_model_is_prefixed_for_google_ai_studio(self, monkeypatch):
        """OpenHands expects Gemini models in gemini/<model> format."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = self._resolve(
            agent="openhands",
            model="gemini-3.1-flash-lite-preview",
            agent_env={"GEMINI_API_KEY": "test-gemini-key"},
        )
        assert result["LLM_MODEL"] == "gemini/gemini-3.1-flash-lite-preview"
        assert result["LLM_API_KEY"] == "test-gemini-key"

    def test_openhands_explicit_llm_model_is_preserved(self, monkeypatch):
        """User-provided LLM_MODEL must win over derived normalization."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = self._resolve(
            agent="openhands",
            model="gemini-3.1-flash-lite-preview",
            agent_env={
                "GEMINI_API_KEY": "test-gemini-key",
                "LLM_MODEL": "litellm/custom-format",
            },
        )
        assert result["LLM_MODEL"] == "litellm/custom-format"

    def test_openhands_vertex_model_is_prefixed_for_vertex(self, monkeypatch, tmp_path):
        """OpenHands expects Vertex Gemini models in vertex_ai/<model> format."""
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text("{}")
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        result = self._resolve(
            agent="openhands",
            model="google-vertex/gemini-2.5-flash",
            agent_env={"GOOGLE_CLOUD_PROJECT": "my-proj"},
        )
        assert result["LLM_MODEL"] == "vertex_ai/gemini-2.5-flash"

    def test_provider_bridge_key_alone_does_not_bypass_required_model_key(
        self, monkeypatch
    ):
        """Only mapped agent-native keys can bypass provider-specific key checks."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="openclaw",
                model="openai/gpt-4.1-mini",
                agent_env={"BENCHFLOW_PROVIDER_API_KEY": "x"},
            )

    def test_required_key_missing_raises(self, monkeypatch, tmp_path):
        """Missing required API key raises ValueError when no subscription auth."""
        # Clear any auto-inherited keys from the environment
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ZAI_API_KEY",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        # Ensure no host subscription auth files are found
        self._patch_expanduser(monkeypatch, tmp_path)
        # Anthropic model
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY required"):
            self._resolve(
                model="claude-haiku-4-5-20251001",
                agent_env={},
            )
        # Custom provider (zai)
        with pytest.raises(ValueError, match="ZAI_API_KEY required"):
            self._resolve(
                model="zai/glm-5",
                agent_env={},
            )
        # OpenAI model
        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            self._resolve(
                agent="codex-acp",
                model="gpt-4o",
                agent_env={},
            )

    def test_vertex_model_requires_adc(self, monkeypatch, tmp_path):
        """Vertex model without ADC raises ValueError."""
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        with pytest.raises(ValueError, match="requires ADC credentials"):
            self._resolve(
                model="google-vertex/gemini-3-flash",
                agent_env={"GOOGLE_CLOUD_PROJECT": "my-proj"},
            )

    def test_vertex_model_requires_project(self, monkeypatch, tmp_path):
        """Vertex model without GOOGLE_CLOUD_PROJECT raises ValueError."""
        adc_dir = tmp_path / ".config" / "gcloud"
        adc_dir.mkdir(parents=True)
        (adc_dir / "application_default_credentials.json").write_text("{}")
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT required"):
            self._resolve(
                model="google-vertex/gemini-3-flash",
                agent_env={},
            )


# ── _resolve_prompts ──


class TestResolvePrompts:
    """Tests for SDK._resolve_prompts — prompt list resolution from instruction.md."""

    def _resolve(self, task_path, prompts):
        from benchflow.sdk import SDK

        return SDK._resolve_prompts(task_path, prompts)

    def test_none_prompts_returns_instruction(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Do the thing.")
        result = self._resolve(tmp_path, prompts=None)
        assert result == ["Do the thing."]

    def test_mixed_list_replaces_nones(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Do the thing.")
        result = self._resolve(tmp_path, prompts=[None, "custom", None])
        assert result == ["Do the thing.", "custom", "Do the thing."]

    def test_all_explicit_preserves_prompts(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Do the thing.")
        result = self._resolve(tmp_path, prompts=["a", "b"])
        assert result == ["a", "b"]

    def test_missing_instruction_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._resolve(tmp_path, prompts=None)

    def test_whitespace_stripped(self, tmp_path):
        (tmp_path / "instruction.md").write_text("  hello  \n")
        result = self._resolve(tmp_path, prompts=None)
        assert result == ["hello"]


# ── _init_trial ──


class TestInitTrial:
    """Tests for SDK._init_trial — trial directory setup."""

    def _init(self, task_path, job_name=None, trial_name=None, jobs_dir="jobs"):
        from benchflow.sdk import SDK

        return SDK._init_trial(task_path, job_name, trial_name, jobs_dir)

    @pytest.fixture()
    def task_dir(self, tmp_path):
        """Minimal Harbor task directory."""
        td = tmp_path / "my-task"
        td.mkdir()
        (td / "task.toml").write_text(
            'version = "1.0"\n\n[verifier]\ntimeout_sec = 900.0\n\n'
            "[agent]\ntimeout_sec = 900.0\n\n[environment]\n"
        )
        (td / "instruction.md").write_text("Do the thing.")
        return td

    def test_trial_dir_created(self, task_dir, tmp_path):
        _, trial_dir, _, _, _, _ = self._init(task_dir, jobs_dir=tmp_path / "jobs")
        assert trial_dir.exists()
        for subdir in ("agent", "verifier", "artifacts", "trajectory"):
            assert (trial_dir / subdir).is_dir()

    def test_default_job_name_format(self, task_dir, tmp_path):
        _, _, _, _, job_name, _ = self._init(task_dir, jobs_dir=tmp_path / "jobs")
        # Default: date-based like 2026-04-08__12-30-45
        assert "__" in job_name
        assert job_name[:4].isdigit()

    def test_custom_job_name(self, task_dir, tmp_path):
        _, _, _, _, job_name, _ = self._init(
            task_dir,
            job_name="my-job",
            jobs_dir=tmp_path / "jobs",
        )
        assert job_name == "my-job"

    def test_trial_name_includes_task(self, task_dir, tmp_path):
        _, _, _, _, _, trial_name = self._init(task_dir, jobs_dir=tmp_path / "jobs")
        assert "my-task" in trial_name

    def test_custom_trial_name(self, task_dir, tmp_path):
        _, _, _, _, _, trial_name = self._init(
            task_dir,
            trial_name="custom-trial",
            jobs_dir=tmp_path / "jobs",
        )
        assert trial_name == "custom-trial"

    def test_started_at_is_datetime(self, task_dir, tmp_path):
        _, _, _, started_at, _, _ = self._init(task_dir, jobs_dir=tmp_path / "jobs")
        assert isinstance(started_at, datetime)


# ── _write_config ──


class TestWriteConfig:
    """Tests for SDK._write_config — writes config.json to trial_dir."""

    def _write(self, trial_dir, **kwargs):
        from benchflow.sdk import SDK

        return SDK._write_config(trial_dir, **kwargs)

    def test_config_json_written(self, tmp_path):
        self._write(
            tmp_path,
            task_path=Path("/tasks/foo"),
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="docker",
            skills_dir=None,
            sandbox_user=None,
            context_root=None,
            timeout=300,
            started_at=datetime(2026, 4, 8, 12, 0),
            agent_env={"ANTHROPIC_API_KEY": "sk-secret", "SOME_VAR": "visible"},
        )
        data = json.loads((tmp_path / "config.json").read_text())
        expected_keys = {
            "task_path",
            "agent",
            "model",
            "environment",
            "skills_dir",
            "sandbox_user",
            "sandbox_locked_paths",
            "context_root",
            "timeout_sec",
            "started_at",
            "agent_env",
        }
        assert expected_keys.issubset(data.keys()), (
            f"missing keys: {expected_keys - data.keys()}"
        )
        assert data["agent"] == "claude-agent-acp"
        assert data["model"] == "claude-haiku-4-5-20251001"
        assert data["environment"] == "docker"
        assert data["timeout_sec"] == 300

    def test_secrets_filtered(self, tmp_path):
        """Keys containing KEY/TOKEN/SECRET not in config.json agent_env."""
        self._write(
            tmp_path,
            task_path=Path("/tasks/foo"),
            agent="test",
            model=None,
            environment="docker",
            skills_dir=None,
            sandbox_user=None,
            context_root=None,
            timeout=300,
            started_at=datetime(2026, 4, 8),
            agent_env={
                "ANTHROPIC_API_KEY": "secret",
                "OPENAI_API_KEY": "secret",
                "MY_TOKEN": "secret",
                "DB_PASSWORD": "pass123",
                "MY_CREDENTIALS": "creds",
                "SAFE_VAR": "visible",
            },
        )
        data = json.loads((tmp_path / "config.json").read_text())
        recorded = data["agent_env"]
        assert "ANTHROPIC_API_KEY" not in recorded
        assert "OPENAI_API_KEY" not in recorded
        assert "MY_TOKEN" not in recorded
        assert "DB_PASSWORD" not in recorded
        assert "MY_CREDENTIALS" not in recorded
        assert recorded["SAFE_VAR"] == "visible"


# ── _build_result ──


class TestBuildResult:
    """Tests for SDK._build_result — builds RunResult and writes output files."""

    def _build(self, trial_dir, **kwargs):
        from benchflow.sdk import SDK

        defaults = dict(
            task_name="my-task",
            trial_name="my-trial",
            agent="claude-agent-acp",
            agent_name="Claude",
            model="claude-haiku-4-5-20251001",
            n_tool_calls=5,
            prompts=["solve it"],
            error=None,
            verifier_error=None,
            trajectory=[{"type": "message", "text": "hello"}],
            partial_trajectory=False,
            rewards={"score": 1.0},
            started_at=datetime(2026, 4, 8, 12, 0),
            timing={"agent_setup": 1.5, "agent_execution": 10.2},
        )
        defaults.update(kwargs)
        return SDK._build_result(trial_dir, **defaults)

    def test_result_json_written(self, tmp_path):
        self._build(tmp_path)
        assert (tmp_path / "result.json").exists()
        data = json.loads((tmp_path / "result.json").read_text())
        assert data["task_name"] == "my-task"
        assert data["rewards"] == {"score": 1.0}
        assert data["error"] is None
        assert data["agent"] == "claude-agent-acp"
        assert data["model"] == "claude-haiku-4-5-20251001"
        assert data["n_tool_calls"] == 5
        assert data["n_prompts"] == 1
        assert "started_at" in data
        assert "finished_at" in data
        assert data["partial_trajectory"] is False

    def test_timing_json_written(self, tmp_path):
        self._build(tmp_path)
        assert (tmp_path / "timing.json").exists()
        data = json.loads((tmp_path / "timing.json").read_text())
        assert "total" in data
        assert "agent_setup" in data
        for k, v in data.items():
            assert v >= 0, f"negative timing: {k}={v}"
            assert v == round(v, 1), f"not rounded: {k}={v}"

    def test_prompts_json_written(self, tmp_path):
        self._build(tmp_path)
        assert (tmp_path / "prompts.json").exists()
        data = json.loads((tmp_path / "prompts.json").read_text())
        assert data == ["solve it"]

    def test_trajectory_saved(self, tmp_path):
        traj_dir = tmp_path / "trajectory"
        traj_dir.mkdir()
        self._build(tmp_path)
        traj_file = traj_dir / "acp_trajectory.jsonl"
        assert traj_file.exists()

    def test_timing_values_rounded(self, tmp_path):
        self._build(tmp_path, timing={"agent_setup": 1.5678})
        data = json.loads((tmp_path / "timing.json").read_text())
        assert data["agent_setup"] == 1.6

    def test_error_in_result(self, tmp_path):
        result = self._build(tmp_path, error="timeout")
        assert result.error == "timeout"
        assert not result.success
