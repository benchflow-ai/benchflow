"""bench init / bench doctor onboarding logic (benchflow.onboarding).

Behavior under test, not implementation: credentials land in a private env
file that later runs can source; preferences round-trip; provider/agent
selection follows the wire-protocol rules the registry already declares; the
smoke ping tells the truth about a working vs broken key; the wizard's output
is a runnable `bench eval run` command.
"""

from __future__ import annotations

from benchflow import onboarding


class TestEnvFile:
    def test_write_creates_private_file_and_read_back(self, tmp_path):
        path = tmp_path / ".benchflow" / ".env"
        onboarding.write_env_file(path, {"DEEPSEEK_API_KEY": "sk-test"})
        assert path.read_text() == 'DEEPSEEK_API_KEY="sk-test"\n'
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_write_merges_and_overwrites_only_given_keys(self, tmp_path):
        path = tmp_path / ".env"
        onboarding.write_env_file(path, {"A": "1", "B": "2"})
        onboarding.write_env_file(path, {"B": "changed"})
        content = onboarding.read_env_file(path)
        assert content == {"A": "1", "B": "changed"}

    def test_load_into_environ_never_overrides_real_env(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        onboarding.write_env_file(
            path, {"BF_PROBE_FROM_FILE": "file", "BF_PROBE_SET": "file"}
        )
        monkeypatch.delenv("BF_PROBE_FROM_FILE", raising=False)
        monkeypatch.setenv("BF_PROBE_SET", "real-env")
        loaded = onboarding.load_env_file(path)
        import os

        assert os.environ["BF_PROBE_FROM_FILE"] == "file"
        assert os.environ["BF_PROBE_SET"] == "real-env"  # file must not clobber
        assert loaded == ["BF_PROBE_FROM_FILE"]
        monkeypatch.delenv("BF_PROBE_FROM_FILE")


class TestPrefs:
    def test_round_trip_and_missing_file_is_empty(self, tmp_path):
        path = tmp_path / "config.toml"
        assert onboarding.load_prefs(path) == {}
        prefs = {
            "agent": "pi-acp",
            "model": "deepseek/deepseek-v4-flash",
            "dataset": "skillsbench",
            "sandbox": "docker",
        }
        onboarding.save_prefs(path, prefs)
        assert onboarding.load_prefs(path) == prefs


class TestProviderResolution:
    def test_prefixed_model_resolves_to_its_provider(self):
        name, cfg = onboarding.resolve_provider("deepseek/deepseek-v4-flash")
        assert name == "deepseek"
        assert cfg.auth_env == "DEEPSEEK_API_KEY"

    def test_bare_model_resolves_via_model_prefixes(self):
        name, _cfg = onboarding.resolve_provider("deepseek-v4-flash")
        assert name == "deepseek"

    def test_unknown_model_returns_none(self):
        assert onboarding.resolve_provider("definitely-not-a-model-9000") is None


class TestAgentPicker:
    def test_deepseek_filter_excludes_wire_incompatible_agents(self):
        """deepseek is openai-completions only: anthropic-messages and
        openai-responses agents (claude-agent-acp, codex-acp) must not be
        offered; chat-compatible core agents must be; oracle (no model) and
        gemini (native provider protocol, bypasses routing) never appear."""
        names = onboarding.compatible_agents("deepseek/deepseek-v4-flash")
        assert "pi-acp" in names and "opencode" in names
        assert "claude-agent-acp" not in names
        assert "codex-acp" not in names
        assert "oracle" not in names and "gemini" not in names


class TestModelPing:
    """A GET /models can 200 while the route is broken (proven in the census);
    only a max_tokens=1 completion exercises key + model id + endpoint."""

    def _transport(self, status, body):
        import httpx

        def handler(request):
            # the ping must hit the chat-completions route with max_tokens=1
            assert request.url.path.endswith("/chat/completions")
            import json

            payload = json.loads(request.content)
            assert payload["max_tokens"] == 1
            return httpx.Response(status, json=body)

        return httpx.MockTransport(handler)

    def test_working_key_reports_ok(self):
        result = onboarding.model_ping(
            "deepseek/deepseek-v4-flash",
            env={"DEEPSEEK_API_KEY": "sk-good"},
            transport=self._transport(200, {"choices": [{}], "model": "x"}),
        )
        assert result.ok
        assert "deepseek" in result.name

    def test_bad_key_reports_failure_with_status(self):
        result = onboarding.model_ping(
            "deepseek/deepseek-v4-flash",
            env={"DEEPSEEK_API_KEY": "sk-bad"},
            transport=self._transport(401, {"error": {"message": "bad key"}}),
        )
        assert not result.ok
        assert "401" in result.detail

    def test_missing_key_fails_before_any_request(self):
        result = onboarding.model_ping("deepseek/deepseek-v4-flash", env={})
        assert not result.ok
        assert "DEEPSEEK_API_KEY" in result.detail


class TestDoctor:
    def test_daytona_without_key_fails_and_docker_binary_checked(self, monkeypatch):
        import httpx

        ok_transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [{}]})
        )
        monkeypatch.setattr("shutil.which", lambda _: None)
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="daytona",
            env={"DEEPSEEK_API_KEY": "sk"},
            ping_transport=ok_transport,
        )
        by_name = {r.name: r for r in results}
        assert not by_name["daytona (DAYTONA_API_KEY)"].ok
        assert "docker" not in " ".join(by_name)  # daytona run: no docker row
        assert by_name["provider key (DEEPSEEK_API_KEY)"].ok
        assert by_name["model ping (deepseek)"].ok

    def test_docker_sandbox_checks_docker_binary(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="docker",
            env={},
            skip_ping=True,
        )
        by_name = {r.name: r for r in results}
        assert not by_name["docker"].ok
        assert not by_name["provider key (DEEPSEEK_API_KEY)"].ok
        assert "model ping (deepseek)" not in by_name


class TestCommandAssembly:
    PREFS: dict = {  # noqa: RUF012 — read-only test fixture
        "agent": "pi-acp",
        "model": "deepseek/deepseek-v4-flash",
        "dataset": "skillsbench",
        "sandbox": "docker",
        "skill_mode": "with-skill",
    }

    def test_final_command_is_a_runnable_eval_run(self):
        cmd = onboarding.final_command(self.PREFS)
        assert cmd == (
            "bench eval run --agent pi-acp --model deepseek/deepseek-v4-flash"
            " -d skillsbench --sandbox docker --skill-mode with-skill"
        )

    def test_tasks_dir_dataset_uses_tasks_dir_flag(self):
        prefs = {**self.PREFS, "dataset": "/data/my-tasks"}
        assert " --tasks-dir /data/my-tasks " in onboarding.final_command(prefs) + " "

    def test_oracle_smoke_argv_swaps_agent_and_pins_one_task(self):
        argv = onboarding.smoke_argv(self.PREFS, task="citation-check")
        assert argv[:4] == ["bench", "eval", "run", "--agent"]
        assert argv[4] == "oracle"
        assert "--include" in argv and "citation-check" in argv
        assert "--model" not in argv  # oracle needs no model
