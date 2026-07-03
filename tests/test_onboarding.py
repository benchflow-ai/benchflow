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


class TestEnvFileRobustness:
    """Hand-edited dotenv dialects and corrupt files must never break the CLI
    (the autoload callback runs on EVERY subcommand — including the ones
    needed to fix the file)."""

    def test_export_prefix_and_single_quotes_parse(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("export DEEPSEEK_API_KEY=sk-abc\nZAI_API_KEY='sk-z'\n")
        assert onboarding.read_env_file(path) == {
            "DEEPSEEK_API_KEY": "sk-abc",
            "ZAI_API_KEY": "sk-z",
        }

    def test_malformed_lines_are_skipped_not_fatal(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text('=oops\n  =also bad\nGOOD_KEY="v"\nBAD KEY=x\n')
        monkeypatch.delenv("GOOD_KEY", raising=False)
        applied = onboarding.load_env_file(path)  # must not raise OSError
        assert applied == ["GOOD_KEY"]
        monkeypatch.delenv("GOOD_KEY")

    def test_unreadable_file_warns_not_crashes(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("A=1")
        path.chmod(0o000)
        try:
            assert onboarding.load_env_file(path) == []  # degraded, no raise
        finally:
            path.chmod(0o600)


class TestModelPingProviderClasses:
    """The ping must use the SAME endpoint join as the run path — no URL
    guessing — and be honest for provider classes it cannot exercise."""

    def _capture(self, status=200, body=None):
        import httpx

        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["headers"] = dict(request.headers)
            return httpx.Response(status, json=body or {"choices": [{}]})

        return httpx.MockTransport(handler), seen

    def test_zai_versioned_base_gets_no_extra_v1(self):
        transport, seen = self._capture()
        result = onboarding.model_ping(
            "zai/glm-5", env={"ZAI_API_KEY": "sk-z"}, transport=transport
        )
        assert result.ok, result.detail
        assert seen["url"] == "https://api.z.ai/api/paas/v4/chat/completions"

    def test_anthropic_only_provider_pings_messages_with_x_api_key(self):
        transport, seen = self._capture(body={"content": [], "type": "message"})
        result = onboarding.model_ping(
            "azure-foundry-anthropic/claude-opus-4-6",
            env={"AZURE_API_KEY": "sk-az", "AZURE_RESOURCE": "myres"},
            transport=transport,
        )
        assert result.ok, result.detail
        assert seen["url"] == (
            "https://myres.services.ai.azure.com/anthropic/v1/messages"
        )
        assert seen["headers"].get("x-api-key") == "sk-az"

    def test_adc_provider_is_honestly_skipped_not_failed(self):
        result = onboarding.model_ping("google-vertex/gemini-3-pro", env={})
        assert result.ok
        assert "skipped" in result.detail

    def test_200_with_non_completion_body_fails(self):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="<html>login page</html>")
        )
        result = onboarding.model_ping(
            "deepseek/deepseek-v4-flash",
            env={"DEEPSEEK_API_KEY": "sk"},
            transport=transport,
        )
        assert not result.ok
        assert "not a completion" in result.detail

    def test_error_detail_strips_terminal_escapes(self):
        import httpx

        transport = httpx.MockTransport(
            lambda req: httpx.Response(500, text="bad \x1b]0;pwned\x07 thing")
        )
        result = onboarding.model_ping(
            "deepseek/deepseek-v4-flash",
            env={"DEEPSEEK_API_KEY": "sk"},
            transport=transport,
        )
        assert not result.ok
        assert "\x1b" not in result.detail and "\x07" not in result.detail


class TestDoctorHardening:
    def test_unknown_sandbox_is_a_failing_row_not_silence(self):
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="dokcer",  # typo
            env={"DEEPSEEK_API_KEY": "sk"},
            skip_ping=True,
        )
        row = next(r for r in results if r.name == "sandbox")
        assert not row.ok
        assert "dokcer" in row.detail

    def test_litellm_route_row_reports_resolution(self):
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="docker",
            env={"DEEPSEEK_API_KEY": "sk"},
            skip_ping=True,
        )
        row = next(r for r in results if r.name.startswith("litellm route"))
        assert row.ok  # deepseek resolves without network

    def test_unregistered_model_with_inferable_key_checks_that_key(self):
        results = onboarding.run_doctor(
            model="claude-opus-4-6",
            sandbox="docker",
            env={},
            skip_ping=True,
        )
        row = next(r for r in results if "ANTHROPIC_API_KEY" in r.name)
        assert not row.ok  # key absent -> honest red row, not "no provider"


class TestEnvFileDataSafety:
    def test_undecodable_file_degrades_never_crashes(self, tmp_path):
        """One stray latin-1 byte must not brick every CLI command (the
        autoload callback runs on ALL subcommands, including the repair
        paths)."""
        path = tmp_path / ".env"
        path.write_bytes(b"NOTE=caf\xe9\nGOOD=1\n")
        assert onboarding.load_env_file(path) == []  # degraded, no raise

    def test_rewrite_preserves_comments_and_unparsed_lines(self, tmp_path):
        """write_env_file must not destroy what the parser skips — a hand
        commented file survives the next `bench init` verbatim."""
        path = tmp_path / ".env"
        path.write_text("# my deepseek key\nA=1\nnot a kv line\n")
        onboarding.write_env_file(path, {"B": "2", "A": "changed"})
        text = path.read_text()
        assert "# my deepseek key" in text
        assert "not a kv line" in text
        assert onboarding.read_env_file(path) == {"A": "changed", "B": "2"}

    def test_rewrite_refuses_to_clobber_undecodable_file(self, tmp_path):
        path = tmp_path / ".env"
        path.write_bytes(b"\xff\xfe binary junk")
        import pytest

        with pytest.raises(OSError):
            onboarding.write_env_file(path, {"A": "1"})
        assert path.read_bytes() == b"\xff\xfe binary junk"  # untouched

    def test_rewrite_retightens_widened_permissions(self, tmp_path):
        path = tmp_path / ".env"
        onboarding.write_env_file(path, {"A": "1"})
        path.chmod(0o644)
        onboarding.write_env_file(path, {"B": "2"})
        assert (path.stat().st_mode & 0o777) == 0o600


class TestSubscriptionAwareDoctor:
    def test_subscription_login_skips_key_route_and_ping_rows(self, monkeypatch):
        """A subscription-onboarded setup (host login files, no API key) must
        not be failed by its own smoke test: key/route/ping rows are skipped,
        not red."""
        monkeypatch.setattr(
            "benchflow.agents.env.check_subscription_auth", lambda a, k: True
        )
        results = onboarding.run_doctor(
            model="claude-opus-4-6",
            sandbox="docker",
            env={},
            agent="claude-agent-acp",
        )
        assert all(r.ok for r in results if r.name != "docker")
        skipped = [r for r in results if r.skipped]
        assert any("subscription" in r.detail for r in skipped)

    def test_skipped_rows_carry_the_skipped_flag(self):
        result = onboarding.model_ping("google-vertex/gemini-3-pro", env={})
        assert result.ok and result.skipped

    def test_modal_sandbox_row_is_sane(self):
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="modal",
            env={"DEEPSEEK_API_KEY": "sk"},
            skip_ping=True,
        )
        row = next(r for r in results if r.name == "sandbox")
        assert row.ok
        assert "unknown" not in row.detail

    def test_route_row_failure_names_the_exception_type(self, monkeypatch):
        def boom(model, env):
            raise ValueError("boom")

        monkeypatch.setattr(
            "benchflow.providers.litellm_config.resolve_litellm_route", boom
        )
        results = onboarding.run_doctor(
            model="deepseek/deepseek-v4-flash",
            sandbox="docker",
            env={"DEEPSEEK_API_KEY": "sk"},
            skip_ping=True,
        )
        row = next(r for r in results if r.name.startswith("litellm route"))
        assert not row.ok
        assert "ValueError" in row.detail


class TestDetectKey:
    """After the model is chosen the wizard must find credentials itself:
    subscription login, then the process environment (which includes the
    saved ~/.benchflow/.env), then a ./.env in the working folder — prompting
    only when all three miss."""

    def test_exported_key_beats_subscription_matching_the_run_path(
        self, monkeypatch, tmp_path
    ):
        """resolve_agent_env inherits an exported key into the agent env and
        uses_native_subscription_auth then returns False — so at RUN time an
        exported key wins over a subscription login. detect_key must report
        the same order or init announces an auth source the run won't use."""
        monkeypatch.setattr(
            "benchflow.agents.env.check_subscription_auth", lambda a, k: True
        )
        monkeypatch.setenv("PROBE_KEY", "from-env")
        source, value = onboarding.detect_key(
            "PROBE_KEY", agent="claude-agent-acp", cwd=tmp_path
        )
        assert source == "environment" and value == "from-env"

    def test_subscription_wins_when_no_key_is_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "benchflow.agents.env.check_subscription_auth", lambda a, k: True
        )
        monkeypatch.delenv("PROBE_KEY", raising=False)
        source, value = onboarding.detect_key(
            "PROBE_KEY", agent="claude-agent-acp", cwd=tmp_path
        )
        assert source == "subscription" and value is None

    def test_process_env_beats_cwd_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROBE_KEY", "from-env")
        (tmp_path / ".env").write_text('PROBE_KEY="from-cwd"\n')
        source, value = onboarding.detect_key("PROBE_KEY", cwd=tmp_path)
        assert source == "environment" and value == "from-env"

    def test_cwd_dotenv_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PROBE_KEY", raising=False)
        (tmp_path / ".env").write_text('PROBE_KEY="from-cwd"\n')
        source, value = onboarding.detect_key("PROBE_KEY", cwd=tmp_path)
        assert source == "./.env" and value == "from-cwd"

    def test_nothing_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PROBE_KEY", raising=False)
        assert onboarding.detect_key("PROBE_KEY", cwd=tmp_path) == (None, None)


class TestDatasetChoices:
    def test_newest_version_first_per_name(self, monkeypatch):
        entries = [
            {"name": "skillsbench", "version": "1.0", "description": "old"},
            {"name": "skillsbench", "version": "1.1", "description": "new"},
        ]
        monkeypatch.setattr(
            "benchflow._utils.dataset_registry.load_registry", lambda src: entries
        )
        choices = onboarding.dataset_choices()
        assert choices[0][0] == "skillsbench@1.1"
        assert "skillsbench@1.0" in [c[0] for c in choices]

    def test_registry_unreachable_degrades_to_empty(self, monkeypatch):
        def boom(src):
            raise OSError("offline")

        monkeypatch.setattr("benchflow._utils.dataset_registry.load_registry", boom)
        assert onboarding.dataset_choices() == []

    def test_malformed_registry_entries_degrade_not_crash(self, monkeypatch):
        monkeypatch.setattr(
            "benchflow._utils.dataset_registry.load_registry",
            lambda src: ["not-a-dict", {"name": "ok", "version": "1.0"}],
        )
        # must not raise; the well-formed entry may or may not survive
        assert isinstance(onboarding.dataset_choices(), list)


class TestAgentPaths:
    def test_full_universe_loads_and_classifies_by_path(self, tmp_path, monkeypatch):
        """The wizard must offer ALL three adaptation paths — core built-ins
        plus the remote manifest catalog (auto-loaded on demand) plus any
        installed plugin packages — grouped by the naming law."""
        from benchflow.agents import remote_manifests

        (tmp_path / "probe-manifest").mkdir()
        (tmp_path / "probe-manifest" / "manifest.toml").write_text(
            'contract_version = "1.0"\nname = "probe-manifest"\n'
            'protocol = "acp"\ninstall_cmd = "true"\nlaunch_cmd = "true"\n'
        )
        monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))
        remote_manifests._reset_for_tests()
        try:
            paths = onboarding.agent_paths()
            assert "probe-manifest" in paths["acp"]  # manifest catalog loaded
            assert "pi-acp" in paths["acp"]  # core
            assert all(a.startswith("ai-sdk") for a in paths.get("ai-sdk", []))
            assert all(a.startswith("omnigent-") for a in paths.get("omnigent", []))
            assert "oracle" not in paths["acp"] and "gemini" not in paths["acp"]
        finally:
            remote_manifests._reset_for_tests()
            from benchflow.agents import registry

            for n in ("probe-manifest",):
                registry.AGENTS.pop(n, None)
                registry.AGENT_INSTALLERS.pop(n, None)
                registry.AGENT_LAUNCH.pop(n, None)


class TestDetectKeySources:
    def test_all_sources_listed_in_run_path_order(self, monkeypatch, tmp_path):
        """The auth menu needs every detected source, ordered the way the run
        path would use them: environment, subscription, ./.env."""
        monkeypatch.setattr(
            "benchflow.agents.env.check_subscription_auth", lambda a, k: True
        )
        monkeypatch.setenv("PROBE_KEY", "from-env")
        (tmp_path / ".env").write_text('PROBE_KEY="from-cwd"\n')
        sources = onboarding.detect_key_sources(
            "PROBE_KEY", agent="claude-agent-acp", cwd=tmp_path
        )
        assert [s for s, _ in sources] == ["environment", "subscription", "./.env"]
