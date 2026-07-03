"""`bench init` / `bench doctor` CLI behavior (thin glue over onboarding).

Non-interactive mode is the contract CI relies on: every prompt has a flag,
so a fully-flagged invocation must never block on stdin.
"""

from __future__ import annotations

from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def _init_args(home, extra=()):
    return [
        "init",
        "--model",
        "deepseek/deepseek-v4-flash",
        "--agent",
        "pi-acp",
        "--dataset",
        "skillsbench@1.1",
        "--sandbox",
        "docker",
        "--api-key",
        "sk-test-123",
        "--skip-smoke",
        *extra,
    ]


def test_non_interactive_writes_files_and_prints_command(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(app, _init_args(tmp_path))
    assert result.exit_code == 0, result.output
    # secrets → private env file; prefs → config; final command printed
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {
        "DEEPSEEK_API_KEY": "sk-test-123"
    }
    assert ((tmp_path / ".env").stat().st_mode & 0o777) == 0o600
    prefs = onboarding.load_prefs(tmp_path / "config.toml")
    assert prefs["agent"] == "pi-acp"
    assert (
        "bench eval run --agent pi-acp --model deepseek/deepseek-v4-flash"
        in result.output
    )


def test_incompatible_agent_for_model_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--agent",
            "codex-acp",  # openai-responses wire: the run path would reject it
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code != 0
    assert "codex-acp" in result.output and "deepseek" in result.output


def test_doctor_without_setup_hints_at_init(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "bench init" in result.output


def test_doctor_reports_rows_and_fails_on_broken_setup(tmp_path, monkeypatch):
    """Saved setup but no key and no docker: every row shows, exit code 1."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    from benchflow import onboarding

    onboarding.save_prefs(
        tmp_path / "config.toml",
        {
            "agent": "pi-acp",
            "model": "deepseek/deepseek-v4-flash",
            "dataset": "skillsbench",
            "sandbox": "docker",
        },
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "docker" in result.output
    assert "DEEPSEEK_API_KEY" in result.output
    assert "❌" in result.output


def test_interactive_wizard_prompts_and_completes(tmp_path, monkeypatch):
    """No flags: the wizard walks numbered menus (provider → model → agent →
    dataset → sandbox), auto-detects credentials, and only prompts for the
    key when nothing is found."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no ./.env here -> key prompt is the fallback
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices",
        lambda: [("skillsbench@1.1", "87-task benchmark")],
    )
    answers = "\n".join(
        [
            "",  # agent menu -> default (pi-acp)
            "",  # provider menu (filtered) -> default (deepseek)
            "deepseek-v4-flash",  # model id (deepseek has no catalog)
            "",  # dataset menu -> default (skillsbench@1.1)
            "",  # sandbox menu -> default (docker)
            "sk-wizard-key",  # hidden api key (nothing auto-detected)
        ]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {
        "DEEPSEEK_API_KEY": "sk-wizard-key"
    }
    assert "bench eval run --agent pi-acp" in result.output


def test_startup_autoloads_saved_env_file(tmp_path, monkeypatch):
    """A key stored by a previous init is visible to later invocations
    without exporting anything (the CLI auto-loads ~/.benchflow/.env)."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from benchflow import onboarding

    onboarding.write_env_file(tmp_path / ".env", {"DEEPSEEK_API_KEY": "sk-saved"})
    result = runner.invoke(app, [*_init_args(tmp_path)[:-3], "--skip-smoke"])
    # (same init args minus --api-key: the saved key must be found)
    assert result.exit_code == 0, result.output
    assert "DEEPSEEK_API_KEY found in your environment" in result.output


def test_full_smoke_runs_credential_free_oracle_stage(tmp_path, monkeypatch):
    """--full-smoke executes the stage-1 oracle eval (sandbox plumbing proof)
    before the credential checks; the exact argv is the assembled smoke
    command."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)

        class R:
            returncode = 0

        return R()

    from benchflow.onboarding import CheckResult

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "benchflow.onboarding.run_doctor",
        lambda *a, **k: [CheckResult("stub", True, "")],
    )
    result = runner.invoke(
        app,
        [*_init_args(tmp_path)[:-1], "--full-smoke", "--smoke-task", "citation-check"],
    )
    assert result.exit_code == 0, result.output
    # doctor checks still run (docker/key/ping will fail here — that's fine,
    # exit code reflects them); the oracle stage must have been invoked first.
    assert any(
        argv[:5] == ["bench", "eval", "run", "--agent", "oracle"] for argv in calls
    ), result.output
    assert any("citation-check" in argv for argv in calls)


def test_full_smoke_oracle_failure_fails_init(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))

    def failing_run(argv, **kwargs):
        class R:
            returncode = 1

        return R()

    monkeypatch.setattr("subprocess.run", failing_run)
    result = runner.invoke(
        app,
        [*_init_args(tmp_path)[:-1], "--full-smoke", "--smoke-task", "citation-check"],
    )
    assert result.exit_code == 1
    assert "oracle" in result.output


def test_full_smoke_without_task_errors_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(app, [*_init_args(tmp_path), "--full-smoke"])
    assert result.exit_code == 2
    assert "--smoke-task" in result.output


def test_api_key_flag_rejected_for_non_api_key_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "google-vertex/gemini-3-pro",
            "--agent",
            "pi-acp",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code != 0
    assert "adc" in result.output.lower()


def test_smoke_verifies_the_key_just_provided(tmp_path, monkeypatch):
    """--api-key must be what the checks verify, even if a different key is
    already exported (verify what was saved, not what happened to be set)."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-stale-exported")
    seen = {}

    def fake_doctor(model, sandbox, env, **kw):
        seen["key"] = env.get("DEEPSEEK_API_KEY")
        from benchflow.onboarding import CheckResult

        return [CheckResult("stub", True, "")]

    monkeypatch.setattr("benchflow.onboarding.run_doctor", fake_doctor)
    result = runner.invoke(app, _init_args(tmp_path)[:-1])  # drop --skip-smoke
    assert result.exit_code == 0, result.output
    assert seen["key"] == "sk-test-123"


def test_bare_model_with_inferable_key_completes(tmp_path, monkeypatch):
    """No registered provider, but a well-known key family (claude-*): the
    wizard proceeds, stores the key under the inferred env var, and skips the
    endpoint ping honestly."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "claude-opus-4-6",
            "--agent",
            "claude-agent-acp",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-ant-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {
        "ANTHROPIC_API_KEY": "sk-ant-x"
    }


def test_subscription_login_skips_key_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda agent, key: True
    )
    # no --api-key, no env key, no stdin: would block on the hidden prompt
    # unless the subscription path answers first.
    result = runner.invoke(app, [*_init_args(tmp_path)[:-3], "--skip-smoke"])
    assert result.exit_code == 0, result.output
    assert "subscription" in result.output.lower()


def test_default_dataset_produces_a_parseable_spec(tmp_path, monkeypatch):
    """The wizard's whole job is handing over a command that RUNS: a
    registry-style dataset must carry a version (name@version) or init must
    reject it — bare 'skillsbench' fails bench eval run's dataset parsing."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--agent",
            "pi-acp",
            "--dataset",
            "skillsbench",  # version-less registry name
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code != 0
    assert "@" in result.output  # points at name@version

    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--agent",
            "pi-acp",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "-d skillsbench@1.1" in result.output
    from benchflow._utils.dataset_registry import parse_dataset_spec

    parse_dataset_spec("skillsbench@1.1")  # the emitted value must parse


def test_doctor_survives_corrupt_and_partial_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text('model = "unterminated')
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "bench init" in result.output

    (tmp_path / "config.toml").write_text('agent = "pi-acp"\n')  # missing keys
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "bench init" in result.output


def test_full_smoke_missing_bench_binary_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))

    def gone(argv, **kw):
        raise FileNotFoundError(2, "No such file", "bench")

    monkeypatch.setattr("subprocess.run", gone)
    result = runner.invoke(
        app,
        [*_init_args(tmp_path)[:-1], "--full-smoke", "--smoke-task", "t1"],
    )
    assert result.exit_code == 1
    assert "PATH" in result.output  # clear message, not a traceback


def test_stale_exported_key_shadow_is_warned(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-stale-exported")
    result = runner.invoke(app, _init_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert "shadow" in result.output.lower()


def test_wizard_is_selection_driven_with_auto_key_detection(tmp_path, monkeypatch):
    """Hermes-style UX: every step is a numbered menu (or Enter for the
    default) — and the key is auto-detected from ./.env in the working
    folder, so the user never types it."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('DEEPSEEK_API_KEY="sk-from-folder"\n')
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices",
        lambda: [("skillsbench@1.1", "87-task benchmark")],
    )
    answers = "\n".join(
        [
            "",  # agent menu -> Enter = default (pi-acp)
            "",  # provider menu (filtered to pi-acp-routable) -> Enter = deepseek
            "deepseek-v4-flash",  # model (free text w/ hint; deepseek has no catalog)
            "",  # dataset menu -> Enter = default (skillsbench@1.1)
            "",  # sandbox menu -> Enter = default (docker)
        ]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    # told the user the absolute source, a key fingerprint, and the destination
    assert str(tmp_path / ".env") in result.output
    assert "…" in result.output and "saved" in result.output
    from benchflow import onboarding

    # passed through into the saved setup for future runs
    assert onboarding.read_env_file(tmp_path / "home" / ".env") == {
        "DEEPSEEK_API_KEY": "sk-from-folder"
    }
    assert (
        "bench eval run --agent pi-acp --model deepseek/deepseek-v4-flash"
        in result.output
    )
    # menus were shown, not free-text demands
    assert "1)" in result.output


def test_provider_menu_is_filtered_by_chosen_agent(tmp_path, monkeypatch):
    """Agent comes first; the provider menu then only offers providers that
    agent can route (codex-acp speaks openai-responses -> deepseek, which is
    completions-only, must not be offered)."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # pick codex-acp by name from the agent menu, then abort at the provider
    # menu with an out-of-range answer + EOF; the menu text is what we assert.
    result = runner.invoke(
        app,
        ["init", "--skip-smoke", "--agent", "codex-acp"],
        input="\n",  # provider menu: Enter default, then model prompt EOFs
    )
    menu = result.output.split("Provider", 1)[-1]
    menu = menu.split("Select", 1)[0]
    assert "openai" in menu
    assert "deepseek" not in menu


def test_local_tasks_dir_bare_name_is_normalized_not_rejected(tmp_path, monkeypatch):
    """Picking 'a local tasks dir' and answering a bare relative name must
    produce a --tasks-dir command, not a registry-spec rejection."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mytasks").mkdir()
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices",
        lambda: [("skillsbench@1.1", "")],
    )
    answers = "\n".join(
        [
            "",  # agent -> pi-acp
            "",  # provider -> deepseek
            "deepseek-v4-flash",
            "2",  # dataset menu: "a local tasks dir"
            "mytasks",  # bare relative name
            "",  # sandbox -> docker
            "sk-k",  # key prompt
        ]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert "--tasks-dir ./mytasks" in result.output
