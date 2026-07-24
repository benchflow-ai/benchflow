"""`bench init` interactive menu and credential-source behavior."""

from __future__ import annotations

from tests.init_cli_helpers import _init_args, app, runner


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


def test_flagged_local_tasks_dir_bare_name_is_normalized(tmp_path, monkeypatch):
    """Guards PR #883: fully flagged local task dirs mirror the prompt path."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mytasks").mkdir()

    result = runner.invoke(
        app,
        [
            "init",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--agent",
            "pi-acp",
            "--dataset",
            "mytasks",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--tasks-dir ./mytasks" in result.output


def test_auto_key_detection_prefers_cwd_dotenv_over_exported_key(tmp_path, monkeypatch):
    """Guards PR #883: init validates the key the final run will use."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-exported")
    (tmp_path / ".env").write_text('DEEPSEEK_API_KEY="sk-from-cwd"\n')

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
            "--skip-smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / "home" / ".env") == {
        "DEEPSEEK_API_KEY": "sk-from-cwd"
    }
    assert str(tmp_path / ".env") in result.output
    assert "shadow" in result.output.lower()


def test_provider_labels_show_the_matched_protocol(tmp_path, monkeypatch):
    """A provider's label must show the endpoint the CHOSEN agent will use —
    aws-bedrock's primary wire is openai-responses, but next to
    claude-agent-acp it serves anthropic-messages and must say so."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "claude-agent-acp"], input="\n"
    )
    menu = result.output.split("Provider", 1)[-1].split("Select", 1)[0]
    assert "openai-responses" not in menu
    assert "anthropic-messages" in menu
    assert "BYO" in menu  # vllm: no canonical URL, caller supplies semantics


def test_interactive_auth_menu_lists_subscription_as_a_choice(tmp_path, monkeypatch):
    """OpenClaw-style auth step: when a subscription login AND a key are both
    available, the user chooses — subscription is a listed option, not a
    silent auto-decision."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-exported-key")
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: True
    )
    monkeypatch.setattr("benchflow.cli.init_cmd._isatty", lambda: True)
    answers = "\n".join(
        [
            "",  # agent -> pi-acp
            "",  # provider -> deepseek
            "deepseek-v4-flash",
            "",  # dataset (registry stubbed below)
            "",  # sandbox -> docker
            "2",  # auth menu: pick the subscription login
        ]
    )
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices", lambda: [("skillsbench@1.1", "")]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert "subscription" in result.output.lower()  # listed as an option
    assert "…-key" in result.output or "…" in result.output  # key fingerprinted


def test_subscription_pick_is_honored_by_the_smoke(tmp_path, monkeypatch):
    """Choosing the subscription login while a key is exported must make the
    smoke verify the SUBSCRIPTION setup (key rows skipped) and warn that the
    shell export will shadow it at run time."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-exported")
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: True
    )
    monkeypatch.setattr("benchflow.cli.init_cmd._isatty", lambda: True)
    seen = {}

    def fake_doctor(model, sandbox, env, **kw):
        seen["key_in_env"] = "DEEPSEEK_API_KEY" in env
        from benchflow.onboarding import CheckResult

        return [CheckResult("stub", True, "")]

    monkeypatch.setattr("benchflow.onboarding.run_doctor", fake_doctor)
    result = runner.invoke(
        app,
        [
            "init",
            "--agent",
            "pi-acp",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
        ],
        input="2\n",  # credentials menu: 1=env key, 2=subscription
    )
    assert result.exit_code == 0, result.output
    assert seen["key_in_env"] is False  # the declined key is NOT verified
    assert "shadow" in result.output.lower()  # told about the shell export


def test_claude_agent_gets_native_anthropic_provider_and_subscription(
    tmp_path, monkeypatch
):
    """An anthropic-native agent (claude-agent-acp) must offer 'anthropic'
    in the provider menu — the registry has no such endpoint, but the run
    path supports it via subscription login / ANTHROPIC_API_KEY — and picking
    it must surface the subscription login in the credentials menu."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: True
    )
    monkeypatch.setattr("benchflow.cli.init_cmd._isatty", lambda: True)
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices", lambda: [("skillsbench@1.1", "")]
    )
    answers = "\n".join(
        [
            "1",  # provider menu: anthropic (native) listed first for this agent
            "",  # model id -> default (claude-sonnet-4-6)
            "",  # dataset -> default
            "",  # sandbox -> docker
            "1",  # credentials menu: subscription login (listed!)
        ]
    )
    result = runner.invoke(
        app,
        ["init", "--skip-smoke", "--agent", "claude-agent-acp"],
        input=answers + "\n",
    )
    assert result.exit_code == 0, result.output
    menu = result.output.split("Provider", 1)[-1]
    assert "anthropic" in menu.split("Select", 1)[0]
    assert "subscription login" in result.output
    assert "--model claude-sonnet-4-6" in result.output


def test_subscription_status_announced_right_after_agent_choice(tmp_path, monkeypatch):
    """Choosing a subscription-capable agent immediately reports whether a
    local login was found — or guides the user to log in."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # found case
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: True
    )
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "claude-agent-acp"], input="\n"
    )
    assert "subscription login found" in result.output.lower()
    # not-found case -> login guidance
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: False
    )
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "claude-agent-acp"], input="\n"
    )
    out = result.output.lower()
    assert "no" in out and "subscription login" in out
    assert "claude setup-token" in result.output  # concrete login guidance


def test_codex_subscription_guidance_uses_codex_login(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchflow.agents.env.check_subscription_auth", lambda a, k: False
    )
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "codex-acp"], input="\n"
    )
    assert "codex login" in result.output


def test_gemini_flags_path_works_via_inferred_key(tmp_path, monkeypatch):
    """gemini is menu-excluded (native wire) but --agent gemini with a
    gemini-* model must work via the inferred GEMINI_API_KEY — not die with a
    bogus 'wire protocol mismatch'."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "--agent",
            "gemini",
            "--model",
            "gemini-3-pro",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-fake",
            "--skip-smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--agent gemini --model gemini-3-pro" in result.output
    from benchflow import onboarding

    assert onboarding.read_env_file(tmp_path / ".env") == {"GEMINI_API_KEY": "sk-fake"}


def test_gemini_with_wrong_family_model_gets_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "init",
            "--agent",
            "gemini",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-fake",
            "--skip-smoke",
        ],
    )
    assert result.exit_code == 1
    assert "gemini" in result.output and "gemini-*" in result.output
    assert "protocol mismatch" not in result.output  # no misleading diagnosis


def test_gemini_interactive_gets_native_provider_menu(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "gemini"], input="\n"
    )
    menu = result.output.split("Provider", 1)[-1].split("Select", 1)[0]
    assert "google" in menu and "GEMINI_API_KEY" in menu
    assert "deepseek" not in menu  # not the 21-provider lie


def test_codex_provider_default_is_openai_not_bedrock(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "codex-acp"], input="\n"
    )
    # the Select prompt default index must point at openai
    menu = result.output.split("Provider", 1)[-1]
    import re

    m = re.search(r"(\d+)\) openai ", menu)
    d = re.search(r"Select \[(\d+)\]", menu)
    assert m and d and m.group(1) == d.group(1), menu[:400]


def test_provider_label_not_a_duplicate_of_the_name(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "--skip-smoke", "--agent", "deepagents"], input="\n"
    )
    menu = result.output.split("Provider", 1)[-1].split("Select", 1)[0]
    assert "deepseek  — deepseek" not in menu  # no information-free labels


def test_prompts_have_line_editing():
    """Arrows/backspace must work in every prompt: click's input() only gets
    line editing when readline is loaded — importing the wizard module must
    load it (guarded for platforms without it)."""
    import sys

    import benchflow.cli.init_cmd  # noqa: F401

    assert "readline" in sys.modules


def test_menu_reprompts_on_non_integer_input(tmp_path, monkeypatch):
    """Typos at a Select menu must re-prompt, not crash: typer.prompt's
    re-prompt loop catches its VENDORED click's UsageError, so a real
    click.IntRange BadParameter escaped as a traceback (auth-matrix find)."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    answers = "\n".join(
        [
            "not-a-number",  # agent menu: garbage -> must re-prompt
            "",  # then Enter = default pi-acp
            "",  # provider -> deepseek
            "deepseek-v4-flash",
            "skillsbench@1.1",
            "docker",
            "sk-k",
        ]
    )
    # dataset/sandbox prompts appear as menus; feed by index-agnostic strings
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices", lambda: [("skillsbench@1.1", "")]
    )
    answers = "\n".join(["not-a-number", "", "", "deepseek-v4-flash", "", "", "sk-k"])
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    assert "Traceback" not in result.output
    assert result.exit_code == 0, result.output


def test_init_smoke_prints_skip_note(tmp_path, monkeypatch):
    """A green smoke with skipped rows must say so (the doctor already
    does): '(N check(s) skipped — not verifiable before run time)'."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from benchflow.onboarding import CheckResult

    monkeypatch.setattr(
        "benchflow.onboarding.run_doctor",
        lambda *a, **k: [
            CheckResult("docker", True, "ok"),
            CheckResult("provider auth", True, "skipped — sub", skipped=True),
        ],
    )
    result = runner.invoke(app, _init_args(tmp_path)[:-1])  # no --skip-smoke
    assert result.exit_code == 0, result.output
    assert "check(s) skipped" in result.output


def test_non_tty_env_source_announced_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-exported-key")
    result = runner.invoke(app, [*_init_args(tmp_path)[:-3], "--skip-smoke"])
    assert result.exit_code == 0, result.output
    assert result.output.count("Using DEEPSEEK_API_KEY from your environment") == 1
