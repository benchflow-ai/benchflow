"""`bench init` catalog-agent and package-path menu behavior."""

from __future__ import annotations

from tests.init_cli_helpers import app, runner


def _manifest_source(tmp_path, monkeypatch, name="probe-flag-agent"):
    from benchflow.agents import remote_manifests

    d = tmp_path / "src" / name
    d.mkdir(parents=True)
    (d / "manifest.toml").write_text(
        f'contract_version = "1.0"\nname = "{name}"\nprotocol = "acp"\n'
        'install_cmd = "true"\nlaunch_cmd = "true"\n'
    )
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path / "src"))
    remote_manifests._reset_for_tests()
    return name


def test_agent_flag_reaches_catalog_agents_via_autoload(tmp_path, monkeypatch):
    """--agent naming a catalog-only agent must work like `bench run` does
    (the miss path autoloads) — not exit with a bogus protocol mismatch."""
    name = _manifest_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    try:
        result = runner.invoke(
            app,
            [
                "init",
                "--agent",
                name,
                "--model",
                "deepseek/deepseek-v4-flash",
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
    finally:
        from benchflow.agents import registry, remote_manifests

        remote_manifests._reset_for_tests()
        registry.AGENTS.pop(name, None)
        registry.AGENT_INSTALLERS.pop(name, None)
        registry.AGENT_LAUNCH.pop(name, None)


def test_unknown_agent_flag_says_unknown_not_protocol_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    from benchflow.agents import remote_manifests

    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    remote_manifests._reset_for_tests()
    result = runner.invoke(
        app,
        [
            "init",
            "--agent",
            "definitely-not-an-agent-9000",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--dataset",
            "skillsbench@1.1",
            "--sandbox",
            "docker",
            "--api-key",
            "sk-x",
            "--skip-smoke",
        ],
    )
    assert result.exit_code == 1
    assert "Unknown agent" in result.output
    assert "protocol mismatch" not in result.output


def test_other_agent_offers_static_catalog_and_fetches_one_manifest(
    tmp_path, monkeypatch
):
    """'other' shows a STATIC catalog list (zero network) and selecting an
    entry fetches ONLY that agent's manifest — never the full repo."""
    from benchflow.agents import registry, remote_manifests

    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    # hermetic single-manifest source
    from benchflow import onboarding

    # Deterministic target regardless of suite order: earlier tests (e.g.
    # manifest parity) may have registered the WHOLE catalog — evict one
    # entry for the duration of this test and restore it afterwards.
    target = onboarding.CATALOG_AGENTS[0]
    saved = (
        registry.AGENTS.pop(target, None),
        registry.AGENT_INSTALLERS.pop(target, None),
        registry.AGENT_LAUNCH.pop(target, None),
    )
    d = tmp_path / "src" / "acp" / target
    d.mkdir(parents=True)
    (d / "manifest.toml").write_text(
        f'contract_version = "1.0"\nname = "{target}"\nprotocol = "acp"\n'
        'api_protocol = "openai-completions"\n'
        'install_cmd = "true"\nlaunch_cmd = "true"\n'
    )
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path / "src"))
    # full-repo loading must NOT happen
    monkeypatch.setattr(
        remote_manifests,
        "autoload_remote_manifest_agents",
        lambda: (_ for _ in ()).throw(AssertionError("full catalog load!")),
    )
    monkeypatch.setattr(
        "benchflow.onboarding.dataset_choices", lambda: [("skillsbench@1.1", "")]
    )
    n_local = len(onboarding.acp_agents())
    target_idx = [n for n, _ in onboarding.path_choices("acp")].index(target) + 1
    answers = "\n".join(
        [
            str(n_local + 1),  # agent menu: "other" -> path menu
            "1",  # path menu: acp
            str(target_idx),  # pick it from the static list
            "",  # provider -> deepseek
            "deepseek-v4-flash",
            "",  # dataset
            "",  # sandbox
            "sk-k",  # key
        ]
    )
    try:
        result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
        assert result.exit_code == 0, result.output
        assert f"--agent {target}" in result.output
    finally:
        registry.AGENTS.pop(target, None)
        registry.AGENT_INSTALLERS.pop(target, None)
        registry.AGENT_LAUNCH.pop(target, None)
        for store, val in zip(
            (registry.AGENTS, registry.AGENT_INSTALLERS, registry.AGENT_LAUNCH),
            saved,
            strict=True,
        ):
            if val is not None:
                store[target] = val


def test_other_shows_three_paths_then_that_paths_agents(tmp_path, monkeypatch):
    """'other' -> path menu (acp / ai-sdk / omnigent with counts) -> that
    path's static agent list. Package paths guide installation instead of
    dead-ending."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from benchflow import onboarding

    n_local = len(onboarding.acp_agents())
    answers = "\n".join(
        [
            str(n_local + 1),  # agent menu -> other
            "3",  # path menu -> omnigent
            "1",  # first omnigent agent (package not installed here? may be)
        ]
    )
    result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
    out = result.output
    assert "acp" in out and "ai-sdk" in out and "omnigent" in out  # path menu
    assert "omnigent-" in out  # that path's agents listed


def test_uninstalled_package_agent_gets_install_guidance(tmp_path, monkeypatch):
    """Selecting a package-path agent that is not importable prints the exact
    install command instead of a crash or a dead end."""
    monkeypatch.setenv("BENCHFLOW_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from benchflow.agents import registry

    # simulate not-installed: evict ai-sdk-codex if some env registered it
    saved = registry.AGENTS.pop("ai-sdk-codex", None)
    try:
        from benchflow import onboarding

        n_local = len(onboarding.acp_agents())
        idx = [n for n, _ in onboarding.path_choices("ai-sdk")].index("ai-sdk-codex")
        answers = "\n".join([str(n_local + 1), "2", str(idx + 1)])
        result = runner.invoke(app, ["init", "--skip-smoke"], input=answers + "\n")
        assert result.exit_code == 1
        assert "uv pip install" in result.output
        assert "ai-sdk/harness-codex" in result.output  # the exact subdirectory
    finally:
        if saved is not None:
            registry.AGENTS["ai-sdk-codex"] = saved
