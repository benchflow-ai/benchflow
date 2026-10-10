use std::path::PathBuf;

use serde_json::Value;

use super::*;

/// 0.8's registry strings (see `testdata/registry-0.8.json`'s `_about`).
fn golden() -> Value {
    serde_json::from_str(include_str!("../../testdata/registry-0.8.json")).unwrap()
}

fn golden_str(path: &[&str]) -> String {
    let golden = golden();
    let value = path.iter().fold(&golden, |value, key| &value[*key]);
    value
        .as_str()
        .unwrap_or_else(|| panic!("no golden string at {path:?}"))
        .to_string()
}

fn builtin(name: &str) -> AgentManifest {
    Registry::builtin().find(name).unwrap().clone()
}

fn parse(text: &str) -> Result<AgentManifest, ManifestError> {
    AgentManifest::parse(
        text,
        ManifestSource::File(PathBuf::from("agents/test.toml")),
    )
}

const MINIMAL: &str = r#"
name = "my-agent"

[install]
npm = "my-agent@1.0.0"
bin = "my-agent"
"#;

#[test]
fn builtins_are_the_four_0_8_agents_with_their_aliases() {
    let registry = Registry::builtin();
    let names: Vec<&str> = registry.agents().iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names[..4], ["claude", "codex", "gemini", "opencode"]);
    for (alias, name) in [
        ("claude-agent-acp", "claude"),
        ("claude-code", "claude"),
        ("codex-acp", "codex"),
        ("gemini", "gemini"),
        ("gemini-cli", "gemini"),
        ("opencode", "opencode"),
    ] {
        assert_eq!(registry.find(alias).unwrap().name, name, "{alias}");
    }
}

#[test]
fn claude_install_matches_0_8_byte_for_byte() {
    let claude = builtin("claude");
    assert_eq!(
        claude.install_script(),
        golden_str(&["claude-agent-acp", "install"])
    );
    assert_eq!(
        claude.launch_command(),
        golden_str(&["claude-agent-acp", "launch"])
    );
    // The launcher hands the pinned CLI to the adapter.
    let InstallMethod::Npm(npm) = &claude.install.method else {
        panic!()
    };
    assert_eq!(
        install::launcher_script(npm),
        golden_str(&["launchers", "claude-agent-acp"])
    );
    assert!(claude.install_script().contains("'2.1.280 (Claude Code)'"));
}

#[test]
fn codex_install_and_launch_match_0_8() {
    let codex = builtin("codex");
    assert_eq!(
        codex.install_script(),
        golden_str(&["codex-acp", "install"])
    );
    // Agent options' flags go where `"$@"` stands; with none set it expands
    // to nothing, which is 0.8's command.
    assert_eq!(
        codex.launch_command().replace(r#" "$@""#, ""),
        golden_str(&["codex-acp", "launch"])
    );
    assert_eq!(
        codex.launch_env()["INITIAL_AGENT_MODE"],
        "agent-full-access"
    );
    assert_eq!(codex.model.format, ModelFormat::ModelEffort);
    assert_eq!(codex.model.select, ModelSelect::SetModel);
}

#[test]
fn gemini_install_and_launch_match_0_8() {
    let gemini = builtin("gemini");
    assert_eq!(gemini.install_script(), golden_str(&["gemini", "install"]));
    assert_eq!(gemini.launch_command(), golden_str(&["gemini", "launch"]));
    assert_eq!(
        gemini.launch_command(),
        "/opt/benchflow/bin/gemini --acp --yolo --skip-trust"
    );
    assert_eq!(gemini.default_model(), Some("gemini-2.5-flash"));
}

#[test]
fn opencode_keeps_0_8_install_but_launches_without_the_gateway_wrapper() {
    let opencode = builtin("opencode");
    assert_eq!(opencode.launch_command(), "/opt/benchflow/bin/opencode acp");
    let InstallMethod::Npm(npm) = &opencode.install.method else {
        panic!()
    };
    let launcher = install::launcher_script(npm);
    // A native bin runs directly, a Node script on BenchFlow's Node (what
    // 0.8's opencode-proxy wrapper did outside proxy mode).
    assert_eq!(
        launcher,
        "#!/bin/sh\n\
         if [ \"$(head -c2 /opt/benchflow/js-agents/bin/opencode 2>/dev/null)\" = \"#!\" ]; then\n  \
         exec /opt/benchflow/node/bin/node /opt/benchflow/js-agents/bin/opencode \"$@\"\n\
         else\n  \
         PATH=\"/opt/benchflow/node/bin:$PATH\" exec /opt/benchflow/js-agents/bin/opencode \"$@\"\n\
         fi\n"
    );
    // Everything but the launcher is 0.8's install.
    let ours = opencode.install_script();
    let theirs = golden_str(&["opencode_js_install"]);
    let cut = |s: &str| {
        let start = s.find("printf '%s\\n'").unwrap();
        let end = s.find(" > /opt/benchflow/bin/opencode").unwrap();
        format!("{}{}", &s[..start], &s[end..])
    };
    assert_eq!(cut(&ours), cut(&theirs));
}

#[test]
fn an_unpinned_package_installs_latest_unless_present_as_0_8_did() {
    let manifest = parse(
        r#"
name = "some-agent"
[install]
npm = "some-agent-pkg"
bin = "some-agent"
"#,
    )
    .unwrap();
    assert_eq!(manifest.install_script(), golden_str(&["unpinned_example"]));
}

#[test]
fn launcher_scripts_match_0_8_for_node_agents() {
    for (name, key) in [("codex", "codex-acp"), ("gemini", "gemini")] {
        let manifest = builtin(name);
        let InstallMethod::Npm(npm) = &manifest.install.method else {
            panic!()
        };
        assert_eq!(
            install::launcher_script(npm),
            golden_str(&["launchers", key]),
            "{name}"
        );
    }
}

#[test]
fn builtin_settings_follow_0_8() {
    let claude = builtin("claude");
    assert_eq!(claude.skill_paths, ["$HOME/.claude/skills"]);
    assert_eq!(claude.home_dirs, [".claude"]);
    assert_eq!(claude.credentials, CredentialFamily::Anthropic);
    assert_eq!(claude.model.select, ModelSelect::ConfigOption);
    assert_eq!(claude.model.config_id, "model");
    assert_eq!(claude.model.effort_config_id.as_deref(), Some("effort"));
    assert!(claude.acp.typed_failures && claude.acp.subagent_transcript);
    assert_eq!(claude.install.timeout, Duration::from_secs(900));
    assert_eq!(builtin("codex").skill_paths, ["$HOME/.agents/skills"]);
    assert_eq!(builtin("codex").credentials, CredentialFamily::OpenAi);
    assert!(!builtin("codex").acp.typed_failures);
    assert_eq!(builtin("gemini").credentials, CredentialFamily::Gemini);
    assert_eq!(builtin("opencode").credentials, CredentialFamily::Model);
    assert_eq!(builtin("opencode").model.format, ModelFormat::ProviderModel);
}

#[test]
fn agents_that_find_skills_their_own_way_take_them() {
    for (name, path, via) in [
        ("kimi-code", "$HOME/.kimi-code/skills", true),
        ("mistral-vibe", "$HOME/.vibe/skills", true),
        ("eve", "$HOME/.benchflow/eve-skills", true),
        ("muse-code", "$HOME/.config/muse/skills", false),
    ] {
        let m = builtin(name);
        assert_eq!(m.skill_paths, [path], "{name}");
        assert_eq!(m.skills_via.is_some(), via, "{name}");
    }
    let kimi = builtin("kimi-code");
    let command = &kimi.headless.as_ref().unwrap().command;
    assert!(command.contains(r#"${BENCHFLOW_SKILLS_DIR:+--skills-dir "$BENCHFLOW_SKILLS_DIR"}"#));
    let vibe = builtin("mistral-vibe");
    for command in [
        vibe.launch.command.as_deref().unwrap(),
        &vibe.headless.as_ref().unwrap().command,
    ] {
        assert!(command.contains("skill_paths = "), "{command}");
    }
}

#[test]
fn skill_dirs_expand_home_and_workspace() {
    let manifest = parse(&format!(
        "skill_paths = [\"$HOME/.claude/skills\", \"$WORKSPACE/.agents/skills\"]\n{MINIMAL}"
    ))
    .unwrap();
    assert_eq!(
        manifest.skill_dirs("/home/agent/", "/app"),
        ["/home/agent/.claude/skills", "/app/.agents/skills"]
    );
}

#[test]
fn fingerprints_follow_the_installed_bits() {
    let claude = builtin("claude");
    let codex = builtin("codex");
    assert_ne!(claude.version_fingerprint(), codex.version_fingerprint());
    assert_eq!(
        claude.version_fingerprint(),
        builtin("claude-code").version_fingerprint()
    );

    // Launch settings do not change the image; a pin does.
    let mut relaunched = claude.clone();
    relaunched.launch.env.insert("X".into(), "1".into());
    assert_eq!(
        relaunched.version_fingerprint(),
        claude.version_fingerprint()
    );
    let bumped = AgentManifest::parse(
        &include_str!("../../agents/claude.toml")
            .replace("claude-agent-acp@0.81.2", "claude-agent-acp@0.81.3"),
        ManifestSource::Builtin("claude.toml"),
    )
    .unwrap();
    assert_ne!(bumped.version_fingerprint(), claude.version_fingerprint());
}

#[test]
fn a_script_install_runs_as_given() {
    let manifest = parse(
        r#"
name = "native"
[install]
script = """
set -e
curl -fsSL https://example.com/install.sh | sh
"""
timeout = 1200
[launch]
command = "/usr/local/bin/native-acp --stdio"
"#,
    )
    .unwrap();
    assert_eq!(
        manifest.install_script(),
        "set -e\ncurl -fsSL https://example.com/install.sh | sh\n"
    );
    assert_eq!(
        manifest.launch_command(),
        "/usr/local/bin/native-acp --stdio"
    );
    assert_eq!(manifest.install.timeout, Duration::from_secs(1200));
}

#[test]
fn default_launch_appends_quoted_args() {
    let manifest = parse(&format!(
        "{MINIMAL}\n[launch]\nargs = [\"acp\", \"--name\", \"two words\"]\n"
    ))
    .unwrap();
    assert_eq!(
        manifest.launch_command(),
        "/opt/benchflow/bin/my-agent acp --name 'two words'"
    );
    assert_eq!(manifest.credentials, CredentialFamily::None);
    assert_eq!(manifest.model.select, ModelSelect::Auto);
    assert_eq!(manifest.model.default, None);
}

#[test]
fn unknown_keys_name_the_line_and_the_likely_key() {
    let err = parse(&format!("{MINIMAL}timout = 5\n")).unwrap_err();
    // `timout` lands in [install], the table open at that point.
    let ManifestError::UnknownKey(details) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(details.key, "install.timout");
    assert_eq!((details.line, details.column), (7, 1));
    assert_eq!(details.suggestion.as_deref(), Some("timeout"));
    assert_eq!(details.table, "[install]");
    assert_eq!(
        err.to_string(),
        "agents/test.toml:7:1: unknown key `install.timout`; did you mean `timeout`? \
         ([install] takes npm, bin, companions, launcher, launcher_env, verify, script, uv-tool, \
         uv-venv, python, installer, links, timeout, binary, npx, uvx)"
    );

    let err = parse(&MINIMAL.replace("name = ", "nmae = ")).unwrap_err();
    assert!(
        err.to_string()
            .contains("unknown key `nmae`; did you mean `name`?"),
        "{err}"
    );

    let err = parse(&format!("{MINIMAL}\n[modle]\nselect = \"auto\"\n")).unwrap_err();
    assert!(
        err.to_string()
            .contains("unknown key `modle`; did you mean `model`?"),
        "{err}"
    );

    // Free-form tables take any variable name.
    parse(&format!("{MINIMAL}\n[launch.env]\nANY_NAME = \"1\"\n")).unwrap();
}

#[test]
fn type_and_syntax_errors_have_positions() {
    let err = parse(&format!("{MINIMAL}timeout = \"soon\"\n")).unwrap_err();
    let ManifestError::Syntax { line, message, .. } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(*line, 7);
    assert!(message.contains("invalid type"), "{message}");

    let err = parse("name = \"x\"\n[install\n").unwrap_err();
    assert!(
        matches!(err, ManifestError::Syntax { line: 2, .. }),
        "{err:?}"
    );

    let err = parse("name = \"x\"\n").unwrap_err();
    assert!(err.to_string().contains("missing field `install`"), "{err}");
}

#[test]
fn invalid_values_are_refused_with_the_fix() {
    let cases: &[(&str, &str)] = &[
        (
            &MINIMAL.replace("my-agent\"\n\n", "My_Agent\"\n\n"),
            "`name`: `My_Agent` is not a valid agent name",
        ),
        (
            &MINIMAL.replace("name = \"my-agent\"", "name = \"oracle\""),
            "reserved for a scripted agent",
        ),
        (
            &MINIMAL.replace("bin = \"my-agent\"\n", ""),
            "`bin` is required with `npm`",
        ),
        (
            &format!("{MINIMAL}script = \"x\"\n"),
            "use either `npm` or `script`",
        ),
        (
            "name = \"x\"\n[install]\nscript = \"true\"\n",
            "`command` is required when the agent is installed with `script`",
        ),
        (
            "name = \"x\"\n[install]\nscript = \"true\"\nbin = \"x\"\n[launch]\ncommand = \"x\"\n",
            "only applies to `npm` or `uv-tool` installs",
        ),
        (
            &format!("{MINIMAL}\n[launch]\ncommand = \"x\"\nargs = [\"y\"]\n"),
            "put the arguments in `command`",
        ),
        (
            &format!("{MINIMAL}\n[credentials]\nfamily = \"antropic\"\n"),
            "did you mean `anthropic`?",
        ),
        (
            &format!("{MINIMAL}\n[credentials]\nfamily = \"openai\"\nenv = [\"X\"]\n"),
            "only read with `family = \"env\"`",
        ),
        (
            &format!("{MINIMAL}\n[model]\nconfig_id = \"m\"\n"),
            "only applies with `select = \"config-option\"`",
        ),
        (
            &format!("{MINIMAL}\n[model]\nformat = \"model[effort]\"\n"),
            "set `select = \"set-model\"`",
        ),
        (
            &format!("skill_paths = [\"/abs\"]\n{MINIMAL}"),
            "must start with `$HOME/` or `$WORKSPACE/`",
        ),
        (
            &format!("skills_via = \"a flag\"\n{MINIMAL}"),
            "needs `skill_paths`",
        ),
        (
            &format!("home_dirs = [\"../x\"]\n{MINIMAL}"),
            "must be a folder under the home directory",
        ),
        (
            &format!("aliases = [\"my-agent\"]\n{MINIMAL}"),
            "listed twice",
        ),
        (
            &MINIMAL.replace("my-agent@1.0.0", "my agent"),
            "is not an npm package spec",
        ),
        (&format!("{MINIMAL}timeout = 0\n"), "at least 1 second"),
        (
            &format!("{MINIMAL}\n[launch.env]\n\"NOT-A-NAME\" = \"1\"\n"),
            "not an environment variable name",
        ),
    ];
    for (text, expected) in cases {
        let err = parse(text).expect_err(text);
        assert!(
            err.to_string().contains(expected),
            "{err}\n--- expected: {expected}"
        );
    }
}

#[test]
fn invalid_value_errors_point_at_the_value() {
    let err = parse(&format!("{MINIMAL}\n[model]\nformat = \"model[effort]\"\n")).unwrap_err();
    let ManifestError::Invalid { key, line, .. } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(key, "model.format");
    assert_eq!(*line, 9);
    let err = parse(&format!("aliases = [\"ok\", \"Bad\"]\n{MINIMAL}")).unwrap_err();
    assert!(
        err.to_string()
            .starts_with("agents/test.toml:1:18: `aliases[1]`"),
        "{err}"
    );
}

#[test]
fn env_credentials_list_the_variables() {
    let manifest = parse(&format!(
        "{MINIMAL}\n[credentials]\nenv = [\"MISTRAL_API_KEY\"]\n"
    ))
    .unwrap();
    assert_eq!(
        manifest.credentials,
        CredentialFamily::Env(vec!["MISTRAL_API_KEY".into()])
    );
    let err = parse(&format!("{MINIMAL}\n[credentials]\nfamily = \"env\"\n")).unwrap_err();
    assert!(err.to_string().contains("list the variables"), "{err}");
}

#[test]
fn registry_finds_names_and_aliases_and_suggests() {
    let registry = Registry::builtin();
    let err = registry.find("cluade").unwrap_err();
    assert_eq!(err.suggestion.as_deref(), Some("claude"));
    let names: Vec<&str> = registry.agents().iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        err.to_string(),
        format!(
            "unknown agent `cluade`; did you mean `claude`? (agents: {}, \
             and the scripted oracle and nop; `bench agents` describes them)",
            names.join(", ")
        )
    );
    assert_eq!(registry.find("zzzz").unwrap_err().suggestion, None);
    // 0.8 spellings resolve.
    assert_eq!(registry.find("codex-acp").unwrap().name, "codex");
}

#[test]
fn project_manifests_add_and_replace_agents() {
    let dir = tempfile::tempdir().unwrap();
    let mine = dir.path().join("mine.toml");
    std::fs::write(&mine, MINIMAL).unwrap();
    let claude_next = dir.path().join("claude.toml");
    std::fs::write(
        &claude_next,
        include_str!("../../agents/claude.toml").replace("0.81.2", "0.81.9"),
    )
    .unwrap();
    std::fs::write(dir.path().join("notes.txt"), "not a manifest").unwrap();

    let registry = Registry::with_manifests([dir.path()]).unwrap();
    // The built-ins plus one; claude was replaced, not added.
    assert_eq!(
        registry.agents().len(),
        Registry::builtin().agents().len() + 1
    );
    let claude = registry.find("claude-agent-acp").unwrap();
    assert!(claude.install_script().contains("claude-agent-acp@0.81.9"));
    assert_eq!(claude.source, ManifestSource::File(claude_next.clone()));
    assert_eq!(
        registry.find("my-agent").unwrap().source.path(),
        Some(mine.as_path())
    );

    // A single file works too.
    let registry = Registry::with_manifests([&mine]).unwrap();
    assert!(registry.find("my-agent").is_ok());
}

#[test]
fn clashing_aliases_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("thief.toml");
    std::fs::write(&path, format!("aliases = [\"codex-acp\"]\n{MINIMAL}")).unwrap();
    let err = Registry::with_manifests([&path]).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!(
            "agent name `codex-acp` is claimed twice: by built-in agents/codex.toml and by {}; rename one of them",
            path.display()
        )
    );
    let missing = dir.path().join("missing.toml");
    let err = Registry::with_manifests([&missing]).unwrap_err();
    assert!(
        err.to_string().starts_with("cannot read agent manifest"),
        "{err}"
    );
}

#[test]
fn optional_tables_take_any_subset_of_their_keys() {
    let manifest = parse(&format!(
        "{MINIMAL}\n[acp]\ntyped_failures = true\n\n[model]\ndefault = \"m\"\n"
    ))
    .unwrap();
    assert!(manifest.acp.typed_failures && !manifest.acp.subagent_transcript);
    assert_eq!(manifest.model.default.as_deref(), Some("m"));
    assert_eq!(manifest.model.config_id, "model");
}

const HEADLESS_ONLY: &str = r#"
name = "my-cli"
[install]
uv-tool = "my-cli==1.2.3"
bin = "my-cli"
python = "3.12"
[headless]
command = 'my-cli --yes --message-file "$BENCHFLOW_INSTRUCTION_FILE"'
output = "text"
env = { MY_CLI_NO_COLOR = "1" }
[[headless.files]]
path = ".my-cli/config.json"
contents = "{}"
mode = 0o600
"#;

#[test]
fn a_headless_only_agent_declares_only_headless() {
    let m = parse(HEADLESS_ONLY).unwrap();
    assert_eq!(m.harness, Harness::Headless);
    assert_eq!(m.harnesses, [Harness::Headless]);
    let headless = m.headless.as_ref().unwrap();
    assert_eq!(headless.output, crate::headless::OutputFormat::Text);
    assert_eq!(headless.files[0].path, ".my-cli/config.json");
    assert_eq!(headless.files[0].mode, 0o600);
    assert!(
        m.pick_harness(Some(Harness::Acp))
            .unwrap_err()
            .contains("has no acp harness")
    );
    let script = m.install_script();
    assert!(
        script.contains("uv tool install --python 3.12 my-cli==1.2.3"),
        "{script}"
    );
    assert!(
        script.contains("UV_TOOL_BIN_DIR=/opt/benchflow/bin"),
        "{script}"
    );
    assert!(
        script.contains("[ -x /opt/benchflow/bin/my-cli ]"),
        "{script}"
    );
}

#[test]
fn an_npm_agent_with_headless_keeps_acp_as_its_default() {
    let m = parse(&format!(
        "{MINIMAL}\n[launch]\nargs = [\"--acp\"]\n[headless]\ncommand = \"my-agent -p\"\noutput = \"claude-stream-json\"\n"
    ))
    .unwrap();
    assert_eq!(m.harness, Harness::Acp);
    assert_eq!(m.harnesses, [Harness::Acp, Harness::Headless]);
    assert_eq!(
        m.pick_harness(Some(Harness::Headless)),
        Ok(Harness::Headless)
    );
    let m = parse(&format!(
        "harness = \"headless\"\n{MINIMAL}\n[launch]\n[headless]\ncommand = \"x\"\noutput = \"text\"\n"
    ))
    .unwrap();
    assert_eq!(m.harness, Harness::Headless);
    // Without [launch], a [headless] table makes the agent headless-only.
    let m = parse(&format!(
        "{MINIMAL}\n[headless]\ncommand = \"x\"\noutput = \"text\"\n"
    ))
    .unwrap();
    assert_eq!(m.harnesses, [Harness::Headless]);
}

#[test]
fn an_installer_runs_in_its_own_home_and_links_its_binaries() {
    let m = parse(
        r#"
name = "fxlike"
[install]
installer = "curl -fsSL https://example.com/setup.sh | bash -s -- v1.0.0"
links = [".local/bin/fx"]
verify = "/opt/benchflow/bin/fx --version"
[headless]
command = "fx ask --json"
output = "fx-json"
"#,
    )
    .unwrap();
    let script = m.install_script();
    assert!(
        script.contains("export HOME=/opt/benchflow/agents/fxlike;"),
        "{script}"
    );
    assert!(
        script.contains("ln -sf /opt/benchflow/agents/fxlike/.local/bin/fx /opt/benchflow/bin/fx"),
        "{script}"
    );
    assert!(
        script.ends_with("&& /opt/benchflow/bin/fx --version"),
        "{script}"
    );
}

#[test]
fn headless_and_install_mistakes_say_what_to_write() {
    let cases: &[(&str, &str)] = &[
        (
            "name = \"x\"\n[install]\nuv-tool = \"aider-chat\"\nbin = \"aider\"\n[headless]\ncommand = \"x\"\noutput = \"text\"\n",
            "write `<package>==<version>`",
        ),
        (
            "name = \"x\"\n[install]\nuv-tool = \"a==1\"\n[headless]\ncommand = \"x\"\noutput = \"text\"\n",
            "`bin` is required with `uv-tool`",
        ),
        (
            "name = \"x\"\n[install]\ninstaller = \"true\"\n[headless]\ncommand = \"x\"\noutput = \"text\"\n",
            "`links` is required with `installer`",
        ),
        (
            "name = \"x\"\n[install]\ninstaller = \"true\"\nlinks = [\"../bin/x\"]\n[headless]\ncommand = \"x\"\noutput = \"text\"\n",
            "must be a file under the installer's home",
        ),
        (
            "name = \"x\"\n[install]\ninstaller = \"true\"\nlinks = [\"x\"]\npython = \"3.12\"\n[headless]\ncommand = \"x\"\noutput = \"text\"\n",
            "only applies to `uv-tool`, `uv-venv`, or `uvx` installs",
        ),
        (
            &format!("{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"stream-json\"\n"),
            "unknown output format `stream-json`",
        ),
        (
            &format!("{MINIMAL}[headless]\noutput = \"text\"\n"),
            "`command` is required",
        ),
        (
            &format!("{MINIMAL}[headless]\ncommand = \"x\"\n"),
            "`output` is required",
        ),
        (
            &format!(
                "{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"text\"\nenv = {{ BENCHFLOW_MODEL = \"y\" }}\n"
            ),
            "is set by BenchFlow for every headless run",
        ),
        (
            &format!(
                "{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"text\"\n[[headless.files]]\npath = \"/etc/passwd\"\n"
            ),
            "must be a file under the agent's home",
        ),
        (
            &format!(
                "{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"text\"\n[[headless.files]]\npath = \"a\"\nmode = 0o4755\n"
            ),
            "write it in octal",
        ),
        (
            &format!("harness = \"headless\"\n{MINIMAL}"),
            "`headless` needs a `[headless]` table",
        ),
        (
            &format!(
                "harness = \"acp\"\n{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"text\"\n"
            ),
            "`acp` needs a `[launch]` table",
        ),
        (
            &format!("harness = \"headles\"\n{MINIMAL}"),
            "did you mean `headless`?",
        ),
    ];
    for (text, expected) in cases {
        let err = parse(text).expect_err(text);
        assert!(
            err.to_string().contains(expected),
            "{err}\n--- expected: {expected}"
        );
    }
    let err = parse(&format!(
        "{MINIMAL}[headless]\ncommand = \"x\"\noutput = \"text\"\n[[headless.files]]\npath = \"a\"\nmod = 1\n"
    ))
    .unwrap_err();
    assert!(
        err.to_string().contains("unknown key `headless.files[0].mod`; did you mean `mode`? (each [[headless.files]] takes path, contents, runner, mode)"),
        "{err}"
    );
}

#[test]
fn builtin_headless_commands_read_the_instruction_file_and_keep_acp_defaults() {
    for m in Registry::builtin().agents() {
        // Agents with an ACP route keep it as their default.
        let expected_default = if HEADLESS_ONLY_BUILTINS.contains(&m.name.as_str()) {
            Harness::Headless
        } else {
            Harness::Acp
        };
        assert_eq!(m.harness, expected_default, "{}", m.name);
        let Some(headless) = m.headless.as_ref() else {
            continue;
        };
        assert!(
            headless.command.contains("$BENCHFLOW_INSTRUCTION_FILE"),
            "{}: the instruction must come from the file",
            m.name
        );
        assert!(
            !headless.command.contains("$BENCHFLOW_INSTRUCTION\""),
            "{}",
            m.name
        );
        // Adding a headless harness never changes what is installed.
        if expected_default == Harness::Acp {
            assert!(m.declares(Harness::Headless));
        }
    }
    // goose and aider take models.dev ids; fx takes gateway ids.
    let goose = builtin("goose");
    assert_eq!(
        goose.cli_model("claude-sonnet-4-6"),
        "anthropic/claude-sonnet-4-6"
    );
    assert_eq!(
        builtin("fx").cli_model("vercel/anthropic/claude-sonnet-4.6"),
        "anthropic/claude-sonnet-4.6"
    );
    assert_eq!(
        builtin("codex").cli_model("openai/gpt-6.1-sol"),
        "gpt-6.1-sol"
    );
}

/// Built-in agents with no ACP route: the headless harness is their only one.
const HEADLESS_ONLY_BUILTINS: &[&str] = &[
    "aider",
    "fx",
    "fx-dev",
    "mini-swe-agent",
    "swe-agent",
    "trae-agent",
    "hermes",
    "openclaw",
    "kimi-code",
    "mcode",
    "code-puppy",
    "rovodev-cli",
    "devin",
    "muse-code",
    "openhands-sdk",
    "nemo-agent",
    "strands",
    "deerflow",
    "antigravity-sdk",
    "langgraph",
    "eve",
];

/// Every ACP-capable agent of the agent catalog (rust-rewrite-1010
/// agents-catalog.md) that is not one of 0.8's first four, with Harbor's,
/// Karotte's, and 0.8's names as aliases, and the ACP registry id it runs.
const CATALOG_ACP_AGENTS: &[(&str, &[&str], Option<&str>)] = &[
    ("goose", &[], Some("goose")),
    ("kimi", &["kimi-cli"], Some("kimi")),
    ("qwen-code", &["qwen-coder"], Some("qwen-code")),
    ("cline", &["cline-cli"], Some("cline")),
    ("pi", &["pi-acp"], Some("pi-acp")),
    ("mistral-vibe", &["vibe"], Some("mistral-vibe")),
    ("grok", &["grok-build"], Some("grok-build")),
    ("deepagents", &[], Some("deepagents")),
    ("mimo", &[], None),
    ("openhands", &["oh"], None),
    ("cursor", &["cursor-cli"], Some("cursor")),
    (
        "copilot",
        &["copilot-cli", "github-copilot-cli"],
        Some("github-copilot-cli"),
    ),
    ("junie", &[], Some("junie")),
    ("cortex-code", &["cortex"], Some("cortex-code")),
    (
        "antigravity",
        &["agy", "antigravity-cli"],
        Some("antigravity-acp"),
    ),
];

#[test]
fn every_catalog_acp_agent_is_built_in_and_valid() {
    let registry = Registry::builtin();
    assert_eq!(
        registry.agents().len(),
        4 + CATALOG_ACP_AGENTS.len() + HEADLESS_ONLY_BUILTINS.len()
    );
    for (name, aliases, registry_id) in CATALOG_ACP_AGENTS {
        let m = registry.find(name).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&m.name, name);
        for alias in *aliases {
            assert_eq!(&registry.find(alias).unwrap().name, name, "{alias}");
        }
        assert_eq!(m.acp_registry.as_deref(), *registry_id, "{name}");
        assert!(!m.description.is_empty(), "{name}");
        // Installs are pinned: the script is the same every time, and it
        // names an exact version.
        let script = m.install_script();
        assert!(!script.contains("@latest"), "{name} is pinned");
        assert_eq!(m.version_fingerprint(), m.version_fingerprint());
        let out = std::process::Command::new("sh")
            .args(["-n", "-c", &script])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(m.launch_command().contains("/opt/benchflow/bin/"), "{name}");
        for path in &m.skill_paths {
            assert!(path.starts_with("$HOME/") || path.starts_with("$WORKSPACE/"));
        }
    }
    // The four 0.8 agents name their registry entries too.
    for (name, id) in [
        ("claude", "claude-acp"),
        ("codex", "codex-acp"),
        ("gemini", "gemini"),
        ("opencode", "opencode"),
    ] {
        assert_eq!(
            registry.find(name).unwrap().acp_registry.as_deref(),
            Some(id)
        );
    }
}

#[test]
fn agents_that_need_a_vendor_account_say_which() {
    let registry = Registry::builtin();
    for (name, vendor) in [
        ("cursor", "Cursor account"),
        ("copilot", "GitHub Copilot subscription"),
        ("junie", "JetBrains AI account"),
        ("cortex-code", "Snowflake account"),
        ("antigravity", "Gemini API key"),
        ("kimi", "Moonshot AI"),
        ("grok", "xAI API key"),
        ("mistral-vibe", "Mistral API key"),
    ] {
        let account = registry.find(name).unwrap().account.clone();
        assert!(
            account.as_deref().is_some_and(|a| a.contains(vendor)),
            "{name}: {account:?}"
        );
    }
    assert_eq!(registry.find("goose").unwrap().account, None);
}

#[test]
fn registry_distributions_in_built_ins_launch_the_real_path() {
    let registry = Registry::builtin();
    let InstallMethod::Binary(b) = &registry.find("cortex-code").unwrap().install.method else {
        panic!("cortex-code installs its release archive")
    };
    assert_eq!(
        b.targets["linux-aarch64"].cmd,
        "coco-1.0.73+180523.e6179a031de9-linux-arm64/cortex"
    );
    let goose = registry.find("goose").unwrap();
    assert!(goose.install_script().contains("bf_sha=bf01cb6359f7"));
    let InstallMethod::Uvx(u) = &registry.find("openhands").unwrap().install.method else {
        panic!("openhands installs from PyPI with uv")
    };
    assert_eq!(
        (u.package.as_str(), u.bin.as_str()),
        ("openhands==1.16.0", "openhands")
    );
}

const SDK_AGENT: &str = r#"
name = "my-sdk-agent"
[install]
uv-venv = ["my-framework[openai]==2.0.1", "helper @ https://github.com/o/r/archive/0123456789abcdef0123456789abcdef01234567.tar.gz#subdirectory=pkg"]
python = "3.12"
verify = "/opt/benchflow/agents/my-sdk-agent/venv/bin/python -c 'import my_framework'"
[headless]
command = 'exec /opt/benchflow/agents/my-sdk-agent/venv/bin/python "$HOME/.benchflow/runners/run_strands.py" "$BENCHFLOW_INSTRUCTION_FILE"'
output = "benchflow-runner"
[[headless.files]]
path = ".benchflow/runners/bf_events.py"
runner = "bf_events.py"
[[headless.files]]
path = ".benchflow/runners/run_strands.py"
runner = "run_strands.py"
[credentials]
env = ["MY_KEY"]
account = "a Vendor account: create a key in the vendor's console"
"#;

#[test]
fn a_uv_venv_agent_gets_its_own_venv_and_runner_files() {
    let m = parse(SDK_AGENT).unwrap();
    assert_eq!(m.harnesses, [Harness::Headless]);
    let script = m.install_script();
    assert!(
        script.contains("/opt/benchflow/uv/uv venv --clear --python 3.12 /opt/benchflow/agents/my-sdk-agent/venv"),
        "{script}"
    );
    assert!(
        script.contains(
            "uv pip install --python /opt/benchflow/agents/my-sdk-agent/venv/bin/python 'my-framework[openai]==2.0.1' 'helper @ https://github.com/o/r/archive/"
        ),
        "{script}"
    );
    assert!(
        script.ends_with(
            "&& /opt/benchflow/agents/my-sdk-agent/venv/bin/python -c 'import my_framework'"
        ),
        "{script}"
    );
    // The runner files carry the embedded scripts.
    let files = &m.headless.as_ref().unwrap().files;
    assert_eq!(
        files[0].contents,
        crate::runners::get("bf_events.py").unwrap()
    );
    assert_eq!(files[1].path, ".benchflow/runners/run_strands.py");
    assert!(files[1].contents.contains("import bf_events"));
    assert_eq!(files[1].mode, 0o644);
    assert_eq!(
        m.account.as_deref(),
        Some("a Vendor account: create a key in the vendor's console")
    );
}

#[test]
fn a_vendor_account_is_described_and_is_the_fix_for_no_login() {
    let manifest = parse(&format!(
        "{MINIMAL}\n[credentials]\nenv = [\"ACME_TOKEN\"]\naccount = \"an Acme account: `export ACME_TOKEN=<token>`\"\n"
    ))
    .unwrap();
    assert_eq!(
        manifest.account.as_deref(),
        Some("an Acme account: `export ACME_TOKEN=<token>`")
    );
    let host = crate::credentials::HostEnv::new(Default::default(), None);
    let err = crate::credentials::find_login(&manifest, None, &host).unwrap_err();
    assert_eq!(
        err.next_step(),
        "this agent needs an Acme account: `export ACME_TOKEN=<token>`"
    );
    let err = parse(&format!(
        "{MINIMAL}\n[credentials]\nenv = [\"ACME_TOKEN\"]\naccount = \" \"\n"
    ))
    .unwrap_err();
    assert!(err.to_string().contains("credentials.account"), "{err}");
    // The built-in agents that need a vendor's account say which.
    for (name, vendor) in [
        ("rovodev-cli", "Atlassian"),
        ("devin", "Cognition"),
        ("muse-code", "Meta"),
    ] {
        let account = builtin(name).account.unwrap_or_default();
        assert!(account.contains(vendor), "{name}: {account}");
    }
}

#[test]
fn trajectory_formats_bring_their_follower() {
    use crate::headless::OutputFormat;
    for name in ["mini-swe-agent", "swe-agent", "trae-agent", "openclaw"] {
        let m = builtin(name);
        let headless = m.headless.as_ref().unwrap();
        let files = headless.output.helper_files();
        assert_eq!(files.len(), 1, "{name}");
        let (path, contents) = files[0];
        assert!(headless.command.contains(path), "{name} runs {path}");
        assert!(contents.contains("BenchFlow: run an agent"), "{name}");
    }
    assert!(OutputFormat::Text.helper_files().is_empty());
}

#[test]
fn sdk_install_and_runner_mistakes_say_what_to_write() {
    let header = "name = \"x\"\n[install]\n";
    let headless = "[headless]\ncommand = \"x\"\noutput = \"benchflow-runner\"\n";
    let cases: &[(&str, &str)] = &[
        (
            "uv-venv = [\"strands-agents\"]\n",
            "`strands-agents` is not pinned",
        ),
        (
            "uv-venv = [\"pkg @ https://github.com/o/r/archive/main.tar.gz\"]\n",
            "with a 40-character commit in the URL",
        ),
        ("uv-venv = []\n", "lists no packages"),
        (
            "uv-venv = [\"a==1\"]\nbin = \"a\"\n",
            "only applies to `npm` or `uv-tool` installs",
        ),
        (
            "uv-venv = [\"a==1\"]\npython = \"three\"\n",
            "is not a Python version",
        ),
    ];
    for (install, want) in cases {
        let err = parse(&format!("{header}{install}{headless}")).unwrap_err();
        assert!(err.to_string().contains(want), "{install}: {err}");
    }
    let err = parse(&format!(
        "{header}uv-venv = [\"a==1\"]\n{headless}[[headless.files]]\npath = \"r.py\"\nrunner = \"run_strand.py\"\n"
    ))
    .unwrap_err();
    assert!(
        err.to_string().contains("`runner = \"run_strand.py\"` is not one of BenchFlow's runner scripts; did you mean `run_strands.py`?"),
        "{err}"
    );
    let err = parse(&format!(
        "{header}uv-venv = [\"a==1\"]\n{headless}[[headless.files]]\npath = \"r.py\"\nrunner = \"run_strands.py\"\ncontents = \"x\"\n"
    ))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("use either `contents` or `runner`"),
        "{err}"
    );
    // A uv-venv install has no executable to launch over ACP.
    let err = parse(&format!("{header}uv-venv = [\"a==1\"]\n[launch]\n")).unwrap_err();
    assert!(err.to_string().contains("`command` is required"), "{err}");
}

#[test]
fn sdk_agents_run_a_runner_in_their_venv() {
    let python = [
        ("openhands-sdk", "run_openhands_sdk.py"),
        ("nemo-agent", "run_nemo_agent.py"),
        ("strands", "run_strands.py"),
        ("deerflow", "run_deerflow.py"),
        ("antigravity-sdk", "run_antigravity_sdk.py"),
        ("langgraph", "run_langgraph.py"),
    ];
    for (name, runner) in python {
        let m = builtin(name);
        let InstallMethod::UvVenv(venv) = &m.install.method else {
            panic!("{name} is not a uv-venv install")
        };
        assert!(
            venv.requirements
                .iter()
                .all(|r| r.contains("==") || r.contains(" @ https://")),
            "{name}"
        );
        let headless = m.headless.as_ref().unwrap();
        assert_eq!(
            headless.output,
            crate::headless::OutputFormat::BenchflowRunner
        );
        assert!(
            headless.command.starts_with(&format!(
                "exec /opt/benchflow/agents/{name}/venv/bin/python -u \"$HOME/.benchflow/runners/{runner}\""
            )),
            "{name}: {}",
            headless.command
        );
        let paths: Vec<&str> = headless.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                ".benchflow/runners/bf_events.py".to_string(),
                format!(".benchflow/runners/{runner}")
            ],
            "{name}"
        );
        assert_eq!(m.model.select, ModelSelect::None, "{name}");
    }
    let eve = builtin("eve");
    assert!(matches!(eve.install.method, InstallMethod::Installer(_)));
    assert!(
        eve.install_script()
            .contains("node-v24.21.0-linux-$arch.tar.gz")
    );
    assert_eq!(
        eve.headless.as_ref().unwrap().files[0].path,
        ".benchflow/runners/run_eve.mjs"
    );
    // antigravity-sdk says it needs a Google account wherever a login is missing.
    let agy = builtin("antigravity-sdk");
    assert!(
        agy.account
            .as_deref()
            .unwrap()
            .starts_with("a Google account")
    );
    assert!(agy.description.contains("Google account"));
    assert_eq!(
        agy.credentials,
        CredentialFamily::Env(vec!["GEMINI_API_KEY".into(), "GOOGLE_API_KEY".into()])
    );
    assert_eq!(
        builtin("nemo-agent").default_model(),
        Some("nvidia/nvidia/nemotron-3-super-120b-a12b")
    );
    assert_eq!(builtin("nvidia-nat").name, "nemo-agent");
}

/// Every agent that has both routes upstream (Harbor runs it headless, and
/// it speaks ACP) declares both, on one install, with ACP the default
/// except where Harbor's headless route is the agent's main one.
#[test]
fn agents_with_both_routes_upstream_declare_both() {
    use crate::headless::OutputFormat;
    let registry = Registry::builtin();
    let both = [
        ("cline", Harness::Acp, OutputFormat::ClineJson),
        ("cortex-code", Harness::Acp, OutputFormat::ClaudeStreamJson),
        ("cursor", Harness::Acp, OutputFormat::CursorStreamJson),
        (
            "antigravity",
            Harness::Acp,
            OutputFormat::AntigravityStreamJson,
        ),
        ("grok", Harness::Acp, OutputFormat::GrokStreamingJson),
        ("junie", Harness::Acp, OutputFormat::Text),
        ("openhands", Harness::Acp, OutputFormat::OpenhandsJson),
        ("kimi", Harness::Acp, OutputFormat::KimiStreamJson),
        ("mimo", Harness::Acp, OutputFormat::OpencodeJson),
        ("pi", Harness::Acp, OutputFormat::PiJson),
        ("qwen-code", Harness::Acp, OutputFormat::QwenSession),
        ("copilot", Harness::Acp, OutputFormat::CopilotJson),
        ("mistral-vibe", Harness::Acp, OutputFormat::VibeStreaming),
        ("hermes", Harness::Headless, OutputFormat::HermesSession),
        ("mcode", Harness::Headless, OutputFormat::McodeStreamJson),
        ("devin", Harness::Headless, OutputFormat::Text),
    ];
    for (name, default, output) in both {
        let m = registry.find(name).unwrap();
        assert_eq!(m.harness, default, "{name}");
        assert!(m.harnesses.contains(&Harness::Acp), "{name}");
        assert!(m.harnesses.contains(&Harness::Headless), "{name}");
        let headless = m.headless.as_ref().unwrap();
        assert_eq!(headless.output, output, "{name}");
        assert!(
            headless.command.contains("$BENCHFLOW_INSTRUCTION_FILE"),
            "{name} reads the instruction file"
        );
        // The commands parse as POSIX shell.
        for script in [headless.command.as_str(), m.launch_command().as_str()] {
            let out = std::process::Command::new("sh")
                .args(["-n", "-c", script])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{name}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

#[test]
fn claude_and_codex_load_their_own_session_files() {
    assert_eq!(
        builtin("claude").resume.load,
        Some(SessionFile::ClaudeJsonl)
    );
    assert_eq!(
        builtin("codex").resume.load,
        Some(SessionFile::CodexRollout)
    );
    for name in ["gemini", "opencode", "goose"] {
        assert_eq!(builtin(name).resume.load, None, "{name}");
    }
}

#[test]
fn session_files_are_checked_and_placed_where_the_cli_looks() {
    let id = "0b9c8a52-6f3e-4f0e-9d7a-2a5c1e7b4d10";
    let line = b"{\"type\":\"user\"}\n\n{\"type\":\"assistant\"}\n";
    let claude = SessionFile::ClaudeJsonl;
    assert_eq!(claude.check(&format!("{id}.jsonl"), line).unwrap(), id);
    assert_eq!(
        claude.place(&format!("{id}.jsonl"), "/app"),
        format!(".claude/projects/-app/{id}.jsonl")
    );
    assert_eq!(
        claude.place("x.jsonl", "/home/agent/my_task"),
        ".claude/projects/-home-agent-my-task/x.jsonl"
    );
    assert!(
        claude
            .check("session.jsonl", line)
            .unwrap_err()
            .contains("<session-id>.jsonl")
    );
    assert!(claude.check(&format!("{id}.json"), line).is_err());
    assert!(
        claude
            .check(&format!("{id}.jsonl"), b"{\"a\":1}\nnot json\n")
            .unwrap_err()
            .contains("line 2")
    );
    assert!(
        claude
            .check(&format!("{id}.jsonl"), b"\n")
            .unwrap_err()
            .contains("no JSON lines")
    );

    let codex = SessionFile::CodexRollout;
    let name = format!("rollout-2026-10-09T14-03-22-{id}.jsonl");
    assert_eq!(codex.check(&name, line).unwrap(), id);
    assert_eq!(
        codex.place(&name, "/app"),
        format!(".codex/sessions/2026/10/09/{name}")
    );
    for bad in [
        format!("{id}.jsonl"),
        format!("rollout-2026-10-09-{id}.jsonl"),
        "rollout-2026-10-09T14-03-22-nope.jsonl".to_string(),
    ] {
        assert!(codex.check(&bad, line).is_err(), "{bad}");
    }
}

#[test]
fn a_load_format_must_be_known_and_resumable() {
    let ok = parse(&format!(
        "{MINIMAL}\n[resume]\nacp = true\nsession_dirs = [\".x\"]\nload = \"claude-jsonl\"\n"
    ))
    .unwrap();
    assert_eq!(ok.resume.load, Some(SessionFile::ClaudeJsonl));
    let unknown = parse(&format!(
        "{MINIMAL}\n[resume]\nacp = true\nsession_dirs = [\".x\"]\nload = \"gemini-chat\"\n"
    ))
    .unwrap_err()
    .to_string();
    assert!(
        unknown.contains("known: claude-jsonl, codex-rollout"),
        "{unknown}"
    );
    let alone = parse(&format!("{MINIMAL}\n[resume]\nload = \"claude-jsonl\"\n"))
        .unwrap_err()
        .to_string();
    assert!(alone.contains("set `acp` or `headless` too"), "{alone}");
}

#[test]
fn resources_set_the_least_the_sandbox_gets() {
    let m = parse(&format!(
        "{MINIMAL}\n[resources]\nmin_memory = \"4 GB\"\nmin_cpus = 2\n"
    ))
    .unwrap();
    assert_eq!(
        m.resources,
        ResourceFloor {
            memory_mb: Some(4096),
            cpus: Some(2)
        }
    );
    assert_eq!(
        parse(&format!(
            "{MINIMAL}\n[resources]\nmin_memory = \"512 MB\"\n"
        ))
        .unwrap()
        .resources
        .memory_mb,
        Some(512)
    );
    assert!(parse(MINIMAL).unwrap().resources.is_empty());
    for bad in ["\"4\"", "\"four GB\"", "\"0 GB\"", "\"4 TB\""] {
        let err = parse(&format!("{MINIMAL}\n[resources]\nmin_memory = {bad}\n")).unwrap_err();
        let ManifestError::Invalid { key, .. } = &err else {
            panic!("{err:?}")
        };
        assert_eq!(key, "resources.min_memory", "{bad}");
    }
    assert!(parse(&format!("{MINIMAL}\n[resources]\nmin_cpus = 0\n")).is_err());
    assert!(parse(&format!("{MINIMAL}\n[resources]\nmemory = \"4 GB\"\n")).is_err());
}

#[test]
fn openclaw_asks_for_the_memory_it_needs() {
    // Killed (exit 137) at Daytona's default size in T7.1; passed with 4 GB.
    assert_eq!(builtin("openclaw").resources.memory_mb, Some(4096));
}
