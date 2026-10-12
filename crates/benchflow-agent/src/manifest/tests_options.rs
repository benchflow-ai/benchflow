//! Agent options (`[options.<name>]`) and `[resume]`.

use std::path::PathBuf;

use super::*;

fn builtin(name: &str) -> AgentManifest {
    Registry::builtin().find(name).unwrap().clone()
}

fn parse(text: &str) -> Result<AgentManifest, ManifestError> {
    AgentManifest::parse(
        text,
        ManifestSource::File(PathBuf::from("agents/test.toml")),
    )
}

fn given(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

const HEADLESS: &str = r#"
name = "t"

[install]
npm = "@scope/t@1.0.0"
bin = "t"
companions = ["t-cli@2.0.0"]
verify = "t --version | grep -q 2.0.0"

[headless]
command = 'printf "[%s]" "$@"'
output = "text"
"#;

#[test]
fn an_unknown_option_gets_a_suggestion_and_the_list() {
    let claude = builtin("claude");
    let e = claude
        .resolve_options(&given(&[("max_turn", "5")]), Harness::Headless)
        .unwrap_err();
    assert_eq!(e.message, "agent `claude` has no option `max_turn`");
    assert!(
        e.next.starts_with("did you mean `max_turns`?"),
        "{}",
        e.next
    );
    assert!(e.next.contains("bench agents claude"), "{}", e.next);

    let e = builtin("kimi")
        .resolve_options(&given(&[("max_turns", "5")]), Harness::Acp)
        .unwrap_err();
    assert!(e.message.contains("takes no options"), "{}", e.message);
    assert!(e.next.contains("--env"), "{}", e.next);
}

#[test]
fn a_wrong_value_says_what_the_option_takes() {
    let claude = builtin("claude");
    let e = claude
        .resolve_options(&given(&[("max_turns", "ten")]), Harness::Headless)
        .unwrap_err();
    assert_eq!(
        e.message,
        "option `max_turns` of claude takes a whole number from 1, not `ten`"
    );
    let e = claude
        .resolve_options(&given(&[("max_turns", "0")]), Harness::Headless)
        .unwrap_err();
    assert!(e.message.contains("from 1, not `0`"), "{}", e.message);
    let e = builtin("codex")
        .resolve_options(&given(&[("web_search", "liev")]), Harness::Acp)
        .unwrap_err();
    assert!(
        e.message
            .contains("one of disabled, cached, live, not `liev` (did you mean `live`?)"),
        "{}",
        e.message
    );
    let e = claude
        .resolve_options(&given(&[("version", "2.1; rm -rf /")]), Harness::Acp)
        .unwrap_err();
    assert!(e.message.contains("takes a version"), "{}", e.message);
    // Guards accbea8f (typed agent options): free text after a flag could
    // read as another flag of the agent's CLI.
    let e = claude
        .resolve_options(
            &given(&[("append_system_prompt", "--some-flag")]),
            Harness::Headless,
        )
        .unwrap_err();
    assert!(
        e.message.contains("does not start with `-`"),
        "{}",
        e.message
    );
    assert!(
        claude
            .resolve_options(
                &given(&[("append_system_prompt", "be brief - always")]),
                Harness::Headless
            )
            .is_ok()
    );
    assert!(
        options::parse_assignment("max_turns").is_err(),
        "a name with no value"
    );
    assert_eq!(
        options::parse_assignment("max-turns=3").unwrap(),
        ("max_turns".to_string(), "3".to_string())
    );
}

#[test]
fn an_option_for_one_harness_is_refused_on_the_other() {
    let e = builtin("claude")
        .resolve_options(&given(&[("max_turns", "5")]), Harness::Acp)
        .unwrap_err();
    assert!(
        e.message
            .contains("reaches the agent only on the headless harness, and this run uses acp"),
        "{}",
        e.message
    );
    assert_eq!(e.next, "add --harness headless, or drop --ak max_turns");
}

#[test]
fn flags_go_where_the_command_says_quoted_for_the_shell() {
    let text = format!(
        "{HEADLESS}\n[options.note]\ntype = \"string\"\ndescription = \"x\"\nflag = \"--note\"\n\n[options.mode]\ntype = \"enum\"\ndescription = \"x\"\nchoices = [\"a b\"]\nflag = \"-c mode={{value}}\"\n\n[options.fast]\ntype = \"bool\"\ndescription = \"x\"\nflag = \"--fast\"\n"
    );
    let m = parse(&text).unwrap();
    let values = m
        .resolve_options(
            &given(&[
                ("note", "it's \"$HOME\" `x`"),
                ("mode", "a b"),
                ("fast", "false"),
            ]),
            Harness::Headless,
        )
        .unwrap();
    let applied = m.with_options(&values);
    let command = &applied.headless.as_ref().unwrap().command;
    assert!(command.starts_with("set -- "), "{command}");
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[--note][it's \"$HOME\" `x`][-c][mode=a b]",
        "each word reaches the agent as written; a false bool adds nothing"
    );
    assert_eq!(
        applied.chosen_options["note"],
        OptionValue::String("it's \"$HOME\" `x`".into())
    );
    // Nothing set: the command is the manifest's.
    let none = m.with_options(&OptionValues::new());
    assert_eq!(none.headless, m.headless);
}

#[test]
fn claude_options_reach_the_headless_command_and_both_harnesses_env() {
    let claude = builtin("claude");
    let values = claude
        .resolve_options(
            &given(&[("max_turns", "7"), ("max_thinking_tokens", "4096")]),
            Harness::Headless,
        )
        .unwrap();
    let m = claude.with_options(&values);
    let h = m.headless.as_ref().unwrap();
    assert!(
        h.command.starts_with("set -- --max-turns 7\n"),
        "{}",
        h.command
    );
    assert_eq!(h.env["MAX_THINKING_TOKENS"], "4096");
    assert_eq!(m.launch.env["MAX_THINKING_TOKENS"], "4096");
    assert_eq!(m.launch_command(), claude.launch_command());
    assert_eq!(
        m.version_fingerprint(),
        claude.version_fingerprint(),
        "nothing installed changed"
    );
}

/// Codex's `-c` overrides reach Codex headless only: codex-acp does not
/// take them (BenchFlow 0.8 documents this, and passes such settings
/// through CODEX_CONFIG), and E7's live run over ACP with
/// `-c web_search=disabled` still sent the web_search tool. Over ACP they
/// are refused before anything runs; headless they are passed.
#[test]
fn codex_flags_reach_the_headless_harness_only() {
    let codex = builtin("codex");
    for option in [
        ("reasoning_summary", "detailed"),
        ("web_search", "disabled"),
    ] {
        assert!(
            codex
                .resolve_options(&given(&[option]), Harness::Acp)
                .is_err(),
            "{option:?} over ACP"
        );
    }
    let values = codex
        .resolve_options(
            &given(&[("reasoning_summary", "detailed")]),
            Harness::Headless,
        )
        .unwrap();
    let m = codex.with_options(&values);
    assert!(
        m.headless
            .as_ref()
            .unwrap()
            .command
            .starts_with("set -- -c model_reasoning_summary=detailed\n")
    );
    assert!(
        !m.launch_command().contains("model_reasoning_summary"),
        "{}",
        m.launch_command()
    );
}

#[test]
fn a_version_option_repins_the_install_and_changes_the_fingerprint() {
    let claude = builtin("claude");
    let values = claude
        .resolve_options(&given(&[("version", "2.1.300")]), Harness::Acp)
        .unwrap();
    let m = claude.with_options(&values);
    let script = m.install_script();
    assert!(
        script.contains("@anthropic-ai/claude-code@2.1.300"),
        "{script}"
    );
    assert!(
        script.contains("'2.1.300 (Claude Code)'"),
        "verify follows: {script}"
    );
    assert!(!script.contains("2.1.280"), "{script}");
    assert!(
        script.contains("@agentclientprotocol/claude-agent-acp@0.81.2"),
        "the adapter keeps its pin"
    );
    assert_ne!(m.version_fingerprint(), claude.version_fingerprint());
    // Asking for the pinned version is the built-in install.
    let same = claude.with_options(
        &claude
            .resolve_options(&given(&[("version", "2.1.280")]), Harness::Acp)
            .unwrap(),
    );
    assert_eq!(same.version_fingerprint(), claude.version_fingerprint());

    for (agent, package, version) in [
        ("aider", "aider-chat==0.90.0", "0.90.0"),
        ("openhands", "openhands==1.17.0", "1.17.0"),
        ("gemini", "@google/gemini-cli@0.43.0", "0.43.0"),
    ] {
        let a = builtin(agent);
        let m = a.with_options(
            &a.resolve_options(&given(&[("version", version)]), a.harness)
                .unwrap(),
        );
        assert!(m.install_script().contains(package), "{agent}");
        assert_ne!(m.version_fingerprint(), a.version_fingerprint(), "{agent}");
    }
}

#[test]
fn a_file_option_sets_its_key_for_both_harnesses() {
    let gemini = builtin("gemini");
    let values = gemini
        .resolve_options(&given(&[("max_turns", "9")]), Harness::Acp)
        .unwrap();
    let m = gemini.with_options(&values);
    for files in [&m.launch.files, &m.headless.as_ref().unwrap().files] {
        let f = files
            .iter()
            .find(|f| f.path == ".gemini/settings.json")
            .expect("the settings file");
        let v: serde_json::Value = serde_json::from_str(&f.contents).unwrap();
        assert_eq!(v["model"]["maxSessionTurns"], 9);
    }
}

#[test]
fn mini_swe_agent_flags_follow_its_defaults() {
    let m = builtin("mini-swe-agent");
    let values = m
        .resolve_options(&given(&[("cost_limit", "1.5")]), Harness::Headless)
        .unwrap();
    let command = m.with_options(&values).headless.unwrap().command;
    assert!(
        command.starts_with("set -- --cost-limit 1.5\n"),
        "{command}"
    );
    assert!(
        !command.contains("\nset --\n"),
        "nothing clears the option's words"
    );
}

#[test]
fn option_tables_are_checked_when_the_manifest_is_read() {
    let bad = |extra: &str| {
        parse(&format!("{HEADLESS}\n{extra}"))
            .unwrap_err()
            .to_string()
    };
    let e = bad("[options.x]\ntyp = \"int\"\n");
    assert!(
        e.contains("unknown key `options.x.typ`; did you mean `type`?"),
        "{e}"
    );
    let e = bad("[options.x]\ntype = \"integer\"\ndescription = \"d\"\nenv = \"X\"\n");
    assert!(e.contains("unknown type `integer`"), "{e}");
    let e = bad("[options.x]\ntype = \"int\"\ndescription = \"d\"\n");
    assert!(e.contains("say how it reaches the agent"), "{e}");
    let e = bad("[options.x]\ntype = \"enum\"\ndescription = \"d\"\nenv = \"X\"\n");
    assert!(e.contains("lists its `choices`"), "{e}");
    let e = bad("[options.x]\ntype = \"int\"\ndescription = \"d\"\nenv = \"BENCHFLOW_MODEL\"\n");
    assert!(e.contains("set by BenchFlow"), "{e}");
    let e = bad("[options.x]\ntype = \"int\"\ndescription = \"d\"\nenv = \"X\"\ndefault = \"3\"\n");
    assert!(
        e.contains("`options.x.default`: must be a whole number"),
        "{e}"
    );
    let e = bad(
        "[options.x]\ntype = \"string\"\ndescription = \"d\"\ninstall = \"t-cli\"\ndefault = \"1.0\"\n",
    );
    assert!(
        e.contains("must be the version the install pins, \"2.0.0\""),
        "{e}"
    );
    let e = bad(
        "[options.x]\ntype = \"string\"\ndescription = \"d\"\ninstall = \"nope\"\ndefault = \"1\"\n",
    );
    assert!(e.contains("pins no package `nope`"), "{e}");
    let e = bad("[options.Max]\ntype = \"int\"\ndescription = \"d\"\nenv = \"X\"\n");
    assert!(e.contains("not an option name"), "{e}");
    // A flag needs "$@" in the command it goes to.
    let e = parse(
        "name = \"t\"\n[install]\nnpm = \"t@1.0.0\"\nbin = \"t\"\n[headless]\ncommand = \"t\"\noutput = \"text\"\n[options.x]\ntype = \"int\"\ndescription = \"d\"\nflag = \"--x\"\n",
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("must say where flags go with \"$@\""), "{e}");
    // An install option repins a companion too, and its verify line.
    let m = parse(&format!(
        "{HEADLESS}\n[options.version]\ntype = \"string\"\ndescription = \"d\"\ninstall = \"t-cli\"\ndefault = \"2.0.0\"\n"
    ))
    .unwrap();
    let v = m.with_options(
        &m.resolve_options(&given(&[("version", "2.1.0")]), Harness::Headless)
            .unwrap(),
    );
    let script = v.install_script();
    assert!(
        script.contains("t-cli@2.1.0") && script.contains("grep -q 2.1.0"),
        "{script}"
    );
}

#[test]
fn resume_tables_are_checked_and_the_built_in_agents_declare_them() {
    let e = parse(&format!(
        "{HEADLESS}\n[resume]\nheadless = true\nsession_dirs = [\".t\"]\n"
    ))
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("must resume when $BENCHFLOW_RESUME is set"),
        "{e}"
    );
    let e = parse(&format!("{HEADLESS}\n[resume]\nacp = true\n"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no acp harness"), "{e}");
    for (agent, acp, headless) in [
        ("claude", true, true),
        ("codex", true, true),
        ("gemini", true, true),
        ("opencode", true, true),
        ("goose", true, true),
        ("aider", false, false),
        ("mini-swe-agent", false, false),
    ] {
        let r = &builtin(agent).resume;
        assert_eq!((r.acp, r.headless), (acp, headless), "{agent}");
        assert_eq!(r.session_dirs.is_empty(), !(acp || headless), "{agent}");
    }
    assert!(
        builtin("claude")
            .headless
            .unwrap()
            .command
            .contains("${BENCHFLOW_RESUME:+--continue}")
    );
    assert!(
        builtin("codex")
            .headless
            .unwrap()
            .command
            .contains("exec ${BENCHFLOW_RESUME:+resume --last} --json")
    );
}

#[test]
fn the_most_used_agents_declare_options() {
    for agent in [
        "claude",
        "codex",
        "gemini",
        "opencode",
        "goose",
        "aider",
        "mini-swe-agent",
        "openhands",
    ] {
        let m = builtin(agent);
        assert!(!m.options.is_empty(), "{agent} declares no options");
        for o in &m.options {
            assert!(!o.description.is_empty(), "{agent}.{}", o.name);
        }
    }
}

#[test]
fn an_install_pin_changes_only_the_whole_version_in_verify() {
    let v = options::replace_version(
        "t --version | grep -q 2.0.0 && test -d /opt/12.0.0 && echo 2.0.0-rc1 2.0.0",
        "2.0.0",
        "3.1.0",
    );
    assert_eq!(
        v,
        "t --version | grep -q 3.1.0 && test -d /opt/12.0.0 && echo 2.0.0-rc1 3.1.0"
    );
    assert_eq!(
        options::replace_version("'2.0.0 (T)'", "2.0.0", "3.1.0"),
        "'3.1.0 (T)'"
    );
}
