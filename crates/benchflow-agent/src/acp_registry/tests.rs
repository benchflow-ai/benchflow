use super::*;
use crate::manifest::{ArchiveKind, CredentialFamily, InstallMethod};

/// The snapshot the agent catalog was built from, and today's index.
const SNAPSHOTS: [(&str, &str); 2] = [
    (
        "2026-09-29",
        include_str!("../../testdata/acp-registry-2026-09-29.json"),
    ),
    ("2026-10-10", SNAPSHOT),
];

fn index(date: &str) -> Index {
    let text = SNAPSHOTS.iter().find(|(d, _)| *d == date).unwrap().1;
    Index::parse(text).unwrap()
}

fn none() -> Registry {
    Registry::from_manifests(Vec::new())
}

/// `sh -n`: the script parses as POSIX shell.
fn assert_parses(script: &str, what: &str) {
    let out = std::process::Command::new("sh")
        .arg("-n")
        .arg("-c")
        .arg(script)
        .output()
        .expect("sh runs");
    assert!(
        out.status.success(),
        "{what}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn every_entry_becomes_a_manifest_or_a_clear_refusal() {
    for (date, text) in SNAPSHOTS {
        let index = Index::parse(text).unwrap();
        assert!(index.agents.len() >= 36, "{date}");
        let mut refused = Vec::new();
        for entry in &index.agents {
            match entry.to_manifest(&Registry::builtin()) {
                Ok(m) => {
                    assert_eq!(m.name, entry.spec());
                    assert_eq!(m.acp_registry.as_deref(), Some(entry.id.as_str()));
                    assert!(
                        m.launch_command()
                            .starts_with(&format!("/opt/benchflow/bin/{}", entry.id)),
                        "{}",
                        m.launch_command()
                    );
                    assert_eq!(m.model.select, crate::manifest::ModelSelect::Auto);
                    assert!(
                        !m.notes.is_empty(),
                        "{} says where its login comes from",
                        entry.id
                    );
                    assert_parses(&m.install_script(), &entry.spec());
                    assert_eq!(m.version_fingerprint().len(), 64);
                }
                Err(r) => {
                    assert!(
                        r.to_string().starts_with(&format!(
                            "acp:{}@{} cannot run in a sandbox: ",
                            r.id, r.version
                        )),
                        "{r}"
                    );
                    refused.push(r);
                }
            }
        }
        // Every entry of both snapshots has a Linux build or a package.
        assert!(refused.is_empty(), "{date}: {refused:?}");
    }
}

#[test]
fn binaries_in_subfolders_launch_by_their_real_path() {
    // Harbor's generic ACP agent runs `dist/<basename of cmd>`, so these
    // three never started there.
    for (date, id, cmd) in [
        ("2026-09-29", "cursor", "dist-package/cursor-agent"),
        ("2026-09-29", "junie", "junie-app/bin/junie"),
        (
            "2026-09-29",
            "cortex-code",
            "coco-1.0.73+180523.e6179a031de9-linux-amd64/cortex",
        ),
        ("2026-10-10", "devin", "bin/devin"),
        ("2026-10-10", "kimchi", "bin/kimchi"),
    ] {
        let m = index(date).find(id).unwrap().to_manifest(&none()).unwrap();
        let InstallMethod::Binary(b) = &m.install.method else {
            panic!("{id} installs its binary")
        };
        assert_eq!(b.targets["linux-x86_64"].cmd, cmd, "{id}");
        let script = m.install_script();
        assert!(
            script.contains(&format!("bf_cmd={}", crate::shell::quote(cmd))),
            "{id}"
        );
        assert!(script.contains("bf_target=\"$bf_dir/dist/$bf_cmd\""));
    }
}

#[test]
fn the_distribution_is_chosen_for_linux_sandboxes() {
    let old = index("2026-09-29");
    let new = index("2026-10-10");
    let kind = |index: &Index, id: &str| {
        let m = index.find(id).unwrap().to_manifest(&none()).unwrap();
        match m.install.method {
            InstallMethod::Binary(b) => format!("binary:{}", b.targets.len()),
            InstallMethod::Npx(n) => format!("npx:{}", n.package),
            InstallMethod::Uvx(u) => format!("uvx:{}", u.package),
            other => panic!("{other:?}"),
        }
    };
    // Builds for both Linux platforms beat a package.
    assert_eq!(kind(&old, "codex-acp"), "binary:2");
    assert_eq!(kind(&new, "kilo"), "binary:2");
    // A package beats a binary for one platform only.
    assert_eq!(kind(&old, "sigit"), "binary:2");
    assert_eq!(
        kind(&new, "codex-acp"),
        "npx:@agentclientprotocol/codex-acp@2.2.2"
    );
    assert_eq!(kind(&new, "qwen-code"), "npx:@qwen-code/qwen-code@0.25.0");
    // uvx's `name@version` becomes uv's `name==version`.
    assert_eq!(kind(&old, "minion-code"), "uvx:minion-code==0.1.44");
    assert_eq!(kind(&new, "fast-agent"), "uvx:fast-agent-acp==0.10.1");
    // One Linux platform, no package: that platform only.
    assert_eq!(kind(&old, "corust-agent"), "binary:1");
    // A bare executable download.
    let grok = old
        .find("grok-build")
        .unwrap()
        .to_manifest(&none())
        .unwrap();
    let InstallMethod::Binary(b) = &grok.install.method else {
        panic!()
    };
    assert_eq!(b.targets["linux-x86_64"].kind, ArchiveKind::File);
    assert_eq!(
        grok.launch_command(),
        "/opt/benchflow/bin/grok-build agent stdio"
    );
}

#[test]
fn entries_keep_their_arguments_variables_and_checksums() {
    let new = index("2026-10-10");
    let vt = new.find("vtcode").unwrap().to_manifest(&none()).unwrap();
    assert_eq!(vt.launch.env["VT_ACP_ENABLED"], "1");
    assert_eq!(vt.launch_command(), "/opt/benchflow/bin/vtcode acp");
    let factory = new
        .find("factory-droid")
        .unwrap()
        .to_manifest(&none())
        .unwrap();
    assert_eq!(
        factory.launch_command(),
        "/opt/benchflow/bin/factory-droid exec --output-format acp-daemon"
    );
    assert_eq!(factory.launch.env["DROID_DISABLE_AUTO_UPDATE"], "true");
    let goose = new.find("goose").unwrap().to_manifest(&none()).unwrap();
    let InstallMethod::Binary(b) = &goose.install.method else {
        panic!()
    };
    let sha = b.targets["linux-x86_64"]
        .sha256
        .clone()
        .expect("goose lists sha256");
    assert!(goose.install_script().contains(&format!("bf_sha={sha};")));
    assert!(goose.source.to_string().contains("acp:goose@1.54.0"));
}

#[test]
fn an_entry_borrows_the_logins_of_the_built_in_that_runs_the_same_agent() {
    let codex = AgentManifest::parse(
        r#"
name = "codex"
acp_registry = "codex-acp"
skill_paths = ["$HOME/.agents/skills"]
[install]
npm = "@agentclientprotocol/codex-acp@2.0.1"
bin = "codex-acp"
[launch]
env = { INITIAL_AGENT_MODE = "agent-full-access" }
[credentials]
family = "openai"
account = "an OpenAI account"
"#,
        ManifestSource::Builtin("codex.toml"),
    )
    .unwrap();
    let agents = Registry::from_manifests(vec![codex]);
    let m = resolve_in(&index("2026-10-10"), "acp:codex-acp", &agents).unwrap();
    assert_eq!(m.name, "acp:codex-acp@2.2.2");
    assert_eq!(m.credentials, CredentialFamily::OpenAi);
    assert_eq!(m.account.as_deref(), Some("an OpenAI account"));
    assert_eq!(m.skill_paths, ["$HOME/.agents/skills"]);
    assert_eq!(m.launch.env["INITIAL_AGENT_MODE"], "agent-full-access");
    assert!(m.notes[0].contains("`codex`"), "{:?}", m.notes);
    // The build is the registry's, not the built-in's.
    assert!(
        m.install_script()
            .contains("@agentclientprotocol/codex-acp@2.2.2")
    );
    // An entry no built-in names runs on no login, and says so.
    let goose = resolve_in(&index("2026-10-10"), "acp:goose", &agents).unwrap();
    assert_eq!(goose.credentials, CredentialFamily::None);
    assert!(goose.notes[0].contains("--env KEY=VALUE"));
}

#[test]
fn unknown_ids_and_versions_say_what_exists() {
    let idx = index("2026-10-10");
    let err = resolve_in(&idx, "acp:gooze", &none()).unwrap_err();
    assert!(matches!(&err, RegistryError::UnknownId { closest, .. } if closest[0] == "goose"));
    assert!(
        err.to_string()
            .starts_with("the public ACP registry has no agent `gooze`; closest: `goose`"),
        "{err}"
    );
    assert_eq!(
        err.next_step(),
        "`bench agents --registry` lists every entry"
    );
    let err = resolve_in(&idx, "acp:copilot", &none()).unwrap_err();
    assert!(err.to_string().contains("`github-copilot-cli`"), "{err}");

    let err = resolve_in(&idx, "acp:goose@0.0.1", &none()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "the public ACP registry has no goose 0.0.1; it lists goose 1.54.0"
    );
    assert_eq!(
        err.next_step(),
        "use acp:goose@1.54.0, or acp:goose for whatever the registry lists"
    );
    // The listed version resolves with or without `@`.
    assert_eq!(
        resolve_in(&idx, "acp:goose@1.54.0", &none()).unwrap().name,
        "acp:goose@1.54.0"
    );
}

#[test]
fn specs_are_checked() {
    assert_eq!(
        parse_spec("acp:goose@1.2.3").unwrap(),
        ("goose".to_string(), Some("1.2.3".to_string()))
    );
    assert_eq!(
        parse_spec("acp:goose").unwrap(),
        ("goose".to_string(), None)
    );
    for bad in ["acp:", "acp:@1", "acp:goose@", "acp:a b", "acp:a/b"] {
        let err = parse_spec(bad).unwrap_err();
        assert!(
            err.to_string().contains("is not an ACP registry agent"),
            "{bad}: {err}"
        );
    }
    assert!(is_spec("acp:x") && !is_spec("codex"));
}

#[test]
fn entries_without_a_linux_build_are_refused_with_the_reason() {
    let mut idx = index("2026-10-10");
    let entry = idx.agents.iter_mut().find(|e| e.id == "goose").unwrap();
    entry
        .distribution
        .binary
        .as_mut()
        .unwrap()
        .retain(|p, _| !p.starts_with("linux"));
    let err = resolve_in(&idx, "acp:goose", &none()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "acp:goose@1.54.0 cannot run in a sandbox: it publishes builds only for darwin-aarch64, darwin-x86_64, windows-x86_64, and sandboxes run Linux"
    );
    let entry = idx.agents.iter_mut().find(|e| e.id == "goose").unwrap();
    entry.distribution.binary = None;
    assert!(
        resolve_in(&idx, "acp:goose", &none())
            .unwrap_err()
            .to_string()
            .ends_with("it lists no binary, npx, or uvx distribution")
    );
}

#[tokio::test]
async fn a_local_index_file_is_used_as_is_and_a_fresh_cache_skips_the_network() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("registry.json");
    std::fs::write(
        &file,
        include_str!("../../testdata/acp-registry-2026-09-29.json"),
    )
    .unwrap();
    let source = Source {
        cache_dir: Some(tmp.path().join("cache")),
        file: Some(file.clone()),
    };
    let loaded = load(&source).await.unwrap();
    assert_eq!(loaded.origin, Origin::File(file));
    assert_eq!(loaded.index.find("goose").unwrap().version, "1.37.0");
    // With the file set, another version is not looked up on the network.
    let err = resolve(&loaded, "acp:goose@1.2.0", &source, &none())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("BENCHFLOW_ACP_REGISTRY is set"),
        "{err}"
    );

    // A cached version resolves without the network.
    let cached = source.version_path("goose", "1.2.0").unwrap();
    std::fs::create_dir_all(cached.parent().unwrap()).unwrap();
    let mut old = loaded.index.find("goose").unwrap().clone();
    old.version = "1.2.0".into();
    std::fs::write(&cached, serde_json::to_string(&old).unwrap()).unwrap();
    let m = resolve(&loaded, "acp:goose@1.2.0", &source, &none())
        .await
        .unwrap();
    assert_eq!(m.name, "acp:goose@1.2.0");

    // A cache under a day old is used without fetching.
    let source = Source {
        cache_dir: Some(tmp.path().join("cache")),
        file: None,
    };
    let path = source.index_path().unwrap();
    std::fs::write(&path, SNAPSHOT).unwrap();
    let loaded = load(&source).await.unwrap();
    assert!(
        matches!(loaded.origin, Origin::Cache { .. }),
        "{:?}",
        loaded.origin
    );
    assert!(loaded.origin.warning().is_none());
}

#[test]
fn closest_ids_rank_substrings_first() {
    let idx = index("2026-10-10");
    assert_eq!(idx.closest("claude", 1), ["claude-acp"]);
    assert_eq!(idx.closest("pi", 1), ["pi-acp"]);
    assert!(idx.closest("zzzzzzzz", 3).is_empty());
}
