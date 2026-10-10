//! The 0.8 task.md and split-layout readers against BenchFlow 0.8 itself.
//!
//! `tests/fixtures/v06/expected.json` records what 0.8's own loader
//! (`benchflow.task.Task`) read from every fixture package under `v06/`,
//! `split/`, and `convert/v06/` (see `make_goldens.py` there): the
//! instruction and the settings a run uses, or the error 0.8 refused the
//! package with. A package 0.8 reads must check without errors here and map
//! to the same settings; a package 0.8 refuses must be an error here, except
//! where Harbor runs it (below).

use std::path::{Path, PathBuf};
use std::time::Duration;

use benchflow_task::package::{ImageRef, TaskPackage};
use benchflow_task::values::Network;
use benchflow_task::{CheckOptions, Format};
use serde_json::Value;

/// Packages 0.8 refuses that the split reader accepts with warnings,
/// because Harbor runs them: 0.8 refuses unknown `task.toml` keys, and
/// Harbor ignores them (here they are X100 and X101 warnings).
const HARBOR_RUNS: &[&str] = &["split/typos"];

/// Missing-file errors. 0.8's loader reads the config without looking at
/// the folders (its `bench tasks check` and its sandboxes refuse a package
/// with no Dockerfile later), so these do not count against the comparison.
const FILE_CODES: &[&str] = &["X112", "X113", "X114", "X208"];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected() -> serde_json::Map<String, Value> {
    let text = std::fs::read_to_string(fixtures().join("v06/expected.json")).unwrap();
    match serde_json::from_str(&text).unwrap() {
        Value::Object(m) => m,
        _ => panic!("expected.json is an object"),
    }
}

fn secs(v: &Value) -> Option<Duration> {
    v.as_f64().map(Duration::from_secs_f64)
}

fn network(mode: &Value, hosts: &Value) -> Option<Network> {
    match mode.as_str()? {
        "no-network" => Some(Network::None),
        "public" => Some(Network::Open),
        "allowlist" => Some(Network::Allow(
            hosts
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h.as_str().unwrap().to_string())
                .collect(),
        )),
        other => panic!("network mode {other}"),
    }
}

fn env(v: &Value) -> Vec<(String, String)> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect()
}

fn sorted(map: &std::collections::BTreeMap<String, String>) -> Vec<(String, String)> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// The split reader leaves a size unset where Harbor does (Harbor has no
/// default CPUs, memory, or disk), so compare only what it sets; a 0.8
/// task.md takes 0.8's defaults, so everything is compared.
fn same_size(key: &str, ours: Option<f64>, theirs: &Value, split: bool) {
    match ours {
        Some(v) => assert_eq!(Some(v), theirs.as_f64(), "{key}"),
        None => assert!(split, "{key}: unset, but 0.8 has {theirs}"),
    }
}

fn compare(key: &str, p: &TaskPackage, want: &Value) {
    let split = p.format == Format::Split;
    // The instruction: Harbor strips leading canary lines before the agent
    // sees them, and a task's prompt_prefix comes first.
    let theirs = want["instruction"].as_str().unwrap();
    let theirs = benchflow_task::split::strip_canary(theirs);
    assert!(
        p.instruction.trim_end().ends_with(theirs.trim_end()),
        "{key}: instruction\n ours: {:?}\n0.8's: {:?}",
        p.instruction,
        theirs
    );
    if let Some(name) = want["name"].as_str() {
        assert_eq!(p.id, name, "{key}: name");
    }

    let agent = &want["agent"];
    assert_eq!(
        p.agent.timeout,
        secs(&agent["timeout_sec"]),
        "{key}: agent timeout"
    );
    assert_eq!(
        p.agent.user.as_deref(),
        agent["user"].as_str(),
        "{key}: agent user"
    );
    assert_eq!(
        p.agent.network,
        network(&agent["network_mode"], &agent["allowed_hosts"]),
        "{key}: agent network"
    );

    let s = &want["sandbox"];
    match s["docker_image"].as_str() {
        Some(image) => assert_eq!(
            p.sandbox.image,
            ImageRef::Pull {
                reference: image.into()
            },
            "{key}: image"
        ),
        None => assert!(
            matches!(p.sandbox.image, ImageRef::Build { .. }),
            "{key}: image"
        ),
    }
    same_size(&format!("{key}: cpus"), p.sandbox.cpus, &s["cpus"], split);
    same_size(
        &format!("{key}: memory"),
        p.sandbox.memory_mb.map(|m| m as f64),
        &s["memory_mb"],
        split,
    );
    same_size(
        &format!("{key}: storage"),
        p.sandbox.disk_mb.map(|m| m as f64),
        &s["storage_mb"],
        split,
    );
    assert_eq!(
        Some(p.sandbox.gpus as u64),
        s["gpus"].as_u64(),
        "{key}: gpus"
    );
    assert_eq!(
        p.sandbox.workdir.as_deref(),
        s["workdir"].as_str(),
        "{key}: workdir"
    );
    assert_eq!(sorted(&p.sandbox.env), env(&s["env"]), "{key}: sandbox env");
    assert_eq!(
        Some(p.sandbox.network.clone()),
        network(&s["network_mode"], &s["allowed_hosts"]),
        "{key}: sandbox network"
    );
    assert_eq!(
        p.sandbox.build_timeout,
        secs(&s["build_timeout_sec"]),
        "{key}: build timeout"
    );

    let v = &want["verifier"];
    assert_eq!(
        p.verifier.timeout,
        secs(&v["timeout_sec"]),
        "{key}: verifier timeout"
    );
    assert_eq!(
        p.verifier.user.as_deref(),
        v["user"].as_str(),
        "{key}: verifier user"
    );
    assert_eq!(
        sorted(&p.verifier.env),
        env(&v["env"]),
        "{key}: verifier env"
    );

    // A package with no oracle folder has no oracle to give the env to.
    if let Some(oracle) = &p.oracle {
        assert_eq!(
            sorted(&oracle.env),
            env(&want["solution"]["env"]),
            "{key}: oracle env"
        );
    }
}

#[test]
fn packages_read_as_0_8_reads_them() {
    let expected = expected();
    assert!(expected.len() >= 20, "every fixture package is recorded");
    for (key, want) in &expected {
        let report = benchflow_task::check(&fixtures().join(key), &CheckOptions { deep: true });
        let errors: Vec<_> = report.diagnostics.iter().filter(|d| d.is_error()).collect();
        if want["ok"] == false {
            if HARBOR_RUNS.contains(&key.as_str()) {
                assert!(errors.is_empty(), "{key}: {errors:#?}");
                assert!(!report.diagnostics.is_empty(), "{key}: warned");
            } else {
                assert!(
                    !errors.is_empty(),
                    "{key}: 0.8 refuses it ({}), and so must this reader",
                    want["error"]
                );
            }
            continue;
        }
        let config_errors: Vec<_> = errors
            .iter()
            .filter(|d| !FILE_CODES.contains(&d.code))
            .collect();
        assert!(
            config_errors.is_empty(),
            "{key}: 0.8 reads it, but: {config_errors:#?}"
        );
        let package = report.package.as_ref().expect("a package");
        compare(key, package, want);
    }
}
