//! BenchFlow 0.8's task.md (YAML frontmatter, then the prompt) through
//! `benchflow_task::check`: strict keys, renamed keys, shorthands and
//! profiles, and the features the runner refuses by name.
//!
//! Fixtures are in `tests/fixtures/v06/` and `tests/fixtures/convert/v06/`;
//! `tests/reference_08.rs` compares every one of them with what 0.8 reads.

use std::path::{Path, PathBuf};
use std::time::Duration;

use benchflow_task::diag::{Diagnostic, Severity};
use benchflow_task::package::ImageRef;
use benchflow_task::values::Network;
use benchflow_task::{CheckOptions, CheckReport, Format};

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(path)
}

fn check(path: &str) -> CheckReport {
    benchflow_task::check(&fixture(path), &CheckOptions { deep: true })
}

fn with_code<'a>(report: &'a CheckReport, code: &str) -> Vec<&'a Diagnostic> {
    report
        .diagnostics
        .iter()
        .filter(|d| d.code == code)
        .collect()
}

#[test]
fn unknown_keys_are_errors_naming_the_key_path_and_line() {
    let report = check("v06/strict");
    assert_eq!(report.format, Some(Format::TaskMdV06));
    let found: Vec<(String, Option<usize>, Option<String>)> = with_code(&report, "X201")
        .into_iter()
        .map(|d| {
            assert_eq!(d.severity, Severity::Error);
            (
                d.message.clone(),
                d.location.as_ref().and_then(|l| l.line),
                d.help.clone(),
            )
        })
        .collect();
    assert_eq!(
        found,
        [
            (
                "unknown key `timout` in `agent`".to_string(),
                Some(5),
                Some("did you mean `timeout_sec`?".to_string())
            ),
            (
                "unknown key `netwrk` in `sandbox`".to_string(),
                Some(8),
                Some("did you mean `network_mode`?".to_string())
            ),
            (
                "unknown key `cpu` in `verifier.sandbox`".to_string(),
                Some(13),
                Some("did you mean `cpus`?".to_string())
            ),
        ]
    );
    assert_eq!(report.diagnostics.len(), 3, "{:#?}", report.diagnostics);
}

#[test]
fn renamed_keys_are_errors_that_say_the_new_name() {
    let report = check("v06/renamed");
    let renamed = with_code(&report, "X203");
    assert_eq!(renamed.len(), 2, "{:#?}", report.diagnostics);
    assert_eq!(
        renamed[0].message,
        "the `environment` key was renamed to `sandbox`"
    );
    assert_eq!(renamed[0].location.as_ref().unwrap().line, Some(2));
    assert_eq!(
        renamed[1].message,
        "the `verifier.environment_mode` key was renamed to `verifier.sandbox_mode`"
    );
    assert_eq!(renamed[1].location.as_ref().unwrap().line, Some(6));
    assert!(renamed.iter().all(|d| d.is_error()));
}

#[test]
fn shorthands_and_profiles_expand_as_in_0_8() {
    let report = check("v06/shorthands");
    assert!(report.diagnostics.is_empty(), "{:#?}", report.diagnostics);
    let p = report.package.unwrap();
    // `name:` without an organization is filed under benchflow/, as 0.8 does.
    assert_eq!(p.id, "benchflow/shorthand-task");
    // `image:` names a prebuilt image, so no environment/ is needed.
    assert_eq!(
        p.sandbox.image,
        ImageRef::Pull {
            reference: "python:3.12-slim".into()
        }
    );
    // The code-change profile fills what the task leaves out ...
    assert_eq!(p.sandbox.cpus, Some(4.0));
    assert_eq!(p.sandbox.memory_mb, Some(8192));
    assert_eq!(p.sandbox.workdir.as_deref(), Some("/repo"));
    assert_eq!(p.sandbox.network, Network::None);
    assert_eq!(p.verifier.timeout, Some(Duration::from_secs(1200)));
    assert_eq!(p.verifier.user.as_deref(), Some("root"));
    // ... and what the task sets wins.
    assert_eq!(p.agent.timeout, Some(Duration::from_secs(600)));
    assert_eq!(p.verifier.env["STRICT"], "1");
    // The prompt as 0.8 sends it: the body, without its final newline.
    assert_eq!(
        p.instruction,
        "Fix the bug in /repo so that the tests pass."
    );
}

#[test]
fn features_the_runner_lacks_are_refused_by_name() {
    let report = check("convert/v06/private-facts-nudges");
    assert!(report.ok(), "{:#?}", report.diagnostics);
    let p = report.package.unwrap();
    let fields: Vec<&str> = p.unsupported.iter().map(|u| u.field.as_str()).collect();
    assert_eq!(
        fields,
        ["scenes", "user", "benchflow.nudges", "## user-persona"]
    );
    assert!(p.unsupported.iter().all(|u| !u.reason.is_empty()));
}

#[test]
fn deprecated_spellings_are_read_with_a_note() {
    let report = check("convert/v06/regex-email-parser");
    assert!(report.ok(), "{:#?}", report.diagnostics);
    let notes = with_code(&report, "X212");
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].severity, Severity::Note);
    assert!(notes[0].message.contains("`allow_internet`"));
    let p = report.package.unwrap();
    assert_eq!(p.sandbox.network, Network::Open);
    assert_eq!(p.unsupported[0].field, "sandbox.skills_dir");
}

#[test]
fn a_task_format_plugin_task_is_refused() {
    let report = check("convert/v06/report-writer");
    let refusal = with_code(&report, "X213");
    assert_eq!(refusal.len(), 1, "{:#?}", report.diagnostics);
    assert!(refusal[0].is_error());
    assert!(
        refusal[0].message.contains("robouse"),
        "{}",
        refusal[0].message
    );
}

/// F1 (rc.2, round 2): in BenchFlow 0.8's task.md the cloud metadata
/// opt-out is `metadata: {x-allow-cloud-metadata: true}`, which 0.8 loads
/// (`tests/reference_08.rs` holds what 0.8 read); in `sandbox` it is an
/// unknown key, which 0.8 refuses, so it is an error here too.
#[test]
fn the_cloud_metadata_opt_out_is_read_from_metadata() {
    let report = check("v06/metadata-opt-out");
    assert_eq!(report.format, Some(Format::TaskMdV06));
    assert!(
        report
            .diagnostics
            .iter()
            .all(|d| d.severity != Severity::Error),
        "{:#?}",
        report.diagnostics
    );
    assert!(
        report
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
    let report = check("v06/metadata-in-sandbox");
    let errors = with_code(&report, "X201");
    assert!(
        errors
            .iter()
            .any(|d| d.message.contains("allow_cloud_metadata")),
        "{:#?}",
        report.diagnostics
    );
}
