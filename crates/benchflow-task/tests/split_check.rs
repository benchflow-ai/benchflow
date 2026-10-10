//! The split layout (`task.toml` + `instruction.md` + `environment/` +
//! `tests/` + `solution/`) through `benchflow_task::check`: warnings at the
//! exact key, safe fixes, the package it maps to, and the fields it refuses.
//!
//! Fixtures are in `tests/fixtures/split/`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use benchflow_task::diag::{Diagnostic, Severity};
use benchflow_task::package::{ImageRef, Isolation};
use benchflow_task::values::Network;
use benchflow_task::{CheckOptions, CheckReport, Format};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/split")
        .join(name)
}

fn check(name: &str) -> CheckReport {
    benchflow_task::check(&fixture(name), &CheckOptions { deep: true })
}

fn find<'a>(report: &'a CheckReport, code: &str, text: &str) -> &'a Diagnostic {
    report
        .diagnostics
        .iter()
        .find(|d| d.code == code && d.message.contains(text))
        .unwrap_or_else(|| panic!("no {code} with {text:?} in {:#?}", report.diagnostics))
}

/// The byte span of the first `needle` in `text` at or after `after`.
fn span_of(text: &str, needle: &str, after: &str) -> (usize, usize) {
    let from = text.find(after).expect("anchor");
    let start = from + text[from..].find(needle).expect("needle");
    (start, start + needle.len())
}

#[test]
fn typos_are_warnings_at_the_key_with_a_safe_fix() {
    let report = check("typos");
    assert_eq!(report.format, Some(Format::Split));
    assert!(report.ok(), "warnings only: {:#?}", report.diagnostics);
    assert_eq!(report.diagnostics.len(), 3, "{:#?}", report.diagnostics);
    let toml = fs::read_to_string(fixture("typos/task.toml")).unwrap();

    let typo = find(&report, "X100", "`timout_sec` in [agent]");
    assert_eq!(typo.severity, Severity::Warning);
    let at = typo.location.as_ref().unwrap();
    assert_eq!(at.line, Some(7));
    assert_eq!(at.span, Some(span_of(&toml, "timout_sec", "[agent]")));
    assert_eq!(typo.label.as_deref(), Some("did you mean `timeout_sec`?"));
    let fix = typo.fix.as_ref().expect("a close typo is fixed");
    assert_eq!(fix.span, at.span.unwrap());
    assert_eq!(fix.replacement, "timeout_sec");

    let misplaced = find(&report, "X101", "`cpus`");
    assert_eq!(
        misplaced.message,
        "`cpus` belongs in [environment], not [agent]; here it is ignored"
    );
    assert_eq!(misplaced.location.as_ref().unwrap().line, Some(8));
    assert!(misplaced.fix.is_none(), "moving a key is not a rename");

    let table = find(&report, "X100", "[verifer]");
    assert_eq!(table.message, "unknown table [verifer] is ignored");
    assert_eq!(table.label.as_deref(), Some("did you mean `verifier`?"));
    assert!(
        table.fix.is_none(),
        "renaming a table is left to the author"
    );

    // The ignored keys take Harbor's defaults, as Harbor would run the task.
    let package = report.package.as_ref().unwrap();
    assert_eq!(package.agent.timeout, None);
    assert_eq!(package.verifier.timeout, Some(Duration::from_secs(600)));
    assert_eq!(package.sandbox.cpus, None);
    assert_eq!(package.sandbox.memory_mb, Some(4096));
    assert_eq!(package.sandbox.network, Network::None);
}

#[test]
fn a_full_harbor_task_maps_every_key() {
    let report = check("full");
    assert!(report.diagnostics.is_empty(), "{:#?}", report.diagnostics);
    let p = report.package.unwrap();
    assert_eq!(p.id, "acme/full");
    assert_eq!(p.name, "full");
    assert_eq!(p.version.as_deref(), Some("2.1.0"));
    // Harbor strips the leading canary lines and the blank line after them.
    assert_eq!(
        p.instruction,
        "Fix the parser in /workspace so that `make test` passes.\n"
    );
    assert_eq!(p.agent.timeout, Some(Duration::from_secs(900)));
    assert_eq!(p.agent.user.as_deref(), Some("agent"));
    assert_eq!(p.verifier.timeout, Some(Duration::from_secs(120)));
    assert_eq!(p.verifier.env["API_URL"], "http://localhost:8000");
    assert_eq!(
        (
            p.verifier.dir.as_path(),
            p.verifier.mount.as_str(),
            p.verifier.entry.as_str()
        ),
        (Path::new("tests"), "/tests", "test.sh")
    );
    assert_eq!(p.verifier.isolation, Isolation::Shared);
    assert_eq!(
        p.sandbox.image,
        ImageRef::Build {
            context: "environment".into(),
            dockerfile: "environment/Dockerfile".into()
        }
    );
    assert_eq!(p.sandbox.cpus, Some(2.0));
    assert_eq!(p.sandbox.memory_mb, Some(4096));
    assert_eq!(p.sandbox.disk_mb, Some(20480));
    assert_eq!(p.sandbox.build_timeout, Some(Duration::from_secs(300)));
    assert_eq!(
        p.sandbox.network,
        Network::Allow(vec!["pypi.org".into(), "*.github.com".into()])
    );
    assert_eq!(p.sandbox.workdir.as_deref(), Some("/workspace"));
    assert_eq!(p.sandbox.env["LANG"], "C.UTF-8");
    let ready = p.sandbox.ready.as_ref().unwrap();
    assert_eq!(ready.run, "test -f /workspace/ready");
    assert_eq!(ready.interval, Duration::from_secs(2));
    assert_eq!(ready.retries, 5);
    let oracle = p.oracle.as_ref().unwrap();
    assert_eq!(
        (
            oracle.dir.as_path(),
            oracle.mount.as_str(),
            oracle.entry.as_str()
        ),
        (Path::new("solution"), "/solution", "solve.sh")
    );
    assert_eq!(oracle.env["HINT"], "1");
    assert!(p.unsupported.is_empty(), "{:?}", p.unsupported);
    assert_eq!(
        p.digest,
        benchflow_task::digest::harbor_digest(&fixture("full")).unwrap()
    );
}

#[test]
fn benchflow_spellings_are_read_with_a_note_for_harbor() {
    let report = check("benchflow-0.7");
    assert!(report.ok(), "{:#?}", report.diagnostics);
    for (key, harbor) in [
        ("`sandbox` is", "`environment`"),
        ("`sandbox_mode` in [verifier]", "`environment_mode`"),
        ("`oracle` is", "`solution`"),
    ] {
        let d = find(&report, "X104", key);
        assert!(d.message.contains(harbor), "{}", d.message);
        assert_eq!(d.severity, Severity::Warning);
    }
    let memory = find(&report, "X102", "`memory` in [sandbox]");
    let fix = memory.fix.as_ref().unwrap();
    assert_eq!(fix.replacement, "memory_mb = 2048");
    let toml = fs::read_to_string(fixture("benchflow-0.7/task.toml")).unwrap();
    assert_eq!(&toml[fix.span.0..fix.span.1], "memory = \"2G\"");

    let p = report.package.unwrap();
    assert_eq!(
        p.instruction, "You are working in a sandbox.\n\nWrite the report.\n",
        "prompt_prefix comes first"
    );
    assert_eq!(p.sandbox.memory_mb, Some(2048));
    assert_eq!(p.agent.timeout, Some(Duration::from_secs(600)));
    assert_eq!(
        (p.verifier.dir.as_path(), p.verifier.mount.as_str()),
        (Path::new("verifier"), "/verifier"),
        "a verifier/ folder wins over tests/, as in 0.8"
    );
    let oracle = p.oracle.unwrap();
    assert_eq!(oracle.dir, Path::new("oracle"));
    assert_eq!(oracle.env["SEED"], "7");
}

#[test]
fn fields_the_runner_cannot_honor_are_refused_by_name() {
    let report = check("refused");
    assert!(report.ok(), "{:#?}", report.diagnostics);
    let p = report.package.unwrap();
    let fields: Vec<&str> = p.unsupported.iter().map(|u| u.field.as_str()).collect();
    assert_eq!(fields, ["[environment] os", "[verifier] reward_range"]);
    assert_eq!(
        p.sandbox.skills_dir.as_deref(),
        Some("/skills"),
        "Harbor's skills_dir is a folder in the sandbox, copied to the agent"
    );
    assert_eq!(p.verifier.entry, "test.bat", "a Windows task's verifier");
    // Steps and GPU types are read; a backend that cannot honor them refuses
    // the trial when it starts.
    assert_eq!(p.extras.gpu_types, ["H100"]);
    let steps = p.extras.steps.as_ref().unwrap();
    let names: Vec<&str> = steps.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["first", "second"]);
}

#[test]
fn broken_tasks_are_errors_that_name_the_file_and_value() {
    let report = check("broken");
    assert!(!report.ok());
    let toml = fs::read_to_string(fixture("broken/task.toml")).unwrap();
    let timeout = find(&report, "X111", "`timeout_sec` in [agent]");
    assert_eq!(timeout.severity, Severity::Error);
    assert_eq!(
        timeout.location.as_ref().unwrap().span,
        Some(span_of(&toml, "\"ten minutes\"", "[agent]"))
    );
    let missing = find(&report, "X112", "instruction.md");
    assert_eq!(missing.severity, Severity::Error);
    assert_eq!(
        find(&report, "X113", "tests/test.sh").severity,
        Severity::Error
    );
    let hosts = find(&report, "X103", "allowed_hosts");
    assert_eq!(hosts.severity, Severity::Warning);
}

#[test]
fn every_code_is_explained() {
    for code in [
        "X100", "X101", "X102", "X103", "X104", "X111", "X112", "X113",
    ] {
        assert!(
            benchflow_task::diag::explain(code).is_some(),
            "bench explain {code}"
        );
    }
}
