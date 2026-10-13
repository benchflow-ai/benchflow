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
    assert_eq!(fields, ["[environment] os"]);
    // `[verifier] reward_range` is honored since 2026-10-11 (break campaign
    // E9), as in BenchFlow 0.8: the trial is graded against it.
    assert_eq!(p.verifier.reward_range, Some((-1.0, 1.0)));
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

#[test]
fn a_task_given_as_a_path_without_a_file_name_is_named_after_its_folder() {
    // `bench run .` passes `.`; a path ending in `..` has no file name
    // either. The task is named after the folder, not `.` (whose trial
    // folders, `.__<id>`, would be hidden).
    let dir = fixture("full").join("environment").join("..");
    assert!(dir.file_name().is_none());
    let report = benchflow_task::check(&dir, &CheckOptions { deep: true });
    assert_eq!(report.package.unwrap().name, "full");
}

#[test]
fn a_task_reached_through_a_symlink_keeps_the_name_given() {
    // A symlinked task folder keeps the link's name: resolving it renamed
    // the task (and so every trial folder) after the link's target.
    let tmp = tempfile::tempdir().unwrap();
    let link = tmp.path().join("fix-auth");
    std::os::unix::fs::symlink(fixture("full"), &link).unwrap();
    let report = benchflow_task::check(&link, &CheckOptions { deep: true });
    assert_eq!(report.package.unwrap().name, "fix-auth");
}

/// 0.8 fills an env value written `${VAR}` from the host's environment;
/// this runner passes no host value into a task, so such a value is refused
/// by name before any sandbox starts, rather than reaching the script as the
/// text `${OPENAI_API_KEY}` (an LLM judge then fails every trial, blamed on
/// the task). A value with other text around it is literal, as in 0.8.
#[test]
fn env_values_taken_from_the_host_are_refused_by_name() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("judged");
    copy(&fixture("full"), &root);
    let toml = fs::read_to_string(root.join("task.toml")).unwrap();
    let toml = toml
        .replace(
            r#"env = { API_URL = "http://localhost:8000" }"#,
            r#"env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", API_URL = "http://localhost:8000", NOTE = "cost ${USD}" }"#,
        )
        .replace(r#"env = { LANG = "C.UTF-8" }"#, r#"env = { LANG = "C.UTF-8", MODE = "${MODE:-fast}" }"#)
        .replace(r#"env = { HINT = "1" }"#, r#"env = { HINT = "1", TOKEN = "${HF_TOKEN}" }"#);
    fs::write(root.join("task.toml"), toml).unwrap();
    let report = benchflow_task::check(&root, &CheckOptions { deep: true });
    let p = report.package.as_ref().expect("a package");
    let fields: Vec<&str> = p.unsupported.iter().map(|u| u.field.as_str()).collect();
    assert_eq!(
        fields,
        [
            r#"[environment] env MODE = "${MODE:-fast}""#,
            r#"[verifier] env OPENAI_API_KEY = "${OPENAI_API_KEY}""#,
            r#"[solution] env TOKEN = "${HF_TOKEN}""#,
        ],
        "{:#?}",
        p.unsupported
    );
    assert!(p.unsupported.iter().all(|u| !u.agents_only));
    assert_eq!(p.verifier.env["NOTE"], "cost ${USD}");
}

fn copy(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy(&e.path(), &dest);
        } else {
            fs::copy(e.path(), dest).unwrap();
        }
    }
}

/// A task given as `.` (`bench run .`) or through `..` is named after its
/// folder, as task.md's are: `.` made trial folders named `.__<id>`, hidden.
#[test]
fn a_task_reached_through_dot_dot_is_named_after_its_folder() {
    let report = benchflow_task::check(
        &fixture("full").join("environment").join(".."),
        &CheckOptions { deep: true },
    );
    let p = report.package.as_ref().expect("a package");
    assert_eq!(p.name, "full");
}

/// A link to a task is named after the link, as 0.8 names a task after its
/// folder's name: a folder holding `full` and `full-latest -> full` holds two
/// tasks with two names and two trial names, not two tasks named `full`
/// (review E4-A, 4). The same for task.md packages.
#[test]
fn a_task_reached_through_a_link_is_named_after_the_link() {
    let d = tempfile::tempdir().unwrap();
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/taskmd/examples");
    for (target, link) in [
        (fixture("full"), "full-latest"),
        (examples.join("hello-world"), "hello-latest"),
    ] {
        let at = d.path().join(link);
        std::os::unix::fs::symlink(&target, &at).unwrap();
        let report = benchflow_task::check(&at, &CheckOptions { deep: true });
        let p = report
            .package
            .as_ref()
            .unwrap_or_else(|| panic!("{link}: {:#?}", report.diagnostics));
        assert_eq!(p.name, link);
    }
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &dest);
        } else {
            fs::copy(e.path(), &dest).unwrap();
        }
    }
}

/// `[verifier] reward_range` may only widen [0, 1], as in BenchFlow 0.8: a
/// range that narrows or shifts it is an error at the key, and a task that
/// declares none keeps [0, 1].
#[test]
fn a_reward_range_may_only_widen_zero_to_one() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("task");
    copy_tree(&fixture("refused"), &root);
    let toml = fs::read_to_string(root.join("task.toml")).unwrap();
    for (range, ok) in [
        ("[-1.0, 1.0]", true),
        ("[0, 10]", true),
        ("[0.2, 1.0]", false),
        ("[0.0, 0.5]", false),
        ("[1.0, -1.0]", false),
        ("\"wide\"", false),
    ] {
        fs::write(
            root.join("task.toml"),
            toml.replace(
                "reward_range = [-1.0, 1.0]",
                &format!("reward_range = {range}"),
            ),
        )
        .unwrap();
        let report = benchflow_task::check(&root, &CheckOptions { deep: true });
        assert_eq!(report.ok(), ok, "{range}: {:#?}", report.diagnostics);
        if !ok {
            assert!(
                report
                    .diagnostics
                    .iter()
                    .any(|d| d.severity == Severity::Error && d.message.contains("reward_range")),
                "{range}: {:#?}",
                report.diagnostics
            );
        }
    }
    fs::write(
        root.join("task.toml"),
        toml.replace("reward_range = [-1.0, 1.0]\n", ""),
    )
    .unwrap();
    let report = benchflow_task::check(&root, &CheckOptions { deep: true });
    assert_eq!(report.package.unwrap().verifier.reward_range, None);
}

/// F1-B-r2 L2: the opt-out's first spelling, `[environment]
/// allow_cloud_metadata`, is still an X100 warning that opts nothing out,
/// and its message names the key's new place. No rename is offered: the
/// key moves to another table.
#[test]
fn the_stale_opt_out_names_its_new_place() {
    let d = tempfile::tempdir().unwrap();
    for (rel, text) in [
        (
            "task.toml",
            "schema_version = \"1.4\"\n\n[environment]\nallow_cloud_metadata = true\n",
        ),
        ("instruction.md", "Do it.\n"),
        ("environment/Dockerfile", "FROM ubuntu:24.04\n"),
        ("tests/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
    ] {
        let path = d.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    let report = benchflow_task::check(d.path(), &CheckOptions { deep: true });
    let x100: Vec<&Diagnostic> = report
        .diagnostics
        .iter()
        .filter(|d| d.code == "X100")
        .collect();
    assert_eq!(x100.len(), 1, "{:#?}", report.diagnostics);
    let d = x100[0];
    assert_eq!(d.severity, Severity::Warning);
    let said = format!("{} {:?} {:?}", d.message, d.label, d.help);
    assert!(
        said.contains("`[metadata] x-allow-cloud-metadata`"),
        "{said}"
    );
    assert!(d.fix.is_none(), "{:?}", d.fix);
    assert!(
        !report
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
}

/// F1 (rc.2, round 2): `[metadata] x-allow-cloud-metadata = true` is
/// bench's opt-out from refusing the cloud metadata endpoints; it is read
/// without a warning, and a value that is not a boolean (as the split
/// layout reads one: `"yes"` is) is not. `[environment]` cannot hold it:
/// BenchFlow 0.8 refuses a key it does not know there, so it is an unknown
/// key here too, and opts nothing out.
#[test]
fn the_cloud_metadata_opt_out_is_read() {
    let d = tempfile::tempdir().unwrap();
    let write = |toml: &str| {
        for (rel, text) in [
            ("task.toml", toml),
            ("instruction.md", "Do it.\n"),
            ("environment/Dockerfile", "FROM ubuntu:24.04\n"),
            ("tests/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
        ] {
            let path = d.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        benchflow_task::check(d.path(), &CheckOptions { deep: true })
    };
    let report = write("schema_version = \"1.4\"\n\n[metadata]\nx-allow-cloud-metadata = true\n");
    assert!(report.diagnostics.is_empty(), "{:#?}", report.diagnostics);
    let pkg = report.package.expect("a package");
    assert!(pkg.sandbox.allow_cloud_metadata);
    assert_eq!(pkg.sandbox.network, Network::Open);
    let report = write("schema_version = \"1.4\"\n\n[environment]\nnetwork_mode = \"public\"\n");
    assert!(
        !report
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
    let report =
        write("schema_version = \"1.4\"\n\n[metadata]\nx-allow-cloud-metadata = \"maybe\"\n");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.message.contains("x-allow-cloud-metadata")),
        "{:#?}",
        report.diagnostics
    );
    let report = write("schema_version = \"1.4\"\n\n[environment]\nallow_cloud_metadata = true\n");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "X100" && d.message.contains("allow_cloud_metadata")),
        "{:#?}",
        report.diagnostics
    );
    assert!(
        !report
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
}
