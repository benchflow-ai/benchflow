//! Differential tests: the task.md checker against task-md's reference
//! checker (`tools/taskmd.py`), on every package under
//! `tests/fixtures/taskmd/{examples,imported,cases}`.
//!
//! `golden/<group>/<name>.json` and `.check.txt` are what the reference's
//! `json` and `check` commands print for each package (`regen.sh` records
//! them). For each package this test compares:
//!
//! - errors and warnings: the same messages, word for word, in the same
//!   files, and on the same lines where the reference names a line;
//! - the parsed config, the blocks, the instruction and its source, the
//!   canary, the title, the mode, and whether the package is ok;
//! - the facts about grading, line for line.
//!
//! Two kinds of message are compared by their first words only: a TOML or
//! JSON syntax error, since the message after `not valid TOML:` comes from
//! the parsing library.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use benchflow_task::check::CheckOptions;
use benchflow_task::diag::{Diagnostic, Severity};
use benchflow_task::taskmd;
use regex::Regex;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/taskmd")
}

/// Every fixture package: (group, name, folder).
fn packages() -> Vec<(String, String, PathBuf)> {
    let mut out = Vec::new();
    for group in ["examples", "imported", "cases"] {
        let root = fixtures().join(group);
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            out.push((group.to_string(), name.clone(), root.join(&name)));
        }
    }
    out
}

/// A diagnostic as both checkers can say it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Said {
    level: &'static str,
    message: String,
    file: String,
    line: Option<usize>,
}

/// Messages whose tail comes from a parsing library; and one where the
/// reference prints a placeholder unformatted (its message is an f-string
/// joined to a plain string, so it says `({why})` where this port says why).
fn normalize_message(message: &str) -> String {
    for prefix in ["config block is not valid TOML:", "not valid JSON:"] {
        if message.starts_with(prefix) {
            return prefix.to_string();
        }
    }
    const NO_OFFSET: &str =
        "clock start is an RFC 3339 time with an offset, such as \"2026-03-14T09:00:00-04:00\" (";
    if let Some(i) = message.find(NO_OFFSET) {
        return format!("{}{{why}})", &message[..i + NO_OFFSET.len()]);
    }
    message.to_string()
}

static WHERE_LINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([^\s:]+):(\d+)$").unwrap());

/// The reference's `where`: `task.md:12`, `verifier/rubric.json criteria[3]`,
/// `[integrity] canary` (a config key, in task.md), or nothing.
fn reference_said(d: &Value) -> Said {
    let level = match d["level"].as_str() {
        Some("error") => "error",
        Some("warning") => "warning",
        other => panic!("unknown level {other:?}"),
    };
    let message = normalize_message(d["message"].as_str().unwrap_or_default());
    let place = d["where"].as_str().unwrap_or("task.md");
    let (file, line) = if place.starts_with('[') {
        ("task.md".to_string(), None)
    } else if let Some(caps) = WHERE_LINE.captures(place) {
        (caps[1].to_string(), caps[2].parse().ok())
    } else {
        (place.split(' ').next().unwrap_or(place).to_string(), None)
    };
    let line = if message.starts_with("config block is not valid TOML") {
        None
    } else {
        line
    };
    Said {
        level,
        message,
        file,
        line,
    }
}

/// The Rust diagnostic in the reference's terms: its file is named the way
/// the reference names it when the reference gives only a file name.
fn rust_said(d: &Diagnostic, reference_file: Option<&str>, reference_line: bool) -> Said {
    let loc = d.location.as_ref();
    let mut file = loc.map_or_else(
        || "task.md".to_string(),
        |l| l.file.to_string_lossy().replace('\\', "/"),
    );
    if let Some(r) = reference_file
        && !r.contains('/')
        && file.ends_with(&format!("/{r}"))
    {
        file = r.to_string();
    }
    let message = normalize_message(&d.message);
    let line = if reference_line && !message.starts_with("config block is not valid TOML") {
        loc.and_then(|l| l.line)
    } else {
        None
    };
    Said {
        level: if d.severity == Severity::Error {
            "error"
        } else {
            "warning"
        },
        message,
        file,
        line,
    }
}

static DATETIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|z|[+-]\d{2}:\d{2})?$")
        .unwrap()
});

/// TOML date-times are strings on both sides, written differently: the
/// reference prints Python's `str(datetime)`.
fn normalize_config(v: &mut Value) {
    match v {
        Value::String(s) => {
            if let Some(c) = DATETIME.captures(s) {
                let frac = c.get(3).map_or(String::new(), |f| {
                    let micros: String = f
                        .as_str()
                        .chars()
                        .chain(std::iter::repeat('0'))
                        .take(6)
                        .collect();
                    if micros == "000000" {
                        String::new()
                    } else {
                        format!(".{micros}")
                    }
                });
                let offset = match c.get(4).map(|m| m.as_str()) {
                    Some("Z" | "z") => "+00:00".to_string(),
                    Some(o) => o.to_string(),
                    None => String::new(),
                };
                *s = format!("{} {}{frac}{offset}", &c[1], &c[2]);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(normalize_config),
        Value::Object(m) => m.values_mut().for_each(normalize_config),
        _ => {}
    }
}

/// The grading lines `check` printed.
fn reference_facts(check: &str) -> Vec<String> {
    check
        .lines()
        .filter_map(|l| l.strip_prefix("  "))
        .filter(|l| {
            l.starts_with("grading needs a model:")
                || l.starts_with("judge ")
                || l.starts_with("worst-case grading time:")
                || l.starts_with("episodes:")
        })
        .map(str::to_string)
        .collect()
}

/// What differs for one package, if anything.
fn compare(group: &str, name: &str, dir: &Path) -> Vec<String> {
    let golden = fixtures().join("golden").join(group);
    let json_text =
        std::fs::read_to_string(golden.join(format!("{name}.json"))).expect("json golden");
    let reference: Value =
        serde_json::from_str(&json_text).unwrap_or_else(|e| panic!("{group}/{name}.json: {e}"));
    let check_text =
        std::fs::read_to_string(golden.join(format!("{name}.check.txt"))).expect("check golden");
    let mut problems = Vec::new();

    let report = taskmd::check(dir, &CheckOptions { deep: true });
    let text = std::fs::read_to_string(dir.join("task.md")).expect("task.md");
    let doc = taskmd::parse(&text, Path::new("task.md"));

    // Diagnostics.
    let ref_diags = reference["diagnostics"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut want: Vec<Said> = ref_diags.iter().map(reference_said).collect();
    let got_diags: Vec<&Diagnostic> = report
        .diagnostics
        .iter()
        .filter(|d| d.severity != Severity::Note)
        .collect();
    // Pair each Rust diagnostic with the reference's naming of its file, by message.
    let mut got: Vec<Said> = got_diags
        .iter()
        .map(|d| {
            let m = normalize_message(&d.message);
            let r = want.iter().find(|w| w.message == m);
            rust_said(
                d,
                r.map(|w| w.file.as_str()),
                r.is_some_and(|w| w.line.is_some()),
            )
        })
        .collect();
    want.sort();
    got.sort();
    if want != got {
        let missing: Vec<&Said> = want.iter().filter(|w| !got.contains(w)).collect();
        let extra: Vec<&Said> = got.iter().filter(|g| !want.contains(g)).collect();
        problems.push(format!(
            "diagnostics differ\n  missing: {missing:#?}\n  extra: {extra:#?}"
        ));
    }
    if reference["ok"].as_bool() != Some(report.ok()) {
        problems.push(format!(
            "ok: reference {}, rust {}",
            reference["ok"],
            report.ok()
        ));
    }

    // The parse.
    let mut want_config = reference["config"].clone();
    let mut got_config = doc.config_value().clone();
    normalize_config(&mut want_config);
    normalize_config(&mut got_config);
    if want_config != got_config {
        problems.push(format!(
            "config differs\n  reference: {want_config}\n  rust:      {got_config}"
        ));
    }
    let checks: [(&str, Value, Value); 6] = [
        (
            "instruction",
            reference["instruction"].clone(),
            Value::from(doc.instruction.clone()),
        ),
        (
            "instruction_source",
            reference["instruction_source"].clone(),
            Value::from(doc.instruction_source.clone()),
        ),
        (
            "canary",
            reference["canary"].clone(),
            doc.canary.clone().map_or(Value::Null, Value::from),
        ),
        ("mode", reference["mode"].clone(), Value::from(doc.mode())),
        (
            "config_source",
            reference["config_source"].clone(),
            Value::from(doc.config_source.as_str()),
        ),
        (
            "title",
            reference["title"].clone(),
            doc.title().map_or(Value::Null, Value::from),
        ),
    ];
    for (what, want, got) in checks {
        if want != got {
            problems.push(format!("{what}: reference {want}, rust {got}"));
        }
    }
    let want_blocks: Vec<Value> = reference["blocks"].as_array().cloned().unwrap_or_default();
    let got_blocks: Vec<Value> = doc
        .blocks
        .iter()
        .map(|b| {
            serde_json::json!({"kind": b.kind.as_str(), "arg": b.arg, "info": b.info, "body": b.body, "line": b.line})
        })
        .collect();
    if want_blocks != got_blocks {
        problems.push(format!(
            "blocks differ\n  reference: {want_blocks:?}\n  rust:      {got_blocks:?}"
        ));
    }
    if doc.diagnostics.iter().all(|d| !d.is_error()) {
        let view = taskmd::agent_view(dir).expect("agent view");
        if view
            != format!(
                "{}\n",
                reference["instruction"].as_str().unwrap_or_default()
            )
        {
            problems.push("agent view differs".into());
        }
    }

    // What grading needs.
    let want_facts = reference_facts(&check_text);
    if want_facts != report.facts {
        problems.push(format!(
            "facts differ\n  reference: {want_facts:#?}\n  rust:      {:#?}",
            report.facts
        ));
    }
    problems
}

#[test]
fn the_checker_agrees_with_the_reference() {
    let mut failures = Vec::new();
    let all = packages();
    assert!(
        all.len() > 150,
        "the fixtures are missing: run tests/fixtures/taskmd/regen.sh"
    );
    for (group, name, dir) in &all {
        let problems = compare(group, name, dir);
        if !problems.is_empty() {
            failures.push(format!("== {group}/{name}\n{}", problems.join("\n")));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} packages differ from the reference:\n\n{}",
        failures.len(),
        all.len(),
        failures.join("\n\n")
    );
}

#[test]
fn every_example_checks_clean() {
    for (group, name, dir) in packages() {
        if group != "examples" {
            continue;
        }
        let report = taskmd::check(&dir, &CheckOptions { deep: true });
        let errors: Vec<&str> = report
            .diagnostics
            .iter()
            .filter(|d| d.is_error())
            .map(|d| d.message.as_str())
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:#?}");
        assert!(report.package.is_some(), "{name}: no package");
    }
}
