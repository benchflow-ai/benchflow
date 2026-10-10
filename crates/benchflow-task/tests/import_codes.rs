//! One case per diagnostic code of the formats BenchFlow imports, and of
//! packages: X1xx for Harbor's split layout (task.toml + instruction.md +
//! environment/ + tests/ + solution/), X2xx for BenchFlow 0.8's task.md
//! (YAML frontmatter, then the prompt), and P0xx for any package. Each case
//! writes the smallest package that hits its code the way authors most
//! often do, checks it with `benchflow_task::check`, and asserts that the
//! code fires and that nothing else does, but for the codes the case names.
//!
//! `tests/codes/import.txt` keeps what a user reads for each case: the
//! code, severity, and message, then where it points, the label under the
//! caret, the notes, the help, and the fix `bench check --fix` applies. A
//! change in wording shows up in review; regenerate it with
//! `UPDATE_GOLDEN=1 cargo test -p benchflow-task --test import_codes`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use benchflow_task::diag::{Diagnostic, Explanation, Severity};
use benchflow_task::{CheckOptions, CheckReport};
use tempfile::TempDir;

/// The package a case starts from.
#[derive(Debug, Clone, Copy)]
enum Base {
    /// A Harbor task in the split layout that checks clean.
    Split,
    /// A BenchFlow 0.8 task.md package that checks clean.
    V06,
    /// An empty folder.
    Empty,
}

/// A file's text that means "leave this file of the base out".
const DELETE: &str = "\0";

const INSTRUCTION: &str = "Fix the parser in /app so that `make test` passes.\n";
const DOCKERFILE: &str = "FROM python:3.12-slim\nWORKDIR /app\nCOPY . /app\n";
const TEST_SH: &str = "#!/bin/bash\nmake -C /app test && echo 1 > /logs/verifier/reward.txt \
                       || echo 0 > /logs/verifier/reward.txt\n";

const SPLIT_TOML: &str = r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[agent]
timeout_sec = 600

[environment]
cpus = 2
memory_mb = 4096
"#;

const SPLIT: &[(&str, &str)] = &[
    ("task.toml", SPLIT_TOML),
    ("instruction.md", INSTRUCTION),
    ("environment/Dockerfile", DOCKERFILE),
    ("tests/test.sh", TEST_SH),
    (
        "solution/solve.sh",
        "#!/bin/bash\ngit -C /app apply /solution/fix.diff\n",
    ),
];

const V06_TASK_MD: &str = r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
agent:
  timeout_sec: 600
sandbox:
  cpus: 2
  memory_mb: 4096
---
Fix the parser in /app so that `make test` passes.
"#;

const V06: &[(&str, &str)] = &[
    ("task.md", V06_TASK_MD),
    ("environment/Dockerfile", DOCKERFILE),
    ("verifier/test.sh", TEST_SH),
    (
        "oracle/solve.sh",
        "#!/bin/bash\ngit -C /app apply /oracle/fix.diff\n",
    ),
];

fn base_files(base: Base) -> &'static [(&'static str, &'static str)] {
    match base {
        Base::Split => SPLIT,
        Base::V06 => V06,
        Base::Empty => &[],
    }
}

struct Case {
    code: &'static str,
    base: Base,
    /// Files written over the base; `DELETE` leaves a base file out.
    files: &'static [(&'static str, &'static str)],
    /// A file of the package that cannot be read while it is checked.
    unreadable: Option<&'static str>,
    /// Other codes the package draws, for the reason given at the case.
    also: &'static [&'static str],
}

fn case(code: &'static str, base: Base, files: &'static [(&'static str, &'static str)]) -> Case {
    Case {
        code,
        base,
        files,
        unreadable: None,
        also: &[],
    }
}

fn cases() -> Vec<Case> {
    vec![
        // P0xx: any package.
        //
        // A Harbor task copied without its task.toml.
        case(
            "P001",
            Base::Empty,
            &[
                ("instruction.md", INSTRUCTION),
                ("environment/Dockerfile", DOCKERFILE),
                ("tests/test.sh", TEST_SH),
            ],
        ),
        // A file the checking user cannot read, such as one a container
        // wrote into the task as root with mode 600.
        Case {
            unreadable: Some("environment/data/fixtures.db"),
            ..case(
                "P002",
                Base::Split,
                &[("environment/data/fixtures.db", "rows\n")],
            )
        },
        // X1xx: the split layout.
        //
        // A misspelled key, which Harbor ignores without a word.
        case(
            "X100",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[environment]
network_mod = "no-network"
"#,
            )],
        ),
        // A sandbox size under [agent].
        case(
            "X101",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[agent]
timeout_sec = 600
cpus = 2
"#,
            )],
        ),
        // Harbor's old size spelling, as SkillsBench tasks write it.
        case(
            "X102",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[environment]
cpus = 2
memory = "4G"
"#,
            )],
        ),
        // The old network switch left beside the new one.
        case(
            "X103",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[environment]
network_mode = "no-network"
allow_internet = false
"#,
            )],
        ),
        // A BenchFlow 0.7 task.toml: [sandbox] for [environment].
        case(
            "X104",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "acme/fix-parser"

[sandbox]
cpus = 2
memory_mb = 4096
"#,
            )],
        ),
        // A string without quotes, as YAML would take it.
        case(
            "X110",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = acme/fix-parser
"#,
            )],
        ),
        // A task name without the organization Harbor requires.
        case(
            "X111",
            Base::Split,
            &[(
                "task.toml",
                r#"schema_version = "1.4"

[task]
name = "fix-parser"
"#,
            )],
        ),
        // The instruction left out.
        case("X112", Base::Split, &[("instruction.md", DELETE)]),
        // A pytest file, but not the test.sh that runs it.
        case(
            "X113",
            Base::Split,
            &[
                ("tests/test.sh", DELETE),
                (
                    "tests/test_outputs.py",
                    "def test_parser():\n    assert True\n",
                ),
            ],
        ),
        // No environment/ folder, and no docker_image.
        case("X114", Base::Split, &[("environment/Dockerfile", DELETE)]),
        // X2xx: BenchFlow 0.8's task.md.
        //
        // A colon inside an unquoted value.
        case(
            "X200",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
  description: Fix the parser: it drops the last line
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // A key 0.8 does not define, which it refuses.
        case(
            "X201",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
agent:
  timeout: 600
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // A time limit written as a duration.
        case(
            "X202",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
agent:
  timeout_sec: 10m
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // Harbor's table name in a 0.8 task.md.
        case(
            "X203",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
environment:
  cpus: 2
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // The prompt split under two `## prompt` headings. Only X204: the
        // prompt is there, twice, not empty.
        case(
            "X204",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
---
## prompt

Fix the parser in /app.

## Prompt

Run `make test` to check your work.
"#,
            )],
        ),
        // A misspelled profile.
        case(
            "X205",
            Base::V06,
            &[(
                "task.md",
                r#"---
profile: code-changes
task:
  name: acme/fix-parser
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // A default strategy that names no strategy.
        case(
            "X206",
            Base::V06,
            &[(
                "verifier/verifier.md",
                r#"---
verifier:
  default_strategy: tests
  strategies:
    deterministic:
      type: script
      command: ./test.sh
---
"#,
            )],
        ),
        // The Harbor task.toml left behind after converting to task.md.
        case("X207", Base::V06, &[("task.toml", SPLIT_TOML)]),
        // No environment/ folder, and no prebuilt image.
        case("X208", Base::V06, &[("environment/Dockerfile", DELETE)]),
        // A scene whose turn misspells the role it hands to.
        case(
            "X209",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
agents:
  roles:
    solver:
      agent: claude-agent-acp
scenes:
  - name: fix
    turns:
      - role: solvr
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // The prompt still holds `bench tasks init`'s placeholder.
        case(
            "X210",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
---
[REPLACE: what the agent must do, in plain words]
"#,
            )],
        ),
        // The old network switch left beside the new one.
        case(
            "X211",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
sandbox:
  network_mode: no-network
  allow_internet: false
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // The old network switch, as tasks converted from SkillsBench have it.
        case(
            "X212",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/fix-parser
sandbox:
  cpus: 1
  allow_internet: true
---
Fix the parser in /app so that `make test` passes.
"#,
            )],
        ),
        // A task built by the robouse task-format plugin.
        case(
            "X213",
            Base::V06,
            &[(
                "task.md",
                r#"---
schema_version: "1.3"
task:
  name: acme/tidy-kitchen
robouse:
  arena: kitchen
---
Put the cups in the cupboard.
"#,
            )],
        ),
    ]
}

/// Keeps a file unreadable while it lives, and makes it readable again on
/// drop, so the temporary folder can be removed.
#[cfg_attr(not(unix), allow(dead_code))]
struct Unreadable(PathBuf);

impl Unreadable {
    #[cfg(unix)]
    fn new(path: &Path) -> Result<Self, String> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o000))
            .map_err(|e| format!("cannot make {} unreadable: {e}", path.display()))?;
        let guard = Self(path.to_path_buf());
        if std::fs::File::open(path).is_ok() {
            return Err(format!(
                "{} is still readable with mode 000; run the tests as a user other than root",
                path.display()
            ));
        }
        Ok(guard)
    }

    #[cfg(not(unix))]
    fn new(path: &Path) -> Result<Self, String> {
        Err(format!(
            "making {} unreadable needs Unix permissions",
            path.display()
        ))
    }
}

impl Drop for Unreadable {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644));
        }
    }
}

/// A case's package, checked. The folder is removed on drop.
struct Checked {
    dir: TempDir,
    report: CheckReport,
}

impl Checked {
    /// The diagnostics with the case's own code.
    fn of(&self, code: &str) -> Vec<&Diagnostic> {
        self.report
            .diagnostics
            .iter()
            .filter(|d| d.code == code)
            .collect()
    }
}

/// Write the case's package into a new folder and check it.
fn check(case: &Case) -> Result<Checked, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let mut files: BTreeMap<&str, &str> = base_files(case.base).iter().copied().collect();
    for &(rel, text) in case.files {
        if text == DELETE {
            files.remove(rel);
        } else {
            files.insert(rel, text);
        }
    }
    for (rel, text) in files {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().expect("a file in the package"))
            .map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
    }
    let guard = match case.unreadable {
        Some(rel) => Some(Unreadable::new(&dir.path().join(rel))?),
        None => None,
    };
    let report = benchflow_task::check(dir.path(), &CheckOptions { deep: true });
    drop(guard);
    Ok(Checked { dir, report })
}

/// A path for the golden file: relative to the package, `.` for the
/// package itself. Locations are relative to the package; one that names
/// an absolute path must lie under `root`.
fn relative(root: &Path, file: &Path) -> Option<String> {
    let rel = if file.is_absolute() {
        file.strip_prefix(root).ok()?
    } else if file
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return None;
    } else {
        file
    };
    let text = rel.to_string_lossy().replace('\\', "/");
    Some(if text.is_empty() { ".".into() } else { text })
}

/// What a user reads for one diagnostic.
fn shown(root: &Path, d: &Diagnostic) -> String {
    let severity = match d.severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    };
    let mut s = String::new();
    let _ = writeln!(s, "{} {severity}: {}", d.code, d.message);
    if let Some(loc) = &d.location {
        let file = relative(root, &loc.file).unwrap_or_else(|| loc.file.display().to_string());
        match loc.line {
            Some(line) => {
                let _ = writeln!(s, "  at {file}:{line}");
            }
            None => {
                let _ = writeln!(s, "  at {file}");
            }
        }
    }
    if let Some(label) = &d.label {
        let _ = writeln!(s, "  label: {label}");
    }
    for note in &d.notes {
        let _ = writeln!(s, "  note: {note}");
    }
    if let Some(help) = &d.help {
        let _ = writeln!(s, "  help: {help}");
    }
    if let Some(fix) = &d.fix {
        let _ = writeln!(s, "  fix: {}", fix.description);
    }
    // A message that names the temporary folder would change on every run.
    s.replace(&root.display().to_string(), "<package>")
}

/// Why a case does not hold, if it does not.
fn verify(case: &Case, checked: &Checked) -> Result<(), String> {
    let found = || -> Vec<String> {
        checked
            .report
            .diagnostics
            .iter()
            .map(|d| format!("{} {}", d.code, d.message))
            .collect()
    };
    if checked.of(case.code).is_empty() {
        return Err(format!("{}: not emitted; found {:#?}", case.code, found()));
    }
    let expected: BTreeSet<&str> = std::iter::once(case.code)
        .chain(case.also.iter().copied())
        .collect();
    let emitted: BTreeSet<&str> = checked.report.diagnostics.iter().map(|d| d.code).collect();
    if emitted != expected {
        return Err(format!(
            "{}: the package draws {emitted:?}, want {expected:?}: {:#?}",
            case.code,
            found()
        ));
    }
    for d in checked.of(case.code) {
        if let Some(loc) = &d.location
            && loc.file.is_absolute()
        {
            return Err(format!(
                "{}: names {} by its full path; callers join locations to the package",
                case.code,
                loc.file.display()
            ));
        }
        if let Some(loc) = &d.location
            && relative(checked.dir.path(), &loc.file).is_none()
        {
            return Err(format!(
                "{}: points outside the package, at {}",
                case.code,
                loc.file.display()
            ));
        }
    }
    Ok(())
}

#[test]
fn every_code_fires() {
    let mut failures = Vec::new();
    for case in cases() {
        let result = check(&case).and_then(|checked| verify(&case, &checked));
        if let Err(e) = result {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn every_explained_code_has_a_case() {
    let tables: [&[Explanation]; 3] = [
        benchflow_task::split::EXPLANATIONS,
        benchflow_task::v06::EXPLANATIONS,
        benchflow_task::load::EXPLANATIONS,
    ];
    let explained: BTreeSet<&str> = tables
        .iter()
        .flat_map(|t| t.iter())
        .map(|e| e.code)
        .collect();
    let mut failures = Vec::new();
    let mut tested = BTreeSet::new();
    for case in cases() {
        if !tested.insert(case.code) {
            failures.push(format!("{} has two cases", case.code));
        }
        for also in case.also {
            if benchflow_task::diag::explain(also).is_none() {
                failures.push(format!("{}: {also} has no explanation", case.code));
            }
        }
    }
    // Codes `bench check` emits itself, after this crate's check: they are
    // triggered through the binary (crates/benchflow-cli/tests/check_codes.rs).
    const CLI_CODES: &[&str] = &["P003"];
    for code in explained.difference(&tested) {
        if !CLI_CODES.contains(code) {
            failures.push(format!("{code} is explained, but no case triggers it"));
        }
    }
    for code in tested.difference(&explained) {
        failures.push(format!("{code} has a case, but no table here explains it"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_base_packages_check_clean() {
    for base in [Base::Split, Base::V06] {
        let checked = check(&case("", base, &[])).unwrap();
        assert!(
            checked.report.diagnostics.is_empty(),
            "{base:?}: {:#?}",
            checked.report.diagnostics
        );
        assert!(checked.report.package.is_some(), "{base:?}: no package");
    }
}

#[test]
fn what_users_read_matches_the_golden_file() {
    let mut cases = cases();
    cases.sort_by_key(|c| c.code);
    let mut blocks = Vec::new();
    let mut failures = Vec::new();
    for case in &cases {
        let checked = match check(case) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{}: {e}", case.code));
                continue;
            }
        };
        let mine = checked.of(case.code);
        if mine.is_empty() {
            failures.push(format!("{}: not emitted", case.code));
        }
        for d in mine {
            blocks.push(shown(checked.dir.path(), d));
        }
    }
    let golden = blocks.join("\n");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/codes/import.txt");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &golden).unwrap();
    } else {
        let want = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        if want != golden {
            failures.push(format!(
                "tests/codes/import.txt is stale; run with UPDATE_GOLDEN=1 and review the diff. Got:\n{golden}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The diagnostic `code` draws for its case.
fn drawn(code: &str) -> Diagnostic {
    let case = cases()
        .into_iter()
        .find(|c| c.code == code)
        .unwrap_or_else(|| panic!("no case for {code}"));
    let checked = check(&case).unwrap_or_else(|e| panic!("{code}: {e}"));
    checked
        .of(code)
        .first()
        .map(|d| (*d).clone())
        .unwrap_or_else(|| panic!("{code} not drawn"))
}

/// Each says what happened where the user looks, and a step that works.
#[test]
fn messages_point_at_the_problem_and_say_what_to_do() {
    // P001: no `bench import` of a folder that holds no task.
    let help = drawn("P001").help.unwrap_or_default();
    assert!(
        help.contains("bench new") && !help.contains("import"),
        "{help}"
    );
    // P002: the file that cannot be read, and what to do about it.
    let d = drawn("P002");
    assert!(
        d.message.contains("environment/data/fixtures.db"),
        "{}",
        d.message
    );
    assert!(d.help.is_some());
    // X200: the line in task.md, not in the frontmatter.
    let d = drawn("X200");
    assert_eq!(
        d.location.as_ref().and_then(|l| l.line),
        Some(5),
        "{}",
        d.message
    );
    assert!(
        d.message.contains("at task.md line 5 column 30"),
        "{}",
        d.message
    );
    // X204: the second heading, and no claim that the prompt is empty.
    let d = drawn("X204");
    assert_eq!(d.location.as_ref().and_then(|l| l.line), Some(10));
    // X205: the profile it was most likely meant to be.
    assert!(
        drawn("X205")
            .message
            .contains("did you mean \"code-change\"?")
    );
    // X206: the strategy named, and the ones there are.
    let m = drawn("X206").message;
    assert!(
        m.contains("\"tests\"") && m.contains("deterministic"),
        "{m}"
    );
    // X207: what to do with the file nothing reads.
    assert!(
        drawn("X207")
            .help
            .is_some_and(|h| h.contains("remove task.toml"))
    );
    // X212: the spelling to write, in YAML.
    let d = drawn("X212");
    assert!(d.message.ends_with("network_mode: public"), "{}", d.message);
    assert_eq!(
        d.help.as_deref(),
        Some("write `network_mode: public` instead")
    );
}
