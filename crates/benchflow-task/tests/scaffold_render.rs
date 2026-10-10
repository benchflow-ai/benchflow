//! `bench new`'s templates: each renders a package that checks clean, whose
//! oracle earns the full reward from its own verifier, and whose empty run
//! earns nothing.
//!
//! The runs happen on the host with the sandbox paths (`/app`,
//! `/logs/verifier`) moved into a temporary folder: the oracle and the
//! verifier scripts run under `bash`, and the python template's tests run
//! under `python3` through a small stand-in for pytest that writes the same
//! CTRF names. A run is skipped, with a note, when `bash` or `python3` is
//! missing.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use benchflow_task::ctrf::Ctrf;
use benchflow_task::rubric::Rubric;
use benchflow_task::scaffold::{self, ScaffoldError};
use benchflow_task::{CheckOptions, Format};

fn render(template: &str, name: &str) -> (tempfile::TempDir, PathBuf, Vec<PathBuf>) {
    let work = tempfile::tempdir().unwrap();
    let dir = work.path().join("task");
    let files = scaffold::render(template, name, &dir).unwrap();
    (work, dir, files)
}

#[test]
fn every_template_writes_its_files_with_the_name_filled_in() {
    for t in scaffold::templates() {
        let (_work, dir, files) = render(t.name, "acme/parse-logs");
        let expected: Vec<PathBuf> = t.paths().map(|p| dir.join(p)).collect();
        assert_eq!(files, expected, "{}", t.name);
        for f in &files {
            let text = fs::read_to_string(f).unwrap();
            assert!(!text.contains("{{"), "{} keeps a placeholder", f.display());
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = fs::metadata(f).unwrap().permissions().mode();
                let script = f.extension().is_some_and(|e| e == "sh");
                assert_eq!(mode & 0o111 != 0, script, "{} executable", f.display());
            }
        }
        let config = if t.name == "split" {
            fs::read_to_string(dir.join("task.toml")).unwrap()
        } else {
            fs::read_to_string(dir.join("task.md")).unwrap()
        };
        assert!(
            config.contains("name = \"acme/parse-logs\""),
            "{}: {config}",
            t.name
        );
    }
}

#[test]
fn every_template_sets_a_memory_limit() {
    // Without one, `bench test` on a new task warns that the sandbox may
    // use all the host's memory.
    for t in scaffold::templates() {
        let (_work, dir, _) = render(t.name, "acme/parse-logs");
        let task = benchflow_task::load(&dir).unwrap_or_else(|d| panic!("{}: {d:?}", t.name));
        assert_eq!(task.sandbox.memory_mb, Some(2048), "{}", t.name);
    }
}

#[test]
fn a_plain_name_is_filed_under_local_with_a_title() {
    let (_work, dir, _) = render("basic", "parse-logs");
    let text = fs::read_to_string(dir.join("task.md")).unwrap();
    assert!(text.contains("name = \"local/parse-logs\""), "{text}");
    assert!(text.contains("title = \"Parse logs\""), "{text}");
}

#[test]
fn render_refuses_a_folder_that_is_not_empty() {
    let work = tempfile::tempdir().unwrap();
    let dir = work.path().join("task");
    fs::create_dir_all(&dir).unwrap();
    // An empty folder is fine.
    scaffold::render("basic", "hello", &dir).unwrap();
    let before = fs::read_to_string(dir.join("task.md")).unwrap();
    let e = scaffold::render("python", "other", &dir).unwrap_err();
    assert!(matches!(e, ScaffoldError::Exists(_)), "{e}");
    assert_eq!(
        fs::read_to_string(dir.join("task.md")).unwrap(),
        before,
        "nothing was overwritten"
    );
    let file = work.path().join("file");
    fs::write(&file, "x").unwrap();
    let e = scaffold::render("basic", "hello", &file).unwrap_err();
    assert!(matches!(e, ScaffoldError::Exists(_)), "{e}");
}

#[test]
fn every_template_checks_clean() {
    for t in scaffold::templates() {
        let (_work, dir, _) = render(t.name, "acme/hello-task");
        let report = benchflow_task::check(&dir, &CheckOptions { deep: true });
        let expected = if t.name == "split" {
            Format::Split
        } else {
            Format::TaskMd
        };
        assert_eq!(report.format, Some(expected), "{}", t.name);
        assert!(
            report.diagnostics.is_empty(),
            "{}: {:#?}",
            t.name,
            report.diagnostics
        );
        let package = report.package.expect("a clean package loads");
        assert!(package.digest.starts_with("sha256:"), "{}", t.name);
        assert!(
            package.unsupported.is_empty(),
            "{}: {:?}",
            t.name,
            package.unsupported
        );
    }
}

#[test]
fn the_split_template_imports_to_a_clean_task_md() {
    let (work, dir, _) = render("split", "acme/hello-task");
    let out = work.path().join("imported");
    benchflow_task::convert::import_to_taskmd(&dir, &out, false).unwrap();
    let report = benchflow_task::check(&out, &CheckOptions { deep: true });
    let errors: Vec<_> = report.diagnostics.iter().filter(|d| d.is_error()).collect();
    assert!(errors.is_empty(), "{errors:#?}");
}

/// The host tool `name`, or `None` (with a note) when it is not installed.
fn tool(name: &str) -> Option<&str> {
    let found = Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !found {
        eprintln!("note: {name} is not installed, so this run is skipped");
    }
    found.then_some(name)
}

/// A sandbox stand-in: `app/` for the agent's workdir and `logs/verifier/`.
struct Host {
    _root: tempfile::TempDir,
    app: PathBuf,
    logs: PathBuf,
}

impl Host {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let app = root.path().join("app");
        let logs = root.path().join("logs/verifier");
        fs::create_dir_all(&app).unwrap();
        fs::create_dir_all(&logs).unwrap();
        Self {
            _root: root,
            app,
            logs,
        }
    }

    /// `text` with the sandbox paths moved here.
    fn remap(&self, text: &str) -> String {
        text.replace("/logs/verifier/", &format!("{}/", self.logs.display()))
            .replace("/app/", &format!("{}/", self.app.display()))
    }

    /// Run a script of the package under bash, with its paths moved here.
    fn run_script(&self, script: &Path) {
        let text = self.remap(&fs::read_to_string(script).unwrap());
        let status = Command::new("bash")
            .arg("-c")
            .arg(text)
            .current_dir(&self.app)
            .status()
            .unwrap();
        assert!(status.success(), "{} failed", script.display());
    }
}

fn grade(dir: &Path, host: &Host) -> f64 {
    let rubric = Rubric::from_path(&dir.join("verifier/rubric.json")).unwrap();
    let ctrf = Ctrf::from_path(&host.logs.join("ctrf.json"));
    rubric.grade(&ctrf, &BTreeMap::new()).reward
}

#[test]
fn basic_oracle_passes_and_an_empty_run_fails() {
    if tool("bash").is_none() {
        return;
    }
    let (_work, dir, _) = render("basic", "acme/hello-task");
    let solved = Host::new();
    solved.run_script(&dir.join("oracle/solve.sh"));
    solved.run_script(&dir.join("verifier/test.sh"));
    assert_eq!(grade(&dir, &solved), 1.0);

    let empty = Host::new();
    empty.run_script(&dir.join("verifier/test.sh"));
    assert_eq!(grade(&dir, &empty), 0.0);

    let wrong = Host::new();
    fs::write(wrong.app.join("hello.txt"), "Hello, world!").unwrap();
    wrong.run_script(&dir.join("verifier/test.sh"));
    assert_eq!(grade(&dir, &wrong), 0.0, "no final newline fails the gate");
}

#[test]
fn split_oracle_passes_and_an_empty_run_fails() {
    if tool("bash").is_none() {
        return;
    }
    let (_work, dir, _) = render("split", "acme/hello-task");
    let solved = Host::new();
    solved.run_script(&dir.join("solution/solve.sh"));
    solved.run_script(&dir.join("tests/test.sh"));
    let rewards = benchflow_task::reward::read_rewards(&solved.logs).unwrap();
    assert_eq!(rewards.reward, 1.0);

    let empty = Host::new();
    empty.run_script(&dir.join("tests/test.sh"));
    let rewards = benchflow_task::reward::read_rewards(&empty.logs).unwrap();
    assert_eq!(rewards.reward, 0.0);
}

/// Runs each `test_*` function of a pytest file and writes the CTRF report
/// pytest-json-ctrf would: one test per function, named `file::function`.
const PYTEST_STAND_IN: &str = r#"
import importlib.util, json, sys
path, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("test_outputs", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tests = []
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)()
        status = "passed"
    except Exception:
        status = "failed"
    tests.append({"name": "test_outputs.py::" + name, "status": status})
with open(out, "w") as f:
    json.dump({"results": {"tool": {"name": "pytest"}, "tests": tests}}, f)
"#;

fn run_python_tests(dir: &Path, host: &Host) {
    let tests = host.app.parent().unwrap().join("test_outputs.py");
    let text = fs::read_to_string(dir.join("verifier/test_outputs.py")).unwrap();
    fs::write(&tests, host.remap(&text)).unwrap();
    let status = Command::new("python3")
        .arg("-I")
        .arg("-c")
        .arg(PYTEST_STAND_IN)
        .arg(&tests)
        .arg(host.logs.join("ctrf.json"))
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn python_oracle_passes_and_an_empty_run_fails() {
    if tool("bash").is_none() || tool("python3").is_none() {
        return;
    }
    let (_work, dir, _) = render("python", "acme/scores");
    let solved = Host::new();
    // The Dockerfile copies the data into the workdir.
    fs::copy(
        dir.join("sandbox/scores.csv"),
        solved.app.join("scores.csv"),
    )
    .unwrap();
    solved.run_script(&dir.join("oracle/solve.sh"));
    run_python_tests(&dir, &solved);
    assert_eq!(grade(&dir, &solved), 1.0);

    let empty = Host::new();
    fs::copy(dir.join("sandbox/scores.csv"), empty.app.join("scores.csv")).unwrap();
    run_python_tests(&dir, &empty);
    assert_eq!(grade(&dir, &empty), 0.0);

    let partial = Host::new();
    fs::write(
        partial.app.join("summary.json"),
        r#"{"count": 5, "mean": 81, "max": {"name": "Grace", "score": 97}}"#,
    )
    .unwrap();
    run_python_tests(&dir, &partial);
    assert_eq!(
        grade(&dir, &partial),
        0.5,
        "the gate passes, and count and max earn 2 of 4 points"
    );
}
