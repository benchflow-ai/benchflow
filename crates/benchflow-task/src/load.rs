//! Finding, detecting, checking, and loading packages in any format.
//!
//! | Folder holds | Format |
//! |---|---|
//! | `task.md` that does not open with `---` | task.md draft 2, BenchFlow's own ([`crate::taskmd`]); a `task.toml` beside it is an error |
//! | `task.md` that opens with `---` | BenchFlow 0.8's task.md ([`crate::v06`]); a `task.toml` beside it is ignored, as in 0.8 |
//! | `task.toml` (and no `task.md`) | The split layout ([`crate::split`]) |
//!
//! Checking a package also computes its content digest
//! ([`crate::digest::digest`]) the way its own format's tools compute it, so
//! a package from [`check`] or [`load`] is ready to run and to record.

use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::check::{CheckOptions, CheckReport};
use crate::diag::{Diagnostic, Explanation};
use crate::package::{Format, TaskPackage};

/// Diagnostic codes this module emits.
pub static EXPLANATIONS: &[Explanation] = &[
    Explanation {
        code: "P001",
        title: "not a task folder",
        body: "A task folder holds task.md (task.md, or BenchFlow 0.8's task.md with YAML frontmatter) or \
task.toml (the split layout Harbor uses). The path given holds neither, is a file, or does not exist. \
`bench new <name>` creates a task folder.",
    },
    Explanation {
        code: "P002",
        title: "cannot read the package's files",
        body: "Checking a package reads every file in it, to compute the digest that identifies the task's \
version. A file could not be read; the message says which and why. Fix its permissions, or remove it if it \
does not belong to the task.",
    },
    Explanation {
        code: "P003",
        title: "a credential file in the build folder",
        body: "The folder a task's image is built from holds a file that looks like a credential: a .netrc, \
.git-credentials, AWS or cloud SDK credentials, a Docker config with registry logins, an .npmrc with an auth \
token, a .pypirc with a password, or a private SSH key. The build receives every file there, so a COPY can put \
it in the image where the agent can read it, and anyone with the image can. BenchFlow names the file and never \
prints its content. Move it out of the build folder, or list it in .dockerignore; pass secrets a build needs as \
build secrets (RUN --mount=type=secret). `bench test` also scans the built image.",
    },
    Explanation {
        code: "P004",
        title: "the rubric's judge settings cannot be used",
        body: "Rubric criteria with judge \"llm\" are judged on the host by a model, one call per criterion. \
The rubric's `x-judge` names the model as {\"model\": \"<provider>/<model>\"}; its `x-reference`, and each \
criterion's `reference`, is text or a file in a private folder of the package (verifier/reference.md, or \
verifier/answers.toml#total for one value of a TOML or JSON file), which the agent never sees. One of them \
could not be used; the message says which. A run would stop on it with R313.",
    },
    Explanation {
        code: "P005",
        title: "checking the task crashed BenchFlow",
        body: "BenchFlow stopped with an internal error while it checked this task, so the task was neither \
loaded nor run. The message holds the error. This is a BenchFlow bug, whatever the task holds: report it with \
the task folder (or the value that set it off). The other tasks are checked as usual.",
    },
];

/// Folders [`discover`] never enters.
const SKIP: [&str; 7] = [
    ".git",
    "node_modules",
    "target",
    "jobs",
    ".cache",
    "__pycache__",
    ".venv",
];

/// The format of the package rooted at `dir`, if it is one. Reads at most
/// the first bytes of `task.md`.
pub fn detect(dir: &Path) -> Option<Format> {
    let md = dir.join("task.md");
    if md.is_file() {
        let mut head = [0u8; 8];
        let n = fs::File::open(&md)
            .and_then(|mut f| f.read(&mut head))
            .unwrap_or(0);
        let text = String::from_utf8_lossy(&head[..n]);
        return Some(if crate::taskmd::is_draft2(&text) {
            Format::TaskMd
        } else {
            Format::TaskMdV06
        });
    }
    dir.join("task.toml").is_file().then_some(Format::Split)
}

/// Package roots under `root` (or `root` itself when it is one), sorted. A
/// package root is the shallowest folder that holds `task.md` or
/// `task.toml`; nothing inside a package is a package. The walk skips
/// `.git`, `node_modules`, `target`, `jobs`, `.cache`, `__pycache__`, and
/// `.venv`, and does not follow links into folders that are not packages.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    visit(root, true, &mut out);
    out.sort();
    out
}

/// Visit one folder: record it when it is a package, else descend.
fn visit(dir: &Path, descend: bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut subdirs: Vec<(PathBuf, bool)> = Vec::new();
    let mut package = false;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Ok(ty) = entry.file_type() else { continue };
        if (name == OsStr::new("task.md") || name == OsStr::new("task.toml"))
            && (ty.is_file() || (ty.is_symlink() && entry.path().is_file()))
        {
            package = true;
            break;
        }
        if !descend || SKIP.iter().any(|s| name == OsStr::new(s)) {
            continue;
        }
        if ty.is_dir() {
            subdirs.push((entry.path(), true));
        } else if ty.is_symlink() && entry.path().is_dir() {
            subdirs.push((entry.path(), false));
        }
    }
    if package {
        out.push(dir.to_path_buf());
        return;
    }
    for (sub, follow) in subdirs {
        visit(&sub, follow, out);
    }
}

/// Check the package at `dir`, whatever its format, and give its package
/// the content digest its format defines.
/// Diagnostics about the folders a run needs rather than the task's own
/// configuration: a missing build folder or Dockerfile, a missing verifier
/// script. [`check_schema`] leaves them out.
const RUNTIME_ASSET_CODES: &[&str] = &["X113", "X114", "X208"];

/// Check only a task's configuration and text, not the folders a run needs:
/// for authoring fixtures that are only a task file (BenchFlow 0.8's
/// `--level schema`).
pub fn check_schema(dir: &Path, opts: &CheckOptions) -> CheckReport {
    let mut report = check(dir, opts);
    keep_schema_only(&mut report);
    report
}

/// Drop a report's diagnostics about the folders a run needs (see
/// [`check_schema`]).
pub fn keep_schema_only(report: &mut CheckReport) {
    report
        .diagnostics
        .retain(|d| !RUNTIME_ASSET_CODES.contains(&d.code));
}

pub fn check(dir: &Path, opts: &CheckOptions) -> CheckReport {
    let Some(format) = detect(dir) else {
        let mut report = not_a_task(dir);
        relative_locations(&mut report, dir);
        return report;
    };
    let mut report = match format {
        Format::TaskMd => crate::taskmd::check(dir, opts),
        Format::TaskMdV06 => crate::v06::check(dir, opts),
        Format::Split => crate::split::check(dir, opts),
    };
    // task.md's checks name files relative to the package already, and a
    // relative name there may begin with a folder named like the package.
    if format != Format::TaskMd {
        relative_locations(&mut report, dir);
    }

    if let Some(package) = report.package.as_ref() {
        report
            .diagnostics
            .extend(package.extras.diagnostics.iter().cloned());
    }
    if let Some(package) = report.package.as_mut() {
        match crate::digest::digest_as(dir, format) {
            Ok(d) => package.digest = d,
            Err(e) => report.diagnostics.push(
                Diagnostic::error("P002", format!("cannot read a file of the package: {e}"))
                    .in_file(".")
                    .help("make it readable (`chmod a+r <file>`), or remove it if it does not belong to the task"),
            ),
        }
    }
    report
}

/// Name every file relative to the package, as callers expect: the split
/// layout's and 0.8's checks name them under `dir`, so a caller that joins
/// them to `dir` (to show the source line, or to apply a fix) read
/// `dir/dir/task.toml` when `dir` was relative.
fn relative_locations(report: &mut CheckReport, dir: &Path) {
    let relative = |file: &mut PathBuf| {
        if let Ok(rel) = file.strip_prefix(dir) {
            *file = if rel.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                rel.to_path_buf()
            };
        }
    };
    for d in &mut report.diagnostics {
        if let Some(loc) = d.location.as_mut() {
            relative(&mut loc.file);
        }
        if let Some(fix) = d.fix.as_mut() {
            relative(&mut fix.file);
        }
    }
}

fn not_a_task(dir: &Path) -> CheckReport {
    let mut report = CheckReport::new(dir, None);
    let message = if dir.is_dir() {
        "not a task: no task.md or task.toml here"
    } else if dir.exists() {
        "not a task: a file, not a folder"
    } else {
        "not a task: there is no such folder"
    };
    report
        .diagnostics
        .push(Diagnostic::error("P001", message).in_file(dir).help(
        "a task is a folder with task.md (or Harbor's task.toml); `bench new <name>` creates one",
    ));
    report
}

/// Load a package for running, with its content digest. Errors carry every
/// diagnostic.
pub fn load(dir: &Path) -> Result<TaskPackage, Vec<Diagnostic>> {
    let mut report = check(dir, &CheckOptions::default());
    let ok = report.ok();
    match report.package.take() {
        Some(package) if ok => Ok(package),
        _ => Err(report.diagnostics),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    #[test]
    fn detects_formats() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        assert_eq!(detect(d), None);
        write(d, "task.toml", "");
        assert_eq!(detect(d), Some(Format::Split));
        write(d, "task.md", "---\nagent: {}\n---\nDo it.\n");
        assert_eq!(detect(d), Some(Format::TaskMdV06));
        write(d, "task.md", "Do it.\n");
        assert_eq!(detect(d), Some(Format::TaskMd));
        write(d, "task.md", "---\r\na: 1\r\n---\r\nDo it.\r\n");
        assert_eq!(detect(d), Some(Format::TaskMdV06));
    }

    #[test]
    fn discovers_the_shallowest_package_roots() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        write(d, "tasks/a/task.md", "Do a.\n");
        write(d, "tasks/a/sandbox/repo/task.md", "nested, not a package\n");
        write(d, "tasks/b/task.toml", "");
        write(d, "tasks/group/c/task.toml", "");
        write(d, "node_modules/x/task.toml", "");
        write(d, ".git/y/task.toml", "");
        write(d, "jobs/run/task.md", "x");
        let found: Vec<PathBuf> = discover(d)
            .into_iter()
            .map(|p| p.strip_prefix(d).unwrap().to_path_buf())
            .collect();
        assert_eq!(
            found,
            vec![
                PathBuf::from("tasks/a"),
                PathBuf::from("tasks/b"),
                PathBuf::from("tasks/group/c")
            ]
        );
        assert_eq!(discover(&d.join("tasks/a")), vec![d.join("tasks/a")]);
    }

    #[test]
    fn a_folder_without_a_task_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let report = check(dir.path(), &CheckOptions::default());
        assert!(!report.ok());
        assert_eq!(report.diagnostics[0].code, "P001");
        assert!(load(dir.path()).is_err());
        let file = dir.path().join("f");
        write(dir.path(), "f", "");
        let report = check(&file, &CheckOptions::default());
        assert_eq!(
            report.diagnostics[0].message,
            "not a task: a file, not a folder"
        );
        // A path that does not exist says so, and the help names a step that
        // works on it (not an import of a folder that holds no task).
        let report = check(&dir.path().join("moved"), &CheckOptions::default());
        let d = &report.diagnostics[0];
        assert_eq!(d.message, "not a task: there is no such folder");
        assert!(
            d.help
                .as_deref()
                .is_some_and(|h| h.contains("bench new") && !h.contains("import"))
        );
    }

    #[test]
    fn an_unreadable_file_is_named() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "task.md",
            "Do it.\n\n```toml task\nname = \"a/b\"\nversion = \"1.0.0\"\n```\n",
        );
        write(dir.path(), "sandbox/Dockerfile", "FROM ubuntu:24.04\n");
        write(
            dir.path(),
            "verifier/test.sh",
            "echo 1 > /logs/verifier/reward.txt\n",
        );
        write(dir.path(), "data/x.db", "x");
        let secret = dir.path().join("data/x.db");
        fs::set_permissions(&secret, fs::Permissions::from_mode(0o000)).unwrap();
        let report = check(dir.path(), &CheckOptions::default());
        fs::set_permissions(&secret, fs::Permissions::from_mode(0o644)).unwrap();
        if nix_root() {
            return;
        }
        let d = report
            .diagnostics
            .iter()
            .find(|d| d.code == "P002")
            .unwrap_or_else(|| panic!("no P002 in {:#?}", report.diagnostics));
        assert!(d.message.contains("data/x.db"), "{}", d.message);
        assert!(d.help.is_some());
    }

    /// Root reads any file, so an unreadable file cannot be shown as root.
    fn nix_root() -> bool {
        std::env::var("USER").is_ok_and(|u| u == "root")
    }

    #[test]
    fn task_toml_beside_a_draft_2_task_md_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "task.md", "Do it.\n");
        write(dir.path(), "task.toml", "");
        let report = check(dir.path(), &CheckOptions::default());
        assert!(
            report
                .diagnostics
                .iter()
                .any(|d| d.code == crate::taskmd::code::HARBOR_TASK_TOML && d.is_error())
        );
    }

    #[test]
    fn checked_packages_carry_their_formats_digest() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        write(
            d,
            "split/task.toml",
            "schema_version = \"1.4\"\n[environment]\nnetwork_mode = \"no-network\"\n",
        );
        write(d, "split/instruction.md", "Do it.\n");
        write(d, "split/environment/Dockerfile", "FROM ubuntu:24.04\n");
        write(
            d,
            "split/tests/test.sh",
            "echo 1 > /logs/verifier/reward.txt\n",
        );
        let report = check(&d.join("split"), &CheckOptions::default());
        let package = report.package.expect("the split task loads");
        assert_eq!(
            package.digest,
            crate::digest::harbor_digest(&d.join("split")).unwrap()
        );
        assert_eq!(
            load(&d.join("split")).unwrap().digest,
            package.digest,
            "load and check agree"
        );
    }
}
