//! Starter tasks for `bench new`.
//!
//! Each template is a complete package whose reference solution passes its
//! own verifier and whose empty run fails it, so a new task passes
//! `bench check` and `bench test` before its author changes anything:
//!
//! | Template | What it sets up |
//! |---|---|
//! | `basic` (the default) | task.md; a shell verifier that writes a test report; a rubric with one gate; an oracle |
//! | `python` | task.md graded by pytest through its CTRF plugin; a rubric with a gate and points |
//! | `split` | the split layout Harbor reads: `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/` |
//!
//! The files are compiled into the binary from `templates/<name>/`. Two
//! placeholders are filled in: `{{name}}`, the task's full `org/name`, and
//! `{{title}}`, a title made from the name. Shell scripts are written
//! executable.

mod name;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub use name::LOCAL_ORG;

/// A starter task.
#[derive(Debug, Clone, Copy)]
pub struct Template {
    /// What `bench new --template` takes, such as `basic`.
    pub name: &'static str,
    /// One line saying what the template sets up.
    pub description: &'static str,
    files: &'static [(&'static str, &'static str)],
}

impl Template {
    /// The files the template writes, relative to the task folder.
    pub fn paths(&self) -> impl Iterator<Item = &'static str> {
        self.files.iter().map(|(path, _)| *path)
    }
}

/// One template file: its path in the task, and its text.
macro_rules! template_file {
    ($template:literal, $path:literal) => {
        (
            $path,
            include_str!(concat!("../../templates/", $template, "/", $path)),
        )
    };
}

static TEMPLATES: &[Template] = &[
    Template {
        name: "basic",
        description: "task.md with a shell verifier that writes a test report, a one-gate rubric, and an oracle",
        files: &[
            template_file!("basic", "task.md"),
            template_file!("basic", "sandbox/Dockerfile"),
            template_file!("basic", "verifier/test.sh"),
            template_file!("basic", "verifier/rubric.json"),
            template_file!("basic", "oracle/solve.sh"),
        ],
    },
    Template {
        name: "python",
        description: "task.md graded by pytest, with a gate and points in the rubric",
        files: &[
            template_file!("python", "task.md"),
            template_file!("python", "sandbox/Dockerfile"),
            template_file!("python", "sandbox/scores.csv"),
            template_file!("python", "verifier/test.sh"),
            template_file!("python", "verifier/test_outputs.py"),
            template_file!("python", "verifier/rubric.json"),
            template_file!("python", "oracle/solve.sh"),
        ],
    },
    Template {
        name: "split",
        description: "the split layout Harbor reads: task.toml, instruction.md, environment/, tests/, solution/",
        files: &[
            template_file!("split", "task.toml"),
            template_file!("split", "instruction.md"),
            template_file!("split", "environment/Dockerfile"),
            template_file!("split", "tests/test.sh"),
            template_file!("split", "solution/solve.sh"),
        ],
    },
];

/// Every template, the default (`basic`) first.
pub fn templates() -> &'static [Template] {
    TEMPLATES
}

/// Why a task could not be created.
#[derive(Debug, thiserror::Error)]
pub enum ScaffoldError {
    /// No template has this name.
    #[error("there is no template `{name}`{}; the templates are {}", hint(.suggestion), template_list())]
    UnknownTemplate {
        name: String,
        suggestion: Option<String>,
    },
    /// The name is not lowercase letters, digits, and hyphens.
    #[error("{}", invalid_name(.name, .suggestion))]
    InvalidName {
        name: String,
        suggestion: Option<String>,
    },
    /// The folder exists and is not empty, or is not a folder.
    #[error("{} already exists and is not an empty folder; pick another name or folder", .0.display())]
    Exists(PathBuf),
    /// A file could not be written.
    #[error("could not write {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

fn hint(suggestion: &Option<String>) -> String {
    suggestion
        .as_ref()
        .map(|s| format!(" (did you mean `{s}`?)"))
        .unwrap_or_default()
}

fn template_list() -> String {
    let names: Vec<&str> = TEMPLATES.iter().map(|t| t.name).collect();
    names.join(", ")
}

fn invalid_name(name: &str, suggestion: &Option<String>) -> String {
    let rule = "a task name is lowercase letters, digits, and hyphens, optionally after an \
organization and a slash, such as `parse-logs` or `acme/parse-logs`";
    if name.is_empty() {
        return format!("the task needs a name: {rule}");
    }
    format!("`{name}` is not a task name{}: {rule}", hint(suggestion))
}

/// The full `org/name` a new task gets for `name`: `name` itself when it
/// names an organization, else `local/<name>` ([`LOCAL_ORG`]).
pub fn full_name(name: &str) -> Result<String, ScaffoldError> {
    name::qualify(name).ok_or_else(|| ScaffoldError::InvalidName {
        name: name.to_string(),
        suggestion: name::suggest(name),
    })
}

/// Create the task `name` from `template` in `dir`, which must not exist or
/// be empty, and return the files written, as `dir` joined with each path.
///
/// `name` is `parse-logs` or `acme/parse-logs` ([`full_name`]); the task is
/// named `local/parse-logs` in the first case. Nothing is left behind when a
/// write fails.
pub fn render(template: &str, name: &str, dir: &Path) -> Result<Vec<PathBuf>, ScaffoldError> {
    let tpl = TEMPLATES
        .iter()
        .find(|t| t.name == template)
        .ok_or_else(|| ScaffoldError::UnknownTemplate {
            name: template.to_string(),
            suggestion: crate::suggest::did_you_mean(template, TEMPLATES.iter().map(|t| t.name)),
        })?;
    let full = full_name(name)?;
    let title = name::title(&full);
    let created = prepare(dir)?;
    match write_files(tpl, dir, &full, &title) {
        Ok(written) => Ok(written),
        Err(e) => {
            undo(dir, created);
            Err(e)
        }
    }
}

/// Make sure `dir` is an empty folder; true when it had to be created.
fn prepare(dir: &Path) -> Result<bool, ScaffoldError> {
    let io_err = |source| ScaffoldError::Io {
        path: dir.to_path_buf(),
        source,
    };
    if dir.exists() {
        let empty = dir.is_dir() && fs::read_dir(dir).map_err(io_err)?.next().is_none();
        if !empty {
            return Err(ScaffoldError::Exists(dir.to_path_buf()));
        }
        return Ok(false);
    }
    fs::create_dir_all(dir).map_err(io_err)?;
    Ok(true)
}

fn write_files(
    tpl: &Template,
    dir: &Path,
    full: &str,
    title: &str,
) -> Result<Vec<PathBuf>, ScaffoldError> {
    let mut written = Vec::with_capacity(tpl.files.len());
    for (rel, text) in tpl.files {
        let path = dir.join(rel);
        let text = text.replace("{{name}}", full).replace("{{title}}", title);
        write(&path, &text, rel.ends_with(".sh")).map_err(|source| ScaffoldError::Io {
            path: path.clone(),
            source,
        })?;
        written.push(path);
    }
    Ok(written)
}

fn write(path: &Path, text: &str, executable: bool) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, text)?;
    if executable {
        make_executable(path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Remove what a failed render wrote: the folder itself when render created
/// it, else everything in it (it was empty before).
fn undo(dir: &Path, created: bool) {
    if created {
        let _ = fs::remove_dir_all(dir);
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let _ = if path.is_dir() && !path.is_symlink() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_has_a_unique_name_and_files() {
        let mut names: Vec<&str> = templates().iter().map(|t| t.name).collect();
        assert_eq!(names[0], "basic", "the default comes first");
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), templates().len());
        for t in templates() {
            assert!(t.paths().count() > 0, "{} has no files", t.name);
            assert!(!t.description.is_empty());
            assert!(!t.description.contains('\n'), "one line");
        }
    }

    #[test]
    fn unknown_templates_suggest_a_close_one() {
        let dir = tempfile::tempdir().unwrap();
        let e = render("pyhton", "x", &dir.path().join("x")).unwrap_err();
        assert!(matches!(
            &e,
            ScaffoldError::UnknownTemplate { suggestion: Some(s), .. } if s == "python"
        ));
        let msg = e.to_string();
        assert!(msg.contains("did you mean `python`"), "{msg}");
        assert!(msg.contains("basic, python, split"), "{msg}");
        assert!(!dir.path().join("x").exists(), "nothing is written");
    }

    #[test]
    fn invalid_names_say_the_rule_and_a_fix() {
        let dir = tempfile::tempdir().unwrap();
        let e = render("basic", "My_Task", &dir.path().join("t")).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("`My_Task` is not a task name"), "{msg}");
        assert!(msg.contains("did you mean `my-task`"), "{msg}");
        let e = render("basic", "", &dir.path().join("t")).unwrap_err();
        assert!(e.to_string().starts_with("the task needs a name"), "{e}");
        assert!(!dir.path().join("t").exists());
    }

    #[test]
    fn full_names_put_plain_names_under_local() {
        assert_eq!(full_name("hello").unwrap(), "local/hello");
        assert_eq!(full_name("acme/hello").unwrap(), "acme/hello");
        assert!(full_name("Hello").is_err());
    }

    #[test]
    fn a_failed_render_leaves_an_existing_folder_empty() {
        let dir = tempfile::tempdir().unwrap();
        // A folder where sandbox/Dockerfile must go makes its write fail
        // after task.md was written.
        let target = dir.path().join("t");
        fs::create_dir_all(&target).unwrap();
        let tpl = &TEMPLATES[0];
        fs::create_dir_all(target.join("sandbox")).unwrap();
        fs::create_dir_all(target.join("sandbox/Dockerfile")).unwrap();
        let e = write_files(tpl, &target, "local/t", "T").unwrap_err();
        assert!(matches!(e, ScaffoldError::Io { .. }), "{e}");
        undo(&target, false);
        assert!(target.is_dir(), "an existing folder is kept");
        assert_eq!(fs::read_dir(&target).unwrap().count(), 0);
    }
}
