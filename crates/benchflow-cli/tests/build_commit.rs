//! The commit in `bench --version`, from a git checkout and from a source
//! archive. `build.rs` is compiled in here too, so its own unit tests run
//! (cargo does not test build scripts).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `git` run in `dir`, with no `GIT_*` variable inherited from the caller.
/// Hooks, `git rebase -x` and `!` aliases export `GIT_DIR` (and hooks
/// `GIT_INDEX_FILE`); inherited, they would point these commands at the
/// caller's repository instead of the one `dir` is in (G9F-B-2).
fn git_in(dir: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.current_dir(dir);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("GIT_") {
            cmd.env_remove(key);
        }
    }
    cmd
}

/// Whether `git` runs at all: `build.rs` builds without it, so the tests
/// that need it skip (G9F-A-5).
fn have_git() -> bool {
    let ok = git_in(Path::new("."))
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipped: no git on PATH");
    }
    ok
}

/// A scratch repository in `dir`, made with no global or system config and
/// no inherited `GIT_*` variable; `run` panics when a git command fails.
struct Scratch {
    dir: PathBuf,
    config: PathBuf,
}

impl Scratch {
    fn new(dir: &Path, config_dir: &Path) -> Scratch {
        let config = config_dir.join("gitconfig");
        std::fs::write(&config, "").unwrap();
        Scratch {
            dir: dir.to_path_buf(),
            config,
        }
    }

    fn run(&self, args: &[&str]) -> String {
        let out = git_in(&self.dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(args)
            .env("GIT_CONFIG_GLOBAL", &self.config)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }
}

/// A copy of this crate's `build.rs` and placeholder archive file in `dir`.
fn copy_crate(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::copy(src.join("build.rs"), dir.join("build.rs")).unwrap();
    std::fs::write(dir.join("archive-commit.txt"), "$Format:%H %cs$\n").unwrap();
}

/// `git` in the repository, or None when this is no git checkout (the tests
/// themselves may run from a source archive, which has no history).
fn git(args: &[&str]) -> Option<String> {
    let out = git_in(&repo())
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The archive file, from the repository root.
const ARCHIVE_FILE: &str = "crates/benchflow-cli/archive-commit.txt";

/// HEAD's full hash and commit date, as the archive file is filled in, or
/// why the checkout tests skip: this is a source archive (its commit file is
/// filled in), or no checkout that tracks this crate. A plain copy may sit
/// inside some unrelated repository (a home folder kept in git), whose HEAD
/// is not this tree's commit and has no such file (G9-R1-A-1).
fn head() -> Result<String, &'static str> {
    let file = std::fs::read_to_string(repo().join(ARCHIVE_FILE)).map_err(|_| "no archive file")?;
    if !file.contains("$Format:") {
        return Err("a source archive");
    }
    git(&["ls-files", "--error-unmatch", ARCHIVE_FILE])
        .ok_or("not a git checkout of this repository")?;
    git(&[
        "-c",
        "log.showSignature=false",
        "log",
        "-1",
        "--format=%H %cs",
    ])
    .ok_or("not a git checkout of this repository")
}

/// A checkout commits the archive file as git's placeholder, never filled
/// in: a filled-in file (a tarball unpacked over a clone and committed)
/// would make every later checkout build name that stale commit, and would
/// make the checkout tests here skip as if in a source archive (G9-R1-B-2).
#[test]
fn a_checkout_commits_the_archive_file_as_the_placeholder() {
    if git(&["ls-files", "--error-unmatch", ARCHIVE_FILE]).is_none() {
        eprintln!("skipped: not a git checkout of this repository");
        return;
    }
    // `./` keeps the path relative to this checkout's folder, as `ls-files`
    // reads it, when the repository tracks it in a subfolder (G9F-A-3).
    let committed = git(&["show", &format!("HEAD:./{ARCHIVE_FILE}")]).unwrap();
    assert_eq!(committed, "$Format:%H %cs$");
}

/// A copy of the crate without `.git` inside a repository that does not
/// track it names no commit: that repository's HEAD says nothing about this
/// build (G9-R1-B-3).
#[test]
fn a_copy_inside_another_repository_names_no_commit() {
    if !have_git() {
        return;
    }
    let outer = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    let crate_dir = outer.path().join("home/benchflow/crates/benchflow-cli");
    std::fs::write(outer.path().join("notes.txt"), "mine\n").unwrap();
    let git = Scratch::new(outer.path(), config.path());
    git.run(&["init", "-q"]);
    git.run(&["add", "notes.txt"]);
    git.run(&["commit", "-q", "-m", "outer"]);
    copy_crate(&crate_dir);
    assert_eq!(build_script::commit(&crate_dir), None);
}

/// Set by [`a_linked_worktree_build_names_head_under_an_exported_git_dir`]
/// for the copy of this test binary it runs.
const PROBE: &str = "BENCH_BUILD_COMMIT_PROBE";

/// Prints `build.rs`'s answer for the folder in [`PROBE`]; does nothing in a
/// normal run.
#[test]
fn probe_commit_under_the_callers_environment() {
    let Some(dir) = std::env::var_os(PROBE) else {
        return;
    };
    let commit = build_script::commit(Path::new(&dir));
    println!("probe-commit={}", commit.as_deref().unwrap_or("none"));
}

/// Hooks, `git rebase -x 'cargo test'` and `!` aliases in a linked worktree
/// export `GIT_DIR` and no `GIT_WORK_TREE`. The build still names the
/// worktree's HEAD, as it did before the `ls-files` check (G9F-B-1).
/// `build.rs` reads the process environment, so the check runs in a child
/// copy of this test binary with `GIT_DIR` set there only.
#[test]
fn a_linked_worktree_build_names_head_under_an_exported_git_dir() {
    if !have_git() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    let main = root.path().join("main");
    let crate_path = "crates/benchflow-cli";
    copy_crate(&main.join(crate_path));
    let git = Scratch::new(&main, config.path());
    git.run(&["init", "-q"]);
    git.run(&["add", "."]);
    git.run(&["commit", "-q", "-m", "base"]);
    let wt = root.path().join("wt");
    git.run(&["worktree", "add", "-q", "-b", "other", wt.to_str().unwrap()]);
    let in_wt = Scratch::new(&wt, config.path());
    in_wt.run(&["commit", "-q", "--allow-empty", "-m", "worktree"]);
    let head = in_wt.run(&["log", "-1", "--format=%H %cs"]);
    let git_dir = in_wt.run(&["rev-parse", "--absolute-git-dir"]);
    assert_ne!(head, git.run(&["log", "-1", "--format=%H %cs"]));

    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "probe_commit_under_the_callers_environment",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE, wt.join(crate_path))
        .env("GIT_DIR", &git_dir)
        .env_remove("GIT_WORK_TREE")
        .env("GIT_CONFIG_GLOBAL", &in_wt.config)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .current_dir(wt.join(crate_path))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout}");
    let (hash, date) = head.split_once(' ').unwrap();
    let expected = format!("probe-commit={} {date}", &hash[..9]);
    assert!(
        stdout.lines().any(|l| l == expected),
        "want {expected:?}: {stdout}"
    );
}

/// `git archive` fills in the commit file (`export-subst`), with the full
/// hash, and `build.rs` reads it as the commit a checkout build names.
#[test]
fn git_archive_writes_the_commit_into_the_archive_file() {
    let head = match head() {
        Ok(head) => head,
        Err(why) => {
            eprintln!("skipped: {why}");
            return;
        }
    };
    let file = ARCHIVE_FILE;
    let mut sh = Command::new("sh");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("GIT_") {
            sh.env_remove(key);
        }
    }
    let out = sh
        .arg("-c")
        .arg(r#"git archive --format=tar HEAD -- "$1" | tar -xOf - "$1""#)
        .arg("sh")
        .arg(file)
        .current_dir(repo())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let archived = String::from_utf8(out.stdout).unwrap();
    assert_eq!(archived.trim(), head, "the archive's {file}");
    let (hash, date) = head.split_once(' ').unwrap();
    assert_eq!(
        build_script::parse_commit(&archived),
        Some(format!("{} {date}", &hash[..9]))
    );
}

/// A checkout build names HEAD with the same 9 characters an archive build
/// prints.
#[test]
fn a_checkout_build_names_heads_first_nine_characters() {
    let head = match head() {
        Ok(head) => head,
        Err(why) => {
            eprintln!("skipped: {why}");
            return;
        }
    };
    let (hash, date) = head.split_once(' ').unwrap();
    let out = assert_cmd::Command::cargo_bin("bench")
        .unwrap()
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    let version = String::from_utf8(out.stdout).unwrap();
    let expected = format!(
        "bench {} ({} {date})",
        env!("CARGO_PKG_VERSION"),
        &hash[..9]
    );
    assert_eq!(version.trim(), expected);
}
