//! Build information for `bench --version`.
//!
//! Built from a git checkout, `bench --version` names the commit and its
//! date, the way `rustc --version` does:
//!
//! ```text
//! bench 0.8.1-dev.0 (f5260eaaa 2026-10-10)
//! ```
//!
//! Built anywhere else (no `git` on PATH, a source archive, a crate unpacked
//! by `cargo package` or from crates.io), it prints just the version. The
//! script asks `git` where things are instead of reading `.git` itself: in a
//! linked worktree `.git` is a file that points elsewhere, and branch refs
//! live in the main repository's git folder. It sets `BENCH_LONG_VERSION`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    println!("cargo::rerun-if-changed=build.rs");
    let long = match commit(&dir) {
        Some(commit) => format!("{version} ({commit})"),
        None => version,
    };
    println!("cargo::rustc-env=BENCH_LONG_VERSION={long}");
}

/// `<short sha> <commit date>` for HEAD, or None when this is not a git
/// checkout of BenchFlow.
fn commit(dir: &Path) -> Option<String> {
    // A packaged crate carries this file. It may be unpacked inside some
    // unrelated repository (a home folder kept in git), whose HEAD says
    // nothing about this build.
    if dir.join(".cargo_vcs_info.json").exists() {
        return None;
    }
    let git_dir = dir.join(git(dir, &["rev-parse", "--git-dir"])?);
    watch_head(dir, &git_dir);
    let line = git(
        dir,
        &[
            "-c",
            "log.showSignature=false",
            "log",
            "-1",
            "--no-color",
            "--abbrev=9",
            "--date=short",
            "--format=%h %cd",
        ],
    )?;
    (!line.is_empty()).then_some(line)
}

/// Rerun this script when HEAD moves: a commit, a checkout, a reset, a pull.
/// HEAD is per worktree; the branch it names is shared, so ask git where the
/// branch's file is.
fn watch_head(dir: &Path, git_dir: &Path) {
    let head = git_dir.join("HEAD");
    watch(&head);
    let Ok(text) = std::fs::read_to_string(&head) else {
        return;
    };
    // A detached HEAD holds the sha itself, so watching HEAD is enough.
    let Some(name) = text.strip_prefix("ref:").map(str::trim) else {
        return;
    };
    let Some(paths) = git(
        dir,
        &["rev-parse", "--git-path", name, "--git-path", "packed-refs"],
    ) else {
        return;
    };
    let mut paths = paths.lines().map(|p| dir.join(p));
    let (Some(loose), Some(packed)) = (paths.next(), paths.next()) else {
        return;
    };
    if loose.is_file() {
        watch(&loose);
    } else {
        // The branch is only in packed-refs. Its next commit writes the loose
        // file, so watch the folder it will appear in.
        watch(&packed);
        if let Some(folder) = loose.ancestors().skip(1).find(|p| p.is_dir()) {
            watch(folder);
        }
    }
}

/// Cargo reruns a script on every build while a path it watches is missing,
/// so only watch paths that exist.
fn watch(path: &Path) {
    if path.exists() {
        println!("cargo::rerun-if-changed={}", path.display());
    }
}

/// Runs `git` in `dir` and returns its trimmed stdout, or None when git is
/// missing, fails, or this is not a repository.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}
