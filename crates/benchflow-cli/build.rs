//! Build information for `bench --version`.
//!
//! `bench --version` names the commit it was built from and that commit's
//! date, the way `rustc --version` does:
//!
//! ```text
//! bench 0.8.1-dev.0 (f5260eaaa 2026-10-10)
//! ```
//!
//! Two places say which commit that is:
//!
//! - **A source archive** (`git archive`, or GitHub's source tarball, which
//!   is made the same way) has no git history, but git writes the commit
//!   into [`ARCHIVE_COMMIT`] as it packs it (`export-subst` in
//!   `.gitattributes`). The file holds `%H`, the full hash: `%h` is as long
//!   as git needs to keep it unique, which grows with the repository, so the
//!   same commit's tarball would change and break anyone pinning its sha256.
//!   In a checkout the file still holds git's placeholder, which is ignored.
//! - **A git checkout:** the script asks `git` where things are instead of
//!   reading `.git` itself: in a linked worktree `.git` is a file that points
//!   elsewhere, and branch refs live in the main repository's git folder.
//!
//! Both print the first 9 characters of the full hash, so an archive build
//! and a checkout build of one commit print the same version. Built anywhere
//! else (no `git` on PATH, a crate unpacked by `cargo package` or from
//! crates.io, a copy inside a repository that does not track this script),
//! it prints just the version. It sets `BENCH_LONG_VERSION`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    println!("cargo::rerun-if-changed=build.rs");
    let long = match archive_commit(&dir).or_else(|| commit(&dir)) {
        Some(commit) => format!("{version} ({commit})"),
        None => version,
    };
    println!("cargo::rustc-env=BENCH_LONG_VERSION={long}");
}

/// The file `git archive` writes the commit into, next to this script.
const ARCHIVE_COMMIT: &str = "archive-commit.txt";

/// How many characters of the hash `bench --version` prints (rc.1 printed
/// `fc2c92dcb`).
const SHORT_HASH: usize = 9;

/// `<short sha> <commit date>` from [`ARCHIVE_COMMIT`], or None while it
/// holds git's placeholder (a checkout) or is missing.
///
/// Read first: only `git archive` fills the file in, so a filled-in file
/// says this tree is that commit's archive, even when it was unpacked inside
/// some unrelated repository (a home folder kept in git) whose HEAD says
/// nothing about this build.
fn archive_commit(dir: &Path) -> Option<String> {
    let path = dir.join(ARCHIVE_COMMIT);
    watch(&path);
    parse_commit(&std::fs::read_to_string(path).ok()?)
}

/// `<short sha> <date>` from `<hash> <YYYY-MM-DD>`, the hash shortened to
/// [`SHORT_HASH`] characters; None for anything else, such as the archive
/// file's unsubstituted `$Format:...$` placeholder.
pub(crate) fn parse_commit(text: &str) -> Option<String> {
    let mut words = text.split_whitespace();
    let (hash, date) = (words.next()?, words.next()?);
    if words.next().is_some() {
        return None;
    }
    // Full hashes are 40 (SHA-1) or 64 (SHA-256) characters; git never
    // abbreviates below 7.
    let hex = |s: &str| s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !(7..=64).contains(&hash.len()) || !hex(hash) {
        return None;
    }
    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    let mut parts = date.split('-');
    let ok_date = matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some(y), Some(m), Some(d), None)
            if y.len() == 4 && m.len() == 2 && d.len() == 2
                && digits(y) && digits(m) && digits(d)
    );
    if !ok_date {
        return None;
    }
    let short = &hash[..hash.len().min(SHORT_HASH)];
    Some(format!("{short} {date}"))
}

/// `<short sha> <commit date>` for HEAD, or None when this is not a git
/// checkout of BenchFlow.
pub(crate) fn commit(dir: &Path) -> Option<String> {
    // A packaged crate carries this file. It may be unpacked inside some
    // unrelated repository (a home folder kept in git), whose HEAD says
    // nothing about this build.
    if dir.join(".cargo_vcs_info.json").exists() {
        return None;
    }
    // The same for a plain copy (rsync, `cp -r`) inside such a repository:
    // only a repository that tracks this script is BenchFlow's (G9-R1-B-3).
    git(dir, &["ls-files", "--error-unmatch", "build.rs"])?;
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
            "--date=short",
            // The full hash, shortened by `parse_commit` as the archive's
            // is: `%h` grows past 9 characters when 9 are ambiguous.
            "--format=%H %cd",
        ],
    )?;
    parse_commit(&line)
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
///
/// No `GIT_*` variable is inherited: hooks, `git rebase -x` and `!` aliases
/// in a linked worktree export `GIT_DIR` without `GIT_WORK_TREE`, and git
/// then takes this crate's folder as the top of the work tree, so the
/// `ls-files` check fails and the build drops its commit (G9F-B-1).
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new("git");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("GIT_") {
            cmd.env_remove(key);
        }
    }
    let out = cmd
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

#[cfg(test)]
mod tests {
    use super::parse_commit;

    #[test]
    fn a_full_hash_is_shortened_to_nine_characters() {
        assert_eq!(
            parse_commit("fc2c92dcb0e1d2c3b4a5968778695a4b3c2d1e0f 2026-10-12\n").as_deref(),
            Some("fc2c92dcb 2026-10-12")
        );
        let sha256 = "ab".repeat(32);
        assert_eq!(
            parse_commit(&format!("{sha256} 2026-10-12")).as_deref(),
            Some("ababababa 2026-10-12")
        );
    }

    #[test]
    fn a_short_hash_is_kept_up_to_nine_characters() {
        assert_eq!(
            parse_commit("fc2c92dcb0e1 2026-10-12").as_deref(),
            Some("fc2c92dcb 2026-10-12")
        );
        assert_eq!(
            parse_commit("fc2c92d 2026-10-12").as_deref(),
            Some("fc2c92d 2026-10-12")
        );
    }

    #[test]
    fn the_unsubstituted_placeholder_is_ignored() {
        // What the file holds in a checkout, and in a tree copied without
        // `git archive`.
        assert_eq!(parse_commit("$Format:%H %cs$\n"), None);
        assert_eq!(parse_commit("$Format:%H %cs$"), None);
    }

    /// The file holds git's placeholder in a checkout and the commit in a
    /// source archive, and the tests run in both: a test that took the
    /// placeholder for granted failed in an archive tree.
    #[test]
    fn the_archive_file_holds_the_placeholder_or_a_commit() {
        let text = include_str!("archive-commit.txt");
        if text.contains("$Format:") {
            assert_eq!(text.trim(), "$Format:%H %cs$");
            assert_eq!(parse_commit(text), None);
        } else {
            let commit = parse_commit(text);
            assert!(commit.is_some(), "{text:?}");
        }
    }

    #[test]
    fn anything_else_is_ignored() {
        for text in [
            "",
            "\n",
            "fc2c92dcb",
            "fc2c92dcb 2026-10-12 extra",
            "fc2c92 2026-10-12",
            "FC2C92DCB 2026-10-12",
            "fc2c92dcg 2026-10-12",
            "fc2c92dcb 2026-1-12",
            "fc2c92dcb 12-10-2026",
            "fc2c92dcb 2026-10-12-01",
            "fc2c92dcb 2026-1o-12",
        ] {
            assert_eq!(parse_commit(text), None, "{text:?}");
        }
        let too_long = "a".repeat(65);
        assert_eq!(parse_commit(&format!("{too_long} 2026-10-12")), None);
    }
}
