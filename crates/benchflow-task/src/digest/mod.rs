//! Content digests of task packages: one `sha256:<hex>` string that changes
//! when any byte of the package changes, computed the way each format's own
//! tools compute it, so a result recorded by Harbor, BenchFlow 0.8, or
//! task.md's tools can be matched to the folder it ran.
//!
//! | Format | Digest | Files | Line per file |
//! |---|---|---|---|
//! | Split layout | [`harbor_digest`], Harbor's packager | task.toml, instruction.md, README.md, trajectory.json, and everything under environment/, tests/, solution/, steps/, minus `.gitignore` (or Harbor's default ignores) | `{path}\0{sha256 hex}\n` |
//! | task.md draft 2 | [`tree_hash`], task.md's tree hash | every regular file except evidence/, changes.json, and .git at the root | `{path}\0{32-byte sha256}` |
//! | BenchFlow 0.8 task.md | [`benchflow_digest`], 0.8's `task_digest` | every regular file except `.benchflow-source.json` | `{path}\0{32-byte sha256}` |
//!
//! Paths are relative to the package and `/`-separated, files are sorted by
//! path, and symbolic links never count (Harbor's digest reads a link to a
//! file as the file, since its packager follows it).

mod gitignore;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub use gitignore::GitIgnore;

use crate::package::Format;

/// One file in a digest: its relative path as bytes, and where it is.
struct Entry {
    rel: Vec<u8>,
    path: PathBuf,
}

/// The digest of the package at `dir`, by its format: Harbor's packager
/// digest for the split layout, the tree hash for task.md, and 0.8's
/// `task_digest` for a 0.8 task.md. A folder that is not a package is an
/// `InvalidInput` error.
pub fn digest(dir: &Path) -> io::Result<String> {
    match crate::load::detect(dir) {
        Some(format) => digest_as(dir, format),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} is not a task package (no task.md or task.toml)",
                dir.display()
            ),
        )),
    }
}

/// The digest of the package at `dir` in a format already detected.
pub fn digest_as(dir: &Path, format: Format) -> io::Result<String> {
    match format {
        Format::Split => harbor_digest(dir),
        Format::TaskMd => tree_hash(dir),
        Format::TaskMdV06 => benchflow_digest(dir),
    }
}

/// Harbor's package digest (`Packager.compute_content_hash`): the files
/// Harbor publishes, filtered by the task's `.gitignore` (or, without one,
/// Harbor's default ignores: `__pycache__/`, `*.pyc`, `.DS_Store`, `*.swp`,
/// `*.swo`, `*~`), sorted by path, each hashed into the line
/// `{path}\0{sha256 hex}\n` of an outer SHA-256.
pub fn harbor_digest(dir: &Path) -> io::Result<String> {
    let mut files: Vec<Entry> = Vec::new();
    for single in [
        "task.toml",
        "instruction.md",
        "README.md",
        "trajectory.json",
    ] {
        let path = dir.join(single);
        if path.is_file() {
            files.push(Entry {
                rel: single.as_bytes().to_vec(),
                path,
            });
        }
    }
    for folder in ["environment", "tests", "solution", "steps"] {
        let top = dir.join(folder);
        if top.is_dir() {
            harbor_walk(&top, folder.as_bytes(), &mut files)?;
        }
    }
    let ignore = match fs::read(dir.join(".gitignore")) {
        Ok(bytes) => GitIgnore::parse(&String::from_utf8_lossy(&bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => GitIgnore::harbor_defaults(),
        Err(e) => return Err(e),
    };
    files.retain(|f| !ignore.is_ignored(&String::from_utf8_lossy(&f.rel)));
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    let mut outer = Sha256::new();
    for f in &files {
        outer.update(&f.rel);
        outer.update(b"\0");
        outer.update(hex::encode(entry_sha256(f)?).as_bytes());
        outer.update(b"\n");
    }
    Ok(format!("sha256:{}", hex::encode(outer.finalize())))
}

/// Python's `Path.rglob("*")` followed by `is_file()`: descend real folders
/// only, and count regular files and links to regular files.
fn harbor_walk(dir: &Path, rel: &[u8], out: &mut Vec<Entry>) -> io::Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry?;
        let name = os_bytes(&entry.file_name());
        let child_rel = join_rel(rel, &name);
        let ty = entry.file_type()?;
        let path = entry.path();
        if ty.is_dir() {
            harbor_walk(&path, &child_rel, out)?;
        } else if ty.is_file() || (ty.is_symlink() && path.is_file()) {
            out.push(Entry {
                rel: child_rel,
                path,
            });
        }
    }
    Ok(())
}

/// task.md's tree hash (`tools/evidence.py`): every regular file, hidden
/// ones and nested repositories included, except `evidence/`,
/// `changes.json`, and `.git` at the package root. Each file adds its path,
/// a zero byte, and the 32-byte SHA-256 of its contents.
pub fn tree_hash(dir: &Path) -> io::Result<String> {
    if !dir.is_dir() {
        return Err(not_a_dir(dir));
    }
    let mut files = Vec::new();
    plain_walk(
        dir,
        b"",
        true,
        &mut |name, is_dir, top| {
            if !top {
                return true;
            }
            if is_dir {
                name != b"evidence" && name != b".git"
            } else {
                name != b"changes.json" && name != b".git"
            }
        },
        &mut files,
    )?;
    hash_entries(files)
}

/// BenchFlow 0.8's `task_digest`: every regular file except ones named
/// `.benchflow-source.json`. Each file adds its path, a zero byte, and the
/// 32-byte SHA-256 of its contents. It byte-matches the digests SkillsBench
/// publishes in its dataset registry.
pub fn benchflow_digest(dir: &Path) -> io::Result<String> {
    if !dir.is_dir() {
        return Err(not_a_dir(dir));
    }
    let mut files = Vec::new();
    plain_walk(
        dir,
        b"",
        true,
        &mut |name, is_dir, _| is_dir || name != b".benchflow-source.json",
        &mut files,
    )?;
    hash_entries(files)
}

/// `os.walk` without following links: regular files only; `keep(name,
/// is_dir, at_root)` drops entries.
fn plain_walk(
    dir: &Path,
    rel: &[u8],
    top: bool,
    keep: &mut dyn FnMut(&[u8], bool, bool) -> bool,
    out: &mut Vec<Entry>,
) -> io::Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry?;
        let name = os_bytes(&entry.file_name());
        let ty = entry.file_type()?;
        if ty.is_symlink() {
            continue;
        }
        if !keep(&name, ty.is_dir(), top) {
            continue;
        }
        let child_rel = join_rel(rel, &name);
        if ty.is_dir() {
            plain_walk(&entry.path(), &child_rel, false, keep, out)?;
        } else if ty.is_file() {
            out.push(Entry {
                rel: child_rel,
                path: entry.path(),
            });
        }
    }
    Ok(())
}

fn hash_entries(mut files: Vec<Entry>) -> io::Result<String> {
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    let mut outer = Sha256::new();
    for f in &files {
        outer.update(&f.rel);
        outer.update(b"\0");
        outer.update(entry_sha256(f)?);
    }
    Ok(format!("sha256:{}", hex::encode(outer.finalize())))
}

/// A package file's SHA-256; an error names the file by its path in the
/// package, so whoever reads it knows which one to fix.
fn entry_sha256(f: &Entry) -> io::Result<Vec<u8>> {
    file_sha256(&f.path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("{}: {e}", String::from_utf8_lossy(&f.rel)),
        )
    })
}

fn file_sha256(path: &Path) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_vec())
}

fn join_rel(rel: &[u8], name: &[u8]) -> Vec<u8> {
    if rel.is_empty() {
        name.to_vec()
    } else {
        let mut v = rel.to_vec();
        v.push(b'/');
        v.extend_from_slice(name);
        v
    }
}

#[cfg(unix)]
fn os_bytes(name: &std::ffi::OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    name.as_bytes().to_vec()
}

#[cfg(not(unix))]
fn os_bytes(name: &std::ffi::OsStr) -> Vec<u8> {
    name.to_string_lossy().into_owned().into_bytes()
}

fn not_a_dir(dir: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotADirectory,
        format!("not a folder: {}", dir.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, body: &[u8]) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    /// The registry construction written out literally, as 0.8's tests do.
    fn spec(entries: &[(&str, &[u8])]) -> String {
        let mut h = Sha256::new();
        for (rel, body) in entries {
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(Sha256::digest(body));
        }
        format!("sha256:{}", hex::encode(h.finalize()))
    }

    #[test]
    fn benchflow_digest_matches_the_spec_and_sorts_by_string() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a/c", b"slash");
        write(dir.path(), "a.b/c", b"dot");
        write(dir.path(), ".benchflow-source.json", b"{}");
        assert_eq!(
            benchflow_digest(dir.path()).unwrap(),
            spec(&[("a.b/c", b"dot"), ("a/c", b"slash")])
        );
    }

    #[test]
    fn tree_hash_skips_root_evidence_and_changes() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "task.md", b"Do it.\n");
        write(dir.path(), "evidence/runs.json", b"[]");
        write(dir.path(), "changes.json", b"[]");
        write(dir.path(), "verifier/evidence/kept.txt", b"k");
        write(dir.path(), "verifier/changes.json", b"k2");
        assert_eq!(
            tree_hash(dir.path()).unwrap(),
            spec(&[
                ("task.md", b"Do it.\n"),
                ("verifier/changes.json", b"k2"),
                ("verifier/evidence/kept.txt", b"k"),
            ])
        );
    }

    #[test]
    fn harbor_digest_reads_only_published_files() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "task.toml", b"[task]\n");
        write(dir.path(), "instruction.md", b"Do it.\n");
        write(dir.path(), "tests/test.sh", b"#!/bin/bash\n");
        write(dir.path(), "tests/__pycache__/x.pyc", b"junk");
        write(dir.path(), "notes.txt", b"not published");
        let mut h = Sha256::new();
        for (rel, body) in [
            ("instruction.md", &b"Do it.\n"[..]),
            ("task.toml", b"[task]\n"),
            ("tests/test.sh", b"#!/bin/bash\n"),
        ] {
            h.update(format!("{rel}\0{}\n", hex::encode(Sha256::digest(body))).as_bytes());
        }
        let expected = format!("sha256:{}", hex::encode(h.finalize()));
        assert_eq!(harbor_digest(dir.path()).unwrap(), expected);
        write(dir.path(), ".gitignore", b"tests/test.sh\n");
        assert_ne!(harbor_digest(dir.path()).unwrap(), expected);
    }

    #[cfg(unix)]
    #[test]
    fn links_never_count_in_tree_hashes() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "task.md", b"x");
        std::os::unix::fs::symlink("task.md", dir.path().join("link.md")).unwrap();
        assert_eq!(tree_hash(dir.path()).unwrap(), spec(&[("task.md", b"x")]));
        assert_eq!(
            benchflow_digest(dir.path()).unwrap(),
            spec(&[("task.md", b"x")])
        );
    }
}
