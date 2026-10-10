//! Walking, hashing, and copying a package's files the way the reference
//! converter does (Python's `sorted(root.rglob("*"))` and `shutil.copy2`).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// A path relative to a package, by its components.
pub(crate) type Rel = Vec<String>;

/// `a/b/c` for a relative path.
pub(crate) fn posix(rel: &[String]) -> String {
    rel.join("/")
}

/// Split `a/b/c` into components.
pub(crate) fn rel(text: &str) -> Rel {
    text.split('/')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// The path of `rel` under `root`.
pub(crate) fn under(root: &Path, rel: &[String]) -> PathBuf {
    let mut p = root.to_path_buf();
    for part in rel {
        p.push(part);
    }
    p
}

/// Every file in a package except its own `.git`, sorted by path
/// components: real folders are walked, links to files count, links to
/// folders are not followed.
pub(crate) fn list(root: &Path) -> io::Result<Vec<Rel>> {
    let mut out = Vec::new();
    walk(root, &mut Vec::new(), &mut out)?;
    out.retain(|r| r.first().map(String::as_str) != Some(".git"));
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, prefix: &mut Vec<String>, out: &mut Vec<Rel>) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if prefix.is_empty() => return Err(e),
        Err(_) => return Ok(()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let ty = entry.file_type()?;
        prefix.push(name);
        if ty.is_dir() {
            walk(&entry.path(), prefix, out)?;
        } else if entry.path().is_file() {
            out.push(prefix.clone());
        }
        prefix.pop();
    }
    Ok(())
}

/// Every file under a folder, `.git` included, sorted by path components.
pub(crate) fn list_all(root: &Path) -> Vec<Rel> {
    let mut out = Vec::new();
    let _ = walk(root, &mut Vec::new(), &mut out);
    out.sort();
    out
}

/// Whether a folder holds at least one file, at any depth.
pub(crate) fn has_files(dir: &Path) -> bool {
    dir.is_dir() && list(dir).is_ok_and(|f| !f.is_empty())
}

/// `sha256:<hex>` of bytes.
pub(crate) fn sha256_bytes(data: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(data)))
}

/// `sha256:<hex>` of a file.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

/// Copy a file with its permissions, making parent folders.
pub(crate) fn copy(from: &Path, to: &Path) -> io::Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(from, to)?;
    Ok(())
}

/// Write a file, making parent folders.
pub(crate) fn write(to: &Path, data: &[u8]) -> io::Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(to, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_files_by_components_without_root_git() {
        let dir = tempfile::tempdir().unwrap();
        for f in ["a/c", "a.b/c", ".git/HEAD", "x/.git/keep", "top"] {
            write(&dir.path().join(f), b"x").unwrap();
        }
        let files: Vec<String> = list(dir.path()).unwrap().iter().map(|r| posix(r)).collect();
        assert_eq!(files, vec!["a/c", "a.b/c", "top", "x/.git/keep"]);
    }
}
