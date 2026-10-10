//! Taking custody of what an agent left: unpacking the outputs archive that
//! [`crate::hardening::pack_outputs`] built in the sandbox, on the host,
//! accepting only regular files and folders at safe relative paths.

use std::io;
use std::path::{Component, Path, PathBuf};

/// What was unpacked.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Unpacked {
    pub files: usize,
    pub bytes: u64,
    /// Entries refused: absolute paths, `..`, links, and special files.
    pub refused: Vec<String>,
    /// Files [`unpack_mapped`] was told to leave out.
    pub left_out: Vec<String>,
}

/// Unpack `archive` into `dest`, keeping each entry's path (relative to the
/// sandbox's `/`). Never follows or creates links; never writes outside
/// `dest`.
pub fn unpack(archive: &Path, dest: &Path) -> io::Result<Unpacked> {
    unpack_mapped(archive, dest, &|rel| Some(rel.to_path_buf()))
}

/// Unpack `archive` into `dest` as [`unpack`] does, with each entry's safe
/// relative path passed through `map`: the path under `dest` to write it
/// at, or `None` to leave it out (counted in [`Unpacked::left_out`]).
/// Each file keeps its modification time.
pub fn unpack_mapped(
    archive: &Path,
    dest: &Path,
    map: &dyn Fn(&Path) -> Option<PathBuf>,
) -> io::Result<Unpacked> {
    let file = std::fs::File::open(archive)?;
    let mut ar = tar::Archive::new(file);
    let mut out = Unpacked::default();
    std::fs::create_dir_all(dest)?;
    for entry in ar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(rel) = safe_relative(&path) else {
            out.refused.push(path.display().to_string());
            continue;
        };
        let kind = entry.header().entry_type();
        let Some(rel) = map(&rel).and_then(|r| safe_relative(&r)) else {
            if kind.is_file() {
                out.left_out.push(rel.display().to_string());
            }
            continue;
        };
        let target = dest.join(&rel);
        if kind.is_dir() {
            std::fs::create_dir_all(&target)?;
        } else if kind.is_file() {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Refuse to write through anything that is already a link.
            if std::fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink()) {
                out.refused.push(path.display().to_string());
                continue;
            }
            let mut f = std::fs::File::create(&target)?;
            let n = io::copy(&mut entry, &mut f)?;
            out.files += 1;
            out.bytes += n;
            if let Ok(t) = entry.header().mtime() {
                let _ = f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(t));
            }
            #[cfg(unix)]
            if let Ok(mode) = entry.header().mode() {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(
                    &target,
                    std::fs::Permissions::from_mode(mode & 0o755),
                );
            }
        } else {
            out.refused.push(path.display().to_string());
        }
    }
    Ok(out)
}

/// The entries of an outputs archive that a verifier's sandbox must not get:
/// anything but regular files and folders, unsafe paths, and paths outside
/// `allowed` (absolute sandbox paths; folders above one of them pass). The
/// archive is unpacked as root at `/`, so an entry elsewhere could replace
/// what the verifier runs.
pub fn stray_entries(archive: &Path, allowed: &[String]) -> io::Result<Vec<String>> {
    let mut ar = tar::Archive::new(std::fs::File::open(archive)?);
    let mut stray = Vec::new();
    for entry in ar.entries()? {
        let entry = entry?;
        let path = entry.path()?.into_owned();
        let kind = entry.header().entry_type();
        if !(kind.is_file() || kind.is_dir()) || !within(&path, kind.is_dir(), allowed) {
            stray.push(path.display().to_string());
        }
    }
    Ok(stray)
}

/// The paths under `folder` (an `outputs/` folder, laid out from `/`) that
/// [`stray_entries`] would refuse.
pub fn stray_files(folder: &Path, allowed: &[String]) -> Vec<String> {
    walkdir::WalkDir::new(folder)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|e| {
            let rel = e.path().strip_prefix(folder).ok()?;
            let kind = e.file_type();
            let ok = (kind.is_file() || kind.is_dir()) && within(rel, kind.is_dir(), allowed);
            (!ok).then(|| rel.display().to_string())
        })
        .collect()
}

/// Whether the relative path `rel` is at or under one of `allowed`, or is a
/// folder above one.
fn within(rel: &Path, is_dir: bool, allowed: &[String]) -> bool {
    let Some(rel) = safe_relative(rel) else {
        return false;
    };
    allowed.iter().any(|a| {
        let a = Path::new(a.trim_start_matches('/'));
        rel.starts_with(a) || (is_dir && a.starts_with(&rel))
    })
}

/// A path with no root, no prefix, and no `..`; `None` otherwise.
fn safe_relative(p: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive(entries: &[(&str, &[u8])], link: Option<(&str, &str)>) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut b = tar::Builder::new(f.reopen().unwrap());
        for (name, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            // Bypass the builder's path checks to write hostile names.
            let gnu = h.as_gnu_mut().unwrap();
            let bytes = name.as_bytes();
            gnu.name[..bytes.len()].copy_from_slice(bytes);
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        if let Some((name, target)) = link {
            let mut h = tar::Header::new_gnu();
            h.set_entry_type(tar::EntryType::Symlink);
            h.set_size(0);
            b.append_link(&mut h, name, target).unwrap();
        }
        b.finish().unwrap();
        f
    }

    #[test]
    fn unpacks_regular_files_and_refuses_the_rest() {
        let a = archive(
            &[
                ("app/hello.txt", b"Hello, world!"),
                ("../escape.txt", b"x"),
                ("/etc/passwd", b"x"),
            ],
            Some(("app/secret", "/etc/shadow")),
        );
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert_eq!(u.files, 1);
        assert_eq!(
            std::fs::read_to_string(d.path().join("app/hello.txt")).unwrap(),
            "Hello, world!"
        );
        assert_eq!(u.refused.len(), 3, "{:?}", u.refused);
        assert!(!d.path().join("app/secret").exists());
        assert!(!d.path().parent().unwrap().join("escape.txt").exists());
    }
}
