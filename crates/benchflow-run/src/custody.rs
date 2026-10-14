//! Taking custody of what an agent left: the outputs archive that
//! [`crate::hardening::pack_outputs`] built in the sandbox, read on the host.
//!
//! The archive is made by the image's own `tar`, which an agent that is or
//! can become root controls, so nothing in it is trusted. [`filter_outputs`]
//! copies it, archive to archive, keeping only regular files inside the
//! task's outputs, folders strictly inside them, and hard links between such
//! files; a separate verifier's sandbox, a regrade and `bench --diff` get
//! only what it keeps, and `outputs/` is unpacked from the same copy.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, RandomState};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::record::RefusedEntry;

/// The most bytes a GNU long-name or PAX entry may hold: the crate reads
/// each one whole, and an honest one names one path.
pub const MAX_NAME_BYTES: u64 = 64 * 1024;
/// The most members an outputs archive may hold: ten times the files the
/// pack saves ([`crate::hardening::MAX_OUTPUT_FILES`]).
pub const MAX_ENTRIES: u64 = 100_000;
/// The most refused entries listed by name; all are counted.
pub const LISTED_REFUSALS: usize = 50;
/// The most parts a kept path may have: the pack saves at most
/// [`crate::hardening::MAX_OUTPUT_DEPTH`] below an output.
pub const MAX_DEPTH: usize = 128;
/// The most bytes the files and links an outputs archive keeps or unpacks
/// may come to, each name at its file's full size: the copy-out's cap
/// ([`crate::copy_out::ARCHIVE_CAP`]), 64 MiB over what the pack saves.
pub const OUTPUT_BUDGET: u64 = crate::copy_out::ARCHIVE_CAP;

#[derive(Debug, Clone, Copy)]
struct Limits {
    name_bytes: u64,
    entries: u64,
    /// Bytes of files and links, each name at its file's full size.
    bytes: u64,
}

const LIMITS: Limits = Limits {
    name_bytes: MAX_NAME_BYTES,
    entries: MAX_ENTRIES,
    bytes: OUTPUT_BUDGET,
};

/// What was unpacked.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Unpacked {
    pub files: usize,
    pub bytes: u64,
    /// Entries refused: absolute paths, `..`, links other than a hard link
    /// to a file written before it, and special files.
    pub refused: Vec<String>,
    /// Files [`unpack_mapped`] was told to leave out.
    pub left_out: Vec<String>,
    /// Of the refused, files and link copies past the byte budget.
    pub over_budget: u64,
}

/// One member of an archive, with GNU long names and PAX records applied.
struct Member {
    path: PathBuf,
    link: Option<PathBuf>,
    header: tar::Header,
    size: u64,
}

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

/// Read each member of `reader`, raw: long-name, long-link and PAX entries
/// are read here, each at most `limits.name_bytes`, and at most
/// `limits.entries` members are read. Either limit ends the walk with
/// [`io::ErrorKind::FileTooLarge`].
fn walk<R: Read + io::Seek>(
    reader: R,
    limits: Limits,
    each: &mut dyn FnMut(&Member, &mut dyn Read) -> io::Result<()>,
) -> io::Result<()> {
    let mut ar = tar::Archive::new(reader);
    let (mut long_name, mut long_link, mut pax) = (None, None, None);
    let mut members = 0u64;
    for entry in ar.entries_with_seek()?.raw(true) {
        let mut entry = entry?;
        let kind = entry.header().entry_type();
        let slot = if kind.is_gnu_longname() {
            Some(&mut long_name)
        } else if kind.is_gnu_longlink() {
            Some(&mut long_link)
        } else if kind.is_pax_local_extensions() {
            Some(&mut pax)
        } else {
            None
        };
        if (slot.is_some() || kind.is_pax_global_extensions()) && entry.size() > limits.name_bytes {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                format!(
                    "a long-name or PAX entry of {} bytes, over the {} allowed",
                    entry.size(),
                    limits.name_bytes
                ),
            ));
        }
        if let Some(slot) = slot {
            if slot.is_some() {
                return Err(invalid("two long-name or PAX entries describe one member"));
            }
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            // A long name is a C string. Its copy adds the NUL back, so
            // one of `name_bytes` or more without it would make a copy the
            // next read refuses (review F2-RA-2): refused here instead.
            if !kind.is_pax_local_extensions() {
                buf = c_string(buf);
                if buf.len() as u64 >= limits.name_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::FileTooLarge,
                        format!(
                            "a long name of {} bytes, at or over the {} allowed",
                            buf.len(),
                            limits.name_bytes
                        ),
                    ));
                }
            }
            *slot = Some(buf);
            continue;
        }
        members += 1;
        if members > limits.entries {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                format!("more than {} entries", limits.entries),
            ));
        }
        let (mut path, mut link) = (None, None);
        if let Some(records) = pax.take() {
            for ext in tar::PaxExtensions::new(&records) {
                let ext = ext?;
                match ext.key_bytes() {
                    b"path" => path = Some(ext.value_bytes().to_vec()),
                    b"linkpath" => link = Some(ext.value_bytes().to_vec()),
                    _ => {}
                }
            }
        }
        let header = entry.header().clone();
        let path = path
            .or(long_name.take().map(c_string))
            .unwrap_or_else(|| header.path_bytes().into_owned());
        let link = link
            .or(long_link.take().map(c_string))
            .or_else(|| header.link_name_bytes().map(|b| b.into_owned()));
        let member = Member {
            path: bytes_path(path),
            link: link.map(bytes_path),
            size: entry.size(),
            header,
        };
        each(&member, &mut entry)?;
    }
    if long_name.is_some() || long_link.is_some() || pax.is_some() {
        return Err(invalid(
            "the archive ends with a long-name or PAX entry that describes no member",
        ));
    }
    Ok(())
}

/// A GNU long name: the bytes before the first NUL.
fn c_string(mut b: Vec<u8>) -> Vec<u8> {
    if let Some(n) = b.iter().position(|&c| c == 0) {
        b.truncate(n);
    }
    b
}

#[cfg(unix)]
fn bytes_path(b: Vec<u8>) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(std::ffi::OsString::from_vec(b))
}

#[cfg(not(unix))]
fn bytes_path(b: Vec<u8>) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(&b).into_owned())
}

/// Why a folder entry at or above one of the task's outputs is refused.
pub const ABOVE_OUTPUTS: &str = "a folder at or above one of the task's outputs";

/// What [`filter_outputs`] kept and refused.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Filtered {
    /// Entries copied: files, folders and hard links.
    pub kept: u64,
    /// The first [`LISTED_REFUSALS`] entries refused.
    pub refused: Vec<RefusedEntry>,
    /// Every entry refused, listed or not.
    pub refused_count: u64,
    /// Of those, folders at or above one of the task's outputs (listed with
    /// the reason [`ABOVE_OUTPUTS`]).
    pub folders_above: u64,
}

impl Filtered {
    fn refuse(&mut self, path: &Path, kind: tar::EntryType, reason: &str) {
        self.refused_count += 1;
        if self.refused.len() < LISTED_REFUSALS {
            self.refused.push(RefusedEntry {
                path: path.display().to_string(),
                kind: kind_name(kind).into(),
                reason: reason.into(),
            });
        }
    }
}

/// Why [`filter_outputs`] stopped: the archive (what came out of the
/// sandbox: unreadable, or past [`MAX_NAME_BYTES`] or [`MAX_ENTRIES`], as
/// [`io::ErrorKind::FileTooLarge`]), or the copy's writer on this machine.
#[derive(Debug)]
pub enum FilterError {
    Archive(io::Error),
    Write(io::Error),
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterError::Archive(e) => write!(f, "the outputs archive could not be read: {e}"),
            FilterError::Write(e) => write!(f, "the checked copy could not be written: {e}"),
        }
    }
}

/// A writer that notes whether it failed, so a write error on this machine
/// is told apart from the archive's own.
struct Noted<'a, W> {
    inner: W,
    failed: &'a Cell<bool>,
}

impl<W: Write> Write for Noted<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf).inspect_err(|_| self.failed.set(true))
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush().inspect_err(|_| self.failed.set(true))
    }
}

/// A writer into memory that refuses to grow past a cap, with
/// [`io::ErrorKind::FileTooLarge`].
pub struct CappedVec<'a> {
    buf: &'a mut Vec<u8>,
    cap: u64,
}

impl<'a> CappedVec<'a> {
    pub fn new(buf: &'a mut Vec<u8>, cap: u64) -> Self {
        CappedVec { buf, cap }
    }
}

impl Write for CappedVec<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.buf.len() as u64 + data.len() as u64 > self.cap {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                format!("a checked copy past {} bytes", self.cap),
            ));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A reader that counts what it read.
struct Counted<'a> {
    inner: &'a mut dyn Read,
    n: u64,
}

impl Read for Counted<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.n += n as u64;
        Ok(n)
    }
}

/// Copy the outputs archive at `archive` into a new archive written to
/// `out`, as a stream, keeping only:
/// - regular files at a safe relative path at or under one of `allowed`
///   (absolute sandbox paths: the task's output paths, as the pack got them);
/// - folders strictly under one of them (a folder at or above an output
///   would give a system folder, or the workdir, the archive's owner and
///   mode);
/// - hard links whose name and target are both allowed and whose target was
///   copied before them as a file (an honest pack writes the second name of
///   a file this way; a git clone's objects are such files).
///
/// Each kept entry keeps its owner (uid, gid and names) and modification
/// time (0 for one that is negative, blank, or 2^33 seconds or more), and
/// loses its setuid, setgid and sticky bits. Everything else (links,
/// special files, paths outside the outputs or with `..`) is refused and
/// counted, never an error: the verifier simply never sees it. So are
/// entries the verifier's `tar` would stop on as far as the archive alone
/// shows (an owner past 32 bits, a name with a NUL, a name longer than
/// Linux allows, more than [`MAX_DEPTH`] path parts, an entry under a path
/// kept as a file, a file at a path kept as a folder; what the grading
/// image already holds is not seen), and files and
/// links past [`OUTPUT_BUDGET`] bytes in all, each name counted at its
/// file's full size, as the pack counts them: a link costs nothing in the
/// archive but a full copy wherever the outputs are unpacked. Nothing
/// passes through this machine's file system, which could fold names that
/// differ by case, refuse long or non-UTF-8 names, or change owners.
pub fn filter_outputs<W: Write>(
    archive: &Path,
    out: W,
    allowed: &[String],
) -> Result<Filtered, FilterError> {
    filter_within(archive, out, allowed, OUTPUT_BUDGET)
}

/// [`filter_outputs`] with a byte budget of `budget` (at most
/// [`OUTPUT_BUDGET`]).
pub fn filter_within<W: Write>(
    archive: &Path,
    out: W,
    allowed: &[String],
    budget: u64,
) -> Result<Filtered, FilterError> {
    let limits = Limits {
        bytes: budget.min(OUTPUT_BUDGET),
        ..LIMITS
    };
    filter_with(archive, out, allowed, limits)
}

/// A keyed hash of each leading part of `rel`: `a`, `a/b`, `a/b/c`. The last
/// is the whole path's. Linear in the path's length, however deep.
fn prefix_keys(keys: &RandomState, rel: &Path) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = keys.build_hasher();
    rel.components()
        .map(|c| {
            c.as_os_str().hash(&mut h);
            h.clone().finish()
        })
        .collect()
}

#[cfg(unix)]
fn has_nul(p: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    p.as_os_str().as_bytes().contains(&0)
}

#[cfg(not(unix))]
fn has_nul(p: &Path) -> bool {
    p.to_string_lossy().contains('\0')
}

/// Whether a path has a part over 255 bytes, or is 4,096 bytes or longer:
/// what Linux refuses to create.
fn too_long(p: &Path) -> bool {
    p.as_os_str().len() >= 4096 || p.components().any(|c| c.as_os_str().len() > 255)
}

/// The header's mtime, or 0 for one the verifier's `tar` would stop on or
/// misread: blank, negative (GNU base-256 with the sign bit set, which the
/// crate reads as about 2^64: a file from before 1970), or 2^33 seconds or
/// more (reviews F2-RA2-1, F2-RB3-1).
fn plain_mtime(h: &tar::Header) -> u64 {
    let first = h.as_old().mtime[0];
    let negative = first & 0x80 != 0 && first & 0x40 != 0;
    match h.mtime() {
        Ok(t) if !negative && t < 1 << 33 => t,
        _ => 0,
    }
}

/// Why a file or link past the byte budget is refused.
pub const OVER_BUDGET: &str =
    "past the outputs' byte budget, each name counted at its file's full size";

fn filter_with<W: Write>(
    archive: &Path,
    out: W,
    allowed: &[String],
    limits: Limits,
) -> Result<Filtered, FilterError> {
    let allowed: Vec<PathBuf> = allowed.iter().filter_map(|a| output_rel(a)).collect();
    let file = std::fs::File::open(archive).map_err(FilterError::Archive)?;
    let failed = Cell::new(false);
    let mut b = tar::Builder::new(Noted {
        inner: out,
        failed: &failed,
    });
    let mut f = Filtered::default();
    // The files copied so far, by a keyed hash of their path, with their
    // size: a hard link is kept only to one of them, and costs its size.
    let keys = RandomState::new();
    let mut files: HashMap<u64, u64> = HashMap::new();
    // The folders kept, as entries or as parents of kept entries.
    let mut folders: HashSet<u64> = HashSet::new();
    let mut spent = 0u64;
    let under = |p: &Path| allowed.iter().any(|a| p.starts_with(a));
    let walked = walk(file, limits, &mut |m, data| {
        let kind = m.header.entry_type();
        let Some(rel) = safe_relative(&m.path) else {
            f.refuse(&m.path, kind, "an absolute path, or one with `..`");
            return Ok(());
        };
        // What the verifier's `tar -xf` stops on (review F2-RA-3).
        if has_nul(&rel) || m.link.as_deref().is_some_and(has_nul) {
            f.refuse(&rel, kind, "a name with a NUL byte");
            return Ok(());
        }
        let owner_ok = |n: io::Result<u64>| n.map_or(true, |n| n <= u64::from(u32::MAX));
        if !owner_ok(m.header.uid()) || !owner_ok(m.header.gid()) {
            f.refuse(&rel, kind, "an owner past 32 bits");
            return Ok(());
        }
        // A name Linux cannot create makes the verifier's `tar` stop
        // (review F2-RA3-1).
        if too_long(&rel) || m.link.as_deref().is_some_and(too_long) {
            f.refuse(&rel, kind, "a name longer than Linux allows");
            return Ok(());
        }
        let prefixes = prefix_keys(&keys, &rel);
        if prefixes.len() > MAX_DEPTH {
            f.refuse(&rel, kind, "deeper than BenchFlow restores");
            return Ok(());
        }
        let key = *prefixes.last().expect("a safe path has a part");
        let parents = &prefixes[..prefixes.len() - 1];
        if parents.iter().any(|p| files.contains_key(p)) {
            f.refuse(&rel, kind, "under a path kept as a file");
            return Ok(());
        }
        // A file or link where a folder was kept, or will be as the parent
        // of a kept entry (review F2-RA2-2).
        if !kind.is_dir() && folders.contains(&key) {
            f.refuse(&rel, kind, "a file at a path kept as a folder");
            return Ok(());
        }
        let keep_parents = |folders: &mut HashSet<u64>| {
            for p in parents.iter().rev() {
                if !folders.insert(*p) {
                    break;
                }
            }
        };
        let mut h = tar::Header::new_gnu();
        // A field that does not parse (blank, as some writers leave it)
        // reads as root's, the epoch, or a plain mode.
        h.set_uid(m.header.uid().unwrap_or(0));
        h.set_gid(m.header.gid().unwrap_or(0));
        h.set_mtime(plain_mtime(&m.header));
        let plain = if kind.is_dir() { 0o755 } else { 0o644 };
        h.set_mode(m.header.mode().unwrap_or(plain) & 0o777);
        if let Ok(Some(u)) = m.header.username() {
            let _ = h.set_username(u);
        }
        if let Ok(Some(g)) = m.header.groupname() {
            let _ = h.set_groupname(g);
        }
        // Files and links count against the budget, each at its file's
        // full size (review F2-RA-1).
        let mut afford = |size: u64, f: &mut Filtered| {
            if spent.saturating_add(size) > limits.bytes {
                f.refuse(&rel, kind, OVER_BUDGET);
                false
            } else {
                spent += size;
                true
            }
        };
        if kind.is_file() {
            if !under(&rel) {
                f.refuse(&rel, kind, "outside the task's outputs");
                return Ok(());
            }
            if !afford(m.size, &mut f) {
                return Ok(());
            }
            h.set_entry_type(tar::EntryType::Regular);
            h.set_size(m.size);
            let mut counted = Counted { inner: data, n: 0 };
            b.append_data(&mut h, &rel, &mut counted)?;
            if counted.n != m.size {
                return Err(invalid("the archive ends inside a file"));
            }
            files.insert(key, m.size);
            keep_parents(&mut folders);
            f.kept += 1;
        } else if kind.is_dir() {
            if files.contains_key(&key) {
                f.refuse(&rel, kind, "a folder at a path kept as a file");
                return Ok(());
            }
            if !allowed.iter().any(|a| rel.starts_with(a) && rel != *a) {
                if allowed.iter().any(|a| a.starts_with(&rel)) {
                    f.folders_above += 1;
                    f.refuse(&rel, kind, ABOVE_OUTPUTS);
                } else {
                    f.refuse(&rel, kind, "outside the task's outputs");
                }
                return Ok(());
            }
            h.set_entry_type(tar::EntryType::Directory);
            h.set_size(0);
            b.append_data(&mut h, &rel, io::empty())?;
            folders.insert(key);
            keep_parents(&mut folders);
            f.kept += 1;
        } else if kind.is_hard_link() {
            let target = m.link.as_deref().and_then(safe_relative);
            let size = target
                .as_deref()
                .and_then(|t| prefix_keys(&keys, t).last().copied())
                .and_then(|k| files.get(&k).copied());
            match (target, size) {
                // A file's own name again: nothing to restore.
                (Some(t), Some(_)) if t == rel => {}
                (Some(t), Some(size)) if under(&rel) && under(&t) => {
                    if !afford(size, &mut f) {
                        return Ok(());
                    }
                    h.set_entry_type(tar::EntryType::Link);
                    h.set_size(0);
                    b.append_link(&mut h, &rel, &t)?;
                    files.insert(key, size);
                    keep_parents(&mut folders);
                    f.kept += 1;
                }
                _ => f.refuse(
                    &rel,
                    kind,
                    "a hard link that is not between two files inside the task's outputs, the first copied before it",
                ),
            }
        } else {
            f.refuse(&rel, kind, "not a regular file or folder");
        }
        Ok(())
    });
    let finished = walked.and_then(|()| b.into_inner().and_then(|mut w| w.flush()));
    match finished {
        Ok(()) => Ok(f),
        Err(e) if failed.get() => Err(FilterError::Write(e)),
        Err(e) => Err(FilterError::Archive(e)),
    }
}

/// The entries of an outputs archive that a verifier's sandbox would not
/// get from [`filter_outputs`], by the same rule.
pub fn stray_entries(archive: &Path, allowed: &[String]) -> io::Result<Filtered> {
    filter_outputs(archive, io::sink(), allowed).map_err(|e| match e {
        FilterError::Archive(e) | FilterError::Write(e) => e,
    })
}

fn kind_name(k: tar::EntryType) -> &'static str {
    if k.is_file() {
        "file"
    } else if k.is_dir() {
        "folder"
    } else if k.is_symlink() {
        "symbolic link"
    } else if k.is_hard_link() {
        "hard link"
    } else if k.is_character_special() || k.is_block_special() {
        "device"
    } else if k.is_fifo() {
        "FIFO"
    } else if k.is_gnu_sparse() {
        "sparse file"
    } else if k.is_pax_global_extensions() {
        "PAX global header"
    } else {
        "other entry"
    }
}

/// Unpack `archive` into `dest`, keeping each entry's path (relative to the
/// sandbox's `/`). Never follows or creates links; never writes outside
/// `dest`.
pub fn unpack(archive: &Path, dest: &Path) -> io::Result<Unpacked> {
    unpack_within(archive, dest, &|rel| Some(rel.to_path_buf()), OUTPUT_BUDGET)
}

/// Unpack `archive` into `dest` as [`unpack`] does, with each entry's safe
/// relative path passed through `map`: the path under `dest` to write it
/// at, or `None` to leave it out (counted in [`Unpacked::left_out`]).
/// Each file keeps its modification time. A hard link to a file written
/// before it is written as a copy of that file. The archive is read under
/// the same limits as [`filter_outputs`].
pub fn unpack_mapped(
    archive: &Path,
    dest: &Path,
    map: &dyn Fn(&Path) -> Option<PathBuf>,
) -> io::Result<Unpacked> {
    unpack_within(archive, dest, map, OUTPUT_BUDGET)
}

/// [`unpack_mapped`] with a byte budget of `budget` (at most
/// [`OUTPUT_BUDGET`]): a file or link copy that would take what is written
/// past it is refused (counted in [`Unpacked::over_budget`]), never an
/// error, and what came before it stays (review F2-RA-1: one file and many
/// hard links to it asked this machine for a full copy each).
pub fn unpack_within(
    archive: &Path,
    dest: &Path,
    map: &dyn Fn(&Path) -> Option<PathBuf>,
    budget: u64,
) -> io::Result<Unpacked> {
    let budget = budget.min(OUTPUT_BUDGET);
    let file = std::fs::File::open(archive)?;
    let mut out = Unpacked::default();
    std::fs::create_dir_all(dest)?;
    // The size each file was written at: an archive can hold one path twice
    // (outputs that overlap, `/app` and `/app/report.md`), and the second
    // entry replaces the first, so the counts are of distinct files, as on
    // disk (E5 review: regrade took the doubled count for damage).
    let mut written: std::collections::BTreeMap<PathBuf, u64> = std::collections::BTreeMap::new();
    let mapped = |p: &Path| map(p).and_then(|r| safe_relative(&r));
    walk(file, LIMITS, &mut |m, data| {
        let Some(rel) = safe_relative(&m.path) else {
            out.refused.push(m.path.display().to_string());
            return Ok(());
        };
        let kind = m.header.entry_type();
        let Some(rel) = mapped(&rel) else {
            if kind.is_file() || kind.is_hard_link() {
                out.left_out.push(rel.display().to_string());
            }
            return Ok(());
        };
        let target = dest.join(&rel);
        if kind.is_dir() {
            std::fs::create_dir_all(&target)?;
            return Ok(());
        }
        let (source, size) = if kind.is_file() {
            (None, m.size)
        } else if kind.is_hard_link() {
            match m
                .link
                .as_deref()
                .and_then(safe_relative)
                .and_then(|t| mapped(&t))
                .and_then(|t| written.get(&t).map(|n| (t, *n)))
            {
                Some((t, _)) if t == rel => return Ok(()),
                Some((t, n)) => (Some(dest.join(t)), n),
                None => {
                    out.refused.push(m.path.display().to_string());
                    return Ok(());
                }
            }
        } else {
            out.refused.push(m.path.display().to_string());
            return Ok(());
        };
        // Each name costs its file's full size, link copies included.
        let before = written.get(&rel).copied().unwrap_or(0);
        if (out.bytes - before).saturating_add(size) > budget {
            out.over_budget += 1;
            out.refused.push(m.path.display().to_string());
            return Ok(());
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Refuse to write through anything that is already a link; a file
        // already there (a path packed twice, its first copy read-only:
        // review F2-RB2-1) is replaced, not opened.
        match std::fs::symlink_metadata(&target) {
            Ok(meta) if meta.file_type().is_symlink() => {
                out.refused.push(m.path.display().to_string());
                return Ok(());
            }
            Ok(meta) if meta.file_type().is_file() => std::fs::remove_file(&target)?,
            _ => {}
        }
        let mut f = std::fs::File::create(&target)?;
        let n = match &source {
            Some(s) => io::copy(&mut std::fs::File::open(s)?.take(size), &mut f)?,
            None => io::copy(&mut data.take(size), &mut f)?,
        };
        match written.insert(rel.clone(), n) {
            Some(before) => out.bytes = out.bytes - before + n,
            None => {
                out.files += 1;
                out.bytes += n;
            }
        }
        if let Ok(t) = m.header.mtime() {
            // Never `+`: an mtime of 2^63 or more would panic (review
            // F2-RA2-1).
            if let Some(at) = std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_secs(t)) {
                let _ = f.set_modified(at);
            }
        }
        #[cfg(unix)]
        if let Ok(mode) = m.header.mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ =
                std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode & 0o755));
        }
        Ok(())
    })?;
    Ok(out)
}

/// The paths under `folder` (an `outputs/` folder, laid out from `/`) that
/// a verifier's sandbox must not get: anything but regular files and
/// folders, and paths outside `allowed` (folders above one of them pass:
/// a folder on disk holds its files' parents; [`filter_outputs`] drops
/// them from the archive packed from it).
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
    allowed
        .iter()
        .filter_map(|a| output_rel(a))
        .any(|a| rel.starts_with(&a) || (is_dir && a.starts_with(&rel)))
}

/// One of the task's output paths (absolute, in the sandbox) relative to
/// `/`; `None` for `/` itself or a path with `..`.
fn output_rel(a: &str) -> Option<PathBuf> {
    safe_relative(Path::new(a.trim_start_matches('/')))
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

    /// E5 review (A4, B2): outputs that overlap put one file in the archive
    /// twice; the counts are of the files on disk, so a regrade does not
    /// take them for damage.
    #[test]
    fn a_file_archived_twice_counts_once() {
        let a = archive(
            &[
                ("app/report.md", b"first draft"),
                ("app/other.txt", b"x"),
                ("app/report.md", b"final"),
            ],
            None,
        );
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert_eq!((u.files, u.bytes), (2, 6));
        assert_eq!(
            std::fs::read_to_string(d.path().join("app/report.md")).unwrap(),
            "final"
        );
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

    /// One raw entry, its name (and a link's target) written straight into
    /// the GNU header, past the builder's checks.
    struct Raw<'a> {
        name: &'a [u8],
        kind: tar::EntryType,
        mode: u32,
        owner: u64,
        data: &'a [u8],
        link: &'a [u8],
    }

    fn raw<'a>(name: &'a str, kind: tar::EntryType, mode: u32) -> Raw<'a> {
        Raw {
            name: name.as_bytes(),
            kind,
            mode,
            owner: 0,
            data: b"",
            link: b"",
        }
    }

    fn write_raw(b: &mut tar::Builder<Vec<u8>>, e: &Raw) {
        let mut h = tar::Header::new_gnu();
        h.set_entry_type(e.kind);
        h.set_size(e.data.len() as u64);
        h.set_mode(e.mode);
        h.set_uid(e.owner);
        h.set_gid(e.owner);
        h.set_mtime(1_700_000_000);
        let gnu = h.as_gnu_mut().unwrap();
        gnu.name[..e.name.len()].copy_from_slice(e.name);
        gnu.linkname[..e.link.len()].copy_from_slice(e.link);
        h.set_cksum();
        b.append(&h, e.data).unwrap();
    }

    fn archive_of(entries: &[Raw]) -> tempfile::NamedTempFile {
        let mut b = tar::Builder::new(Vec::new());
        for e in entries {
            write_raw(&mut b, e);
        }
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), b.into_inner().unwrap()).unwrap();
        f
    }

    /// A GNU long-name entry of `len` bytes for the member after it.
    fn long_name(b: &mut tar::Builder<Vec<u8>>, name: &[u8]) {
        let mut h = tar::Header::new_gnu();
        h.set_entry_type(tar::EntryType::GNULongName);
        h.set_size(name.len() as u64);
        h.as_gnu_mut().unwrap().name[..13].copy_from_slice(b"././@LongLink");
        h.set_cksum();
        b.append(&h, name).unwrap();
    }

    /// (path, kind, mode, uid, data) of each entry of `bytes`.
    fn entries_of(bytes: &[u8]) -> Vec<(String, tar::EntryType, u32, u64, Vec<u8>)> {
        let mut ar = tar::Archive::new(bytes);
        ar.entries()
            .unwrap()
            .map(|e| {
                let mut e = e.unwrap();
                let mut data = Vec::new();
                e.read_to_end(&mut data).unwrap();
                let h = e.header();
                let path = String::from_utf8_lossy(&e.path_bytes())
                    .trim_end_matches('/')
                    .to_string();
                (
                    path,
                    h.entry_type(),
                    h.mode().unwrap(),
                    h.uid().unwrap(),
                    data,
                )
            })
            .collect()
    }

    /// Review F2-A-r12 RESTORE: the outputs archive comes from the image's
    /// own `tar`, which a root agent controls. The one rule a separate
    /// verifier, a regrade and `--diff` share keeps regular files inside the
    /// outputs, folders strictly inside them and honest hard links, with
    /// owners kept and setuid, setgid and sticky bits cleared; everything
    /// else is refused and counted, never an error.
    #[test]
    fn the_filter_keeps_only_files_and_folders_inside_the_outputs() {
        use tar::EntryType as K;
        let answer = Raw {
            owner: 1000,
            data: b"42",
            ..raw("app/answer.txt", K::Regular, 0o4755)
        };
        let a = archive_of(&[
            answer,
            Raw {
                data: b"upper",
                ..raw("app/Makefile", K::Regular, 0o644)
            },
            Raw {
                data: b"lower",
                ..raw("app/makefile", K::Regular, 0o644)
            },
            raw("app/sub", K::Directory, 0o1777),
            raw("app", K::Directory, 0o777),
            raw("usr", K::Directory, 0o777),
            Raw {
                data: b"x",
                ..raw("usr/bin/python3", K::Regular, 0o755)
            },
            Raw {
                data: b"{\"reward\":1}",
                ..raw("logs/verifier/reward.json", K::Regular, 0o644)
            },
            Raw {
                data: b"x",
                ..raw("../escape.txt", K::Regular, 0o644)
            },
            Raw {
                data: b"x",
                ..raw("/etc/passwd", K::Regular, 0o644)
            },
            Raw {
                link: b"/etc/shadow",
                ..raw("app/secret", K::Symlink, 0o777)
            },
            Raw {
                link: b"app/answer.txt",
                ..raw("app/same.txt", K::Link, 0o644)
            },
            Raw {
                link: b"etc/passwd",
                ..raw("app/passwd", K::Link, 0o644)
            },
            Raw {
                link: b"app/later.txt",
                ..raw("app/early.txt", K::Link, 0o644)
            },
            raw("app/pipe", K::Fifo, 0o644),
        ]);
        let mut out = Vec::new();
        let f = filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        let kept = entries_of(&out);
        let names: Vec<&str> = kept.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "app/answer.txt",
                "app/Makefile",
                "app/makefile",
                "app/sub",
                "app/same.txt"
            ]
        );
        assert_eq!(f.kept, 5);
        let answer = &kept[0];
        assert_eq!(
            (answer.1, answer.2, answer.3),
            (K::Regular, 0o755, 1000),
            "owner kept, setuid cleared"
        );
        assert_eq!(answer.4, b"42");
        assert_eq!(
            (kept[3].1, kept[3].2),
            (K::Directory, 0o777),
            "sticky bit cleared"
        );
        assert_eq!(kept[4].1, K::Link);
        let refused: Vec<(&str, &str)> = f
            .refused
            .iter()
            .map(|r| (r.path.as_str(), r.kind.as_str()))
            .collect();
        assert_eq!(
            refused,
            [
                ("app", "folder"),
                ("usr", "folder"),
                ("usr/bin/python3", "file"),
                ("logs/verifier/reward.json", "file"),
                ("../escape.txt", "file"),
                ("/etc/passwd", "file"),
                ("app/secret", "symbolic link"),
                ("app/passwd", "hard link"),
                ("app/early.txt", "hard link"),
                ("app/pipe", "FIFO"),
            ]
        );
        assert_eq!((f.refused_count, f.folders_above), (10, 1));
        assert_eq!(f.refused[0].reason, ABOVE_OUTPUTS);
        // The same rule, for a regrade and `--diff`.
        let s = stray_entries(a.path(), &["/app".into()]).unwrap();
        assert_eq!(s, f);
    }

    /// Review F2-A-r12 RESTORE: the crate reads a long-name or PAX entry
    /// whole, so each is capped, and so is the count of members; past
    /// either the archive is the sandbox's, never this machine's error.
    #[test]
    fn long_names_and_entry_counts_are_bounded() {
        let over = |name_len: usize| {
            let mut b = tar::Builder::new(Vec::new());
            // Parts of 99 bytes and a slash: names Linux can create.
            let body: String = (0..name_len - 4)
                .map(|i| if i % 100 == 99 { '/' } else { 'n' })
                .collect();
            let name = format!("app/{body}");
            long_name(&mut b, name.as_bytes());
            write_raw(
                &mut b,
                &Raw {
                    data: b"x",
                    ..raw("app/short", tar::EntryType::Regular, 0o644)
                },
            );
            let f = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(f.path(), b.into_inner().unwrap()).unwrap();
            (f, name)
        };
        // Under the cap: the long name is kept whole.
        let (a, name) = over(300);
        let mut out = Vec::new();
        filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        assert_eq!(entries_of(&out)[0].0, name);
        // Over it: refused before it is read.
        let (a, _) = over(MAX_NAME_BYTES as usize + 1);
        match filter_outputs(a.path(), io::sink(), &["/app".into()]) {
            Err(FilterError::Archive(e)) => assert_eq!(e.kind(), io::ErrorKind::FileTooLarge),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            unpack(a.path(), tempfile::tempdir().unwrap().path())
                .unwrap_err()
                .kind(),
            io::ErrorKind::FileTooLarge
        );
        // The count of members.
        let files: Vec<Raw> = (0..4)
            .map(|_| Raw {
                data: b"x",
                ..raw("app/f", tar::EntryType::Regular, 0o644)
            })
            .collect();
        let a = archive_of(&files);
        let few = Limits {
            name_bytes: MAX_NAME_BYTES,
            entries: 3,
            ..LIMITS
        };
        match filter_with(a.path(), io::sink(), &["/app".into()], few) {
            Err(FilterError::Archive(e)) => assert_eq!(e.kind(), io::ErrorKind::FileTooLarge),
            other => panic!("{other:?}"),
        }
        let enough = Limits {
            name_bytes: MAX_NAME_BYTES,
            entries: 4,
            ..LIMITS
        };
        assert!(filter_with(a.path(), io::sink(), &["/app".into()], enough).is_ok());
    }

    /// A copy that cannot be written is this machine's failure, told apart
    /// from an archive that cannot be read.
    #[test]
    fn a_copy_that_cannot_be_written_is_not_the_archives_fault() {
        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::StorageFull, "full"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let a = archive_of(&[Raw {
            data: b"x",
            ..raw("app/f", tar::EntryType::Regular, 0o644)
        }]);
        assert!(matches!(
            filter_outputs(a.path(), Full, &["/app".into()]),
            Err(FilterError::Write(_))
        ));
        let junk = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(junk.path(), vec![7u8; 1024]).unwrap();
        assert!(matches!(
            filter_outputs(junk.path(), io::sink(), &["/app".into()]),
            Err(FilterError::Archive(_))
        ));
    }

    /// The second name of a file an honest pack saved (a git clone's
    /// objects) is unpacked as a copy, not refused.
    #[test]
    fn an_honest_hard_link_is_unpacked_as_a_copy() {
        use tar::EntryType as K;
        let a = archive_of(&[
            Raw {
                data: b"blob",
                ..raw("app/.git/objects/ab", K::Regular, 0o444)
            },
            Raw {
                link: b"app/.git/objects/ab",
                ..raw("app/clone/.git/objects/ab", K::Link, 0o444)
            },
            Raw {
                link: b"etc/passwd",
                ..raw("app/passwd", K::Link, 0o644)
            },
        ]);
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert_eq!((u.files, u.bytes), (2, 8));
        assert_eq!(
            std::fs::read(d.path().join("app/clone/.git/objects/ab")).unwrap(),
            b"blob"
        );
        assert_eq!(u.refused, ["app/passwd"]);
    }

    /// Review F2-RA-1: one file and many hard links to it cost a full copy
    /// of the file per name wherever they are unpacked. The filter and
    /// `unpack` keep what fits the byte budget, each name at its file's
    /// full size, and refuse the rest, never as an error.
    #[test]
    fn one_file_and_many_links_stay_within_the_byte_budget() {
        let cap = crate::copy_out::ARCHIVE_CAP;
        let mib = 1u64 << 20;
        let body = vec![b'x'; 4 << 20];
        let mut b = tar::Builder::new(Vec::new());
        write_raw(
            &mut b,
            &Raw {
                data: &body,
                ..raw("app/big", tar::EntryType::Regular, 0o644)
            },
        );
        let names: Vec<String> = (0..300).map(|i| format!("app/l{i}")).collect();
        for n in &names {
            write_raw(
                &mut b,
                &Raw {
                    link: b"app/big",
                    ..raw(n, tar::EntryType::Link, 0o644)
                },
            );
        }
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
        // 301 names of 4 MiB: 1,204 MiB, past the budget of 1,088 MiB.
        let fits = cap / (4 * mib);
        let f = filter_outputs(a.path(), io::sink(), &["/app".into()]).unwrap();
        assert_eq!(f.kept, fits);
        assert_eq!(f.refused_count, 301 - fits);
        assert!(
            f.refused.iter().all(|r| r.reason.contains("budget")),
            "{:?}",
            f.refused
        );
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert!(u.bytes <= cap, "{} bytes", u.bytes);
        assert_eq!(u.files as u64, fits);
        assert_eq!(u.refused.len() as u64, 301 - fits);
        let on_disk: u64 = walkdir::WalkDir::new(d.path())
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(on_disk <= cap, "{on_disk} bytes on disk");
    }

    /// A path written as a file, then as a hard link, then as a file again
    /// ends with the last one; the file the link named keeps its own.
    #[test]
    fn a_path_written_again_after_a_link_replaces_only_itself() {
        use tar::EntryType as K;
        let a = archive_of(&[
            Raw {
                data: b"one",
                ..raw("app/a", K::Regular, 0o644)
            },
            Raw {
                link: b"app/a",
                ..raw("app/b", K::Link, 0o644)
            },
            Raw {
                data: b"two",
                ..raw("app/b", K::Regular, 0o644)
            },
        ]);
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert_eq!((u.files, u.bytes), (2, 6));
        assert_eq!(std::fs::read(d.path().join("app/a")).unwrap(), b"one");
        assert_eq!(std::fs::read(d.path().join("app/b")).unwrap(), b"two");
    }

    /// Review F2-RA-2: the filter's copy of a long name adds a NUL, so a
    /// name the first filter keeps must pass the second, at the limit too.
    #[test]
    fn the_filter_is_idempotent_at_the_long_name_limit() {
        for len in [MAX_NAME_BYTES as usize - 1, MAX_NAME_BYTES as usize] {
            let name = format!("app/{}", "n".repeat(len - 4));
            let mut b = tar::Builder::new(Vec::new());
            long_name(&mut b, name.as_bytes());
            write_raw(
                &mut b,
                &Raw {
                    data: b"x",
                    ..raw("app/short", tar::EntryType::Regular, 0o644)
                },
            );
            let a = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
            let mut first = Vec::new();
            if filter_outputs(a.path(), &mut first, &["/app".into()]).is_err() {
                continue;
            }
            let again = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(again.path(), &first).unwrap();
            let second = filter_outputs(again.path(), io::sink(), &["/app".into()]);
            assert!(second.is_ok(), "a name of {len} bytes: {second:?}");
        }
    }

    /// Review F2-RA-3: entries the verifier's `tar -xf` stops on (an owner
    /// past 32 bits, a member under a path kept as a file, a folder there,
    /// a name with a NUL) would let a root agent end its own trial as the
    /// sandbox's; they are refused instead.
    #[test]
    fn entries_the_verifiers_tar_stops_on_are_refused() {
        use tar::EntryType as K;
        let mut b = tar::Builder::new(Vec::new());
        for e in [
            Raw {
                data: b"f",
                ..raw("app/f", K::Regular, 0o644)
            },
            Raw {
                data: b"g",
                ..raw("app/f/g", K::Regular, 0o644)
            },
            raw("app/f", K::Directory, 0o755),
            Raw {
                owner: 1 << 40,
                data: b"h",
                ..raw("app/huge", K::Regular, 0o644)
            },
        ] {
            write_raw(&mut b, &e);
        }
        // A PAX path with a NUL in it.
        let record = b"app/a\0b";
        let line = format!("{} path=", record.len() + 9);
        let mut pax = line.into_bytes();
        pax.extend_from_slice(record);
        pax.push(b'\n');
        let mut h = tar::Header::new_ustar();
        h.set_entry_type(K::XHeader);
        h.set_size(pax.len() as u64);
        h.set_path("PaxHeader").unwrap();
        h.set_cksum();
        b.append(&h, &pax[..]).unwrap();
        write_raw(
            &mut b,
            &Raw {
                data: b"n",
                ..raw("app/plain", K::Regular, 0o644)
            },
        );
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
        let mut out = Vec::new();
        let f = filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        let names: Vec<String> = entries_of(&out).into_iter().map(|e| e.0).collect();
        assert_eq!(names, ["app/f"], "{:?}", f.refused);
        assert_eq!(f.refused_count, 4, "{:?}", f.refused);
    }

    /// Review F2-RA2-1: an mtime of 2^63 seconds or more panicked the
    /// unpack (`UNIX_EPOCH + Duration`), so the trial was never recorded. A
    /// raw archive, as the session's is, unpacks without it; the filter
    /// refuses an mtime the verifier's `tar` stops on.
    #[test]
    fn a_huge_modification_time_neither_panics_nor_reaches_the_verifier() {
        let mut b = tar::Builder::new(Vec::new());
        let mut h = tar::Header::new_gnu();
        h.set_size(1);
        h.set_mode(0o644);
        h.set_mtime(u64::MAX);
        b.append_data(&mut h, "home/agent/.fake/s.jsonl", &b"x"[..])
            .unwrap();
        let mut h = tar::Header::new_gnu();
        h.set_size(1);
        h.set_mode(0o644);
        h.set_mtime(1 << 33);
        b.append_data(&mut h, "home/agent/.fake/t.jsonl", &b"y"[..])
            .unwrap();
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
        let d = tempfile::tempdir().unwrap();
        let u = unpack_mapped(a.path(), d.path(), &|rel| {
            rel.strip_prefix("home/agent").ok().map(Path::to_path_buf)
        })
        .unwrap();
        assert_eq!(u.files, 2);
        let f = filter_outputs(a.path(), io::sink(), &["/home/agent".into()]).unwrap();
        assert_eq!((f.kept, f.refused_count), (2, 0), "{:?}", f.refused);
    }

    /// Review F2-RA2-2: a file at a path kept as a folder (as an entry, or
    /// as the parent of a kept entry) makes the verifier's `tar` stop.
    #[test]
    fn a_file_at_a_path_kept_as_a_folder_is_refused() {
        use tar::EntryType as K;
        let a = archive_of(&[
            raw("app/d", K::Directory, 0o755),
            Raw {
                data: b"x",
                ..raw("app/d/x", K::Regular, 0o644)
            },
            Raw {
                data: b"d",
                ..raw("app/d", K::Regular, 0o644)
            },
            Raw {
                data: b"y",
                ..raw("app/e/y", K::Regular, 0o644)
            },
            Raw {
                link: b"app/d/x",
                ..raw("app/e", K::Link, 0o644)
            },
        ]);
        let mut out = Vec::new();
        let f = filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        let names: Vec<String> = entries_of(&out).into_iter().map(|e| e.0).collect();
        assert_eq!(names, ["app/d", "app/d/x", "app/e/y"], "{:?}", f.refused);
        assert_eq!(f.refused_count, 2);
    }

    /// Review F2-RB2-1: honest outputs that overlap (`/app` and
    /// `/app/ro.txt`) pack a read-only file twice; the second copy replaces
    /// the first instead of failing the whole unpack.
    #[test]
    fn a_read_only_file_packed_twice_unpacks() {
        use tar::EntryType as K;
        let a = archive_of(&[
            Raw {
                data: b"one",
                ..raw("app/ro.txt", K::Regular, 0o444)
            },
            Raw {
                data: b"two",
                ..raw("app/ro.txt", K::Regular, 0o444)
            },
        ]);
        let d = tempfile::tempdir().unwrap();
        let u = unpack(a.path(), d.path()).unwrap();
        assert_eq!((u.files, u.bytes), (1, 3));
        assert_eq!(std::fs::read(d.path().join("app/ro.txt")).unwrap(), b"two");
    }

    /// Reviews F2-RB3-1, F2-RA3-2: an mtime the verifier's `tar` would stop
    /// on or misread is written as 0, never a reason to drop the file. GNU
    /// tar writes a file from before 1970 as negative base-256, which the
    /// crate reads as about 2^64. Every other mtime is kept as it is.
    #[test]
    fn odd_modification_times_are_written_as_zero_and_the_file_kept() {
        let mut b = tar::Builder::new(Vec::new());
        let cases: [(&str, Option<u64>); 4] = [
            ("app/epoch", Some(0)),
            ("app/last", Some((1 << 33) - 1)),
            ("app/before-1970", None),
            ("app/max", Some(u64::MAX)),
        ];
        for (name, t) in cases {
            let mut h = tar::Header::new_gnu();
            h.set_size(1);
            h.set_mode(0o644);
            match t {
                Some(t) => h.set_mtime(t),
                // -1 in GNU base-256.
                None => h.as_old_mut().mtime = [0xff; 12],
            }
            h.set_path(name).unwrap();
            h.set_cksum();
            b.append(&h, &b"x"[..]).unwrap();
        }
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
        let mut out = Vec::new();
        let f = filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        assert_eq!((f.kept, f.refused_count), (4, 0), "{:?}", f.refused);
        let mut ar = tar::Archive::new(&out[..]);
        let mtimes: Vec<u64> = ar
            .entries()
            .unwrap()
            .map(|e| e.unwrap().header().mtime().unwrap())
            .collect();
        assert_eq!(mtimes, [0, (1 << 33) - 1, 0, 0]);
        let d = tempfile::tempdir().unwrap();
        assert_eq!(unpack(a.path(), d.path()).unwrap().files, 4);
    }

    /// Review F2-RA3-1: a name Linux cannot create (a part over 255 bytes,
    /// or 4,096 bytes or more in all) makes the verifier's `tar` stop.
    #[test]
    fn names_longer_than_linux_allows_are_refused() {
        use tar::EntryType as K;
        let part = format!("app/{}", "p".repeat(256));
        let long = format!("app/{}", vec!["d".repeat(200); 21].join("/"));
        let mut b = tar::Builder::new(Vec::new());
        for name in [part.as_str(), long.as_str()] {
            long_name(&mut b, name.as_bytes());
            write_raw(
                &mut b,
                &Raw {
                    data: b"x",
                    ..raw("app/x", K::Regular, 0o644)
                },
            );
        }
        write_raw(
            &mut b,
            &Raw {
                data: b"y",
                ..raw("app/ok", K::Regular, 0o644)
            },
        );
        long_name(&mut b, part.as_bytes());
        write_raw(
            &mut b,
            &Raw {
                link: b"app/ok",
                ..raw("app/x", K::Link, 0o644)
            },
        );
        let mut h = tar::Header::new_gnu();
        h.set_entry_type(K::GNULongLink);
        h.set_size(part.len() as u64);
        h.as_gnu_mut().unwrap().name[..13].copy_from_slice(b"././@LongLink");
        h.set_cksum();
        b.append(&h, part.as_bytes()).unwrap();
        write_raw(&mut b, &raw("app/to-long", K::Link, 0o644));
        let a = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(a.path(), b.into_inner().unwrap()).unwrap();
        let mut out = Vec::new();
        let f = filter_outputs(a.path(), &mut out, &["/app".into()]).unwrap();
        let names: Vec<String> = entries_of(&out).into_iter().map(|e| e.0).collect();
        assert_eq!(names, ["app/ok"], "{:?}", f.refused);
        assert_eq!(f.refused_count, 4, "{:?}", f.refused);
        assert!(
            f.refused
                .iter()
                .all(|r| r.reason.contains("longer than Linux")),
            "{:?}",
            f.refused
        );
    }
}
