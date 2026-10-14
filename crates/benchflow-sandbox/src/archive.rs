//! Tar archives for moving files in and out of sandboxes.
//!
//! Both backends move folders as one tar stream: the host packs, the sandbox
//! runs `tar -xf -`; the sandbox runs `tar -cf -`, the host unpacks.
//!
//! Packing ([`pack`], [`pack_filtered`], [`pack_bytes`]) walks a local file
//! or folder without following symlinks and writes regular files and folders
//! only, owned by root, with their permission bits (setuid, setgid and sticky
//! bits are dropped). Symlinks and special files are left out and reported.
//!
//! Unpacking ([`unpack`]) treats the archive as hostile, because it comes out
//! of a sandbox an agent controlled. It refuses absolute paths, `..`,
//! symlinks, hard links, devices, FIFOs and sockets; it never writes through
//! a symlink that already exists under the destination; it can stop at a
//! size cap before reading a huge (or sparse) file; and it reports what it
//! skipped and why.

use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use tar::{EntryType, Header};

/// A path left out of an archive or an extraction, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub path: String,
    pub reason: &'static str,
}

/// What a pack wrote.
#[derive(Debug, Clone, Default)]
pub struct Packed {
    pub files: u64,
    pub dirs: u64,
    /// Bytes of file content.
    pub bytes: u64,
    /// sha256 over each entry's path, kind, mode, and content, in archive
    /// order. Modification times are left out, so a fresh checkout of the
    /// same files has the same digest.
    pub digest: String,
    pub skipped: Vec<Skipped>,
}

/// Which paths [`pack_filtered`] includes. Paths are relative to the packed
/// folder and use `/`.
pub trait PackFilter {
    /// Include this file or folder entry.
    fn keep(&self, rel: &str, is_dir: bool) -> bool;
    /// Walk into this folder at all. Return false only when nothing below it
    /// can be kept.
    fn descend(&self, _rel_dir: &str) -> bool {
        true
    }
}

/// Includes everything.
pub struct KeepAll;

impl PackFilter for KeepAll {
    fn keep(&self, _rel: &str, _is_dir: bool) -> bool {
        true
    }
}

/// Pack a local file or folder. A folder's contents go at the archive root;
/// a file becomes one entry named after it.
pub fn pack(local: &Path, out: impl Write) -> io::Result<Packed> {
    pack_filtered(local, &KeepAll, out)
}

/// Pack a local file as one entry called `name`.
pub fn pack_file_as(local: &Path, name: &str, out: impl Write) -> io::Result<Packed> {
    let meta = fs::symlink_metadata(local).map_err(|e| with_path(e, "read", local))?;
    if !meta.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is not a regular file", local.display()),
        ));
    }
    let mut packer = Packer::new(out);
    packer.file(local, name, &meta)?;
    packer.finish()
}

/// [`pack`], including only what `filter` keeps.
pub fn pack_filtered(local: &Path, filter: &dyn PackFilter, out: impl Write) -> io::Result<Packed> {
    let meta = fs::symlink_metadata(local).map_err(|e| with_path(e, "read", local))?;
    let mut packer = Packer::new(out);
    if meta.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} is a symlink; BenchFlow uploads regular files and folders only",
                local.display()
            ),
        ));
    } else if meta.is_file() {
        let name = local
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".to_string());
        packer.file(local, &name, &meta)?;
    } else if meta.is_dir() {
        packer.tree(local, filter)?;
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is not a regular file or folder", local.display()),
        ));
    }
    packer.finish()
}

/// Write an archive holding one file, `name`, with `contents` and `mode`.
pub fn pack_bytes(name: &str, contents: &[u8], mode: u32, out: impl Write) -> io::Result<()> {
    let mut builder = tar::Builder::new(out);
    let mut header = header(EntryType::Regular, mode, contents.len() as u64, now_secs());
    builder.append_data(&mut header, name, contents)?;
    builder.into_inner()?.flush()
}

struct Packer<W: Write> {
    builder: tar::Builder<W>,
    hasher: Sha256,
    packed: Packed,
}

impl<W: Write> Packer<W> {
    fn new(out: W) -> Self {
        let mut builder = tar::Builder::new(out);
        builder.follow_symlinks(false);
        Self {
            builder,
            hasher: Sha256::new(),
            packed: Packed::default(),
        }
    }

    fn tree(&mut self, root: &Path, filter: &dyn PackFilter) -> io::Result<()> {
        let mut walk = walkdir::WalkDir::new(root)
            .min_depth(1)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter();
        while let Some(entry) = walk.next() {
            let entry = entry.map_err(|e| {
                let path = e
                    .path()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| root.to_path_buf());
                io::Error::other(format!("cannot read {}: {e}", path.display()))
            })?;
            let rel = relative(root, entry.path());
            let file_type = entry.file_type();
            if file_type.is_dir() {
                if !filter.descend(&rel) {
                    walk.skip_current_dir();
                    continue;
                }
                if filter.keep(&rel, true) {
                    let meta = entry.metadata().map_err(io::Error::other)?;
                    self.dir(&rel, &meta)?;
                }
            } else if file_type.is_symlink() {
                if filter.keep(&rel, false) {
                    self.skip(rel, "symlink");
                }
            } else if file_type.is_file() {
                if filter.keep(&rel, false) {
                    let meta = entry.metadata().map_err(io::Error::other)?;
                    self.file(entry.path(), &rel, &meta)?;
                }
            } else if filter.keep(&rel, false) {
                self.skip(rel, "special file (FIFO, socket or device)");
            }
        }
        Ok(())
    }

    fn skip(&mut self, path: String, reason: &'static str) {
        self.packed.skipped.push(Skipped { path, reason });
    }

    fn dir(&mut self, rel: &str, meta: &fs::Metadata) -> io::Result<()> {
        let mode = meta.mode() & 0o777;
        self.hash_entry(b'd', rel, mode, 0);
        let mut h = header(EntryType::Directory, mode, 0, mtime(meta));
        self.builder
            .append_data(&mut h, format!("{rel}/"), io::empty())?;
        self.packed.dirs += 1;
        Ok(())
    }

    fn file(&mut self, path: &Path, name: &str, meta: &fs::Metadata) -> io::Result<()> {
        let size = meta.len();
        let mode = meta.mode() & 0o777;
        self.hash_entry(b'f', name, mode, size);
        let file = fs::File::open(path).map_err(|e| with_path(e, "open", path))?;
        let mut reader = Hashing {
            inner: file.take(size),
            hasher: &mut self.hasher,
            read: 0,
        };
        let mut h = header(EntryType::Regular, mode, size, mtime(meta));
        self.builder.append_data(&mut h, name, &mut reader)?;
        if reader.read != size {
            return Err(io::Error::other(format!(
                "{} changed size while it was being packed ({} of {size} bytes read); try again",
                path.display(),
                reader.read
            )));
        }
        self.packed.files += 1;
        self.packed.bytes += size;
        Ok(())
    }

    fn hash_entry(&mut self, kind: u8, rel: &str, mode: u32, size: u64) {
        self.hasher.update([kind]);
        self.hasher.update(rel.as_bytes());
        self.hasher.update([0]);
        self.hasher.update(mode.to_le_bytes());
        self.hasher.update(size.to_le_bytes());
    }

    fn finish(self) -> io::Result<Packed> {
        let Packer {
            builder,
            hasher,
            mut packed,
        } = self;
        builder.into_inner()?.flush()?;
        packed.digest = hex::encode(hasher.finalize());
        Ok(packed)
    }
}

/// Hashes what passes through it and counts it.
struct Hashing<'a, R> {
    inner: R,
    hasher: &'a mut Sha256,
    read: u64,
}

impl<R: Read> Read for Hashing<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        self.read += n as u64;
        Ok(n)
    }
}

fn header(kind: EntryType, mode: u32, size: u64, mtime: u64) -> Header {
    let mut h = Header::new_gnu();
    h.set_entry_type(kind);
    h.set_mode(mode & 0o777);
    h.set_size(size);
    h.set_mtime(mtime);
    h.set_uid(0);
    h.set_gid(0);
    // Names are best effort; numeric ids decide ownership.
    let _ = h.set_username("root");
    let _ = h.set_groupname("root");
    h
}

fn mtime(meta: &fs::Metadata) -> u64 {
    u64::try_from(meta.mtime()).unwrap_or(0)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn relative(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn with_path(err: io::Error, action: &str, path: &Path) -> io::Error {
    io::Error::new(
        err.kind(),
        format!("cannot {action} {}: {err}", path.display()),
    )
}

/// What an extraction wrote.
#[derive(Debug, Clone, Default)]
pub struct Unpacked {
    pub files: u64,
    pub dirs: u64,
    /// Bytes of file content written.
    pub bytes: u64,
    /// Entries refused, with the reason.
    pub skipped: Vec<Skipped>,
    /// Entries refused past the first [`MAX_SKIPPED_KEPT`], counted only.
    pub skipped_more: u64,
    /// The archive's top entry was a regular file (written at `dest`), not a
    /// folder.
    pub root_is_file: bool,
}

/// The unpacker's error for an archive with nothing under the path asked
/// for. Its kind is `NotFound`, as a missing path's is; this type tells it
/// apart from this machine's own `NotFound` (a folder gone while the copy
/// was unpacked): what streams out is the sandbox's (review F2-A-r11 L2).
#[derive(Debug)]
struct EmptyArchive(String);

impl std::fmt::Display for EmptyArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EmptyArchive {}

/// True when `e` is [`unpack`]'s error for an archive that holds nothing
/// under the path asked for.
pub fn is_empty_archive(e: &io::Error) -> bool {
    e.get_ref().is_some_and(|inner| inner.is::<EmptyArchive>())
}

/// The largest GNU long name or long link (`L`, `K`) or PAX header (`x`,
/// `g`) an archive may hold. They are read into memory whole, and charged
/// to no content cap, so code inside a sandbox could otherwise make one as
/// large as this machine's memory (review F2-A-r12-2).
pub const MAX_HEADER_EXTENSION: u64 = 64 * 1024;

/// The refused entries [`Unpacked::skipped`] keeps; the rest are counted.
pub const MAX_SKIPPED_KEPT: usize = 100;

/// The longest path a [`Skipped`] record shows, in characters.
const MAX_SHOWN: usize = 256;

/// What an extraction may take from an archive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// File content, in bytes.
    pub bytes: Option<u64>,
    /// Archive entries (headers of every kind, refused ones included) plus
    /// the folders the extraction creates on the way to them. Folders and
    /// empty files cost no content, but each costs this machine an inode
    /// (review F2-A-r12-3).
    pub entries: Option<u64>,
}

/// The entries a capped copy may bring when its caller names no budget.
pub const DEFAULT_ENTRIES: u64 = 100_000;

impl Limits {
    /// `max_bytes` of content, and [`DEFAULT_ENTRIES`].
    pub fn capped(max_bytes: u64) -> Self {
        Self {
            bytes: Some(max_bytes),
            entries: Some(DEFAULT_ENTRIES),
        }
    }
}

/// What an extraction may take from a copy of one file (an archive, a
/// piece of evidence, a log): its one entry, with room for the headers a
/// `tar` may put before it.
pub const FILE_ENTRIES: u64 = 8;

/// Extract an archive made by `tar -cf - -C <parent> <root>`, so that `root`
/// itself lands at `dest`: a folder's contents go into `dest` (created), a
/// file is written at `dest`. With `root` equal to `.`, every entry goes
/// under `dest`.
///
/// Entries outside `root`, absolute paths, `..`, links and special files are
/// skipped and listed in the result. More than `max_bytes` of file content
/// fails the extraction before the file that would cross the cap is read.
/// Folders keep their permission bits plus owner `rwx`, so the result can
/// always be cleaned up; files keep their permission bits.
pub fn unpack(
    archive: impl Read,
    root: &str,
    dest: &Path,
    max_bytes: Option<u64>,
) -> io::Result<Unpacked> {
    unpack_limited(
        archive,
        root,
        dest,
        Limits {
            bytes: max_bytes,
            entries: None,
        },
    )
}

/// The error for an archive past one of its limits: `FileTooLarge`, which
/// every caller already maps as a copy past its cap.
fn too_large(what: String) -> io::Error {
    io::Error::new(io::ErrorKind::FileTooLarge, what)
}

/// Counts entries against [`Limits::entries`].
struct Budget {
    used: u64,
    max: Option<u64>,
}

impl Budget {
    fn charge(&mut self) -> io::Result<()> {
        self.used += 1;
        match self.max {
            Some(max) if self.used > max => Err(too_large(format!(
                "the archive holds more than {max} entries (folders created on the way included); \
                 the rest was not copied"
            ))),
            _ => Ok(()),
        }
    }
}

/// Keep a refused entry, its path cut to [`MAX_SHOWN`] characters, or
/// count it past [`MAX_SKIPPED_KEPT`].
fn skip(out: &mut Unpacked, path: &str, reason: &'static str) {
    if out.skipped.len() >= MAX_SKIPPED_KEPT {
        out.skipped_more += 1;
        return;
    }
    let path = match path.char_indices().nth(MAX_SHOWN) {
        Some((at, _)) => format!("{}…", &path[..at]),
        None => path.to_string(),
    };
    out.skipped.push(Skipped { path, reason });
}

/// What the extension headers before an entry said about it.
#[derive(Default)]
struct Pending {
    long_name: Option<Vec<u8>>,
    long_link: bool,
    pax: Option<Vec<u8>>,
}

impl Pending {
    fn is_empty(&self) -> bool {
        self.long_name.is_none() && !self.long_link && self.pax.is_none()
    }
}

/// Read an extension header's body, refusing one past
/// [`MAX_HEADER_EXTENSION`] before any of it is read.
fn extension_body(entry: &mut tar::Entry<'_, impl Read>, kind: &str) -> io::Result<Vec<u8>> {
    let size = entry.size();
    if size > MAX_HEADER_EXTENSION {
        return Err(too_large(format!(
            "the archive's {kind} header holds {size} bytes, over the {MAX_HEADER_EXTENSION}-byte \
             limit; the rest was not copied"
        )));
    }
    let mut body = Vec::with_capacity(size as usize);
    entry.take(MAX_HEADER_EXTENSION).read_to_end(&mut body)?;
    Ok(body)
}

/// [`unpack`] under `limits`. The archive's extension headers (GNU long
/// names, PAX) are read here, under [`MAX_HEADER_EXTENSION`], not by the
/// tar crate, which reads them whole.
pub fn unpack_limited(
    archive: impl Read,
    root: &str,
    dest: &Path,
    limits: Limits,
) -> io::Result<Unpacked> {
    let max_bytes = limits.bytes;
    let mut budget = Budget {
        used: 0,
        max: limits.entries,
    };
    let mut archive = tar::Archive::new(archive);
    let mut out = Unpacked::default();
    let root = root.trim_matches('/');
    let mut saw_root = false;
    let mut pending = Pending::default();

    for entry in archive.entries()?.raw(true) {
        let mut entry = entry?;
        budget.charge()?;
        let kind = entry.header().entry_type();
        // Extension headers describe the entry after them (two of a kind
        // for one entry is refused, as the tar crate does).
        let twice = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "two extension headers of one kind describe the same entry",
            )
        };
        match kind {
            EntryType::GNULongName => {
                if pending.long_name.is_some() {
                    return Err(twice());
                }
                let mut name = extension_body(&mut entry, "long name")?;
                while name.last() == Some(&0) {
                    name.pop();
                }
                pending.long_name = Some(name);
                continue;
            }
            EntryType::GNULongLink => {
                if pending.long_link {
                    return Err(twice());
                }
                // Links are skipped, so the name is not kept.
                extension_body(&mut entry, "long link name")?;
                pending.long_link = true;
                continue;
            }
            EntryType::XHeader => {
                if pending.pax.is_some() {
                    return Err(twice());
                }
                pending.pax = Some(extension_body(&mut entry, "PAX")?);
                continue;
            }
            // Global headers set defaults this unpacker does not use.
            EntryType::XGlobalHeader => {
                extension_body(&mut entry, "PAX global")?;
                continue;
            }
            _ => {}
        }
        let described = std::mem::take(&mut pending);
        let mut pax_path = None;
        if let Some(pax) = &described.pax {
            for ext in tar::PaxExtensions::new(pax) {
                let ext = ext?;
                match ext.key_bytes() {
                    b"path" => pax_path = Some(ext.value_bytes().to_vec()),
                    // The crate frames the entry by its header's size, so a
                    // PAX size that says otherwise is refused (an honest
                    // `tar` writes one only past what the header can hold,
                    // far past every cap).
                    b"size" => {
                        let size: Option<u64> =
                            ext.value().ok().and_then(|v| v.trim().parse().ok());
                        if size != Some(entry.size()) {
                            return Err(match size {
                                Some(s) if max_bytes.is_some_and(|cap| s > cap) => too_large(
                                    format!("an entry of {s} bytes is past the download's cap"),
                                ),
                                _ => io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "an entry's PAX size disagrees with its header",
                                ),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        let raw_path: PathBuf = match described.long_name.or(pax_path) {
            Some(bytes) => {
                use std::os::unix::ffi::OsStringExt;
                PathBuf::from(OsString::from_vec(bytes))
            }
            None => entry.path()?.into_owned(),
        };
        let shown = raw_path.to_string_lossy().into_owned();
        let parts = match clean_parts(&raw_path) {
            Ok(parts) => parts,
            Err(reason) => {
                skip(&mut out, &shown, reason);
                continue;
            }
        };
        let rest: Vec<OsString> = if root.is_empty() || root == "." {
            parts
        } else if parts.first().is_some_and(|p| p == root) {
            parts[1..].to_vec()
        } else {
            skip(&mut out, &shown, "outside the requested path");
            continue;
        };
        let kind = entry.header().entry_type();
        if rest.is_empty() {
            saw_root = true;
        }
        match kind {
            EntryType::Directory => match safe_dir(&mut budget, dest, &rest)? {
                Ok(dir) => {
                    let mode = entry.header().mode().unwrap_or(0o755) & 0o777;
                    // Applied now, with the owner's rwx kept, so a
                    // read-only folder never stops its own contents from
                    // being written, and nothing is held per folder until
                    // the end (review of 661c655e: a flood of deep folder
                    // entries held their paths in memory).
                    if !rest.is_empty() {
                        fs::set_permissions(&dir, fs::Permissions::from_mode(mode | 0o700))?;
                    }
                    out.dirs += 1;
                }
                Err(reason) => skip(&mut out, &shown, reason),
            },
            EntryType::Regular | EntryType::Continuous => {
                let size = entry.size();
                if let Some(cap) = max_bytes
                    && out.bytes.saturating_add(size) > cap
                {
                    return Err(io::Error::new(
                        io::ErrorKind::FileTooLarge,
                        format!(
                            "{shown} ({size} bytes) would take the download past its {cap}-byte cap \
                             ({} bytes written so far); the rest was not copied",
                            out.bytes
                        ),
                    ));
                }
                let target = if rest.is_empty() {
                    out.root_is_file = true;
                    if let Some(parent) = dest.parent()
                        && !parent.as_os_str().is_empty()
                    {
                        fs::create_dir_all(parent)?;
                    }
                    Ok(dest.to_path_buf())
                } else {
                    match safe_dir(&mut budget, dest, &rest[..rest.len() - 1])? {
                        Ok(dir) => Ok(dir.join(&rest[rest.len() - 1])),
                        Err(reason) => Err(reason),
                    }
                };
                let target = match target {
                    Ok(t) => t,
                    Err(reason) => {
                        skip(&mut out, &shown, reason);
                        continue;
                    }
                };
                match fs::symlink_metadata(&target) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        skip(
                            &mut out,
                            &shown,
                            "a symlink is in the way at the destination",
                        );
                        continue;
                    }
                    Ok(meta) if meta.is_dir() => {
                        skip(
                            &mut out,
                            &shown,
                            "a folder is in the way at the destination",
                        );
                        continue;
                    }
                    // A file already there (an entry repeated, maybe with a
                    // mode that forbids writing it) is replaced, never
                    // opened for writing: the archive's modes must not make
                    // this machine's own write fail (review F2-A-r11 M2).
                    Ok(_) => {
                        fs::remove_file(&target).map_err(|e| with_path(e, "replace", &target))?
                    }
                    Err(_) => {}
                }
                let mode = entry.header().mode().unwrap_or(0o644) & 0o777;
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&target)
                    .map_err(|e| with_path(e, "write", &target))?;
                let written = io::copy(&mut (&mut entry).take(size), &mut file)?;
                file.flush()?;
                drop(file);
                fs::set_permissions(&target, fs::Permissions::from_mode(mode))?;
                out.files += 1;
                out.bytes += written;
            }
            // Read raw, a sparse entry would not be framed as `tar -S` wrote
            // it; no copy BenchFlow makes asks for one.
            EntryType::GNUSparse => skip(&mut out, &shown, "sparse file"),
            EntryType::Symlink => skip(&mut out, &shown, "symlink"),
            EntryType::Link => skip(&mut out, &shown, "hard link"),
            EntryType::Char | EntryType::Block | EntryType::Fifo => {
                skip(&mut out, &shown, "special file (device or FIFO)")
            }
            EntryType::XGlobalHeader | EntryType::XHeader => {}
            _ => skip(&mut out, &shown, "unsupported entry type"),
        }
    }

    if !pending.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the archive ends with extension headers that describe no entry",
        ));
    }
    if !saw_root
        && out.files == 0
        && out.dirs == 0
        && out.skipped.is_empty()
        && out.skipped_more == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            EmptyArchive(format!("the archive holds nothing under {root}")),
        ));
    }

    Ok(out)
}

/// The normal components of an archive path, or why it is refused.
fn clean_parts(path: &Path) -> Result<Vec<OsString>, &'static str> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(p) => parts.push(p.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir => return Err("path with .."),
            Component::RootDir | Component::Prefix(_) => return Err("absolute path"),
        }
    }
    Ok(parts)
}

/// Create `dest/<rel...>` one component at a time, never following a
/// symlink below `dest`. `dest` itself is the caller's choice and is created
/// with its parents.
fn safe_dir(
    budget: &mut Budget,
    dest: &Path,
    rel: &[OsString],
) -> io::Result<Result<PathBuf, &'static str>> {
    match fs::symlink_metadata(dest) {
        Ok(meta) if meta.file_type().is_symlink() && !dest.is_dir() => {
            return Ok(Err("the destination is a symlink"));
        }
        Ok(meta) if !meta.is_dir() && !meta.file_type().is_symlink() => {
            return Ok(Err("a file is in the way at the destination"));
        }
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => fs::create_dir_all(dest)?,
        Err(e) => return Err(e),
    }
    let mut dir = dest.to_path_buf();
    for part in rel {
        dir.push(part);
        match fs::symlink_metadata(&dir) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Ok(Err("a symlink is in the way at the destination"));
            }
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => return Ok(Err("a file is in the way at the destination")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                budget.charge()?;
                fs::create_dir(&dir)?;
                // 0o755 until the folder's own entry, if any, sets its mode (the
                // owner's rwx kept).
                fs::set_permissions(&dir, fs::Permissions::from_mode(0o755))?;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(Ok(dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn mode_of(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    /// One hand-made archive entry: path, kind, mode, data, link target.
    type RawEntry<'a> = (&'a str, EntryType, u32, &'a [u8], Option<&'a str>);

    /// Build an archive by hand, entry by entry.
    fn handmade(entries: &[RawEntry<'_>]) -> Vec<u8> {
        let mut b = tar::Builder::new(Vec::new());
        for (path, kind, mode, data, link) in entries {
            let mut h = Header::new_gnu();
            h.set_entry_type(*kind);
            h.set_mode(*mode);
            h.set_size(data.len() as u64);
            if let Some(link) = link {
                h.set_link_name(link).unwrap();
            }
            // Write the raw name so unsafe paths survive (set_path refuses them).
            let name = h.as_old_mut().name.as_mut();
            name[..path.len()].copy_from_slice(path.as_bytes());
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        b.into_inner().unwrap()
    }

    #[test]
    fn folder_round_trip_keeps_contents_and_modes() {
        let src = tempfile::tempdir().unwrap();
        let root = src.path().join("out");
        fs::create_dir_all(root.join("sub/deeper")).unwrap();
        fs::write(root.join("run.sh"), "#!/bin/sh\necho hi\n").unwrap();
        fs::set_permissions(root.join("run.sh"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(root.join("sub/secret.txt"), "s3cret").unwrap();
        fs::set_permissions(
            root.join("sub/secret.txt"),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        fs::write(root.join("sub/deeper/empty"), "").unwrap();
        symlink("/etc/passwd", root.join("link")).unwrap();

        let mut tar_bytes = Vec::new();
        let packed = pack(&root, &mut tar_bytes).unwrap();
        assert_eq!(packed.files, 3);
        assert_eq!(packed.dirs, 2);
        assert_eq!(
            packed.skipped,
            vec![Skipped {
                path: "link".into(),
                reason: "symlink"
            }]
        );

        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("copy");
        let out = unpack(&tar_bytes[..], ".", &target, None).unwrap();
        assert_eq!(out.files, 3);
        assert_eq!(
            fs::read_to_string(target.join("run.sh")).unwrap(),
            "#!/bin/sh\necho hi\n"
        );
        assert_eq!(mode_of(&target.join("run.sh")), 0o755);
        assert_eq!(mode_of(&target.join("sub/secret.txt")), 0o600);
        assert!(target.join("sub/deeper/empty").is_file());
        assert!(!target.join("link").exists());
    }

    #[test]
    fn digest_ignores_mtimes_but_not_content_or_mode() {
        let a = tempfile::tempdir().unwrap();
        fs::write(a.path().join("f"), "one").unwrap();
        let d1 = pack(a.path(), io::sink()).unwrap().digest;
        // Touch: same content, new mtime.
        fs::write(a.path().join("f"), "one").unwrap();
        assert_eq!(pack(a.path(), io::sink()).unwrap().digest, d1);
        fs::set_permissions(a.path().join("f"), fs::Permissions::from_mode(0o700)).unwrap();
        let d2 = pack(a.path(), io::sink()).unwrap().digest;
        assert_ne!(d2, d1);
        fs::write(a.path().join("f"), "two").unwrap();
        assert_ne!(pack(a.path(), io::sink()).unwrap().digest, d2);
    }

    #[test]
    fn a_single_file_lands_at_the_destination_path() {
        let src = tempfile::tempdir().unwrap();
        let f = src.path().join("reward.txt");
        fs::write(&f, "1.0").unwrap();
        let mut tar_bytes = Vec::new();
        pack(&f, &mut tar_bytes).unwrap();
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("deep/dir/r.txt");
        let out = unpack(&tar_bytes[..], "reward.txt", &target, None).unwrap();
        assert!(out.root_is_file);
        assert_eq!(fs::read_to_string(&target).unwrap(), "1.0");
    }

    /// An entry repeated, first with a mode that forbids writing it, is
    /// replaced, not opened for writing again: the archive's modes cannot
    /// make this machine's own write fail (review F2-A-r11 M2).
    #[test]
    fn a_repeated_entry_replaces_a_read_only_one() {
        let r = EntryType::Regular;
        let file = handmade(&[
            ("a.tar", r, 0o400, b"junk", None),
            ("a.tar", r, 0o644, b"real", None),
        ]);
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("a.tar");
        unpack(&file[..], "a.tar", &target, None).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"real");
        assert_eq!(mode_of(&target), 0o644);
        let folder = handmade(&[
            ("v/reward.txt", r, 0o400, b"0", None),
            ("v/reward.txt", r, 0o000, b"1", None),
        ]);
        let target = dst.path().join("v");
        unpack(&folder[..], "v", &target, None).unwrap();
        fs::set_permissions(target.join("reward.txt"), fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(fs::read(target.join("reward.txt")).unwrap(), b"1");
    }

    /// An archive with nothing under the path is told apart from this
    /// machine's own `NotFound` (review F2-A-r11 L2).
    #[test]
    fn an_empty_archive_is_told_apart() {
        let dst = tempfile::tempdir().unwrap();
        let e = unpack(&[0u8; 1024][..], "v", &dst.path().join("v"), None).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
        assert!(is_empty_archive(&e), "{e}");
        assert!(!is_empty_archive(&io::Error::from(io::ErrorKind::NotFound)));
    }

    /// A reader that counts what was read from it.
    struct Counting<R>(R, std::rc::Rc<std::cell::Cell<u64>>);

    impl<R: Read> Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.0.read(buf)?;
            self.1.set(self.1.get() + n as u64);
            Ok(n)
        }
    }

    /// One extension header claiming `size` bytes, followed by 2 MiB of
    /// its body: what a replaced `tar` can stream.
    fn claimed(kind: EntryType, size: u64) -> (impl Read, std::rc::Rc<std::cell::Cell<u64>>) {
        let mut h = Header::new_gnu();
        h.set_entry_type(kind);
        h.set_size(size);
        h.set_mode(0o644);
        let name = h.as_old_mut().name.as_mut();
        name[..13].copy_from_slice(b"././@LongLink");
        h.set_cksum();
        let read = std::rc::Rc::new(std::cell::Cell::new(0));
        let stream = io::Cursor::new(h.as_bytes().to_vec()).chain(io::repeat(b'a').take(2 << 20));
        (Counting(stream, read.clone()), read)
    }

    /// Review F2-A-r12-2: a long name or PAX header past 64 KiB is refused
    /// from its header, before its body is read: in memory it would be
    /// charged to no cap. 8 GiB takes a base-256 size field, 1 GiB an octal
    /// one.
    #[test]
    fn an_oversized_extension_header_fails_before_its_body_is_read() {
        let dst = tempfile::tempdir().unwrap();
        for (kind, size) in [
            (EntryType::GNULongName, 8 << 30),
            (EntryType::GNULongName, 1 << 30),
            (EntryType::GNULongLink, 1 << 30),
            (EntryType::XHeader, 1 << 30),
            (EntryType::XGlobalHeader, 1 << 30),
            (EntryType::GNULongName, MAX_HEADER_EXTENSION + 1),
        ] {
            let (stream, read) = claimed(kind, size);
            let limits = Limits::capped(1 << 20);
            let e = unpack_limited(stream, "v", &dst.path().join("v"), limits).unwrap_err();
            assert_eq!(
                e.kind(),
                io::ErrorKind::FileTooLarge,
                "{kind:?} {size}: {e}"
            );
            assert!(
                read.get() <= 1 << 20,
                "{kind:?} {size}: read {}",
                read.get()
            );
        }
    }

    /// Long names, long links and PAX headers stacked before an entry are
    /// read as before; a global PAX header is accepted and set aside.
    #[test]
    fn extension_headers_still_name_their_entry() {
        let r = EntryType::Regular;
        let long = format!("v/{}/f.txt", "d".repeat(150));
        let pax_name = format!("v/{}/g.txt", "e".repeat(150));
        let pax = format!("{} path={pax_name}\n", pax_name.len() + 7);
        let pax = format!("{} path={pax_name}\n", pax.len());
        let stream = handmade(&[
            (
                "pax_global_header",
                EntryType::XGlobalHeader,
                0o644,
                b"20 comment=honest\n",
                None,
            ),
            (
                "././@LongLink",
                EntryType::GNULongName,
                0o644,
                long.as_bytes(),
                None,
            ),
            (
                "././@LongLink",
                EntryType::GNULongLink,
                0o644,
                b"target\0",
                None,
            ),
            ("PaxHeader", EntryType::XHeader, 0o644, b"", None),
            ("v/short", r, 0o644, b"one", None),
            ("PaxHeader", EntryType::XHeader, 0o644, pax.as_bytes(), None),
            ("v/other", r, 0o644, b"two", None),
        ]);
        let dst = tempfile::tempdir().unwrap();
        let out = unpack_limited(
            &stream[..],
            "v",
            &dst.path().join("v"),
            Limits::capped(1 << 20),
        )
        .unwrap();
        assert_eq!(out.files, 2, "{:?}", out.skipped);
        let base = dst.path().join("v");
        assert_eq!(fs::read(base.join(&long[2..])).unwrap(), b"one");
        assert_eq!(fs::read(base.join(&pax_name[2..])).unwrap(), b"two");
        // Extension headers with no entry after them are refused.
        let dangling = handmade(&[("././@LongLink", EntryType::GNULongName, 0o644, b"v/x", None)]);
        let e = unpack(&dangling[..], "v", &dst.path().join("w"), None).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{e}");
    }

    /// Review F2-A-r12-3: folders and empty files cost no content, so an
    /// entry budget bounds them, and each folder made on the way counts.
    #[test]
    fn floods_of_folders_or_empty_files_stop_at_the_entry_budget() {
        let limits = Limits {
            bytes: Some(1),
            entries: Some(50),
        };
        let folders: Vec<String> = (0..60).map(|i| format!("v/d{i}")).collect();
        let files: Vec<String> = (0..60).map(|i| format!("v/f{i}")).collect();
        let deep = format!("v/{}f", "a/".repeat(60));
        let mut entries: Vec<Vec<RawEntry<'_>>> = vec![
            folders
                .iter()
                .map(|p| (p.as_str(), EntryType::Directory, 0o755, &b""[..], None))
                .collect(),
            files
                .iter()
                .map(|p| (p.as_str(), EntryType::Regular, 0o644, &b""[..], None))
                .collect(),
        ];
        // One long name 60 folders deep: the folders made count too.
        let deep_stream = {
            let mut b = tar::Builder::new(Vec::new());
            let mut h = Header::new_gnu();
            h.set_size(0);
            h.set_mode(0o644);
            b.append_data(&mut h, &deep, &b""[..]).unwrap();
            b.into_inner().unwrap()
        };
        let dst = tempfile::tempdir().unwrap();
        for (i, list) in entries.drain(..).enumerate() {
            let stream = handmade(&list);
            let e = unpack_limited(&stream[..], "v", &dst.path().join(format!("v{i}")), limits)
                .unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::FileTooLarge, "{e}");
        }
        let e =
            unpack_limited(&deep_stream[..], "v", &dst.path().join("deep"), limits).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::FileTooLarge, "{e}");
        // An honest folder of many small files: headers and padding are not
        // charged against the content cap.
        let names: Vec<String> = (0..300).map(|i| format!("v/s{i}")).collect();
        let honest: Vec<RawEntry<'_>> = names
            .iter()
            .map(|p| (p.as_str(), EntryType::Regular, 0o644, &b"x"[..], None))
            .collect();
        let stream = handmade(&honest);
        let out = unpack_limited(
            &stream[..],
            "v",
            &dst.path().join("ok"),
            Limits::capped(300),
        )
        .unwrap();
        assert_eq!(out.files, 300);
    }

    /// Refused entries keep at most 100 records, each path cut short.
    #[test]
    fn refused_entries_are_kept_short_and_counted() {
        let names: Vec<String> = (0..150)
            .map(|i| format!("v/{i}-{}", "n".repeat(90)))
            .collect();
        let list: Vec<RawEntry<'_>> = names
            .iter()
            .map(|p| (p.as_str(), EntryType::Fifo, 0o644, &b""[..], None))
            .collect();
        let long = format!("v/{}", "z".repeat(2_000));
        let mut stream = handmade(&[(
            "././@LongLink",
            EntryType::GNULongName,
            0o644,
            long.as_bytes(),
            None,
        )]);
        stream.truncate(stream.len() - 1024);
        let mut fifo = Header::new_gnu();
        fifo.set_entry_type(EntryType::Fifo);
        fifo.set_size(0);
        fifo.as_old_mut().name.as_mut()[..3].copy_from_slice(b"v/x");
        fifo.set_cksum();
        stream.extend_from_slice(fifo.as_bytes());
        stream.extend(handmade(&list));
        let dst = tempfile::tempdir().unwrap();
        let out = unpack(&stream[..], "v", &dst.path().join("v"), None).unwrap();
        assert_eq!(out.skipped.len(), MAX_SKIPPED_KEPT);
        assert_eq!(out.skipped_more, 51);
        assert!(out.skipped[0].path.chars().count() <= MAX_SHOWN + 1);
        assert!(out.skipped[0].path.starts_with("v/zzz"));
    }

    #[test]
    fn pack_bytes_writes_one_entry_with_its_mode() {
        let mut tar_bytes = Vec::new();
        pack_bytes("cfg.json", b"{}", 0o640, &mut tar_bytes).unwrap();
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("cfg.json");
        unpack(&tar_bytes[..], "cfg.json", &target, None).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"{}");
        assert_eq!(mode_of(&target), 0o640);
    }

    #[test]
    fn refuses_escapes_links_and_special_files() {
        let archive = handmade(&[
            ("out/", EntryType::Directory, 0o755, b"", None),
            ("out/ok.txt", EntryType::Regular, 0o644, b"fine", None),
            ("/etc/evil", EntryType::Regular, 0o644, b"x", None),
            ("out/../../evil", EntryType::Regular, 0o644, b"x", None),
            (
                "out/sym",
                EntryType::Symlink,
                0o777,
                b"",
                Some("/etc/passwd"),
            ),
            ("out/hard", EntryType::Link, 0o644, b"", Some("out/ok.txt")),
            ("out/fifo", EntryType::Fifo, 0o644, b"", None),
            ("other/file", EntryType::Regular, 0o644, b"x", None),
        ]);
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("got");
        let out = unpack(&archive[..], "out", &target, None).unwrap();
        assert_eq!(out.files, 1);
        assert_eq!(fs::read_to_string(target.join("ok.txt")).unwrap(), "fine");
        let reasons: Vec<_> = out
            .skipped
            .iter()
            .map(|s| (s.path.as_str(), s.reason))
            .collect();
        assert_eq!(
            reasons,
            vec![
                ("/etc/evil", "absolute path"),
                ("out/../../evil", "path with .."),
                ("out/sym", "symlink"),
                ("out/hard", "hard link"),
                ("out/fifo", "special file (device or FIFO)"),
                ("other/file", "outside the requested path"),
            ]
        );
        assert!(!dst.path().join("evil").exists());
    }

    #[test]
    fn never_writes_through_an_existing_symlink() {
        let outside = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("got");
        fs::create_dir_all(&target).unwrap();
        symlink(outside.path(), target.join("sub")).unwrap();
        let archive = handmade(&[
            ("out/sub/", EntryType::Directory, 0o755, b"", None),
            ("out/sub/planted", EntryType::Regular, 0o644, b"x", None),
        ]);
        let out = unpack(&archive[..], "out", &target, None).unwrap();
        assert_eq!(out.files, 0);
        assert!(!outside.path().join("planted").exists());
        assert!(
            out.skipped
                .iter()
                .all(|s| s.reason == "a symlink is in the way at the destination")
        );
    }

    #[test]
    fn stops_at_the_size_cap_before_reading_a_big_file() {
        let archive = handmade(&[
            ("out/", EntryType::Directory, 0o755, b"", None),
            ("out/a", EntryType::Regular, 0o644, &[b'a'; 600], None),
            ("out/b", EntryType::Regular, 0o644, &[b'b'; 600], None),
        ]);
        let dst = tempfile::tempdir().unwrap();
        let err = unpack(&archive[..], "out", &dst.path().join("got"), Some(1000)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::FileTooLarge);
        assert!(err.to_string().contains("1000-byte cap"), "{err}");
    }

    #[test]
    fn read_only_folders_still_get_their_contents() {
        let archive = handmade(&[
            ("out/", EntryType::Directory, 0o755, b"", None),
            ("out/ro/", EntryType::Directory, 0o555, b"", None),
            ("out/ro/f", EntryType::Regular, 0o444, b"data", None),
        ]);
        let dst = tempfile::tempdir().unwrap();
        let target = dst.path().join("got");
        unpack(&archive[..], "out", &target, None).unwrap();
        assert_eq!(fs::read_to_string(target.join("ro/f")).unwrap(), "data");
        assert_eq!(mode_of(&target.join("ro")), 0o755);
        assert_eq!(mode_of(&target.join("ro/f")), 0o444);
    }

    #[test]
    fn an_empty_archive_is_not_found() {
        let archive = tar::Builder::new(Vec::new()).into_inner().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let err = unpack(&archive[..], "x", &dst.path().join("x"), None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    struct SkipLogs;
    impl PackFilter for SkipLogs {
        fn keep(&self, rel: &str, _is_dir: bool) -> bool {
            !rel.starts_with("logs")
        }
        fn descend(&self, rel_dir: &str) -> bool {
            rel_dir != "logs"
        }
    }

    #[test]
    fn filters_leave_paths_out() {
        let src = tempfile::tempdir().unwrap();
        fs::create_dir_all(src.path().join("logs")).unwrap();
        fs::write(src.path().join("logs/big.log"), "x").unwrap();
        fs::write(src.path().join("keep.txt"), "k").unwrap();
        let mut tar_bytes = Vec::new();
        let packed = pack_filtered(src.path(), &SkipLogs, &mut tar_bytes).unwrap();
        assert_eq!((packed.files, packed.dirs), (1, 0));
    }
}
