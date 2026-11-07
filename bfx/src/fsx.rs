//! Path resolution beneath a held directory: never a link, never a mount.
//!
//! Every file primitive starts from a *root*: an absolute directory opened
//! from `/` one component at a time with `O_NOFOLLOW`, so no link on the way
//! is followed (mounts on the way to the root are allowed; the host names
//! roots such as a bind-mounted `/logs/verifier`). Beneath the root, every
//! name is resolved one component at a time from the directory already held:
//!
//! - with `openat2` and `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
//!   RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV`, so the kernel refuses a link
//!   (`ELOOP`) and a mount point (`EXDEV`);
//! - where `openat2` is missing (`ENOSYS`, or `EPERM` from an older seccomp
//!   profile), with `openat(..., O_NOFOLLOW)` and a check that each opened
//!   component has the root's device and mount id (`mnt_id` in
//!   `/proc/self/fdinfo`), giving the same results. Without a procfs that
//!   check cannot be made, and every operation beneath a root fails.
//!
//! Path arguments are relative, with no empty, `.` or `..` component; names
//! containing dots (`..a`, `...`) are ordinary names. A failure that comes
//! from a link anywhere on the path is reported as `ELOOP` in both modes.
//! Every operation then acts on the descriptor it opened, never on a path a
//! second time.

use crate::err::{Error, Result};
use rustix::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use rustix::fs::{self, AtFlags, Mode, OFlags, ResolveFlags, Stat};
use rustix::io::Errno;
use serde_json::{json, Map, Value};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Set by a request's `"resolver": "walk"`: use the component walk even where
/// `openat2` exists (tests compare the two).
pub static FORCE_WALK: AtomicBool = AtomicBool::new(false);

const PROC_SUPER_MAGIC: u64 = 0x9fa0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolver {
    Openat2,
    Walk,
}

impl Resolver {
    pub fn name(self) -> &'static str {
        match self {
            Resolver::Openat2 => "openat2",
            Resolver::Walk => "walk",
        }
    }
}

pub fn resolver() -> Resolver {
    static PROBE: OnceLock<Resolver> = OnceLock::new();
    if FORCE_WALK.load(Ordering::Relaxed) {
        return Resolver::Walk;
    }
    *PROBE.get_or_init(|| {
        match fs::openat2(
            fs::CWD,
            "/",
            OFlags::PATH | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::NO_SYMLINKS,
        ) {
            Err(Errno::NOSYS) | Err(Errno::PERM) => Resolver::Walk,
            _ => Resolver::Openat2,
        }
    })
}

struct Proc {
    fd: OwnedFd,
    fdinfo: OwnedFd,
}

/// `/proc/self/fd` and `/proc/self/fdinfo`, opened once and checked to be
/// procfs. None when there is no procfs.
fn proc_dirs() -> Option<&'static Proc> {
    static P: OnceLock<Option<Proc>> = OnceLock::new();
    P.get_or_init(|| {
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC;
        let fd = fs::open("/proc/self/fd", flags, Mode::empty()).ok()?;
        let fdinfo = fs::open("/proc/self/fdinfo", flags, Mode::empty()).ok()?;
        for d in [&fd, &fdinfo] {
            if fs::fstatfs(d).ok()?.f_type as u64 != PROC_SUPER_MAGIC {
                return None;
            }
        }
        Some(Proc { fd, fdinfo })
    })
    .as_ref()
}

pub fn have_proc() -> bool {
    proc_dirs().is_some()
}

fn no_proc() -> Error {
    Error::new("no-proc", "/proc is not procfs here, so mount boundaries and descriptors cannot be checked")
}

/// The mount id of an open descriptor, from `/proc/self/fdinfo`.
pub fn mnt_id(fd: BorrowedFd) -> Result<u64> {
    let p = proc_dirs().ok_or_else(no_proc)?;
    let name = fd.as_raw_fd().to_string();
    let f = fs::openat(&p.fdinfo, name.as_str(), OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty())
        .map_err(|e| Error::sys(e, "reading fdinfo"))?;
    let text = read_to_end(f.as_fd(), 1 << 16)?;
    for line in text.split(|b| *b == b'\n') {
        if let Some(rest) = line.strip_prefix(b"mnt_id:") {
            let s = std::str::from_utf8(rest).unwrap_or("").trim();
            if let Ok(n) = s.parse::<u64>() {
                return Ok(n);
            }
        }
    }
    Err(Error::new("no-proc", "fdinfo has no mnt_id"))
}

/// The `/proc/self/fd/<n>` name of an open descriptor, relative to the held
/// `/proc/self/fd`: operations that have no descriptor form (`chmod` on an
/// `O_PATH` descriptor, `getxattr`) reach the open file through it.
pub fn chmod_fd(fd: BorrowedFd, mode: u32) -> Result<()> {
    let p = proc_dirs().ok_or_else(no_proc)?;
    let name = fd.as_raw_fd().to_string();
    fs::chmodat(&p.fd, name.as_str(), Mode::from_raw_mode(mode), AtFlags::empty())
        .map_err(|e| Error::sys(e, "chmod"))
}

pub fn proc_fd_path(fd: BorrowedFd) -> Result<String> {
    proc_dirs().ok_or_else(no_proc)?;
    Ok(format!("/proc/self/fd/{}", fd.as_raw_fd()))
}

pub fn read_to_end(fd: BorrowedFd, limit: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = match rustix::io::read(fd, &mut buf[..]) {
            Ok(n) => n,
            Err(Errno::INTR) => continue,
            Err(e) => return Err(Error::sys(e, "read")),
        };
        if n == 0 {
            return Ok(out);
        }
        out.extend_from_slice(&buf[..n]);
        if out.len() > limit {
            return Err(Error::new("too-large", "input is larger than allowed"));
        }
    }
}

/// A held directory with the device and mount every entry beneath it must
/// share.
pub struct Node {
    pub fd: OwnedFd,
    pub dev: u64,
    pub mnt: Option<u64>,
}

impl Node {
    fn new(fd: OwnedFd) -> Result<Node> {
        let st = fs::fstat(&fd).map_err(|e| Error::sys(e, "fstat"))?;
        let mnt = if resolver() == Resolver::Walk { Some(mnt_id(fd.as_fd())?) } else { None };
        Ok(Node { fd, dev: st.st_dev as u64, mnt })
    }

    /// A directory opened beneath this one: same mount by construction.
    fn child(&self, fd: OwnedFd) -> Node {
        Node { fd, dev: self.dev, mnt: self.mnt }
    }

    pub fn stat(&self) -> Result<Stat> {
        fs::fstat(&self.fd).map_err(|e| Error::sys(e, "fstat"))
    }
}

pub fn os(name: &[u8]) -> &OsStr {
    OsStr::from_bytes(name)
}

pub fn show(name: &[u8]) -> String {
    String::from_utf8_lossy(name).into_owned()
}

/// Split a relative path into its components; refuses absolute paths and
/// empty, `.` and `..` components. The empty path is the root itself.
pub fn split_rel(rel: &[u8]) -> Result<Vec<&[u8]>> {
    if rel.is_empty() {
        return Ok(Vec::new());
    }
    if rel[0] == b'/' {
        return Err(Error::bad(format!("{:?} must be relative to its root", show(rel))));
    }
    let mut out = Vec::new();
    for c in rel.split(|b| *b == b'/') {
        if c.is_empty() || c == b"." || c == b".." || c.contains(&0) {
            return Err(Error::bad(format!("{:?} has an empty, '.' or '..' component", show(rel))));
        }
        out.push(c);
    }
    Ok(out)
}

/// Split an absolute path; `/` alone has no components.
pub fn split_abs(path: &[u8]) -> Result<Vec<&[u8]>> {
    if path.first() != Some(&b'/') {
        return Err(Error::bad(format!("{:?} must be absolute", show(path))));
    }
    if path == b"/" {
        return Ok(Vec::new());
    }
    split_rel(&path[1..])
}

/// Open an absolute directory as a root: from `/`, one component at a time,
/// never following a link. Mounts on the way are allowed.
pub fn open_root(path: &str) -> Result<Node> {
    open_root_bytes(path.as_bytes())
}

/// [`open_root`] for a path given as raw bytes.
pub fn open_root_bytes(path: &[u8]) -> Result<Node> {
    let comps = split_abs(path)?;
    let dir = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut cur = fs::open("/", dir, Mode::empty()).map_err(|e| Error::sys(e, "/"))?;
    for (i, c) in comps.iter().enumerate() {
        cur = match fs::openat(&cur, os(c), dir, Mode::empty()) {
            Ok(fd) => fd,
            Err(e) => {
                let e = if matches!(e, Errno::NOTDIR | Errno::LOOP) && is_link(cur.as_fd(), c) { Errno::LOOP } else { e };
                let at = format!("/{}", comps[..=i].iter().map(|c| show(c)).collect::<Vec<_>>().join("/"));
                return Err(Error::sys(e, at));
            }
        };
    }
    Node::new(cur)
}

fn is_link(dir: BorrowedFd, name: &[u8]) -> bool {
    matches!(fs::statat(dir, os(name), AtFlags::SYMLINK_NOFOLLOW), Ok(st) if ftype(st.st_mode as u32) == "link")
}

/// Open one name in *parent*, never following a link and never crossing a
/// mount. `flags` are the open flags; `O_NOFOLLOW` and `O_CLOEXEC` are
/// always added.
pub fn open_one(parent: &Node, name: &[u8], flags: OFlags) -> std::result::Result<OwnedFd, Errno> {
    let flags = flags | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let r = match resolver() {
        Resolver::Openat2 => fs::openat2(
            &parent.fd,
            os(name),
            flags,
            Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS | ResolveFlags::NO_XDEV,
        ),
        Resolver::Walk => fs::openat(&parent.fd, os(name), flags, Mode::empty()).and_then(|fd| {
            let st = fs::fstat(&fd)?;
            let same_dev = st.st_dev as u64 == parent.dev;
            let same_mnt = match parent.mnt {
                Some(m) => mnt_id(fd.as_fd()).map_err(|_| Errno::NOSYS)? == m,
                None => false,
            };
            if same_dev && same_mnt {
                Ok(fd)
            } else {
                Err(Errno::XDEV)
            }
        }),
    };
    r.map_err(|e| {
        // The same result in both modes: whatever made a link fail.
        if matches!(e, Errno::NOTDIR | Errno::LOOP) && is_link(parent.fd.as_fd(), name) {
            Errno::LOOP
        } else {
            e
        }
    })
}

fn join(comps: &[&[u8]]) -> String {
    comps.iter().map(|c| show(c)).collect::<Vec<_>>().join("/")
}

/// The directory at *comps* beneath *root*.
pub fn open_dir(root: &Node, comps: &[&[u8]]) -> Result<Node> {
    let dir = OFlags::RDONLY | OFlags::DIRECTORY;
    if comps.is_empty() {
        let fd = fs::openat(&root.fd, ".", dir | OFlags::CLOEXEC, Mode::empty()).map_err(|e| Error::sys(e, "."))?;
        return Ok(root.child(fd));
    }
    let mut cur: Option<Node> = None;
    for (i, c) in comps.iter().enumerate() {
        let parent = cur.as_ref().unwrap_or(root);
        let fd = open_one(parent, c, dir).map_err(|e| Error::sys(e, join(&comps[..=i])))?;
        cur = Some(root.child(fd));
    }
    Ok(cur.unwrap())
}

/// The parent directory of a non-empty *comps*, and the last name.
pub fn open_parent<'c>(root: &Node, comps: &[&'c [u8]]) -> Result<(Node, &'c [u8])> {
    let (last, dirs) = comps.split_last().ok_or_else(|| Error::bad("this operation needs a path beneath its root"))?;
    Ok((open_dir(root, dirs)?, *last))
}

/// An entry of a directory, opened `O_PATH` (no side effects) unless it is a
/// mount point, which is only reported.
pub struct Entry {
    pub st: Stat,
    pub mount: bool,
    pub fd: Option<OwnedFd>,
}

pub fn entry(parent: &Node, name: &[u8]) -> std::result::Result<Entry, Errno> {
    match open_one(parent, name, OFlags::PATH) {
        Ok(fd) => {
            let st = fs::fstat(&fd)?;
            Ok(Entry { st, mount: false, fd: Some(fd) })
        }
        Err(Errno::XDEV) => {
            let st = fs::statat(&parent.fd, os(name), AtFlags::SYMLINK_NOFOLLOW)?;
            Ok(Entry { st, mount: true, fd: None })
        }
        Err(e) => Err(e),
    }
}

/// The directory an `O_PATH` entry names, opened for reading through that
/// descriptor (no second lookup by name).
pub fn reopen_dir(parent: &Node, e: &Entry) -> std::result::Result<Node, Errno> {
    let fd = e.fd.as_ref().ok_or(Errno::XDEV)?;
    let dir = fs::openat(fd, ".", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty())?;
    Ok(parent.child(dir))
}

/// A regular file entry opened for reading through its held `O_PATH`
/// descriptor (`/proc/self/fd`), so no second lookup by name can open
/// another inode, a FIFO or a device first. Without /proc (or a held
/// descriptor) it opens by name, non-blocking; callers still compare
/// (dev, ino) with the entry they saw.
pub fn reopen_file(parent: &Node, name: &[u8], e: &Entry) -> std::result::Result<OwnedFd, Errno> {
    let flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC;
    if let Some(held) = e.fd.as_ref() {
        if let Ok(path) = proc_fd_path(held.as_fd()) {
            return fs::open(path.as_str(), flags, Mode::empty());
        }
    }
    open_one(parent, name, flags)
}

/// The names in a directory, sorted, without `.` and `..`.
pub fn names(dir: &Node) -> Result<Vec<Vec<u8>>> {
    let mut d = fs::Dir::read_from(&dir.fd).map_err(|e| Error::sys(e, "reading a directory"))?;
    let mut out = Vec::new();
    while let Some(ent) = d.read() {
        let ent = ent.map_err(|e| Error::sys(e, "reading a directory"))?;
        let n = ent.file_name().to_bytes();
        if n != b"." && n != b".." {
            out.push(n.to_vec());
        }
    }
    out.sort();
    Ok(out)
}

pub fn ftype(mode: u32) -> &'static str {
    match mode & 0o170000 {
        0o100000 => "file",
        0o040000 => "dir",
        0o120000 => "link",
        0o010000 => "fifo",
        0o140000 => "sock",
        0o020000 => "chr",
        0o060000 => "blk",
        _ => "unknown",
    }
}

pub fn stat_json(st: &Stat) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".into(), json!(ftype(st.st_mode as u32)));
    m.insert("mode".into(), json!(st.st_mode as u32 & 0o7777));
    m.insert("uid".into(), json!(st.st_uid as u32));
    m.insert("gid".into(), json!(st.st_gid as u32));
    m.insert("size".into(), json!(st.st_size as i64));
    m.insert("nlink".into(), json!(st.st_nlink as u64));
    m.insert("dev".into(), json!(st.st_dev as u64));
    m.insert("ino".into(), json!(st.st_ino as u64));
    m.insert("mtime".into(), json!(st.st_mtime as i64));
    if matches!(ftype(st.st_mode as u32), "chr" | "blk") {
        m.insert("rdev".into(), json!(st.st_rdev as u64));
    }
    m
}

/// A path for JSON output: a string when it is UTF-8, else base64 bytes
/// under `path_b64`.
pub fn put_path(m: &mut Map<String, Value>, key: &str, p: &[u8]) {
    match std::str::from_utf8(p) {
        Ok(s) => {
            m.insert(key.into(), json!(s));
        }
        Err(_) => {
            m.insert(format!("{key}_b64"), json!(crate::b64::encode(p)));
        }
    }
}

pub fn readlink_at(parent: &Node, name: &[u8]) -> Option<Vec<u8>> {
    fs::readlinkat(&parent.fd, os(name), Vec::new()).ok().map(|c| c.into_bytes())
}

/// `rel/name` as bytes.
pub fn child_path(rel: &[u8], name: &[u8]) -> Vec<u8> {
    if rel.is_empty() {
        name.to_vec()
    } else {
        let mut p = rel.to_vec();
        p.push(b'/');
        p.extend_from_slice(name);
        p
    }
}
