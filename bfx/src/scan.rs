//! `scan`: facts about the image, no decisions (results/TR-PROTOCOL.md v1).
//! The host's checklist (`benchflow.trust.checklist`) drives it in rounds
//! and decides what is sealed; scan only reports.
//!
//! `{paths: [abs..], walks: [{root, depth}..], contents: [abs..],
//!   elf: [abs..], interpreters: [abs..], processes=false, sysctl: [name..],
//!   mounts=true, acl=true, content_max_bytes=4 MiB, max_entries=2000000,
//!   max_bytes=128 MiB, timeout_ms=10000, as={uid: 65534, gid: 65534}}`
//!
//! - `paths`: each looked at component by component from `/`, never
//!   following a link: an entry for every component prefix, stopping at the
//!   first that is missing (`"exists": false`) or not a directory.
//! - `walks`: `root` listed to `depth` levels (1 = its own entries); links
//!   and mount points are listed (`"mount": true`), never entered. A
//!   directory whose names are all present has `"listing": true`.
//! - Every entry: lstat facts, `link` (raw bytes) for a link, `acl` when the
//!   inode has a POSIX ACL, `mount_id` from fdinfo. Entries are unique by
//!   path.
//! - `contents`: the bytes of each regular file (never through a link),
//!   `too-large` above `content_max_bytes`. `elf`: ELF facts (elf.rs), or
//!   `elf: null` and the shebang line.
//! - `processes`: every other process's ids, groups, exe, descriptors and
//!   mapped files. `sysctl`: values under /proc/sys. `mounts`: mountinfo.
//! - `interpreters`: each run twice as `as` (never root), with an empty
//!   environment, `/` as its folder, no_new_privs, its own session and a
//!   timeout: with `-I`, then with `-S -s -E`, to report what it says its
//!   paths, prefixes and ABI tags are.
//! - `errors`: every per-path failure `{path, op, error}`. More than
//!   `max_entries` entries, or an answer above `max_bytes`, fails the
//!   request: an answer is never truncated.
//! - A path that is not UTF-8 is given as base64 bytes: `paths_b64`,
//!   `contents_b64` (answered in `contents_b64`, keyed by the base64 text)
//!   and a walk's `root_b64`. Paths in the answer are text, or `path_b64`.

use crate::args::Args;
use crate::err::{Error, Result};
use crate::exec::{spawn_capture, Caps, PrepareData};
use crate::fsx::{self, entry, ftype, names, reopen_dir, split_abs, Node};
use crate::ops::entry_json;
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::fs::{self, AtFlags, Mode, OFlags};
use rustix::io::Errno;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PY_FACTS: &str = r#"
import sys, json
try:
    import sysconfig
    cv = sysconfig.get_config_var
    paths = sysconfig.get_paths()
except Exception:
    cv = lambda k: None
    paths = {}
try:
    import site
    sp = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    usp = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
    eus = getattr(site, "ENABLE_USER_SITE", None)
except Exception:
    sp, usp, eus = [], None, None
v = sys.version_info
out = {
    "executable": sys.executable,
    "version": [v[0], v[1], v[2], v[3], v[4]],
    "hexversion": sys.hexversion,
    "implementation": sys.implementation.name,
    "cache_tag": sys.implementation.cache_tag,
    "prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "exec_prefix": sys.exec_prefix,
    "base_exec_prefix": getattr(sys, "base_exec_prefix", sys.exec_prefix),
    "path": sys.path,
    "site_packages": sp,
    "user_site": usp,
    "enable_user_site": eus,
    "sysconfig_paths": paths,
    "soabi": cv("SOABI"),
    "ext_suffix": cv("EXT_SUFFIX"),
    "multiarch": cv("MULTIARCH"),
    "abiflags": getattr(sys, "abiflags", ""),
    "platform": sys.platform,
    "maxsize": sys.maxsize,
    "byteorder": sys.byteorder,
    "isolated": sys.flags.isolated,
    "no_site": sys.flags.no_site,
}
sys.stdout.write(json.dumps(out))
"#;

/// The two interpreter runs (TR-PROTOCOL.md §3.6).
const INTERPRETER_RUNS: [&[&str]; 2] = [&["-I"], &["-S", "-s", "-E"]];

const DIR: OFlags = OFlags::RDONLY.union(OFlags::DIRECTORY).union(OFlags::NOFOLLOW).union(OFlags::CLOEXEC);

fn acl_entries(raw: &[u8]) -> Option<Value> {
    // posix_acl_xattr: u32 version (2), then (u16 tag, u16 perm, u32 id).
    if raw.len() < 4 || (raw.len() - 4) % 8 != 0 {
        return None;
    }
    let mut out = Vec::new();
    for e in raw[4..].chunks(8) {
        let tag = u16::from_le_bytes([e[0], e[1]]);
        let perm = u16::from_le_bytes([e[2], e[3]]);
        let id = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
        out.push(json!([tag, perm, id]));
    }
    Some(Value::Array(out))
}

/// The inode's POSIX ACLs; `{"error"}` when one is present but cannot be
/// read or decoded (the host then treats the inode as writable by all).
fn acl_of(fd: BorrowedFd) -> Option<Value> {
    let path = match fsx::proc_fd_path(fd) {
        Ok(p) => p,
        Err(_) => return Some(json!({"error": "no /proc to read ACLs through"})),
    };
    let mut m = Map::new();
    for (key, name) in [("access", "system.posix_acl_access"), ("default", "system.posix_acl_default")] {
        // Most inodes have no ACL: a small buffer, grown only when it is short.
        let mut buf = vec![0u8; 256];
        let mut got = fs::getxattr(path.as_str(), name, &mut buf[..]);
        if got == Err(Errno::RANGE) {
            buf = vec![0u8; 65536];
            got = fs::getxattr(path.as_str(), name, &mut buf[..]);
        }
        match got {
            Ok(n) => match acl_entries(&buf[..n]) {
                Some(v) => {
                    m.insert(key.into(), v);
                }
                None => return Some(json!({"error": format!("{name} cannot be decoded")})),
            },
            Err(Errno::NODATA) | Err(Errno::OPNOTSUPP) => {}
            Err(e) => return Some(json!({"error": format!("{name}: {}", errno_name(e))})),
        }
    }
    if m.is_empty() {
        None
    } else {
        Some(Value::Object(m))
    }
}

/// The mount id of an open inode: one statx (Linux 5.8+), else fdinfo.
fn mount_id(fd: BorrowedFd) -> Option<u64> {
    match fs::statx(fd, "", AtFlags::EMPTY_PATH, fs::StatxFlags::MNT_ID) {
        Ok(x) if x.stx_mask & fs::StatxFlags::MNT_ID.bits() != 0 => Some(x.stx_mnt_id),
        _ => fsx::mnt_id(fd).ok(),
    }
}

fn errno_name(e: Errno) -> String {
    match e {
        Errno::NOENT => "missing".into(),
        Errno::LOOP => "link".into(),
        Errno::NOTDIR => "not-directory".into(),
        Errno::ACCESS | Errno::PERM => "permission".into(),
        Errno::XDEV => "mount".into(),
        _ => format!("errno {}", e.raw_os_error()),
    }
}

fn is_link_at(dir: BorrowedFd, name: &[u8]) -> bool {
    matches!(fs::statat(dir, fsx::os(name), AtFlags::SYMLINK_NOFOLLOW), Ok(st) if ftype(st.st_mode as u32) == "link")
}

/// `prefix/name` for an absolute prefix (`/` included).
fn join_abs(prefix: &[u8], name: &[u8]) -> Vec<u8> {
    let mut p = prefix.to_vec();
    if p.last() != Some(&b'/') {
        p.push(b'/');
    }
    p.extend_from_slice(name);
    p
}

/// The regular file at *path*, reached from `/` without following any link,
/// opened for reading only once it is known to be a regular file (so no
/// device or FIFO is ever opened). Errors are TR-PROTOCOL's names.
fn open_regular(path: &[u8]) -> std::result::Result<(OwnedFd, fs::Stat), String> {
    let comps = split_abs(path).map_err(|_| "bad-path".to_string())?;
    let (last, dirs) = comps.split_last().ok_or_else(|| "not-regular".to_string())?;
    let mut dir = fs::open("/", DIR, Mode::empty()).map_err(errno_name)?;
    for c in dirs {
        dir = match fs::openat(&dir, fsx::os(c), DIR, Mode::empty()) {
            Ok(fd) => fd,
            Err(e) if matches!(e, Errno::NOTDIR | Errno::LOOP) && is_link_at(dir.as_fd(), c) => return Err("link".into()),
            Err(e) => return Err(errno_name(e)),
        };
    }
    let held = fs::openat(&dir, fsx::os(last), OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty()).map_err(errno_name)?;
    let st = fs::fstat(&held).map_err(errno_name)?;
    match ftype(st.st_mode as u32) {
        "file" => {}
        "link" => return Err("link".into()),
        _ => return Err("not-regular".into()),
    }
    // Reopen the inode itself through /proc (no second lookup by name);
    // without /proc, open by name and check it is the same inode.
    let fd = match fsx::proc_fd_path(held.as_fd()) {
        Ok(p) => fs::open(p.as_str(), OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOCTTY, Mode::empty()).map_err(errno_name)?,
        Err(_) => {
            let fd = fs::openat(&dir, fsx::os(last), OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC, Mode::empty()).map_err(errno_name)?;
            let again = fs::fstat(&fd).map_err(errno_name)?;
            if (again.st_dev, again.st_ino) != (st.st_dev, st.st_ino) {
                return Err("changed".into());
            }
            fd
        }
    };
    Ok((fd, st))
}

struct Scan {
    acl: bool,
    max_entries: u64,
    entries: Vec<Value>,
    /// Each entry's index, by path: entries are unique by path.
    seen: HashMap<Vec<u8>, usize>,
    errors: Vec<Value>,
}

impl Scan {
    /// The index of *path*'s entry: *m* when it is new, else the one kept.
    fn push(&mut self, path: &[u8], m: Map<String, Value>) -> Result<usize> {
        if let Some(i) = self.seen.get(path) {
            return Ok(*i);
        }
        if self.entries.len() as u64 >= self.max_entries {
            return Err(Error::new("too-many", format!("more than {} entries", self.max_entries)));
        }
        self.entries.push(Value::Object(m));
        self.seen.insert(path.to_vec(), self.entries.len() - 1);
        Ok(self.entries.len() - 1)
    }

    fn error(&mut self, path: &[u8], op: &str, why: String) {
        let mut m = Map::new();
        fsx::put_path(&mut m, "path", path);
        m.insert("op".into(), json!(op));
        m.insert("error".into(), json!(why));
        self.errors.push(Value::Object(m));
    }

    fn missing(&mut self, path: &[u8]) -> Result<()> {
        let mut m = Map::new();
        fsx::put_path(&mut m, "path", path);
        m.insert("exists".into(), json!(false));
        self.push(path, m).map(|_| ())
    }

    /// The facts of an inode held by *fd* (O_PATH, or a directory).
    fn facts(&self, path: &[u8], st: &fs::Stat, fd: Option<BorrowedFd>, link: Option<Vec<u8>>) -> Map<String, Value> {
        let mut m = fsx::stat_json(st);
        fsx::put_path(&mut m, "path", path);
        m.insert("exists".into(), json!(true));
        if let Some(t) = link {
            fsx::put_path(&mut m, "link", &t);
        }
        if let Some(fd) = fd {
            if let Some(id) = mount_id(fd) {
                m.insert("mount_id".into(), json!(id));
            }
            if self.acl && ftype(st.st_mode as u32) != "link" {
                if let Some(a) = acl_of(fd) {
                    m.insert("acl".into(), a);
                }
            }
        }
        m
    }

    /// `paths`: every component prefix of *path*, from `/`.
    fn prefixes(&mut self, path: &[u8]) -> Result<()> {
        let comps = match split_abs(path) {
            Ok(c) => c,
            Err(_) => {
                self.error(path, "path", "bad-path".into());
                return Ok(());
            }
        };
        let mut dir = match fs::open("/", DIR, Mode::empty()) {
            Ok(fd) => fd,
            Err(e) => {
                self.error(b"/", "path", errno_name(e));
                return Ok(());
            }
        };
        if !self.seen.contains_key(&b"/"[..]) {
            let st = fs::fstat(&dir).map_err(|e| Error::sys(e, "/"))?;
            let m = self.facts(b"/", &st, Some(dir.as_fd()), None);
            self.push(b"/", m)?;
        }
        let mut at: Vec<u8> = b"/".to_vec();
        for c in comps {
            at = join_abs(&at, c);
            let held = match fs::openat(&dir, fsx::os(c), OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty()) {
                Ok(fd) => fd,
                Err(Errno::NOENT) => return self.missing(&at),
                Err(e) => {
                    self.error(&at, "path", errno_name(e));
                    return Ok(());
                }
            };
            let st = fs::fstat(&held).map_err(|e| Error::sys(e, fsx::show(&at)))?;
            let kind = ftype(st.st_mode as u32);
            let link = if kind == "link" { fs::readlinkat(&dir, fsx::os(c), Vec::new()).ok().map(|t| t.into_bytes()) } else { None };
            let m = self.facts(&at, &st, Some(held.as_fd()), link);
            self.push(&at, m)?;
            if kind != "dir" {
                return Ok(());
            }
            dir = match fs::openat(&held, ".", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty()) {
                Ok(fd) => fd,
                Err(e) => {
                    self.error(&at, "path", errno_name(e));
                    return Ok(());
                }
            };
        }
        Ok(())
    }

    /// `walks`: *root* and its names down to *depth* levels.
    fn walk(&mut self, root: &[u8], depth: u64) -> Result<()> {
        let node = match fsx::open_root_bytes(root) {
            Ok(n) => n,
            Err(e) if e.is(Errno::NOENT) => return self.missing(root),
            Err(e) => {
                let why = e.errno.map(|n| errno_name(Errno::from_raw_os_error(n))).unwrap_or_else(|| e.kind.to_string());
                self.error(root, "walk", why);
                return Ok(());
            }
        };
        let st = node.stat()?;
        let m = self.facts(root, &st, Some(node.fd.as_fd()), None);
        let at = self.push(root, m)?;
        if depth > 0 && self.list(&node, root, 1, depth)? {
            self.entries[at]["listing"] = json!(true);
        }
        Ok(())
    }

    /// List *dir* (at *level*); true when every name was read.
    fn list(&mut self, dir: &Node, abs: &[u8], level: u64, depth: u64) -> Result<bool> {
        let ns = match names(dir) {
            Ok(n) => n,
            Err(e) => {
                self.error(abs, "list", e.msg.clone());
                return Ok(false);
            }
        };
        for n in ns {
            let p = join_abs(abs, &n);
            let e = match entry(dir, &n) {
                Ok(e) => e,
                Err(Errno::NOENT) => continue,
                Err(err) => {
                    self.error(&p, "lstat", errno_name(err));
                    continue;
                }
            };
            let mut m = entry_json(&p, dir, &n, &e);
            m.insert("exists".into(), json!(true));
            if let Some(fd) = &e.fd {
                if let Some(id) = mount_id(fd.as_fd()) {
                    m.insert("mount_id".into(), json!(id));
                }
                if self.acl && ftype(e.st.st_mode as u32) != "link" {
                    if let Some(a) = acl_of(fd.as_fd()) {
                        m.insert("acl".into(), a);
                    }
                }
            }
            let at = self.push(&p, m)?;
            if !e.mount && ftype(e.st.st_mode as u32) == "dir" && level < depth {
                match reopen_dir(dir, &e) {
                    Ok(child) => {
                        if self.list(&child, &p, level + 1, depth)? {
                            self.entries[at]["listing"] = json!(true);
                        }
                    }
                    Err(err) => self.error(&p, "list", errno_name(err)),
                }
            }
        }
        Ok(true)
    }
}

fn contents_of(path: &[u8], cap: u64) -> Value {
    let (fd, st) = match open_regular(path) {
        Ok(x) => x,
        Err(why) => return json!({"error": why}),
    };
    if st.st_size as u64 > cap {
        return json!({"error": "too-large", "size": st.st_size});
    }
    match fsx::read_to_end(fd.as_fd(), cap as usize) {
        Ok(data) => json!({
            "b64": crate::b64::encode(&data),
            "size": data.len(),
            "sha256": crate::ops::hex(&Sha256::digest(&data)),
        }),
        Err(e) if e.kind == "too-large" => json!({"error": "too-large"}),
        Err(e) => json!({"error": e.msg}),
    }
}

fn elf_of(path: &str) -> Value {
    let (fd, _) = match open_regular(path.as_bytes()) {
        Ok(x) => x,
        Err(why) => return json!({"error": why}),
    };
    let mut head = [0u8; 256];
    let n = rustix::io::pread(&fd, &mut head[..], 0).unwrap_or(0);
    let head = &head[..n];
    match crate::elf::facts(fd.as_fd(), head) {
        Some(f) if f.get("error").is_some() => f,
        Some(f) => json!({"elf": f}),
        None => {
            let mut m = Map::new();
            m.insert("elf".into(), Value::Null);
            if head.starts_with(b"#!") {
                let end = head.iter().position(|b| *b == b'\n').unwrap_or(head.len());
                m.insert("shebang".into(), json!(String::from_utf8_lossy(&head[..end])));
            }
            Value::Object(m)
        }
    }
}

fn mounts() -> Value {
    let fd = match fs::open("/proc/self/mountinfo", OFlags::RDONLY | OFlags::CLOEXEC, fs::Mode::empty()) {
        Ok(fd) => fd,
        Err(_) => return Value::Null,
    };
    let text = fsx::read_to_end(fd.as_fd(), 16 << 20).unwrap_or_default();
    let unescape = |s: &str| -> String {
        let b = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'\\' && i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
                out.push(u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 4]).unwrap(), 8).unwrap_or(b'?'));
                i += 4;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    let mut out = Vec::new();
    for line in String::from_utf8_lossy(&text).lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let Some(dash) = f.iter().position(|x| *x == "-") else { continue };
        if f.len() < dash + 4 || dash < 6 {
            continue;
        }
        out.push(json!({
            "id": f[0].parse::<u64>().ok(),
            "parent": f[1].parse::<u64>().ok(),
            "dev": f[2],
            "root": unescape(f[3]),
            "mount_point": unescape(f[4]),
            "options": f[5],
            "fstype": f[dash + 1],
            "source": unescape(f[dash + 2]),
            "super_options": f[dash + 3],
        }));
    }
    Value::Array(out)
}

fn read_proc(path: &str, limit: usize) -> std::result::Result<Vec<u8>, Errno> {
    let fd = fs::open(path, OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW, Mode::empty())?;
    fsx::read_to_end(fd.as_fd(), limit).map_err(|e| e.errno.map(Errno::from_raw_os_error).unwrap_or(Errno::IO))
}

fn id_line(status: &str, key: &str) -> Vec<u64> {
    status
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .map(|rest| rest.split_whitespace().filter_map(|n| n.parse().ok()).collect())
        .unwrap_or_default()
}

/// One process's descriptor facts (TR-PROTOCOL.md §3.5).
fn process(pid: u32) -> Option<Value> {
    let base = format!("/proc/{pid}");
    let status = match read_proc(&format!("{base}/status"), 1 << 16) {
        Ok(s) => String::from_utf8_lossy(&s).into_owned(),
        Err(Errno::NOENT) | Err(Errno::SRCH) => return None,
        Err(e) => return Some(json!({"pid": pid, "error": errno_name(e)})),
    };
    let fail = |op: &str, e: Errno| Some(json!({"pid": pid, "error": format!("{op}: {}", errno_name(e))}));
    let exe = match fs::readlink(format!("{base}/exe").as_str(), Vec::new()) {
        Ok(t) => t.into_bytes(),
        Err(Errno::NOENT) => Vec::new(), // a kernel thread or a zombie: no program
        Err(e) => return fail("exe", e),
    };
    let fd_dir = match fs::open(format!("{base}/fd").as_str(), OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty()) {
        Ok(fd) => fd,
        Err(Errno::NOENT) | Err(Errno::SRCH) => return None,
        Err(e) => return fail("fd", e),
    };
    let mut fds = Vec::new();
    let mut d = match fs::Dir::read_from(&fd_dir) {
        Ok(d) => d,
        Err(e) => return fail("fd", e),
    };
    let mut numbers: Vec<u32> = Vec::new();
    while let Some(ent) = d.read() {
        let Ok(ent) = ent else { return fail("fd", Errno::IO) };
        if let Ok(n) = std::str::from_utf8(ent.file_name().to_bytes()).unwrap_or("").parse::<u32>() {
            numbers.push(n);
        }
    }
    numbers.sort_unstable();
    for n in numbers {
        let target = match fs::readlinkat(&fd_dir, n.to_string().as_str(), Vec::new()) {
            Ok(t) => t.into_bytes(),
            Err(Errno::NOENT) => continue,
            Err(e) => return fail("fd", e),
        };
        let info = match read_proc(&format!("{base}/fdinfo/{n}"), 1 << 16) {
            Ok(i) => String::from_utf8_lossy(&i).into_owned(),
            Err(Errno::NOENT) => continue,
            Err(e) => return fail("fdinfo", e),
        };
        let field = |k: &str| info.lines().find_map(|l| l.strip_prefix(k)).map(|v| v.trim().to_string());
        let mut m = Map::new();
        m.insert("fd".into(), json!(n));
        fsx::put_path(&mut m, "target", &target);
        m.insert("flags".into(), json!(field("flags:")));
        m.insert("mnt_id".into(), json!(field("mnt_id:").and_then(|v| v.parse::<u64>().ok())));
        m.insert("ino".into(), json!(field("ino:").and_then(|v| v.parse::<u64>().ok())));
        fds.push(Value::Object(m));
    }
    let maps = match read_proc(&format!("{base}/maps"), 64 << 20) {
        Ok(text) => {
            let mut seen = BTreeMap::new();
            for line in text.split(|b| *b == b'\n') {
                // address perms offset dev inode path
                let mut rest = line;
                for _ in 0..5 {
                    let skip = rest.iter().position(|b| *b == b' ').map(|i| i + 1).unwrap_or(rest.len());
                    rest = &rest[skip..];
                    let pad = rest.iter().position(|b| *b != b' ').unwrap_or(rest.len());
                    rest = &rest[pad..];
                }
                if rest.first() == Some(&b'/') {
                    seen.insert(rest.to_vec(), ());
                }
            }
            seen.into_keys()
                .map(|p| match String::from_utf8(p) {
                    Ok(s) => json!(s),
                    Err(e) => json!({"path_b64": crate::b64::encode(e.as_bytes())}),
                })
                .collect::<Vec<_>>()
        }
        Err(Errno::NOENT) | Err(Errno::SRCH) => return None,
        Err(e) => return fail("maps", e),
    };
    let mut m = Map::new();
    m.insert("pid".into(), json!(pid));
    m.insert("uid".into(), json!(id_line(&status, "Uid:")));
    m.insert("gid".into(), json!(id_line(&status, "Gid:")));
    m.insert("groups".into(), json!(id_line(&status, "Groups:")));
    fsx::put_path(&mut m, "exe", &exe);
    m.insert("fds".into(), Value::Array(fds));
    m.insert("maps".into(), Value::Array(maps));
    Some(Value::Object(m))
}

fn processes() -> Value {
    let me = std::process::id();
    let mut pids: Vec<u32> = Vec::new();
    if let Ok(dir) = fs::open("/proc", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty()) {
        if let Ok(mut d) = fs::Dir::read_from(&dir) {
            while let Some(Ok(ent)) = d.read() {
                if let Ok(n) = std::str::from_utf8(ent.file_name().to_bytes()).unwrap_or("").parse::<u32>() {
                    if n != me {
                        pids.push(n);
                    }
                }
            }
        }
    }
    pids.sort_unstable();
    Value::Array(pids.into_iter().filter_map(process).collect())
}

fn sysctl_value(name: &str) -> Value {
    let ok = !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-')
        && !name.starts_with('.')
        && !name.contains("..");
    if !ok {
        return json!({"error": "bad-name"});
    }
    match read_proc(&format!("/proc/sys/{}", name.replace('.', "/")), 1 << 16) {
        Ok(v) => json!(String::from_utf8_lossy(&v).trim_end_matches('\n')),
        Err(e) => json!({"error": errno_name(e)}),
    }
}

fn interpreter_run(path: &str, flags: &[&str], uid: u32, gid: u32, timeout: Duration) -> Value {
    let started = Instant::now();
    let mut c = Command::new(path);
    c.args(flags).arg("-c").arg(PY_FACTS);
    c.env_clear();
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let data = PrepareData {
        cwd: None,
        umask: 0o077,
        rlimits: Vec::new(),
        setsid: true,
        caps: Caps::Launch,
        groups: Vec::new(),
        uid,
        gid,
        nnp: true,
    };
    // SAFETY: see exec::command.
    unsafe {
        c.pre_exec(move || {
            rustix::process::chdir("/").map_err(|e| std::io::Error::from_raw_os_error(e.raw_os_error()))?;
            crate::exec::prepare(&data)
        });
    }
    let r = spawn_capture(c, None, Some(timeout), 4 << 20);
    let ms = started.elapsed().as_millis() as u64;
    match r {
        Err(e) => json!({"flags": flags, "error": e.to_string(), "ms": ms}),
        Ok(out) => {
            let facts: Option<Value> = if out.status == 0 { serde_json::from_slice(&out.stdout).ok() } else { None };
            json!({
                "flags": flags,
                "status": out.status,
                "timed_out": out.timed_out,
                "facts": facts,
                "stderr": String::from_utf8_lossy(&out.stderr[..out.stderr.len().min(4000)]),
                "ms": ms,
            })
        }
    }
}

/// `uname -srvm`, from /proc/sys/kernel and this binary's architecture.
fn uname() -> String {
    let read = |name: &str| {
        read_proc(&format!("/proc/sys/kernel/{name}"), 4096)
            .map(|v| String::from_utf8_lossy(&v).trim_end_matches('\n').to_string())
            .unwrap_or_default()
    };
    format!("{} {} {} {}", read("ostype"), read("osrelease"), read("version"), std::env::consts::ARCH)
}

pub fn scan(v: &Value) -> Result<Value> {
    let a = Args::new(
        v,
        &[
            "paths", "paths_b64", "walks", "contents", "contents_b64", "elf", "interpreters", "processes", "sysctl", "mounts", "acl",
            "content_max_bytes", "max_entries", "max_bytes", "timeout_ms", "as", "resolver",
        ],
    )?;
    let started = Instant::now();
    let (uid, gid) = match a.opt("as") {
        None => (65534, 65534),
        Some(x) => {
            let xa = Args::new(x, &["uid", "gid"])?;
            (xa.u32("uid")?, xa.u32("gid")?)
        }
    };
    if uid == 0 {
        return Err(Error::bad("interpreters are never run as root"));
    }
    let content_cap = a.opt_u64("content_max_bytes")?.unwrap_or(4 << 20);
    let max_bytes = a.opt_u64("max_bytes")?.unwrap_or(128 << 20);
    let timeout = Duration::from_millis(a.opt_u64("timeout_ms")?.unwrap_or(10_000));
    let mut walks = Vec::new();
    for w in a.opt_arr("walks")?.map(|v| v.as_slice()).unwrap_or(&[]) {
        let wa = Args::new(w, &["root", "root_b64", "depth"])?;
        walks.push((wa.path("root")?, wa.u64("depth")?));
    }
    let mut s = Scan {
        acl: a.bool("acl", true)?,
        max_entries: a.opt_u64("max_entries")?.unwrap_or(2_000_000),
        entries: Vec::new(),
        seen: HashMap::new(),
        errors: Vec::new(),
    };
    for p in a.path_list("paths")? {
        s.prefixes(&p)?;
    }
    for (root, depth) in &walks {
        s.walk(root, *depth)?;
    }
    let mut contents = Map::new();
    for p in a.str_list("contents")? {
        let c = contents_of(p.as_bytes(), content_cap);
        if let Some(why) = c.get("error").and_then(Value::as_str) {
            s.error(p.as_bytes(), "contents", why.to_string());
        }
        contents.insert(p.to_string(), c);
    }
    // A path given as bytes is answered under the base64 text it was given as.
    let mut contents_b64 = Map::new();
    for b in a.str_list("contents_b64")? {
        let p = crate::b64::decode(b).ok_or_else(|| Error::bad("contents_b64 must hold base64"))?;
        let c = contents_of(&p, content_cap);
        if let Some(why) = c.get("error").and_then(Value::as_str) {
            s.error(&p, "contents", why.to_string());
        }
        contents_b64.insert(b.to_string(), c);
    }
    let mut elf = Map::new();
    for p in a.str_list("elf")? {
        let f = elf_of(p);
        if let Some(why) = f.get("error").and_then(Value::as_str) {
            s.error(p.as_bytes(), "elf", why.to_string());
        }
        elf.insert(p.to_string(), f);
    }
    // Before any interpreter runs, so no child of the scanner is listed.
    let procs = if a.bool("processes", false)? { processes() } else { Value::Array(Vec::new()) };
    let mut sysctl = Map::new();
    for name in a.str_list("sysctl")? {
        sysctl.insert(name.to_string(), sysctl_value(name));
    }
    let mount_list = if a.bool("mounts", true)? { mounts() } else { Value::Array(Vec::new()) };
    let interps: Vec<Value> = a
        .str_list("interpreters")?
        .into_iter()
        .map(|p| {
            if !p.starts_with('/') {
                json!({"path": p, "error": "not an absolute path"})
            } else {
                let runs: Vec<Value> = INTERPRETER_RUNS.iter().map(|f| interpreter_run(p, f, uid, gid, timeout)).collect();
                json!({"path": p, "runs": runs})
            }
        })
        .collect();
    let answer = json!({
        "protocol": crate::serve::PROTOCOL,
        "bfx": env!("CARGO_PKG_VERSION"),
        "arch": std::env::consts::ARCH,
        "uname": uname(),
        "resolver": fsx::resolver().name(),
        "entries": s.entries,
        "contents": contents,
        "contents_b64": contents_b64,
        "elf": elf,
        "interpreters": interps,
        "processes": procs,
        "sysctl": sysctl,
        "mounts": mount_list,
        "errors": s.errors,
        "ms": started.elapsed().as_millis() as u64,
    });
    let size = serde_json::to_vec(&answer).map(|v| v.len() as u64).unwrap_or(u64::MAX);
    if size > max_bytes {
        return Err(Error::new("too-large", format!("the answer is {size} bytes, more than max_bytes {max_bytes}")));
    }
    Ok(answer)
}
