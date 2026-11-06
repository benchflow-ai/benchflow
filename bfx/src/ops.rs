//! The file primitives: mkdir, chown, chmod, clear, remove, write, walk and
//! stat. Each takes a `root` (absolute) and a `path` relative to it, and
//! acts only through descriptors opened by `fsx` (no link followed, no mount
//! crossed). Recursive operations visit mount points as entries and never
//! descend into them; links are acted on as links (chown) or skipped
//! (chmod), never followed. `root` and `path` may instead be given as raw
//! bytes, base64, under `root_b64` and `path_b64`: a name need not be UTF-8.

use crate::args::{mode_value, Args};
use crate::err::{Error, Result};
use crate::fsx::{self, child_path, entry, ftype, names, open_dir, open_one, open_parent, os, reopen_dir, show, split_rel, Entry, Node};
use crate::glob;
use rustix::fd::AsFd;
use rustix::fs::{self, AtFlags, Gid, Mode, OFlags, RenameFlags, Uid};
use rustix::io::Errno;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const MAX_DEPTH: usize = 512;
/// How deep one walk goes before it hands a folder back (`deeper`): it holds
/// one descriptor per level, well within the usual 1024.
const WALK_DEPTH: usize = 256;

/// A regular file with another name somewhere: never changed in place.
fn shared(e: &Entry) -> bool {
    ftype(e.st.st_mode as u32) == "file" && e.st.st_nlink > 1
}

/// `root` and `path`, each as text or as base64 bytes (`root_b64`,
/// `path_b64`): a name need not be UTF-8.
fn root_and_path(a: &Args) -> Result<(Node, Vec<u8>)> {
    let root = fsx::open_root_bytes(&a.path("root")?)?;
    Ok((root, a.opt_path("path")?.unwrap_or_default()))
}

const PATH_ARGS: [&str; 4] = ["root", "root_b64", "path", "path_b64"];

fn allowed<'k>(more: &[&'k str]) -> Vec<&'k str> {
    PATH_ARGS.iter().copied().chain(more.iter().copied()).collect()
}

fn uid(n: Option<u32>) -> Option<Uid> {
    n.map(Uid::from_raw_unchecked)
}

fn gid(n: Option<u32>) -> Option<Gid> {
    n.map(Gid::from_raw_unchecked)
}

/// `mkdir {root, path, mode=0o755, uid?, gid?, parents=false, exist_ok=false}`
pub fn mkdir(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["mode", "uid", "gid", "parents", "exist_ok", "resolver"]))?;
    let (root, path) = root_and_path(&a)?;
    let comps = split_rel(&path)?;
    if comps.is_empty() {
        return Err(Error::bad("mkdir needs a path beneath its root"));
    }
    let mode = a.opt("mode").map(|m| mode_value(m, "mode")).transpose()?.unwrap_or(0o755);
    let (u, g) = (a.opt_u32("uid")?, a.opt_u32("gid")?);
    let parents = a.bool("parents", false)?;
    let exist_ok = a.bool("exist_ok", false)? || parents;
    let mut created: Vec<String> = Vec::new();
    let r = mkdir_steps(&root, &comps, mode, u, g, parents, exist_ok, &mut created);
    match r {
        Ok(()) => Ok(json!({"created": created})),
        Err(e) => Err(e.with_detail(json!({"created": created}))),
    }
}

#[allow(clippy::too_many_arguments)]
fn mkdir_steps(root: &Node, comps: &[&[u8]], mode: u32, u: Option<u32>, g: Option<u32>, parents: bool, exist_ok: bool, created: &mut Vec<String>) -> Result<()> {
    let dir = OFlags::RDONLY | OFlags::DIRECTORY;
    let mut cur: Option<Node> = None;
    for (i, c) in comps.iter().enumerate() {
        let last = i + 1 == comps.len();
        let parent = cur.as_ref().unwrap_or(root);
        let at = comps[..=i].iter().map(|c| show(c)).collect::<Vec<_>>().join("/");
        let fd = match open_one(parent, c, dir) {
            Ok(fd) => {
                if last && !exist_ok {
                    return Err(Error::sys(Errno::EXIST, &at));
                }
                fd
            }
            Err(Errno::NOENT) if last || parents => {
                match fs::mkdirat(&parent.fd, os(c), Mode::from_raw_mode(0o700)) {
                    Ok(()) | Err(Errno::EXIST) => {}
                    Err(e) => return Err(Error::sys(e, &at)),
                }
                let fd = open_one(parent, c, dir).map_err(|e| Error::sys(e, &at))?;
                if u.is_some() || g.is_some() {
                    fs::fchown(&fd, uid(u), gid(g)).map_err(|e| Error::sys(e, &at))?;
                }
                fs::fchmod(&fd, Mode::from_raw_mode(mode)).map_err(|e| Error::sys(e, &at))?;
                created.push(at.clone());
                fd
            }
            Err(e) => return Err(Error::sys(e, &at)),
        };
        cur = Some(Node { fd, dev: root.dev, mnt: root.mnt });
    }
    Ok(())
}

/// Visit *e* (named *rel*) and, when it is a directory and *recursive*,
/// everything beneath it: `f(rel, parent, name, entry)` before descending.
fn visit(
    parent: &Node,
    name: &[u8],
    rel: &[u8],
    recursive: bool,
    depth: usize,
    f: &mut dyn FnMut(&[u8], &Node, &[u8], &Entry) -> Result<()>,
) -> Result<()> {
    let e = entry(parent, name).map_err(|e| Error::sys(e, show(rel)))?;
    f(rel, parent, name, &e)?;
    if recursive && !e.mount && ftype(e.st.st_mode as u32) == "dir" {
        if depth >= MAX_DEPTH {
            return Err(Error::new("too-deep", format!("{} is deeper than {MAX_DEPTH} levels", show(rel))));
        }
        let node = reopen_dir(parent, &e).map_err(|err| Error::sys(err, show(rel)))?;
        for n in names(&node)? {
            visit(&node, &n, &child_path(rel, &n), true, depth + 1, f)?;
        }
    }
    Ok(())
}

/// The target of a chown/chmod: the root itself, or a name in its parent.
fn for_target(
    root: &Node,
    path: &[u8],
    recursive: bool,
    f: &mut dyn FnMut(&[u8], &Node, &[u8], &Entry) -> Result<()>,
) -> Result<()> {
    let comps = split_rel(path)?;
    if comps.is_empty() {
        // The root: act on it, then on its children.
        let st = root.stat()?;
        let fd = fs::openat(&root.fd, ".", OFlags::PATH | OFlags::CLOEXEC, Mode::empty()).map_err(|e| Error::sys(e, "."))?;
        let e = Entry { st, mount: false, fd: Some(fd) };
        let here = open_dir(root, &[])?;
        f(b"", &here, b".", &e)?;
        if recursive {
            for n in names(&here)? {
                visit(&here, &n, &n, true, 1, f)?;
            }
        }
        return Ok(());
    }
    let (parent, name) = open_parent(root, &comps)?;
    visit(&parent, name, path, recursive, 0, f)
}

/// `chown {root, path, uid?, gid?, recursive=false}`: links are changed
/// themselves (never their targets); mount points are skipped and listed.
pub fn chown(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["uid", "gid", "recursive", "resolver"]))?;
    let (root, path) = root_and_path(&a)?;
    let (u, g) = (a.opt_u32("uid")?, a.opt_u32("gid")?);
    let recursive = a.bool("recursive", false)?;
    let mut changed = 0u64;
    let (mut mounts, mut hardlinks) = (Vec::new(), Vec::new());
    let r = for_target(&root, &path, recursive, &mut |rel, _parent, _name, e| {
        match &e.fd {
            None => mounts.push(show(rel)),
            Some(_) if shared(e) => hardlinks.push(show(rel)),
            Some(fd) => {
                fs::chownat(fd, "", uid(u), gid(g), AtFlags::EMPTY_PATH).map_err(|err| Error::sys(err, show(rel)))?;
                changed += 1;
            }
        }
        Ok(())
    });
    let done = json!({"changed": changed, "skipped_mounts": mounts, "skipped_hardlinks": hardlinks});
    r.map(|_| done.clone()).map_err(|e| e.with_detail(done))
}

/// `chmod {root, path, mode? | clear?/set?, recursive=false}`: `mode` sets
/// the bits; `clear` and `set` change some (`{"clear": "022"}` is `go-w`).
/// Links have no mode and are skipped; mount points are skipped and listed.
pub fn chmod(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["mode", "clear", "set", "recursive", "resolver"]))?;
    let (root, path) = root_and_path(&a)?;
    let mode = a.opt("mode").map(|m| mode_value(m, "mode")).transpose()?;
    let clear = a.opt("clear").map(|m| mode_value(m, "clear")).transpose()?.unwrap_or(0);
    let set = a.opt("set").map(|m| mode_value(m, "set")).transpose()?.unwrap_or(0);
    if mode.is_some() == (clear != 0 || set != 0) {
        return Err(Error::bad("give either mode, or clear and/or set"));
    }
    let recursive = a.bool("recursive", false)?;
    let (mut changed, mut links) = (0u64, 0u64);
    let (mut mounts, mut hardlinks) = (Vec::new(), Vec::new());
    let r = for_target(&root, &path, recursive, &mut |rel, _parent, _name, e| {
        let old = e.st.st_mode as u32;
        match &e.fd {
            None => mounts.push(show(rel)),
            Some(_) if ftype(old) == "link" => links += 1,
            Some(_) if shared(e) => hardlinks.push(show(rel)),
            Some(fd) => {
                let new = mode.unwrap_or((old & 0o7777 & !clear) | set);
                if new != old & 0o7777 {
                    fsx::chmod_fd(fd.as_fd(), new).map_err(|err| Error { msg: format!("{}: {}", show(rel), err.msg), ..err })?;
                    changed += 1;
                }
            }
        }
        Ok(())
    });
    let done = json!({"changed": changed, "skipped_links": links, "skipped_mounts": mounts, "skipped_hardlinks": hardlinks});
    r.map(|_| done.clone()).map_err(|e| e.with_detail(done))
}

/// Remove everything in *dir*; mount points (and anything that cannot be
/// removed) are reported and left. Returns how many entries went.
fn empty_dir(dir: &Node, rel: &[u8], depth: usize, left: &mut Vec<Value>) -> Result<u64> {
    if depth >= MAX_DEPTH {
        return Err(Error::new("too-deep", format!("{} is deeper than {MAX_DEPTH} levels", show(rel))));
    }
    let mut removed = 0;
    for n in names(dir)? {
        let p = child_path(rel, &n);
        let e = match entry(dir, &n) {
            Ok(e) => e,
            Err(Errno::NOENT) => continue,
            Err(err) => {
                left.push(json!({"path": show(&p), "error": Error::sys(err, show(&p)).to_json()}));
                continue;
            }
        };
        if e.mount {
            left.push(json!({"path": show(&p), "error": Error::sys(Errno::XDEV, show(&p)).to_json()}));
            continue;
        }
        let is_dir = ftype(e.st.st_mode as u32) == "dir";
        if is_dir {
            match reopen_dir(dir, &e) {
                Ok(child) => removed += empty_dir(&child, &p, depth + 1, left)?,
                Err(err) => {
                    left.push(json!({"path": show(&p), "error": Error::sys(err, show(&p)).to_json()}));
                    continue;
                }
            }
        }
        let flags = if is_dir { AtFlags::REMOVEDIR } else { AtFlags::empty() };
        match fs::unlinkat(&dir.fd, os(&n), flags) {
            Ok(()) => removed += 1,
            Err(Errno::NOENT) => {}
            Err(err) => left.push(json!({"path": show(&p), "error": Error::sys(err, show(&p)).to_json()})),
        }
    }
    Ok(removed)
}

/// `clear {root, path}`: empty a directory, keeping it.
pub fn clear(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["resolver"]))?;
    let (root, path) = root_and_path(&a)?;
    let dir = open_dir(&root, &split_rel(&path)?)?;
    let mut left = Vec::new();
    let removed = empty_dir(&dir, &path, 0, &mut left)?;
    if left.is_empty() {
        Ok(json!({"removed": removed}))
    } else {
        Err(Error::new("partial", format!("{} entries could not be removed", left.len()))
            .with_detail(json!({"removed": removed, "left": left})))
    }
}

/// `remove {root, path, expect?: {dev, ino}, recursive=false, missing_ok=false}`.
/// With `expect`, the entry must still be the one an earlier scan saw. The
/// expectation is checked on the entry's own descriptor and the name is then
/// unlinked, so it holds once no other process can rename entries in that
/// folder (after K1's kill). Against a live writer, a swap of the name
/// between the check and the unlink removes the entry swapped in.
pub fn remove(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["expect", "recursive", "missing_ok", "resolver"]))?;
    let (root, raw) = root_and_path(&a)?;
    let comps = split_rel(&raw)?;
    let path = show(&raw);
    let path = path.as_str();
    let recursive = a.bool("recursive", false)?;
    let missing_ok = a.bool("missing_ok", false)?;
    let expect = match a.opt("expect") {
        None => None,
        Some(x) => {
            let ea = Args::new(x, &["dev", "ino"])?;
            Some((ea.u64("dev")?, ea.u64("ino")?))
        }
    };
    let (parent, name) = match open_parent(&root, &comps) {
        Err(e) if missing_ok && e.is(Errno::NOENT) => return Ok(json!({"removed": 0})),
        r => r?,
    };
    let e = match entry(&parent, name) {
        Err(Errno::NOENT) if missing_ok => return Ok(json!({"removed": 0})),
        r => r.map_err(|e| Error::sys(e, path))?,
    };
    if let Some((dev, ino)) = expect {
        if (e.st.st_dev as u64, e.st.st_ino as u64) != (dev, ino) {
            return Err(Error::new("changed", format!("{path} is no longer the entry that was scanned"))
                .with_detail(json!({"dev": e.st.st_dev as u64, "ino": e.st.st_ino as u64})));
        }
    }
    if e.mount {
        return Err(Error::sys(Errno::XDEV, path));
    }
    let mut removed = 0;
    let is_dir = ftype(e.st.st_mode as u32) == "dir";
    if is_dir {
        if !recursive {
            return Err(Error::sys(Errno::ISDIR, path));
        }
        let dir = reopen_dir(&parent, &e).map_err(|err| Error::sys(err, path))?;
        let mut left = Vec::new();
        removed += empty_dir(&dir, &raw, 0, &mut left)?;
        if !left.is_empty() {
            return Err(Error::new("partial", format!("{} entries could not be removed", left.len()))
                .with_detail(json!({"removed": removed, "left": left})));
        }
    }
    let flags = if is_dir { AtFlags::REMOVEDIR } else { AtFlags::empty() };
    fs::unlinkat(&parent.fd, os(name), flags).map_err(|err| Error::sys(err, path))?;
    Ok(json!({"removed": removed + 1}))
}

static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A new private file in *dir* (`O_EXCL`, so never through a link): its
/// name and descriptor.
pub fn temp_file(dir: &Node) -> Result<(Vec<u8>, rustix::fd::OwnedFd)> {
    let pid = rustix::process::getpid().as_raw_nonzero().get();
    for _ in 0..1000 {
        let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        let name = format!(".bfx-{pid}-{seq}-{nanos:x}").into_bytes();
        match fs::openat(
            &dir.fd,
            os(&name),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(fd) => return Ok((name, fd)),
            Err(Errno::EXIST) => continue,
            Err(e) => return Err(Error::sys(e, "creating a temporary file")),
        }
    }
    Err(Error::new("os", "no free temporary name"))
}

pub fn write_all(fd: rustix::fd::BorrowedFd, mut data: &[u8]) -> Result<()> {
    while !data.is_empty() {
        match rustix::io::write(fd, data) {
            Ok(0) => return Err(Error::new("os", "short write")),
            Ok(n) => data = &data[n..],
            Err(Errno::INTR) => continue,
            Err(e) => return Err(Error::sys(e, "write")),
        }
    }
    Ok(())
}

/// `write {root, path, data | data_b64, mode=0o644, uid?, gid?, replace=true, sync=false}`:
/// the bytes go to a new private file in the same directory, which then
/// replaces the name (a link there is replaced, never written through).
pub fn write(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["data", "data_b64", "mode", "uid", "gid", "replace", "sync", "resolver"]))?;
    let (root, raw) = root_and_path(&a)?;
    let comps = split_rel(&raw)?;
    let path = show(&raw);
    let path = path.as_str();
    let data: Vec<u8> = match (a.opt_str("data")?, a.opt_str("data_b64")?) {
        (Some(s), None) => s.as_bytes().to_vec(),
        (None, Some(b)) => crate::b64::decode(b).ok_or_else(|| Error::bad("data_b64 is not base64"))?,
        _ => return Err(Error::bad("give exactly one of data and data_b64")),
    };
    let mode = a.opt("mode").map(|m| mode_value(m, "mode")).transpose()?.unwrap_or(0o644);
    let (u, g) = (a.opt_u32("uid")?, a.opt_u32("gid")?);
    let (parent, name) = open_parent(&root, &comps)?;
    let (tmp, fd) = temp_file(&parent)?;
    let done = (|| -> Result<()> {
        write_all(fd.as_fd(), &data)?;
        if u.is_some() || g.is_some() {
            fs::fchown(&fd, uid(u), gid(g)).map_err(|e| Error::sys(e, "fchown"))?;
        }
        fs::fchmod(&fd, Mode::from_raw_mode(mode)).map_err(|e| Error::sys(e, "fchmod"))?;
        if a.bool("sync", false)? {
            fs::fsync(&fd).map_err(|e| Error::sys(e, "fsync"))?;
        }
        let flags = if a.bool("replace", true)? { RenameFlags::empty() } else { RenameFlags::NOREPLACE };
        fs::renameat_with(&parent.fd, os(&tmp), &parent.fd, os(name), flags).map_err(|e| Error::sys(e, path))
    })();
    if let Err(e) = done {
        let _ = fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty());
        return Err(e);
    }
    let st = fs::fstat(&fd).map_err(|e| Error::sys(e, "fstat"))?;
    Ok(json!({"size": data.len(), "dev": st.st_dev as u64, "ino": st.st_ino as u64}))
}

/// The SHA-256 of a regular file entry, read through its held descriptor
/// (fsx::reopen_file) and checked to be the same (dev, ino); None when
/// larger than *max*.
pub fn hash_entry(parent: &Node, name: &[u8], e: &Entry, max: u64) -> Result<Option<String>> {
    if e.st.st_size as u64 > max {
        return Ok(None);
    }
    let fd = fsx::reopen_file(parent, name, e).map_err(|err| Error::sys(err, show(name)))?;
    let st = fs::fstat(&fd).map_err(|err| Error::sys(err, "fstat"))?;
    if (st.st_dev, st.st_ino) != (e.st.st_dev, e.st.st_ino) || ftype(st.st_mode as u32) != "file" {
        return Err(Error::new("changed", format!("{} changed while it was read", show(name))));
    }
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        let n = match rustix::io::read(&fd, &mut buf[..]) {
            Ok(n) => n,
            Err(Errno::INTR) => continue,
            Err(err) => return Err(Error::sys(err, "read")),
        };
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > max {
            return Ok(None);
        }
        h.update(&buf[..n]);
    }
    Ok(Some(hex(&h.finalize())))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn entry_json(rel: &[u8], parent: &Node, name: &[u8], e: &Entry) -> Map<String, Value> {
    let mut m = fsx::stat_json(&e.st);
    fsx::put_path(&mut m, "path", rel);
    if e.mount {
        m.insert("mount".into(), json!(true));
    }
    if ftype(e.st.st_mode as u32) == "link" {
        if let Some(t) = fsx::readlink_at(parent, name) {
            fsx::put_path(&mut m, "link", &t);
        }
    }
    m
}

/// `walk {root, path="", patterns?, hash_max_bytes?, max_entries=1000000}`:
/// every entry beneath, in sorted depth-first order, with its lstat facts.
/// Patterns (`*`, `?`, `**`) select which entries are listed; the walk still
/// goes everywhere. Mount points are listed (`"mount": true`), not entered.
///
/// The walk is complete or it fails, with two exceptions it reports:
/// - `vanished`: an entry that was listed but is gone when it is looked at
///   (or a folder removed while it is read) is skipped, as `find` skips it,
///   and the rest of its folder is still walked;
/// - `deeper`: a folder `WALK_DEPTH` levels beneath the start is not
///   entered; its facts are listed there, and the caller walks it as a root
///   of its own. So no tree is too deep to walk, and no walk holds more than
///   one descriptor per level.
///
/// Any other failure beneath the start fails the walk.
pub fn walk(v: &Value) -> Result<Value> {
    let a = Args::new(v, &allowed(&["patterns", "hash_max_bytes", "max_entries", "resolver"]))?;
    let (root, path) = root_and_path(&a)?;
    let patterns = a.str_list("patterns")?;
    let hash_max = a.opt_u64("hash_max_bytes")?;
    let max = a.opt_u64("max_entries")?.unwrap_or(1_000_000);
    let comps = split_rel(&path)?;
    let start = open_dir(&root, &comps)?;
    let mut w = Walk { patterns, hash_max, max, seen: 0, out: Vec::new(), vanished: Vec::new(), deeper: Vec::new() };
    for n in names(&start)? {
        let rel = child_path(&path, &n);
        w.visit(&start, &n, &rel, 1)?;
    }
    Ok(json!({"entries": w.out, "vanished": w.vanished, "deeper": w.deeper}))
}

struct Walk<'p> {
    patterns: Vec<&'p str>,
    hash_max: Option<u64>,
    max: u64,
    seen: u64,
    out: Vec<Value>,
    vanished: Vec<Value>,
    deeper: Vec<Value>,
}

impl Walk<'_> {
    fn gone(&mut self, rel: &[u8]) {
        let mut m = Map::new();
        fsx::put_path(&mut m, "path", rel);
        self.vanished.push(Value::Object(m));
    }

    fn visit(&mut self, parent: &Node, name: &[u8], rel: &[u8], depth: usize) -> Result<()> {
        let e = match entry(parent, name) {
            Ok(e) => e,
            Err(Errno::NOENT) => {
                self.gone(rel);
                return Ok(());
            }
            Err(err) => return Err(Error::sys(err, show(rel))),
        };
        self.seen += 1;
        if self.seen > self.max {
            return Err(Error::new("too-many", format!("more than {} entries", self.max)));
        }
        let is_dir = !e.mount && ftype(e.st.st_mode as u32) == "dir";
        if self.patterns.is_empty() || self.patterns.iter().any(|p| glob::matches(p.as_bytes(), rel)) {
            let mut m = entry_json(rel, parent, name, &e);
            if let (Some(hm), "file", false) = (self.hash_max, ftype(e.st.st_mode as u32), e.mount) {
                match hash_entry(parent, name, &e, hm) {
                    Ok(Some(h)) => {
                        m.insert("sha256".into(), json!(h));
                    }
                    Ok(None) => {
                        m.insert("sha256_skipped".into(), json!("size"));
                    }
                    Err(err) if err.is(Errno::NOENT) || err.kind == "changed" => {
                        self.gone(rel);
                        return Ok(());
                    }
                    Err(err) => return Err(err),
                }
            }
            self.out.push(Value::Object(m));
        }
        if !is_dir {
            return Ok(());
        }
        if depth >= WALK_DEPTH {
            self.deeper.push(Value::Object(entry_json(rel, parent, name, &e)));
            return Ok(());
        }
        let node = match reopen_dir(parent, &e) {
            Ok(n) => n,
            Err(Errno::NOENT) => {
                self.gone(rel);
                return Ok(());
            }
            Err(err) => return Err(Error::sys(err, show(rel))),
        };
        drop(e);
        // A folder removed while it is read lists as ENOENT (a dead
        // directory): it is gone, like an entry.
        let kids = match names(&node) {
            Ok(k) => k,
            Err(err) if err.is(Errno::NOENT) => {
                self.gone(rel);
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        for n in kids {
            self.visit(&node, &n, &child_path(rel, &n), depth + 1)?;
        }
        Ok(())
    }
}

/// `stat {root, paths: [..]}`: lstat facts of each path, or its error.
pub fn stat(v: &Value) -> Result<Value> {
    let a = Args::new(v, &["root", "paths", "resolver"])?;
    let root = fsx::open_root(a.str("root")?)?;
    let mut out = Vec::new();
    for p in a.str_list("paths")? {
        let r = (|| -> Result<Value> {
            let comps = split_rel(p.as_bytes())?;
            if comps.is_empty() {
                let mut m = fsx::stat_json(&root.stat()?);
                m.insert("path".into(), json!(""));
                return Ok(Value::Object(m));
            }
            let (parent, name) = open_parent(&root, &comps)?;
            let e = entry(&parent, name).map_err(|e| Error::sys(e, p))?;
            Ok(Value::Object(entry_json(p.as_bytes(), &parent, name, &e)))
        })();
        out.push(match r {
            Ok(v) => v,
            Err(e) => json!({"path": p, "error": e.to_json()}),
        });
    }
    Ok(json!({"entries": out}))
}

/// `bfx _populate DIR` (the volume fill): copy this binary to `DIR/bfx`,
/// 0555, through a new private file that is fsync'd and renamed into place,
/// so a reader never sees a partial file.
pub fn populate(dir: &str) -> Result<Value> {
    let root = fsx::open_root(dir)?;
    let exe = fs::open("/proc/self/exe", OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty()).map_err(|e| Error::sys(e, "/proc/self/exe"))?;
    let data = fsx::read_to_end(exe.as_fd(), 256 << 20)?;
    let (tmp, fd) = temp_file(&root)?;
    let done = (|| -> Result<()> {
        write_all(fd.as_fd(), &data)?;
        fs::fchmod(&fd, Mode::from_raw_mode(0o555)).map_err(|e| Error::sys(e, "fchmod"))?;
        fs::fsync(&fd).map_err(|e| Error::sys(e, "fsync"))?;
        fs::renameat_with(&root.fd, os(&tmp), &root.fd, "bfx", RenameFlags::empty()).map_err(|e| Error::sys(e, "rename"))?;
        fs::fsync(&root.fd).map_err(|e| Error::sys(e, "fsync"))
    })();
    if let Err(e) = done {
        let _ = fs::unlinkat(&root.fd, os(&tmp), AtFlags::empty());
        return Err(e);
    }
    Ok(json!({"sha256": hex(&Sha256::digest(&data))}))
}
