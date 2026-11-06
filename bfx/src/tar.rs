//! tar-out and tar-in: POSIX ustar with pax extended headers, beneath a held
//! root.
//!
//! tar-out records links as links (never their targets), skips mount points
//! (listed) and, by default, special files (listed); a regular file is read
//! through the walk's own held descriptor (fsx::reopen_file), and must still
//! be the entry the walk saw.
//!
//! tar-in treats the archive as data. A member name must be relative with
//! no `..` component (a leading `./` is dropped); every directory on the way
//! is opened beneath the root without following a link or crossing a mount,
//! so a link the archive made earlier is never followed by a later member.
//! A file is written to a new private file in its directory and renamed over
//! the name. Symbolic links are created as links. A hard link may only name
//! a regular file this same extraction created. Device and FIFO members are
//! refused unless `specials` is "skip". Ownership is the request's (or
//! root's), never the archive's; modes are masked (by default to 0o777, so
//! no set-id or sticky bit comes from the archive), and a mask that keeps a
//! set-id bit is refused, so no set-id bit ever comes from an archive.
//! `refuse_root_owner` (the host sets it for agent-derived bytes) refuses a
//! request that does not give every entry a non-root uid and gid. A
//! directory member that names a directory which was there before the
//! extraction leaves it as it was (owner, mode, times) and is listed in the
//! answer's `kept`, so the host can add it to the agent's write set if it
//! must be the agent's.
//!
//! tar-out writes a hard-linked file as separate regular members, one per
//! name (link identity is not kept, and the bytes are stored once per name):
//! a restore never makes two names share an inode the archive chose. Its
//! `max_bytes` failure is `too-large`, as tar-in's.

use crate::args::{mode_value, Args};
use crate::err::{Error, Result};
use crate::fsx::{self, child_path, entry, ftype, names, open_one, os, reopen_dir, show, split_rel, Node};
use crate::ops::{temp_file, write_all};
use rustix::fd::AsFd;
use rustix::event::Timespec;
use rustix::fs::{self, AtFlags, Gid, Mode, OFlags, RenameFlags, Timestamps, Uid};
use rustix::io::Errno;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{Read, Write};

const BLOCK: usize = 512;
const META_LIMIT: u64 = 1 << 20;

// ---------------------------------------------------------------- writing

fn octal(field: &mut [u8], v: u64) -> bool {
    let digits = field.len() - 1;
    let s = format!("{v:o}");
    if s.len() > digits {
        return false;
    }
    let pad = digits - s.len();
    for b in field[..pad].iter_mut() {
        *b = b'0';
    }
    field[pad..digits].copy_from_slice(s.as_bytes());
    field[digits] = 0;
    true
}

fn pax_record(key: &str, value: &[u8]) -> Vec<u8> {
    let body = key.len() + value.len() + 3; // " " "=" "\n"
    let mut len = body + 1;
    while format!("{len}").len() + body != len {
        len = format!("{len}").len() + body;
    }
    let mut r = format!("{len} {key}=").into_bytes();
    r.extend_from_slice(value);
    r.push(b'\n');
    r
}

pub struct Writer<W: Write> {
    w: W,
    pub bytes: u64,
}

impl<W: Write> Writer<W> {
    pub fn new(w: W) -> Self {
        Writer { w, bytes: 0 }
    }

    pub fn put(&mut self, b: &[u8]) -> Result<()> {
        self.w.write_all(b).map_err(|e| Error::io(e, "writing the archive"))?;
        self.bytes += b.len() as u64;
        Ok(())
    }

    pub fn header(&mut self, name: &[u8], kind: u8, mode: u32, uid: u64, gid: u64, size: u64, mtime: i64, link: &[u8]) -> Result<()> {
        let mut pax = Vec::new();
        let mut h = [0u8; BLOCK];
        if name.len() <= 100 && name.is_ascii() {
            h[..name.len()].copy_from_slice(name);
        } else {
            pax.extend(pax_record("path", name));
            let short = &name[..name.len().min(100)];
            h[..short.len()].copy_from_slice(short);
        }
        octal(&mut h[100..108], (mode & 0o7777) as u64);
        if !octal(&mut h[108..116], uid) {
            pax.extend(pax_record("uid", uid.to_string().as_bytes()));
            octal(&mut h[108..116], 0);
        }
        if !octal(&mut h[116..124], gid) {
            pax.extend(pax_record("gid", gid.to_string().as_bytes()));
            octal(&mut h[116..124], 0);
        }
        if !octal(&mut h[124..136], size) {
            pax.extend(pax_record("size", size.to_string().as_bytes()));
            octal(&mut h[124..136], 0);
        }
        let mt = mtime.max(0) as u64;
        if !octal(&mut h[136..148], mt) {
            pax.extend(pax_record("mtime", mt.to_string().as_bytes()));
            octal(&mut h[136..148], 0);
        }
        h[156] = kind;
        if link.len() <= 100 && link.is_ascii() {
            h[157..157 + link.len()].copy_from_slice(link);
        } else {
            pax.extend(pax_record("linkpath", link));
            let short = &link[..link.len().min(100)];
            h[157..157 + short.len()].copy_from_slice(short);
        }
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        if !pax.is_empty() {
            let mut p = [0u8; BLOCK];
            p[..14].copy_from_slice(b"././@PaxHeader");
            octal(&mut p[100..108], 0o644);
            octal(&mut p[108..116], 0);
            octal(&mut p[116..124], 0);
            octal(&mut p[124..136], pax.len() as u64);
            octal(&mut p[136..148], 0);
            p[156] = b'x';
            p[257..263].copy_from_slice(b"ustar\0");
            p[263..265].copy_from_slice(b"00");
            checksum(&mut p);
            self.put(&p)?;
            self.put(&pax)?;
            self.pad(pax.len() as u64)?;
        }
        checksum(&mut h);
        self.put(&h)
    }

    pub fn pad(&mut self, size: u64) -> Result<()> {
        let rem = (size % BLOCK as u64) as usize;
        if rem != 0 {
            self.put(&[0u8; BLOCK][..BLOCK - rem])?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<W> {
        self.put(&[0u8; BLOCK * 2])?;
        self.w.flush().map_err(|e| Error::io(e, "writing the archive"))?;
        Ok(self.w)
    }
}

fn checksum(h: &mut [u8; BLOCK]) {
    h[148..156].copy_from_slice(b"        ");
    let sum: u64 = h.iter().map(|b| *b as u64).sum();
    let s = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(s.as_bytes());
}

struct OutReport {
    entries: u64,
    skipped: Vec<Value>,
    mounts: Vec<String>,
    changed: Vec<String>,
    content_bytes: u64,
    max_content_bytes: u64,
    max_entries: u64,
    recursive: bool,
}

fn out_entry<W: Write>(w: &mut Writer<W>, parent: &Node, name: &[u8], rel: &[u8], specials_error: bool, rep: &mut OutReport, depth: usize) -> Result<()> {
    let e = match entry(parent, name) {
        Ok(e) => e,
        Err(Errno::NOENT) => return Ok(()),
        Err(err) => return Err(Error::sys(err, show(rel))),
    };
    if e.mount {
        rep.mounts.push(show(rel));
        return Ok(());
    }
    if rep.entries + rep.skipped.len() as u64 >= rep.max_entries {
        return Err(Error::new("too-large", "the archive has more entries than allowed"));
    }
    let st = &e.st;
    let (mode, uid, gid, mtime) = (st.st_mode as u32, st.st_uid as u64, st.st_gid as u64, st.st_mtime as i64);
    match ftype(mode) {
        "dir" => {
            let mut n = rel.to_vec();
            n.push(b'/');
            w.header(&n, b'5', mode, uid, gid, 0, mtime, b"")?;
            rep.entries += 1;
            if depth > 512 {
                return Err(Error::new("too-deep", show(rel)));
            }
            if rep.recursive {
                let node = reopen_dir(parent, &e).map_err(|err| Error::sys(err, show(rel)))?;
                for c in names(&node)? {
                    out_entry(w, &node, &c, &child_path(rel, &c), specials_error, rep, depth + 1)?;
                }
            }
        }
        "link" => {
            let target = fsx::readlink_at(parent, name).ok_or_else(|| Error::new("changed", show(rel)))?;
            w.header(rel, b'2', mode, uid, gid, 0, mtime, &target)?;
            rep.entries += 1;
        }
        "file" => {
            let fd = fsx::reopen_file(parent, name, &e).map_err(|err| Error::sys(err, show(rel)))?;
            let now = fs::fstat(&fd).map_err(|err| Error::sys(err, "fstat"))?;
            if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino) || ftype(now.st_mode as u32) != "file" {
                return Err(Error::new("changed", format!("{} changed while it was archived", show(rel))));
            }
            let size = now.st_size as u64;
            rep.content_bytes = rep.content_bytes.checked_add(size)
                .ok_or_else(|| Error::new("too-large", "the archive content is larger than allowed"))?;
            if rep.content_bytes > rep.max_content_bytes {
                return Err(Error::new("too-large", "the archive content is larger than allowed"));
            }
            w.header(rel, b'0', now.st_mode as u32, uid, gid, size, now.st_mtime as i64, b"")?;
            let mut left = size;
            let mut buf = vec![0u8; 1 << 16];
            while left > 0 {
                let want = (left as usize).min(buf.len());
                let n = match rustix::io::read(&fd, &mut buf[..want]) {
                    Ok(n) => n,
                    Err(Errno::INTR) => continue,
                    Err(err) => return Err(Error::sys(err, show(rel))),
                };
                if n == 0 {
                    // Shrank while read: the header's size stands, zeros fill.
                    rep.changed.push(show(rel));
                    let zeros = vec![0u8; left as usize];
                    w.put(&zeros)?;
                    break;
                }
                w.put(&buf[..n])?;
                left -= n as u64;
            }
            w.pad(size)?;
            rep.entries += 1;
        }
        other => {
            if specials_error {
                return Err(Error::new("special", format!("{} is a {other}", show(rel))));
            }
            rep.skipped.push(json!({"path": show(rel), "type": other}));
        }
    }
    Ok(())
}

/// A writer that refuses to pass *limit* bytes.
pub struct Bounded<W: Write> {
    w: W,
    left: u64,
}

impl<W: Write> Write for Bounded<W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if b.len() as u64 > self.left {
            return Err(std::io::Error::new(std::io::ErrorKind::FileTooLarge, "the archive is larger than allowed"));
        }
        let n = self.w.write(b)?;
        self.left -= n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// `tar-out {root, paths=[""], specials="skip", max_bytes?,
/// max_content_bytes?, max_entries?, recursive=true}` into *w*.
/// Nonrecursive paths archive a directory's own entry without its children;
/// the host can select a bounded inventory without interpreting image code.
pub fn tar_out<W: Write>(v: &Value, w: W) -> Result<(Value, W)> {
    tar_out_bounded(v, w, None)
}

/// As `tar_out`, never writing more than the smaller of *cap* and the
/// request's `max_bytes`.
pub fn tar_out_bounded<W: Write>(v: &Value, w: W, cap: Option<u64>) -> Result<(Value, W)> {
    let a = Args::new(v, &["root", "paths", "specials", "max_bytes", "max_content_bytes", "max_entries", "recursive", "resolver"])?;
    let limit = match (a.opt_u64("max_bytes")?, cap) {
        (Some(x), Some(c)) => x.min(c),
        (Some(x), None) => x,
        (None, Some(c)) => c,
        (None, None) => u64::MAX,
    };
    let (rep, b) = tar_out_inner(&a, Bounded { w, left: limit })?;
    Ok((rep, b.w))
}

fn tar_out_inner<W: Write>(a: &Args, w: W) -> Result<(Value, W)> {
    let root = fsx::open_root(a.str("root")?)?;
    let mut paths = a.str_list("paths")?;
    if a.opt("paths").is_none() {
        paths.push("");
    }
    let specials_error = match a.opt_str("specials")?.unwrap_or("skip") {
        "skip" => false,
        "error" => true,
        _ => return Err(Error::bad("specials must be skip or error")),
    };
    let mut rep = OutReport {
        entries: 0, skipped: Vec::new(), mounts: Vec::new(), changed: Vec::new(),
        content_bytes: 0,
        max_content_bytes: a.opt_u64("max_content_bytes")?.unwrap_or(u64::MAX),
        max_entries: a.opt_u64("max_entries")?.unwrap_or(u64::MAX),
        recursive: a.bool("recursive", true)?,
    };
    let mut w = Writer::new(w);
    for p in paths {
        let comps = split_rel(p.as_bytes())?;
        if comps.is_empty() {
            let here = fsx::open_dir(&root, &[])?;
            for c in names(&here)? {
                out_entry(&mut w, &here, &c, &c, specials_error, &mut rep, 1)?;
            }
        } else {
            let (parent, name) = fsx::open_parent(&root, &comps)?;
            out_entry(&mut w, &parent, name, p.as_bytes(), specials_error, &mut rep, 1)?;
        }
    }
    let bytes = w.bytes + (BLOCK * 2) as u64;
    let w = w.finish()?;
    Ok((
        json!({"entries": rep.entries, "bytes": bytes, "skipped": rep.skipped, "mounts": rep.mounts, "changed": rep.changed}),
        w,
    ))
}

// ---------------------------------------------------------------- reading

pub struct Member {
    pub name: Vec<u8>,
    pub kind: u8,
    pub mode: u32,
    pub size: u64,
    pub mtime: i64,
    pub link: Vec<u8>,
}

pub struct Reader<R: Read> {
    r: R,
    done: bool,
}

fn parse_num(f: &[u8]) -> Option<u64> {
    if f.first().map(|b| b & 0x80 != 0).unwrap_or(false) {
        // base-256: big-endian, first byte's top bit is the marker.
        if f[0] & 0x40 != 0 {
            return None; // negative
        }
        let mut v: u64 = (f[0] & 0x3f) as u64;
        for b in &f[1..] {
            v = v.checked_mul(256)?.checked_add(*b as u64)?;
        }
        return Some(v);
    }
    let s: Vec<u8> = f.iter().copied().take_while(|b| *b != 0).filter(|b| *b != b' ').collect();
    if s.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(std::str::from_utf8(&s).ok()?, 8).ok()
}

fn cstr(f: &[u8]) -> Vec<u8> {
    f.iter().copied().take_while(|b| *b != 0).collect()
}

impl<R: Read> Reader<R> {
    pub fn new(r: R) -> Self {
        Reader { r, done: false }
    }

    fn block(&mut self) -> Result<Option<[u8; BLOCK]>> {
        let mut b = [0u8; BLOCK];
        let mut got = 0;
        while got < BLOCK {
            match self.r.read(&mut b[got..]) {
                Ok(0) => {
                    return if got == 0 { Ok(None) } else { Err(Error::new("bad-archive", "the archive ends inside a block")) };
                }
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Error::io(e, "reading the archive")),
            }
        }
        Ok(Some(b))
    }

    /// Pass the member's data (and its padding) to *sink*.
    pub fn data(&mut self, size: u64, sink: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let mut left = size;
        let mut buf = vec![0u8; 1 << 16];
        while left > 0 {
            let want = (left as usize).min(buf.len());
            let n = match self.r.read(&mut buf[..want]) {
                Ok(0) => return Err(Error::new("bad-archive", "the archive ends inside a member")),
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(Error::io(e, "reading the archive")),
            };
            sink(&buf[..n])?;
            left -= n as u64;
        }
        let rem = (size % BLOCK as u64) as usize;
        if rem != 0 {
            let mut pad = vec![0u8; BLOCK - rem];
            self.r.read_exact(&mut pad).map_err(|e| Error::io(e, "reading the archive"))?;
        }
        Ok(())
    }

    fn meta(&mut self, size: u64) -> Result<Vec<u8>> {
        if size > META_LIMIT {
            return Err(Error::new("bad-archive", "an extended header is too large"));
        }
        let mut v = Vec::new();
        self.data(size, &mut |b| {
            v.extend_from_slice(b);
            Ok(())
        })?;
        Ok(v)
    }

    pub fn next(&mut self) -> Result<Option<Member>> {
        if self.done {
            return Ok(None);
        }
        let (mut path, mut linkpath, mut psize, mut pmtime) = (None, None, None, None);
        loop {
            let h = match self.block()? {
                None => {
                    self.done = true;
                    return Ok(None);
                }
                Some(h) => h,
            };
            if h.iter().all(|b| *b == 0) {
                self.done = true;
                return Ok(None);
            }
            let want = parse_num(&h[148..156]).ok_or_else(|| Error::new("bad-archive", "bad checksum field"))?;
            let mut unsigned: u64 = 0;
            let mut signed: i64 = 0;
            for (i, b) in h.iter().enumerate() {
                let b = if (148..156).contains(&i) { b' ' } else { *b };
                unsigned += b as u64;
                signed += (b as i8) as i64;
            }
            if want != unsigned && want as i64 != signed {
                return Err(Error::new("bad-archive", "a header's checksum does not match"));
            }
            let kind = h[156];
            let size = parse_num(&h[124..136]).ok_or_else(|| Error::new("bad-archive", "bad size field"))?;
            match kind {
                b'x' => {
                    let data = self.meta(size)?;
                    parse_pax(&data, &mut path, &mut linkpath, &mut psize, &mut pmtime)?;
                }
                b'g' => {
                    self.meta(size)?;
                }
                b'L' => path = Some(cstr(&self.meta(size)?)),
                b'K' => linkpath = Some(cstr(&self.meta(size)?)),
                _ => {
                    let mut name = cstr(&h[0..100]);
                    if &h[257..263] == b"ustar\0" {
                        let prefix = cstr(&h[345..500]);
                        if !prefix.is_empty() {
                            let mut p = prefix;
                            p.push(b'/');
                            p.extend_from_slice(&name);
                            name = p;
                        }
                    }
                    let m = Member {
                        name: path.unwrap_or(name),
                        kind,
                        mode: parse_num(&h[100..108]).unwrap_or(0) as u32 & 0o7777,
                        size: psize.unwrap_or(size),
                        mtime: pmtime.unwrap_or_else(|| parse_num(&h[136..148]).unwrap_or(0) as i64),
                        link: linkpath.unwrap_or_else(|| cstr(&h[157..257])),
                    };
                    return Ok(Some(m));
                }
            }
        }
    }
}

fn parse_pax(data: &[u8], path: &mut Option<Vec<u8>>, link: &mut Option<Vec<u8>>, size: &mut Option<u64>, mtime: &mut Option<i64>) -> Result<()> {
    let bad = || Error::new("bad-archive", "a malformed pax record");
    let mut rest = data;
    while !rest.is_empty() {
        let sp = rest.iter().position(|b| *b == b' ').ok_or_else(bad)?;
        let len: usize = std::str::from_utf8(&rest[..sp]).map_err(|_| bad())?.parse().map_err(|_| bad())?;
        if len <= sp + 1 || len > rest.len() || rest[len - 1] != b'\n' {
            return Err(bad());
        }
        let rec = &rest[sp + 1..len - 1];
        let eq = rec.iter().position(|b| *b == b'=').ok_or_else(bad)?;
        let (k, v) = (&rec[..eq], &rec[eq + 1..]);
        match k {
            b"path" => *path = Some(v.to_vec()),
            b"linkpath" => *link = Some(v.to_vec()),
            b"size" => *size = Some(std::str::from_utf8(v).map_err(|_| bad())?.parse().map_err(|_| bad())?),
            b"mtime" => {
                let s = std::str::from_utf8(v).map_err(|_| bad())?;
                let whole = s.split('.').next().unwrap_or("0");
                *mtime = Some(whole.parse().map_err(|_| bad())?);
            }
            _ => {}
        }
        rest = &rest[len..];
    }
    Ok(())
}

// ------------------------------------------------------------- extracting

struct Extract {
    root: Node,
    uid: Option<u32>,
    gid: Option<u32>,
    mask: u32,
    symlinks: bool,
    skip_specials: bool,
    created: HashSet<(u64, u64)>,
    /// Directories this extraction made: only these take the request's
    /// owner and the archive's mode.
    made_dirs: HashSet<(u64, u64)>,
    dir_times: Vec<(Vec<u8>, i64)>,
}

fn times(mtime: i64) -> Timestamps {
    let t = Timespec { tv_sec: mtime, tv_nsec: 0 };
    Timestamps { last_access: t, last_modification: t }
}

/// The member's name as components: a leading "./" and trailing "/" are
/// dropped; anything absolute or with "..", "." or empty parts is refused.
fn member_comps(name: &[u8]) -> Result<Vec<&[u8]>> {
    let mut n = name;
    while let Some(rest) = n.strip_prefix(b"./") {
        n = rest;
    }
    while let Some(rest) = n.strip_suffix(b"/") {
        n = rest;
    }
    if n == b"." {
        n = b"";
    }
    split_rel(n).map_err(|_| Error::new("bad-member", format!("member name {:?} is absolute or has an empty, '.' or '..' part", show(name))))
}

impl Extract {
    fn own(&self, fd: rustix::fd::BorrowedFd) -> Result<()> {
        if self.uid.is_some() || self.gid.is_some() {
            fs::fchown(fd, self.uid.map(Uid::from_raw_unchecked), self.gid.map(Gid::from_raw_unchecked)).map_err(|e| Error::sys(e, "fchown"))?;
        }
        Ok(())
    }

    /// The directory that holds *comps*' last name, creating missing ones.
    fn parent(&mut self, comps: &[&[u8]]) -> Result<Node> {
        let dir = OFlags::RDONLY | OFlags::DIRECTORY;
        let mut cur = fsx::open_dir(&self.root, &[])?;
        for (i, c) in comps[..comps.len() - 1].iter().enumerate() {
            let at = comps[..=i].iter().map(|c| show(c)).collect::<Vec<_>>().join("/");
            let fd = match open_one(&cur, c, dir) {
                Ok(fd) => fd,
                Err(Errno::NOENT) => {
                    match fs::mkdirat(&cur.fd, os(c), Mode::from_raw_mode(0o700)) {
                        Ok(()) | Err(Errno::EXIST) => {}
                        Err(e) => return Err(Error::sys(e, &at)),
                    }
                    let fd = open_one(&cur, c, dir).map_err(|e| Error::sys(e, &at))?;
                    self.own(fd.as_fd())?;
                    fs::fchmod(&fd, Mode::from_raw_mode(0o755 & self.mask)).map_err(|e| Error::sys(e, &at))?;
                    let st = fs::fstat(&fd).map_err(|e| Error::sys(e, "fstat"))?;
                    self.made_dirs.insert((st.st_dev as u64, st.st_ino as u64));
                    fd
                }
                Err(e) => return Err(Error::sys(e, &at)),
            };
            cur = Node { fd, dev: self.root.dev, mnt: self.root.mnt };
        }
        Ok(cur)
    }

    fn place(&self, parent: &Node, tmp: &[u8], name: &[u8], at: &str) -> Result<()> {
        fs::renameat_with(&parent.fd, os(tmp), &parent.fd, os(name), RenameFlags::empty()).map_err(|e| {
            let _ = fs::unlinkat(&parent.fd, os(tmp), AtFlags::empty());
            Error::sys(e, at)
        })
    }

    fn member<R: Read>(&mut self, r: &mut Reader<R>, m: &Member, rep: &mut InReport) -> Result<()> {
        let comps = member_comps(&m.name)?;
        let at = show(&m.name);
        if comps.is_empty() {
            if m.kind == b'5' {
                return r.data(m.size, &mut |_| Ok(()));
            }
            return Err(Error::new("bad-member", format!("member {at:?} names the root itself")));
        }
        let parent = self.parent(&comps)?;
        let name = *comps.last().unwrap();
        match m.kind {
            b'0' | 0 | b'7' => {
                let (tmp, fd) = temp_file(&parent)?;
                let mut fail = None;
                r.data(m.size, &mut |b| {
                    if fail.is_none() {
                        if let Err(e) = write_all(fd.as_fd(), b) {
                            fail = Some(e);
                        }
                    }
                    Ok(())
                })?;
                let done = fail.map_or(Ok(()), Err).and_then(|_| {
                    self.own(fd.as_fd())?;
                    fs::fchmod(&fd, Mode::from_raw_mode(m.mode & self.mask)).map_err(|e| Error::sys(e, &at))?;
                    fs::futimens(&fd, &times(m.mtime)).map_err(|e| Error::sys(e, &at))
                });
                if let Err(e) = done {
                    let _ = fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty());
                    return Err(e);
                }
                self.place(&parent, &tmp, name, &at)?;
                let st = fs::fstat(&fd).map_err(|e| Error::sys(e, "fstat"))?;
                self.created.insert((st.st_dev as u64, st.st_ino as u64));
                rep.files += 1;
                rep.bytes += m.size;
            }
            b'5' => {
                r.data(m.size, &mut |_| Ok(()))?;
                let mut made = false;
                let node = match entry(&parent, name) {
                    Ok(e) if e.mount => return Err(Error::sys(Errno::XDEV, &at)),
                    Ok(e) if ftype(e.st.st_mode as u32) == "dir" => reopen_dir(&parent, &e).map_err(|err| Error::sys(err, &at))?,
                    other => {
                        if other.is_ok() {
                            fs::unlinkat(&parent.fd, os(name), AtFlags::empty()).map_err(|e| Error::sys(e, &at))?;
                        } else if let Err(err) = other {
                            if err != Errno::NOENT {
                                return Err(Error::sys(err, &at));
                            }
                        }
                        match fs::mkdirat(&parent.fd, os(name), Mode::from_raw_mode(0o700)) {
                            Ok(()) | Err(Errno::EXIST) => {}
                            Err(e) => return Err(Error::sys(e, &at)),
                        }
                        let fd = open_one(&parent, name, OFlags::RDONLY | OFlags::DIRECTORY).map_err(|e| Error::sys(e, &at))?;
                        made = true;
                        Node { fd, dev: self.root.dev, mnt: self.root.mnt }
                    }
                };
                let st = fs::fstat(&node.fd).map_err(|e| Error::sys(e, "fstat"))?;
                let id = (st.st_dev as u64, st.st_ino as u64);
                if made || self.made_dirs.contains(&id) {
                    self.made_dirs.insert(id);
                    self.own(node.fd.as_fd())?;
                    fs::fchmod(&node.fd, Mode::from_raw_mode(m.mode & self.mask)).map_err(|e| Error::sys(e, &at))?;
                    self.dir_times.push((comps.join(&b'/'), m.mtime));
                } else {
                    // A directory that was there before keeps its owner,
                    // mode and times (a task's root-owned folder never
                    // becomes the agent's); the answer lists it.
                    rep.kept.push(at.clone());
                }
                rep.dirs += 1;
            }
            b'2' => {
                r.data(m.size, &mut |_| Ok(()))?;
                if !self.symlinks {
                    return Err(Error::new("link-refused", format!("member {at:?} is a symbolic link")));
                }
                if m.link.is_empty() || m.link.contains(&0) {
                    return Err(Error::new("bad-member", format!("member {at:?} has an empty link target")));
                }
                let tmp = {
                    let (tmp, fd) = temp_file(&parent)?;
                    drop(fd);
                    fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty()).map_err(|e| Error::sys(e, &at))?;
                    tmp
                };
                fs::symlinkat(os(&m.link), &parent.fd, os(&tmp)).map_err(|e| Error::sys(e, &at))?;
                if self.uid.is_some() || self.gid.is_some() {
                    fs::chownat(
                        &parent.fd,
                        os(&tmp),
                        self.uid.map(Uid::from_raw_unchecked),
                        self.gid.map(Gid::from_raw_unchecked),
                        AtFlags::SYMLINK_NOFOLLOW,
                    )
                    .map_err(|e| {
                        let _ = fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty());
                        Error::sys(e, &at)
                    })?;
                }
                self.place(&parent, &tmp, name, &at)?;
                rep.links += 1;
            }
            b'1' => {
                r.data(m.size, &mut |_| Ok(()))?;
                let tcomps = member_comps(&m.link)?;
                let (tparent, tname) = fsx::open_parent(&self.root, &tcomps)?;
                let te = entry(&tparent, tname).map_err(|e| Error::sys(e, show(&m.link)))?;
                let key = (te.st.st_dev as u64, te.st.st_ino as u64);
                if te.mount || ftype(te.st.st_mode as u32) != "file" || !self.created.contains(&key) {
                    return Err(Error::new("link-refused", format!("hard link {at:?} names {:?}, not a file this archive made", show(&m.link))));
                }
                let tmp = {
                    let (tmp, fd) = temp_file(&parent)?;
                    drop(fd);
                    fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty()).map_err(|e| Error::sys(e, &at))?;
                    tmp
                };
                fs::linkat(&tparent.fd, os(tname), &parent.fd, os(&tmp), AtFlags::empty()).map_err(|e| Error::sys(e, &at))?;
                let made = fs::statat(&parent.fd, os(&tmp), AtFlags::SYMLINK_NOFOLLOW).map_err(|e| Error::sys(e, &at))?;
                if (made.st_dev as u64, made.st_ino as u64) != key {
                    let _ = fs::unlinkat(&parent.fd, os(&tmp), AtFlags::empty());
                    return Err(Error::new("changed", format!("hard link target {:?} changed", show(&m.link))));
                }
                self.place(&parent, &tmp, name, &at)?;
                rep.links += 1;
            }
            b'3' | b'4' | b'6' => {
                r.data(m.size, &mut |_| Ok(()))?;
                if !self.skip_specials {
                    return Err(Error::new("special", format!("member {at:?} is a device or FIFO")));
                }
                rep.skipped.push(at);
            }
            k => {
                return Err(Error::new("bad-member", format!("member {at:?} has unsupported type {:?}", k as char)));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct InReport {
    members: u64,
    files: u64,
    dirs: u64,
    links: u64,
    bytes: u64,
    skipped: Vec<String>,
    kept: Vec<String>,
}

/// `tar-in {root, uid?, gid?, refuse_root_owner=false, mode_mask=0o777, symlinks="create", specials="refuse", mtime=true, max_bytes?, max_members=1000000}`
/// from *r*. Stops at the first member it refuses; what came before stays.
/// The request is checked before the root is opened or any member extracted.
pub fn tar_in<R: Read>(v: &Value, r: R) -> Result<Value> {
    let a = Args::new(v, &["root", "uid", "gid", "refuse_root_owner", "mode_mask", "symlinks", "specials", "mtime", "max_bytes", "max_members", "resolver", "data_b64"])?;
    let (uid, gid) = (a.opt_u32("uid")?, a.opt_u32("gid")?);
    if a.bool("refuse_root_owner", false)? && !matches!((uid, gid), (Some(u), Some(g)) if u != 0 && g != 0) {
        return Err(Error::new(
            "root-owner",
            "refuse_root_owner: the request must give extracted entries a non-root uid and gid",
        ));
    }
    let mask = a.opt("mode_mask").map(|m| mode_value(m, "mode_mask")).transpose()?.unwrap_or(0o777);
    if mask & 0o6000 != 0 {
        return Err(Error::bad("mode_mask must not keep a set-id bit"));
    }
    let max_bytes = a.opt_u64("max_bytes")?.unwrap_or(u64::MAX);
    let max_members = a.opt_u64("max_members")?.unwrap_or(1_000_000);
    let root = fsx::open_root(a.str("root")?)?;
    let symlinks = match a.opt_str("symlinks")?.unwrap_or("create") {
        "create" => true,
        "refuse" => false,
        _ => return Err(Error::bad("symlinks must be create or refuse")),
    };
    let skip_specials = match a.opt_str("specials")?.unwrap_or("refuse") {
        "refuse" => false,
        "skip" => true,
        _ => return Err(Error::bad("specials must be refuse or skip")),
    };
    let mut x = Extract {
        root,
        uid,
        gid,
        mask,
        symlinks,
        skip_specials,
        created: HashSet::new(),
        made_dirs: HashSet::new(),
        dir_times: Vec::new(),
    };
    let mut reader = Reader::new(r);
    let mut rep = InReport::default();
    loop {
        let m = match reader.next() {
            Ok(Some(m)) => m,
            Ok(None) => break,
            Err(e) => return Err(with_progress(e, &rep)),
        };
        rep.members += 1;
        if rep.members > max_members || rep.bytes.saturating_add(m.size) > max_bytes {
            return Err(with_progress(Error::new("too-large", "the archive has more members or bytes than allowed"), &rep));
        }
        if let Err(e) = x.member(&mut reader, &m, &mut rep) {
            return Err(with_progress(e, &rep));
        }
    }
    if a.bool("mtime", true)? {
        for (p, t) in x.dir_times.iter().rev() {
            let comps = split_rel(p)?;
            if let Ok(node) = fsx::open_dir(&x.root, &comps) {
                let _ = fs::futimens(&node.fd, &times(*t));
            }
        }
    }
    Ok(json!({"members": rep.members, "files": rep.files, "dirs": rep.dirs, "links": rep.links, "bytes": rep.bytes, "skipped": rep.skipped, "kept": rep.kept}))
}

fn with_progress(e: Error, rep: &InReport) -> Error {
    let d = json!({"members": rep.members, "files": rep.files, "dirs": rep.dirs, "links": rep.links, "bytes": rep.bytes});
    e.with_detail(d)
}

#[cfg(test)]
mod evidence_capture_regressions {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> std::path::PathBuf {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("bfx-evidence-{}-{stamp}", std::process::id()));
        fs::create_dir_all(root.join("cache")).unwrap();
        fs::write(root.join("cache/secret"), b"not selected").unwrap();
        fs::write(root.join("answer"), b"42\n").unwrap();
        root
    }

    /// Guards ac3bedaa1: explicit directories must not import excluded descendants.
    #[test]
    fn nonrecursive_inventory_has_no_unselected_bytes() {
        let root = fixture();
        let (report, bytes) = tar_out(&json!({
            "root": root, "paths": ["cache", "answer"], "recursive": false,
            "max_content_bytes": 3, "max_entries": 2,
        }), Vec::new()).unwrap();
        assert_eq!(report["entries"], 2);
        let mut reader = Reader::new(std::io::Cursor::new(bytes));
        let mut names = Vec::new();
        while let Some(member) = reader.next().unwrap() {
            names.push(String::from_utf8(member.name.clone()).unwrap());
            reader.data(member.size, &mut |_| Ok(())).unwrap();
        }
        assert_eq!(names, ["cache/", "answer"]);
        fs::remove_dir_all(root).unwrap();
    }

    /// Guards ac3bedaa1: reject oversized content before even its header travels.
    #[test]
    fn content_bound_precedes_file_transfer() {
        let root = fixture();
        let mut bytes = Vec::new();
        let error = tar_out(&json!({
            "root": root, "paths": ["answer"], "max_content_bytes": 2,
        }), &mut bytes).unwrap_err();
        assert_eq!(error.kind, "too-large");
        assert!(bytes.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    /// Guards ac3bedaa1: skipped entries consume the same manifest budget.
    #[test]
    fn skipped_special_also_consumes_an_entry() {
        let root = fixture();
        rustix::fs::mknodat(rustix::fs::CWD, root.join("pipe").as_path(),
            rustix::fs::FileType::Fifo, rustix::fs::Mode::from_raw_mode(0o600), 0).unwrap();
        let error = tar_out(&json!({
            "root": root, "paths": ["answer", "pipe"], "max_entries": 1,
        }), Vec::new()).unwrap_err();
        assert_eq!(error.kind, "too-large");
        fs::remove_dir_all(root).unwrap();
    }
}
