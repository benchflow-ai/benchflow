//! Property tests of the file primitives and tar, on generated trees and
//! archives (testgen; no attack string is written by hand).
//!
//! They run as root in a container that mounts a tmpfs at `/bfxt/root/m`,
//! a mount point inside the test root (scripts/test_bfx.sh); elsewhere the
//! mount cases are skipped and the ownership cases need root. Run with
//! `--test-threads=1`: the tests share `/bfxt` and the resolver switch.
//!
//! What they establish, for every generated case and in both resolver
//! modes (openat2 and the component walk):
//! - resolution beneath a root succeeds exactly when every component is a
//!   real directory on the root's mount (links: "link", mounts: "mount");
//! - no primitive and no archive changes anything outside the root or inside
//!   the mount: a snapshot of both is identical before and after;
//! - tar-in never leaves a device, FIFO or set-id bit;
//! - tar-out then tar-in reproduces a tree (types, modes, link targets,
//!   contents).

use crate::fsx::{self, FORCE_WALK};
use crate::testgen::Rng;
use crate::{ops, tar};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

const BASE: &str = "/bfxt";

fn base() -> PathBuf {
    if Path::new(BASE).is_dir() {
        PathBuf::from(BASE)
    } else {
        let p = std::env::temp_dir().join(format!("bfxt-{}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }
}

fn root_dir() -> PathBuf {
    let r = base().join("root");
    fs::create_dir_all(&r).unwrap();
    r
}

fn outside_dir() -> PathBuf {
    base().join("outside")
}

/// The mount point inside the root, when the container provides one.
fn mount() -> Option<PathBuf> {
    let m = root_dir().join("m");
    let (a, b) = (fs::symlink_metadata(&m).ok()?, fs::symlink_metadata(root_dir()).ok()?);
    if a.is_dir() && a.dev() != b.dev() {
        Some(m)
    } else {
        None
    }
}

fn is_root() -> bool {
    crate::sys::getresuid().1 == 0
}

fn p(b: &[u8]) -> &Path {
    Path::new(OsStr::from_bytes(b))
}

fn empty(dir: &Path, keep_mount: bool) {
    for e in fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        if keep_mount && e.file_name() == "m" && mount().is_some() {
            continue;
        }
        let ft = e.file_type().unwrap();
        if ft.is_dir() {
            fs::remove_dir_all(e.path()).unwrap();
        } else {
            fs::remove_file(e.path()).unwrap();
        }
    }
}

/// Empty the root (keeping the mount point), refill the outside and the
/// mount with known content.
fn reset() {
    empty(&root_dir(), true);
    let out = outside_dir();
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(out.join("d")).unwrap();
    fs::write(out.join("secret"), b"outside").unwrap();
    fs::set_permissions(out.join("secret"), fs::Permissions::from_mode(0o600)).unwrap();
    if let Some(m) = mount() {
        empty(&m, false);
        fs::set_permissions(&m, fs::Permissions::from_mode(0o755)).unwrap();
        fs::create_dir_all(m.join("inner")).unwrap();
        fs::write(m.join("inner/file"), b"mounted").unwrap();
        fs::set_permissions(m.join("inner/file"), fs::Permissions::from_mode(0o640)).unwrap();
    }
}

type Snap = BTreeMap<PathBuf, (String, u32, u32, u32, Vec<u8>)>;

/// Everything under *dir* (and *dir* itself): path -> (type, mode, uid,
/// gid, content or link target).
fn snapshot(dir: &Path) -> Snap {
    let mut out = BTreeMap::new();
    if let Ok(m) = fs::symlink_metadata(dir) {
        out.insert(dir.to_path_buf(), ("top".to_string(), m.mode() & 0o7777, m.uid(), m.gid(), Vec::new()));
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for e in rd {
            let path = e.unwrap().path();
            let m = fs::symlink_metadata(&path).unwrap();
            let ft = m.file_type();
            let (t, data) = if ft.is_symlink() {
                ("link", fs::read_link(&path).unwrap().as_os_str().as_bytes().to_vec())
            } else if ft.is_dir() {
                stack.push(path.clone());
                ("dir", Vec::new())
            } else if ft.is_file() {
                ("file", fs::read(&path).unwrap())
            } else {
                ("special", Vec::new())
            };
            out.insert(path, (t.to_string(), m.mode() & 0o7777, m.uid(), m.gid(), data));
        }
    }
    out
}

fn protected() -> Snap {
    let mut s = snapshot(&outside_dir());
    if let Some(m) = mount() {
        s.extend(snapshot(&m));
    }
    s
}

/// A generated tree in the root: directories, files and links whose
/// targets lead outside, into the mount, up the tree, or nowhere.
fn gen_tree(r: &mut Rng, n: usize) -> Vec<Vec<u8>> {
    let root = root_dir();
    let mut dirs: Vec<Vec<u8>> = vec![Vec::new()];
    let mut all: Vec<Vec<u8>> = Vec::new();
    let out = outside_dir();
    let targets: Vec<Vec<u8>> = vec![
        out.as_os_str().as_bytes().to_vec(),
        out.join("secret").as_os_str().as_bytes().to_vec(),
        out.join("d").as_os_str().as_bytes().to_vec(),
        b"/".to_vec(),
        b"..".to_vec(),
        b"../..".to_vec(),
        b"../../outside".to_vec(),
        b"m".to_vec(),
        b"m/inner".to_vec(),
        b"does-not-exist".to_vec(),
    ];
    for _ in 0..n {
        let parent = r.pick(&dirs).clone();
        let name = r.name();
        let rel = fsx::child_path(&parent, &name);
        if rel == b"m" {
            continue;
        }
        let path = root.join(p(&rel));
        if fs::symlink_metadata(&path).is_ok() {
            continue;
        }
        match r.below(3) {
            0 => {
                fs::create_dir(&path).unwrap();
                dirs.push(rel.clone());
            }
            1 => fs::write(&path, &name).unwrap(),
            _ => {
                let t = if r.chance(1, 3) && !all.is_empty() { r.pick(&all).clone() } else { r.pick(&targets).clone() };
                symlink(p(&t), &path).unwrap();
            }
        }
        all.push(rel);
    }
    all
}

/// A random path through the tree: known names, the mount, and new names.
fn gen_path(r: &mut Rng, known: &[Vec<u8>]) -> Vec<u8> {
    let mut path: Vec<u8> = if !known.is_empty() && r.chance(2, 3) { r.pick(known).clone() } else { Vec::new() };
    for _ in 0..r.below(3) {
        let extra: Vec<u8> = match r.below(4) {
            0 => b"m".to_vec(),
            1 => b"inner".to_vec(),
            _ => r.name(),
        };
        path = fsx::child_path(&path, &extra);
    }
    if path.is_empty() {
        path = r.name();
    }
    path
}

/// The kind open_dir must report for *rel*, from lstat alone.
fn model_dir(rel: &[u8]) -> Option<&'static str> {
    let root = root_dir();
    let root_dev = fs::metadata(&root).unwrap().dev();
    let mut cur = root.clone();
    for c in rel.split(|b| *b == b'/') {
        cur = cur.join(p(c));
        let m = match fs::symlink_metadata(&cur) {
            Ok(m) => m,
            Err(_) => return Some("missing"),
        };
        if m.file_type().is_symlink() {
            return Some("link");
        }
        if m.dev() != root_dev {
            return Some("mount");
        }
        if !m.is_dir() {
            return Some("not-dir");
        }
    }
    None
}

fn both_modes(f: &mut dyn FnMut()) {
    for walk in [false, true] {
        FORCE_WALK.store(walk, Ordering::Relaxed);
        f();
    }
    FORCE_WALK.store(false, Ordering::Relaxed);
}

fn root_arg() -> String {
    root_dir().to_str().unwrap().to_string()
}

#[test]
fn resolution_matches_the_lstat_model_in_both_modes() {
    let mut r = Rng::new(1);
    for case in 0..60 {
        reset();
        let known = gen_tree(&mut r, 25);
        for _ in 0..40 {
            let rel = gen_path(&mut r, &known);
            let want = model_dir(&rel);
            both_modes(&mut || {
                let root = fsx::open_root(&root_arg()).unwrap();
                let comps = fsx::split_rel(&rel).unwrap();
                let got = fsx::open_dir(&root, &comps).err().map(|e| e.kind);
                assert_eq!(got, want, "case {case} path {:?} walk={}", String::from_utf8_lossy(&rel), FORCE_WALK.load(Ordering::Relaxed));
            });
        }
    }
}

#[test]
fn names_made_of_dots_are_names_and_dot_components_are_refused() {
    for bad in [&b"/a"[..], b"a/../b", b"..", b".", b"a/./b", b"a//b", b"a/"] {
        assert!(fsx::split_rel(bad).is_err(), "{:?}", String::from_utf8_lossy(bad));
    }
    for good in [&b"..a"[..], b"...", b"a..", b".a/..b/..."] {
        assert!(fsx::split_rel(good).is_ok());
    }
}

#[test]
fn primitives_never_change_anything_outside_the_root_or_in_a_mount() {
    let mut r = Rng::new(2);
    let owner = if is_root() { Some(4242) } else { None };
    let (mut ok, mut all) = (0, 0);
    for case in 0..80 {
        reset();
        let known = gen_tree(&mut r, 30);
        let before = protected();
        both_modes(&mut || {
            for _ in 0..15 {
                let rel = String::from_utf8_lossy(&gen_path(&mut r, &known)).into_owned();
                let v = match r.below(8) {
                    0 => ops::chmod(&json!({"root": root_arg(), "path": rel, "mode": 0o777, "recursive": true})),
                    1 => ops::chmod(&json!({"root": root_arg(), "path": "", "clear": 0o022, "recursive": true})),
                    2 => match owner {
                        Some(u) => ops::chown(&json!({"root": root_arg(), "path": rel, "uid": u, "gid": u, "recursive": true})),
                        None => continue,
                    },
                    3 => ops::write(&json!({"root": root_arg(), "path": rel, "data": "written", "mode": 0o666})),
                    4 => ops::mkdir(&json!({"root": root_arg(), "path": rel, "parents": true, "mode": 0o777})),
                    5 => ops::remove(&json!({"root": root_arg(), "path": rel, "recursive": true, "missing_ok": true})),
                    6 => ops::clear(&json!({"root": root_arg(), "path": rel})),
                    _ => ops::walk(&json!({"root": root_arg(), "hash_max_bytes": 1 << 20})),
                };
                all += 1;
                ok += v.is_ok() as u32;
            }
        });
        assert_eq!(protected(), before, "case {case}");
    }
    // Not vacuous: most operations act.
    assert!(ok * 3 > all, "{ok} of {all} operations succeeded");
}

#[test]
fn the_container_provides_the_mount_point() {
    // In the test container (scripts/test_bfx.sh) the mount cases must run.
    if Path::new(BASE).is_dir() {
        assert!(mount().is_some(), "no tmpfs at /bfxt/root/m");
    }
}

#[test]
fn remove_refuses_an_entry_that_is_not_the_scanned_one() {
    reset();
    fs::write(root_dir().join("f"), b"x").unwrap();
    let m = fs::symlink_metadata(root_dir().join("f")).unwrap();
    let wrong = ops::remove(&json!({"root": root_arg(), "path": "f", "expect": {"dev": m.dev(), "ino": m.ino() + 1}}));
    assert_eq!(wrong.unwrap_err().kind, "changed");
    assert!(root_dir().join("f").exists());
    ops::remove(&json!({"root": root_arg(), "path": "f", "expect": {"dev": m.dev(), "ino": m.ino()}})).unwrap();
    assert!(!root_dir().join("f").exists());
}

#[test]
fn write_replaces_a_link_and_never_writes_through_it() {
    reset();
    symlink(outside_dir().join("secret"), root_dir().join("l")).unwrap();
    let before = protected();
    ops::write(&json!({"root": root_arg(), "path": "l", "data": "new"})).unwrap();
    assert_eq!(protected(), before);
    assert_eq!(fs::read(root_dir().join("l")).unwrap(), b"new");
    assert!(!fs::symlink_metadata(root_dir().join("l")).unwrap().file_type().is_symlink());
}

// ------------------------------------------------------------------- tar

/// A generated archive: members with absolute, `..`, `./`, through-a-link
/// and into-the-mount names; links whose targets leave the root; hard links
/// to outside files; devices and FIFOs; long (pax) names; ordinary members.
fn gen_archive(r: &mut Rng) -> Vec<u8> {
    let mut w = tar::Writer::new(Vec::new());
    let mut names: Vec<Vec<u8>> = Vec::new();
    let mut links: Vec<Vec<u8>> = Vec::new();
    let out = outside_dir();
    for _ in 0..(5 + r.below(20)) {
        let mut name = if !names.is_empty() && r.chance(1, 3) { fsx::child_path(r.pick(&names), &r.name()) } else { r.name() };
        match r.below(10) {
            0 => name = [out.as_os_str().as_bytes(), b"/", &r.name()].concat(),
            1 => name = fsx::child_path(b"..", &name),
            2 => name = fsx::child_path(&name, b".."),
            3 if !links.is_empty() => name = fsx::child_path(r.pick(&links), &r.name()),
            4 => name = fsx::child_path(b".", &name),
            5 => name = fsx::child_path(b"m", &name),
            6 => {
                while name.len() < 180 {
                    name = fsx::child_path(&name, &r.name());
                }
            }
            _ => {}
        }
        let mode = r.below(0o10000) as u32;
        match r.below(7) {
            0 | 1 => {
                let data: Vec<u8> = (0..r.below(3000)).map(|_| r.below(256) as u8).collect();
                w.header(&name, b'0', mode, 0, 0, data.len() as u64, 1_700_000_000, b"").unwrap();
                w.put(&data).unwrap();
                w.pad(data.len() as u64).unwrap();
            }
            2 => w.header(&name, b'5', mode, 0, 0, 0, 1_700_000_000, b"").unwrap(),
            3 => {
                let t = match r.below(4) {
                    0 => out.as_os_str().as_bytes().to_vec(),
                    1 => b"../../outside".to_vec(),
                    2 => b"m".to_vec(),
                    _ => r.name(),
                };
                w.header(&name, b'2', 0o777, 0, 0, 0, 1_700_000_000, &t).unwrap();
                links.push(name.clone());
            }
            4 => {
                let t = match r.below(3) {
                    0 => out.join("secret").as_os_str().as_bytes().to_vec(),
                    1 => b"../outside/secret".to_vec(),
                    _ if !names.is_empty() => r.pick(&names).clone(),
                    _ => b"x".to_vec(),
                };
                w.header(&name, b'1', 0o644, 0, 0, 0, 1_700_000_000, &t).unwrap();
            }
            5 => w.header(&name, b'6', 0o666, 0, 0, 0, 1_700_000_000, b"").unwrap(),
            _ => w.header(&name, b'3', 0o666, 0, 0, 0, 1_700_000_000, b"").unwrap(),
        }
        names.push(name);
    }
    w.finish().unwrap()
}

fn no_specials_or_setid(dir: &Path) {
    for (path, (t, mode, ..)) in snapshot(dir) {
        assert_ne!(t, "special", "{path:?}");
        if t != "link" && t != "top" {
            assert_eq!(mode & 0o7000, 0, "{path:?} has a set-id or sticky bit");
        }
    }
}

#[test]
fn tar_in_of_generated_archives_stays_beneath_the_root() {
    let mut r = Rng::new(3);
    for case in 0..300 {
        reset();
        let data = gen_archive(&mut r);
        let before = protected();
        both_modes(&mut || {
            let _ = tar::tar_in(&json!({"root": root_arg()}), std::io::Cursor::new(data.clone()));
        });
        assert_eq!(protected(), before, "case {case}");
        let mut in_root = snapshot(&root_dir());
        if let Some(m) = mount() {
            in_root.retain(|k, _| !k.starts_with(&m));
        }
        let fake = in_root.keys().cloned().collect::<Vec<_>>();
        for path in fake {
            assert!(path.starts_with(root_dir()), "case {case}: {path:?}");
        }
        no_specials_or_setid(&root_dir());
    }
}

#[test]
fn tar_in_refuses_each_hostile_member_kind() {
    let mut r = Rng::new(4);
    let out = outside_dir();
    let n = r.name();
    let cases: Vec<(Vec<u8>, u8, Vec<u8>, &str)> = vec![
        ([out.as_os_str().as_bytes(), b"/", &n].concat(), b'0', vec![], "bad-member"),
        (fsx::child_path(b"..", &n), b'0', vec![], "bad-member"),
        (n.clone(), b'6', vec![], "special"),
        (n.clone(), b'3', vec![], "special"),
        (n.clone(), b'1', out.join("secret").as_os_str().as_bytes().to_vec(), "bad-member"),
    ];
    for (name, kind, link, want) in cases {
        reset();
        let mut w = tar::Writer::new(Vec::new());
        w.header(&name, kind, 0o644, 0, 0, 0, 0, &link).unwrap();
        let e = tar::tar_in(&json!({"root": root_arg()}), std::io::Cursor::new(w.finish().unwrap())).unwrap_err();
        assert_eq!(e.kind, want, "{:?}", String::from_utf8_lossy(&name));
    }
    // A link member, then a member through it: the second is refused.
    reset();
    let mut w = tar::Writer::new(Vec::new());
    w.header(&n, b'2', 0o777, 0, 0, 0, 0, out.as_os_str().as_bytes()).unwrap();
    let through = fsx::child_path(&n, &r.name());
    w.header(&through, b'0', 0o644, 0, 0, 1, 0, b"").unwrap();
    w.put(b"x").unwrap();
    w.pad(1).unwrap();
    let before = protected();
    let e = tar::tar_in(&json!({"root": root_arg()}), std::io::Cursor::new(w.finish().unwrap())).unwrap_err();
    assert_eq!(e.kind, "link");
    assert_eq!(protected(), before);
}

/// An archive of a set-uid file, a set-gid directory and a sticky file.
fn setid_archive() -> Vec<u8> {
    let mut w = tar::Writer::new(Vec::new());
    w.header(b"d", b'5', 0o2775, 0, 0, 0, 0, b"").unwrap();
    w.header(b"d/su", b'0', 0o4755, 0, 0, 1, 0, b"").unwrap();
    w.put(b"x").unwrap();
    w.pad(1).unwrap();
    w.header(b"st", b'0', 0o1777, 0, 0, 1, 0, b"").unwrap();
    w.put(b"y").unwrap();
    w.pad(1).unwrap();
    w.finish().unwrap()
}

#[test]
fn tar_in_never_leaves_a_set_id_bit_and_refuses_a_mask_that_keeps_one() {
    reset();
    tar::tar_in(&json!({"root": root_arg()}), std::io::Cursor::new(setid_archive())).unwrap();
    no_specials_or_setid(&root_dir());
    for mask in [0o7777, 0o4777, 0o2777, 0o6000] {
        reset();
        let e = tar::tar_in(&json!({"root": root_arg(), "mode_mask": mask}), std::io::Cursor::new(setid_archive())).unwrap_err();
        assert_eq!(e.kind, "bad-request", "{mask:o}");
        assert!(fs::read_dir(root_dir()).unwrap().all(|e| e.unwrap().file_name() == "m"), "nothing extracted");
    }
    // The sticky bit alone is not a set-id bit: a mask may keep it.
    reset();
    tar::tar_in(&json!({"root": root_arg(), "mode_mask": 0o1777}), std::io::Cursor::new(setid_archive())).unwrap();
    for (path, (t, mode, ..)) in snapshot(&root_dir()) {
        if t != "link" && t != "top" {
            assert_eq!(mode & 0o6000, 0, "{path:?}");
        }
    }
}

#[test]
fn tar_in_with_refuse_root_owner_refuses_any_root_ownership() {
    let bad = [
        json!({}),
        json!({"uid": 0, "gid": 0}),
        json!({"uid": 0, "gid": 1000}),
        json!({"uid": 1000, "gid": 0}),
        json!({"uid": 1000}),
        json!({"gid": 1000}),
    ];
    for owner in bad {
        reset();
        let mut v = json!({"root": root_arg(), "refuse_root_owner": true});
        v.as_object_mut().unwrap().extend(owner.as_object().unwrap().clone());
        let e = tar::tar_in(&v, std::io::Cursor::new(setid_archive())).unwrap_err();
        assert_eq!(e.kind, "root-owner", "{v}");
        assert!(fs::read_dir(root_dir()).unwrap().all(|e| e.unwrap().file_name() == "m"), "nothing extracted: {v}");
    }
    if !is_root() {
        return;
    }
    reset();
    let v = json!({"root": root_arg(), "refuse_root_owner": true, "uid": 1000, "gid": 1001});
    tar::tar_in(&v, std::io::Cursor::new(setid_archive())).unwrap();
    let snap = snapshot(&root_dir());
    let made: Vec<_> = snap.iter().filter(|(_, (t, ..))| t != "top").filter(|(k, _)| !k.starts_with(root_dir().join("m"))).collect();
    assert_eq!(made.len(), 3, "{made:?}");
    for (path, _) in made {
        let md = fs::symlink_metadata(path).unwrap();
        assert_eq!((md.uid(), md.gid()), (1000, 1001), "{path:?}");
        assert_eq!(md.mode() & 0o6000, 0, "{path:?}");
    }
}

#[test]
fn tar_in_leaves_a_directory_that_was_there_as_it_was_and_lists_it() {
    if !is_root() {
        return;
    }
    reset();
    let keep = root_dir().join("keep");
    fs::create_dir(&keep).unwrap();
    fs::set_permissions(&keep, fs::Permissions::from_mode(0o750)).unwrap();
    let before = fs::symlink_metadata(&keep).unwrap();
    let mut w = tar::Writer::new(Vec::new());
    w.header(b"keep", b'5', 0o777, 0, 0, 0, 0, b"").unwrap();
    w.header(b"keep/f", b'0', 0o644, 0, 0, 1, 0, b"").unwrap();
    w.put(b"x").unwrap();
    w.pad(1).unwrap();
    // A directory this extraction makes, named twice, is its own.
    w.header(b"new", b'5', 0o755, 0, 0, 0, 0, b"").unwrap();
    w.header(b"new", b'5', 0o750, 0, 0, 0, 0, b"").unwrap();
    let v = json!({"root": root_arg(), "refuse_root_owner": true, "uid": 1000, "gid": 1001});
    let answer = tar::tar_in(&v, std::io::Cursor::new(w.finish().unwrap())).unwrap();
    assert_eq!(answer["kept"], json!(["keep"]));
    let after = fs::symlink_metadata(&keep).unwrap();
    assert_eq!((after.uid(), after.gid(), after.mode() & 0o7777), (0, 0, 0o750));
    assert_eq!(after.mtime(), before.mtime());
    let f = fs::symlink_metadata(keep.join("f")).unwrap();
    assert_eq!((f.uid(), f.gid()), (1000, 1001));
    let new = fs::symlink_metadata(root_dir().join("new")).unwrap();
    assert_eq!((new.uid(), new.gid(), new.mode() & 0o7777), (1000, 1001, 0o750));
}

/// Three regular members of 1000 bytes each: a, b, c.
fn three_files() -> Vec<u8> {
    let mut w = tar::Writer::new(Vec::new());
    for name in [b"a", b"b", b"c"] {
        w.header(name, b'0', 0o644, 0, 0, 1000, 0, b"").unwrap();
        w.put(&[b'z'; 1000]).unwrap();
        w.pad(1000).unwrap();
    }
    w.finish().unwrap()
}

#[test]
fn tar_in_stops_at_its_member_and_byte_bounds() {
    for bound in [json!({"max_members": 2}), json!({"max_bytes": 2500})] {
        reset();
        let mut v = json!({"root": root_arg()});
        v.as_object_mut().unwrap().extend(bound.as_object().unwrap().clone());
        let e = tar::tar_in(&v, std::io::Cursor::new(three_files())).unwrap_err();
        assert_eq!(e.kind, "too-large", "{bound}");
        // What came before the bound stays; the third member never lands.
        assert!(root_dir().join("a").exists() && root_dir().join("b").exists(), "{bound}");
        assert!(!root_dir().join("c").exists(), "{bound}");
    }
}

#[test]
fn tar_out_stops_at_its_byte_bound_with_too_large() {
    reset();
    for name in ["a", "b", "c"] {
        fs::write(root_dir().join(name), [b'z'; 1000]).unwrap();
    }
    let (_, whole) = tar::tar_out(&json!({"root": root_arg(), "paths": ["a", "b", "c"]}), Vec::new()).unwrap();
    let e = tar::tar_out(&json!({"root": root_arg(), "paths": ["a", "b", "c"], "max_bytes": 2500}), Vec::new()).unwrap_err();
    assert_eq!(e.kind, "too-large");
    assert!(whole.len() > 2500);
}

#[test]
fn tar_out_then_tar_in_reproduces_a_tree() {
    let mut r = Rng::new(5);
    let copy = base().join("copy");
    for case in 0..60 {
        reset();
        gen_tree(&mut r, 30);
        let m = mount();
        let rel = |snap: Snap, top: &Path| -> BTreeMap<PathBuf, (String, u32, Vec<u8>)> {
            snap.into_iter()
                .filter(|(k, (t, ..))| t != "top" && m.as_ref().map_or(true, |m| !k.starts_with(m)))
                .map(|(k, (t, mode, _, _, d))| (k.strip_prefix(top).unwrap().to_path_buf(), (t, mode & 0o777, d)))
                .collect()
        };
        let want = rel(snapshot(&root_dir()), &root_dir());
        let (rep, data) = tar::tar_out(&json!({"root": root_arg()}), Vec::new()).unwrap();
        if m.is_some() {
            assert_eq!(rep["mounts"], json!(["m"]), "case {case}");
        }
        let _ = fs::remove_dir_all(&copy);
        fs::create_dir_all(&copy).unwrap();
        tar::tar_in(&json!({"root": copy.to_str().unwrap()}), std::io::Cursor::new(data)).unwrap();
        assert_eq!(rel(snapshot(&copy), &copy), want, "case {case}");
    }
    let _ = fs::remove_dir_all(&copy);
}

#[test]
fn tar_out_records_links_as_links_and_skips_fifos() {
    reset();
    symlink(outside_dir().join("secret"), root_dir().join("l")).unwrap();
    rustix::fs::mknodat(rustix::fs::CWD, root_dir().join("fifo").as_path(), rustix::fs::FileType::Fifo, rustix::fs::Mode::from_raw_mode(0o644), 0).unwrap();
    let (rep, data) = tar::tar_out(&json!({"root": root_arg()}), Vec::new()).unwrap();
    assert_eq!(rep["skipped"], json!([{"path": "fifo", "type": "fifo"}]));
    let mut rd = tar::Reader::new(std::io::Cursor::new(data));
    let mut seen = Vec::new();
    while let Some(m) = rd.next().unwrap() {
        rd.data(m.size, &mut |_| Ok(())).unwrap();
        seen.push((String::from_utf8_lossy(&m.name).into_owned(), m.kind, m.link.clone()));
    }
    assert!(seen.contains(&("l".to_string(), b'2', outside_dir().join("secret").as_os_str().as_bytes().to_vec())));
    assert!(!seen.iter().any(|(n, ..)| n == "fifo"));
}

// ------------------------------------------------------------------ exec

#[test]
fn exec_gives_exactly_the_requested_env_ids_and_no_new_privs() {
    if !is_root() || !Path::new("/bin/sh").exists() {
        return;
    }
    // cat changes nothing in its environment: what it reads is what exec gave.
    let run = |what: &str| {
        let v = crate::exec::run(&json!({
            "argv": ["/bin/cat", format!("/proc/self/{what}")],
            "env": {"A": "1", "B": "two words"},
            "uid": 65534, "gid": 65533, "groups": [65532, 65531],
            "cwd": "/",
        }))
        .unwrap();
        String::from_utf8_lossy(&crate::b64::decode(v["stdout_b64"].as_str().unwrap()).unwrap()).into_owned()
    };
    let (status, environ) = (run("status"), run("environ"));
    let line = |k: &str| status.lines().find(|l| l.starts_with(k)).unwrap().split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
    assert_eq!(line("Uid:"), "65534 65534 65534 65534");
    assert_eq!(line("Gid:"), "65533 65533 65533 65533");
    assert_eq!(line("Groups:"), "65531 65532");
    assert_eq!(line("NoNewPrivs:"), "1");
    assert_eq!(line("CapAmb:"), "0000000000000000");
    assert_eq!(line("CapInh:"), "0000000000000000");
    let bnd = u64::from_str_radix(&line("CapBnd:"), 16).unwrap();
    assert_eq!(bnd & ((1 << 12) | (1 << 13)), 0, "NET_ADMIN and NET_RAW are out of the bounding set");
    let mut env: Vec<&str> = environ.split('\0').filter(|s| !s.is_empty()).collect();
    env.sort();
    assert_eq!(env, vec!["A=1", "B=two words"]);
}

#[test]
fn exec_refuses_a_relative_program_and_a_cwd_through_a_link() {
    let e = crate::exec::run(&json!({"argv": ["sh", "-c", "true"]})).unwrap_err();
    assert_eq!(e.kind, "bad-request");
    reset();
    symlink(outside_dir(), root_dir().join("l")).unwrap();
    let e = crate::exec::run(&json!({"argv": ["/bin/true"], "cwd": {"root": root_arg(), "path": "l"}})).unwrap_err();
    assert_eq!(e.kind, "link");
}

// ---------------------------------------------------------------- kill

#[test]
fn kill_is_k1s_kill_only_and_refuses_keeping_options() {
    // K2's keeping kill runs only on today's route in 0.8.3.
    for extra in [json!({"keep_ports": [8080]}), json!({"keep_ports": []}), json!({"wait": true})] {
        let mut v = json!({"accounts": [{"uid": 65534, "gid": 65534}]});
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let e = crate::kill::kill(&v).unwrap_err();
        assert_eq!(e.kind, "bad-request", "{v}");
    }
}

// ---------------------------------------------------------------- scan
// The facts TR-PROTOCOL.md v1 asks for, on a small tree under the test
// root and on the test image's own files.

fn scan(args: serde_json::Value) -> serde_json::Value {
    crate::scan::scan(&args).unwrap()
}

fn entry_of<'a>(answer: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    answer["entries"].as_array().unwrap().iter().find(|e| e["path"] == path)
}

#[test]
fn scan_paths_report_every_prefix_and_stop_at_a_link_or_a_missing_name() {
    reset();
    let r = root_dir();
    fs::create_dir_all(r.join("a/b")).unwrap();
    symlink("../elsewhere", r.join("a/l")).unwrap();
    let root = r.to_str().unwrap();
    let answer = scan(json!({"paths": [format!("{root}/a/l/x"), format!("{root}/a/b/missing/y")]}));
    let mut prefix = String::new();
    assert_eq!(entry_of(&answer, "/").unwrap()["type"], "dir");
    for c in root.trim_start_matches('/').split('/') {
        prefix = format!("{prefix}/{c}");
        assert_eq!(entry_of(&answer, &prefix).unwrap()["type"], "dir", "{prefix}");
    }
    let link = entry_of(&answer, &format!("{root}/a/l")).unwrap();
    assert_eq!((link["type"].as_str(), link["link"].as_str()), (Some("link"), Some("../elsewhere")));
    assert!(entry_of(&answer, &format!("{root}/a/l/x")).is_none());
    let missing = entry_of(&answer, &format!("{root}/a/b/missing")).unwrap();
    assert_eq!(missing["exists"], false);
    assert!(entry_of(&answer, &format!("{root}/a/b/missing/y")).is_none());
    assert!(entry_of(&answer, &format!("{root}/a/b")).unwrap()["mount_id"].is_u64());
    // Entries are unique by path though both paths share their prefixes.
    let paths: Vec<_> = answer["entries"].as_array().unwrap().iter().map(|e| e["path"].clone()).collect();
    let unique: std::collections::BTreeSet<_> = paths.iter().map(|p| p.to_string()).collect();
    assert_eq!(paths.len(), unique.len());
}

#[test]
fn scan_walks_list_to_their_depth_and_never_enter_links_or_mounts() {
    reset();
    let r = root_dir();
    fs::create_dir_all(r.join("d1/d2/d3")).unwrap();
    fs::write(r.join("d1/d2/d3/deep"), b"x").unwrap();
    symlink(outside_dir(), r.join("out")).unwrap();
    let root = r.to_str().unwrap().to_string();
    let one = scan(json!({"walks": [{"root": root, "depth": 1}]}));
    assert_eq!(entry_of(&one, &root).unwrap()["listing"], true);
    assert!(entry_of(&one, &format!("{root}/d1")).unwrap().get("listing").is_none());
    assert!(entry_of(&one, &format!("{root}/d1/d2")).is_none());
    assert_eq!(entry_of(&one, &format!("{root}/out")).unwrap()["type"], "link");
    assert!(entry_of(&one, &format!("{root}/out/d")).is_none());
    let three = scan(json!({"walks": [{"root": root, "depth": 3}]}));
    assert_eq!(entry_of(&three, &format!("{root}/d1/d2")).unwrap()["listing"], true);
    assert!(entry_of(&three, &format!("{root}/d1/d2/d3")).unwrap().get("listing").is_none());
    assert!(entry_of(&three, &format!("{root}/d1/d2/d3/deep")).is_none());
    if mount().is_some() {
        let m = entry_of(&three, &format!("{root}/m")).unwrap();
        assert_eq!(m["mount"], true);
        assert!(entry_of(&three, &format!("{root}/m/inner")).is_none());
    }
    let gone = scan(json!({"walks": [{"root": format!("{root}/nothing"), "depth": 2}]}));
    assert_eq!(entry_of(&gone, &format!("{root}/nothing")).unwrap()["exists"], false);
}

#[test]
fn scan_contents_are_the_bytes_of_regular_files_only() {
    reset();
    let r = root_dir();
    fs::write(r.join("conf"), b"include /etc/x\n").unwrap();
    symlink("conf", r.join("alias")).unwrap();
    fs::create_dir(r.join("dir")).unwrap();
    fs::write(r.join("big"), vec![b'a'; 64]).unwrap();
    let root = r.to_str().unwrap();
    let names = ["conf", "alias", "dir", "big", "none"].map(|n| format!("{root}/{n}"));
    let answer = scan(json!({"contents": names, "content_max_bytes": 32}));
    let c = &answer["contents"];
    assert_eq!(c[&names[0]]["b64"], crate::b64::encode(b"include /etc/x\n"));
    assert_eq!(c[&names[0]]["sha256"], ops::hex(&<sha2::Sha256 as sha2::Digest>::digest(b"include /etc/x\n")));
    assert_eq!(c[&names[1]]["error"], "link");
    assert_eq!(c[&names[2]]["error"], "not-regular");
    assert_eq!(c[&names[3]]["error"], "too-large");
    assert_eq!(c[&names[4]]["error"], "missing");
    assert_eq!(answer["errors"].as_array().unwrap().len(), 4);
}

#[test]
fn scan_elf_reports_load_references_tags_and_shebangs() {
    reset();
    let r = root_dir();
    fs::write(r.join("script"), b"#!/bin/sh -e\necho hi\n").unwrap();
    let script = format!("{}/script", r.to_str().unwrap());
    let mut want = vec![script.clone()];
    let busybox = Path::new("/bin/busybox").is_file();
    if busybox {
        want.push("/bin/busybox".into());
    }
    let answer = scan(json!({"elf": want}));
    assert_eq!(answer["elf"][&script], json!({"elf": null, "shebang": "#!/bin/sh -e"}));
    if busybox {
        let f = &answer["elf"]["/bin/busybox"]["elf"];
        assert!(f["interp"].as_str().unwrap().starts_with("/lib/ld-musl"));
        assert!(f["needed"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("musl")));
        for key in ["audit", "depaudit", "filter", "auxiliary", "unknown_tags"] {
            assert_eq!(f[key], json!([]), "{key}");
        }
        assert!(f["flags_1"].is_u64() && f["flags"].is_u64());
    }
}

#[test]
fn scan_lists_other_processes_with_their_ids_descriptors_and_maps() {
    let mut child = std::process::Command::new("/bin/sleep").arg("5").spawn().unwrap();
    let answer = scan(json!({"processes": true}));
    let _ = child.kill();
    let _ = child.wait();
    let procs = answer["processes"].as_array().unwrap();
    assert!(procs.iter().all(|p| p["pid"] != std::process::id()));
    let p = procs.iter().find(|p| p["pid"] == child.id()).expect("the child is listed");
    assert_eq!(p["uid"].as_array().unwrap().len(), 4);
    assert_eq!(p["gid"].as_array().unwrap().len(), 4);
    assert!(p["exe"].as_str().unwrap().starts_with('/'));
    assert!(p["fds"].as_array().unwrap().iter().any(|f| f["fd"] == 0 && f["flags"].is_string()));
    assert!(!p["maps"].as_array().unwrap().is_empty());
}

#[test]
fn scan_sysctl_reads_proc_sys_and_refuses_other_names() {
    let answer = scan(json!({"sysctl": ["kernel.ostype", "../etc/passwd", "fs.no_such_knob"]}));
    assert_eq!(answer["sysctl"]["kernel.ostype"], "Linux");
    assert_eq!(answer["sysctl"]["../etc/passwd"]["error"], "bad-name");
    assert_eq!(answer["sysctl"]["fs.no_such_knob"]["error"], "missing");
}

#[test]
fn scan_fails_rather_than_truncates() {
    reset();
    let root = root_dir().to_str().unwrap().to_string();
    for i in 0..10 {
        fs::write(root_dir().join(format!("f{i}")), b"").unwrap();
    }
    let e = crate::scan::scan(&json!({"walks": [{"root": root, "depth": 1}], "max_entries": 5})).unwrap_err();
    assert_eq!(e.kind, "too-many");
    let e = crate::scan::scan(&json!({"walks": [{"root": root, "depth": 1}], "max_bytes": 100})).unwrap_err();
    assert_eq!(e.kind, "too-large");
    let ok = scan(json!({"walks": [{"root": root, "depth": 1}]}));
    assert_eq!(ok["protocol"], 1);
    assert!(ok["uname"].as_str().unwrap().starts_with("Linux "));
}

// --- walk completeness and byte paths (TR-P2 r3: H1, M1) ---------------------

fn walk_ok(args: serde_json::Value) -> serde_json::Value {
    ops::walk(&args).unwrap()
}

fn listed(answer: &serde_json::Value, key: &str) -> Vec<String> {
    answer[key].as_array().unwrap().iter().map(|e| e["path"].as_str().unwrap_or("").to_string()).collect()
}

#[test]
fn walk_skips_an_entry_that_vanishes_and_walks_the_rest_of_its_folder() {
    reset();
    let r = root_dir();
    fs::create_dir_all(r.join("churn")).unwrap();
    fs::create_dir_all(r.join("keep/x")).unwrap();
    fs::write(r.join("keep/x/conftest.py"), b"x").unwrap();
    fs::write(r.join("churn/zz-conftest.py"), b"x").unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let churn = {
        let (stop, dir) = (stop.clone(), r.join("churn"));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for i in 0..300 {
                    let _ = fs::write(dir.join(format!("f{i}")), b"");
                }
                for i in 0..300 {
                    let _ = fs::remove_file(dir.join(format!("f{i}")));
                }
            }
        })
    };
    let root = r.to_str().unwrap().to_string();
    let mut vanished = 0;
    let mut failed = Vec::new();
    for _ in 0..200 {
        match ops::walk(&json!({"root": root, "patterns": ["**/conftest.py", "**/zz-*"]})) {
            Ok(a) => {
                let got = listed(&a, "entries");
                assert!(got.contains(&"keep/x/conftest.py".to_string()), "{got:?}");
                assert!(got.contains(&"churn/zz-conftest.py".to_string()), "{got:?}");
                vanished += a["vanished"].as_array().unwrap().len();
            }
            Err(e) => failed.push(e.msg),
        }
    }
    stop.store(true, Ordering::Relaxed);
    churn.join().unwrap();
    assert!(failed.is_empty(), "{} walks failed, e.g. {:?}", failed.len(), failed.first());
    assert!(vanished > 0, "the churn never raced the walk");
}

#[test]
fn walk_hands_back_folders_beyond_its_depth_and_never_fails_on_depth() {
    reset();
    let r = root_dir();
    let mut deep = r.clone();
    for _ in 0..600 {
        deep.push("a");
    }
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("conftest.py"), b"x").unwrap();
    let root = r.to_str().unwrap().to_string();
    // Each walk goes 256 levels and hands back the folder there; walking
    // each handed-back folder in turn reaches the bottom.
    let (mut top, mut levels, mut found) = (root.clone(), 0, Vec::new());
    loop {
        let a = walk_ok(json!({"root": top, "patterns": ["**/conftest.py"]}));
        found.extend(listed(&a, "entries").into_iter().map(|p| format!("{top}/{p}")));
        let deeper = listed(&a, "deeper");
        if deeper.is_empty() {
            break;
        }
        assert_eq!(deeper.len(), 1);
        assert_eq!(deeper[0].split('/').count(), 256);
        assert_eq!(a["deeper"][0]["type"], "dir");
        top = format!("{top}/{}", deeper[0]);
        levels += 256;
    }
    assert_eq!(levels, 512);
    assert_eq!(found, vec![format!("{root}/{}conftest.py", "a/".repeat(600))]);
    fs::remove_dir_all(r.join("a")).unwrap();
}

#[test]
fn paths_that_are_not_text_are_given_and_answered_as_bytes() {
    reset();
    let r = root_dir();
    let odd = r.join(OsStr::from_bytes(b"\xff\xfe"));
    fs::create_dir_all(&odd).unwrap();
    fs::write(odd.join("conftest.py"), b"odd").unwrap();
    let root = r.to_str().unwrap().to_string();
    let b64 = |b: &[u8]| crate::b64::encode(b);
    let mut abs = r.as_os_str().as_bytes().to_vec();
    abs.extend_from_slice(b"/\xff\xfe");
    // The walk answers the name as bytes.
    let a = walk_ok(json!({"root": root, "patterns": ["**/conftest.py"]}));
    assert_eq!(a["entries"][0]["path_b64"], b64(b"\xff\xfe/conftest.py"));
    // scan takes it as bytes in paths, walks and contents.
    let mut file = abs.clone();
    file.extend_from_slice(b"/conftest.py");
    let s = scan(json!({"paths_b64": [b64(&file)], "walks": [{"root_b64": b64(&abs), "depth": 1}], "contents_b64": [b64(&file)]}));
    let rows = s["entries"].as_array().unwrap();
    assert!(rows.iter().any(|e| e["path_b64"] == b64(&file) && e["type"] == "file"));
    assert!(rows.iter().any(|e| e["path_b64"] == b64(&abs) && e["listing"] == true));
    assert_eq!(crate::b64::decode(s["contents_b64"][b64(&file)]["b64"].as_str().unwrap()).unwrap(), b"odd");
    // Every file primitive takes root_b64 and path_b64.
    let e = &a["entries"][0];
    let gone = ops::remove(&json!({"root_b64": b64(&abs), "path": "conftest.py", "expect": {"dev": e["dev"], "ino": e["ino"]}})).unwrap();
    assert_eq!(gone["removed"], 1);
    ops::write(&json!({"root": root, "path_b64": b64(b"\xff\xfe/w"), "data": "w"})).unwrap();
    assert_eq!(fs::read(odd.join("w")).unwrap(), b"w");
    ops::remove(&json!({"root": root, "path_b64": b64(b"\xff\xfe"), "recursive": true})).unwrap();
    assert!(!odd.exists());
    let both = ops::remove(&json!({"root": root, "path": "x", "path_b64": b64(b"x")})).unwrap_err();
    assert_eq!(both.kind, "bad-request");
}
