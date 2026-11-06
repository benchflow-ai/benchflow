//! `exec`: start a program with an exact environment and identity.
//!
//! `{argv, env={}, uid=0, gid=0, groups=[], cwd?, umask=0o022, rlimits={},
//! nnp=true, caps="launch", setsid=false}`
//!
//! - `argv[0]` is an absolute path; there is no PATH search.
//! - The environment is exactly `env`; nothing of bfx's own is passed on.
//! - `cwd` is either an absolute directory (opened from `/` without following
//!   a link) or `{root, path}` (the path opened beneath the root, never a
//!   link or another mount). The process enters it through the descriptor.
//! - Identity: setgroups(groups), setresgid(gid x3), setresuid(uid x3), each
//!   checked afterwards with getresuid/getresgid/getgroups.
//! - Capabilities, per the agent launch rules in docs/sandboxes/threat-model.md:
//!   `"launch"` removes NET_ADMIN and NET_RAW from the bounding set and
//!   clears the inheritable and ambient sets; `"keep"` clears only the
//!   inheritable and ambient sets. A non-root uid holds no capability after
//!   the switch in either case.
//! - `nnp` sets no_new_privs, so set-id files and file capabilities grant
//!   nothing.
//!
//! One-shot `bfx exec` replaces itself with the program (the caller's pid
//! and stdio are the program's); `serve` starts it as a child and returns its
//! status and output.

use crate::args::{mode_value, Args};
use crate::err::{Error, Result};
use crate::fsx::{self, split_rel};
use crate::sys;
use rustix::fd::{AsFd, OwnedFd};
use rustix::process::{Resource, Rlimit};
use rustix::thread::CapabilitySet;
use serde_json::{json, Value};
use std::ffi::OsString;
use std::io;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Caps {
    Launch,
    Keep,
}

pub struct Spec {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub uid: u32,
    pub gid: u32,
    pub groups: Vec<u32>,
    pub cwd: Option<OwnedFd>,
    pub umask: u32,
    pub rlimits: Vec<(Resource, Option<u64>, Option<u64>)>,
    pub nnp: bool,
    pub caps: Caps,
    pub setsid: bool,
}

fn resource(name: &str) -> Option<Resource> {
    Some(match name {
        "cpu" => Resource::Cpu,
        "fsize" => Resource::Fsize,
        "data" => Resource::Data,
        "stack" => Resource::Stack,
        "core" => Resource::Core,
        "nproc" => Resource::Nproc,
        "nofile" => Resource::Nofile,
        "memlock" => Resource::Memlock,
        "as" => Resource::As,
        _ => return None,
    })
}

pub const EXEC_KEYS: &[&str] = &["argv", "env", "uid", "gid", "groups", "cwd", "umask", "rlimits", "nnp", "caps", "setsid", "resolver"];

pub fn parse(a: &Args) -> Result<Spec> {
    let argv: Vec<String> = a.str_list("argv")?.into_iter().map(String::from).collect();
    match argv.first() {
        Some(p) if p.starts_with('/') => {}
        _ => return Err(Error::bad("argv[0] must be an absolute path")),
    }
    if argv.iter().any(|s| s.contains('\0')) {
        return Err(Error::bad("argv holds a NUL byte"));
    }
    let mut env = Vec::new();
    if let Some(e) = a.opt("env") {
        let m = e.as_object().ok_or_else(|| Error::bad("env must be an object"))?;
        for (k, v) in m {
            let v = v.as_str().ok_or_else(|| Error::bad("env values must be strings"))?;
            if k.is_empty() || k.contains('=') || k.contains('\0') || v.contains('\0') {
                return Err(Error::bad(format!("env name {k:?} or its value is not allowed")));
            }
            env.push((k.clone(), v.to_string()));
        }
    }
    let cwd = match a.opt("cwd") {
        None => None,
        Some(Value::String(p)) => Some(fsx::open_root(p)?.fd),
        Some(v @ Value::Object(_)) => {
            let ca = Args::new(v, &["root", "path"])?;
            let root = fsx::open_root(ca.str("root")?)?;
            let comps = split_rel(ca.opt_str("path")?.unwrap_or("").as_bytes())?;
            Some(fsx::open_dir(&root, &comps)?.fd)
        }
        Some(_) => return Err(Error::bad("cwd must be a path or {root, path}")),
    };
    let mut rlimits = Vec::new();
    if let Some(r) = a.opt("rlimits") {
        let m = r.as_object().ok_or_else(|| Error::bad("rlimits must be an object"))?;
        for (k, v) in m {
            let res = resource(k).ok_or_else(|| Error::bad(format!("unknown rlimit {k:?}")))?;
            let pair = v.as_array().filter(|p| p.len() == 2).ok_or_else(|| Error::bad("an rlimit is [soft, hard]"))?;
            let lim = |x: &Value| -> Result<Option<u64>> {
                if x.is_null() {
                    Ok(None)
                } else {
                    x.as_u64().map(Some).ok_or_else(|| Error::bad("an rlimit value is an integer or null"))
                }
            };
            rlimits.push((res, lim(&pair[0])?, lim(&pair[1])?));
        }
    }
    let caps = match a.opt_str("caps")?.unwrap_or("launch") {
        "launch" => Caps::Launch,
        "keep" => Caps::Keep,
        _ => return Err(Error::bad("caps must be launch or keep")),
    };
    Ok(Spec {
        argv,
        env,
        uid: a.opt_u32("uid")?.unwrap_or(0),
        gid: a.opt_u32("gid")?.unwrap_or(0),
        groups: a.u32_list("groups")?,
        cwd,
        umask: a.opt("umask").map(|m| mode_value(m, "umask")).transpose()?.unwrap_or(0o022),
        rlimits,
        nnp: a.bool("nnp", true)?,
        caps,
        setsid: a.bool("setsid", false)?,
    })
}

fn errno(e: rustix::io::Errno) -> io::Error {
    io::Error::from_raw_os_error(e.raw_os_error())
}

/// Status line field reading for the checks below.
fn proc_ids() -> Vec<u32> {
    let mut g = rustix::process::getgroups().unwrap_or_default().iter().map(|g| g.as_raw()).collect::<Vec<_>>();
    g.sort_unstable();
    g.dedup();
    g
}

/// The identity, capability and resource steps, in the process that will
/// exec the program (the forked child, or bfx itself for one-shot exec).
/// Single-threaded at this point, so the per-thread credential calls set
/// the whole process.
pub fn prepare(s: &PrepareData) -> io::Result<()> {
    if let Some(fd) = &s.cwd {
        rustix::process::fchdir(fd.as_fd()).map_err(errno)?;
    }
    rustix::process::umask(rustix::fs::Mode::from_raw_mode(s.umask));
    for (res, soft, hard) in &s.rlimits {
        rustix::process::setrlimit(*res, Rlimit { current: *soft, maximum: *hard }).map_err(errno)?;
    }
    if s.setsid {
        rustix::process::setsid().map_err(errno)?;
    }
    if s.caps == Caps::Launch {
        for cap in [CapabilitySet::NET_ADMIN, CapabilitySet::NET_RAW] {
            if rustix::thread::capability_is_in_bounding_set(cap).map_err(errno)? {
                rustix::thread::remove_capability_from_bounding_set(cap).map_err(errno)?;
            }
        }
    }
    let mut want = s.groups.clone();
    want.sort_unstable();
    want.dedup();
    let already = sys::getresuid() == (s.uid, s.uid, s.uid) && sys::getresgid() == (s.gid, s.gid, s.gid) && proc_ids() == want;
    // Only an identity that differs is switched (an unprivileged caller can
    // still run as itself); either way it is checked afterwards.
    if !already {
        let groups: Vec<rustix::process::Gid> = s.groups.iter().map(|g| rustix::process::Gid::from_raw_unchecked(*g)).collect();
        rustix::thread::set_thread_groups(&groups).map_err(errno)?;
        let g = rustix::process::Gid::from_raw_unchecked(s.gid);
        rustix::thread::set_thread_res_gid(g, g, g).map_err(errno)?;
        let u = rustix::process::Uid::from_raw_unchecked(s.uid);
        rustix::thread::set_thread_res_uid(u, u, u).map_err(errno)?;
    }
    let denied = || io::Error::from_raw_os_error(libc::EPERM);
    if sys::getresuid() != (s.uid, s.uid, s.uid) || sys::getresgid() != (s.gid, s.gid, s.gid) {
        return Err(denied());
    }
    if proc_ids() != want {
        return Err(denied());
    }
    // Before 4.3 there is no ambient set to clear.
    match rustix::thread::clear_ambient_capability_set() {
        Ok(()) | Err(rustix::io::Errno::INVAL) => {}
        Err(e) => return Err(errno(e)),
    }
    let mut sets = rustix::thread::capabilities(None).map_err(errno)?;
    sets.inheritable = CapabilitySet::empty();
    rustix::thread::set_capabilities(None, sets).map_err(errno)?;
    if s.nnp {
        rustix::thread::set_no_new_privs(true).map_err(errno)?;
    }
    Ok(())
}

/// What `prepare` needs, owned by the closure.
pub struct PrepareData {
    pub cwd: Option<OwnedFd>,
    pub umask: u32,
    pub rlimits: Vec<(Resource, Option<u64>, Option<u64>)>,
    pub setsid: bool,
    pub caps: Caps,
    pub groups: Vec<u32>,
    pub uid: u32,
    pub gid: u32,
    pub nnp: bool,
}

pub fn command(s: Spec) -> Command {
    let mut c = Command::new(&s.argv[0]);
    c.args(&s.argv[1..]);
    c.env_clear();
    for (k, v) in &s.env {
        c.env(OsString::from_vec(k.clone().into_bytes()), OsString::from_vec(v.clone().into_bytes()));
    }
    let data = PrepareData {
        cwd: s.cwd,
        umask: s.umask,
        rlimits: s.rlimits,
        setsid: s.setsid,
        caps: s.caps,
        groups: s.groups,
        uid: s.uid,
        gid: s.gid,
        nnp: s.nnp,
    };
    // SAFETY: `prepare` makes only system calls and reads data the closure
    // owns; nothing in it allocates after fork except error values on
    // failure, and bfx never runs threads that could hold a lock.
    unsafe {
        c.pre_exec(move || prepare(&data));
    }
    c
}

/// One-shot: become the program. Returns only on failure.
pub fn exec_now(v: &Value) -> Error {
    let r = Args::new(v, EXEC_KEYS).and_then(|a| parse(&a));
    match r {
        Err(e) => e,
        Ok(s) => {
            let argv0 = s.argv[0].clone();
            let err = command(s).exec();
            Error::io(err, format!("exec {argv0}"))
        }
    }
}

/// Serve: run the program to its end, with optional stdin bytes and a
/// timeout (its process group is killed when it runs out).
pub fn run(v: &Value) -> Result<Value> {
    let mut keys = EXEC_KEYS.to_vec();
    keys.extend(["stdin_b64", "timeout_ms"]);
    let a = Args::new(v, &keys)?;
    let mut s = parse(&a)?;
    let stdin = a.opt_str("stdin_b64")?.map(|b| crate::b64::decode(b).ok_or_else(|| Error::bad("stdin_b64 is not base64"))).transpose()?;
    let timeout = a.opt_u64("timeout_ms")?.map(Duration::from_millis);
    s.setsid = true;
    let argv0 = s.argv[0].clone();
    let mut c = command(s);
    c.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
    c.stdout(Stdio::piped());
    c.stderr(Stdio::piped());
    let out = spawn_capture(c, stdin, timeout, 64 << 20).map_err(|e| Error::io(e, format!("exec {argv0}")))?;
    Ok(json!({
        "status": out.status,
        "timed_out": out.timed_out,
        "stdout_b64": crate::b64::encode(&out.stdout),
        "stderr_b64": crate::b64::encode(&out.stderr),
    }))
}

pub struct Captured {
    pub status: i32,
    pub timed_out: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Start *c* (its own session), feed it *stdin*, read both outputs up to
/// *limit* bytes each, and wait for it; on *timeout*, kill its group.
/// The status is the exit code, or 128 + n for signal n.
pub fn spawn_capture(mut c: Command, stdin: Option<Vec<u8>>, timeout: Option<Duration>, limit: usize) -> io::Result<Captured> {
    use rustix::event::{poll, PollFd, PollFlags};
    use std::io::Write;
    let mut child = c.spawn()?;
    let pid = child.id() as i32;
    let start = Instant::now();
    let mut input = child.stdin.take();
    let mut data = stdin.unwrap_or_default();
    let mut out = child.stdout.take();
    let mut err = child.stderr.take();
    for f in [out.as_ref().map(|o| o.as_fd()), err.as_ref().map(|e| e.as_fd()), input.as_ref().map(|i| i.as_fd())].into_iter().flatten() {
        let fl = rustix::fs::fcntl_getfl(f).map_err(errno)?;
        rustix::fs::fcntl_setfl(f, fl | rustix::fs::OFlags::NONBLOCK).map_err(errno)?;
    }
    if data.is_empty() {
        input = None;
    }
    let (mut ob, mut eb) = (Vec::new(), Vec::new());
    let mut timed_out = false;
    let mut buf = vec![0u8; 1 << 16];
    while out.is_some() || err.is_some() || input.is_some() {
        let left = match timeout {
            Some(t) => {
                let e = start.elapsed();
                if e >= t {
                    timed_out = true;
                    let _ = sys::kill_group(pid, libc::SIGKILL);
                    break;
                }
                Some(t - e)
            }
            None => None,
        };
        let mut fds = Vec::new();
        let mut which = Vec::new();
        if let Some(o) = &out {
            fds.push(PollFd::new(o, PollFlags::IN));
            which.push(1);
        }
        if let Some(e) = &err {
            fds.push(PollFd::new(e, PollFlags::IN));
            which.push(2);
        }
        if let Some(i) = &input {
            fds.push(PollFd::new(i, PollFlags::OUT));
            which.push(0);
        }
        let ts = left.map(|d| rustix::event::Timespec { tv_sec: d.as_secs() as i64, tv_nsec: d.subsec_nanos() as i64 });
        match poll(&mut fds, ts.as_ref()) {
            Ok(_) | Err(rustix::io::Errno::INTR) => {}
            Err(e) => return Err(errno(e)),
        }
        let ready: Vec<(i32, PollFlags)> = fds.iter().zip(&which).map(|(f, w)| (*w, f.revents())).collect();
        drop(fds);
        for (w, ev) in ready {
            if ev.is_empty() {
                continue;
            }
            match w {
                0 => {
                    let i = input.as_mut().unwrap();
                    match i.write(&data) {
                        Ok(n) => {
                            data.drain(..n);
                            if data.is_empty() {
                                input = None;
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                        Err(_) => input = None,
                    }
                }
                1 => pump(&mut out, &mut ob, &mut buf, limit),
                2 => pump(&mut err, &mut eb, &mut buf, limit),
                _ => {}
            }
        }
    }
    let status = child.wait()?;
    use std::os::unix::process::ExitStatusExt;
    let code = status.code().unwrap_or_else(|| 128 + status.signal().unwrap_or(0));
    Ok(Captured { status: code, timed_out, stdout: ob, stderr: eb })
}

/// Read what is there from *src* into *dst* (up to *limit* kept); at EOF or
/// an error the stream is dropped.
fn pump<R: std::io::Read>(src: &mut Option<R>, dst: &mut Vec<u8>, buf: &mut [u8], limit: usize) {
    let Some(r) = src.as_mut() else { return };
    match r.read(buf) {
        Ok(0) => *src = None,
        Ok(n) => {
            if dst.len() < limit {
                let take = n.min(limit - dst.len());
                dst.extend_from_slice(&buf[..take]);
            }
        }
        Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::Interrupted => {}
        Err(_) => *src = None,
    }
}
