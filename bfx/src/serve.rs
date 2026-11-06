//! Command dispatch for one-shot use and for `bfx serve` (JSON lines).
//!
//! Protocol version 1. `serve [--journal PATH]` reads one JSON object per
//! line and answers each request by its id. Answers come in request order,
//! with one exception: a refusal found while a line is parsed (a bad line,
//! an unknown protocol version) is written at once, before the answers of
//! requests read earlier that have not run yet.
//!
//! - A request is `{"id": n, "cmd": name, "args": {...}, "v": 1?, "r": s?}`
//!   with a numeric id (a non-negative integer) and, optionally, the seconds
//!   its caller has left. A cancel is `{"cancel": n}`.
//! - Before a command other than `version` starts, bfx appends `started <n>`
//!   to the journal (fsync'd; the file is created new, root's 0600, never
//!   through a link) and writes `{"id": n, "started": true}`. Then the answer:
//!   `{"id": n, "ok": true, "result": {..}}` or `{"id": n, "ok": false,
//!   "error": {kind, message, errno?, detail?}}`.
//! - All input already there is read before a command starts, so a cancel
//!   sent while its request waited withdraws it; a request whose time left
//!   ran out on bfx's monotonic clock is refused. Both answer
//!   `{"ok": false, "error": {"kind": "cancelled" | "deadline", ..},
//!   "started": false}` and nothing ran.
//! - `r` is checked only before a command starts. A command that has
//!   started (a walk, tar-out, scan, kill) runs to its end even past its
//!   caller's deadline, and holds the session meanwhile; the caller that
//!   stopped waiting treats its outcome as ambiguous (fail-closed).
//! - When the channel dies, a request whose started line or journal entry
//!   exists may or may not have run: the host never runs it a second time.
//!
//! Byte payloads travel as base64 (`tar-in` `args.data_b64`, `tar-out`
//! `result.data_b64`, `exec`'s stdin and output). Lines are bounded
//! (MAX_LINE); in serve, `exec` runs the program to its end as a child.

use crate::err::{Error, Result};
use crate::{exec, fsx, kill, ops, scan, tar};
use rustix::event::{poll, PollFd, PollFlags};
use rustix::fd::{BorrowedFd, OwnedFd};
use rustix::fs::{self, Mode, OFlags};
use rustix::io::Errno;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::ffi::OsStr;
use std::io::Write;
use std::time::{Duration, Instant};

pub const PROTOCOL: u64 = 1;
pub const MAX_LINE: usize = 256 << 20;
/// The largest tar-out a serve answer carries (it is held in memory).
pub const SERVE_TAR_MAX: u64 = 128 << 20;

pub const COMMANDS: &[&str] = &["version", "exec", "kill", "tar-out", "tar-in", "mkdir", "chown", "chmod", "clear", "remove", "write", "walk", "stat", "scan"];

/// `"resolver": "walk"` in any request forces the component walk (tests).
pub fn set_resolver(args: &Value) -> Result<()> {
    match args.get("resolver") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "walk" => fsx::FORCE_WALK.store(true, std::sync::atomic::Ordering::Relaxed),
        Some(Value::String(s)) if s == "auto" => fsx::FORCE_WALK.store(false, std::sync::atomic::Ordering::Relaxed),
        Some(_) => return Err(Error::bad("resolver must be auto or walk")),
    }
    Ok(())
}

pub fn version() -> Value {
    json!({
        "name": "bfx",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": PROTOCOL,
        "arch": std::env::consts::ARCH,
        "pid": std::process::id(),
        "sha256": crate::self_sha256(),
        "resolver": fsx::resolver().name(),
        "procfs": fsx::have_proc(),
        "commands": COMMANDS,
    })
}

/// Run one command whose input and output are JSON values (tar data as
/// base64).
pub fn dispatch(cmd: &str, args: &Value) -> Result<Value> {
    set_resolver(args)?;
    let r = match cmd {
        "version" => Ok(version()),
        "exec" => exec::run(args),
        "kill" => kill::kill(args),
        "mkdir" => ops::mkdir(args),
        "chown" => ops::chown(args),
        "chmod" => ops::chmod(args),
        "clear" => ops::clear(args),
        "remove" => ops::remove(args),
        "write" => ops::write(args),
        "walk" => ops::walk(args),
        "stat" => ops::stat(args),
        "scan" => scan::scan(args),
        "tar-in" => {
            let data = args.get("data_b64").and_then(|d| d.as_str()).ok_or_else(|| Error::bad("data_b64 is required in serve"))?;
            let bytes = crate::b64::decode(data).ok_or_else(|| Error::bad("data_b64 is not base64"))?;
            tar::tar_in(args, std::io::Cursor::new(bytes))
        }
        "tar-out" => {
            let (mut v, buf) = tar::tar_out_bounded(args, Vec::new(), Some(SERVE_TAR_MAX))?;
            v["data_b64"] = json!(crate::b64::encode(&buf));
            Ok(v)
        }
        _ => Err(Error::bad(format!("unknown command {cmd:?}"))),
    };
    fsx::FORCE_WALK.store(false, std::sync::atomic::Ordering::Relaxed);
    r
}

pub fn answer(id: Value, r: Result<Value>) -> Value {
    match r {
        Ok(v) => json!({"id": id, "ok": true, "result": v}),
        Err(e) => json!({"id": id, "ok": false, "error": e.to_json()}),
    }
}

fn not_started(id: u64, kind: &'static str, why: &str) -> Value {
    json!({"id": id, "ok": false, "started": false, "error": {"kind": kind, "message": why}})
}

struct Input {
    buf: Vec<u8>,
    eof: bool,
}

impl Input {
    /// Read what is there (waiting up to *wait* for the first bytes).
    fn fill(&mut self, wait: Option<Duration>) {
        let stdin = unsafe { BorrowedFd::borrow_raw(0) };
        let mut fds = [PollFd::from_borrowed_fd(stdin, PollFlags::IN)];
        let ts = wait.map(|d| rustix::event::Timespec { tv_sec: d.as_secs() as i64, tv_nsec: d.subsec_nanos() as i64 });
        match poll(&mut fds, ts.as_ref()) {
            Ok(0) => return,
            Ok(_) => {}
            Err(_) => return,
        }
        let mut chunk = vec![0u8; 1 << 20];
        loop {
            match rustix::io::read(stdin, &mut chunk[..]) {
                Ok(0) => {
                    self.eof = true;
                    return;
                }
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(Errno::INTR) => {}
                Err(Errno::AGAIN) => return,
                Err(_) => {
                    self.eof = true;
                    return;
                }
            }
        }
    }

    fn line(&mut self) -> Option<std::result::Result<Vec<u8>, ()>> {
        match self.buf.iter().position(|b| *b == b'\n') {
            Some(i) => {
                let l = self.buf[..i].to_vec();
                self.buf.drain(..=i);
                Some(Ok(l))
            }
            None if self.buf.len() > MAX_LINE => Some(Err(())),
            None => None,
        }
    }
}

struct Request {
    id: u64,
    cmd: String,
    args: Value,
    due: Option<Instant>,
}

fn write_line(v: &Value) -> bool {
    let mut text = serde_json::to_vec(v).unwrap_or_else(|_| b"{}".to_vec());
    text.push(b'\n');
    let mut out = std::io::stdout().lock();
    out.write_all(&text).and_then(|_| out.flush()).is_ok()
}

fn open_journal(path: &OsStr) -> Option<OwnedFd> {
    fs::open(
        path,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::APPEND | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .ok()
}

fn record(journal: &Option<OwnedFd>, id: u64) -> bool {
    match journal {
        None => true,
        Some(j) => {
            let line = format!("started {id}\n");
            rustix::io::write(j, line.as_bytes()).map(|n| n == line.len()).unwrap_or(false) && fs::fsync(j).is_ok()
        }
    }
}

pub fn serve(journal_path: Option<&OsStr>) -> i32 {
    let journal = match journal_path {
        None => None,
        Some(p) => match open_journal(p) {
            Some(fd) => Some(fd),
            None => {
                let _ = std::io::stderr().write_all(b"bfx serve: cannot create the journal\n");
                return 2;
            }
        },
    };
    let stdin = unsafe { BorrowedFd::borrow_raw(0) };
    if let Ok(fl) = fs::fcntl_getfl(stdin) {
        let _ = fs::fcntl_setfl(stdin, fl | OFlags::NONBLOCK);
    }
    let mut input = Input { buf: Vec::new(), eof: false };
    let mut queue: VecDeque<Request> = VecDeque::new();
    let mut cancelled: HashSet<u64> = HashSet::new();
    loop {
        // Everything already there, before anything starts.
        input.fill(if queue.is_empty() && !input.eof { None } else { Some(Duration::ZERO) });
        loop {
            let line = match input.line() {
                None => break,
                Some(Err(())) => {
                    write_line(&answer(Value::Null, Err(Error::bad("a request line is longer than allowed"))));
                    return 2;
                }
                Some(Ok(l)) => l,
            };
            if line.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            let req: Value = match serde_json::from_slice(&line) {
                Ok(v) => v,
                Err(e) => {
                    if !write_line(&answer(Value::Null, Err(Error::bad(format!("not a JSON line: {e}"))))) {
                        return 1;
                    }
                    continue;
                }
            };
            if let Some(c) = req.get("cancel").and_then(|c| c.as_u64()) {
                cancelled.insert(c);
                continue;
            }
            let Some(id) = req.get("id").and_then(|i| i.as_u64()) else {
                if !write_line(&answer(req.get("id").cloned().unwrap_or(Value::Null), Err(Error::bad("id must be a non-negative integer")))) {
                    return 1;
                }
                continue;
            };
            if let Some(v) = req.get("v") {
                if v.as_u64() != Some(PROTOCOL) {
                    write_line(&not_started(id, "bad-request", &format!("protocol {v} is not {PROTOCOL}")));
                    continue;
                }
            }
            let due = req.get("r").and_then(|r| r.as_f64()).map(|r| Instant::now() + Duration::from_secs_f64(r.max(0.0)));
            let cmd = req.get("cmd").and_then(|c| c.as_str()).unwrap_or("").to_string();
            let args = req.get("args").cloned().unwrap_or_else(|| json!({}));
            queue.push_back(Request { id, cmd, args, due });
        }
        let Some(r) = queue.pop_front() else {
            if input.eof {
                return 0;
            }
            continue;
        };
        let ok = if cancelled.remove(&r.id) {
            write_line(&not_started(r.id, "cancelled", "the caller withdrew the request before it started"))
        } else if r.due.is_some_and(|d| Instant::now() > d) {
            write_line(&not_started(r.id, "deadline", "deadline passed before the request started"))
        } else if r.cmd == "version" {
            write_line(&answer(json!(r.id), dispatch(&r.cmd, &r.args)))
        } else if !COMMANDS.contains(&r.cmd.as_str()) {
            write_line(&not_started(r.id, "bad-request", &format!("unknown command {:?}", r.cmd)))
        } else if !record(&journal, r.id) {
            write_line(&not_started(r.id, "journal", "could not record the start"))
        } else {
            write_line(&json!({"id": r.id, "started": true})) && write_line(&answer(json!(r.id), dispatch(&r.cmd, &r.args)))
        };
        if !ok {
            return 1;
        }
    }
}

/// `--json -`: the arguments are one line on stdin (so secrets never
/// appear in argv). Reads byte by byte up to the newline, leaving the rest
/// of stdin to the command (`exec`'s program, `tar-in`'s archive).
pub fn read_args_line() -> std::result::Result<Vec<u8>, &'static str> {
    let stdin = unsafe { BorrowedFd::borrow_raw(0) };
    let mut out = Vec::new();
    let mut b = [0u8; 1];
    loop {
        match rustix::io::read(stdin, &mut b) {
            Ok(0) => return if out.is_empty() { Err("no arguments on stdin") } else { Ok(out) },
            Ok(_) if b[0] == b'\n' => return Ok(out),
            Ok(_) => {
                out.push(b[0]);
                if out.len() > MAX_LINE {
                    return Err("the arguments line is longer than allowed");
                }
            }
            Err(Errno::INTR) | Err(Errno::AGAIN) => {}
            Err(_) => return Err("cannot read stdin"),
        }
    }
}
