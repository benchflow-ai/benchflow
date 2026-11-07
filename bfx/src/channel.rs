//! `bfx serve --frames [JOURNAL]`: the exec channel's helper
//! (src/benchflow/sandbox/_exec_channel.py), with the same frames and the
//! same behaviour as the Python helper it replaces.
//!
//! In, from the host: a 4-byte big-endian length, that much JSON
//! (`{"i": id, "a": argv, "d": cwd, "n": stdin bytes or -1, "r": seconds
//! left or null}` or a cancel `{"c": id, "n": -1}`), then the stdin bytes.
//! Out: one kind byte, a 4-byte id, a 4-byte length, the payload. Kinds:
//! R ready (the helper's pid), S starting (after the journal lists the
//! step), O and E output chunks, X exit (the status as text), F not started
//! (the reason).
//!
//! - Each step is the command the per-command `docker exec` would run, as a
//!   child with the environment this helper was given (byte for byte), in
//!   its own session, its folder entered by path, stdin `/dev/null` unless
//!   it has input.
//! - Before a step can exist, `started <id>` is appended to the journal
//!   (fsync'd) and an S frame is sent; a step that fails to start then
//!   gets an F frame.
//! - All input that is there is read before any step starts, so a cancel
//!   already sent withdraws its step; a step whose time left ran out on this
//!   helper's monotonic clock is refused.
//! - The exit status is the child's, `128 + n` when signal n ended it. After
//!   the child exits, output is still read for up to 2 s, as Docker does.
//! - When stdin closes, the helper exits at once; running steps go on.
//!
//! `bfx serve --verdict JOURNAL PID` runs after a channel died: it kills that
//! helper (only while its command line still names JOURNAL), waits until it
//! is gone, then prints `J` and the journal.

use rustix::event::{poll, PollFd, PollFlags, Timespec};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::fs::{self, Mode, OFlags};
use rustix::io::Errno;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

const GRACE: Duration = Duration::from_secs(2);

fn out(kind: u8, id: u32, data: &[u8]) {
    let mut f = Vec::with_capacity(9 + data.len());
    f.push(kind);
    f.extend_from_slice(&id.to_be_bytes());
    f.extend_from_slice(&(data.len() as u32).to_be_bytes());
    f.extend_from_slice(data);
    let stdout = unsafe { BorrowedFd::borrow_raw(1) };
    let mut rest = &f[..];
    while !rest.is_empty() {
        match rustix::io::write(stdout, rest) {
            Ok(n) => rest = &rest[n..],
            Err(Errno::INTR) | Err(Errno::AGAIN) => {}
            // The host is gone: nothing can be reported any more.
            Err(_) => std::process::exit(0),
        }
    }
}

struct Job {
    child: Child,
    pidfd: Option<OwnedFd>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    stdin: Option<ChildStdin>,
    input: Vec<u8>,
    end: Option<Instant>,
    code: Option<i32>,
}

fn nonblocking(fd: BorrowedFd) {
    if let Ok(fl) = fs::fcntl_getfl(fd) {
        let _ = fs::fcntl_setfl(fd, fl | OFlags::NONBLOCK);
    }
}

struct Helper {
    journal: Option<OwnedFd>,
    jobs: BTreeMap<u32, Job>,
    inbuf: Vec<u8>,
}

impl Helper {
    fn start(&mut self, head: &Value, body: Vec<u8>, due: Option<Instant>) {
        let id = head["i"].as_u64().unwrap_or(0) as u32;
        if let Some(due) = due {
            if Instant::now() > due {
                out(b'F', id, b"deadline passed before the step started");
                return;
            }
        }
        if let Some(j) = &self.journal {
            let line = format!("started {id}\n");
            let ok = rustix::io::write(j, line.as_bytes()).map(|n| n == line.len()).unwrap_or(false) && fs::fsync(j).is_ok();
            if !ok {
                out(b'F', id, b"could not record the start");
                return;
            }
        }
        // Said before the step can exist: the host that reads it knows the
        // step may have started.
        out(b'S', id, b"");
        let argv: Vec<&str> = head["a"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        let n = head["n"].as_i64().unwrap_or(-1);
        let Some((prog, args)) = argv.split_first() else {
            out(b'F', id, b"empty argv");
            return;
        };
        let mut c = Command::new(prog);
        c.args(args);
        if let Some(d) = head["d"].as_str() {
            c.current_dir(d);
        }
        c.stdin(if n >= 0 { Stdio::piped() } else { Stdio::null() });
        c.stdout(Stdio::piped()).stderr(Stdio::piped());
        // SAFETY: setsid is a single system call.
        unsafe {
            c.pre_exec(|| rustix::process::setsid().map(|_| ()).map_err(|e| std::io::Error::from_raw_os_error(e.raw_os_error())));
        }
        let mut child = match c.spawn() {
            Ok(ch) => ch,
            Err(e) => {
                let msg = format!("{e}");
                out(b'F', id, &msg.as_bytes()[..msg.len().min(2000)]);
                return;
            }
        };
        let pidfd = rustix::process::Pid::from_raw(child.id() as i32).and_then(|p| rustix::process::pidfd_open(p, rustix::process::PidfdFlags::empty()).ok());
        let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
        let mut stdin = child.stdin.take();
        for f in [stdout.as_ref().map(|x| x.as_fd()), stderr.as_ref().map(|x| x.as_fd()), stdin.as_ref().map(|x| x.as_fd())].into_iter().flatten() {
            nonblocking(f);
        }
        if body.is_empty() {
            stdin = None; // closes it: the command reads EOF
        }
        self.jobs.insert(id, Job { child, pidfd, stdout, stderr, stdin, input: body, end: None, code: None });
    }

    /// Every whole frame in, then the steps no cancel frame withdrew.
    fn take(&mut self) {
        let mut steps = Vec::new();
        let mut cancelled = HashSet::new();
        loop {
            if self.inbuf.len() < 4 {
                break;
            }
            let size = u32::from_be_bytes([self.inbuf[0], self.inbuf[1], self.inbuf[2], self.inbuf[3]]) as usize;
            if self.inbuf.len() < 4 + size {
                break;
            }
            let head: Value = match serde_json::from_slice(&self.inbuf[4..4 + size]) {
                Ok(v) => v,
                // A frame the helper cannot read: the host speaks another
                // protocol; nothing can be trusted to start.
                Err(_) => std::process::exit(2),
            };
            let n = head["n"].as_i64().unwrap_or(-1).max(0) as usize;
            if self.inbuf.len() < 4 + size + n {
                break;
            }
            let body = self.inbuf[4 + size..4 + size + n].to_vec();
            self.inbuf.drain(..4 + size + n);
            if let Some(c) = head.get("c").and_then(|c| c.as_u64()) {
                cancelled.insert(c as u32);
                continue;
            }
            let due = head.get("r").and_then(|r| r.as_f64()).map(|r| Instant::now() + Duration::from_secs_f64(r.max(0.0)));
            steps.push((head, body, due));
        }
        for (head, body, due) in steps {
            let id = head["i"].as_u64().unwrap_or(0) as u32;
            if cancelled.contains(&id) {
                out(b'F', id, b"cancelled before the step started");
            } else {
                self.start(&head, body, due);
            }
        }
    }

    fn reap(&mut self) {
        let now = Instant::now();
        let mut done = Vec::new();
        for (id, j) in self.jobs.iter_mut() {
            if j.code.is_none() {
                if let Ok(Some(st)) = j.child.try_wait() {
                    j.code = Some(st.code().unwrap_or_else(|| 128 + st.signal().unwrap_or(0)));
                    j.end = Some(now);
                    j.pidfd = None;
                }
            }
            if let Some(end) = j.end {
                if (j.stdout.is_none() && j.stderr.is_none()) || now.duration_since(end) >= GRACE {
                    done.push(*id);
                }
            }
        }
        for id in done {
            if let Some(j) = self.jobs.remove(&id) {
                out(b'X', id, j.code.unwrap_or(0).to_string().as_bytes());
            }
        }
    }
}

pub fn frames(journal: Option<&OsStr>) -> ! {
    let journal = journal.map(|p| {
        match fs::open(
            p,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::APPEND | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(fd) => fd,
            Err(_) => std::process::exit(2),
        }
    });
    let stdin = unsafe { BorrowedFd::borrow_raw(0) };
    nonblocking(stdin);
    let mut h = Helper { journal, jobs: BTreeMap::new(), inbuf: Vec::new() };
    out(b'R', 0, std::process::id().to_string().as_bytes());
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let now = Instant::now();
        let mut wait: Option<Duration> = h.jobs.values().filter_map(|j| j.end).map(|e| (e + GRACE).saturating_duration_since(now)).min();
        if h.jobs.values().any(|j| j.code.is_none() && j.pidfd.is_none()) {
            // No pidfd (a kernel before 5.3): look for exits every 10 ms.
            wait = Some(wait.map_or(Duration::from_millis(10), |w| w.min(Duration::from_millis(10))));
        }
        // What to wait on: 0 stdin, then per job its fds.
        let mut tags: Vec<(u32, u8)> = vec![(0, b'H')];
        let mut fds: Vec<PollFd> = vec![PollFd::from_borrowed_fd(stdin, PollFlags::IN)];
        for (id, j) in &h.jobs {
            if let Some(o) = &j.stdout {
                fds.push(PollFd::from_borrowed_fd(o.as_fd(), PollFlags::IN));
                tags.push((*id, b'O'));
            }
            if let Some(e) = &j.stderr {
                fds.push(PollFd::from_borrowed_fd(e.as_fd(), PollFlags::IN));
                tags.push((*id, b'E'));
            }
            if let (Some(i), false) = (&j.stdin, j.input.is_empty()) {
                fds.push(PollFd::from_borrowed_fd(i.as_fd(), PollFlags::OUT));
                tags.push((*id, b'I'));
            }
            if let (Some(p), None) = (&j.pidfd, j.code) {
                fds.push(PollFd::from_borrowed_fd(p.as_fd(), PollFlags::IN));
                tags.push((*id, b'P'));
            }
        }
        let ts = wait.map(|d| Timespec { tv_sec: d.as_secs() as i64, tv_nsec: d.subsec_nanos() as i64 });
        let _ = poll(&mut fds, ts.as_ref());
        let ready: Vec<(u32, u8)> = fds.iter().zip(tags).filter(|(f, _)| !f.revents().is_empty()).map(|(_, t)| t).collect();
        drop(fds);
        for (id, kind) in ready {
            match kind {
                b'H' => {
                    // Read all there is before starting anything, so a
                    // cancel that is already here is seen.
                    let mut got = false;
                    loop {
                        match rustix::io::read(stdin, &mut buf[..]) {
                            Ok(0) => std::process::exit(0),
                            Ok(n) => {
                                h.inbuf.extend_from_slice(&buf[..n]);
                                got = true;
                            }
                            Err(Errno::INTR) => {}
                            Err(Errno::AGAIN) => break,
                            Err(_) => std::process::exit(0),
                        }
                    }
                    if got {
                        h.take();
                    }
                }
                b'I' => {
                    if let Some(j) = h.jobs.get_mut(&id) {
                        let r = j.stdin.as_mut().map(|s| s.write(&j.input[..j.input.len().min(65536)]));
                        match r {
                            Some(Ok(n)) => {
                                j.input.drain(..n);
                            }
                            Some(Err(e)) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::Interrupted => {}
                            _ => j.input.clear(),
                        }
                        if j.input.is_empty() {
                            j.stdin = None;
                        }
                    }
                }
                b'P' => {}
                b'O' | b'E' => {
                    if let Some(j) = h.jobs.get_mut(&id) {
                        let src = if kind == b'O' { j.stdout.as_mut().map(|s| s as &mut dyn Read) } else { j.stderr.as_mut().map(|s| s as &mut dyn Read) };
                        let r = src.map(|s| s.read(&mut buf[..65536]));
                        match r {
                            Some(Ok(0)) | None => {
                                if kind == b'O' {
                                    j.stdout = None;
                                } else {
                                    j.stderr = None;
                                }
                            }
                            Some(Ok(n)) => out(kind, id, &buf[..n]),
                            Some(Err(e)) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::Interrupted => {}
                            Some(Err(_)) => {
                                if kind == b'O' {
                                    j.stdout = None;
                                } else {
                                    j.stderr = None;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        h.reap();
    }
}

fn names_journal(pid: i32, mark: &[u8]) -> bool {
    match fs::open(format!("/proc/{pid}/cmdline").as_str(), OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty()) {
        Ok(fd) => crate::fsx::read_to_end(fd.as_fd(), 1 << 20).map(|c| c.split(|b| *b == 0).any(|a| a == mark)).unwrap_or(false),
        Err(_) => false,
    }
}

pub fn verdict(path: &OsStr, pid: &str) -> i32 {
    let pid: i32 = match pid.parse() {
        Ok(p) => p,
        Err(_) => return 2,
    };
    let mark = path.as_bytes();
    if pid > 1 && names_journal(pid, mark) {
        let _ = crate::sys::kill_one(pid, libc::SIGKILL);
    }
    let end = Instant::now() + Duration::from_secs(10);
    while pid > 1 && names_journal(pid, mark) {
        if Instant::now() > end {
            return 3;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let fd = match fs::open(path, OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC, Mode::empty()) {
        Ok(fd) => fd,
        Err(_) => return 1,
    };
    let data = match crate::fsx::read_to_end(fd.as_fd(), 64 << 20) {
        Ok(d) => d,
        Err(_) => return 1,
    };
    let mut o = std::io::stdout().lock();
    let _ = o.write_all(b"J");
    let _ = o.write_all(&data);
    let _ = o.flush();
    0
}
