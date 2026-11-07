//! `kill`: stop every process of the given accounts and prove none is left
//! (K1's kill, which keeps nothing). This is the semantics of lockdown.py's
//! `_KILL_PRELUDE` and `_KILL_SANDBOX_USER_PROCS_BODY`, with bfx itself doing
//! what setpriv or su did there (acting as the account, `asuser`). K2's
//! keeping kill is not here: in 0.8.3 it runs only on today's route,
//! and a request naming `keep_ports` or `wait` is refused.
//!
//! `{accounts: [{uid, gid}]}`
//!
//! - A process is an account's when its real, effective or saved UID is the
//!   account's. A zombie counts only while its thread group still runs
//!   (State Z with Threads above 1).
//! - Plain kill: freeze (kill(-1, SIGSTOP) as each account, then per-pid
//!   SIGSTOP pass after pass until a pass finds nothing new), SIGTERM and
//!   SIGCONT to the frozen set with up to 1 s to exit, then up to 8 rounds of
//!   freeze, SIGKILL to the frozen set and kill(-1, SIGKILL) as each account.
//! - Per-pid signals to a process whose real or saved UID is the account's
//!   go as the account (a pid root's process took meanwhile answers EPERM);
//!   only a process whose effective UID alone is the account's gets them
//!   from root.
//! - The answer carries the receipt line the shell printed (`BF killed <n>
//!   end`) and the same fact as a field.

use crate::args::Args;
use crate::err::{Error, Result};
use crate::fsx::read_to_end;
use crate::{asuser, sys};
use rustix::fd::AsFd;
use rustix::fs::{self, Mode, OFlags};
use rustix::io::Errno;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::thread::sleep;
use std::time::Duration;

#[derive(Clone, Copy)]
struct Account {
    uid: u32,
    gid: u32,
}

#[derive(Clone, Copy, Debug)]
struct Status {
    state: u8,
    uids: [u32; 4],
    threads: u32,
}

fn read_file(path: &str, limit: usize) -> std::result::Result<Vec<u8>, Errno> {
    let fd = fs::open(path, OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW, Mode::empty())?;
    read_to_end(fd.as_fd(), limit).map_err(|e| e.errno.map(Errno::from_raw_os_error).unwrap_or(Errno::IO))
}

/// A process's State, Uid and Threads lines; None when it is gone.
fn status(pid: i32) -> Option<std::result::Result<Status, Errno>> {
    let text = match read_file(&format!("/proc/{pid}/status"), 1 << 16) {
        Ok(t) => t,
        Err(Errno::NOENT) | Err(Errno::SRCH) => return None,
        Err(e) => return Some(Err(e)),
    };
    let mut st = Status { state: 0, uids: [u32::MAX; 4], threads: 0 };
    let (mut has_state, mut has_uid) = (false, false);
    for line in String::from_utf8_lossy(&text).lines() {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("State:") => {
                if let Some(s) = f.next() {
                    st.state = s.as_bytes()[0];
                    has_state = true;
                }
            }
            Some("Uid:") => {
                let v: Vec<u32> = f.filter_map(|x| x.parse().ok()).collect();
                if v.len() >= 4 {
                    st.uids = [v[0], v[1], v[2], v[3]];
                    has_uid = true;
                }
            }
            Some("Threads:") => st.threads = f.next().and_then(|x| x.parse().ok()).unwrap_or(0),
            _ => {}
        }
    }
    if !has_state || !has_uid {
        // A status with no State or Uid: the process went away mid-read.
        return None;
    }
    Some(Ok(st))
}

fn command_line(pid: i32) -> String {
    let raw = read_file(&format!("/proc/{pid}/cmdline"), 1 << 16).unwrap_or_default();
    let s: String = raw.iter().map(|b| match b {
        0 => ' ',
        b'\n' => '?',
        b => *b as char,
    }).collect();
    s.chars().take(200).collect()
}

fn pids() -> Result<Vec<i32>> {
    let fd = fs::open("/proc", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty())
        .map_err(|e| Error::sys(e, "Process table is unavailable"))?;
    let mut d = fs::Dir::read_from(&fd).map_err(|e| Error::sys(e, "Process table is unavailable"))?;
    let mut out = Vec::new();
    while let Some(e) = d.read() {
        let e = e.map_err(|e| Error::sys(e, "reading /proc"))?;
        if let Ok(n) = std::str::from_utf8(e.file_name().to_bytes()).unwrap_or("").parse::<i32>() {
            out.push(n);
        }
    }
    Ok(out)
}

/// A live process of the accounts: `owner` is the account to signal it as,
/// or None when only its effective UID is the account's (root signals it).
#[derive(Clone, Copy, Debug)]
struct Live {
    pid: i32,
    owner: Option<u32>,
}

struct K {
    accounts: Vec<Account>,
    uids: HashSet<u32>,
    receipt: std::cell::RefCell<Vec<String>>,
}

impl K {
    fn receipt_line(&self, line: String) {
        self.receipt.borrow_mut().push(line);
    }

    fn gid_of(&self, uid: u32) -> u32 {
        self.accounts.iter().find(|a| a.uid == uid).map(|a| a.gid).unwrap_or(uid)
    }

    fn role(&self, st: &Status) -> Option<Option<u32>> {
        let [r, e, s, _] = st.uids;
        if self.uids.contains(&r) {
            Some(Some(r))
        } else if self.uids.contains(&s) {
            Some(Some(s))
        } else if self.uids.contains(&e) {
            Some(None)
        } else {
            None
        }
    }

    fn live(&self) -> Result<Vec<Live>> {
        let mut out = Vec::new();
        for pid in pids()? {
            let st = match status(pid) {
                None => continue,
                Some(Ok(st)) => st,
                Some(Err(e)) => {
                    if fs::stat(format!("/proc/{pid}").as_str()).is_ok() {
                        return Err(Error::sys(e, format!("Cannot inspect process {pid}")).with_kind("unobservable"));
                    }
                    continue;
                }
            };
            let Some(owner) = self.role(&st) else { continue };
            if st.state != b'Z' || st.threads > 1 {
                out.push(Live { pid, owner });
            }
        }
        Ok(out)
    }

    fn mine(&self) -> Result<Vec<i32>> {
        Ok(self.live()?.iter().map(|l| l.pid).collect())
    }

    /// kill(-1, sig) as every account; each must be proven delivered.
    fn everyone(&self, sig: i32) -> Result<()> {
        for a in &self.accounts {
            let r = asuser::run(a.uid, a.gid, &[json!({"broadcast": sig})])?;
            if r[0]["delivered"] != json!(true) {
                return Err(Error::new(
                    "cannot-signal",
                    format!("Cannot send signal {sig} as UID {} GID {}: {}", a.uid, a.gid, r[0]["error"].as_str().unwrap_or("kill(-1) failed")),
                ));
            }
        }
        Ok(())
    }

    fn each(&self, sig: i32, list: &[Live]) -> Result<()> {
        let mut by: BTreeMap<u32, Vec<i32>> = BTreeMap::new();
        for l in list {
            match l.owner {
                None => {
                    let _ = sys::kill_one(l.pid, sig);
                }
                Some(u) => by.entry(u).or_default().push(l.pid),
            }
        }
        for (u, ps) in by {
            asuser::run(u, self.gid_of(u), &[json!({"signal": sig, "pids": ps})])?;
        }
        Ok(())
    }
}

trait WithKind {
    fn with_kind(self, k: &'static str) -> Self;
}

impl WithKind for Error {
    fn with_kind(mut self, k: &'static str) -> Self {
        self.kind = k;
        self
    }
}

fn survivors(pids: &[i32]) -> Error {
    let remain: Vec<Value> = pids.iter().map(|p| json!({"pid": p, "command": command_line(*p)})).collect();
    let names: String = pids.iter().map(|p| format!(" {p} ({})", command_line(*p))).collect();
    Error::new("survivors", format!("Sandbox-user writers remain:{names}")).with_detail(json!({"remain": remain}))
}

const STOP: i32 = libc::SIGSTOP;
const CONT: i32 = libc::SIGCONT;
const TERM: i32 = libc::SIGTERM;
const KILL: i32 = libc::SIGKILL;

fn tick() {
    sleep(Duration::from_millis(100));
}

fn plain(k: &mut K) -> Result<u64> {
    let mut killed: BTreeSet<i32> = BTreeSet::new();
    let freeze = |k: &K, killed: &mut BTreeSet<i32>| -> Result<Vec<Live>> {
        k.everyone(STOP)?;
        let mut stopped = HashSet::new();
        for _ in 0..100 {
            let new: Vec<Live> = k.live()?.into_iter().filter(|l| !stopped.contains(&l.pid)).collect();
            if new.is_empty() {
                break;
            }
            k.each(STOP, &new)?;
            stopped.extend(new.iter().map(|l| l.pid));
        }
        let found = k.live()?;
        killed.extend(found.iter().map(|l| l.pid));
        Ok(found)
    };
    let found = freeze(k, &mut killed)?;
    k.each(TERM, &found)?;
    k.each(CONT, &found)?;
    k.everyone(CONT)?;
    if !found.is_empty() {
        for _ in 0..10 {
            if k.live()?.is_empty() {
                break;
            }
            tick();
        }
    }
    for _ in 0..8 {
        let found = freeze(k, &mut killed)?;
        k.each(KILL, &found)?;
        k.everyone(KILL)?;
        if k.live()?.is_empty() {
            return Ok(killed.len() as u64);
        }
        sleep(Duration::from_secs(1));
    }
    let left = k.mine()?;
    if left.is_empty() {
        return Ok(killed.len() as u64);
    }
    Err(survivors(&left))
}

pub fn kill(v: &Value) -> Result<Value> {
    let a = Args::new(v, &["accounts"])?;
    let mut accounts = Vec::new();
    for x in a.arr("accounts")? {
        let xa = Args::new(x, &["uid", "gid"])?;
        let (uid, gid) = (xa.u32("uid")?, xa.u32("gid")?);
        if uid == 0 {
            return Err(Error::new("invalid-account", "Invalid sandbox UID: root's processes are never killed"));
        }
        accounts.push(Account { uid, gid });
    }
    if accounts.is_empty() {
        return Err(Error::bad("accounts must name at least one account"));
    }
    if sys::getresuid().1 != 0 {
        return Err(Error::new("not-root", "The kill before grading is not running as root"));
    }
    if read_file("/proc/self/status", 1 << 16).is_err() {
        return Err(Error::new("unobservable", "Process table is unavailable"));
    }
    let mut k = K { uids: accounts.iter().map(|a| a.uid).collect(), accounts, receipt: Default::default() };
    let r = plain(&mut k).map(|n| {
        k.receipt_line(format!("BF killed {n} end"));
        json!({"killed": n})
    });
    let receipt = k.receipt.borrow().join("\n");
    match r {
        Ok(mut v) => {
            v["receipt"] = json!(receipt + "\n");
            Ok(v)
        }
        Err(e) => {
            let d = match e.detail.clone() {
                Some(Value::Object(mut m)) => {
                    m.insert("receipt".into(), json!(receipt));
                    Value::Object(m)
                }
                _ => json!({"receipt": receipt}),
            };
            Err(e.with_detail(d))
        }
    }
}
