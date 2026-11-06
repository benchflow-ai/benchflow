//! Work done as an account rather than as root: bfx starts itself
//! (`/proc/self/exe _as-user <json>`) after setgroups([]), setresgid and
//! setresuid to the account, with no_new_privs and an empty environment.
//! The child proves its identity (getresuid/getresgid) in its answer, and
//! refuses to run as root at all.
//!
//! Actions, in order:
//! - `{"broadcast": sig}`: kill(-1, sig) as the account. A child of the
//!   helper is the signal's recipient even when the account has no other
//!   process, so success is never an ESRCH; the child is killed and reaped
//!   afterwards. Reports whether kill(-1) succeeded.
//! - `{"signal": sig, "pids": [..]}`: kill(pid, sig) for each; the kernel's
//!   permission rule means a pid that root's process took meanwhile answers
//!   EPERM and is not signalled.

use crate::err::{Error, Result};
use crate::exec::{spawn_capture, Caps, PrepareData};
use crate::sys;
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Duration;

pub fn run(uid: u32, gid: u32, actions: &[Value]) -> Result<Vec<Value>> {
    let arg = json!({"uid": uid, "gid": gid, "actions": actions}).to_string();
    let mut c = Command::new("/proc/self/exe");
    c.arg0("bfx").arg("_as-user").arg(arg);
    c.env_clear();
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let data = PrepareData {
        cwd: None,
        umask: 0o077,
        rlimits: Vec::new(),
        setsid: false,
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
    let out = spawn_capture(c, None, Some(Duration::from_secs(30)), 16 << 20).map_err(|e| Error::io(e, "starting bfx as the account"))?;
    let fail = |why: &str| {
        Error::new("cannot-signal", format!("bfx could not act as UID {uid} GID {gid}: {why}"))
            .with_detail(json!({"stderr": String::from_utf8_lossy(&out.stderr)}))
    };
    if out.status != 0 {
        return Err(fail(&format!("exit status {}", out.status)));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|_| fail("unreadable answer"))?;
    if v.get("uid") != Some(&json!([uid, uid, uid])) || v.get("gid") != Some(&json!([gid, gid, gid])) {
        return Err(fail("its answer does not prove the identity"));
    }
    let results = v.get("results").and_then(|r| r.as_array()).cloned().unwrap_or_default();
    if results.len() != actions.len() {
        return Err(fail("missing results"));
    }
    Ok(results)
}

/// The helper's own side (`bfx _as-user <json>`).
pub fn main(arg: &str) -> i32 {
    let v: Value = match serde_json::from_str(arg) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    let (want_u, want_g) = (v["uid"].as_u64().unwrap_or(0) as u32, v["gid"].as_u64().unwrap_or(0) as u32);
    let (u, g) = (sys::getresuid(), sys::getresgid());
    if want_u == 0 || u != (want_u, want_u, want_u) || g != (want_g, want_g, want_g) {
        eprintln!("bfx _as-user: not running as the requested account");
        return 3;
    }
    let mut results = Vec::new();
    for a in v["actions"].as_array().cloned().unwrap_or_default() {
        results.push(if let Some(sig) = a.get("broadcast").and_then(|s| s.as_i64()) {
            broadcast(sig as i32)
        } else if let Some(sig) = a.get("signal").and_then(|s| s.as_i64()) {
            let mut sent = 0;
            for p in a["pids"].as_array().cloned().unwrap_or_default() {
                if let Some(pid) = p.as_i64() {
                    if sys::kill_one(pid as i32, sig as i32).is_ok() {
                        sent += 1;
                    }
                }
            }
            json!({"sent": sent})
        } else {
            json!({"error": "unknown action"})
        });
    }
    println!("{}", json!({"uid": [u.0, u.1, u.2], "gid": [g.0, g.1, g.2], "results": results}));
    0
}

fn broadcast(sig: i32) -> Value {
    let probe = Command::new("/proc/self/exe").arg0("bfx").arg("_probe").env_clear().stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    let mut probe = match probe {
        Ok(p) => p,
        Err(e) => return json!({"delivered": false, "error": format!("no recipient: {e}")}),
    };
    let r = sys::kill_all(sig);
    let _ = sys::kill_one(probe.id() as i32, libc::SIGKILL);
    let _ = probe.wait();
    match r {
        Ok(()) => json!({"delivered": true}),
        Err(e) => json!({"delivered": false, "error": e.to_string()}),
    }
}
