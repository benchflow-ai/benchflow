//! bfx: the one program BenchFlow runs as root in a sandbox for its own
//! purposes after the agent has worked there (TRUSTED-RUNTIME.md §3 I1, §4.1).
//!
//! It is static (no PT_INTERP, no DT_NEEDED), pinned by SHA-256 in the
//! Python package, started by absolute path from a read-only location, and
//! before reading any input it marks itself non-dumpable and closes every
//! inherited descriptor above 2. It never reads its own environment.
//!
//! Usage:
//!   bfx version
//!   bfx <command> --json '<arguments>'     one command, one JSON answer
//!   bfx <command> --json -                 the arguments are stdin's first line
//!   bfx serve [--journal JOURNAL]          JSON lines in, JSON lines out
//!   bfx serve --frames [JOURNAL]           the exec channel's helper
//!   bfx serve --verdict JOURNAL PID        after a channel died
//!
//! Commands: exec, kill, scan, tar-out, tar-in, mkdir, chown, chmod, clear,
//! remove, write, walk, stat (see each module). One-shot `exec` replaces
//! bfx with the program; one-shot `tar-out` writes the archive to stdout and
//! its answer to stderr; one-shot `tar-in` reads the archive from stdin.
//! Exit status: 0 done, 1 the command failed (the answer says why), 2 a
//! usage error; for `exec`, 126 when the program could not be started (127
//! when it does not exist).

mod args;
mod asuser;
mod b64;
mod channel;
mod elf;
mod err;
mod exec;
mod fsx;
mod glob;
mod harden;
mod kill;
mod ops;
mod scan;
mod serve;
mod sys;
mod tar;
#[cfg(test)]
mod testgen;
#[cfg(test)]
mod tests;

use err::Error;
use rustix::fd::AsFd;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;

/// The SHA-256 of the running binary, read through /proc/self/exe.
pub fn self_sha256() -> Option<String> {
    let fd = rustix::fs::open("/proc/self/exe", rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC, rustix::fs::Mode::empty()).ok()?;
    let data = fsx::read_to_end(fd.as_fd(), 256 << 20).ok()?;
    Some(ops::hex(&Sha256::digest(&data)))
}

fn usage() -> i32 {
    let _ = std::io::stderr().write_all(b"usage: bfx version | bfx <command> --json '<arguments>'|- | bfx serve [--journal JOURNAL | --frames [JOURNAL] | --verdict JOURNAL PID]\n");
    2
}

fn print_line(to_stderr: bool, v: &Value) {
    let mut text = serde_json::to_vec(v).unwrap_or_default();
    text.push(b'\n');
    let _ = if to_stderr { std::io::stderr().write_all(&text) } else { std::io::stdout().write_all(&text).and_then(|_| std::io::stdout().flush()) };
}

fn envelope(r: Result<Value, Error>) -> (Value, i32) {
    match r {
        Ok(v) => (json!({"ok": true, "result": v}), 0),
        Err(e) => (json!({"ok": false, "error": e.to_json()}), 1),
    }
}

fn one_shot(cmd: &str, text: &[u8]) -> i32 {
    let v: Value = match serde_json::from_slice(text) {
        Ok(v) => v,
        Err(e) => {
            print_line(false, &envelope(Err(Error::bad(format!("--json is not JSON: {e}")))).0);
            return 2;
        }
    };
    match cmd {
        "exec" => {
            if let Err(e) = serve::set_resolver(&v) {
                print_line(true, &envelope(Err(e)).0);
                return 2;
            }
            let e = exec::exec_now(&v);
            let code = if e.is(rustix::io::Errno::NOENT) { 127 } else if e.kind == "bad-request" { 2 } else { 126 };
            print_line(true, &envelope(Err(e)).0);
            code
        }
        "tar-out" => {
            let r = serve::set_resolver(&v).and_then(|_| {
                let w = std::io::BufWriter::with_capacity(1 << 16, std::io::stdout().lock());
                tar::tar_out(&v, w).map(|(rep, _)| rep)
            });
            let (env, code) = envelope(r);
            print_line(true, &env);
            code
        }
        "tar-in" => {
            let r = serve::set_resolver(&v).and_then(|_| tar::tar_in(&v, std::io::BufReader::with_capacity(1 << 16, std::io::stdin().lock())));
            let (env, code) = envelope(r);
            print_line(false, &env);
            code
        }
        _ => {
            let (env, code) = envelope(serve::dispatch(cmd, &v));
            print_line(false, &env);
            code
        }
    }
}

fn run() -> i32 {
    let args: Vec<OsString> = std::env::args_os().collect();
    if let Err(why) = harden::startup() {
        let _ = std::io::stderr().write_all(format!("bfx: {why}\n").as_bytes());
        return 125;
    }
    let a: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    match a.get(1..).unwrap_or(&[]) {
        [b"version"] => {
            print_line(false, &serve::version());
            0
        }
        [b"serve"] => serve::serve(None),
        [b"serve", b"--journal", _] => serve::serve(Some(&args[3])),
        [b"serve", b"--frames"] => channel::frames(None),
        [b"serve", b"--frames", _] => channel::frames(Some(&args[3])),
        [b"serve", b"--verdict", _, pid] => channel::verdict(&args[3], std::str::from_utf8(pid).unwrap_or("")),
        [b"_as-user", arg] => asuser::main(std::str::from_utf8(arg).unwrap_or("")),
        [b"_populate", dir] => {
            let (env, code) = envelope(ops::populate(std::str::from_utf8(dir).unwrap_or("")));
            print_line(false, &env);
            code
        }
        [b"_probe"] => {
            std::thread::sleep(std::time::Duration::from_secs(60));
            0
        }
        [cmd, b"--json", text] => match std::str::from_utf8(cmd) {
            Ok(c) if serve::COMMANDS.contains(&c) => {
                if *text == b"-" {
                    // Secrets (an exec's environment) never appear in argv.
                    match serve::read_args_line() {
                        Ok(line) => one_shot(c, &line),
                        Err(why) => {
                            print_line(false, &envelope(Err(Error::bad(why))).0);
                            2
                        }
                    }
                } else {
                    one_shot(c, text)
                }
            }
            _ => usage(),
        },
        _ => usage(),
    }
}

fn main() {
    std::process::exit(run());
}
