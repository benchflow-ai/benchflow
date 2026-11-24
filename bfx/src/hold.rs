//! `bfx hold`: a held compose service's entrypoint until BenchFlow releases it,
//! so nothing of the task image's runs before the service's rules are in.
//!
//!   bfx hold --token TOKEN [--setenv NAME=VALUE]... -- ARGV...
//!                                       wait for SIGUSR1, then become ARGV
//!   bfx hold --ready TOKEN              is the waiter ready to be released?
//!   bfx hold --release TOKEN            release it (SIGUSR1)
//!   bfx hold --run-if-released --token TOKEN -- ARGV...
//!                                       become ARGV only when no waiter
//!                                       holds the token
//!
//! The waiter runs as the service's PID 1 (or not, when the service shares
//! another container's pid namespace), as the service's own user. Before the
//! release it opens no file of the image's: it reads only its argv, and it
//! does not read its environment or search PATH.
//!
//! 1. The handlers for SIGUSR1 (release) and SIGTERM, SIGINT, SIGQUIT and
//!    SIGHUP (stop) go in before anything else: PID 1 drops a signal from the
//!    host whose disposition is SIG_DFL, even a blocked one. `--ready` reads
//!    exactly those five bits of the waiter's `SigCgt`.
//! 2. It waits in `sigsuspend`, with the five blocked outside it: no CPU, no
//!    timer, no child.
//! 3. A stop signal before the release exits 128 + its number. A stop wins
//!    over a release delivered in the same wake-up.
//! 4. The first SIGUSR1 releases. A later one, before the exec, is
//!    discarded. Every disposition bfx changed (the five, and SIGPIPE, which
//!    Rust's runtime ignores) goes back to SIG_DFL, the signal mask is
//!    cleared, and ARGV replaces bfx with the environment it was given,
//!    byte for byte, with each `--setenv` applied now and only now: the
//!    inherited entries of those names are dropped and each NAME follows the
//!    rest once, in the order the options first name it, with its last
//!    VALUE. bfx's own environment never changes. A stop signal still
//!    pending then reaches the program.
//! 5. ARGV[0] without a `/` is looked up only now, in that environment's
//!    PATH (Docker's default when it has none): the first regular file with
//!    an execute bit, as runc does. A failed exec prints
//!    `bfx hold: cannot start ARGV[0]: <strerror>` and exits 127 (not found)
//!    or 126.
//!
//! A `--setenv` NAME is `[A-Za-z_][A-Za-z0-9_]*`; VALUE is everything after
//! the first `=`, bytes as given (empty, or holding `=`). Anything else is a
//! usage error. The values are in the waiter's cmdline, which any process in
//! the container can read: they are never secrets.
//!
//! `--ready` and `--release` run as `docker exec` steps inside the container.
//! They find the waiter by its `/proc/<pid>/cmdline` (`/.bfx/bfx hold --token
//! TOKEN [--setenv NAME=VALUE]... -- ...`), read only `/proc`, and exit 0 (ready / released), 1 (not
//! yet, or not found) or 3 (more than one waiter holds the token). `--release`
//! signals nothing unless exactly one waiter is ready.
//!
//! `--run-if-released` runs BenchFlow's exec of a command into a held
//! service: ARGV only when no process holds the token's waiter cmdline. A
//! released waiter has become the service's program, so its cmdline no
//! longer matches; a container restarted by its restart policy starts in the
//! waiter again, with the same token, so a command sent into it after the
//! restart finds that waiter and runs nothing (exit 75, one line). An exec
//! started before a restart ends with the container. Before the check it
//! reads only its argv and `/proc`; then ARGV replaces bfx with the
//! environment it was given, SIGPIPE back at its default, and ARGV[0] looked
//! up as after a release (step 5). A `/proc` it can't list runs nothing
//! (exit 125).

use std::ffi::{CStr, CString, OsString};
use std::io::Write;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

/// Where the harness volume puts bfx in every held service; the waiter's
/// argv[0].
pub const PATH: &str = "/.bfx/bfx";
/// The release.
pub const RELEASE: libc::c_int = libc::SIGUSR1;
/// The stop signals.
pub const STOPS: [libc::c_int; 4] = [libc::SIGTERM, libc::SIGINT, libc::SIGQUIT, libc::SIGHUP];
/// The exit status of a usage error.
pub const USAGE: i32 = 2;
/// `--ready`/`--release`: more than one waiter holds the token.
pub const AMBIGUOUS: i32 = 3;
/// `--run-if-released`: a waiter holds the token; nothing ran.
pub const HELD: i32 = 75;
/// `--run-if-released`: `/proc` could not be listed; nothing ran.
pub const UNCHECKED: i32 = 125;
/// Docker's PATH when the environment has none.
pub const DEFAULT_PATH: &[u8] = b"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

static RELEASED: AtomicBool = AtomicBool::new(false);
static STOPPED: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(sig: libc::c_int) {
    if sig == RELEASE {
        RELEASED.store(true, Ordering::SeqCst);
    } else {
        let _ = STOPPED.compare_exchange(0, sig, Ordering::SeqCst, Ordering::SeqCst);
    }
}

extern "C" {
    // The process environment as the kernel and musl left it.
    static environ: *const *const libc::c_char;
}

fn usage(why: &str) -> i32 {
    let _ = std::io::stderr().write_all(
        format!(
            "bfx hold: {why}; usage: bfx hold --token TOKEN [--setenv NAME=VALUE]... -- ARGV... | --ready TOKEN | --release TOKEN | --run-if-released --token TOKEN -- ARGV...\n"
        )
        .as_bytes(),
    );
    USAGE
}

/// `[A-Za-z0-9]{16,64}`.
pub fn token_ok(t: &[u8]) -> bool {
    (16..=64).contains(&t.len()) && t.iter().all(|c| c.is_ascii_alphanumeric())
}

/// A `--setenv` NAME: `[A-Za-z_][A-Za-z0-9_]*`.
pub fn name_ok(n: &[u8]) -> bool {
    match n.split_first() {
        Some((first, rest)) => {
            (first.is_ascii_alphabetic() || *first == b'_') && rest.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
        }
        None => false,
    }
}

/// A waiter's argument line, `ARGV0 hold --token TOKEN [--setenv NAME=VALUE]...
/// -- ARGV...`.
#[derive(Debug, PartialEq)]
pub struct WaiterArgs<'a> {
    pub token: &'a [u8],
    /// (NAME, VALUE) in the order given.
    pub setenv: Vec<(&'a [u8], &'a [u8])>,
    /// Where ARGV starts in the argument line.
    pub argv_at: usize,
}

/// Parse a waiter's argument line (argv[0] included); Err(why) for a usage
/// error. The token's characters are checked last, as before.
pub fn parse_waiter<'a>(a: &[&'a [u8]]) -> Result<WaiterArgs<'a>, &'static str> {
    const SHAPE: &str = "expected --token TOKEN [--setenv NAME=VALUE]... -- ARGV...";
    let (Some(b"hold"), Some(b"--token"), Some(token)) = (a.get(1).copied(), a.get(2).copied(), a.get(3).copied()) else {
        return Err(SHAPE);
    };
    let mut setenv = Vec::new();
    let mut i = 4;
    while a.get(i).copied() == Some(b"--setenv".as_slice()) {
        let Some(pair) = a.get(i + 1).copied() else { return Err("--setenv needs NAME=VALUE") };
        let Some(eq) = pair.iter().position(|c| *c == b'=') else { return Err("--setenv needs NAME=VALUE") };
        let (name, value) = (&pair[..eq], &pair[eq + 1..]);
        if !name_ok(name) {
            return Err("a --setenv NAME is letters, digits and _, not starting with a digit");
        }
        setenv.push((name, value));
        i += 2;
    }
    if a.get(i).copied() != Some(b"--".as_slice()) {
        return Err(SHAPE);
    }
    if a.len() == i + 1 {
        return Err("no program after --");
    }
    if !token_ok(token) {
        return Err("the token must be 16 to 64 letters and digits");
    }
    Ok(WaiterArgs { token, setenv, argv_at: i + 1 })
}

/// The environment the program starts with: *inherited* as it is, except
/// that an entry named by a `--setenv` is dropped, then each such NAME once,
/// in the order the options first name it, with its last VALUE.
pub fn release_env(inherited: &[&[u8]], setenv: &[(&[u8], &[u8])]) -> Vec<Vec<u8>> {
    let named = |entry: &[u8]| {
        setenv.iter().any(|(n, _)| entry.len() > n.len() && entry.starts_with(n) && entry[n.len()] == b'=')
    };
    let mut out: Vec<Vec<u8>> = inherited.iter().filter(|e| !named(e)).map(|e| e.to_vec()).collect();
    let mut done: Vec<&[u8]> = Vec::new();
    for (name, _) in setenv {
        if done.contains(name) {
            continue;
        }
        done.push(name);
        let value = setenv.iter().rev().find(|(n, _)| n == name).map(|(_, v)| *v).unwrap_or_default();
        let mut entry = name.to_vec();
        entry.push(b'=');
        entry.extend_from_slice(value);
        out.push(entry);
    }
    out
}

fn sigset(signals: &[libc::c_int]) -> libc::sigset_t {
    // SAFETY: sigemptyset/sigaddset initialise and fill a local set.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for s in signals {
            libc::sigaddset(&mut set, *s);
        }
        set
    }
}

fn five() -> Vec<libc::c_int> {
    let mut v = vec![RELEASE];
    v.extend_from_slice(&STOPS);
    v
}

fn set_disposition(sig: libc::c_int, handler: libc::sighandler_t, mask: &libc::sigset_t) {
    // SAFETY: a zeroed sigaction with a handler (a plain extern "C" function,
    // SIG_IGN or SIG_DFL) and a valid mask.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler;
        sa.sa_mask = *mask;
        sa.sa_flags = 0;
        libc::sigaction(sig, &sa, std::ptr::null_mut());
    }
}

/// Step 1: block the five, then install their handler. Blocked first, so a
/// signal that arrives now waits for `sigsuspend` and is never lost between
/// a flag check and the wait. Returns the mask to wait with.
pub fn install() -> libc::sigset_t {
    let five = five();
    let set = sigset(&five);
    // SAFETY: a valid set and an out-pointer for the old mask.
    let mut old: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe { libc::sigprocmask(libc::SIG_BLOCK, &set, &mut old) };
    for s in &five {
        set_disposition(*s, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t, &set);
    }
    let mut wait = old;
    for s in &five {
        // SAFETY: removing members from a valid local set.
        unsafe { libc::sigdelset(&mut wait, *s) };
    }
    wait
}

/// Step 2-3: wait for the release; a stop exits here.
pub fn wait(mask: &libc::sigset_t) {
    loop {
        let stop = STOPPED.load(Ordering::SeqCst);
        if stop != 0 {
            // SAFETY: _exit, with no atexit work to run.
            unsafe { libc::_exit(128 + stop) };
        }
        if RELEASED.load(Ordering::SeqCst) {
            return;
        }
        // SAFETY: a valid mask; returns after a handler ran.
        unsafe { libc::sigsuspend(mask) };
    }
}

/// Step 4: undo every disposition bfx changed and clear the mask. A SIGUSR1
/// already pending again is discarded (SIG_IGN drops it), and SIGUSR1 stays
/// ignored until the mask is clear, so a late one meets SIG_DFL only in the
/// last step before the exec. A pending stop signal stays pending: it reaches
/// the program (or, if it lands before the exec, stops the process).
pub fn restore() {
    let empty = sigset(&[]);
    set_disposition(RELEASE, libc::SIG_IGN, &empty);
    for s in STOPS.into_iter().chain([libc::SIGPIPE]) {
        set_disposition(s, libc::SIG_DFL, &empty);
    }
    // SAFETY: an empty set.
    unsafe { libc::sigprocmask(libc::SIG_SETMASK, &empty, std::ptr::null_mut()) };
    set_disposition(RELEASE, libc::SIG_DFL, &empty);
}

/// The inherited environment's entries, read only after the release.
fn inherited_env() -> Vec<&'static [u8]> {
    let mut out = Vec::new();
    // SAFETY: environ is a NULL-terminated array of NUL-terminated strings,
    // which bfx never changes.
    unsafe {
        let mut p = environ;
        if p.is_null() {
            return out;
        }
        while !(*p).is_null() {
            out.push(CStr::from_ptr(*p).to_bytes());
            p = p.add(1);
        }
    }
    out
}

/// PATH in *env*, the first entry that sets it.
fn path_of(env: &[Vec<u8>]) -> Option<&[u8]> {
    env.iter().find_map(|e| e.strip_prefix(b"PATH="))
}

/// Step 5: the file to exec. With a `/`, ARGV[0] as given; else the first
/// regular file with an execute bit in PATH, in order (an empty entry is the
/// working directory), as runc's lookup does. Err(errno) when there is none.
pub fn resolve(argv0: &[u8], path: Option<&[u8]>) -> Result<Vec<u8>, i32> {
    if argv0.contains(&b'/') {
        return Ok(argv0.to_vec());
    }
    if argv0.is_empty() {
        return Err(libc::ENOENT);
    }
    for dir in path.unwrap_or(DEFAULT_PATH).split(|c| *c == b':') {
        let dir: &[u8] = if dir.is_empty() { b"." } else { dir };
        let mut cand = dir.to_vec();
        cand.push(b'/');
        cand.extend_from_slice(argv0);
        let Ok(c) = CString::new(cand.clone()) else { continue };
        // SAFETY: stat into a local buffer through a valid C string.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::stat(c.as_ptr(), &mut st) } == 0
            && (st.st_mode & libc::S_IFMT) == libc::S_IFREG
            && st.st_mode & 0o111 != 0
        {
            return Ok(cand);
        }
    }
    Err(libc::ENOENT)
}

fn strerror(errno: i32) -> String {
    // SAFETY: strerror returns a static, NUL-terminated message.
    unsafe { CStr::from_ptr(libc::strerror(errno)) }.to_string_lossy().into_owned()
}

/// Steps 4-5 after the release. Returns only on failure, with the exit status.
pub fn become_program(argv: &[OsString], setenv: &[(&[u8], &[u8])]) -> i32 {
    restore();
    exec_argv(argv, release_env(&inherited_env(), setenv))
}

/// Step 5: ARGV, looked up in *env*'s PATH, replaces bfx with *env*. Returns
/// only on failure, with the exit status.
fn exec_argv(argv: &[OsString], env: Vec<Vec<u8>>) -> i32 {
    let argv0 = argv[0].as_bytes();
    let fail = |errno: i32| -> i32 {
        let _ = std::io::stderr().write_all(
            format!("bfx hold: cannot start {}: {}\n", String::from_utf8_lossy(argv0), strerror(errno)).as_bytes(),
        );
        if errno == libc::ENOENT {
            127
        } else {
            126
        }
    };
    let file = match resolve(argv0, path_of(&env)) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    let Ok(file) = CString::new(file) else { return fail(libc::ENOENT) };
    let args: Vec<CString> = match argv.iter().map(|a| CString::new(a.clone().into_vec())).collect() {
        Ok(a) => a,
        Err(_) => return fail(libc::EINVAL),
    };
    let mut ptrs: Vec<*const libc::c_char> = args.iter().map(|a| a.as_ptr()).collect();
    ptrs.push(std::ptr::null());
    // Entries come from C strings and from argv, so none holds a NUL.
    let Ok(env) = env.into_iter().map(CString::new).collect::<Result<Vec<CString>, _>>() else { return fail(libc::EINVAL) };
    let mut envp: Vec<*const libc::c_char> = env.iter().map(|e| e.as_ptr()).collect();
    envp.push(std::ptr::null());
    // SAFETY: a valid path, and NULL-terminated argv and envp arrays of C
    // strings that outlive the call.
    unsafe { libc::execve(file.as_ptr(), ptrs.as_ptr(), envp.as_ptr()) };
    fail(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EINVAL))
}

/// The waiter: `bfx hold --token TOKEN [--setenv NAME=VALUE]... -- ARGV...`.
/// The handlers go in before the arguments are checked, so the waiter is
/// ready to count as soon as it can be found.
fn waiter(args: &[OsString]) -> i32 {
    let mask = install();
    let a: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    let parsed = match parse_waiter(&a) {
        Ok(p) => p,
        Err(why) => return usage(why),
    };
    let _ = rustix::process::set_dumpable_behavior(rustix::process::DumpableBehavior::NotDumpable);
    std::panic::set_hook(Box::new(|_| {
        let _ = std::io::stderr().write_all(b"bfx: internal error\n");
    }));
    wait(&mask);
    become_program(&args[parsed.argv_at..], &parsed.setenv)
}

/// A waiter for *token*: `/proc/<pid>/cmdline` is `/.bfx/bfx hold --token
/// TOKEN [--setenv NAME=VALUE]... -- ARGV...` with ARGV not empty.
pub fn is_waiter(cmdline: &[u8], token: &[u8], argv0: &[u8]) -> bool {
    let fields: Vec<&[u8]> = cmdline.strip_suffix(b"\0").unwrap_or(cmdline).split(|c| *c == 0).collect();
    fields.first() == Some(&argv0) && parse_waiter(&fields).is_ok_and(|w| w.token == token)
}

/// The `SigCgt` mask in a `/proc/<pid>/status` text.
pub fn caught(status: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(status).ok()?;
    let hex = text.lines().find_map(|l| l.strip_prefix("SigCgt:"))?.trim();
    u64::from_str_radix(hex, 16).ok()
}

/// Whether the five handlers are installed.
pub fn handlers_in(sigcgt: u64) -> bool {
    five().iter().all(|s| sigcgt & (1u64 << (*s - 1)) != 0)
}

fn read_proc(pid: &str, name: &str) -> Option<Vec<u8>> {
    std::fs::read(format!("/proc/{pid}/{name}")).ok()
}

/// The pids whose cmdline names *token*'s waiter.
pub fn find(token: &[u8], argv0: &[u8]) -> Vec<i32> {
    waiters_in(std::path::Path::new("/proc"), token, argv0).unwrap_or_default()
}

/// The pids under *proc* whose cmdline names *token*'s waiter; Err when
/// *proc* can't be listed. A process gone between the listing and its read
/// is skipped.
pub fn waiters_in(proc: &std::path::Path, token: &[u8], argv0: &[u8]) -> std::io::Result<Vec<i32>> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(proc)?.flatten() {
        let name = e.file_name();
        let Some(pid) = name.to_str().filter(|n| n.bytes().all(|c| c.is_ascii_digit())) else { continue };
        if let Ok(cmd) = std::fs::read(proc.join(pid).join("cmdline")) {
            if is_waiter(&cmd, token, argv0) {
                if let Ok(n) = pid.parse() {
                    found.push(n);
                }
            }
        }
    }
    Ok(found)
}

/// The check's answer for the waiters found: None to run ARGV, else the
/// exit status and its line.
fn decide(found: std::io::Result<Vec<i32>>) -> Option<(i32, String)> {
    match found {
        Ok(found) if found.is_empty() => None,
        Ok(_) => Some((HELD, "bfx hold: the waiter for this token has not been released; nothing ran\n".into())),
        Err(e) => Some((UNCHECKED, format!("bfx hold: cannot list /proc: {e}; nothing ran\n"))),
    }
}

/// `bfx hold --run-if-released --token TOKEN -- ARGV...`: ARGV only when no
/// waiter holds the token.
fn run_if_released(args: &[OsString]) -> i32 {
    let a: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    let token = match a.as_slice() {
        [_, b"hold", b"--run-if-released", b"--token", token, b"--", _, ..] => *token,
        [_, b"hold", b"--run-if-released", b"--token", _, b"--"] => return usage("no program after --"),
        _ => return usage("expected --run-if-released --token TOKEN -- ARGV..."),
    };
    if !token_ok(token) {
        return usage("the token must be 16 to 64 letters and digits");
    }
    if let Some((code, line)) = decide(waiters_in(std::path::Path::new("/proc"), token, PATH.as_bytes())) {
        let _ = std::io::stderr().write_all(line.as_bytes());
        return code;
    }
    set_disposition(libc::SIGPIPE, libc::SIG_DFL, &sigset(&[]));
    exec_argv(&args[6..], inherited_env().iter().map(|e| e.to_vec()).collect())
}

/// `--ready` (signal = false) and `--release` (true), for waiters started as
/// *argv0*.
pub fn control(token: &[u8], release: bool, argv0: &[u8]) -> i32 {
    let pids = find(token, argv0);
    let pid = match pids.as_slice() {
        [] => return 1,
        [p] => *p,
        _ => {
            let _ = std::io::stderr().write_all(format!("bfx hold: {} waiters hold this token\n", pids.len()).as_bytes());
            return AMBIGUOUS;
        }
    };
    let ready = read_proc(&pid.to_string(), "status").as_deref().and_then(caught).map(handlers_in).unwrap_or(false);
    if !ready {
        return 1;
    }
    if !release {
        return 0;
    }
    match crate::sys::kill_one(pid, RELEASE) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// `bfx hold ...`, before bfx's own startup: the waiter's handlers come first.
pub fn main(args: &[OsString]) -> i32 {
    let a: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    match a.as_slice() {
        [_, b"hold", b"--token", ..] => waiter(args),
        [_, b"hold", b"--run-if-released", ..] => run_if_released(args),
        [_, b"hold", flag @ (b"--ready" | b"--release"), token] => {
            if let Err(why) = crate::harden::startup() {
                let _ = std::io::stderr().write_all(format!("bfx: {why}\n").as_bytes());
                return 125;
            }
            if !token_ok(token) {
                return usage("the token must be 16 to 64 letters and digits");
            }
            control(token, *flag == b"--release", PATH.as_bytes())
        }
        _ => usage("expected --token, --ready, --release or --run-if-released"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_16_to_64_letters_and_digits() {
        assert!(token_ok(b"abcdefghij012345"));
        assert!(token_ok(&[b'Z'; 64]));
        assert!(!token_ok(b"abcdefghij01234"));
        assert!(!token_ok(&[b'a'; 65]));
        assert!(!token_ok(b"abcdefghij01234-"));
        assert!(!token_ok(b"abcdefghij01234\xc3\xa9"));
    }

    #[test]
    fn a_waiter_is_found_by_its_exact_cmdline() {
        let t = b"abcdefghij012345";
        assert!(is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--\0/bin/sh\0", t, PATH.as_bytes()));
        assert!(is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--\0x\0\0y\0", t, PATH.as_bytes()));
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--\0", t, PATH.as_bytes()));
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012346\0--\0x\0", t, PATH.as_bytes()));
        assert!(!is_waiter(b"/usr/bin/bfx\0hold\0--token\0abcdefghij012345\0--\0x\0", t, PATH.as_bytes()));
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--ready\0abcdefghij012345\0", t, PATH.as_bytes()));
    }

    #[test]
    fn a_waiter_with_setenv_is_found_by_its_cmdline() {
        let t = b"abcdefghij012345";
        let p = PATH.as_bytes();
        assert!(is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--setenv\0A=1\0--\0x\0", t, p));
        assert!(is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--setenv\0A=--\0--setenv\0B=\0--\0x\0", t, p));
        // A pair without its value, a bad NAME, or no program: not a waiter.
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--setenv\0--\0x\0", t, p));
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--setenv\01A=x\0--\0x\0", t, p));
        assert!(!is_waiter(b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--setenv\0A=1\0--\0", t, p));
    }

    #[test]
    fn setenv_names_and_values() {
        for good in [&b"A"[..], b"_", b"_a1", b"PYTHONPATH", b"a_B_9"] {
            assert!(name_ok(good), "{good:?}");
        }
        for bad in [&b""[..], b"1A", b"A-B", b"A B", b"A.B", b"\xc3\x89", b"A=B"] {
            assert!(!name_ok(bad), "{bad:?}");
        }
        let a: Vec<&[u8]> = vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv", b"E=", b"--setenv", b"Q=a=b", b"--", b"x", b"--setenv"];
        let w = parse_waiter(&a).unwrap();
        assert_eq!(w.setenv, vec![(&b"E"[..], &b""[..]), (&b"Q"[..], &b"a=b"[..])]);
        assert_eq!(w.argv_at, 9);
        let bad: Vec<Vec<&[u8]>> = vec![
            vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv"],
            vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv", b"NOEQ", b"--", b"x"],
            vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv", b"=x", b"--", b"x"],
            vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv", b"A=x", b"x"],
            vec![b"/.bfx/bfx", b"hold", b"--token", b"abcdefghij012345", b"--setenv", b"A=x", b"--"],
        ];
        for a in bad {
            assert!(parse_waiter(&a).is_err(), "{a:?}");
        }
    }

    #[test]
    fn the_release_env_replaces_named_entries_and_keeps_the_rest() {
        let inherited: Vec<&[u8]> = vec![b"PATH=/bin", b"A=1", b"AB=2", b"NOEQ", b"A=dup", b"Z=9"];
        let setenv: Vec<(&[u8], &[u8])> = vec![(b"A", b"x"), (b"R", b"first"), (b"E", b""), (b"R", b"second")];
        let got = release_env(&inherited, &setenv);
        let want: Vec<&[u8]> = vec![b"PATH=/bin", b"AB=2", b"NOEQ", b"Z=9", b"A=x", b"R=second", b"E="];
        assert_eq!(got, want.iter().map(|e| e.to_vec()).collect::<Vec<_>>());
        assert_eq!(release_env(&inherited, &[]), inherited.iter().map(|e| e.to_vec()).collect::<Vec<_>>());
        assert_eq!(path_of(&release_env(&inherited, &[(b"PATH", b"/x")])), Some(&b"/x"[..]));
    }

    #[test]
    fn waiters_are_found_by_token_and_an_unlisted_proc_is_an_error() {
        let root = std::env::temp_dir().join(format!("bfx-proc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let entries: [(&str, &[u8]); 4] = [
            ("10", b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--\0x\0"),
            ("11", b"/.bfx/bfx\0hold\0--token\0abcdefghij012346\0--\0x\0"),
            ("12", b"/.bfx/bfx\0hold\0--run-if-released\0--token\0abcdefghij012345\0--\0x\0"),
            ("self", b"/.bfx/bfx\0hold\0--token\0abcdefghij012345\0--\0x\0"),
        ];
        for (pid, cmd) in entries {
            std::fs::create_dir_all(root.join(pid)).unwrap();
            std::fs::write(root.join(pid).join("cmdline"), cmd).unwrap();
        }
        std::fs::create_dir_all(root.join("13")).unwrap();
        assert_eq!(waiters_in(&root, b"abcdefghij012345", PATH.as_bytes()).unwrap(), vec![10]);
        assert!(waiters_in(&root.join("none"), b"abcdefghij012345", PATH.as_bytes()).is_err());
        let _ = std::fs::remove_dir_all(&root);
        // Run only when the listing worked and found none.
        assert_eq!(decide(Ok(vec![])), None);
        assert_eq!(decide(Ok(vec![7])).map(|d| d.0), Some(HELD));
        assert_eq!(decide(Err(std::io::Error::from_raw_os_error(libc::EACCES))).map(|d| d.0), Some(UNCHECKED));
    }

    #[test]
    fn ready_needs_all_five_handlers() {
        let all: u64 = five().iter().map(|s| 1u64 << (*s - 1)).sum();
        assert!(handlers_in(all));
        for s in five() {
            assert!(!handlers_in(all & !(1u64 << (s - 1))));
        }
        assert_eq!(caught(b"Name:\tbfx\nSigCgt:\t0000000000004a07\n"), Some(0x4a07));
        assert_eq!(caught(b"Name:\tbfx\n"), None);
    }

    #[test]
    fn a_name_without_a_slash_is_looked_up_in_path_only() {
        assert_eq!(resolve(b"./x", None), Ok(b"./x".to_vec()));
        assert_eq!(resolve(b"/no/such", None), Ok(b"/no/such".to_vec()));
        assert_eq!(resolve(b"", Some(b"/bin")), Err(libc::ENOENT));
        assert_eq!(resolve(b"bfx-no-such-program", Some(b"/bin:/usr/bin")), Err(libc::ENOENT));
        let sh = resolve(b"sh", Some(b"/nonexistent:/bin:/usr/bin")).unwrap();
        assert!(sh == b"/bin/sh" || sh == b"/usr/bin/sh", "{:?}", String::from_utf8_lossy(&sh));
    }
}
