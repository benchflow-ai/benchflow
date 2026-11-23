//! `bfx hold` end to end, against the hold's contract: the built binary, started
//! with argv[0] `/.bfx/bfx` (as a held service's entrypoint) and its controls
//! run the same way. Linux only; any uid (the EACCES row needs a non-root
//! user or a file with no execute bit, which holds for root too). The strace
//! row needs strace on PATH.
//!
//!   cargo test --release --target x86_64-unknown-linux-musl --test hold

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const BFX: &str = env!("CARGO_BIN_EXE_bfx");
const AS: &str = "/.bfx/bfx";

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn token() -> String {
    format!("T{:08}x{:08}", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst))
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bfx-hold-{}-{}-{tag}", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A waiter: `/.bfx/bfx hold --token TOKEN -- ARGV...` with exactly *env*.
fn waiter<S: AsRef<OsStr>>(tok: &str, argv: &[S], env: &[(OsString, OsString)]) -> Child {
    waiter_setenv(tok, &[], argv, env)
}

/// A waiter with a `--setenv` for each of *setenv* (`NAME=VALUE`).
fn waiter_setenv<S: AsRef<OsStr>>(tok: &str, setenv: &[OsString], argv: &[S], env: &[(OsString, OsString)]) -> Child {
    let mut c = Command::new(BFX);
    c.arg0(AS).args(["hold", "--token", tok]);
    for pair in setenv {
        c.arg("--setenv").arg(pair);
    }
    c.arg("--").args(argv).env_clear();
    for (k, v) in env {
        c.env(k, v);
    }
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap()
}

/// NUL-terminated `NAME=VALUE` entries, as `/proc/<pid>/environ` holds them.
fn environ_bytes(entries: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (k, v) in entries {
        out.extend_from_slice(k);
        out.push(b'=');
        out.extend_from_slice(v);
        out.push(0);
    }
    out
}

/// `--ready` / `--release`: (exit status, stderr).
fn control(flag: &str, tok: &str) -> (i32, String) {
    let out = Command::new(BFX).arg0(AS).args(["hold", flag, tok]).env_clear().output().unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn wait_ready(tok: &str) {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let (code, err) = control("--ready", tok);
        if code == 0 {
            return;
        }
        assert_eq!((code, err.as_str()), (1, ""), "--ready");
        assert!(Instant::now() < end, "the waiter never became ready");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn status_field(pid: u32, field: &str) -> u64 {
    let text = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
    let hex = text.lines().find_map(|l| l.strip_prefix(field)).unwrap().trim();
    u64::from_str_radix(hex, 16).unwrap()
}

fn bit(sig: i32) -> u64 {
    1u64 << (sig - 1)
}

const FIVE: [i32; 5] = [libc::SIGUSR1, libc::SIGTERM, libc::SIGINT, libc::SIGQUIT, libc::SIGHUP];

fn finish(mut c: Child) -> (std::process::ExitStatus, String) {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(st) = c.try_wait().unwrap() {
            let mut err = String::new();
            c.stderr.take().unwrap().read_to_string(&mut err).unwrap();
            return (st, err);
        }
        if Instant::now() > end {
            let _ = c.kill();
            let _ = c.wait();
            panic!("the process did not finish");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Whether this process holds CAP_SYS_PTRACE (bit 19 of `CapEff`).
/// `bfx hold --run-if-released --token TOKEN -- ARGV...` with exactly *env*:
/// (exit status, stdout, stderr).
fn run_if_released<S: AsRef<OsStr>>(tok: &str, argv: &[S], env: &[(OsString, OsString)]) -> (i32, Vec<u8>, String) {
    let mut c = Command::new(BFX);
    c.arg0(AS).args(["hold", "--run-if-released", "--token", tok, "--"]).args(argv).env_clear();
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.stdin(Stdio::null()).output().unwrap();
    (out.status.code().unwrap_or(-1), out.stdout, String::from_utf8_lossy(&out.stderr).into_owned())
}

/// A waiter for *tok*, released, its program running (its cmdline no longer
/// the waiter's).
fn released_waiter(tok: &str) -> Child {
    let c = waiter(tok, &["/bin/sleep", "30"], &[]);
    wait_ready(tok);
    assert_eq!(control("--release", tok).0, 0);
    let end = Instant::now() + Duration::from_secs(10);
    while std::fs::read(format!("/proc/{}/cmdline", c.id())).unwrap_or_default().starts_with(AS.as_bytes()) {
        assert!(Instant::now() < end, "the released waiter never became its program");
        std::thread::sleep(Duration::from_millis(5));
    }
    c
}

fn end(mut c: Child) {
    let _ = c.kill();
    let _ = c.wait();
}

fn has_cap_sys_ptrace() -> bool {
    status_field(std::process::id(), "CapEff:") & (1 << 19) != 0
}

fn os(s: &str) -> OsString {
    OsString::from(s)
}

fn env(pairs: &[(&str, &[u8])]) -> Vec<(OsString, OsString)> {
    pairs.iter().map(|(k, v)| (os(k), OsString::from_vec(v.to_vec()))).collect()
}

fn sh() -> String {
    if Path::new("/bin/sh").exists() { "/bin/sh".into() } else { "/usr/bin/sh".into() }
}

#[test]
fn a_usage_error_exits_2_with_one_line() {
    let tok = token();
    // Sixteen letters and a dash: the right length, one character outside the set.
    let bad_char = format!("{}-", "a".repeat(16));
    let cases: Vec<Vec<&str>> = vec![
        vec!["hold"],
        vec!["hold", "--token", &tok],
        vec!["hold", "--token", &tok, "--"],
        vec!["hold", "--token", &tok, "/bin/true"],
        vec!["hold", "--token", "short", "--", "/bin/true"],
        vec!["hold", "--token", &bad_char, "--", "/bin/true"],
        vec!["hold", "--ready", "short"],
        vec!["hold", "--release"],
        vec!["hold", "--release", &tok, "extra"],
        vec!["hold", "--frobnicate", &tok],
        // --setenv: a NAME outside [A-Za-z_][A-Za-z0-9_]*, no `=`, no pair, no `--`.
        vec!["hold", "--token", &tok, "--setenv", "1A=x", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "A-B=x", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "=x", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "\u{c9}=x", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "NOEQ", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "--", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv"],
        vec!["hold", "--token", &tok, "--setenv", "A=x", "/bin/true"],
        vec!["hold", "--token", &tok, "--setenv", "A=x", "--"],
        // --run-if-released: no token, a bad one, no `--`, no program.
        vec!["hold", "--run-if-released"],
        vec!["hold", "--run-if-released", "--token", &tok],
        vec!["hold", "--run-if-released", "--token", &tok, "--"],
        vec!["hold", "--run-if-released", "--token", "short", "--", "/bin/true"],
        vec!["hold", "--run-if-released", "--token", &tok, "/bin/true"],
    ];
    for args in cases {
        // Bounded: a waiter that took a bad argument line would wait forever.
        let c = Command::new(BFX)
            .arg0(AS)
            .args(&args)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let (st, err) = finish(c);
        assert_eq!(st.code(), Some(2), "{args:?}");
        assert!(err.starts_with("bfx hold: ") && err.ends_with('\n') && err.matches('\n').count() == 1, "{args:?}: {err:?}");
    }
}

#[test]
fn ready_only_once_the_five_handlers_are_in() {
    let tok = token();
    assert_eq!(control("--ready", &tok), (1, String::new()), "no waiter yet");
    let c = waiter(&tok, &["/bin/true"], &[]);
    wait_ready(&tok);
    let cgt = status_field(c.id(), "SigCgt:");
    for s in FIVE {
        assert!(cgt & bit(s) != 0, "signal {s} has no handler: {cgt:x}");
    }
    // Waiting, not working: no child, and asleep.
    let stat = std::fs::read_to_string(format!("/proc/{}/stat", c.id())).unwrap();
    assert!(stat.split(") ").nth(1).unwrap().starts_with('S'), "{stat}");
    assert_eq!(control("--release", &tok).0, 0);
    let (st, _) = finish(c);
    assert_eq!(st.code(), Some(0));
}

#[test]
fn the_release_execs_the_exact_argv_and_environment() {
    let out = scratch("exact");
    let script = r#"cat /proc/$$/cmdline > "$OUT/cmdline"; cat /proc/$$/environ > "$OUT/environ"; echo $$ > "$OUT/pid""#;
    let argv: Vec<OsString> = vec![
        os(&sh()),
        os("-c"),
        os(script),
        os("sh0"),
        os("two words"),
        os(""),
        OsString::from_vec(b"\xff\xfe raw".to_vec()),
    ];
    let e = env(&[
        ("OUT", out.as_os_str().as_bytes()),
        ("A", b"1"),
        ("SPACES", b"x  y "),
        ("BYTES", b"\xff\xfe"),
        ("EMPTY", b""),
    ]);
    let tok = token();
    let c = waiter(&tok, &argv, &e);
    let pid = c.id();
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    let mut want_cmd = Vec::new();
    for a in &argv {
        want_cmd.extend_from_slice(a.as_bytes());
        want_cmd.push(0);
    }
    assert_eq!(std::fs::read(out.join("cmdline")).unwrap(), want_cmd);
    // std's Command passes the environment sorted by name.
    let mut sorted = e.clone();
    sorted.sort();
    let mut want_env = Vec::new();
    for (k, v) in &sorted {
        want_env.extend_from_slice(k.as_bytes());
        want_env.push(b'=');
        want_env.extend_from_slice(v.as_bytes());
        want_env.push(0);
    }
    assert_eq!(std::fs::read(out.join("environ")).unwrap(), want_env);
    assert_eq!(std::fs::read_to_string(out.join("pid")).unwrap().trim(), pid.to_string(), "the program keeps bfx's pid");
}

#[test]
fn setenv_applies_byte_exact_values_at_the_release_and_not_before() {
    let out = scratch("setenv");
    // Not the script's last command: a shell may exec its last one in its own
    // place, with the environment it rebuilt (BusyBox ash adds SHLVL and PWD).
    let script = r#"cat /proc/$$/environ > "$OUT/environ"; :"#;
    let inherited = env(&[("OUT", out.as_os_str().as_bytes()), ("KEEP", b"k k"), ("RAWIN", b"\xff")]);
    let setenv: Vec<OsString> = vec![
        os("SPACES= a  b "),
        os("EQ=a=b==c="),
        OsString::from_vec(b"RAW=\xff\xfe x".to_vec()),
        os("EMPTY="),
        os("PYTHONPATH=/opt/app/lib"),
    ];
    let tok = token();
    let c = waiter_setenv(&tok, &setenv, &[sh().as_str(), "-c", script], &inherited);
    let pid = c.id();
    wait_ready(&tok);
    // Held: the waiter's environment is what it was given, without the values.
    // The waiter is not dumpable, so only a reader with CAP_SYS_PTRACE sees it
    // (scripts/test_bfx.sh runs these rows with it).
    let mut sorted: Vec<(&[u8], &[u8])> = inherited.iter().map(|(k, v)| (k.as_bytes(), v.as_bytes())).collect();
    sorted.sort();
    match std::fs::read(format!("/proc/{pid}/environ")) {
        Ok(held) => assert_eq!(held, environ_bytes(&sorted)),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied && !has_cap_sys_ptrace() => {
            eprintln!("the held waiter's environment needs CAP_SYS_PTRACE to read; that check skipped");
        }
        Err(e) => panic!("/proc/{pid}/environ: {e}"),
    }
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    // Released: the inherited entries as they were, then each value, byte for byte.
    let mut want = sorted.clone();
    want.extend_from_slice(&[
        (&b"SPACES"[..], &b" a  b "[..]),
        (b"EQ", b"a=b==c="),
        (b"RAW", b"\xff\xfe x"),
        (b"EMPTY", b""),
        (b"PYTHONPATH", b"/opt/app/lib"),
    ]);
    assert_eq!(std::fs::read(out.join("environ")).unwrap(), environ_bytes(&want));
}

#[test]
fn setenv_replaces_an_inherited_name_and_the_last_value_wins() {
    let out = scratch("setenv-over");
    let script = r#"cat /proc/$$/environ > "$OUT/environ"; printf '%s\n' "$@" > "$OUT/args""#;
    let inherited = env(&[("OUT", out.as_os_str().as_bytes()), ("A", b"inherited"), ("Z", b"z")]);
    let setenv = vec![os("A=first"), os("R=one"), os("A=last"), os("R=two")];
    let tok = token();
    // A `--setenv` after `--` is the program's own argument.
    let c = waiter_setenv(&tok, &setenv, &[sh().as_str(), "-c", script, "sh0", "--setenv", "B=no"], &inherited);
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    let want: Vec<(&[u8], &[u8])> = vec![
        (b"OUT", out.as_os_str().as_bytes()),
        (b"Z", b"z"),
        (b"A", b"last"),
        (b"R", b"two"),
    ];
    assert_eq!(std::fs::read(out.join("environ")).unwrap(), environ_bytes(&want));
    assert_eq!(std::fs::read_to_string(out.join("args")).unwrap(), "--setenv\nB=no\n");
}

#[test]
fn a_program_name_is_looked_up_in_the_path_setenv_gives() {
    let dir = scratch("setenv-path");
    let prog = dir.join("bfxprog");
    std::fs::write(&prog, format!("#!{}\necho ran > \"$OUT/ran\"\n", sh())).unwrap();
    std::fs::set_permissions(&prog, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = scratch("setenv-path-out");
    let tok = token();
    let path = format!("PATH=/nonexistent:{}", dir.display());
    let c = waiter_setenv(&tok, &[os(&path)], &["bfxprog"], &env(&[("PATH", b"/bin:/usr/bin"), ("OUT", out.as_os_str().as_bytes())]));
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    assert!(out.join("ran").exists());
}

#[test]
fn setenv_starts_a_program_where_the_image_has_no_env_or_shell() {
    // A distroless-style root: bfx at /.bfx/bfx and one static program, no
    // /usr/bin/env, no shell, no libc, no /proc. Needs root for chroot.
    // SAFETY: geteuid only reads.
    if unsafe { libc::geteuid() } != 0 {
        eprintln!("needs root for chroot; skipped");
        return;
    }
    let root = scratch("distroless");
    std::fs::create_dir_all(root.join(".bfx")).unwrap();
    std::fs::copy(BFX, root.join(".bfx/bfx")).unwrap();
    std::fs::copy(std::env::current_exe().unwrap(), root.join("envdump")).unwrap();
    assert!(!root.join("usr/bin/env").exists() && !root.join("bin/sh").exists());
    let start = |tok: &str, setenv: &[&str], argv: &[&str]| -> Child {
        let root = root.clone();
        let mut c = Command::new(AS);
        c.args(["hold", "--token", tok]);
        for pair in setenv {
            c.args(["--setenv", pair]);
        }
        c.arg("--").args(argv).env_clear();
        // SAFETY: chroot and chdir between fork and exec, with a path built
        // before the fork.
        let path = std::ffi::CString::new(root.as_os_str().as_bytes()).unwrap();
        unsafe {
            c.pre_exec(move || {
                if libc::chroot(path.as_ptr()) != 0 || libc::chdir(c"/".as_ptr()) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap()
    };
    // Restoring a value through the image's env program cannot start here.
    let tok = token();
    let c = start(&tok, &[], &["/usr/bin/env", "PYTHONPATH=/opt/app/lib", "/envdump"]);
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(127), "bfx hold: cannot start /usr/bin/env: No such file or directory\n"));
    // bfx sets it itself.
    let tok = token();
    let mut c = start(&tok, &["PYTHONPATH=/opt/app/lib", "BFX_HOLD_ENVDUMP=1"], &["/envdump"]);
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let mut dumped = Vec::new();
    c.stdout.take().unwrap().read_to_end(&mut dumped).unwrap();
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    assert_eq!(dumped, environ_bytes(&[(b"PYTHONPATH", b"/opt/app/lib"), (b"BFX_HOLD_ENVDUMP", b"1")]));
}

#[test]
fn run_if_released_runs_the_command_once_the_waiter_is_released() {
    let out = scratch("rir-run");
    let tok = token();
    let released = released_waiter(&tok);
    // A held waiter of another token is not this one's.
    let other = token();
    let held = waiter(&other, &["/bin/true"], &[]);
    wait_ready(&other);
    let (code, stdout, err) = run_if_released(&tok, &[sh().as_str(), "-c", r#"echo ran > "$OUT/ran"; echo ok"#], &env(&[("OUT", out.as_os_str().as_bytes())]));
    assert_eq!((code, stdout.as_slice(), err.as_str()), (0, &b"ok\n"[..], ""));
    assert!(out.join("ran").exists());
    // A token no waiter ever held: nothing holds it.
    let (code, _, _) = run_if_released(&token(), &["/bin/true"], &[]);
    assert_eq!(code, 0);
    end(released);
    end(held);
}

#[test]
fn run_if_released_runs_nothing_while_a_waiter_holds_the_token() {
    let out = scratch("rir-held");
    let script = r#"echo ran > "$OUT/ran""#;
    let e = env(&[("OUT", out.as_os_str().as_bytes())]);
    let refused = (75, Vec::new(), "bfx hold: the waiter for this token has not been released; nothing ran\n".to_string());
    // Never released.
    let tok = token();
    let held = waiter(&tok, &["/bin/true"], &[]);
    wait_ready(&tok);
    assert_eq!(run_if_released(&tok, &[sh().as_str(), "-c", script], &e), refused);
    // Released, then a waiter with the same token again (a restarted container's).
    let tok2 = token();
    let released = released_waiter(&tok2);
    let again = waiter(&tok2, &["/bin/true"], &[]);
    let end_at = Instant::now() + Duration::from_secs(10);
    while control("--ready", &tok2).0 != 0 {
        assert!(Instant::now() < end_at, "the second waiter never became ready");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(run_if_released(&tok2, &[sh().as_str(), "-c", script], &e), refused);
    assert!(!out.join("ran").exists(), "the command ran");
    for c in [held, released, again] {
        end(c);
    }
}

#[test]
fn run_if_released_execs_the_exact_argv_and_environment() {
    let out = scratch("rir-exact");
    let tok = token();
    let released = released_waiter(&tok);
    let script = r#"cat /proc/$$/cmdline > "$OUT/cmdline"; cat /proc/$$/environ > "$OUT/environ"; :"#;
    let argv: Vec<OsString> = vec![os(&sh()), os("-c"), os(script), os("sh0"), os("two words"), os(""), OsString::from_vec(b"\xff\xfe raw".to_vec())];
    let e = env(&[("OUT", out.as_os_str().as_bytes()), ("SPACES", b"x  y "), ("BYTES", b"\xff\xfe"), ("EMPTY", b""), ("EQ", b"a=b")]);
    let (code, _, err) = run_if_released(&tok, &argv, &e);
    assert_eq!((code, err.as_str()), (0, ""));
    let mut want_cmd = Vec::new();
    for a in &argv {
        want_cmd.extend_from_slice(a.as_bytes());
        want_cmd.push(0);
    }
    assert_eq!(std::fs::read(out.join("cmdline")).unwrap(), want_cmd);
    let mut sorted: Vec<(&[u8], &[u8])> = e.iter().map(|(k, v)| (k.as_bytes(), v.as_bytes())).collect();
    sorted.sort();
    assert_eq!(std::fs::read(out.join("environ")).unwrap(), environ_bytes(&sorted));
    // SIGPIPE, which Rust's runtime ignores, is back at its default.
    let cat = if Path::new("/bin/cat").exists() { "/bin/cat" } else { "/usr/bin/cat" };
    let (code, status, _) = run_if_released(&tok, &[cat, "/proc/self/status"], &[]);
    assert_eq!(code, 0);
    let status = String::from_utf8(status).unwrap();
    let ign = u64::from_str_radix(status.lines().find_map(|l| l.strip_prefix("SigIgn:")).unwrap().trim(), 16).unwrap();
    assert_eq!(ign & bit(libc::SIGPIPE), 0);
    end(released);
}

#[test]
fn run_if_released_looks_a_name_up_in_path() {
    let dir = scratch("rir-path");
    let prog = dir.join("bfxprog");
    std::fs::write(&prog, format!("#!{}\necho found\n", sh())).unwrap();
    std::fs::set_permissions(&prog, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!("/nonexistent:{}", dir.display());
    let (code, stdout, _) = run_if_released(&token(), &["bfxprog"], &env(&[("PATH", path.as_bytes())]));
    assert_eq!((code, stdout.as_slice()), (0, &b"found\n"[..]));
    let (code, _, err) = run_if_released(&token(), &["bfx-no-such-program"], &env(&[("PATH", b"/bin:/usr/bin")]));
    assert_eq!((code, err.as_str()), (127, "bfx hold: cannot start bfx-no-such-program: No such file or directory\n"));
}

#[test]
fn run_if_released_reads_only_proc_before_the_command() {
    let Some(strace) = ["/usr/bin/strace", "/bin/strace"].into_iter().find(|p| Path::new(p).exists()) else {
        eprintln!("strace is not installed; skipped");
        return;
    };
    let dir = scratch("rir-strace");
    let tok = token();
    for held in [false, true] {
        let w = held.then(|| {
            let c = waiter(&tok, &["/bin/true"], &[]);
            wait_ready(&tok);
            c
        });
        let log = dir.join(format!("log-{held}"));
        let st = Command::new(strace)
            .args(["-f", "-qq", "-e", "trace=%file,%process", "-o"])
            .arg(&log)
            .args(["--", BFX, "hold", "--run-if-released", "--token", &tok, "--", "/bin/true"])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert_eq!(st.code(), Some(if held { 75 } else { 0 }));
        let text = std::fs::read_to_string(&log).unwrap();
        let mut execs = 0;
        for line in text.lines() {
            let call = line.split_once(' ').map(|x| x.1).unwrap_or(line).trim_start();
            let name = call.split('(').next().unwrap_or("");
            if name == "execve" {
                execs += 1;
                if execs == 2 {
                    assert!(!held && call.starts_with("execve(\"/bin/true\""), "{line}");
                    break;
                }
                continue;
            }
            if ["clone", "clone3", "fork", "vfork", "exit", "exit_group", "wait4", "waitid"].contains(&name) || call.starts_with("+++") {
                continue;
            }
            let path = call.split_once('"').map(|x| x.1).unwrap_or("");
            assert!(path.starts_with("/proc"), "before the command: {line}");
        }
        assert_eq!(execs, if held { 1 } else { 2 }, "{text}");
        if let Some(c) = w {
            end(c);
        }
    }
}

#[test]
fn the_program_starts_with_the_dispositions_bfx_changed_back_at_default() {
    // cat changes no signal: its status at start is what bfx left. (A shell
    // is no witness: BusyBox ash ignores SIGQUIT itself.) Everything bfx
    // touched is SIG_DFL, whatever the waiter inherited, and the mask is clear.
    let cat = if Path::new("/bin/cat").exists() { "/bin/cat" } else { "/usr/bin/cat" };
    let tok = token();
    let c = waiter(&tok, &[cat, "/proc/self/status"], &[]);
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let mut c = c;
    let mut status = String::new();
    c.stdout.take().unwrap().read_to_string(&mut status).unwrap();
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    let field = |f: &str| u64::from_str_radix(status.lines().find_map(|l| l.strip_prefix(f)).unwrap().trim(), 16).unwrap();
    assert_eq!(field("SigBlk:"), 0, "the mask is clear");
    assert_eq!(field("SigCgt:"), 0, "no handler survives the exec");
    for s in [libc::SIGUSR1, libc::SIGPIPE, libc::SIGTERM, libc::SIGINT, libc::SIGQUIT, libc::SIGHUP] {
        assert_eq!(field("SigIgn:") & bit(s), 0, "signal {s} is ignored in the program");
    }
}

fn inotify_on(dir: &Path) -> i32 {
    // SAFETY: a new inotify descriptor and a watch on a NUL-terminated path.
    unsafe {
        let fd = libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC);
        assert!(fd >= 0);
        let p = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert!(libc::inotify_add_watch(fd, p.as_ptr(), libc::IN_OPEN | libc::IN_ACCESS | libc::IN_ATTRIB) >= 0);
        fd
    }
}

fn inotify_events(fd: i32) -> usize {
    let mut buf = [0u8; 4096];
    // SAFETY: reading into a local buffer from our own descriptor.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
    if n <= 0 {
        0
    } else {
        n as usize
    }
}

#[test]
fn path_is_searched_only_after_the_release() {
    let dir = scratch("path");
    let prog = dir.join("bfxprog");
    std::fs::write(&prog, format!("#!{}\necho ran > \"$OUT/ran\"\n", sh())).unwrap();
    std::fs::set_permissions(&prog, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = scratch("path-out");
    let ino = inotify_on(&dir);
    let tok = token();
    let path = format!("/nonexistent:{}", dir.display());
    let c = waiter(&tok, &["bfxprog"], &env(&[("PATH", path.as_bytes()), ("OUT", out.as_os_str().as_bytes())]));
    wait_ready(&tok);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(inotify_events(ino), 0, "the PATH folder was read before the release");
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    assert!(inotify_events(ino) > 0, "the program was opened from PATH");
    assert!(out.join("ran").exists());
    // No PATH at all: Docker's default.
    let out2 = scratch("default-path");
    let tok = token();
    let c = waiter(&tok, &["sh", "-c", r#"echo ok > "$OUT/ran""#], &env(&[("OUT", out2.as_os_str().as_bytes())]));
    wait_ready(&tok);
    assert_eq!(control("--release", &tok).0, 0);
    let (st, err) = finish(c);
    assert_eq!((st.code(), err.as_str()), (Some(0), ""));
    assert!(out2.join("ran").exists());
}

#[test]
fn a_stop_signal_before_the_release_exits_128_plus_its_number() {
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGQUIT, libc::SIGHUP] {
        let tok = token();
        let c = waiter(&tok, &["/bin/true"], &[]);
        wait_ready(&tok);
        // SAFETY: signalling our own child.
        assert_eq!(unsafe { libc::kill(c.id() as i32, sig) }, 0);
        let (st, err) = finish(c);
        assert_eq!((st.code(), st.signal(), err.as_str()), (Some(128 + sig), None, ""), "signal {sig}");
    }
}

#[test]
fn a_failed_exec_exits_127_or_126_with_one_line() {
    let dir = scratch("fail");
    let noexec = dir.join("noexec");
    std::fs::write(&noexec, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&noexec, std::fs::Permissions::from_mode(0o644)).unwrap();
    let junk = dir.join("junk");
    std::fs::write(&junk, b"\x01\x02 not a program\n").unwrap();
    std::fs::set_permissions(&junk, std::fs::Permissions::from_mode(0o755)).unwrap();
    let cases: Vec<(String, Vec<(OsString, OsString)>, i32, &str)> = vec![
        ("/nonexistent/prog".into(), vec![], 127, "No such file or directory"),
        ("bfx-no-such-program".into(), env(&[("PATH", b"/bin:/usr/bin")]), 127, "No such file or directory"),
        (noexec.display().to_string(), vec![], 126, "Permission denied"),
        (junk.display().to_string(), vec![], 126, "Exec format error"),
    ];
    for (prog, e, code, why) in cases {
        let tok = token();
        let c = waiter(&tok, &[prog.as_str()], &e);
        wait_ready(&tok);
        assert_eq!(control("--release", &tok).0, 0);
        let (st, err) = finish(c);
        assert_eq!(st.code(), Some(code), "{prog}");
        assert_eq!(err, format!("bfx hold: cannot start {prog}: {why}\n"));
    }
}

#[test]
fn a_second_release_is_ignored_and_the_program_starts_with_defaults() {
    // The program handles SIGUSR1 itself: it sees it delivered, not ignored
    // or blocked.
    let out = scratch("usr1");
    let script = r#"trap 'echo got > "$OUT/got"' USR1; kill -USR1 $$; cat /proc/$$/status > "$OUT/status"; echo done > "$OUT/done""#;
    for round in 0..20 {
        let tok = token();
        let c = waiter(&tok, &[sh().as_str(), "-c", script], &env(&[("OUT", out.as_os_str().as_bytes())]));
        let pid = c.id() as i32;
        wait_ready(&tok);
        // Two releases back to back: the second lands while bfx is still
        // the waiter, and is dropped.
        // SAFETY: signalling our own child.
        unsafe {
            libc::kill(pid, libc::SIGUSR1);
            libc::kill(pid, libc::SIGUSR1);
        }
        let (st, err) = finish(c);
        assert_eq!((st.code(), st.signal(), err.as_str()), (Some(0), None, ""), "round {round}");
        assert!(out.join("got").exists() && out.join("done").exists(), "round {round}");
        let status = std::fs::read_to_string(out.join("status")).unwrap();
        let field = |f: &str| u64::from_str_radix(status.lines().find_map(|l| l.strip_prefix(f)).unwrap().trim(), 16).unwrap();
        assert_eq!(field("SigIgn:") & bit(libc::SIGUSR1), 0);
        assert_eq!(field("SigBlk:"), 0);
        for f in ["got", "done", "status"] {
            let _ = std::fs::remove_file(out.join(f));
        }
    }
}

#[test]
fn ready_and_release_need_exactly_one_waiter_with_the_token() {
    let tok = token();
    let a = waiter(&tok, &["/bin/true"], &[]);
    wait_ready(&tok);
    // A wrong token: not found, nothing sent.
    let other = token();
    assert_eq!(control("--ready", &other), (1, String::new()));
    assert_eq!(control("--release", &other), (1, String::new()));
    // A second waiter with the same token: ambiguous, nothing sent to either.
    let b = waiter(&tok, &["/bin/true"], &[]);
    let end = Instant::now() + Duration::from_secs(10);
    while control("--ready", &tok).0 != 3 {
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(control("--release", &tok).0, 3);
    std::thread::sleep(Duration::from_millis(200));
    for c in [&a, &b] {
        let stat = std::fs::read_to_string(format!("/proc/{}/stat", c.id())).unwrap();
        assert!(stat.contains(") S "), "a waiter was released or ended: {stat}");
    }
    // One left (neither is PID 1): released.
    // SAFETY: stopping our own child.
    unsafe { libc::kill(b.id() as i32, libc::SIGTERM) };
    let (st, _) = finish(b);
    assert_eq!(st.code(), Some(128 + libc::SIGTERM));
    assert_eq!(control("--release", &tok).0, 0);
    let (st, _) = finish(a);
    assert_eq!(st.code(), Some(0));
}

#[test]
fn the_binary_is_static() {
    let data = std::fs::read(BFX).unwrap();
    assert_eq!(&data[..4], b"\x7fELF");
    assert_eq!(data[4], 2, "64-bit");
    let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as usize;
    let (phoff, phentsize, phnum) = (u64_at(0x20), u16_at(0x36), u16_at(0x38));
    for i in 0..phnum {
        assert_ne!(u32_at(phoff + i * phentsize), 3, "PT_INTERP: a dynamic loader would run");
    }
}

#[test]
fn nothing_outside_proc_self_is_opened_before_the_release() {
    let Some(strace) = ["/usr/bin/strace", "/bin/strace"].into_iter().find(|p| Path::new(p).exists()) else {
        eprintln!("strace is not installed; skipped");
        return;
    };
    let dir = scratch("strace");
    let log = dir.join("log");
    let tok = token();
    let mut c = Command::new(strace)
        .args(["-f", "-qq", "-e", "trace=%file,%process", "-e", "signal=SIGUSR1", "-o"])
        .arg(&log)
        // By name: a PATH lookup before the release would show as a stat.
        // A --setenv is applied only at the release, with no file call.
        .args(["--", BFX, "hold", "--token", &tok, "--setenv", "PYTHONPATH=/opt/app/lib", "--", "true"])
        .env_clear()
        .env("PATH", "/bin:/usr/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // strace runs bfx under its own name, so find it by its pid and handlers.
    let end = Instant::now() + Duration::from_secs(10);
    let pid = loop {
        let found = std::fs::read_dir("/proc").unwrap().flatten().find_map(|e| {
            let n: i32 = e.file_name().to_str()?.parse().ok()?;
            let cmd = std::fs::read(format!("/proc/{n}/cmdline")).ok()?;
            let f: Vec<&[u8]> = cmd.split(|b| *b == 0).collect();
            (f.len() > 3 && f[0] == BFX.as_bytes() && f[1] == b"hold" && f[3] == tok.as_bytes()).then_some(n)
        });
        if let Some(p) = found {
            let cgt = status_field(p as u32, "SigCgt:");
            if FIVE.iter().all(|s| cgt & bit(*s) != 0) {
                break p;
            }
        }
        assert!(Instant::now() < end, "no waiter under strace");
        std::thread::sleep(Duration::from_millis(5));
    };
    std::thread::sleep(Duration::from_millis(200));
    // SAFETY: signalling our own grandchild.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGUSR1) }, 0);
    let st = c.wait().unwrap();
    assert!(st.success());
    let text = std::fs::read_to_string(&log).unwrap();
    let mut before = Vec::new();
    let mut released = false;
    for line in text.lines() {
        if line.contains("--- SIGUSR1") {
            released = true;
            break;
        }
        before.push(line);
    }
    assert!(released, "{text}");
    // Traced: every call that takes a file name (%file) and the process
    // calls (%process). Before the release only the process calls that take
    // no name may appear, past the first execve (strace starting bfx), and a
    // file call must name a path under /proc/self. A path strace can't read
    // (bfx is not dumpable, so an unprivileged strace sees a pointer) counts
    // as a file opened.
    const NO_NAME: [&str; 8] = ["clone", "clone3", "fork", "vfork", "exit", "exit_group", "wait4", "waitid"];
    let mut first_exec = true;
    for line in before {
        let call = line.split_once(' ').map(|x| x.1).unwrap_or(line).trim_start();
        let name = call.split('(').next().unwrap_or("");
        if name == "execve" && first_exec {
            first_exec = false;
            continue;
        }
        if NO_NAME.contains(&name) || call.starts_with("+++") {
            continue;
        }
        let path = call.split_once('"').map(|x| x.1).unwrap_or("");
        assert!(path.starts_with("/proc/self/"), "before the release: {line}");
    }
}

#[test]
fn a_process_with_the_waiters_cmdline_but_not_its_handlers_is_not_ready() {
    // This test binary, started as `/.bfx/bfx hold --token TOKEN -- x`,
    // blocks before the test harness starts (`impostor` below): the cmdline
    // matches, the five handlers are not there. (A shell won't do: BusyBox
    // picks its applet by argv[0].)
    let tok = token();
    let mut c = Command::new(std::env::current_exe().unwrap())
        .arg0(AS)
        .args(["hold", "--token", &tok, "--", "x"])
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let cmd = std::fs::read(format!("/proc/{}/cmdline", c.id())).unwrap_or_default();
        if cmd.starts_with(AS.as_bytes()) {
            break;
        }
        assert!(Instant::now() < end, "the impostor never started");
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(c.try_wait().unwrap().is_none(), "the impostor exited");
    assert_eq!(status_field(c.id(), "SigCgt:") & bit(libc::SIGUSR1), 0);
    assert_eq!(control("--ready", &tok), (1, String::new()));
    assert_eq!(control("--release", &tok), (1, String::new()));
    std::thread::sleep(Duration::from_millis(200));
    assert!(c.try_wait().unwrap().is_none(), "--release signalled a process that is not ready");
    c.kill().unwrap();
    c.wait().unwrap();
}

/// Before the test harness reads its arguments: when this binary was started
/// with a waiter's argv (`... hold --token ...`), it waits for SIGKILL with no
/// handler installed, as the lookalike row needs. Any other start goes on to
/// the tests.
#[used]
#[link_section = ".init_array"]
static IMPOSTOR: extern "C" fn() = impostor;

extern "C" fn impostor() {
    // Started by a release with BFX_HOLD_ENVDUMP set: write the environment
    // to stdout, NUL-terminated entries, and exit (no /proc needed).
    if std::env::var_os("BFX_HOLD_ENVDUMP").is_some() {
        let mut out = Vec::new();
        for (k, v) in std::env::vars_os() {
            out.extend_from_slice(k.as_bytes());
            out.push(b'=');
            out.extend_from_slice(v.as_bytes());
            out.push(0);
        }
        // SAFETY: writing a local buffer to stdout, then _exit.
        unsafe {
            libc::write(1, out.as_ptr().cast(), out.len());
            libc::_exit(0);
        }
    }
    let Ok(cmd) = std::fs::read("/proc/self/cmdline") else { return };
    let f: Vec<&[u8]> = cmd.split(|b| *b == 0).collect();
    if f.len() > 2 && f[1] == b"hold" && f[2] == b"--token" {
        loop {
            // SAFETY: pause only waits for a signal.
            unsafe { libc::pause() };
        }
    }
}
