//! `bfx launch -- PROGRAM [ARG...]`: start PROGRAM held, and nothing
//! else.
//!
//! The verifier's test.sh runs the agent's code as root. Its launch must keep
//! NET_ADMIN, NET_RAW and SYS_PTRACE from it whatever the task image holds:
//! util-linux setpriv is missing from Alpine, BusyBox and distroless images
//! (BusyBox's own setpriv applet has no --bounding-set). In order:
//!
//! 1. as root, NET_ADMIN (12), NET_RAW (13) and SYS_PTRACE (19) leave the
//!    bounding set, so no later exec can bring them back;
//! 2. the ambient set is cleared and the inheritable set emptied;
//! 3. no_new_privs is set, so no setuid program or file capability adds any;
//! 4. PROGRAM replaces bfx, with bfx's arguments and environment as given.
//!
//! A step that fails refuses the launch: nothing is started and the exit
//! status is 97, the verifier's "launch refused". The started shell checks
//! its own capabilities again before test.sh (verifier_network._caps_check).

use std::ffi::OsString;
use std::io::{self, Write};
use std::os::unix::process::CommandExt;

/// The exit status of a refused launch (`VERIFIER_LAUNCH_REFUSED`).
pub const REFUSED: i32 = 97;
/// CAP_NET_ADMIN, CAP_NET_RAW, CAP_SYS_PTRACE.
pub const DROPPED: [u32; 3] = [12, 13, 19];

const CAP_VERSION_3: u32 = 0x2008_0522;

#[repr(C)]
struct CapHeader {
    version: u32,
    pid: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}

fn refuse(why: String) -> i32 {
    let _ = io::stderr().write_all(format!("BF verifier-launch refused: bfx launch: {why}\n").as_bytes());
    REFUSED
}

fn prctl(option: libc::c_int, arg: libc::c_ulong) -> io::Result<()> {
    // SAFETY: prctl with integer arguments only; no memory is passed.
    if unsafe { libc::prctl(option, arg, 0 as libc::c_ulong, 0 as libc::c_ulong, 0 as libc::c_ulong) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Steps 1-3. Nothing is started here.
pub fn hold() -> Result<(), String> {
    // SAFETY: plain getters.
    let root = unsafe { libc::getuid() } == 0 || unsafe { libc::geteuid() } == 0;
    if root {
        for cap in DROPPED {
            prctl(libc::PR_CAPBSET_DROP, cap as libc::c_ulong)
                .map_err(|e| format!("capability {cap} stays in the bounding set ({e})"))?;
        }
    }
    if let Err(e) = prctl(libc::PR_CAP_AMBIENT, libc::PR_CAP_AMBIENT_CLEAR_ALL as libc::c_ulong) {
        // EINVAL: a kernel before 4.3, which has no ambient set to clear.
        if e.raw_os_error() != Some(libc::EINVAL) {
            return Err(format!("the ambient set cannot be cleared ({e})"));
        }
    }
    let mut header = CapHeader { version: CAP_VERSION_3, pid: 0 };
    let mut data = [CapData::default(); 2];
    // SAFETY: header and two data records, as capget(2) version 3 takes.
    if unsafe { libc::syscall(libc::SYS_capget, &mut header as *mut CapHeader, data.as_mut_ptr()) } != 0 {
        return Err(format!("capget failed ({})", io::Error::last_os_error()));
    }
    for d in data.iter_mut() {
        d.inheritable = 0;
    }
    // SAFETY: as above; only the inheritable set changes (a reduction).
    if unsafe { libc::syscall(libc::SYS_capset, &mut header as *mut CapHeader, data.as_ptr()) } != 0 {
        return Err(format!("the inheritable set cannot be emptied ({})", io::Error::last_os_error()));
    }
    prctl(libc::PR_SET_NO_NEW_PRIVS, 1).map_err(|e| format!("no_new_privs cannot be set ({e})"))?;
    Ok(())
}

pub fn main(argv: &[OsString]) -> i32 {
    let Some((program, rest)) = argv.split_first() else {
        return refuse("no program to start".into());
    };
    if let Err(why) = hold() {
        return refuse(why);
    }
    let err = std::process::Command::new(program).args(rest).exec();
    let _ = io::stderr().write_all(format!("bfx launch: {}: {err}\n", program.to_string_lossy()).as_bytes());
    if err.kind() == io::ErrorKind::NotFound {
        127
    } else {
        126
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(field: &str) -> String {
        let text = std::fs::read_to_string("/proc/self/status").unwrap();
        text.lines().find_map(|l| l.strip_prefix(field)).unwrap_or("").trim().to_string()
    }

    #[test]
    fn a_held_process_has_no_new_privileges_and_inherits_nothing() {
        // In a child, so the test process itself stays as it was.
        // SAFETY: fork in a test; the child only calls hold, reads and exits.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let ok = hold().is_ok()
                && status("NoNewPrivs:") == "1"
                && u64::from_str_radix(&status("CapInh:"), 16) == Ok(0)
                && u64::from_str_radix(&status("CapAmb:"), 16).unwrap_or(0) == 0
                && (unsafe { libc::geteuid() } != 0
                    || DROPPED.iter().all(|c| u64::from_str_radix(&status("CapBnd:"), 16).unwrap() & (1 << c) == 0));
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        let mut code = 0;
        unsafe { libc::waitpid(pid, &mut code, 0) };
        assert!(libc::WIFEXITED(code) && libc::WEXITSTATUS(code) == 0);
    }

    #[test]
    fn no_program_refuses() {
        assert_eq!(main(&[]), REFUSED);
    }
}
