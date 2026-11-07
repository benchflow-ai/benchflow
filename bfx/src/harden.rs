//! What bfx does first, before it reads any input.
//!
//! 1. PR_SET_DUMPABLE 0: no process of another uid (and no unprivileged
//!    process of root's in another user namespace) may ptrace it or read its
//!    memory, including in the window of a later identity change.
//! 2. Every inherited descriptor above 2 is closed.
//! 3. The panic hook prints a fixed line; nothing reads the environment
//!    (bfx never consults its own environment: no locale, no PATH, no
//!    RUST_* variables).

use crate::sys;
use std::io::Write;

pub fn startup() -> Result<(), &'static str> {
    rustix::process::set_dumpable_behavior(rustix::process::DumpableBehavior::NotDumpable).map_err(|_| "cannot mark itself non-dumpable")?;
    close_inherited();
    std::panic::set_hook(Box::new(|_| {
        let _ = std::io::stderr().write_all(b"bfx: internal error\n");
    }));
    Ok(())
}

fn close_inherited() {
    if sys::close_from(3) {
        return;
    }
    // Before Linux 5.9: list /proc/self/fd, else close up to the limit.
    use rustix::fs::{self, Mode, OFlags};
    if let Ok(dir) = fs::open("/proc/self/fd", OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty()) {
        let own = rustix::fd::AsRawFd::as_raw_fd(&dir);
        let mut fds = Vec::new();
        if let Ok(mut d) = fs::Dir::read_from(&dir) {
            while let Some(Ok(e)) = d.read() {
                if let Ok(n) = std::str::from_utf8(e.file_name().to_bytes()).unwrap_or("").parse::<i32>() {
                    fds.push(n);
                }
            }
        }
        drop(dir);
        for fd in fds {
            if fd > 2 && fd != own {
                // SAFETY: closing a descriptor bfx did not open and does not use.
                unsafe { libc::close(fd) };
            }
        }
        return;
    }
    let max = rustix::process::getrlimit(rustix::process::Resource::Nofile).current.unwrap_or(1 << 20).min(1 << 20);
    for fd in 3..max as i32 {
        // SAFETY: as above.
        unsafe { libc::close(fd) };
    }
}
