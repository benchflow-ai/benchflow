//! The few calls rustix does not wrap, through musl (already linked by std).
//! Everything else goes through rustix.

use std::io;

/// kill(-1, sig): every process the caller may signal except itself and
/// PID 1, in one pass under the kernel's task-list lock.
pub fn kill_all(sig: i32) -> io::Result<()> {
    // SAFETY: a plain system call with integer arguments.
    let r = unsafe { libc::kill(-1, sig) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// kill(pid, sig) with a raw signal number.
pub fn kill_one(pid: i32, sig: i32) -> io::Result<()> {
    if pid <= 0 {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    // SAFETY: a plain system call with integer arguments; pid > 0.
    let r = unsafe { libc::kill(pid, sig) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// kill(-pgid, sig): a whole process group.
pub fn kill_group(pgid: i32, sig: i32) -> io::Result<()> {
    if pgid <= 1 {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    // SAFETY: a plain system call with integer arguments; the group is > 1.
    let r = unsafe { libc::kill(-pgid, sig) };
    if r == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

pub fn getresuid() -> (u32, u32, u32) {
    let (mut r, mut e, mut s) = (0, 0, 0);
    // SAFETY: three valid out-pointers.
    unsafe { libc::getresuid(&mut r, &mut e, &mut s) };
    (r, e, s)
}

pub fn getresgid() -> (u32, u32, u32) {
    let (mut r, mut e, mut s) = (0, 0, 0);
    // SAFETY: three valid out-pointers.
    unsafe { libc::getresgid(&mut r, &mut e, &mut s) };
    (r, e, s)
}

/// close_range(low, ~0, 0); false when the kernel lacks it (before 5.9).
pub fn close_from(low: u32) -> bool {
    // SAFETY: closes descriptors this process does not use past `low`.
    unsafe { libc::syscall(libc::SYS_close_range, low as libc::c_long, u32::MAX as libc::c_long, 0 as libc::c_long) == 0 }
}
