//! Who made a sandbox, so a sweep removes a crashed run's leftovers and
//! never a live run's sandboxes.
//!
//! Every container and Daytona sandbox BenchFlow creates carries five
//! labels:
//!
//! - [`OWNER_LABEL`](crate::OWNER_LABEL): the owner scope, `bench` unless
//!   `BENCHFLOW_OWNER` names another (teams sharing one Daytona key or one
//!   Docker daemon can keep their sweeps apart this way);
//! - [`HOST_LABEL`](crate::HOST_LABEL): this machine's host name;
//! - [`INSTALL_LABEL`]: a random id kept in this machine's BenchFlow cache
//!   folder (with the PID namespace on Linux);
//! - [`USER_LABEL`]: the user id of the BenchFlow process;
//! - [`PID_LABEL`](crate::PID_LABEL): the id of the BenchFlow process.
//!
//! A sandbox is a leftover when its owner scope is ours, its host, install
//! and user are this process's, and no process with its pid is alive here.
//! Host names alone collide (VMs named `localhost`, Macs named
//! `MacBook-Pro.local`), and a pid is invisible across PID namespaces and
//! users (`hidepid`): without the install and user labels another live
//! run's sandboxes could read as leftovers. A pid that is alive (even one
//! reused by an unrelated process) keeps the sandbox: a sweep may miss
//! garbage, but it never deletes a live run's sandbox.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::{HOST_LABEL, OWNER_LABEL, PID_LABEL};

/// Environment variable that overrides the owner scope.
pub const OWNER_SCOPE_ENV: &str = "BENCHFLOW_OWNER";

const DEFAULT_SCOPE: &str = "bench";
const MAX_SCOPE_LEN: usize = 48;

/// The owner scope: `BENCHFLOW_OWNER` (letters, digits, `.`, `_`, `-`; other
/// characters become `-`) or `bench`.
pub fn scope() -> String {
    std::env::var(OWNER_SCOPE_ENV)
        .ok()
        .map(|raw| sanitize_scope(&raw))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_SCOPE.to_string())
}

fn sanitize_scope(raw: &str) -> String {
    let mapped: String = raw
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    mapped
        .trim_matches(|c| c == '-' || c == '.')
        .chars()
        .take(MAX_SCOPE_LEN)
        .collect()
}

/// Where this process's pids mean something, read once, for names that
/// must tell apart runs whose pids could be confused (checkpoints): the
/// machine id (`/etc/machine-id` or D-Bus's copy on Linux, the platform
/// UUID on macOS), the host name, and on Linux the PID namespace. Two
/// machines with one host name (laptops both called `macbook`) differ by
/// machine id; two containers from one image that baked a machine id in
/// differ by host name and PID namespace. An id that is not 32 hex digits
/// (an image's empty or `uninitialized` file) is left out.
pub fn machine_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let mut parts = Vec::new();
        if let Some(id) = read_machine_id() {
            parts.push(format!("machine-id:{id}"));
        }
        parts.push(format!("host:{}", hostname()));
        if let Ok(ns) = std::fs::read_link("/proc/self/ns/pid") {
            parts.push(format!("pid-ns:{}", ns.to_string_lossy()));
        }
        parts.join(" ")
    })
}

fn read_machine_id() -> Option<String> {
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Some(id) = usable_machine_id(&text)
        {
            return Some(id);
        }
    }
    if cfg!(target_os = "macos") {
        let out = std::process::Command::new("/usr/sbin/ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(rest) = line.split("\"IOPlatformUUID\"").nth(1) {
                let id = rest.trim().trim_start_matches('=').trim().trim_matches('"');
                if let Some(id) = usable_machine_id(id) {
                    return Some(id);
                }
            }
        }
    }
    None
}

/// A machine id as 32 hex digits (dashes dropped, lowercased), or `None`.
fn usable_machine_id(text: &str) -> Option<String> {
    let id: String = text
        .trim()
        .chars()
        .filter(|c| *c != '-')
        .collect::<String>()
        .to_ascii_lowercase();
    (id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()) && id.bytes().any(|b| b != b'0'))
        .then_some(id)
}

/// This machine's host name, read once.
pub fn hostname() -> &'static str {
    static HOST: OnceLock<String> = OnceLock::new();
    HOST.get_or_init(read_hostname)
}

fn read_hostname() -> String {
    if let Ok(name) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
        let name = name.trim();
        if !name.is_empty() {
            return name.to_string();
        }
    }
    for program in ["hostname", "uname"] {
        let mut cmd = std::process::Command::new(program);
        if program == "uname" {
            cmd.arg("-n");
        }
        if let Ok(out) = cmd.output()
            && out.status.success()
        {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !name.is_empty() {
                return name;
            }
        }
    }
    "unknown-host".to_string()
}

/// Label: this installation's random id (see [`install`]).
pub const INSTALL_LABEL: &str = "ai.benchflow.install";
/// Label: the user id of the process that made the sandbox.
pub const USER_LABEL: &str = "ai.benchflow.uid";

/// This installation: a random id kept in BenchFlow's cache folder
/// (`BENCHFLOW_CACHE_DIR`, `$XDG_CACHE_HOME/benchflow` or
/// `~/.cache/benchflow`), plus the PID namespace on Linux, where pids are
/// only comparable inside one namespace. Read once. When the id cannot be
/// kept, a new one per process: this process's leftovers are then left to
/// the age rules, never mistaken for another's.
pub fn install() -> &'static str {
    static INSTALL: OnceLock<String> = OnceLock::new();
    INSTALL.get_or_init(|| {
        let id = read_install_id(&cache_dir())
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        match std::fs::read_link("/proc/self/ns/pid") {
            Ok(ns) => {
                let ns: String = ns
                    .to_string_lossy()
                    .chars()
                    .filter(char::is_ascii_digit)
                    .collect();
                format!("{id}-{ns}")
            }
            Err(_) => id,
        }
    })
}

fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("BENCHFLOW_CACHE_DIR") {
        return dir.into();
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return std::path::PathBuf::from(xdg).join("benchflow");
    }
    dirs::home_dir()
        .map(|h| h.join(".cache"))
        .unwrap_or_else(std::env::temp_dir)
        .join("benchflow")
}

/// The id in `dir/install-id`, made when missing. `None` when it can be
/// neither read nor made.
fn read_install_id(dir: &std::path::Path) -> Option<String> {
    let path = dir.join("install-id");
    let valid = |s: &str| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit());
    if let Ok(text) = std::fs::read_to_string(&path)
        && valid(text.trim())
    {
        return Some(text.trim().to_string());
    }
    std::fs::create_dir_all(dir).ok()?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    // Two processes starting at once: the first file written wins.
    let tmp = dir.join(format!("install-id.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &id).ok()?;
    let _ = std::fs::hard_link(&tmp, &path);
    let _ = std::fs::remove_file(&tmp);
    std::fs::read_to_string(&path)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| valid(t))
}

/// The user id of this process.
pub fn user() -> String {
    #[cfg(unix)]
    {
        // SAFETY: getuid never fails and touches no memory.
        #[allow(unsafe_code)]
        let uid = unsafe { libc::getuid() };
        uid.to_string()
    }
    #[cfg(not(unix))]
    {
        "0".to_string()
    }
}

/// The labels a new sandbox carries: owner scope, host, install, user, and
/// this process id.
pub fn labels() -> BTreeMap<String, String> {
    let mut labels = BTreeMap::new();
    labels.insert(OWNER_LABEL.to_string(), scope());
    labels.insert(HOST_LABEL.to_string(), hostname().to_string());
    labels.insert(INSTALL_LABEL.to_string(), install().to_string());
    labels.insert(USER_LABEL.to_string(), user());
    labels.insert(PID_LABEL.to_string(), std::process::id().to_string());
    labels
}

/// Whether a process exists on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Alive,
    Gone,
    /// Neither `/proc` nor `ps` could tell. Treated as alive.
    Unknown,
}

/// Whether process `pid` exists on this machine: `kill(pid, 0)`, where a
/// refusal (EPERM: another user's process) means it is alive. `/proc`
/// hides other users' processes under `hidepid`, so it is not asked.
pub async fn process_state(pid: u32) -> ProcessState {
    if pid == std::process::id() {
        return ProcessState::Alive;
    }
    #[cfg(unix)]
    {
        let Ok(raw) = libc::pid_t::try_from(pid) else {
            return ProcessState::Unknown;
        };
        if raw <= 0 {
            return ProcessState::Unknown;
        }
        // SAFETY: signal 0 sends nothing; it only checks that the process
        // exists, and touches no memory.
        #[allow(unsafe_code)]
        let rc = unsafe { libc::kill(raw, 0) };
        if rc == 0 {
            return ProcessState::Alive;
        }
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ESRCH) => ProcessState::Gone,
            Some(libc::EPERM) => ProcessState::Alive,
            _ => ProcessState::Unknown,
        }
    }
    #[cfg(not(unix))]
    {
        if std::path::Path::new("/proc/self").exists() {
            return if std::path::Path::new(&format!("/proc/{pid}")).exists() {
                ProcessState::Alive
            } else {
                ProcessState::Gone
            };
        }
        // macOS and other systems without /proc: `ps -p` exits 0 when the
        // process exists and 1 when it does not.
        match tokio::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "pid="])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
        {
            Ok(status) if status.success() => ProcessState::Alive,
            Ok(status) if status.code() == Some(1) => ProcessState::Gone,
            _ => ProcessState::Unknown,
        }
    }
}

/// Why a sandbox is (or is not) a leftover, judged from its labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Not ours: no owner label, or another owner scope.
    Foreign,
    /// Ours, but made on another machine; its process cannot be checked here.
    OtherHost,
    /// Ours, made here, by a process that is still running (or may be).
    Live,
    /// Ours, made here, by a process that no longer exists.
    Leftover,
}

/// Judge a sandbox by its labels. `label` looks a label up by name.
pub async fn judge<'a>(label: impl Fn(&str) -> Option<&'a str>) -> Verdict {
    if label(OWNER_LABEL) != Some(scope().as_str()) {
        return Verdict::Foreign;
    }
    if label(HOST_LABEL) != Some(hostname())
        || label(INSTALL_LABEL) != Some(install())
        || label(USER_LABEL) != Some(user().as_str())
    {
        return Verdict::OtherHost;
    }
    let Some(pid) = label(PID_LABEL).and_then(|p| p.trim().parse::<u32>().ok()) else {
        // Ours and from this host, but without a usable pid: leave it to the
        // age-based rules rather than guess.
        return Verdict::Live;
    };
    match process_state(pid).await {
        ProcessState::Gone => Verdict::Leftover,
        ProcessState::Alive | ProcessState::Unknown => Verdict::Live,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_is_sanitized() {
        assert_eq!(sanitize_scope("team a/b"), "team-a-b");
        assert_eq!(sanitize_scope("  ..x.. "), "x");
        assert_eq!(sanitize_scope(&"y".repeat(100)).len(), MAX_SCOPE_LEN);
    }

    #[test]
    fn the_install_id_is_kept_and_reused() {
        let dir = tempfile::tempdir().unwrap();
        let first = read_install_id(dir.path()).unwrap();
        assert_eq!(first.len(), 32);
        assert_eq!(read_install_id(dir.path()).unwrap(), first);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn labels_name_this_process() {
        let labels = labels();
        assert_eq!(labels[PID_LABEL], std::process::id().to_string());
        assert_eq!(labels[HOST_LABEL], hostname());
        assert!(!hostname().is_empty());
    }

    #[tokio::test]
    async fn this_process_is_alive_and_a_dead_one_is_gone() {
        assert_eq!(process_state(std::process::id()).await, ProcessState::Alive);
        // Spawn a short-lived child, wait for it, and check its pid is gone.
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let pid = child.id().unwrap();
        child.wait().await.unwrap();
        assert_eq!(process_state(pid).await, ProcessState::Gone);
    }

    #[tokio::test]
    async fn judges_by_owner_host_and_pid() {
        let me = labels();
        let get = |m: &BTreeMap<String, String>, k: &str| m.get(k).cloned();

        // This process: live.
        let verdict = judge(|k| me.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::Live);

        // A dead process on this host: leftover.
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let pid = child.id().unwrap();
        child.wait().await.unwrap();
        let mut dead = me.clone();
        dead.insert(PID_LABEL.into(), pid.to_string());
        let verdict = judge(|k| dead.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::Leftover);

        // Review of rust/d4-chaos (Daytona 2): the same host name (VMs
        // named `localhost`) but another install, or another user: its pid
        // means nothing here.
        let mut other_install = dead.clone();
        other_install.insert(
            INSTALL_LABEL.into(),
            "0123456789abcdef0123456789abcdef".into(),
        );
        let verdict = judge(|k| other_install.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::OtherHost);
        let mut other_user = dead.clone();
        other_user.insert(USER_LABEL.into(), "4242424".into());
        let verdict = judge(|k| other_user.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::OtherHost);
        let mut unlabelled = dead.clone();
        unlabelled.remove(INSTALL_LABEL);
        let verdict = judge(|k| unlabelled.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::OtherHost);
        // Another user's live process answers EPERM: alive.
        assert_eq!(process_state(1).await, ProcessState::Alive);

        // Another host: never judged dead from here.
        let mut other = dead.clone();
        other.insert(HOST_LABEL.into(), "some-other-host".into());
        let verdict = judge(|k| other.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::OtherHost);

        // Another owner: foreign.
        let mut foreign = dead.clone();
        foreign.insert(OWNER_LABEL.into(), "someone-else".into());
        let verdict = judge(|k| foreign.get(k).map(String::as_str)).await;
        assert_eq!(verdict, Verdict::Foreign);
        assert_eq!(get(&foreign, OWNER_LABEL).as_deref(), Some("someone-else"));
    }
}

#[cfg(test)]
mod machine_id_tests {
    use super::*;

    #[test]
    fn only_a_real_machine_id_is_used() {
        assert_eq!(
            usable_machine_id("0123456789abcdef0123456789ABCDEF\n").as_deref(),
            Some("0123456789abcdef0123456789abcdef")
        );
        assert_eq!(
            usable_machine_id("12345678-9ABC-DEF0-1234-56789ABCDEF0").as_deref(),
            Some("123456789abcdef0123456789abcdef0")
        );
        for bad in [
            "",
            "uninitialized\n",
            "0123",
            &"0".repeat(32),
            &"g".repeat(32),
        ] {
            assert_eq!(usable_machine_id(bad), None, "{bad:?}");
        }
        // Read once, the same each time, and never empty.
        assert!(!machine_id().is_empty());
        assert_eq!(machine_id(), machine_id());
    }
}
