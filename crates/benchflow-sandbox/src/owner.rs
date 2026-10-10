//! Who made a sandbox, so a sweep removes a crashed run's leftovers and
//! never a live run's sandboxes.
//!
//! Every container and Daytona sandbox BenchFlow creates carries three
//! labels:
//!
//! - [`OWNER_LABEL`](crate::OWNER_LABEL): the owner scope, `bench` unless
//!   `BENCHFLOW_OWNER` names another (teams sharing one Daytona key or one
//!   Docker daemon can keep their sweeps apart this way);
//! - [`HOST_LABEL`](crate::HOST_LABEL): this machine's host name;
//! - [`PID_LABEL`](crate::PID_LABEL): the id of the BenchFlow process.
//!
//! A sandbox is a leftover when its owner scope is ours, its host is this
//! machine, and no process with its pid is alive here. A pid that is alive
//! (even one reused by an unrelated process) keeps the sandbox: a sweep may
//! miss garbage, but it never deletes a live run's sandbox.

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

/// The labels a new sandbox carries: owner scope, host, and this process id.
pub fn labels() -> BTreeMap<String, String> {
    let mut labels = BTreeMap::new();
    labels.insert(OWNER_LABEL.to_string(), scope());
    labels.insert(HOST_LABEL.to_string(), hostname().to_string());
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

/// Whether process `pid` exists on this machine.
pub async fn process_state(pid: u32) -> ProcessState {
    if pid == std::process::id() {
        return ProcessState::Alive;
    }
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
    if label(HOST_LABEL) != Some(hostname()) {
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
