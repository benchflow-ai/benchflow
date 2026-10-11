//! Confinement steps a trial applies around the agent, each returning the
//! defense it records: system folders locked, `/etc/ld.so.preload` kept as
//! it was, the agent's processes first in line for the kernel's
//! out-of-memory killer, the memory limit, the runtime, and a verifier
//! environment free of loader variables.
//!
//! Each step runs one of [`crate::hardening`]'s programs and reads back what
//! it did, so the record says what held, not what was asked. `bench doctor
//! --confinement` runs the same steps.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use benchflow_sandbox::{Exec, ExecOutput, Process, Sandbox, SandboxError};

use benchflow_task::Format;

use crate::hardening::{self, Report};
use crate::outcome::{Defense, DefenseState, Owner, Phase, Problem};

const STEP_TIMEOUT: Duration = Duration::from_secs(180);

fn defense(name: &str, state: DefenseState, note: impl Into<String>) -> Defense {
    Defense {
        name: name.to_string(),
        state,
        note: Some(note.into()),
    }
}

async fn root(sb: &dyn Sandbox, script: String) -> Result<ExecOutput, SandboxError> {
    sb.exec(
        Exec::new(crate::hardening::with_plumbing_umask(&script))
            .system()
            .timeout(STEP_TIMEOUT),
    )
    .await
}

/// Remove group and other write permission from the system folders, leaving
/// the workdir and the agent's home alone. Records `locked_system_dirs`.
pub async fn lock_system_dirs(
    sb: &dyn Sandbox,
    user: &str,
    workdir: &str,
    home: &str,
) -> Result<Defense, SandboxError> {
    let out = root(sb, hardening::lock_system_dirs(workdir, home, user)).await?;
    Ok(judge_system_dirs(&out))
}

/// The `locked_system_dirs` defense from [`hardening::lock_system_dirs`]'s
/// report.
fn judge_system_dirs(out: &ExecOutput) -> Defense {
    let base = judge_system_write(out);
    let r = Report::parse(&out.stdout);
    let owned: usize = r
        .first("sysowned-count")
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    if owned == 0 {
        return base;
    }
    let named = r.all("sysowned");
    let more = owned.saturating_sub(named.len());
    let mut note = format!(
        "the agent user owns {owned} paths in system folders, which it can change whatever their permissions: {}{}",
        named.join(", "),
        if more > 0 {
            format!(", and {more} more")
        } else {
            String::new()
        }
    );
    if base.state == DefenseState::Failed
        && let Some(n) = &base.note
    {
        note = format!("{note}; {n}");
    }
    defense("locked_system_dirs", DefenseState::Failed, note)
}

fn judge_system_write(out: &ExecOutput) -> Defense {
    let r = Report::parse(&out.stdout);
    let changed: usize = r
        .first("syswrite-count")
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    let left: Option<usize> = r.first("syswrite-left").and_then(|n| n.parse().ok());
    let named: Vec<String> = r
        .all("syswrite")
        .iter()
        .map(|l| match l.split_once(' ') {
            Some((mode, path)) => format!("{path} ({mode})"),
            None => l.to_string(),
        })
        .collect();
    match left {
        None => defense(
            "locked_system_dirs",
            DefenseState::Failed,
            format!(
                "the lock did not finish: {}",
                out.tail(300).replace('\n', " ")
            ),
        ),
        Some(n) if n > 0 => defense(
            "locked_system_dirs",
            DefenseState::Failed,
            format!("{n} paths in system folders are still writable by group or others"),
        ),
        Some(_) if changed == 0 => defense(
            "locked_system_dirs",
            DefenseState::Enforced,
            "system folders were already writable by root alone",
        ),
        // As with `locked_system_path`, a lock the image made necessary is
        // reported as not having held by itself, so `bench test` shows it to
        // the task's author.
        Some(_) => {
            let more = changed.saturating_sub(named.len());
            defense(
                "locked_system_dirs",
                DefenseState::Failed,
                format!(
                    "the image lets group or others write {changed} paths in system folders; removed that permission before the agent ran: {}{}",
                    named.join(", "),
                    if more > 0 {
                        format!(", and {more} more")
                    } else {
                        String::new()
                    }
                ),
            )
        }
    }
}

/// Snapshot `/etc/ld.so.preload` before the agent starts. Returns what the
/// snapshot found (`absent`, or `present <lines>`), or `None` when it failed.
pub async fn snapshot_preload(sb: &dyn Sandbox) -> Result<Option<String>, SandboxError> {
    let out = root(sb, hardening::preload_snapshot()).await?;
    Ok(Report::parse(&out.stdout)
        .first("preload")
        .map(str::to_string))
}

/// Compare `/etc/ld.so.preload` with its snapshot and restore it. Records
/// `ld_so_preload`; a file that could not be put back refuses grading.
pub async fn check_preload(sb: &dyn Sandbox) -> Result<Result<Defense, Problem>, SandboxError> {
    let out = root(sb, hardening::preload_check()).await?;
    let state = Report::parse(&out.stdout)
        .first("preload")
        .unwrap_or("")
        .to_string();
    Ok(match state.as_str() {
        "absent" => Ok(defense(
            "ld_so_preload",
            DefenseState::Enforced,
            "absent before and after the agent's turn",
        )),
        "unchanged" => Ok(defense(
            "ld_so_preload",
            DefenseState::Enforced,
            "unchanged since before the agent's turn",
        )),
        "restored" => Ok(defense(
            "ld_so_preload",
            DefenseState::Enforced,
            "the agent's turn changed it; restored to its state from before the agent's turn",
        )),
        "no-baseline" => Ok(defense(
            "ld_so_preload",
            DefenseState::Failed,
            "no snapshot from before the agent's turn to compare with",
        )),
        "" => Err(crate::trial::unreported("the /etc/ld.so.preload check", &out)),
        _ => Err(Problem::new(
            "R309",
            Owner::Sandbox,
            Phase::Handoff,
            "/etc/ld.so.preload changed during the agent's turn and could not be restored, so the trial was not graded (every program the verifier starts would load what it names)",
            "retry; if it repeats for this task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
        )
        .detail(out.combined())),
    })
}

/// A sandbox that puts [`hardening::oom_prefix`] before every command, so
/// every process the agent starts is the first the kernel stops when memory
/// runs out. Hand it to whatever runs the agent.
pub struct AgentSandbox<'a> {
    inner: &'a dyn Sandbox,
}

impl<'a> AgentSandbox<'a> {
    pub fn new(inner: &'a dyn Sandbox) -> Self {
        Self { inner }
    }

    fn wrap(mut exec: Exec) -> Exec {
        exec.command = format!("{}{}", hardening::oom_prefix(), exec.command);
        exec
    }
}

#[async_trait]
impl Sandbox for AgentSandbox<'_> {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn backend(&self) -> &str {
        self.inner.backend()
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        self.inner.exec(Self::wrap(exec)).await
    }
    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        self.inner.spawn(Self::wrap(exec)).await
    }
    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError> {
        self.inner.upload(local, remote).await
    }
    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        mode: u32,
    ) -> Result<(), SandboxError> {
        self.inner.write_file(remote, contents, mode).await
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        self.inner.download(remote, local).await
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        self.inner.destroy().await
    }
    async fn presence(&self) -> Result<benchflow_sandbox::Presence, SandboxError> {
        self.inner.presence().await
    }
    async fn download_capped(
        &self,
        remote: &str,
        local: &Path,
        max_bytes: u64,
    ) -> Result<(), SandboxError> {
        self.inner.download_capped(remote, local, max_bytes).await
    }
}

/// Read back the `oom_score_adj` an agent command gets through
/// [`AgentSandbox`], and the one root commands (the runner's, the
/// verifier's) keep. Records `oom_order`.
pub async fn oom_order(sb: &dyn Sandbox, user: &str) -> Result<Defense, SandboxError> {
    let read = "cat /proc/self/oom_score_adj 2>/dev/null || echo unreadable";
    let agent = AgentSandbox::new(sb)
        .exec(Exec::new(read).user(user).timeout(STEP_TIMEOUT))
        .await?;
    let runner = root(sb, read.to_string()).await?;
    let agent = agent.stdout.trim().to_string();
    let runner = runner.stdout.trim().to_string();
    let want = hardening::AGENT_OOM_SCORE_ADJ.to_string();
    Ok(match (agent.parse::<i32>(), runner.parse::<i32>()) {
        (Ok(a), Ok(r)) if agent == want && a > r => defense(
            "oom_order",
            DefenseState::Enforced,
            format!(
                "agent processes start at oom_score_adj {a}, the runner and the verifier at {r}, so the kernel stops the agent's first"
            ),
        ),
        (Ok(a), Ok(r)) => defense(
            "oom_order",
            DefenseState::Failed,
            format!("agent processes got oom_score_adj {a}, not {want}; root commands have {r}"),
        ),
        _ => defense(
            "oom_order",
            DefenseState::NotSupported,
            format!("the sandbox does not expose oom_score_adj (agent: {agent}, root: {runner})"),
        ),
    })
}

/// The memory limit the kernel enforces in the sandbox, in bytes, and the
/// sandbox's total memory in bytes, as [`hardening::memory_limit`] reads
/// them.
pub fn parse_memory(output: &str) -> (Option<u64>, Option<u64>, String) {
    let r = Report::parse(output);
    let total = r
        .first("memtotal")
        .and_then(|kb| kb.trim().parse::<u64>().ok())
        .map(|kb| kb.saturating_mul(1024));
    let line = r.first("mem").unwrap_or("none").to_string();
    let mut parts = line.split_whitespace();
    let version = parts.next().unwrap_or("none");
    let limit = parts
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        // cgroup v1 shows "no limit" as a number near 2^63.
        .filter(|b| *b < 1 << 60);
    let source = match version {
        "v2" => "cgroup v2 memory.max",
        "v1" => "cgroup v1 memory.limit_in_bytes",
        _ => "no memory cgroup visible",
    };
    (limit, total, source.to_string())
}

fn mb(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

/// Read the sandbox's memory limit and compare it with what the task asked
/// for. Records `memory_limit`.
pub async fn memory_limit(
    sb: &dyn Sandbox,
    requested_mb: Option<u64>,
    backend: &str,
) -> Result<Defense, SandboxError> {
    let out = root(sb, hardening::memory_limit()).await?;
    let (limit, total, source) = parse_memory(&out.stdout);
    Ok(memory_defense(limit, total, &source, requested_mb, backend))
}

/// The `memory_limit` defense for what the sandbox shows (`limit` and
/// `total` in bytes) and what the task asked for. A limit above the one
/// asked for counts only as the backend's rounding up to whole GiB; more
/// than that, and the cap is not the task's, so it is recorded as failed.
fn memory_defense(
    limit: Option<u64>,
    total: Option<u64>,
    source: &str,
    requested_mb: Option<u64>,
    backend: &str,
) -> Defense {
    let total_note = total
        .map(|t| format!("; the sandbox sees {} MB in total", mb(t)))
        .unwrap_or_default();
    match (limit, requested_mb) {
        (Some(l), Some(req)) if mb(l) > req.div_ceil(1024).saturating_mul(1024) => defense(
            "memory_limit",
            DefenseState::Failed,
            format!(
                "{} MB ({source}), above the {req} MB the task asked for",
                mb(l)
            ),
        ),
        (Some(l), Some(req)) => defense(
            "memory_limit",
            DefenseState::Enforced,
            if mb(l) == req {
                format!("{req} MB ({source})")
            } else {
                format!(
                    "{} MB ({source}); the task asked for {req} MB and the {backend} backend rounded it",
                    mb(l)
                )
            },
        ),
        (Some(l), None) => defense(
            "memory_limit",
            DefenseState::Enforced,
            format!(
                "{} MB ({source}), the {backend} backend's own; the task sets none",
                mb(l)
            ),
        ),
        (None, Some(req)) => defense(
            "memory_limit",
            DefenseState::NotSupported,
            format!(
                "the task asks for {req} MB, and the sandbox shows no memory limit ({source}){total_note}; the {backend} backend may cap the machine itself, which the sandbox cannot confirm"
            ),
        ),
        (None, None) => defense(
            "memory_limit",
            DefenseState::Off,
            match total {
                Some(t) => format!(
                    "the task sets no memory limit, so the sandbox may use all {} MB its host has",
                    mb(t)
                ),
                None => {
                    "the task sets no memory limit, so the sandbox may use all its host has".into()
                }
            },
        ),
    }
}

/// The runtime a sandbox was asked to start under, as a defense. `None`
/// when no runtime was asked for: the backend's default is not recorded.
pub fn runtime(requested: Option<&str>) -> Option<Defense> {
    let r = requested?;
    let what = if r == "runsc" {
        "runsc (gVisor): the sandbox's system calls go to gVisor's user-space kernel, not the host's".to_string()
    } else {
        format!("{r}, as asked with --runtime")
    };
    Some(defense("runtime", DefenseState::Enforced, what))
}

/// The confinement a trial applies after the agent user exists and before
/// the agent starts: system folders locked (for a non-root agent),
/// `/etc/ld.so.preload` snapshotted, the out-of-memory order checked, and
/// the memory limit read. Returns the defenses to record.
pub async fn before_agent(
    sb: &dyn Sandbox,
    user: &str,
    workdir: &str,
    home: &str,
    requested_memory_mb: Option<u64>,
    backend: &str,
) -> Result<Vec<Defense>, SandboxError> {
    let mut out = Vec::new();
    if user != "root" && user != "0" {
        out.push(lock_system_dirs(sb, user, workdir, home).await?);
    } else {
        out.push(defense(
            "locked_system_dirs",
            DefenseState::Off,
            "the task runs its agent as root, which can write system folders whatever their permissions",
        ));
    }
    if snapshot_preload(sb).await?.is_none() {
        out.push(defense(
            "ld_so_preload",
            DefenseState::Failed,
            "could not snapshot /etc/ld.so.preload before the agent's turn",
        ));
    }
    out.push(oom_order(sb, user).await?);
    out.push(memory_limit(sb, requested_memory_mb, backend).await?);
    Ok(out)
}

/// Find credential files in the sandbox's filesystem (run before the agent
/// starts), and whether `user` can read each.
pub async fn scan_image(
    sb: &dyn Sandbox,
    user: &str,
) -> Result<Vec<crate::credentials::Finding>, SandboxError> {
    let out = root(sb, crate::credentials::scan_program()).await?;
    let mut found = crate::credentials::parse_scan(&out.stdout);
    if found.is_empty() {
        return Ok(found);
    }
    let paths: Vec<String> = found.iter().map(|f| f.path.clone()).collect();
    let readable = sb
        .exec(
            Exec::new(crate::credentials::readable_program(&paths))
                .user(user)
                .timeout(STEP_TIMEOUT),
        )
        .await?;
    let readable = Report::parse(&readable.stdout);
    let readable = readable.all("readable");
    for f in &mut found {
        f.agent_can_read = Some(readable.contains(&f.path.as_str()));
    }
    Ok(found)
}

/// What the verifier's environment check found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifierEnv {
    /// Loader variables the sandbox's own environment held, unset for the
    /// verifier.
    pub dropped: Vec<String>,
    /// Loader variables still set after the unset (must be empty).
    pub still_set: Vec<String>,
    pub safepath: String,
    /// `PATH` folders that are relative or writable by someone other than
    /// root, with why.
    pub bad_path: Vec<String>,
    pub path: String,
    /// The check ran to its end.
    pub done: bool,
}

impl VerifierEnv {
    pub fn parse(output: &str) -> Self {
        let r = Report::parse(output);
        let own = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
        Self {
            dropped: own(r.all("venv-dropped")),
            still_set: own(r.all("venv-set")),
            safepath: r.first("venv-safepath").unwrap_or("").to_string(),
            bad_path: own(r.all("venv-path-bad")),
            path: r.first("venv-path").unwrap_or("").to_string(),
            done: r.first("venv-done").is_some(),
        }
    }
}

/// The verifier's command: [`hardening::verifier_env_prefix`], then `exec`
/// the entry.
pub fn verifier_command(vars: &[&str], entry: &str) -> String {
    format!(
        "{}exec {}",
        hardening::verifier_env_prefix(vars),
        crate::shell_quote(entry)
    )
}

/// Check the environment `exec_like` (the verifier's user, folder, and
/// variables) gives the verifier, under the unset of `vars`. Returns the
/// `clean_verifier_env` defense, or the problem that refuses grading.
///
/// `safepath_off` says why `PYTHONSAFEPATH` may be off ([`safepath_off`]);
/// `None` when it must be 1.
///
/// `image_loader` names the loader variables the image's environment set,
/// read before the agent ran. When given, the check (a root shell, as a
/// rule, after the agent's turn) starts without `vars` (`Exec::unset`), so
/// it neither loads nor reads what they name, and reports those as dropped
/// instead of seeing them itself.
pub async fn check_verifier_env(
    sb: &dyn Sandbox,
    exec_like: &Exec,
    vars: &[&str],
    safepath_off: Option<&str>,
    image_loader: Option<&[String]>,
) -> Result<Result<Defense, Problem>, SandboxError> {
    let mut exec = exec_like.clone();
    exec.command = hardening::verifier_env_check(vars);
    exec.timeout = Some(STEP_TIMEOUT);
    if image_loader.is_some() {
        exec = exec.unset(vars.iter().copied());
    }
    let out = sb.exec(exec).await?;
    let mut env = VerifierEnv::parse(&out.stdout);
    for name in image_loader.unwrap_or_default() {
        if vars.contains(&name.as_str()) && !env.dropped.contains(name) {
            env.dropped.push(name.clone());
        }
    }
    Ok(judge_verifier_env(&env, safepath_off, &out))
}

fn judge_verifier_env(
    env: &VerifierEnv,
    safepath_off: Option<&str>,
    out: &ExecOutput,
) -> Result<Defense, Problem> {
    let refuse = |owner: Owner, why: String, next: &str| {
        Err(Problem::new(
            "R310",
            owner,
            Phase::Verify,
            format!("the verifier's environment could not be made clean, so the trial was not graded: {why}"),
            next,
        )
        .detail(out.combined()))
    };
    if !env.done {
        let mut p = crate::trial::unreported("the check of the verifier's environment", out);
        p.phase = Phase::Verify;
        return Err(p);
    }
    if !env.still_set.is_empty() {
        return refuse(
            Owner::Task,
            format!(
                "{} stayed set after unset (the image's shell may mark them read-only)",
                env.still_set.join(", ")
            ),
            "do not mark loader variables read-only in the image's shell start-up files",
        );
    }
    if !env.bad_path.is_empty() {
        return refuse(
            Owner::Task,
            format!(
                "PATH holds folders someone other than root could write: {}",
                env.bad_path.join(", ")
            ),
            "make every folder on the image's PATH owned by root and not writable by group or others (RUN chmod go-w <folder> in the Dockerfile), or set [verifier] env PATH to root-owned folders",
        );
    }
    let safepath = if env.safepath == "1" {
        "PYTHONSAFEPATH=1".to_string()
    } else if let Some(why) = safepath_off {
        format!("PYTHONSAFEPATH off: {why}")
    } else {
        return refuse(
            Owner::Benchflow,
            format!("PYTHONSAFEPATH is `{}`, not 1", env.safepath),
            "report this with the trial folder",
        );
    };
    let dropped = if env.dropped.is_empty() {
        String::new()
    } else {
        format!(
            "; unset from the sandbox's own environment: {}",
            env.dropped.join(", ")
        )
    };
    Ok(defense(
        "clean_verifier_env",
        if safepath_off.is_some() && env.safepath != "1" {
            DefenseState::Off
        } else {
            DefenseState::Enforced
        },
        format!("no loader variables, {safepath}, PATH of root-owned folders only{dropped}"),
    ))
}

/// Add `PYTHONSAFEPATH=1` to the verifier's variables, unless the task's
/// `[verifier] env` sets it (an empty value turns it off) or its format
/// leaves it off (see [`safepath_off`]): Python then never puts the
/// verifier's folder, or the workdir it starts in, ahead of the standard
/// library, so a planted `pytest.py` or `json.py` there is not imported.
pub fn add_safepath(
    env: &mut BTreeMap<String, String>,
    task_env: &BTreeMap<String, String>,
    format: Format,
) {
    match task_env.get("PYTHONSAFEPATH") {
        Some(v) => {
            env.insert("PYTHONSAFEPATH".into(), v.clone());
        }
        None if format == Format::TaskMd => {
            env.insert("PYTHONSAFEPATH".into(), "1".into());
        }
        None => {}
    }
}

/// Why `PYTHONSAFEPATH` may be off for a task's verifier, or `None` when it
/// must be 1. Tasks in Harbor's split layout and 0.8's task.md run their
/// verifiers as Harbor and 0.8 do, which leave it to the image: such
/// verifiers may run a script that imports a module beside it, which the
/// flag breaks (SkillsBench's adaptive-cruise-control, Terminal-Bench 2's
/// cancel-async-tasks). A task turns it on with `PYTHONSAFEPATH = "1"` in
/// `[verifier] env`.
pub fn safepath_off(format: Format, task_env: &BTreeMap<String, String>) -> Option<&'static str> {
    match task_env.get("PYTHONSAFEPATH") {
        Some(v) if v == "1" => None,
        Some(_) => Some("turned off by the task's [verifier] env"),
        None if format == Format::TaskMd => None,
        None => Some("left to the image, as 0.8 runs this task format"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Files the agent user owns in system folders are not made safe by
    /// taking group and other write permission away, so the lock is not
    /// recorded as holding.
    #[test]
    fn system_folders_the_agent_owns_fail_the_lock() {
        let out = |stdout: &str| ExecOutput {
            stdout: stdout.to_string(),
            ..ExecOutput::default()
        };
        let d = judge_system_dirs(&out("BF syswrite-count 0\nBF syswrite-left 0\n"));
        assert_eq!(d.state, DefenseState::Enforced);
        let d = judge_system_dirs(&out(
            "BF syswrite-count 0\nBF syswrite-left 0\nBF sysowned /opt/venv\nBF sysowned-count 3\n",
        ));
        assert_eq!(d.state, DefenseState::Failed);
        let note = d.note.unwrap();
        assert!(
            note.contains("owns 3 paths") && note.contains("/opt/venv, and 2 more"),
            "{note}"
        );
        let d = judge_system_dirs(&out(
            "BF syswrite-count 0\nBF syswrite-left 0\nBF sysowned-count 0\n",
        ));
        assert_eq!(d.state, DefenseState::Enforced);
    }

    #[test]
    fn memory_reads_cgroup_v2_v1_and_none() {
        let (l, t, s) = parse_memory("BF mem v2 1073741824 1073741824\nBF memtotal 2048000\n");
        assert_eq!(l, Some(1 << 30));
        assert_eq!(t, Some(2048000 * 1024));
        assert!(s.contains("v2"));
        let (l, _, _) = parse_memory("BF mem v2 max max\n");
        assert_eq!(l, None);
        let (l, _, _) = parse_memory("BF mem v1 9223372036854771712 -\n");
        assert_eq!(l, None, "v1's no-limit number is not a limit");
        let (l, t, s) = parse_memory("BF mem none\n");
        assert_eq!((l, t), (None, None));
        assert!(s.contains("no memory cgroup"));
    }

    #[test]
    fn huge_memory_numbers_saturate() {
        // The sandbox's own report and the task's size: both outside input.
        let (_, t, _) = parse_memory("BF mem none\nBF memtotal 18446744073709551615\n");
        assert_eq!(t, Some(u64::MAX));
        let d = memory_defense(Some(1 << 30), None, "v2", Some(u64::MAX), "fake");
        assert_eq!(d.state, DefenseState::Enforced);
    }

    #[test]
    fn a_memory_limit_above_the_one_asked_for_is_not_enforced() {
        let gib = 1024 * 1024 * 1024;
        let d = memory_defense(Some(gib), None, "v2", Some(1024), "fake");
        assert_eq!(d.state, DefenseState::Enforced);
        // Rounded up to whole GiB: still the task's cap.
        let d = memory_defense(Some(2 * gib), None, "v2", Some(1500), "fake");
        assert_eq!(d.state, DefenseState::Enforced);
        let d = memory_defense(Some(8 * gib), None, "v2", Some(1024), "fake");
        assert_eq!(d.state, DefenseState::Failed);
        assert!(d.note.unwrap().contains("above the 1024 MB"));
    }

    fn out() -> ExecOutput {
        ExecOutput::default()
    }

    #[test]
    fn a_clean_environment_passes_and_says_what_was_dropped() {
        let env = VerifierEnv::parse(
            "BF venv-dropped LD_LIBRARY_PATH\nBF venv-safepath 1\nBF venv-path /usr/bin:/bin\nBF venv-done\n",
        );
        let d = judge_verifier_env(&env, None, &out()).unwrap();
        assert_eq!(d.state, DefenseState::Enforced);
        assert!(d.note.unwrap().contains("LD_LIBRARY_PATH"));
    }

    #[test]
    fn an_unclean_environment_refuses_grading_with_an_owner() {
        let p = judge_verifier_env(&VerifierEnv::parse("BF venv-safepath 1\n"), None, &out())
            .unwrap_err();
        assert_eq!(
            p.code, "R307",
            "a check that does not finish did not report back"
        );
        let cases = [
            (
                "BF venv-set LD_PRELOAD\nBF venv-safepath 1\nBF venv-done\n",
                Owner::Task,
            ),
            (
                "BF venv-safepath 1\nBF venv-path-bad /usr/local/bin drwxrwxrwx\nBF venv-done\n",
                Owner::Task,
            ),
            ("BF venv-safepath \nBF venv-done\n", Owner::Benchflow),
        ];
        for (text, owner) in cases {
            let p = judge_verifier_env(&VerifierEnv::parse(text), None, &out()).unwrap_err();
            assert_eq!(p.code, "R310");
            assert_eq!(p.owner, owner, "{text}");
        }
    }

    #[test]
    fn a_task_may_turn_safepath_off() {
        let mut env = BTreeMap::new();
        let mut task = BTreeMap::new();
        add_safepath(&mut env, &task, Format::TaskMd);
        assert_eq!(env["PYTHONSAFEPATH"], "1");
        task.insert("PYTHONSAFEPATH".to_string(), String::new());
        add_safepath(&mut env, &task, Format::TaskMd);
        assert_eq!(env["PYTHONSAFEPATH"], "");
        let why = safepath_off(Format::TaskMd, &task);
        assert_eq!(why, Some("turned off by the task's [verifier] env"));
        let v = VerifierEnv::parse("BF venv-safepath \nBF venv-done\n");
        let d = judge_verifier_env(&v, why, &out()).unwrap();
        assert_eq!(d.state, DefenseState::Off);
    }

    /// Harbor's and 0.8's formats run verifiers as Harbor and 0.8 do: a
    /// script that imports a module beside it must keep working (T4.7).
    #[test]
    fn imported_formats_leave_safepath_to_the_image() {
        let none = BTreeMap::new();
        for format in [Format::Split, Format::TaskMdV06] {
            let mut env = BTreeMap::new();
            add_safepath(&mut env, &none, format);
            assert!(!env.contains_key("PYTHONSAFEPATH"), "{format:?}");
            let why = safepath_off(format, &none);
            assert!(why.is_some_and(|w| w.contains("0.8")), "{why:?}");
            let v = VerifierEnv::parse("BF venv-safepath \nBF venv-done\n");
            assert_eq!(
                judge_verifier_env(&v, why, &out()).unwrap().state,
                DefenseState::Off
            );
        }
        // Native task.md keeps it, and any format can ask for it.
        assert_eq!(safepath_off(Format::TaskMd, &none), None);
        let mut on = BTreeMap::new();
        on.insert("PYTHONSAFEPATH".to_string(), "1".to_string());
        let mut env = BTreeMap::new();
        add_safepath(&mut env, &on, Format::Split);
        assert_eq!(env["PYTHONSAFEPATH"], "1");
        assert_eq!(safepath_off(Format::Split, &on), None);
    }

    #[test]
    fn the_verifier_command_unsets_then_execs() {
        let c = verifier_command(&["LD_PRELOAD", "BASH_ENV"], "/verifier/test.sh");
        assert_eq!(c, "unset LD_PRELOAD BASH_ENV\nexec /verifier/test.sh");
        assert_eq!(verifier_command(&[], "/v/t.sh"), "exec /v/t.sh");
    }

    #[test]
    fn the_oom_prefix_leads_every_agent_command() {
        let e = AgentSandbox::wrap(Exec::new("echo hi"));
        assert!(
            e.command
                .starts_with("{ echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null\n")
        );
        assert!(e.command.ends_with("echo hi"));
    }
}
