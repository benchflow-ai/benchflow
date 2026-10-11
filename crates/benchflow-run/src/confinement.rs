//! `bench doctor --confinement`: start one sandbox, apply the confinement a
//! trial applies, and report each setting by reading the sandbox's state
//! back, with the evidence.
//!
//! Nothing here attacks the sandbox: every check reads a file, a mode, or a
//! process's settings, as root or as the agent user.

use std::collections::BTreeMap;
use std::time::Duration;

use benchflow_sandbox::{
    Exec, Image, ImageSource, ImageSpec, NetworkPolicy, Provider, Sandbox, SandboxError,
    SandboxSpec,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::confine;
use crate::hardening::{self, Report};
use crate::outcome::DefenseState;

/// The image the check starts from unless told otherwise.
pub const DEFAULT_IMAGE: &str = "ubuntu:24.04";
/// The memory limit the check asks for, so there is one to read back.
pub const DEFAULT_MEMORY_MB: u64 = 1024;
const WORKDIR: &str = "/app";
const STEP_TIMEOUT: Duration = Duration::from_secs(180);

/// What to check with.
#[derive(Debug, Clone)]
pub struct Options {
    pub image: String,
    pub memory_mb: u64,
    pub runtime: Option<String>,
    pub pids_limit: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            image: DEFAULT_IMAGE.into(),
            memory_mb: DEFAULT_MEMORY_MB,
            runtime: None,
            pids_limit: 4096,
        }
    }
}

/// Whether a setting holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum State {
    /// Applied, and read back from the sandbox.
    Enforced,
    /// The backend cannot provide it.
    NotSupported,
    /// Not in place in this sandbox.
    NotSet,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Enforced => "enforced",
            State::NotSupported => "not supported",
            State::NotSet => "not set",
        }
    }
}

/// One setting, read back.
#[derive(Debug, Clone, Serialize)]
pub struct Setting {
    pub name: &'static str,
    pub state: State,
    /// One line: what holds.
    pub summary: String,
    /// What was read to say so.
    pub evidence: String,
}

/// The whole report.
#[derive(Debug, Clone, Serialize)]
pub struct Confinement {
    pub backend: String,
    pub image: String,
    pub sandbox: String,
    pub runtime: Option<String>,
    pub settings: Vec<Setting>,
}

/// Why the check could not run, and what to do.
#[derive(Debug, Clone)]
pub struct Failure {
    pub message: String,
    pub next: String,
}

impl Failure {
    fn sandbox(what: &str, e: SandboxError) -> Self {
        Self {
            message: format!("the confinement check could not {what}: {e}"),
            next: "run `bench doctor -s <backend>` to check the backend, then retry".into(),
        }
    }
}

/// Refuse a runtime the backend cannot start sandboxes under, before any
/// sandbox starts. `Ok` for no runtime, and for backends that take any.
pub async fn check_runtime(provider: &dyn Provider, runtime: Option<&str>) -> Result<(), Failure> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    let backend = provider.name();
    if backend == "daytona" {
        return Err(Failure {
            message: format!(
                "--runtime {runtime} asks for a container runtime, and Daytona chooses the runtime of its sandboxes itself"
            ),
            next: format!(
                "run with `-s docker` on a Docker host that has `{runtime}`, or drop --runtime"
            ),
        });
    }
    match provider.runtimes().await {
        Ok(Some(have)) if !have.iter().any(|h| h == runtime) => Err(Failure {
            message: format!(
                "the {backend} daemon has no `{runtime}` runtime (it has: {}), so no sandbox can start under it",
                have.join(", ")
            ),
            next: if runtime == "runsc" {
                "install gVisor on the daemon's host (https://gvisor.dev/docs/user_guide/install/), register it with `sudo runsc install` and `sudo systemctl reload docker`, and check `docker info --format '{{json .Runtimes}}'`; or drop --runtime".into()
            } else {
                format!(
                    "register `{runtime}` with the daemon (`runtimes` in /etc/docker/daemon.json), or drop --runtime"
                )
            },
        }),
        Ok(_) => Ok(()),
        Err(e) => Err(Failure {
            message: format!("could not list the {backend} daemon's runtimes: {e}"),
            next: format!("run `{backend} info` and fix what it reports"),
        }),
    }
}

fn pull_spec(reference: &str) -> ImageSpec {
    // The same digest a task with `[sandbox] image = "<reference>"` gets, so
    // the check reuses that image.
    let mut h = Sha256::new();
    h.update(b"benchflow-image-v1\0");
    h.update(b"pull\0");
    h.update(reference.as_bytes());
    ImageSpec {
        source: ImageSource::Pull {
            reference: reference.to_string(),
        },
        layers: Vec::new(),
        name: crate::images::sanitize_name(reference),
        digest: hex::encode(h.finalize()),
        build_timeout: Some(Duration::from_secs(1800)),
        platform: None,
    }
}

/// Start one sandbox on `provider`, confine it as a trial would, read each
/// setting back, and destroy the sandbox.
pub async fn check(provider: &dyn Provider, opts: &Options) -> Result<Confinement, Failure> {
    check_runtime(provider, opts.runtime.as_deref()).await?;
    let image = provider
        .prepare(&pull_spec(&opts.image), &|_line: &str| {})
        .await
        .map_err(|e| Failure::sandbox(&format!("prepare the image {}", opts.image), e))?;
    let sb = start(provider, &image, opts)
        .await
        .map_err(|e| Failure::sandbox("start a sandbox", e))?;
    let result = read_back(sb.as_ref(), provider.name(), opts).await;
    let _ = sb.destroy().await;
    let mut settings = result.map_err(|e| Failure::sandbox("read the sandbox's settings", e))?;
    if provider.capabilities().root_service {
        settings.push(Setting {
            name: "root_service",
            state: State::NotSupported,
            summary: "a service in the sandbox runs commands as root for any user who reaches it; this check installs no guard, so the agent user can reach it here (trials install an iptables guard and record root_service)".into(),
            evidence: format!("{}: its daemon answers on port 2280 inside the sandbox without credentials", provider.name()),
        });
    }
    Ok(Confinement {
        backend: provider.name().to_string(),
        image: opts.image.clone(),
        sandbox: sb.id().to_string(),
        runtime: opts.runtime.clone(),
        settings,
    })
}

async fn start(
    provider: &dyn Provider,
    image: &Image,
    opts: &Options,
) -> Result<Box<dyn Sandbox>, SandboxError> {
    let mut labels = BTreeMap::new();
    labels.insert("ai.benchflow.trial".into(), "doctor-confinement".into());
    provider
        .create(&SandboxSpec {
            image: image.clone(),
            name: "doctor-confinement".into(),
            cpus: Some(1.0),
            memory_mb: Some(opts.memory_mb),
            disk_mb: None,
            gpus: 0,
            pids_limit: Some(opts.pids_limit),
            network: NetworkPolicy::Open,
            env: BTreeMap::new(),
            workdir: Some(WORKDIR.into()),
            labels,
            runtime: opts.runtime.clone(),
        })
        .await
}

async fn root(sb: &dyn Sandbox, script: String) -> Result<String, SandboxError> {
    Ok(sb
        .exec(Exec::new(script).system().timeout(STEP_TIMEOUT))
        .await?
        .stdout)
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// A program, run as root, that reads the settings a trial leaves.
fn state_program(agent_pid: &str) -> String {
    let dirs: Vec<String> = hardening::SYSTEM_DIRS
        .iter()
        .map(|d| crate::shell_quote(d))
        .collect();
    format!(
        r#"n=0
for r in {dirs}; do
  [ -d "$r" ] && [ ! -L "$r" ] || continue
  c=$(find "$r" -xdev \( -path {workdir} -o -path /home \) -prune -o ! -type l \( -perm -020 -o -perm -002 \) -print 2>/dev/null | wc -l | tr -d ' ')
  n=$((n + c))
done
echo "BF writable $n"
for d in /usr /usr/bin /usr/local/bin /usr/lib /etc /opt; do [ -e "$d" ] && echo "BF mode $(stat -c '%A %U' "$d") $d"; done
if [ -e /etc/ld.so.preload ]; then echo "BF preload-file present $(stat -c '%A %U' /etc/ld.so.preload)"; else echo "BF preload-file absent"; fi
[ -f {state}/preload.sum ] && echo "BF preload-snapshot yes" || echo "BF preload-snapshot no"
p={pid}
echo "BF agent-oom $(cat /proc/$p/oom_score_adj 2>/dev/null || echo gone) $(stat -c %U /proc/$p 2>/dev/null || echo -)"
echo "BF root-oom $(cat /proc/self/oom_score_adj 2>/dev/null || echo unreadable)"
if [ -r /proc/sys/kernel/yama/ptrace_scope ]; then echo "BF ptrace $(cat /proc/sys/kernel/yama/ptrace_scope)"; else echo "BF ptrace none"; fi
echo "BF kernel $(cat /proc/version 2>/dev/null | cut -c1-80)"
echo "BF dmesg $(dmesg 2>/dev/null | head -n 2 | tr '\n' ' ' | cut -c1-120)"
{memory}"#,
        dirs = dirs.join(" "),
        workdir = crate::shell_quote(WORKDIR),
        state = crate::shell_quote(hardening::STATE_DIR),
        pid = crate::shell_quote(agent_pid),
        memory = hardening::memory_limit(),
    )
}

/// A program, run as the agent user, that tries to open the environment of
/// every root process (reading one byte, printing nothing of it).
const ENVIRON_PROGRAM: &str = r#"t=0; r=
for e in /proc/[0-9]*; do
  [ "$(stat -c %u "$e" 2>/dev/null)" = 0 ] || continue
  t=$((t + 1))
  head -c 1 "$e/environ" >/dev/null 2>&1 && r="$r ${e#/proc/}"
done
echo "BF environ $t$r""#;

async fn read_back(
    sb: &dyn Sandbox,
    backend: &str,
    opts: &Options,
) -> Result<Vec<Setting>, SandboxError> {
    let user = hardening::DEFAULT_USER;
    // Set up as a trial does: the agent user, PATH folders, system folders,
    // the ld.so.preload snapshot, the out-of-memory order, the memory limit.
    let out = root(sb, hardening::setup_user(user, WORKDIR)).await?;
    let home = Report::parse(&out)
        .first("user")
        .and_then(|l| l.split_whitespace().nth(2).map(str::to_string))
        .unwrap_or_else(|| format!("/home/{user}"));
    root(sb, hardening::lock_system_path(WORKDIR)).await?;
    let applied =
        confine::before_agent(sb, user, WORKDIR, &home, Some(opts.memory_mb), backend).await?;
    let applied: BTreeMap<String, crate::outcome::Defense> =
        applied.into_iter().map(|d| (d.name.clone(), d)).collect();

    // One agent process to read the settings of.
    let pid = confine::AgentSandbox::new(sb)
        .exec(
            Exec::new("nohup sleep 300 >/dev/null 2>&1 & echo $!")
                .user(user)
                .cwd(WORKDIR)
                .timeout(STEP_TIMEOUT),
        )
        .await?;
    let pid = first_line(&pid.stdout);
    let state = Report::parse(&root(sb, state_program(&pid)).await?);
    let environ = sb
        .exec(Exec::new(ENVIRON_PROGRAM).user(user).timeout(STEP_TIMEOUT))
        .await?;
    let environ = first_line(
        Report::parse(&environ.stdout)
            .first("environ")
            .unwrap_or(""),
    );

    // The verifier's environment, as a trial builds and checks it.
    let path = Report::parse(&root(sb, hardening::trusted_path(WORKDIR)).await?)
        .first("path")
        .map(str::to_string);
    let mut env = BTreeMap::new();
    if let Some(p) = &path {
        env.insert("PATH".to_string(), p.clone());
    }
    confine::add_safepath(&mut env, &BTreeMap::new(), benchflow_task::Format::TaskMd);
    let unset = hardening::unset_vars(&BTreeMap::new());
    let exec = Exec::new(String::new()).root().cwd(WORKDIR).envs(env);
    // A fresh sandbox no agent has used: the check may see the image's
    // variables itself.
    let venv = confine::check_verifier_env(sb, &exec, &unset, None, None).await?;

    let mut s = Vec::new();
    s.push(system_folders(&state, applied.get("locked_system_dirs")));
    s.push(match venv {
        Ok(d) => Setting {
            name: "verifier_env",
            state: State::Enforced,
            summary: "the verifier starts with no loader variables, PYTHONSAFEPATH=1, and a PATH of root-owned folders".into(),
            evidence: format!(
                "{}; PATH={}",
                d.note.unwrap_or_default(),
                path.unwrap_or_default()
            ),
        },
        Err(p) => Setting {
            name: "verifier_env",
            state: State::NotSet,
            summary: p.message.clone(),
            evidence: p.detail.unwrap_or_default().replace('\n', " "),
        },
    });
    s.push(preload(&state));
    s.push(oom(&state, &pid));
    s.push(memory(applied.get("memory_limit")));
    s.push(runtime(&state, backend, opts.runtime.as_deref()));
    s.push(ptrace(&state));
    s.push(root_environ(&environ));
    Ok(s)
}

fn system_folders(state: &Report, applied: Option<&crate::outcome::Defense>) -> Setting {
    let left: Option<u64> = state.first("writable").and_then(|n| n.parse().ok());
    let modes = state.all("mode").join(", ");
    let changed = applied.and_then(|d| d.note.clone()).unwrap_or_default();
    match left {
        Some(0) => Setting {
            name: "system_folders",
            state: State::Enforced,
            summary: format!(
                "nothing under {} is writable by group or others",
                hardening::SYSTEM_DIRS.join(" ")
            ),
            evidence: format!("{modes}; setup: {changed}"),
        },
        Some(n) => Setting {
            name: "system_folders",
            state: State::NotSet,
            summary: format!("{n} paths in system folders are writable by group or others"),
            evidence: format!("{modes}; setup: {changed}"),
        },
        None => Setting {
            name: "system_folders",
            state: State::NotSet,
            summary: "could not read the system folders' permissions".into(),
            evidence: changed,
        },
    }
}

fn preload(state: &Report) -> Setting {
    let file = state.first("preload-file").unwrap_or("unknown").to_string();
    let snap = state.first("preload-snapshot") == Some("yes");
    Setting {
        name: "ld_so_preload",
        state: if snap { State::Enforced } else { State::NotSet },
        summary: if snap {
            "snapshotted before the agent's turn; compared and restored before grading".into()
        } else {
            "no snapshot was taken, so a change during the agent's turn would go unseen".into()
        },
        evidence: format!("/etc/ld.so.preload: {file}"),
    }
}

fn oom(state: &Report, pid: &str) -> Setting {
    let agent = state.first("agent-oom").unwrap_or("").to_string();
    let mut parts = agent.split_whitespace();
    let value = parts.next().unwrap_or("");
    let owner = parts.next().unwrap_or("-");
    let root = state.first("root-oom").unwrap_or("").to_string();
    let evidence = format!(
        "/proc/{pid}/oom_score_adj = {value} (process owned by {owner}); a root command's = {root}"
    );
    match (value.parse::<i32>(), root.parse::<i32>()) {
        (Ok(a), Ok(r)) if a == hardening::AGENT_OOM_SCORE_ADJ && a > r => Setting {
            name: "oom_score_adj",
            state: State::Enforced,
            summary: "the kernel stops the agent's processes first when memory runs out".into(),
            evidence,
        },
        (Ok(_), Ok(_)) => Setting {
            name: "oom_score_adj",
            state: State::NotSet,
            summary: "agent processes do not rank ahead of root's for the out-of-memory killer"
                .into(),
            evidence,
        },
        _ => Setting {
            name: "oom_score_adj",
            state: State::NotSupported,
            summary: "the sandbox does not expose oom_score_adj".into(),
            evidence,
        },
    }
}

fn memory(applied: Option<&crate::outcome::Defense>) -> Setting {
    let note = applied.and_then(|d| d.note.clone()).unwrap_or_default();
    let state = match applied.map(|d| d.state) {
        Some(DefenseState::Enforced) => State::Enforced,
        Some(DefenseState::NotSupported) => State::NotSupported,
        _ => State::NotSet,
    };
    Setting {
        name: "memory_limit",
        state,
        summary: match state {
            State::Enforced => "the kernel caps the sandbox's memory".into(),
            State::NotSupported => "no memory limit is visible inside the sandbox".into(),
            State::NotSet => "the sandbox has no memory limit".into(),
        },
        evidence: note,
    }
}

fn runtime(state: &Report, backend: &str, requested: Option<&str>) -> Setting {
    let kernel = state.first("kernel").unwrap_or("").to_string();
    let dmesg = state.first("dmesg").unwrap_or("").trim().to_string();
    let gvisor =
        dmesg.contains("gVisor") || kernel.contains("4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016");
    let evidence = format!(
        "/proc/version: {kernel}{}",
        if dmesg.is_empty() {
            String::new()
        } else {
            format!("; dmesg: {dmesg}")
        }
    );
    match (requested, backend) {
        (Some(r), _) if r != "runsc" || gvisor => Setting {
            name: "runtime",
            state: State::Enforced,
            summary: if r == "runsc" {
                "gVisor (runsc): system calls go to gVisor's user-space kernel".into()
            } else {
                format!("{r}, as asked")
            },
            evidence,
        },
        (Some(_), _) => Setting {
            name: "runtime",
            state: State::NotSet,
            summary: "runsc was asked for, and the sandbox shows the host's kernel".into(),
            evidence,
        },
        (None, "daytona") => Setting {
            name: "runtime",
            state: State::NotSupported,
            summary: "Daytona chooses its sandboxes' runtime; BenchFlow cannot pick gVisor there"
                .into(),
            evidence,
        },
        (None, _) if gvisor => Setting {
            name: "runtime",
            state: State::Enforced,
            summary: "gVisor is the daemon's default runtime".into(),
            evidence,
        },
        (None, _) => Setting {
            name: "runtime",
            state: State::NotSet,
            summary: "the daemon's default runtime (runc), sharing the host's kernel; --runtime runsc adds gVisor where it is installed".into(),
            evidence,
        },
    }
}

fn ptrace(state: &Report) -> Setting {
    let v = state.first("ptrace").unwrap_or("none").to_string();
    match v.as_str() {
        "none" => Setting {
            name: "ptrace_scope",
            state: State::NotSupported,
            summary: "the kernel has no Yama module, so ptrace follows the plain user rules".into(),
            evidence: "/proc/sys/kernel/yama/ptrace_scope is missing".into(),
        },
        "0" => Setting {
            name: "ptrace_scope",
            state: State::NotSet,
            summary: "a process may trace any other process of its user (root's stay out of reach of the agent user)".into(),
            evidence: "kernel.yama.ptrace_scope = 0".into(),
        },
        other => Setting {
            name: "ptrace_scope",
            state: State::Enforced,
            summary: "a process may trace only its own descendants, or none".into(),
            evidence: format!("kernel.yama.ptrace_scope = {other}"),
        },
    }
}

fn root_environ(line: &str) -> Setting {
    let mut parts = line.split_whitespace();
    let total = parts.next().unwrap_or("?").to_string();
    let readable: Vec<&str> = parts.collect();
    if total == "?" || total.is_empty() {
        return Setting {
            name: "root_environ",
            state: State::NotSet,
            summary: "could not check root processes' environments".into(),
            evidence: line.to_string(),
        };
    }
    if readable.is_empty() {
        Setting {
            name: "root_environ",
            state: State::Enforced,
            summary: "the agent user cannot read any root process's environment".into(),
            evidence: format!(
                "opened /proc/<pid>/environ of {total} root processes as the agent user: all refused"
            ),
        }
    } else {
        Setting {
            name: "root_environ",
            state: State::NotSet,
            summary: format!(
                "the agent user can read the environment of {} root processes",
                readable.len()
            ),
            evidence: format!(
                "readable: /proc/{{{}}}/environ (of {total} root processes)",
                readable.join(",")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_state_program_is_valid_shell() {
        for prog in [state_program("123"), ENVIRON_PROGRAM.to_string()] {
            let out = std::process::Command::new("sh")
                .arg("-n")
                .arg("-c")
                .arg(&prog)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    #[test]
    fn settings_read_their_evidence() {
        let r = Report::parse(
            "BF writable 0\nBF mode drwxr-xr-x root /usr\nBF preload-file absent\nBF preload-snapshot yes\nBF agent-oom 1000 agent\nBF root-oom 0\nBF ptrace 1\nBF kernel Linux version 6.8.0\n",
        );
        assert_eq!(system_folders(&r, None).state, State::Enforced);
        assert_eq!(preload(&r).state, State::Enforced);
        assert_eq!(oom(&r, "42").state, State::Enforced);
        assert_eq!(ptrace(&r).state, State::Enforced);
        assert_eq!(runtime(&r, "docker", None).state, State::NotSet);
        assert_eq!(runtime(&r, "daytona", None).state, State::NotSupported);
        assert_eq!(runtime(&r, "docker", Some("runsc")).state, State::NotSet);
        assert_eq!(root_environ("12").state, State::Enforced);
        let leak = root_environ("12 1 7");
        assert_eq!(leak.state, State::NotSet);
        assert!(leak.evidence.contains("/proc/{1,7}/environ"));
        let r = Report::parse("BF agent-oom 0 agent\nBF root-oom 0\nBF ptrace 0\n");
        assert_eq!(oom(&r, "42").state, State::NotSet);
        assert_eq!(ptrace(&r).state, State::NotSet);
    }
}
