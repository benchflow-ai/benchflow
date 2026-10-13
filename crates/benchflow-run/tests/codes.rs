//! Every run failure code `bench explain` lists, triggered through the real
//! engine: a scripted in-memory sandbox stands in for what a sandbox does
//! (an image that fails to build, a container that never starts, a verifier
//! that writes no reward), and a scripted ACP agent, driven by the real ACP
//! client, for what agents do (exit, go silent, crash, answer with an
//! error). Each case checks the code and whose problem it is.
//!
//! `tests/codes/run.txt` keeps what a user reads for each case: the code,
//! whose problem it is, the message and the next step, as `bench run`'s
//! report shows them. A change in wording shows up in review; regenerate it
//! with `UPDATE_GOLDEN=1 cargo test -p benchflow-run --test codes`. The test
//! also fails when `bench explain` lists a code no case here triggers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::ManifestSource;
use benchflow_run::acp_agent::AcpAgent;
use benchflow_run::agent::{AgentContext, AgentOutput, Attempt, ModelAgent};
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{EXPLANATIONS, Owner, Problem};
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::services::{ServiceHost, ServiceSpec};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, ProcessHandle, Provider,
    ProviderStatus, Sandbox, SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::*;
use benchflow_task::values::Network;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// What the scripted sandbox does wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Fault {
    #[default]
    None,
    /// The task's Dockerfile fails to build.
    TaskImage,
    /// The backend fails to prepare the image for a reason outside the task.
    ImageBackend,
    /// The image builds, but not with the agent's install layers on top.
    AgentLayers,
    /// The container fails to start.
    Start,
    /// The backend cannot give the task what it asks for.
    NoGpus,
    /// The image has no useradd or adduser.
    Useradd,
    /// The readiness check never passes.
    NotReady,
    /// The verifier's files cannot be read on this machine.
    LocalRead,
    /// The connection to the sandbox drops before grading.
    Transport,
    /// Agent processes survive the kill before grading.
    Survivors,
    /// The sandbox is deleted while the kill before grading runs: the
    /// program ends without its report.
    KillGone,
    /// A program on the image's PATH answers in place of the shell.
    Shadowed,
    /// As `Shadowed`, and the planted shell answers any later command with
    /// the container CLI's "No such container", though the sandbox runs.
    ShadowedLies,
    /// The agent's turn changed /etc/ld.so.preload, and it cannot be put back.
    PreloadStuck,
    /// The agent took ownership of BenchFlow's state folder before grading.
    StateTampered,
    /// The image's PATH holds a folder others can write.
    UncleanVerifierEnv,
    /// The workdir cannot be given to root before grading.
    OwnFails,
    /// A hook before grading exits with an error.
    HookFails,
    /// A service beside the sandbox exits as it starts.
    ServiceFails,
    /// The image has no iptables for a blocklist's firewall.
    NoIptables,
    /// The firewall self-test reaches a blocked address.
    Leaky,
    /// The backend loses the sandbox during the firewall self-test (its
    /// commands still come back, as Daytona's did for a deleted sandbox).
    LeakyGone,
    /// The verifier writes no reward.
    NoReward,
    /// The verifier writes a reward that is not a number.
    BadReward,
    /// The verifier runs past its timeout.
    VerifierTimeout,
    /// The verifier exits with an error and writes no reward.
    VerifierFails,
    /// The connection to the sandbox drops while it sets up.
    SetupGone,
    /// The sandbox hangs while it sets up (for a Ctrl-C there).
    SetupHangs,
    /// The connection to the sandbox drops while the verifier runs.
    VerifierGone,
    /// The sandbox is killed under the verifier (exit 137) and is gone.
    VerifierKilled,
    /// The verifier runs, but its folder cannot be fetched.
    DownloadFails,
    /// The verifier removes its own folder (`rm -rf /logs/verifier`).
    VerifierFolderGone,
    /// The verifier's folder comes out of the sandbox but cannot be
    /// unpacked on this machine.
    LocalUnpack,
    /// The copy of the verifier's folder breaks off after `reward.txt`
    /// (the connection drops).
    DownloadBreaksOff,
    /// This machine's disk fills while the verifier's folder is copied in.
    HostDiskFull,
    /// The first copy of the verifier's folder breaks off; the second
    /// works.
    DownloadBreaksOnce,
    /// The verifier's folder cannot be copied (tar refuses a file in it),
    /// though the sandbox runs.
    DownloadRefused,
    /// The verifier's folder is over the copy's cap.
    DownloadOverCap,
    /// The connection to the running sandbox breaks off during both copies.
    DownloadChannel,
    /// The verifier's code stops the sandbox (`kill -TERM 1`): the backend
    /// still has it, stopped, and nothing more can be run or copied.
    VerifierStopsSandbox,
    /// The verifier's code stops the sandbox, and the backend cannot say
    /// what stopped it (Podman; Daytona's `error`).
    VerifierHaltsSandbox,
    /// The verifier runs past its timeout, and its folder cannot be copied.
    TimeoutDownloadFails,
    /// The backend does not answer when the sandbox is removed.
    DestroyFails,
    /// The verifier hangs (for a Ctrl-C while grading).
    VerifierHangs,
    /// The verifier fails on a full disk and writes no reward.
    DiskFull,
    /// The out-of-memory killer ends the agent mid-prompt.
    AgentOom,
    /// The sandbox cannot start the agent's process: it is gone.
    AgentSpawnGone,
    /// The sandbox goes away while the agent works (the agent's process
    /// ends with it).
    AgentGone,
    /// The sandbox stops while the agent works (the backend still has it,
    /// stopped), and the agent's process ends with it.
    AgentStopped,
    /// The sandbox halts while the agent works (the backend cannot say
    /// what stopped it), and the agent's process ends with it.
    AgentHalted,
    /// The task sets a disk limit the backend does not enforce, and the
    /// sandbox's writable layer passes it while the agent works: the
    /// disk limit's watch stops the sandbox.
    DiskOver,
    /// As `DiskOver`, but the layer held most of the limit before the
    /// agent's turn began (an earlier verifier, BenchFlow's own setup).
    DiskOverBefore,
    /// The daemon caps the disk; the agent's turn fills the cap, and the
    /// verifier then fails on the full disk.
    DiskFilled,
    /// As `DiskOver`, but after the first measure the daemon cannot measure
    /// the layer, not even with the sandbox frozen, while it still runs.
    DiskUnmeasurable,
    /// The daemon caps the disk's bytes; the agent's turn floods it with
    /// empty files (inodes the cap does not hold).
    CapInodes,
    /// The verifier fails on a full disk the agent did not fill (no cap:
    /// the host's disk).
    DaemonDiskFull,
    /// The kill before grading cannot start: the sandbox is at its own
    /// process limit.
    ForkBomb,
    /// The kill before grading cannot start, and the sandbox is below its
    /// own process limit (the host ran out).
    HostPids,
    /// The verifier's output says "No space left on device" (the agent's
    /// program printed it), but the disk has room.
    FakeFull,
    /// Under a cap the daemon enforces, the daemon's own disk fills from
    /// outside; the verifier fails on it, with the sandbox's quota unspent.
    CapHostFull,
    /// The host's out-of-memory killer ends the agent mid-prompt; the
    /// sandbox's own limit was never hit.
    HostOom,
    /// Review F2-C, F-2: the sandbox's own limit ran out once, and the
    /// out-of-memory killer ended two processes (the host's killed one).
    MixedOom,
    /// Review F2-C, F-1 (I1): under a cap, the agent's background writer
    /// fills the cap after its attempt returned; the sandbox's `df` shows
    /// it full only once the agent user's processes were killed. The
    /// verifier then fails on the full disk.
    CapFilledAfter,
    /// Review F2-C, F-1 (I2): on a watched layer, the agent's writer takes
    /// the layer past the limit after the attempt returned, before the kill
    /// at the end of its turn.
    LateWriter,
    /// I8: the layer passes the limit while the verifier runs (outside the
    /// agent's windows).
    VerifyWriter,
    /// Review F2-C, F-3 (I9): under a cap, a sparse file whose size is far
    /// past the cap; the cap is not full.
    CapSparse,
    /// Review F2-C, F-4 (I6): the verifier fails on a disk whose blocks are
    /// free and whose inodes are all used.
    InodesFull,
    /// Review F2-C, F-6 (I5): the verifier writes a reward of 0 and prints
    /// "No space left on device"; the sandbox's `df` shows the disk full,
    /// the probe of the daemon's disk does not.
    ZeroOnFullDf,
    /// As `ZeroOnFullDf`, but neither shows it full.
    ZeroWithRoom,
    /// As `ZeroOnFullDf`, but the probe of the daemon's disk shows it full,
    /// and the sandbox's `df` does not.
    ZeroOnFullHost,
    /// As `ZeroOnFullDf`, but the reward is 1.
    PassOnFullDf,
    /// Review F2-C (I10): under a cap, a root agent's `df` says the disk has
    /// room, the daemon measures the layer at the cap, and the verifier
    /// fails on the full disk.
    CapFakedRoom,
    /// Review F2-C-r4 H1 / F2-D-r4 H1 (blocks): under a cap, a neighbour
    /// fills the daemon's disk during the turn; the sandbox's `df` follows
    /// the host's free space (600 MB free at the open, none at the close),
    /// and the probe of the daemon's disk shows it full. The verifier fails
    /// on the full disk.
    CapHostFillBlocks,
    /// Review F2-C-r5 H1: as `CapHostFillBlocks`, but the daemon's disk has
    /// room again when grading fails (the probe shows room).
    CapHostFillRecovered,
    /// The same with the inodes: the capped sandbox's inode counts are the
    /// host's (3,000 free at the open, 1,000 at the close, the agent's
    /// failing commands and neighbours making the files), blocks with room.
    CapHostInodes,
    /// Review F2-C-r4 M1: under a cap, a rootless agent's sparse 20 GB file
    /// makes the layer's apparent size pass the cap; its `df` and the
    /// daemon's disk have room; the verifier prints the disk words and
    /// writes 0.
    CapSparseZero,
    /// Review F2-integrate-A A-1: the daemon cannot measure the layer from
    /// its first read, so the agent never runs (R110 before the turn).
    DiskUnmeasurableFromStart,
    /// Review F2-A-r4 M4: a rootless agent at its own process limit (the
    /// kill at the end of its turn cannot start) also left a writer that
    /// passes the watched limit after the window closed.
    ForkBombWriter,
    /// As `CapFakedRoom`, but the daemon cannot measure the layer, and its
    /// own disk has room: the quota may be full, so not "not full".
    CapUnknownLayer,
    /// Review F2-C, F-5 (I11): under a cap, the verifier writes no reward,
    /// with no word of a full disk.
    CapNoWords,
    /// The oracle's solve.sh (a scripted attempt, whose failure is the
    /// sandbox's) stops its sandbox from inside.
    OracleStops,
    /// The oracle's sandbox halts for a reason the backend cannot name.
    OracleHalts,
    /// The verifier's channel fails (a toolbox 502) while the backend's
    /// own API is down too: an outage the control plane confirms.
    Outage,
}

/// What the scripted ACP agent does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Act {
    #[default]
    Solve,
    /// Exits before answering `initialize`.
    ExitAtStart,
    /// Never answers the prompt.
    Silent,
    /// Exits in the middle of the prompt.
    Crash,
    /// Answers the prompt with a JSON-RPC error.
    Error(&'static str),
    /// The same, after writing a line to its log (stderr).
    ErrorLogged(&'static str, &'static str),
    /// Answers the prompt with a JSON-RPC error whose `data.details` is
    /// this JSON (the ACP TypeScript SDK's place for the thrown error).
    ErrorDetails(&'static str, &'static str),
    /// Answers the prompt with codex-acp's usage-limit error.
    Spent,
    /// Ends its turn with this as its only message, and no tool call.
    Reply(&'static str),
    /// The same, from a model that was reached: the turn reports output
    /// tokens.
    Answer(&'static str),
    /// Refuses `session/new` until signed in with the vendor's account.
    NeedsSignIn,
}

#[derive(Default)]
struct World {
    solved: bool,
    /// The sandbox exists but was stopped from inside.
    stopped: bool,
    halted: bool,
    api_down: bool,
    /// Copies of the verifier's folder so far.
    downloads: usize,
    /// The sandbox is gone: every later command fails.
    gone: bool,
    /// The sandbox was killed from outside (the disk limit's watch).
    killed: bool,
    /// Reads of the sandbox's memory counters so far.
    memory_reads: u32,
    /// The agent's process was started.
    spawned: bool,
    /// Kills of the agent user's processes so far.
    user_kills: u32,
    /// The verifier began.
    verifying: bool,
    /// Probes of the daemon's disk, and measures of the layer, after the
    /// agent's turn.
    host_probes: u32,
    /// What happened, in order: `kill`, `df`, `verifier`.
    log: Vec<&'static str>,
}

/// The fake sandbox's writable layer, in MB by the fault: `DiskOver` 20,
/// then 400; `DiskOverBefore` 280, then 400; `DiskUnmeasurable` 20, then
/// failing; `LateWriter` 20 until the agent user's processes are killed,
/// then 400; `VerifyWriter` 20 until the verifier begins, then 400;
/// `CapSparse` 20 GB (a sparse file). 10 entries (`CapInodes`: then
/// 200,000). The stop leaves the sandbox stopped, as `docker kill` does.
struct FakeLayer {
    world: Arc<Mutex<World>>,
    fault: Fault,
    reads: Mutex<u32>,
    counts: Mutex<u32>,
}

#[async_trait]
impl benchflow_sandbox::disk::Layer for FakeLayer {
    async fn bytes(&self, _timeout: Duration) -> Result<u64, SandboxError> {
        let mut reads = self.reads.lock().unwrap();
        *reads += 1;
        let first = *reads == 1;
        let w = self.world.lock().unwrap();
        let mb = match self.fault {
            Fault::DiskUnmeasurableFromStart => {
                return Err(SandboxError::Transport(
                    "`docker container inspect --size` was killed after 10s".into(),
                ));
            }
            Fault::DiskOverBefore if first => 280,
            Fault::DiskUnmeasurable if !first => {
                return Err(SandboxError::Transport(
                    "`docker container inspect --size` was killed after 10s".into(),
                ));
            }
            Fault::DiskOver | Fault::DiskOverBefore if !first => 400,
            Fault::LateWriter if w.user_kills > 0 => 400,
            Fault::VerifyWriter if w.verifying => 400,
            Fault::ForkBombWriter if w.user_kills >= 2 => 400,
            Fault::CapSparse => 20 * 1024,
            _ => 20,
        };
        Ok(mb * 1024 * 1024)
    }
    async fn entries(&self, _timeout: Duration) -> Result<u64, SandboxError> {
        let mut counts = self.counts.lock().unwrap();
        *counts += 1;
        Ok(if self.fault == Fault::CapInodes && *counts > 1 {
            200_000
        } else {
            10
        })
    }
    async fn status(&self) -> Result<bool, SandboxError> {
        Ok(!self.world.lock().unwrap().killed)
    }
    async fn unpause(&self) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        self.world.lock().unwrap().killed = true;
        Ok(())
    }
}

/// The fake sandbox's `df` of each path asked: blocks and inodes, by the
/// fault and what happened so far.
fn fake_df(fault: Fault, w: &World, command: &str) -> String {
    const MB: u64 = 1024;
    let (avail, used, inodes) = match fault {
        Fault::DiskFull | Fault::DaemonDiskFull | Fault::ZeroOnFullDf | Fault::PassOnFullDf => {
            (0, 300 * MB, "3000000 100 2999900")
        }
        Fault::DiskFilled if w.spawned => (0, 300 * MB, "3000000 100 2999900"),
        Fault::DiskFilled => (290 * MB, 10 * MB, "3000000 100 2999900"),
        Fault::CapFilledAfter if w.user_kills > 0 => (10 * MB, 290 * MB, "3000000 100 2999900"),
        Fault::CapFilledAfter => (200 * MB, 100 * MB, "3000000 100 2999900"),
        Fault::InodesFull => (5_000 * MB, 300 * MB, "3000000 3000000 0"),
        Fault::CapHostFillBlocks | Fault::CapHostFillRecovered if w.spawned => {
            (0, 1024 * MB, "3000000 100 2999900")
        }
        Fault::CapHostFillBlocks | Fault::CapHostFillRecovered => {
            (600 * MB, 424 * MB, "3000000 100 2999900")
        }
        Fault::CapHostInodes if w.spawned => (900 * MB, 100 * MB, "10000000 9999000 1000"),
        Fault::CapHostInodes => (900 * MB, 100 * MB, "10000000 9997000 3000"),
        _ => (5_000 * MB, 300 * MB, "3000000 100 2999900"),
    };
    let n = command
        .split("-Pi")
        .nth(1)
        .map(|r| {
            r.split_whitespace()
                .filter(|t| !t.starts_with("2>"))
                .count()
        })
        .unwrap_or(1)
        .max(1);
    let mut out = "Filesystem 1024-blocks Used Available Capacity Mounted on\n".to_string();
    for _ in 0..n {
        out += &format!("overlay {} {used} {avail} 50% /\n", used + avail);
    }
    out += "BF-I\nFilesystem Inodes IUsed IFree IUse% Mounted on\n";
    for _ in 0..n {
        out += &format!("overlay {inodes} 1% /\n");
    }
    out
}

/// Probes of the daemon's disk, or measures of the layer, made for a
/// `CapNoWords` trial (review F2-C, F-5: there must be none).
static PROBED_WITHOUT_WORDS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn gone() -> SandboxError {
    SandboxError::Transport("the sandbox's API stopped answering (connection reset by peer)".into())
}

struct FakeSandbox {
    world: Arc<Mutex<World>>,
    fault: Fault,
    act: Act,
}

fn output(code: i32, stdout: &str, stderr: &str) -> ExecOutput {
    ExecOutput {
        exit_code: code,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        timed_out: false,
        duration: Duration::from_millis(1),
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-codes"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command;
        let silent = matches!(self.fault, Fault::Shadowed | Fault::ShadowedLies);
        if self.world.lock().unwrap().gone || self.world.lock().unwrap().killed {
            return Err(gone());
        }
        let verifier = c.contains("/verifier/test.sh") || c.contains("/tests/test.sh");
        if c.contains("kill -9 -1") {
            let mut w = self.world.lock().unwrap();
            w.user_kills += 1;
            w.log.push("kill");
        }
        if verifier {
            let mut w = self.world.lock().unwrap();
            w.verifying = true;
            w.log.push("verifier");
        }
        if self.fault == Fault::ShadowedLies && c == "true" {
            return Err(SandboxError::Transport(
                "the sandbox container fake-codes is no longer running (Error response from daemon: No such container: fake-codes)".into(),
            ));
        }
        match self.fault {
            Fault::OracleStops | Fault::OracleHalts if c.contains("solve.sh") => {
                let mut w = self.world.lock().unwrap();
                if self.fault == Fault::OracleStops {
                    w.stopped = true;
                } else {
                    w.halted = true;
                }
                return Err(SandboxError::Transport(
                    "the sandbox container fake-codes is no longer running".into(),
                ));
            }
            Fault::Outage if verifier => {
                self.world.lock().unwrap().api_down = true;
                return Err(SandboxError::Transport(
                    "the toolbox answered HTTP 502 while running the verifier".into(),
                ));
            }
            Fault::SetupGone if c.contains("useradd") => return Err(gone()),
            Fault::SetupHangs if c.contains("useradd") => {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Fault::VerifierGone if verifier => return Err(gone()),
            Fault::VerifierHangs if verifier => {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Fault::VerifyWriter if verifier => {
                // Long enough for the watch to measure the layer meanwhile.
                tokio::time::sleep(Duration::from_millis(400)).await;
                return Ok(output(0, "1 passed\n", ""));
            }
            Fault::ZeroOnFullDf
            | Fault::ZeroWithRoom
            | Fault::ZeroOnFullHost
            | Fault::PassOnFullDf
            | Fault::CapSparseZero
                if verifier =>
            {
                return Ok(output(
                    0,
                    "",
                    "ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device\n",
                ));
            }
            Fault::CapNoWords if verifier => {
                return Ok(output(
                    1,
                    "",
                    "/verifier/test.sh: line 3: pytest: command not found\n",
                ));
            }
            Fault::DiskFull
            | Fault::DiskFilled
            | Fault::DaemonDiskFull
            | Fault::FakeFull
            | Fault::CapHostFull
            | Fault::CapFilledAfter
            | Fault::InodesFull
            | Fault::CapFakedRoom
            | Fault::CapUnknownLayer
            | Fault::CapHostFillBlocks
            | Fault::CapHostFillRecovered
            | Fault::CapHostInodes
                if verifier =>
            {
                return Ok(output(
                    1,
                    "",
                    "OSError: [Errno 28] No space left on device: '/logs/verifier/reward.txt'\n",
                ));
            }
            Fault::VerifierKilled if verifier => {
                self.world.lock().unwrap().gone = true;
                return Ok(output(137, "", "Killed\n"));
            }
            Fault::VerifierStopsSandbox if verifier => {
                self.world.lock().unwrap().stopped = true;
                return Ok(output(143, "", "Terminated\n"));
            }
            Fault::VerifierHaltsSandbox if verifier => {
                self.world.lock().unwrap().halted = true;
                return Ok(output(143, "", "Terminated\n"));
            }
            _ if self.world.lock().unwrap().stopped || self.world.lock().unwrap().halted => {
                return Err(SandboxError::Transport(
                    "the sandbox container fake-codes is no longer running (Error response from daemon: container is not running)".into(),
                ));
            }
            _ => {}
        }
        Ok(if c.contains("memory.events") {
            // Before the turn, then after it.
            let mut w = self.world.lock().unwrap();
            w.memory_reads += 1;
            let after = w.memory_reads > 1;
            let (oom, kills) = match self.fault {
                Fault::AgentOom if after => (1, 1),
                Fault::HostOom if after => (0, 1),
                Fault::MixedOom if after => (1, 2),
                _ => (0, 0),
            };
            output(
                0,
                &format!("BF mem-max 2147483648\nBF mem-oom {oom}\nBF mem-oom_kill {kills}\n"),
                "",
            )
        } else if c.contains("BF-I") {
            let mut w = self.world.lock().unwrap();
            w.log.push("df");
            output(0, &fake_df(self.fault, &w, &c), "")
        } else if c.contains("kill -9 -1")
            && matches!(
                self.fault,
                Fault::ForkBomb | Fault::HostPids | Fault::ForkBombWriter
            )
        {
            if self.fault == Fault::ForkBombWriter && self.world.lock().unwrap().user_kills >= 2 {
                // The handoff's try: long enough for the watch to measure.
                tokio::time::sleep(Duration::from_millis(400)).await;
            }
            output(
                126,
                "",
                "OCI runtime exec failed: exec failed: unable to start container process: read init-p: connection reset by peer: unknown\n",
            )
        } else if c.contains("jump_new") {
            // The network policy's firewall install.
            if self.fault == Fault::NoIptables {
                output(0, "BF fw-error no-iptables\n", "")
            } else {
                output(
                    0,
                    "BF fw /usr/sbin/iptables v6=absent fw6=none reject=REJECT\nBF fw-sum 42\n",
                    "",
                )
            }
        } else if c.contains("BF canary") {
            // The firewall self-test, as the agent user.
            let verdict = if matches!(self.fault, Fault::Leaky | Fault::LeakyGone) {
                "reached"
            } else {
                "refused"
            };
            output(
                0,
                &format!("BF canary block {verdict} 192.0.2.50:443\nBF tamper tried\n"),
                "",
            )
        } else if c.contains("fingerprint") {
            output(0, "BF fw-sum 42\n", "")
        } else if c.contains("useradd") {
            if self.fault == Fault::Useradd {
                output(
                    1,
                    "",
                    "sh: 1: useradd: not found\nsh: 1: adduser: not found\n",
                )
            } else {
                output(0, "BF user agent 1000 /home/agent\n", "")
            }
        } else if c.contains("check-ready") {
            if self.fault == Fault::NotReady {
                output(
                    1,
                    "",
                    "curl: (7) Failed to connect to localhost port 8080\n",
                )
            } else {
                output(0, "", "")
            }
        } else if c.contains("kill -9 -1") {
            match self.fault {
                Fault::Transport => {
                    return Err(SandboxError::Transport(
                        "the sandbox's API stopped answering (connection reset by peer)".into(),
                    ));
                }
                Fault::Survivors => output(1, "BF survivors 4242 sleep\n", ""),
                Fault::KillGone => {
                    // What Daytona answered live: exit 0, no output.
                    self.world.lock().unwrap().gone = true;
                    output(0, "", "")
                }
                _ if silent => output(0, "", ""),
                _ => output(0, "BF killed 1\n", ""),
            }
        } else if c.contains("ld.so.preload") {
            match self.fault {
                Fault::PreloadStuck => output(1, "BF preload unrestorable\n", ""),
                _ => output(0, "BF preload absent\n", ""),
            }
        } else if c.contains("BF venv-done") {
            match self.fault {
                Fault::UncleanVerifierEnv => output(
                    0,
                    "BF venv-safepath 1\nBF venv-path-bad /usr/local/bin drwxrwxrwx\nBF venv-done\n",
                    "",
                ),
                _ => output(
                    0,
                    "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n",
                    "",
                ),
            }
        } else if c.contains("/root/snapshot/status.txt") && self.fault == Fault::HookFails {
            output(
                2,
                "",
                "cp: cannot stat '/app/status.txt': No such file or directory\n",
            )
        } else if c.contains("BF handoff done") {
            match self.fault {
                Fault::StateTampered => output(
                    0,
                    "BF state-owner /var/lib/benchflow/baseline/list\nBF handoff stopped\n",
                    "",
                ),
                _ => output(0, if silent { "" } else { "BF handoff done\n" }, ""),
            }
        } else if c.contains("BF owned") && self.fault == Fault::OwnFails {
            output(
                3,
                "",
                "chown: changing ownership of '/app/data': Read-only file system\nBF error chown /app\n",
            )
        } else if c.contains("BF owned") {
            output(0, if silent { "" } else { "BF owned root\n" }, "")
        } else if c.contains("BF stage-ready") {
            output(0, if silent { "" } else { "BF stage-ready\n" }, "")
        } else if c.contains("BF tree placed") {
            output(0, if silent { "" } else { "BF tree placed\n" }, "")
        } else if c.contains("BF baseline-added") {
            output(0, if silent { "" } else { "BF baseline-added 0\n" }, "")
        } else if c.contains("BF staged") {
            output(0, if silent { "" } else { "BF staged\n" }, "")
        } else if c.contains("BF placement done") {
            output(0, if silent { "" } else { "BF placement done\n" }, "")
        } else if c.contains("BF path") {
            output(
                0,
                if silent {
                    ""
                } else {
                    "BF path /usr/local/bin:/usr/bin:/bin\n"
                },
                "",
            )
        } else if c.contains("baseline") && c.contains("files.tar") {
            output(0, "BF baseline 2\n", "")
        } else if c.contains("/verifier/test.sh") || c.contains("/tests/test.sh") {
            match self.fault {
                Fault::VerifierTimeout | Fault::TimeoutDownloadFails => ExecOutput {
                    timed_out: true,
                    ..output(124, "collecting ...\n", "")
                },
                Fault::VerifierFails => output(
                    2,
                    "",
                    "/verifier/test.sh: line 3: pytest: command not found\n",
                ),
                _ => output(0, "1 passed\n", ""),
            }
        } else if c.contains("/oracle/solve.sh") {
            self.world.lock().unwrap().solved = true;
            output(0, "", "")
        } else {
            output(0, "", "")
        })
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        self.world.lock().unwrap().spawned = true;
        match self.fault {
            Fault::AgentSpawnGone => return Err(gone()),
            Fault::AgentOom | Fault::HostOom | Fault::MixedOom => {
                return Ok(fake_acp_agent(Act::Crash, self.world.clone()));
            }
            Fault::AgentGone => {
                self.world.lock().unwrap().gone = true;
                return Ok(fake_acp_agent(Act::Crash, self.world.clone()));
            }
            Fault::AgentStopped => {
                self.world.lock().unwrap().stopped = true;
                return Ok(fake_acp_agent(Act::Crash, self.world.clone()));
            }
            Fault::AgentHalted => {
                self.world.lock().unwrap().halted = true;
                return Ok(fake_acp_agent(Act::Crash, self.world.clone()));
            }
            _ => {}
        }
        Ok(fake_acp_agent(self.act, self.world.clone()))
    }
    async fn upload(&self, local: &Path, _remote: &str) -> Result<(), SandboxError> {
        if self.fault == Fault::LocalRead && local.ends_with("verifier") {
            return Err(SandboxError::io(
                format!("reading {} to upload it", local.display()),
                std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            ));
        }
        Ok(())
    }
    async fn write_file(&self, _remote: &str, _c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote.ends_with("outputs.tar") {
            // The outputs a separate verifier gets: none.
            tar::Builder::new(std::fs::File::create(local).unwrap())
                .finish()
                .unwrap();
            return Ok(());
        }
        if remote == "/logs/verifier" {
            let attempt = {
                let mut w = self.world.lock().unwrap();
                w.downloads += 1;
                w.downloads
            };
            if matches!(self.fault, Fault::DownloadFails | Fault::DownloadBreaksOff) {
                // The backend lost the sandbox as its folder was copied.
                self.world.lock().unwrap().gone = true;
            }
            if self.world.lock().unwrap().gone && self.fault != Fault::DownloadBreaksOff {
                return Err(gone());
            }
            if self.fault == Fault::LocalUnpack {
                return Err(SandboxError::io(
                    format!(
                        "copying /logs/verifier out of the sandbox into {}",
                        local.display()
                    ),
                    std::io::Error::new(std::io::ErrorKind::NotFound, "no such file or directory"),
                ));
            }
            if self.fault == Fault::VerifierFolderGone {
                return Err(SandboxError::io(
                    "downloading /logs/verifier",
                    std::io::Error::new(std::io::ErrorKind::NotFound, "no such file or directory"),
                ));
            }
            if self.world.lock().unwrap().stopped || self.world.lock().unwrap().halted {
                return Err(SandboxError::Transport(
                    "the sandbox container fake-codes is no longer running".into(),
                ));
            }
            match self.fault {
                Fault::DownloadRefused | Fault::TimeoutDownloadFails => {
                    return Err(SandboxError::Content(
                        "copying /logs/verifier out in fake-codes failed (exit code 2): tar: verifier/secret: Cannot open: Permission denied"
                            .into(),
                    ));
                }
                Fault::DownloadOverCap => {
                    return Err(SandboxError::io(
                        format!(
                            "copying {remote} out of the sandbox into {}",
                            local.display()
                        ),
                        std::io::Error::new(
                            std::io::ErrorKind::FileTooLarge,
                            "verifier/huge.bin would take the download past its cap",
                        ),
                    ));
                }
                Fault::DownloadChannel => return Err(gone()),
                _ => {}
            }
            std::fs::create_dir_all(local).unwrap();
            let solved = self.world.lock().unwrap().solved;
            // Part of the folder arrived (a reward of 0, the rest of the
            // verifier's files did not), then the copy failed.
            match self.fault {
                Fault::DownloadBreaksOff => {
                    std::fs::write(local.join("reward.txt"), "0\n").unwrap();
                    return Err(gone());
                }
                Fault::DownloadBreaksOnce if attempt == 1 => {
                    // A reward.json of 0 would win over the second copy's
                    // reward.txt if the folder were not cleared.
                    std::fs::write(local.join("reward.json"), "{\"reward\": 0}\n").unwrap();
                    return Err(gone());
                }
                Fault::HostDiskFull => {
                    std::fs::write(local.join("reward.txt"), "0\n").unwrap();
                    return Err(SandboxError::io(
                        format!(
                            "copying {remote} out of the sandbox into {}",
                            local.display()
                        ),
                        std::io::Error::from(std::io::ErrorKind::StorageFull),
                    ));
                }
                _ => {}
            }
            match self.fault {
                Fault::NoReward
                | Fault::VerifierFails
                | Fault::VerifierTimeout
                | Fault::DiskFull
                | Fault::DiskFilled
                | Fault::DaemonDiskFull
                | Fault::FakeFull
                | Fault::CapHostFull
                | Fault::CapFilledAfter
                | Fault::InodesFull
                | Fault::CapFakedRoom
                | Fault::CapUnknownLayer
                | Fault::CapNoWords
                | Fault::CapHostFillBlocks
                | Fault::CapHostFillRecovered
                | Fault::CapHostInodes => {}
                Fault::ZeroOnFullDf
                | Fault::ZeroWithRoom
                | Fault::ZeroOnFullHost
                | Fault::CapSparseZero => std::fs::write(local.join("reward.txt"), "0\n").unwrap(),
                Fault::PassOnFullDf => std::fs::write(local.join("reward.txt"), "1\n").unwrap(),
                Fault::BadReward => std::fs::write(local.join("reward.txt"), "passed\n").unwrap(),
                _ => std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" })
                    .unwrap(),
            }
        }
        Ok(())
    }
    fn own_cgroup(&self) -> bool {
        true
    }
    async fn processes(&self) -> Result<(u64, u64), SandboxError> {
        Ok(match self.fault {
            Fault::ForkBomb | Fault::ForkBombWriter => (4096, 4096),
            _ => (312, 4096),
        })
    }
    async fn host_df(&self) -> Option<benchflow_sandbox::disk::DfRow> {
        if self.fault == Fault::CapNoWords {
            PROBED_WITHOUT_WORDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        self.world.lock().unwrap().host_probes += 1;
        let full = matches!(
            self.fault,
            Fault::CapHostFull
                | Fault::ZeroOnFullHost
                | Fault::CapHostFillBlocks
                | Fault::CapHostInodes
        );
        let room = matches!(
            self.fault,
            Fault::ZeroOnFullDf
                | Fault::ZeroWithRoom
                | Fault::PassOnFullDf
                | Fault::CapFakedRoom
                | Fault::CapUnknownLayer
                | Fault::DiskFilled
                | Fault::CapFilledAfter
                | Fault::CapSparseZero
                | Fault::CapHostFillRecovered
        );
        (full || room).then_some(benchflow_sandbox::disk::DfRow {
            total_kb: 100 << 20,
            used_kb: if full { 100 << 20 } else { 1 << 20 },
            avail_kb: if full { 0 } else { 99 << 20 },
            inodes: benchflow_sandbox::disk::Inodes::NotApplicable,
        })
    }
    async fn layer_bytes(&self) -> Option<u64> {
        if self.fault == Fault::CapNoWords {
            PROBED_WITHOUT_WORDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        self.world.lock().unwrap().host_probes += 1;
        match self.fault {
            Fault::CapFakedRoom => Some(300 << 20),
            Fault::CapSparseZero => Some(20 << 30),
            _ => None,
        }
    }
    fn disk_limit(&self) -> benchflow_sandbox::DiskLimit {
        match self.fault {
            Fault::DiskOver
            | Fault::DiskOverBefore
            | Fault::DiskUnmeasurable
            | Fault::LateWriter
            | Fault::VerifyWriter
            | Fault::ForkBombWriter
            | Fault::DiskUnmeasurableFromStart => benchflow_sandbox::DiskLimit::NotEnforced(
                300,
                "the fake daemon cannot cap it".into(),
            ),
            Fault::DiskFilled
            | Fault::CapHostFull
            | Fault::CapInodes
            | Fault::CapFilledAfter
            | Fault::CapSparse
            | Fault::CapFakedRoom
            | Fault::CapUnknownLayer
            | Fault::CapNoWords
            | Fault::CapHostFillBlocks
            | Fault::CapHostFillRecovered
            | Fault::CapHostInodes
            | Fault::CapSparseZero => benchflow_sandbox::DiskLimit::Enforced(300),
            _ => benchflow_sandbox::DiskLimit::None,
        }
    }
    async fn watch_layer(
        &self,
        limit_mb: u64,
        watched: benchflow_sandbox::disk::Watched,
    ) -> Option<(
        benchflow_sandbox::LayerWatch,
        benchflow_sandbox::disk::StartState,
    )> {
        let layer = Arc::new(FakeLayer {
            world: self.world.clone(),
            fault: self.fault,
            reads: Mutex::new(0),
            counts: Mutex::new(0),
        });
        let ms = Duration::from_millis;
        Some(
            benchflow_sandbox::LayerWatch::start(
                limit_mb,
                benchflow_sandbox::disk::Pace {
                    bytes: ms(5),
                    entries: ms(5),
                    most: ms(50),
                    floor: ms(1),
                    base_timeout: ms(200),
                    entries_retry: ms(300),
                    bytes_timeout: ms(100),
                    bytes_bound: ms(150),
                    entries_timeout: ms(100),
                    entries_bound: ms(150),
                    settle: Duration::from_secs(5),
                },
                watched,
                layer,
            )
            .await,
        )
    }
    async fn presence(&self) -> Result<benchflow_sandbox::Presence, SandboxError> {
        let w = self.world.lock().unwrap();
        if w.killed {
            return Err(SandboxError::Transport(
                "the sandbox fake-codes was stopped from outside (docker kill)".into(),
            ));
        }
        if w.api_down {
            return Err(SandboxError::Transport(
                "the backend's API answered HTTP 503".into(),
            ));
        }
        Ok(if w.gone || self.fault == Fault::LeakyGone {
            benchflow_sandbox::Presence::Gone
        } else if w.stopped {
            benchflow_sandbox::Presence::Stopped
        } else if w.halted {
            benchflow_sandbox::Presence::Halted
        } else {
            benchflow_sandbox::Presence::Running
        })
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        if self.fault == Fault::DestroyFails {
            return Err(gone());
        }
        Ok(())
    }
}

/// A scripted ACP agent as a process: stdin, stdout, stderr and a handle.
fn fake_acp_agent(act: Act, world: Arc<Mutex<World>>) -> Process {
    let (client_stdin, agent_stdin) = tokio::io::duplex(64 * 1024);
    let (mut agent_stdout, client_stdout) = tokio::io::duplex(64 * 1024);
    let (mut agent_stderr, client_stderr) = tokio::io::duplex(64 * 1024);
    let task = tokio::spawn(async move {
        if act == Act::ExitAtStart {
            let _ = agent_stderr
                .write_all(b"fatal: no such file: /opt/fake/config.json\n")
                .await;
            return 1;
        }
        let mut lines = BufReader::new(agent_stdin).lines();
        let mut pending: Option<Value> = None;
        while let Ok(Some(line)) = lines.next_line().await {
            let msg: Value = serde_json::from_str(&line).unwrap();
            let id = msg.get("id").cloned();
            let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
            let update = |u: Value| {
                json!({"jsonrpc": "2.0", "method": "session/update",
                       "params": {"sessionId": "s1", "update": u}})
            };
            let mut out: Vec<Value> = Vec::new();
            match msg["method"].as_str().unwrap_or_default() {
                "initialize" => out.push(reply(json!({"protocolVersion": 1,
                    "agentCapabilities": {}, "authMethods": [],
                    "agentInfo": {"name": "fake", "version": "1.0"}}))),
                "session/new" if matches!(act, Act::NeedsSignIn) => {
                    out.push(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000,
                        "message": "Authentication required: Call authenticate before starting a session"}}));
                }
                "session/new" => out.push(reply(json!({"sessionId": "s1",
                    "models": {"currentModelId": "fake-model",
                        "availableModels": [{"modelId": "fake-model", "name": "Fake"}]},
                    "configOptions": [{"id": "model", "name": "Model", "type": "select",
                        "currentValue": "fake-model",
                        "options": [{"value": "fake-model", "name": "Fake"}]}]}))),
                "session/set_config_option" => {
                    let value = msg["params"]["value"].as_str().unwrap_or_default();
                    if value == "fake-model" {
                        out.push(reply(json!({"configOptions": []})));
                    } else {
                        out.push(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602,
                            "message": format!("Invalid value for config option model: {value}")}}));
                    }
                }
                "session/set_model" => out.push(reply(json!({}))),
                "session/prompt" => match act {
                    Act::Solve => {
                        out.push(update(json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Done."}})));
                        world.lock().unwrap().solved = true;
                        out.push(reply(json!({"stopReason": "end_turn"})));
                    }
                    Act::Crash => {
                        let _ = agent_stderr
                            .write_all(b"thread 'main' panicked at src/agent.rs:88: index out of bounds\n")
                            .await;
                        return 101;
                    }
                    Act::Error(message) => out.push(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32603, "message": message}})),
                    Act::ErrorLogged(message, log) => {
                        let _ = agent_stderr.write_all(format!("{log}\n").as_bytes()).await;
                        let _ = agent_stderr.flush().await;
                        out.push(json!({"jsonrpc": "2.0", "id": id,
                            "error": {"code": -32603, "message": message}}));
                    }
                    Act::ErrorDetails(message, details) => out.push(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32603, "message": message, "data": {"details":
                            serde_json::from_str::<serde_json::Value>(details).unwrap()}}})),
                    Act::Spent => out.push(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32603, "message": "Internal error",
                            "data": {"codexErrorInfo": "usageLimitExceeded",
                                "message": "You've hit your usage limit. Upgrade to Pro or try again at Oct 3rd, 2026 5:05 PM."}}})),
                    Act::Reply(text) => {
                        out.push(update(json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text}})));
                        out.push(reply(json!({"stopReason": "end_turn"})));
                    }
                    Act::Answer(text) => {
                        out.push(update(json!({"sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text}})));
                        out.push(reply(json!({"stopReason": "end_turn",
                            "usage": {"inputTokens": 900, "outputTokens": 12, "totalTokens": 912}})));
                    }
                    Act::Silent | Act::ExitAtStart | Act::NeedsSignIn => pending = id.clone(),
                },
                "session/cancel" => {
                    if let Some(pid) = pending.take() {
                        out.push(json!({"jsonrpc": "2.0", "id": pid,
                            "result": {"stopReason": "cancelled"}}));
                    }
                }
                _ => {
                    if id.is_some() {
                        out.push(json!({"jsonrpc": "2.0", "id": id,
                            "error": {"code": -32601, "message": "method not found"}}));
                    }
                }
            }
            for v in out {
                if agent_stdout
                    .write_all(format!("{v}\n").as_bytes())
                    .await
                    .is_err()
                {
                    return 0;
                }
            }
        }
        0
    });
    Process {
        stdin: Box::pin(client_stdin),
        stdout: Box::pin(client_stdout),
        stderr: Box::pin(client_stderr),
        handle: Box::new(FakeHandle { task: Some(task) }),
    }
}

struct FakeHandle {
    task: Option<tokio::task::JoinHandle<i32>>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        match self.task.take() {
            Some(t) => Ok(t.await.unwrap_or(137)),
            None => Ok(137),
        }
    }
    async fn kill(&mut self) -> Result<(), SandboxError> {
        if let Some(t) = self.task.take() {
            t.abort();
        }
        Ok(())
    }
}

struct FakeProvider {
    fault: Fault,
    act: Act,
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            network_none: true,
            pids_limit: true,
            build_from_dockerfile: true,
            ..Capabilities::default()
        }
    }
    async fn status(&self) -> ProviderStatus {
        ProviderStatus {
            ok: true,
            summary: "fake".into(),
            fix: None,
        }
    }
    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        log("building");
        let fail = |task_fault: bool, message: &str, tail: &str| SandboxError::Image {
            name: spec.name.clone(),
            message: message.to_string(),
            log_tail: tail.to_string(),
            task_fault,
        };
        match self.fault {
            Fault::TaskImage => Err(fail(
                true,
                "the task's Dockerfile failed to build (RUN pip install numpy==9.9). This is a problem in the task: fix the Dockerfile",
                "ERROR: No matching distribution found for numpy==9.9",
            )),
            Fault::ImageBackend => Err(fail(
                false,
                "`docker build` failed for a reason outside the task (failed to fetch oauth token: dial tcp: i/o timeout). Check the network and the docker host, then run again.",
                "dial tcp: i/o timeout",
            )),
            Fault::AgentLayers if !spec.layers.is_empty() => Err(fail(
                false,
                "installing the agent on top of the task image failed (npm ERR! 404 Not Found - GET https://registry.npmjs.org/@fake%2fagent). The agent's install step or the network is at fault, not the task; the log tail shows the failing command.",
                "npm ERR! 404 Not Found - GET https://registry.npmjs.org/@fake%2fagent",
            )),
            _ => Ok(Image {
                reference: format!("fake/{}:{}", spec.name, spec.layers.len()),
                fresh: true,
                duration: Duration::from_millis(1),
            }),
        }
    }
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        match self.fault {
            Fault::Start => {
                return Err(SandboxError::Start(
                    "the container exited at once: exec /entrypoint.sh: no such file or directory"
                        .into(),
                ));
            }
            Fault::NoGpus => {
                return Err(SandboxError::Unsupported(format!(
                    "{} asks for 1 GPU(s), and BenchFlow's Daytona backend has none; run it with `--sandbox docker` on a host with GPUs",
                    spec.name
                )));
            }
            _ => {}
        }
        Ok(Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World::default())),
            fault: self.fault,
            act: self.act,
        }))
    }
    fn services(&self) -> Option<&dyn ServiceHost> {
        Some(self as &dyn ServiceHost)
    }
}

#[async_trait]
impl ServiceHost for FakeProvider {
    async fn create_network(
        &self,
        name: &str,
        _labels: &BTreeMap<String, String>,
    ) -> Result<String, SandboxError> {
        Ok(format!("net-{name}"))
    }
    async fn start_service(
        &self,
        _network: &str,
        spec: &ServiceSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        if self.fault == Fault::ServiceFails {
            return Err(SandboxError::Start(format!(
                "{} exited at once (exit code 1): Error: Cannot find module '/srv/server.js'",
                spec.name
            )));
        }
        Ok(Box::new(FakeSandbox {
            world: Arc::new(Mutex::new(World::default())),
            fault: Fault::None,
            act: self.act,
        }))
    }
    async fn create_on_network(
        &self,
        spec: &SandboxSpec,
        _network: &str,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        self.create(spec).await
    }
    async fn logs(&self, _id: &str) -> String {
        String::new()
    }
    async fn remove_network(&self, _network: &str) -> Result<(), SandboxError> {
        Ok(())
    }
}

const MANIFEST: &str = r#"
name = "fake"
description = "a scripted ACP agent"
[install]
script = "npm install -g @fake/agent"
[launch]
command = "/opt/fake/bin/fake-acp"
[credentials]
env = ["FAKE_API_KEY"]
[model]
select = "set-model"
default = "fake-model"
"#;

fn manifest() -> AgentManifest {
    AgentManifest::parse(MANIFEST, ManifestSource::Builtin("fake.toml")).unwrap()
}

/// The scripted agent selecting its model as a session config option.
fn config_option_manifest() -> AgentManifest {
    let text = MANIFEST.replace(
        "select = \"set-model\"",
        "select = \"config-option\"\nconfig_id = \"model\"",
    );
    AgentManifest::parse(&text, ManifestSource::Builtin("fake.toml")).unwrap()
}

fn logged_in() -> HostEnv {
    HostEnv::new(
        BTreeMap::from([(
            "FAKE_API_KEY".to_string(),
            "fake-key-0123456789".to_string(),
        )]),
        None,
    )
}

/// An agent that ends the trial ungraded, as `bench shell --no-grade` does.
struct NoGrade;

#[async_trait]
impl ModelAgent for NoGrade {
    fn name(&self) -> &str {
        "shell"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, _ctx: AgentContext<'_>) -> AgentOutput {
        AgentOutput {
            skip_grading: true,
            ..AgentOutput::default()
        }
    }
}

/// A native-style agent whose provider rate-limited it until its time ran
/// out: its model calls log holds four 429s among six calls.
struct Throttled;

#[async_trait]
impl ModelAgent for Throttled {
    fn name(&self) -> &str {
        "throttled"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let dir = ctx.trial_dir.join("agent");
        std::fs::create_dir_all(&dir).unwrap();
        let calls: String = [200, 429, 429, 200, 429, 429]
            .iter()
            .map(|s| format!("{{\"seq\":1,\"status\":{s}}}\n"))
            .collect();
        std::fs::write(dir.join("model_calls.jsonl"), calls).unwrap();
        AgentOutput {
            timed_out: true,
            tool_calls: 2,
            ..AgentOutput::default()
        }
    }
}

/// An agent harness with a bug: it panics.
struct Panics;

#[async_trait]
impl ModelAgent for Panics {
    fn name(&self) -> &str {
        "buggy"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> Result<(), Problem> {
        Ok(())
    }
    async fn run(&self, _ctx: AgentContext<'_>) -> AgentOutput {
        panic!("index out of bounds: the len is 0 but the index is 0")
    }
}

fn task(root: &Path) -> TaskPackage {
    std::fs::create_dir_all(root.join("oracle")).unwrap();
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("oracle/solve.sh"), "#!/bin/bash\n").unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: "fixtures/codes".into(),
        name: "codes".into(),
        title: None,
        version: None,
        instruction: "Make the test pass.".into(),
        stages: Vec::new(),
        agent: AgentPolicy {
            timeout: Some(Duration::from_secs(60)),
            ..AgentPolicy::default()
        },
        sandbox: SandboxPolicy {
            image: ImageRef::Pull {
                reference: "ubuntu:24.04".into(),
            },
            cpus: None,
            memory_mb: None,
            disk_mb: None,
            gpus: 0,
            build_timeout: None,
            network: Network::Open,
            workdir: Some("/app".into()),
            env: BTreeMap::new(),
            ready: None,
            outputs: Vec::new(),
            skills: None,
            skills_dir: None,
            mcp: Vec::new(),
            allow_cloud_metadata: false,
        },
        verifier: VerifierPolicy {
            dir: root.join("verifier"),
            mount: "/verifier".into(),
            entry: "test.sh".into(),
            starts_in_mount: false,
            timeout: Some(Duration::from_secs(60)),
            user: None,
            env: BTreeMap::new(),
            network: None,
            isolation: Isolation::Shared,
            rubric: None,
            reward_range: None,
        },
        oracle: Some(OraclePolicy {
            dir: root.join("oracle"),
            mount: "/oracle".into(),
            entry: "solve.sh".into(),
            env: BTreeMap::new(),
        }),
        controls: Vec::new(),
        unsupported: Vec::new(),
        extras: Default::default(),
        config: json!({}),
        digest: "sha256:codes".into(),
    }
}

/// Give the task a rubric with one criterion a model judges.
fn judged_rubric(t: &mut TaskPackage, reference: Option<&str>) {
    let mut c = json!({"id": "memo", "text": "The memo names the total.", "judge": "llm",
        "points": 1, "evidence": ["memo.md"]});
    if let Some(r) = reference {
        c["reference"] = json!(r);
    }
    t.verifier.rubric = Some(
        benchflow_task::rubric::Rubric::from_value(&json!({
            "$schema": "https://task.md/schema/rubric-1.json", "criteria": [c]
        }))
        .unwrap(),
    );
}

/// Who attempts the task in a case.
enum Who {
    Oracle,
    /// The scripted ACP agent, on a login, with this model.
    Acp(Option<&'static str>),
    /// The scripted ACP agent asked for a reasoning effort.
    AcpEffort(&'static str),
    /// The scripted ACP agent selecting models as a config option.
    AcpOption(&'static str),
    /// The scripted ACP agent with no login on this machine.
    AcpNoLogin,
    NoGrade,
    Panics,
    Throttled,
}

struct Case {
    code: &'static str,
    owner: Owner,
    what: &'static str,
    who: Who,
    fault: Fault,
    act: Act,
    /// Change the task.
    tweak: fn(&mut TaskPackage),
    /// Run on this fixture task (`tests/fixtures/tasks/<name>`) instead.
    fixture: Option<&'static str>,
    /// Change the run's options.
    options: fn(&mut RunOptions),
    /// Stop the run (Ctrl-C) this long after it starts.
    stop_after: Option<Duration>,
}

impl Case {
    fn new(code: &'static str, owner: Owner, what: &'static str) -> Self {
        Self {
            code,
            owner,
            what,
            who: Who::Oracle,
            fault: Fault::None,
            act: Act::Solve,
            tweak: |_| {},
            fixture: None,
            options: |_| {},
            stop_after: None,
        }
    }
    fn who(mut self, who: Who) -> Self {
        self.who = who;
        self
    }
    fn fault(mut self, fault: Fault) -> Self {
        self.fault = fault;
        self
    }
    fn act(mut self, act: Act) -> Self {
        self.who = Who::Acp(None);
        self.act = act;
        self
    }
    fn tweak(mut self, tweak: fn(&mut TaskPackage)) -> Self {
        self.tweak = tweak;
        self
    }
    fn fixture(mut self, name: &'static str) -> Self {
        self.fixture = Some(name);
        self
    }
    fn options(mut self, options: fn(&mut RunOptions)) -> Self {
        self.options = options;
        self
    }
}

fn cases() -> Vec<Case> {
    use Owner::*;
    vec![
        Case::new(
            "R000",
            You,
            "an interactive session the user chose not to grade",
        )
        .who(Who::NoGrade),
        Case::new("R001", You, "Ctrl-C while the agent works")
            .act(Act::Silent)
            .options(|o| o.idle_timeout = Duration::ZERO),
        Case::new("R101", Task, "the task's Dockerfile fails to build").fault(Fault::TaskImage),
        Case::new(
            "R101",
            Task,
            "the task's Dockerfile fails under a model agent's install layers",
        )
        .act(Act::Solve)
        .fault(Fault::TaskImage),
        Case::new("R102", Sandbox, "the backend cannot prepare the image")
            .fault(Fault::ImageBackend),
        Case::new("R103", Sandbox, "the container fails to start").fault(Fault::Start),
        Case::new(
            "R103",
            Sandbox,
            "the connection to the sandbox drops before grading",
        )
        .fault(Fault::Transport),
        Case::new("R104", Task, "the image has no useradd or adduser").fault(Fault::Useradd),
        Case::new(
            "R108",
            Task,
            "a blocklist on an image with no iptables to enforce it",
        )
        .fault(Fault::NoIptables)
        .tweak(|t| t.sandbox.network = Network::Block(vec!["192.0.2.50".into()])),
        Case::new(
            "R109",
            Benchflow,
            "the firewall self-test reaches a blocked address",
        )
        .fault(Fault::Leaky)
        .tweak(|t| t.sandbox.network = Network::Block(vec!["192.0.2.50".into()])),
        Case::new("R105", Task, "the readiness check never passes")
            .fault(Fault::NotReady)
            .tweak(|t| {
                t.sandbox.ready = Some(Ready {
                    run: "check-ready".into(),
                    interval: Duration::from_millis(10),
                    timeout: Duration::from_secs(5),
                    start_period: Duration::ZERO,
                    start_interval: Duration::from_millis(10),
                    retries: 2,
                })
            }),
        Case::new(
            "R107",
            You,
            "the verifier's files cannot be read on this machine",
        )
        .fault(Fault::LocalRead),
        Case::new(
            "R110",
            Sandbox,
            "the sandbox's layer, mostly full before the agent's turn, passes a disk limit the daemon cannot cap",
        )
        .act(Act::Silent)
        .fault(Fault::DiskOverBefore)
        .tweak(|t| t.sandbox.disk_mb = Some(300)),
        Case::new(
            "R305",
            Sandbox,
            "the agent's turn fills the disk the daemon caps, and the verifier fails on it",
        )
        .act(Act::Solve)
        .fault(Fault::DiskFilled)
        .tweak(|t| t.sandbox.disk_mb = Some(300)),
        Case::new(
            "R316",
            Agent,
            "the sandbox's own memory limit runs out and the out-of-memory killer ends the agent",
        )
        .act(Act::Solve)
        .fault(Fault::AgentOom),
        Case::new(
            "R201",
            Agent,
            "the agent's install layer fails on a task image that builds",
        )
        .act(Act::Solve)
        .fault(Fault::AgentLayers),
        Case::new("R202", Agent, "the agent exits before the ACP handshake").act(Act::ExitAtStart),
        Case::new("R203", Agent, "the agent goes silent during the prompt")
            .act(Act::Silent)
            .options(|o| o.idle_timeout = Duration::from_secs(1)),
        Case::new("R204", Agent, "the agent crashes during the prompt").act(Act::Crash),
        Case::new(
            "R205",
            Agent,
            "the agent answers the prompt with an error of its own",
        )
        .act(Act::Error("Internal error: tool registry is empty")),
        Case::new(
            "R206",
            Agent,
            "the agent does nothing before its time runs out",
        )
        .act(Act::Silent)
        .options(|o| o.idle_timeout = Duration::ZERO)
        .tweak(|t| t.agent.timeout = Some(Duration::from_secs(1))),
        Case::new("R211", You, "no login for the agent on this machine").who(Who::AcpNoLogin),
        Case::new("R212", Model, "the provider rejects the key")
            .act(Act::Error("Invalid API key · Fix external API key")),
        Case::new("R213", Model, "the login's usage limit is reached").act(Act::Spent),
        Case::new("R214", Model, "the provider rate-limits the agent")
            .act(Act::Error("API Error: 429 rate limit exceeded")),
        Case::new(
            "R215",
            You,
            "the agent does not offer the model the run names",
        )
        .act(Act::Solve)
        .who(Who::AcpOption("other-model")),
        Case::new(
            "R215",
            You,
            "the run asks for a reasoning effort the agent lacks",
        )
        .act(Act::Solve)
        .who(Who::AcpEffort("high")),
        Case::new("R301", Task, "the verifier writes no reward").fault(Fault::NoReward),
        Case::new(
            "R302",
            Task,
            "the verifier writes a reward that is not a number",
        )
        .fault(Fault::BadReward),
        Case::new("R303", Task, "the verifier runs past its timeout").fault(Fault::VerifierTimeout),
        Case::new(
            "R304",
            Task,
            "the verifier exits with an error and writes no reward",
        )
        .fault(Fault::VerifierFails),
        Case::new(
            "R305",
            Sandbox,
            "agent processes survive the kill before grading",
        )
        .fault(Fault::Survivors),
        Case::new(
            "R306",
            Task,
            "--agent oracle on a task without a reference solution",
        )
        .tweak(|t| t.oracle = None),
        Case::new(
            "R307",
            Task,
            "a program on the image's PATH answers in place of the shell",
        )
        .fault(Fault::Shadowed),
        Case::new(
            "R309",
            Sandbox,
            "/etc/ld.so.preload cannot be put back before grading",
        )
        .fault(Fault::PreloadStuck),
        Case::new(
            "R310",
            Task,
            "the image's PATH holds a folder others can write",
        )
        .fault(Fault::UncleanVerifierEnv),
        Case::new("R311", Task, "a hook before grading fails")
            .fixture("snapshot-hook")
            .fault(Fault::HookFails),
        Case::new("R312", Model, "the rubric's model judge cannot be reached")
            .tweak(|t| judged_rubric(t, None))
            .options(|o| {
                o.judge_model = Some("vercel/judge/model".into());
                o.judge_env = benchflow_run::config::JudgeEnv(BTreeMap::from([
                    (
                        "AI_GATEWAY_API_KEY".to_string(),
                        "vck_codes_test_key".to_string(),
                    ),
                    // Nothing listens on the discard port.
                    (
                        "BENCHFLOW_NATIVE_BASE_URL".to_string(),
                        "http://127.0.0.1:9/v1".to_string(),
                    ),
                ]));
            }),
        Case::new(
            "R313",
            Task,
            "a rubric's reference names a file the package lacks",
        )
        .tweak(|t| judged_rubric(t, Some("verifier/answers.md"))),
        Case::new(
            "R314",
            Agent,
            "BenchFlow's state folder is no longer root's at the handoff",
        )
        .fault(Fault::StateTampered),
        Case::new(
            "R315",
            Task,
            "the workdir cannot be given to root before grading",
        )
        .fault(Fault::OwnFails),
        Case::new(
            "R106",
            Task,
            "a service beside the sandbox exits as it starts",
        )
        .fixture("sidecar")
        .fault(Fault::ServiceFails),
        Case::new(
            "R401",
            You,
            "the backend has no GPUs for a task that asks for one",
        )
        .fault(Fault::NoGpus),
        Case::new("R401", You, "the task asks for a feature this runner lacks").tweak(|t| {
            t.unsupported.push(Unsupported {
                field: "[sandbox] clock".into(),
                reason: "a fixed clock is not in the Rust runner yet".into(),
                agents_only: false,
            })
        }),
        Case::new("R901", Benchflow, "a bug in an agent harness: it panics").who(Who::Panics),
    ]
}

/// The case's trial record, with the temporary folder it ran in written as
/// `<tmp>` so the golden file does not change from run to run.
async fn run(case: &Case) -> Option<TrialRecord> {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().display().to_string();
    let mut t = match case.fixture {
        Some(name) => benchflow_task::load(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tasks")
                .join(name),
        )
        .unwrap_or_else(|d| panic!("{name}: {d:#?}")),
        None => task(&tmp.path().join("codes")),
    };
    (case.tweak)(&mut t);
    let attempt = match &case.who {
        Who::Oracle => Attempt::Oracle,
        Who::Acp(model) => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            *model,
            None,
            &logged_in(),
        ))),
        Who::AcpEffort(effort) => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            None,
            Some(effort),
            &logged_in(),
        ))),
        Who::AcpOption(model) => Attempt::Model(Arc::new(AcpAgent::new(
            config_option_manifest(),
            Some(model),
            None,
            &logged_in(),
        ))),
        Who::AcpNoLogin => Attempt::Model(Arc::new(AcpAgent::new(
            manifest(),
            None,
            None,
            &HostEnv::new(BTreeMap::new(), None),
        ))),
        Who::NoGrade => Attempt::Model(Arc::new(NoGrade)),
        Who::Panics => Attempt::Model(Arc::new(Panics)),
        Who::Throttled => Attempt::Model(Arc::new(Throttled)),
    };
    let mut options = RunOptions {
        retries: 0,
        unexplained_loss_retry_secs: 0,
        idle_timeout: Duration::from_secs(30),
        cache_dir: tmp.path().join("cache"),
        ..RunOptions::default()
    };
    (case.options)(&mut options);
    let (stop_tx, stop_rx) = stop_switch();
    let rt = Arc::new(Runtime {
        provider: Arc::new(FakeProvider {
            fault: case.fault,
            act: case.act,
        }),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: stop_rx,
    });
    if let Some(after) = case.stop_after {
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let _ = stop_tx.send(true);
        });
    } else {
        // A switch nobody holds any more must not read as a stop.
        drop(stop_tx);
    }
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![attempt])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("codes".into()),
        fresh: true,
        concurrency: 1,
        source: json!({"path": tmp.path()}),
    };
    let out = job::run(spec, rt).await.unwrap();
    out.records.first().cloned().map(|mut rec| {
        if let Some(p) = rec.problem.as_mut() {
            for text in [&mut p.message, &mut p.next] {
                *text = text.replace(&root, "<tmp>");
            }
        }
        rec
    })
}

/// A case's problem as a user reads it in `bench run`'s report.
fn shown(case: &Case, p: &Problem) -> String {
    let mut s = format!(
        "{} · {} · {} · {}\n  {}\n  next: {}\n",
        p.code,
        p.owner.problem(),
        p.phase.label(),
        case.what,
        p.message,
        p.next
    );
    if let Some(d) = &p.detail {
        let first = d.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        s.push_str(&format!("  detail: {first}\n"));
    }
    s
}

#[tokio::test]
async fn every_run_code_is_triggered_and_reads_as_written() {
    let mut cases = cases();
    for c in &mut cases {
        if c.code == "R001" {
            c.stop_after = Some(Duration::from_millis(400));
        }
    }
    let mut failures = Vec::new();
    let mut golden = String::new();
    for case in &cases {
        let Some(rec) = run(case).await else {
            failures.push(format!("{} ({}): no trial record", case.code, case.what));
            continue;
        };
        let Some(p) = &rec.problem else {
            failures.push(format!(
                "{} ({}): no problem recorded",
                case.code, case.what
            ));
            continue;
        };
        if p.code != case.code || p.owner != case.owner {
            failures.push(format!(
                "{} ({}): got {} owned by {}: {}",
                case.code,
                case.what,
                p.code,
                p.owner.label(),
                p.message
            ));
        }
        for (field, text) in [("message", &p.message), ("next step", &p.next)] {
            if text.trim().is_empty() {
                failures.push(format!("{}: an empty {field}", case.code));
            }
        }
        golden.push_str(&shown(case, p));
        golden.push('\n');
    }
    let triggered: BTreeSet<&str> = cases.iter().map(|c| c.code).collect();
    let listed: BTreeSet<&str> = EXPLANATIONS.iter().map(|e| e.code).collect();
    for code in listed.difference(&triggered) {
        failures.push(format!(
            "{code} is in `bench explain` but no case triggers it"
        ));
    }
    for code in triggered.difference(&listed) {
        failures.push(format!(
            "{code} is triggered here but `bench explain` lacks it"
        ));
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/codes/run.txt");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &golden).unwrap();
    } else {
        let want = std::fs::read_to_string(&path).unwrap_or_default();
        if want != golden {
            failures.push(format!(
                "tests/codes/run.txt is stale; run with UPDATE_GOLDEN=1 and review the diff. Got:\n{golden}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Failures beyond one per code: the status each records, and that a
/// sandbox or BenchFlow problem is never a score or the task's fault.
/// D4 chaos runs: a sandbox whose removal failed was named only in the
/// trial's log; the run now lists it at the end.
#[tokio::test]
async fn a_sandbox_that_cannot_be_removed_is_listed() {
    let case = Case::new("R000", Owner::You, "removal fails")
        .act(Act::Solve)
        .fault(Fault::DestroyFails);
    let rec = run(&case).await.expect("a trial record");
    assert_eq!(rec.status, benchflow_run::outcome::Status::Scored);
    assert!(
        benchflow_run::trial::left_behind()
            .iter()
            .any(|s| s.ends_with(" fake-codes")),
        "{:?}",
        benchflow_run::trial::left_behind()
    );
}

/// Review of rust/d4-chaos: a copy of the verifier's folder that broke
/// off once was never tried again, so a dropped connection cost the trial.
#[tokio::test]
async fn a_copy_that_breaks_off_once_is_made_again() {
    let case = Case::new("R000", Owner::You, "the copy breaks off once")
        .act(Act::Solve)
        .fault(Fault::DownloadBreaksOnce);
    let rec = run(&case).await.expect("a trial record");
    assert_eq!(
        rec.status,
        benchflow_run::outcome::Status::Scored,
        "{:?}",
        rec.problem
    );
    assert_eq!(rec.reward, Some(1.0));
}

/// F2 round 4: a verifier's words of a full disk change a score only on
/// evidence the agent cannot fake (review F2-C, F-6), a passed score always
/// stands, and a sparse file under a cap neither stops the trial nor scores
/// it 0 (F-3).
#[tokio::test]
async fn a_score_stands_without_evidence_of_a_full_disk() {
    let cases = [
        (
            Case::new("-", Owner::Agent, "a rootless agent's disk has room")
                .act(Act::Solve)
                .fault(Fault::ZeroWithRoom),
            Some(0.0),
        ),
        (
            Case::new(
                "-",
                Owner::Agent,
                "a root agent's df says full, the daemon's probe has room",
            )
            .act(Act::Solve)
            .fault(Fault::ZeroOnFullDf)
            .tweak(|t| t.agent.user = Some("root".into())),
            Some(0.0),
        ),
        (
            Case::new("-", Owner::Agent, "a passed score on a full disk")
                .act(Act::Solve)
                .fault(Fault::PassOnFullDf),
            Some(1.0),
        ),
        (
            Case::new(
                "-",
                Owner::Agent,
                "a rootless agent's sparse file past the cap does not fake a full quota",
            )
            .act(Act::Solve)
            .fault(Fault::CapSparseZero)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Some(0.0),
        ),
        (
            Case::new("-", Owner::Agent, "a sparse file far past the cap")
                .act(Act::Solve)
                .fault(Fault::CapSparse)
                .tweak(|t| t.sandbox.disk_mb = Some(300)),
            None,
        ),
    ];
    for (case, reward) in &cases {
        let rec = run(case).await.expect("a trial record");
        assert_eq!(
            rec.status,
            benchflow_run::outcome::Status::Scored,
            "{}: {:?}",
            case.what,
            rec.problem
        );
        assert!(rec.problem.is_none(), "{}: {:?}", case.what, rec.problem);
        if let Some(r) = reward {
            assert_eq!(rec.reward, Some(*r), "{}", case.what);
        }
    }
}

#[tokio::test]
async fn failures_record_the_right_status_owner_and_phase() {
    use Owner::*;
    use benchflow_run::outcome::{Phase, Status};
    let mut cases: Vec<(Case, Status, Phase)> = vec![
        (
            Case::new("R103", Sandbox, "the sandbox goes away during setup")
                .fault(Fault::SetupGone),
            Status::Errored,
            Phase::Setup,
        ),
        // Convergence round: every rerun path is capped. The task's setup
        // runs code inside the sandbox too; an unexplained loss there,
        // again right after one, is not run again.
        (
            Case::new(
                "R103",
                Sandbox,
                "the connection breaks during setup, in two attempts in a row",
            )
            .fault(Fault::SetupGone)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Setup,
        ),
        (
            Case::new("R103", Sandbox, "the sandbox cannot start the agent")
                .act(Act::Solve)
                .fault(Fault::AgentSpawnGone),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the sandbox goes away while the agent works",
            )
            .act(Act::Solve)
            .fault(Fault::AgentGone),
            Status::Errored,
            Phase::Agent,
        ),
        // Round-2 reviews of rust/e9-decisions (E9-B high, E9-A M1): a
        // sandbox the backend says code inside stopped while the agent
        // worked is the agent's failure (a root agent can stop it; a rerun
        // would reward that), not a lost sandbox rerun on every resume.
        (
            Case::new("R204", Agent, "the agent's code stops its sandbox while it works")
                .act(Act::Solve)
                .fault(Fault::AgentStopped),
            Status::Errored,
            Phase::Agent,
        ),
        // A stop the backend cannot explain is lost once (rerun)...
        (
            Case::new("R103", Sandbox, "the sandbox halts while the agent works, for no reason the backend can name")
                .act(Act::Solve)
                .fault(Fault::AgentHalted),
            Status::Errored,
            Phase::Agent,
        ),
        // ...and right after a lost attempt it is the agent's failure,
        // capped: a resumed job keeps it.
        (
            Case::new("R103", Sandbox, "the sandbox halts while the agent works, in two attempts in a row")
                .act(Act::Solve)
                .fault(Fault::AgentHalted)
                .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Agent,
        ),
        // Convergence reviews E9-D-A 1, E9-D-B 1: a scripted attempt (the
        // oracle, a script, a native loop's tool call) fails as the
        // sandbox's, not the agent's; the same one counter applies.
        (
            Case::new("R204", Agent, "the oracle's solve.sh stops its own sandbox")
                .who(Who::Oracle)
                .fault(Fault::OracleStops),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R103", Sandbox, "the oracle's sandbox halts for no reason the backend can name")
                .who(Who::Oracle)
                .fault(Fault::OracleHalts),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the oracle's sandbox halts for no reason the backend can name, in two attempts in a row",
            )
            .who(Who::Oracle)
            .fault(Fault::OracleHalts)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Agent,
        ),
        // E9-D-A 3: only the backend's own API can exempt an outage. The
        // toolbox fails while the control plane fails too: a confirmed
        // loss, rerun every time, never capped. (The same toolbox failure
        // while the control plane answers is capped after one rerun: the
        // "connection breaks ... in two attempts in a row" cases.)
        (
            Case::new(
                "R103",
                Sandbox,
                "the toolbox and the backend's API both fail while verifying, in two attempts in a row",
            )
            .act(Act::Solve)
            .fault(Fault::Outage)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R001", You, "Ctrl-C while the verifier runs").fault(Fault::VerifierHangs),
            Status::Skipped,
            Phase::Verify,
        ),
        // Decision 21, as clarified: a full disk nothing shows the agent
        // filled is the sandbox's, run again once; never the task's. Words
        // alone (the agent's program printing them) change nothing.
        (
            Case::new("R305", Sandbox, "the verifier fails on a full disk").fault(Fault::DiskFull),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "the verifier fails on a full disk the daemon does not cap (the host's)",
            )
            .act(Act::Solve)
            .fault(Fault::DaemonDiskFull),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap the daemon enforces, the daemon's own disk fills from outside and the verifier fails on it",
            )
            .act(Act::Solve)
            .fault(Fault::CapHostFull)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R304",
                Task,
                "the verifier's output says the disk is full, and it is not",
            )
            .fault(Fault::FakeFull),
            Status::Unscored,
            Phase::Verify,
        ),
        // The agent's 0, on the sandbox's own evidence only.
        (
            // Round 6 (DECISIONS row 13): a full capped disk is the sandbox's
            // R305, rerun once then kept, whoever filled it.
            Case::new(
                "R305",
                Sandbox,
                "the agent's turn fills the disk the daemon caps, and the verifier fails on it",
            )
            .act(Act::Solve)
            .fault(Fault::DiskFilled)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "the agent's turn writes past a disk limit the daemon cannot cap",
            )
            .act(Act::Silent)
            .fault(Fault::DiskOver)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Scored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "the agent's turn floods the disk the daemon caps with empty files",
            )
            .act(Act::Silent)
            .fault(Fault::CapInodes)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Scored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R110",
                Sandbox,
                "the layer cannot be measured, even frozen, while the sandbox runs",
            )
            .act(Act::Silent)
            .fault(Fault::DiskUnmeasurable)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "the kill before grading cannot start: the sandbox is at its own process limit",
            )
            .act(Act::Solve)
            .fault(Fault::ForkBomb),
            Status::Scored,
            // F2 round 5 (review F2-A-r4 M4): read when the kill at the
            // end of the agent's turn cannot start.
            Phase::Agent,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "the kill before grading cannot start, below the sandbox's own process limit",
            )
            .act(Act::Solve)
            .fault(Fault::HostPids),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new(
                "R204",
                Sandbox,
                "the host's out-of-memory killer ends the agent",
            )
            .act(Act::Solve)
            .fault(Fault::HostOom),
            Status::Errored,
            Phase::Agent,
        ),
        // F2 round 4 (reviews F2-A-r3, F2-B-r3, F2-C).
        (
            Case::new(
                "R204",
                Sandbox,
                "the sandbox's own memory limit runs out once and two processes are killed",
            )
            .act(Act::Solve)
            .fault(Fault::MixedOom),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, the agent's background writer fills the cap after its attempt returns",
            )
            .act(Act::Solve)
            .fault(Fault::CapFilledAfter)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, the host's disk is full at the close of an honest turn and has room again by grading",
            )
            .act(Act::Solve)
            .fault(Fault::CapHostFillRecovered)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "on a watched layer, the agent's writer passes the limit after its attempt returns",
            )
            .act(Act::Solve)
            .fault(Fault::LateWriter)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Scored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R110",
                Sandbox,
                "the layer passes the limit while the verifier runs",
            )
            .act(Act::Solve)
            .fault(Fault::VerifyWriter)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "the verifier fails on a disk whose inodes are used up",
            )
            .fault(Fault::InodesFull),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "the verifier scores 0 on a full disk, shown full by a rootless agent's sandbox",
            )
            .act(Act::Solve)
            .fault(Fault::ZeroOnFullDf),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "a root agent's verifier scores 0 on a full disk the daemon's probe shows full",
            )
            .act(Act::Solve)
            .fault(Fault::ZeroOnFullHost)
            .tweak(|t| t.agent.user = Some("root".into())),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, a root agent's df says there is room, the daemon measures the layer at the cap, and the verifier fails on it",
            )
            .act(Act::Solve)
            .fault(Fault::CapFakedRoom)
            .tweak(|t| {
                t.sandbox.disk_mb = Some(300);
                t.agent.user = Some("root".into());
            }),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, a root agent's layer cannot be measured, the daemon's disk has room, and the verifier fails on a full disk",
            )
            .act(Act::Solve)
            .fault(Fault::CapUnknownLayer)
            .tweak(|t| {
                t.sandbox.disk_mb = Some(300);
                t.agent.user = Some("root".into());
            }),
            Status::Errored,
            Phase::Verify,
        ),
        // F2 round 5.
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, a neighbour fills the daemon's disk during an honest agent's turn and the verifier fails on it",
            )
            .act(Act::Solve)
            .fault(Fault::CapHostFillBlocks)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "under a cap, the host's inodes run low during an honest agent's turn and the verifier fails",
            )
            .act(Act::Solve)
            .fault(Fault::CapHostInodes)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "a separate verifier scores 0 on a full disk",
            )
            .act(Act::Solve)
            .fault(Fault::ZeroOnFullDf)
            .tweak(|t| t.verifier.isolation = benchflow_task::package::Isolation::Separate),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "at its own process limit when its turn ends, with a writer that passes the limit after",
            )
            .act(Act::Solve)
            .fault(Fault::ForkBombWriter)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Scored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R110",
                Sandbox,
                "the layer cannot be measured before the agent's first turn",
            )
            .act(Act::Solve)
            .fault(Fault::DiskUnmeasurableFromStart)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Setup,
        ),
        // No word of a full disk: no probe of the daemon's disk, no
        // measure of the layer (checked after the cases run).
        (
            Case::new(
                "R304",
                Task,
                "under a cap, the verifier writes no reward, with no word of a full disk",
            )
            .act(Act::Solve)
            .fault(Fault::CapNoWords)
            .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Unscored,
            Phase::Verify,
        ),
        // Decision 22: the disk limit's watch stops a sandbox whose layer
        // passed the limit. Nothing yet says the agent wrote it, so it is
        // the sandbox's, rerun once; a repeat is kept as it is.
        (
            Case::new("R110", Sandbox, "the disk limit's watch stops the sandbox")
                .act(Act::Silent)
                .fault(Fault::DiskOverBefore)
                .tweak(|t| t.sandbox.disk_mb = Some(300)),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R110",
                Sandbox,
                "the disk limit's watch stops the sandbox, in two attempts in a row",
            )
            .act(Act::Silent)
            .fault(Fault::DiskOverBefore)
            .tweak(|t| t.sandbox.disk_mb = Some(300))
            .options(|o| {
                o.retries = 1;
                o.unexplained_loss_retry_secs = 0;
            }),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R214", Model, "429s use up the agent's time").who(Who::Throttled),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R316",
                Agent,
                "the sandbox's own memory limit runs out and the out-of-memory killer ends the agent",
            )
            .act(Act::Solve)
            .fault(Fault::AgentOom),
            Status::Scored,
            Phase::Agent,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the connection drops while the verifier runs",
            )
            .fault(Fault::VerifierGone),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R103", Sandbox, "the sandbox is killed under the verifier")
                .fault(Fault::VerifierKilled),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R103", Sandbox, "the verifier's folder cannot be fetched")
                .fault(Fault::DownloadFails),
            Status::Errored,
            Phase::Verify,
        ),
        // D4 chaos runs (rust/d4-chaos): a full disk on this machine while
        // the verifier's folder was copied in scored the oracle 0 from the
        // files that arrived. Review of rust/d4-chaos (Outcomes 1): under
        // the copy's cap, this machine's full disk is yours, and rerun.
        (
            Case::new("R107", You, "this machine's disk fills while the verifier's folder is copied")
                .act(Act::Solve)
                .fault(Fault::HostDiskFull),
            Status::Errored,
            Phase::Verify,
        ),
        // A copy the folder stops is never graded, and never run again (the
        // verifier may have run the agent's code).
        (
            Case::new("R301", Task, "tar refuses a file in the verifier's folder")
                .act(Act::Solve)
                .fault(Fault::DownloadRefused),
            Status::Unscored,
            Phase::Verify,
        ),
        (
            Case::new("R301", Task, "the verifier's folder is over the copy's cap")
                .act(Act::Solve)
                .fault(Fault::DownloadOverCap),
            Status::Unscored,
            Phase::Verify,
        ),
        // Review of rust/d4-chaos (Outcomes 2): a connection that breaks
        // off while the sandbox runs is the sandbox's, and rerun.
        (
            Case::new("R103", Sandbox, "the connection breaks off during both copies")
                .act(Act::Solve)
                .fault(Fault::DownloadChannel),
            Status::Errored,
            Phase::Verify,
        ),
        // Final review of rust/d4-chaos (finding 2): code inside (root)
        // can break the connection the same way in every attempt, buying a
        // rerun each time. Lost in the same phase again right after the
        // retry (or a resume), while the backend still has the sandbox, the
        // trial is not run again (capped), and counts as not passed. It
        // stays the sandbox's (round-2 review E9-B M2: an outage looks the
        // same, so the task is not blamed).
        (
            Case::new(
                "R103",
                Sandbox,
                "the connection breaks off during both copies, in two attempts in a row",
            )
            .act(Act::Solve)
            .fault(Fault::DownloadChannel)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the connection drops before grading, in two attempts in a row",
            )
            .act(Act::Solve)
            .fault(Fault::Transport)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Handoff,
        ),
        // A loss the backend confirms (the sandbox deleted) is rerun every
        // time.
        (
            Case::new(
                "R103",
                Sandbox,
                "the sandbox is deleted during the kill before grading, in two attempts in a row",
            )
            .act(Act::Solve)
            .fault(Fault::KillGone)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Handoff,
        ),
        // Review of rust/d4-chaos (Outcomes 3): code the verifier runs
        // stops the sandbox; it still exists, so the failure stays the
        // task's instead of buying a rerun.
        (
            Case::new("R301", Task, "the verifier's code stops the sandbox")
                .act(Act::Solve)
                .fault(Fault::VerifierStopsSandbox),
            Status::Unscored,
            Phase::Verify,
        ),
        // Diff review of rust/e9-decisions (H1): on Podman (or Daytona's
        // `error`) the backend cannot say who stopped a sandbox. The first
        // such stop is lost and retried, but code inside that stops it in
        // every attempt must not buy a rerun each time: the repeat is the
        // task's.
        (
            Case::new("R103", Sandbox, "the verifier's code halts a sandbox the backend cannot explain")
                .act(Act::Solve)
                .fault(Fault::VerifierHaltsSandbox),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R103",
                Sandbox,
                "the verifier's code halts a sandbox the backend cannot explain, in two attempts in a row",
            )
            .act(Act::Solve)
            .fault(Fault::VerifierHaltsSandbox)
            .options(|o| o.retries = 1),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new("R303", Task, "the verifier times out and its folder cannot be copied")
                .fault(Fault::TimeoutDownloadFails),
            Status::Unscored,
            Phase::Verify,
        ),
        // Reviews of rust/d4-chaos: an agent's planted shell answered the
        // liveness check with the CLI's "No such container", which turned
        // the task's R307 into a rerun.
        (
            Case::new("R307", Task, "a planted shell claims the sandbox is gone")
                .act(Act::Solve)
                .fault(Fault::ShadowedLies),
            Status::Unscored,
            Phase::Handoff,
        ),
        (
            Case::new("R103", Sandbox, "the backend loses the sandbox during the firewall self-test")
                .fault(Fault::LeakyGone)
                .tweak(|t| t.sandbox.network = Network::Block(vec!["192.0.2.50".into()])),
            Status::Errored,
            Phase::Setup,
        ),
        // D4 chaos runs: a sandbox deleted through Daytona's API during the
        // kill before grading was R307, the task's image.
        (
            Case::new("R103", Sandbox, "the sandbox is deleted during the kill before grading")
                .act(Act::Solve)
                .fault(Fault::KillGone),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new("R103", Sandbox, "the copy of the verifier's folder breaks off")
                .act(Act::Solve)
                .fault(Fault::DownloadBreaksOff),
            Status::Errored,
            Phase::Verify,
        ),
        (
            // The sandbox still answers: no reward is the task's.
            Case::new("R301", Task, "the verifier removes its own folder")
                .fault(Fault::VerifierFolderGone),
            Status::Unscored,
            Phase::Verify,
        ),
        (
            // A failure on this machine is the user's, not the task's,
            // even when it reads "not found" (FIX-REVIEW-B 4 on 61deae52).
            Case::new(
                "R107",
                You,
                "the verifier's folder cannot be unpacked on this machine",
            )
            .fault(Fault::LocalUnpack),
            Status::Errored,
            Phase::Verify,
        ),
        (
            Case::new(
                "R305",
                Sandbox,
                "agent processes survive the kill before grading",
            )
            .fault(Fault::Survivors),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new("R309", Sandbox, "/etc/ld.so.preload cannot be put back")
                .fault(Fault::PreloadStuck),
            Status::Errored,
            Phase::Handoff,
        ),
        (
            Case::new("R301", Task, "the verifier writes no reward").fault(Fault::NoReward),
            Status::Unscored,
            Phase::Verify,
        ),
        (
            Case::new("R001", You, "Ctrl-C while the sandbox is set up").fault(Fault::SetupHangs),
            Status::Skipped,
            Phase::Setup,
        ),
        (
            Case::new("R212", Model, "the provider answers 401").act(Act::Error(
                "API Error: 401 {\"error\":{\"message\":\"Invalid API key\"}}",
            )),
            Status::Errored,
            Phase::Agent,
        ),
        (
            // hermes-acp today: a dead key's 401 returned as its reply.
            Case::new("R212", Model, "the provider's 401 is the agent's only reply").act(
                Act::Reply(
                    "HTTP 401: login fail: Please carry the API secret key in the 'X-Api-Key' field of the request header",
                ),
            ),
            Status::Errored,
            Phase::Agent,
        ),
        (
            // cline-acp and kimi-acp today.
            Case::new("R211", You, "the ACP server needs the vendor's sign-in")
                .act(Act::NeedsSignIn),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R215", You, "the provider retired the model")
                .act(Act::Error("Internal error: Exception: [410] Gone")),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R214", Model, "the provider answers 500")
                .act(Act::Error("API Error: 500 Internal Server Error")),
            Status::Errored,
            Phase::Agent,
        ),
        (
            Case::new("R214", Model, "the provider times out")
                .act(Act::Error("API Error: Request timed out.")),
            Status::Errored,
            Phase::Agent,
        ),
    ];
    for (c, _, _) in &mut cases {
        if matches!(c.fault, Fault::SetupHangs | Fault::VerifierHangs) {
            c.stop_after = Some(Duration::from_millis(400));
        }
    }
    let mut failures = Vec::new();
    for (case, status, phase) in &cases {
        let rec = run(case).await.expect("a trial record");
        let Some(p) = rec.problem.as_ref() else {
            failures.push(format!(
                "{}: no problem ({:?}, reward {:?})",
                case.what, rec.status, rec.reward
            ));
            continue;
        };
        if p.code != case.code
            || p.owner != case.owner
            || rec.status != *status
            || p.phase != *phase
        {
            failures.push(format!(
                "{}: want {} {:?} {:?} {:?}, got {} {:?} {:?} {:?}: {}",
                case.what,
                case.code,
                case.owner,
                status,
                phase,
                p.code,
                p.owner,
                rec.status,
                p.phase,
                p.message
            ));
        }
        // Review of rust/d4-chaos: the R103 kept R307's text.
        if matches!(case.fault, Fault::KillGone | Fault::LeakyGone)
            && (p.message.contains("PATH") || p.message.contains("reached"))
        {
            failures.push(format!(
                "{}: the message is not the sandbox's: {p:?}",
                case.what
            ));
        }
        // A repeat is capped (kept by a resume) and says so; a single loss
        // is neither.
        let repeat = (case.what.contains("in two attempts in a row")
            && !matches!(case.fault, Fault::KillGone | Fault::Outage))
            || case.what.contains("stops its");
        if repeat != p.rerun_capped || (repeat && !p.message.contains("not run again")) {
            failures.push(format!(
                "{}: capped {} (want {repeat}), or the repeat is not named: {p:?}",
                case.what, p.rerun_capped
            ));
        }
        if matches!(case.fault, Fault::DiskOver | Fault::DiskOverBefore)
            && !(p.message.contains("400 MB") && p.message.contains("disk limit of 300 MB"))
        {
            failures.push(format!(
                "{}: the layer and limit are not named: {p:?}",
                case.what
            ));
        }
        if case.fault == Fault::DiskOverBefore
            && !rec.defenses.get("disk_limit").is_some_and(|d| {
                d.note
                    .as_deref()
                    .is_some_and(|n| n.starts_with("disk limit not enforced"))
            })
        {
            failures.push(format!(
                "{}: no disk_limit record: {:?}",
                case.what, rec.defenses
            ));
        }
        if case.fault == Fault::DiskUnmeasurable
            && !(p.message.contains("could not be measured")
                && rec.defenses.get("disk_limit").is_some_and(|d| {
                    d.note
                        .as_deref()
                        .is_some_and(|n| n.contains("could go unmeasured for"))
                }))
        {
            failures.push(format!(
                "{}: the unmeasured layer is not named: {p:?} {:?}",
                case.what, rec.defenses
            ));
        }
        // The agent never ran: F1's metadata warning must not count it.
        if case.fault == Fault::DiskUnmeasurableFromStart
            && (rec.agent_started() || rec.model_reached_metadata().is_some())
        {
            failures.push(format!("{}: counted as an agent that ran", case.what));
        }
        if case.fault == Fault::CapNoWords
            && PROBED_WITHOUT_WORDS.load(std::sync::atomic::Ordering::SeqCst) > 0
        {
            failures.push(format!(
                "{}: the daemon's disk was probed with no word of a full disk",
                case.what
            ));
        }
        if case.fault == Fault::CapInodes && !p.message.contains("200000 entries") {
            failures.push(format!("{}: the entries are not named: {p:?}", case.what));
        }
        if case.fault == Fault::DiskFull && !p.message.contains("disk was full") {
            failures.push(format!("{}: the full disk is not named: {p:?}", case.what));
        }
        if case.fault == Fault::AgentOom
            && !(p.message.contains("out-of-memory killer") && p.next.contains("memory"))
        {
            failures.push(format!(
                "{}: the memory kill is not named: {p:?}",
                case.what
            ));
        }
        // The agent's 0 for exhausting its own sandbox is a score of 0.
        if p.code == "R316" && (rec.reward != Some(0.0) || rec.passed != Some(false)) {
            failures.push(format!(
                "{}: R316 not scored 0: {:?}",
                case.what, rec.reward
            ));
        }
        if p.owner != Task && p.code != "R316" && rec.reward.is_some() {
            failures.push(format!(
                "{}: scored though {}",
                case.what,
                p.owner.problem()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Only the provider's own failure leaves a trial ungraded: a model that
/// was reached and answered with words that read like an error is graded,
/// and a key in the reply never reaches the record.
#[tokio::test]
async fn an_answer_that_reads_like_a_provider_error_is_graded() {
    use benchflow_run::outcome::Status;
    let answered = Case::new("R214", Owner::Model, "a model answers like an error")
        .act(Act::Answer("Error: rate limit exceeded, try again later"));
    let rec = run(&answered).await.unwrap();
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(rec.reward, Some(0.0));

    let failed = Case::new("R212", Owner::Model, "a 401 echoing the key")
        .act(Act::Reply("HTTP 401: invalid api key fake-key-0123456789"));
    let rec = run(&failed).await.unwrap();
    assert_eq!(rec.status, Status::Errored);
    let p = rec.problem.unwrap();
    assert!(p.message.contains("HTTP 401"), "{}", p.message);
    assert!(!p.message.contains("fake-key-0123456789"), "{}", p.message);

    // A stream that broke after the model began: the partial answer's
    // tokens do not make the provider's failure the agent's (FIX-REVIEW-B
    // 6 on 61deae52).
    let broke = Case::new("R214", Owner::Model, "a stream broken mid-response")
        .act(Act::Answer("API Error: Server error mid-response"));
    let rec = run(&broke).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    assert_eq!(rec.problem.unwrap().code, "R214");
    // The same after a partial answer, where only the last line is the
    // error (REVIEW-CORRECT L2).
    let partial = Case::new(
        "R214",
        Owner::Model,
        "a partial answer, then a broken stream",
    )
    .act(Act::Answer(
        "Here is the fix: change the loop bound.\nAPI Error: Server error mid-response",
    ));
    let rec = run(&partial).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    assert_eq!(rec.problem.unwrap().code, "R214");
    // ...and with no line break before the error (REVIEW-OUTCOMES L3).
    let joined = Case::new(
        "R214",
        Owner::Model,
        "a partial answer run into a broken stream",
    )
    .act(Act::Answer(
        "Here is the fix: change the loop bound.API Error: Server error mid-response",
    ));
    let rec = run(&joined).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    assert_eq!(rec.problem.unwrap().code, "R214");
    // ...and after a partial answer cut mid-word, as a broken stream
    // usually leaves it (REVIEW-B (d7d) F1: graded as the answer).
    let cut = Case::new(
        "R214",
        Owner::Model,
        "a partial answer cut mid-word, then a broken stream",
    )
    .act(Act::Answer(
        "Here is the fix: change the loop bouAPI Error: Server error mid-response",
    ));
    let rec = run(&cut).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    assert_eq!(rec.problem.unwrap().code, "R214");
    // A partial answer, then the provider failing outright, is the
    // provider's too; an answer that only quotes such an error mid-sentence
    // is graded (REVIEW-OUTCOMES (d7c) L4).
    let overloaded = Case::new("R214", Owner::Model, "a partial answer, then a 529")
        .act(Act::Answer("Here is the fix.\nAPI Error: 529 Overloaded"));
    let rec = run(&overloaded).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    let quoted = Case::new("R214", Owner::Model, "an answer that quotes an error").act(
        Act::Answer("To handle `API Error: Server error mid-response`, retry with backoff."),
    );
    let rec = run(&quoted).await.unwrap();
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    // A quote that ends the answer is still the answer.
    let quoted = Case::new("R214", Owner::Model, "an answer that ends quoting an error").act(
        Act::Answer("Claude Code then prints `API Error: Server error mid-response`."),
    );
    let rec = run(&quoted).await.unwrap();
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    // A spent credit balance in the error's own words is the login's limit
    // (R213), not the agent's error (REVIEW-OUTCOMES (d7c) L2).
    let spent = Case::new("R213", Owner::Model, "a credit balance spent, in words").act(
        Act::Error("Internal error: Your credit balance is too low to access the Anthropic API"),
    );
    let rec = run(&spent).await.unwrap();
    let p = rec.problem.unwrap();
    assert_eq!(p.code, "R213", "{}", p.message);
    // A quota error whose retry delay is on another line of the message,
    // or only in `data.details`, is a per-minute rate limit (R214, retried);
    // one whose per-day quota id is only in the details is spent (R213).
    // Both reviews' inputs (REVIEW-C (orch) 1).
    for (code, transient, why, message, details) in [
        (
            "R214",
            true,
            "a quota error, its retry delay on the next line",
            "Internal error: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\nPlease retry in 20.5s.",
            r#"{"retryDelay": "20s"}"#,
        ),
        (
            "R214",
            true,
            "a quota error, its retry delay only in the details",
            "Internal error: You exceeded your current quota",
            r#"{"retryDelay": "20s"}"#,
        ),
        (
            "R213",
            false,
            "a daily quota error, its day only in the details",
            "Internal error: You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\nPlease retry in 33s.",
            r#"{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}], "links": [{"url": "https://ai.google.dev/gemini-api/docs/rate-limits"}], "retryDelay": "33s"}"#,
        ),
    ] {
        let case = Case::new(code, Owner::Model, why).act(Act::ErrorDetails(message, details));
        let rec = run(&case).await.unwrap();
        let p = rec.problem.unwrap();
        assert_eq!(
            (p.code.as_str(), p.transient),
            (code, transient),
            "{why}: {}",
            p.message
        );
    }
    let delay = Case::new("R214", Owner::Model, "a quota error with a retry delay").act(
        Act::Error("Internal error: You exceeded your current quota. Please retry in 20.5s."),
    );
    let p = run(&delay).await.unwrap().problem.unwrap();
    assert_eq!(
        (p.code.as_str(), p.owner, p.transient),
        ("R214", Owner::Model, true),
        "{}",
        p.message
    );
    // A JSON-RPC error that is a refused request: the provider's, not
    // retried; one that names the context window: the agent's, whatever
    // the log says (REVIEW-OUTCOMES L1, L7).
    let refused = Case::new("R214", Owner::Model, "a JSON-RPC refused request").act(Act::Error(
        "Internal error: HTTP 400: Unsupported parameter 'temperature'",
    ));
    let rec = run(&refused).await.unwrap();
    let p = rec.problem.unwrap();
    assert_eq!(
        (p.code.as_str(), p.transient),
        ("R214", false),
        "{}",
        p.message
    );
    // Its log ends in a tool's HTTP 400, which must not make it the
    // provider's (REVIEW-OUTCOMES (d7c) L6: the case read no log before).
    let overflow =
        Case::new("R205", Owner::Agent, "a JSON-RPC context overflow").act(Act::ErrorLogged(
            "Internal error: API Error: prompt is too long: 210000 tokens > 200000 maximum",
            "curl: (22) The requested URL returned error: HTTP 400 Bad Request",
        ));
    let rec = run(&overflow).await.unwrap();
    let p = rec.problem.unwrap();
    assert_eq!(
        (p.code.as_str(), p.owner),
        ("R205", Owner::Agent),
        "{}",
        p.message
    );
    // A refused request is the provider's but never retried
    // (REVIEW-CORRECT M1; 0.8's provider_rejected is permanent).
    let refused = Case::new("R214", Owner::Model, "a refused request")
        .act(Act::Reply("HTTP 400: Unsupported parameter 'temperature'"));
    let rec = run(&refused).await.unwrap();
    assert_eq!(rec.status, Status::Errored, "{:?}", rec.problem);
    let p = rec.problem.unwrap();
    assert_eq!((p.code.as_str(), p.transient), ("R214", false));
}
