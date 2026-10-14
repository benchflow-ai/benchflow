//! What happened in a trial, whose problem it was, and what to do next.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Whose problem a failure is. Every failure has exactly one owner, so the
/// report can say who should act.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum Owner {
    /// The task package: its Dockerfile, verifier, or oracle.
    Task,
    /// The agent or its harness: it failed to install, start, or speak ACP.
    Agent,
    /// The model provider: authentication, rate limits, usage limits.
    Model,
    /// The sandbox backend: the daemon, the cloud API, the network.
    Sandbox,
    /// The person running BenchFlow: a missing login, a bad flag.
    You,
    /// BenchFlow itself: a bug.
    Benchflow,
}

impl Owner {
    pub fn label(self) -> &'static str {
        match self {
            Owner::Task => "the task",
            Owner::Agent => "the agent",
            Owner::Model => "the model provider",
            Owner::Sandbox => "the sandbox",
            Owner::You => "your setup",
            Owner::Benchflow => "BenchFlow (a bug)",
        }
    }

    /// Whose problem, as a phrase for reports: `the task's problem`, `a
    /// BenchFlow bug`.
    pub fn problem(self) -> String {
        match self {
            Owner::Benchflow => "a BenchFlow bug".to_string(),
            other => format!("{}'s problem", other.label()),
        }
    }
}

impl fmt::Display for Owner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The phase of a trial a failure happened in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Phase {
    Image,
    Start,
    Setup,
    Agent,
    Handoff,
    Verify,
    Save,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Image => "preparing the image",
            Phase::Start => "starting the sandbox",
            Phase::Setup => "setting up the sandbox",
            Phase::Agent => "running the agent",
            Phase::Handoff => "handing off to the verifier",
            Phase::Verify => "verifying",
            Phase::Save => "saving results",
        }
    }
}

/// A failure, with its owner and the next step.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Problem {
    /// Stable code, such as `R301`; `bench explain R301` says more.
    pub code: String,
    pub owner: Owner,
    pub phase: Phase,
    /// One line: what happened.
    pub message: String,
    /// What to do next.
    pub next: String,
    /// A log tail or other evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Retrying the same trial may succeed.
    pub transient: bool,
    /// For a sandbox lost (R103) while the agent works, handing off or
    /// verifying: whether the backend confirmed the loss (deleted, stopped
    /// from outside, or its API unavailable). An unconfirmed loss may be
    /// code inside breaking the connection, so a second one in a row is
    /// not run again (`rerun_capped`); written only on such losses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_confirmed: Option<bool>,
    /// The same unexplained loss came back in the attempt after it, so the
    /// trial is not run again: a resumed job keeps it as it is (an errored
    /// trial, counted as not passed) unless `--rerun errored` asks.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rerun_capped: bool,
}

impl Problem {
    pub fn new(
        code: &str,
        owner: Owner,
        phase: Phase,
        message: impl Into<String>,
        next: impl Into<String>,
    ) -> Self {
        Self {
            code: code.to_string(),
            owner,
            phase,
            message: message.into(),
            next: next.into(),
            detail: None,
            transient: false,
            loss_confirmed: None,
            rerun_capped: false,
        }
    }
    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        let d = detail.into();
        if !d.trim().is_empty() {
            self.detail = Some(d);
        }
        self
    }
    pub fn transient(mut self) -> Self {
        self.transient = true;
        self
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

/// How a trial ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// The verifier produced a reward.
    Scored,
    /// The agent ran but the verifier produced no usable reward (a task
    /// problem): the trial says nothing about the agent.
    Unscored,
    /// The trial could not run: the sandbox, the agent's install, a login.
    Errored,
    /// Not run: a budget ran out or the run was cancelled.
    Skipped,
}

/// Whether a defense held for a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum DefenseState {
    /// Applied and confirmed.
    Enforced,
    /// The backend cannot provide it.
    NotSupported,
    /// The task or run turned it off (an agent running as root, say).
    Off,
    /// Tried and failed.
    Failed,
}

/// One defense and its state, recorded per trial.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Defense {
    pub name: String,
    pub state: DefenseState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The defenses a trial records, by name.
pub type Defenses = BTreeMap<String, Defense>;

pub fn defense(defenses: &mut Defenses, name: &str, state: DefenseState, note: Option<String>) {
    defenses.insert(
        name.to_string(),
        Defense {
            name: name.to_string(),
            state,
            note,
        },
    );
}

/// Every runtime failure code, with an explanation for `bench explain`.
pub struct Explanation {
    pub code: &'static str,
    pub title: &'static str,
    pub owner: Owner,
    pub body: &'static str,
}

pub static EXPLANATIONS: &[Explanation] = &[
    Explanation {
        code: "R000",
        title: "not graded, as asked",
        owner: Owner::You,
        body: "`bench shell --no-grade` ended the trial without running the verifier.",
    },
    Explanation {
        code: "R001",
        title: "stopped before it finished",
        owner: Owner::You,
        body: "The run was stopped while this trial was running: a second Ctrl-C (the first only drains the run: no new trial starts and running ones finish), a `kill`, or a closed terminal. Its sandbox was removed. Such trials say nothing about the agent, so the run's summary and `bench results` leave them out of the pass rate and list them apart. Rerun the same command to resume the job: finished trials are kept and this one runs again.",
    },
    Explanation {
        code: "R101",
        title: "the task's image failed to build",
        owner: Owner::Task,
        body: "A step in the task's Dockerfile failed, or the image the task names cannot be pulled. The trial folder's `image.log` has the build output; the last lines usually name the failing command.\n\nFix the Dockerfile (or the image reference), then run `bench test <task>` to rebuild. Pin versions (`pip install pkg==1.2.3`, `apt-get install pkg=1.2-3`) so a later rebuild cannot break the same way. When a model agent's run fails here, BenchFlow first checks that the task's image builds on its own; if it does, the failure is the agent's install (R201).",
    },
    Explanation {
        code: "R102",
        title: "the image could not be prepared",
        owner: Owner::Sandbox,
        body: "The sandbox backend could not build or pull the image for reasons outside the Dockerfile: the daemon, the registry, disk space, or the network. The detail has the backend's message.\n\nRetry; if it persists, run `bench doctor`.",
    },
    Explanation {
        code: "R103",
        title: "the sandbox failed",
        owner: Owner::Sandbox,
        body: "The sandbox did not start, or the backend lost it (or the connection to it) while the trial ran. Starting fails on resources (memory, disk) or a bad entrypoint on Docker, and on quota or capacity on Daytona; later failures are a container that vanished or an API that stopped answering. A sandbox the backend stopped itself is lost too: an archived Daytona sandbox, or one stopped after Daytona saw no activity from it (or BenchFlow's keepalive could not reach it) for 10 minutes; a request Daytona's control-plane API refused (its rate limit, HTTP 429, or this account), not its toolbox, whose answers come from inside the sandbox; a container `docker stop` or `docker kill` ended, as the daemon's own events say. Words a command inside prints never count. A server error (5xx) or a broken connection is not confirmed either: it looks the same as code inside breaking the connection. One that code inside stopped (`kill -TERM 1`) is not: it is the task's while grading and the agent's while the agent works. One stopped for a reason the backend cannot name (a Podman container, a Docker container with no record of dying, a Daytona sandbox in `error`) is lost once.\n\nRetry, or lower `--concurrency`. `bench doctor` checks the backend. Any other loss (the backend cannot name the cause, or the sandbox still runs and the connection to it broke) is rerun once per trial, two minutes later within a run so that a short outage does not catch the retry too: once an earlier attempt of the trial lost its sandbox that way, in any phase, the next such loss is not run again, since code inside the sandbox can break it every time and an outage only the sandbox's own channel shows looks the same. It stays this code and counts as not passed. A sandbox code inside stopped while the agent (or the oracle, or a script) worked is the agent's own failure (R204), not run again either. A resumed job keeps such trials; `bench run --rerun errored` runs them again.",
    },
    Explanation {
        code: "R104",
        title: "the sandbox could not be set up",
        owner: Owner::Task,
        body: "BenchFlow could not create the non-root agent user, prepare the workdir, or give the agent user its files. Images without `useradd` or `adduser` (some distroless images) cannot host a separate agent user. The agent user copies its own login and settings files into its home from a folder root makes under `/var/lib/benchflow-place`: a folder on the way it cannot write (root's, read-only, or a mount), or a `/`, `/var`, or `/var/lib` that users other than root can change, stops that.\n\nAdd a shell and `useradd` to the image, make the agent's home and the folders in it the agent user's, or set `[agent] user = \"root\"` to run the agent as root (the agent-user defense is then off). With `bench agents check --image`, the image is yours to change. In a task with stages, it is also given when a later stage's files or mounts cannot be put in place: the agent user copies them in, and a copy that fails under a path the task itself placed (its mounts) or a mount point is the task's; and a stage that mounts where an earlier mount or the image left a folder the agent user cannot write, while an earlier ungraded stage's agent still runs, is refused before any copy, since neither BenchFlow nor the agent user can put the files there safely. A fault found after an earlier stage's agent worked leaves the trial unscored, not run again.",
    },
    Explanation {
        code: "R105",
        title: "the readiness check never passed",
        owner: Owner::Task,
        body: "The task's `[sandbox] ready` command did not succeed within its retries, so the agent was never started.\n\nTry the command by hand in `bench shell <task>` (it needs Docker or Podman). If the service is only slow to start, give it a `start_period` or more `retries`: `ready = { run = \"...\", start_period = \"30s\", retries = 10 }`.",
    },
    Explanation {
        code: "R106",
        title: "a service did not start",
        owner: Owner::Task,
        body: "A service the task declares beside the agent's container (a Compose service, or `[[sandbox.services]]`) exited right after it started, or never passed its readiness check, so the agent was never started.\n\nThe trial's `services/<name>.log` has the service's output. Check its image, command, and healthcheck; services run on a private network with no internet, so a service that downloads at start fails there.",
    },
    Explanation {
        code: "R107",
        title: "a file on this machine could not be read or written",
        owner: Owner::You,
        body: "BenchFlow failed to read or write something on the machine it runs on: a task file it uploads to the sandbox, a file in the job folder, a temporary file, or the container CLI it starts (`docker`, `podman`). The message names the file or program and the operating system's error.\n\nCheck that the file exists and you can read it, that the job folder is writable and the disk has room, and that the container CLI is installed and on your PATH; then run again. Copies out of a sandbox are capped (in bytes, in entries, and in the size of each archive header), so what code inside sends cannot fill this machine on its own; but on Docker, a sandbox with no `[sandbox] disk` limit writes to the daemon's disk, and when that is this machine's disk, a sandbox that fills it during a copy shows as this code.",
    },
    Explanation {
        code: "R108",
        title: "the network policy could not be enforced",
        owner: Owner::Task,
        body: "The task limits the network, and BenchFlow could not put the limit in place, so the phase did not run rather than run unprotected. The limit needs BenchFlow's firewall for none in one phase, a host allowlist, or a blocklist; on Daytona, whose own network block leaves IPv6 open, for every none and allowlist too. The message says why it could not: the image has no iptables and none could be installed, root in the sandbox cannot change packet filters, the sandbox has IPv6 but no ip6tables, a listed host does not resolve, or the agent runs as root (root could remove the firewall).\n\nWhose problem it is depends on the cause, and the trial's problem names it: the task's when Docker would refuse it too (fix the Dockerfile, the host name, or `[agent] user = \"root\"`); yours when Docker would run it (only Daytona's IPv6 gap needs the firewall, or your `sandbox-user = \"root\"` setting made the agent root): run it with `--sandbox docker`, add iptables and ip6tables to the image, or set a non-root sandbox user; the sandbox's when root there cannot change packet filters; BenchFlow's when a rule it wrote failed (report it with the trial folder).",
    },
    Explanation {
        code: "R109",
        title: "the network self-test failed",
        owner: Owner::Benchflow,
        body: "Before the agent (and again before the verifier) runs, BenchFlow tests the network policy as that phase's user: an allowed host must answer, a destination outside the policy must not, an unlisted name must not resolve, and the agent user must fail to change the firewall. One of these did not hold, so the phase did not run. The detail shows each canary.\n\nIf an allowed host did not answer even for root, the host or the backend's network is at fault: retry. If a blocked destination answered, report it with the trial folder. If the image has no bash or python3 to run the canaries with, add bash to the task's Dockerfile.",
    },
    Explanation {
        code: "R110",
        title: "the sandbox passed its disk limit",
        owner: Owner::Sandbox,
        body: "The task sets `[sandbox] disk`, and from the agent's first turn to the end of grading BenchFlow measures the sandbox's writable layer from outside it, as the daemon sees it. Where the daemon cannot cap the disk (Docker on overlay2 over ext4, Docker Desktop, the containerd image store) it measures the bytes (a sparse file at its full size) and the files, folders and links the layer added, one 4 KiB block each, and charges whichever is more; where the daemon caps it, the cap holds the bytes, and BenchFlow counts only the added entries, 2 KiB each, whose inodes the cap does not hold. A measure that succeeded put the layer past the limit, so BenchFlow stopped the whole sandbox from outside, so that one trial cannot fill a disk other trials share. The trial was not graded.\n\nThe same code is given when the layer could not be measured: before the agent's first turn (nothing would have held the limit, so the agent did not run), or for longer than the trial's bound (30 s for the bytes and 135 s for the entries on a healthy daemon; longer on a daemon slow before the agent's first turn, in proportion to how long that first measure took) while the daemon said the sandbox still ran. The message says so. A sandbox the daemon no longer has is never stopped this way.\n\nNothing shows the agent filled the layer: the measure that stopped it began outside the agent's turns (BenchFlow's own setup or grading, the verifier's own writes, a root agent's process still writing after its turn), or the agent's turns added less than half the limit (an earlier stage's verifier, a large workdir), or the stop itself failed. So this is the sandbox's error: it is run again once, and an attempt after an earlier unexplained failure of the same trial is kept as it is (`--rerun errored` runs it again). When the agent's turns did add at least half, the trial is the agent's R316 instead. If the task's work, its verifier included, needs more disk, raise its `[sandbox] disk`. The trial's `disk_limit` defense says how the limit held.",
    },
    Explanation {
        code: "R201",
        title: "the agent could not be installed",
        owner: Owner::Agent,
        body: "The agent's install step failed while building its image layer on top of the task's image (which builds on its own). It needs network access to npm or the agent's download site, and a few base tools (curl, tar, xz).\n\nThe trial's `image.log` has the output. `bench agents <name>` shows the install script, and `bench agents check <name> -s <backend>` tries it on a plain image.",
    },
    Explanation {
        code: "R202",
        title: "the agent did not start or did not speak ACP",
        owner: Owner::Agent,
        body: "The agent process exited, or did not answer the Agent Client Protocol handshake in time.\n\nThe trial's `agent/stderr.txt` has the agent's own output.",
    },
    Explanation {
        code: "R203",
        title: "the agent stopped responding",
        owner: Owner::Agent,
        body: "The agent sent nothing for longer than the idle timeout while a prompt was running. A hung tool call or a stalled model connection are the usual causes.\n\nRaise `--idle-timeout` if the task has legitimately long silent steps.",
    },
    Explanation {
        code: "R204",
        title: "the agent crashed",
        owner: Owner::Agent,
        body: "The agent process exited in the middle of a prompt. `agent/stderr.txt` in the trial folder has its last words.\n\nWhen the connection to the agent broke instead (the sandbox's side), the report says it is the sandbox's problem, and the trial is retried when retries are on.",
    },
    Explanation {
        code: "R205",
        title: "the agent failed its turn",
        owner: Owner::Agent,
        body: "The agent answered a request with an error of its own, or broke the Agent Client Protocol (a malformed answer). The message is the agent's. In a task with stages, it is also given when what the agent did in an earlier stage stopped BenchFlow from putting the next stage's files in place: the agent's own files in its home, the stage's files, which the agent user copies into the workdir, or, after an ungraded stage, a mount under a folder the agent can write, which it copies in too (a file where a folder belongs, a folder made read-only, a pipe where a file goes). A path the task itself placed, or a mount, is the task's instead (R104).\n\nThe trial's `agent/` folder has the agent's log. If it names a model or a setting, check `bench agents <name>`.",
    },
    Explanation {
        code: "R206",
        title: "the agent did nothing before its time ran out",
        owner: Owner::Agent,
        body: "The agent's wall-clock budget ran out before it made a tool call, wrote a message or a thought, or produced an output token. That is not a score of 0: the agent most likely never reached its model (a wrong base URL, a login prompt, a blocked network). The trial is not graded, as in BenchFlow 0.8.\n\nRead `agent/<agent>.txt` in the trial folder, then try the agent on a small task with `bench run -a <agent> --tasks <one task>`.",
    },
    Explanation {
        code: "R211",
        title: "no login for this agent",
        owner: Owner::You,
        body: "The agent needs credentials for its model provider and none were found on this machine, or its ACP server refused to start a session until it is signed in with its vendor's own account (`Authentication required`), which no key BenchFlow passes can satisfy; run such an agent headless, or pick another.\n\n`bench agents <name>` lists what it accepts, in order. For Claude Code: `claude setup-token`, then `export CLAUDE_CODE_OAUTH_TOKEN=<token>`. For Codex: `codex login`.",
    },
    Explanation {
        code: "R212",
        title: "the model provider rejected the login",
        owner: Owner::Model,
        body: "The provider answered with an authentication error: the key is wrong, revoked, or for another provider.\n\nCheck the key with a direct request to the provider before running again.\n\nA 403 whose words say the plan or tier lacks the model (\"Free tier users do not have access to this model\") is different: the login works, and the message says so. Upgrade the plan, use a key on a plan that has the model, or pick another model.",
    },
    Explanation {
        code: "R213",
        title: "the login's usage limit was reached",
        owner: Owner::Model,
        body: "The subscription or key ran out of quota. BenchFlow stops starting new trials on this login; the trials not run are listed in the summary.\n\nRerun the same command after the reset to resume the job (finished trials are kept), or switch to another login first. `bench doctor` shows each login.",
    },
    Explanation {
        code: "R214",
        title: "the model provider failed",
        owner: Owner::Model,
        body: "The model provider rate-limited the request, was unavailable, or rejected it, or BenchFlow could not list the upstream providers a gateway would route the model to (it sends nothing it cannot keep off Baseten). The run retries the trial when retries are on, except for a refused request (an HTTP 400 that is not about the context window), which a retry would not change. A prompt over the context window is the agent's (R205) for ACP and headless agents; BenchFlow's own native loops report it here, not retried.\n\nIf it keeps failing, check the provider's status page, or lower `--concurrency` to stay under its rate limit.",
    },
    Explanation {
        code: "R215",
        title: "the agent cannot do what the run asked",
        owner: Owner::You,
        body: "The agent does not offer the requested model, or lacks a setting the run asked for (a reasoning effort, an MCP transport, a provider's endpoint), or no allowed upstream serves the model on a gateway.\n\n`bench agents <name>` shows how the agent selects models and which providers it can use; the message lists the models the agent offered. When the agent would send its requests to a router itself (a router's key it reads directly), the report says it is the agent's problem: BenchFlow cannot keep such requests off Baseten.",
    },
    Explanation {
        code: "R301",
        title: "the verifier wrote no reward",
        owner: Owner::Task,
        body: "The verifier finished but `/logs/verifier` holds neither `reward.json` nor `reward.txt`. (A task with a rubric is different: with no `ctrf.json`, every test criterion fails and the trial is graded, as the task.md format says.)\n\n`verifier/test-stdout.txt` in the trial folder has the verifier's output. A test.sh that exits early (set -e on a failing install) is the usual cause: write the reward on every path, even when tests fail.\n\nThe same code is given when `/logs/verifier` could not be copied out of the sandbox, twice, because of the folder: more than 1 GiB of file content in it, more than 500,000 entries (each header counts, long-name, PAX and refused ones included, and each folder counts again when it is created; up to 100,000 files, links and folders in all always fit), a long-name or PAX header over 64 KiB, files `tar` refuses, names this machine cannot store, a missing or broken `tar` (written by the verifier, or by the agent's code it ran); or because code inside stopped the sandbox. The trial is not graded from a partial copy, and not run again, since the agent's code may have caused it. A copy that fails on this machine (its own disk, quota or permissions, or its job folder gone) is R107, and one whose connection to a running sandbox breaks is R103; both are run again. The message has the error.",
    },
    Explanation {
        code: "R302",
        title: "the reward file could not be read",
        owner: Owner::Task,
        body: "`reward.txt` must hold one number; `reward.json` must be an object of finite numbers (or hold a numeric `reward`). Empty files, NaN, and text are refused.",
    },
    Explanation {
        code: "R303",
        title: "the verifier timed out",
        owner: Owner::Task,
        body: "The verifier ran longer than `[verifier] timeout` and was killed. Install test dependencies in the image rather than in test.sh, and raise the timeout if the tests are genuinely slow.",
    },
    Explanation {
        code: "R304",
        title: "the verifier failed to run",
        owner: Owner::Task,
        body: "test.sh exited with an error and wrote no reward. A non-zero exit after writing a reward is fine; without one, the trial is unscored.",
    },
    Explanation {
        code: "R305",
        title: "hardening before grading failed",
        owner: Owner::Sandbox,
        body: "BenchFlow could not confirm that every agent process was gone before grading (or could not start the stop at all), the check of its root-only state folder failed, a test-runner file the agent planted or changed could not be removed or put back, or the handoff before grading ran out of time, so it did not grade. Processes stuck in uninterruptible I/O, a full disk, or a process table the host ran out of are the usual causes; very many files in the workdir or the temp folders (often ones the agent left) of the last. The same code is given when the verifier, or a later stage's setup, failed on a full disk, or the verifier scored a trial that did not pass while printing \"No space left on device\", and nothing shows the agent filled the disk. A root agent's own `df` cannot show the disk has room: for one, only the daemon's view (a measure of the layer under a cap, a probe of the daemon's disk) can.\n\nThe agent may have caused it, but nothing the agent cannot fake shows that it did (when the sandbox's own process limit shows it did, the trial is R316 instead; a full disk at grading never is), so it is the sandbox's: the trial is run again once, and an attempt after an earlier failure of the sandbox that the backend could not explain is kept as it is, never scored (`--rerun errored` runs it again). If it repeats for one task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder.",
    },
    Explanation {
        code: "R306",
        title: "the task's reference solution or control is missing",
        owner: Owner::Task,
        body: "`--agent oracle` runs the task's reference solution, `oracle/solve.sh` (or `solution/solve.sh` in the split layout), and the task has none; or `bench test` runs a negative control from `controls/` that cannot be read.\n\nAdd the reference solution, or fix the control file the message names.",
    },
    Explanation {
        code: "R307",
        title: "a hardening program did not report back",
        owner: Owner::Task,
        body: "BenchFlow's programs before grading (the kill of the agent's processes, the handoff, the trusted PATH check), and the one that gives the workdir back to the agent user before a later stage, print a line when they finish. One exited without it, so something on the image's PATH answered in place of `sh`, `bash`, or a tool they use: usually an image that lets everyone write a folder like /usr/local/bin, where the agent replaced the shell. No defense of that trial can be trusted, so it is not graded.\n\nMake every folder on the image's PATH root-owned and not writable by group or others. The same code is given when the image has no `find`, which the handoff needs to check BenchFlow's state folder and clean up after the agent (when the agent ran as root or could become root, the handoff goes on without it and records `special_file_custody` failed): add `find` to the image.",
    },
    Explanation {
        code: "R309",
        title: "/etc/ld.so.preload could not be restored",
        owner: Owner::Sandbox,
        body: "Every dynamically linked program loads the libraries `/etc/ld.so.preload` names, the verifier included. BenchFlow snapshots that file before the agent's turn and puts it back before grading; this time it could not, so the trial was not graded.\n\nRetry the trial; if it repeats for one task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder.",
    },
    Explanation {
        code: "R310",
        title: "the verifier's environment could not be made clean",
        owner: Owner::Task,
        body: "The verifier runs with loader variables (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `BASH_ENV`, ...) unset, `PYTHONSAFEPATH=1`, and a `PATH` of folders only root can write. BenchFlow checks that environment, as the verifier's own user, right before grading, and refuses to grade when it does not hold. The message says which part failed.\n\nThe usual cause is an image whose `PATH` includes a folder others can write: make it root-owned and `chmod go-w` it in the Dockerfile, or set `PATH` in `[verifier] env`. A task can set a loader variable or `PYTHONSAFEPATH` (empty turns it off) in `[verifier] env` when its verifier needs it.",
    },
    Explanation {
        code: "R311",
        title: "a hook before grading failed",
        owner: Owner::Task,
        body: "A command the task runs after the agent's turn and before grading (task.toml's `[[verifier.collect]]`, task.md's `[[verifier.snapshot]]`) exited with an error or ran past its timeout, so the state it saves is missing and the trial was not graded.\n\nThe trial's `hooks/<n>.txt` has the command and its output.",
    },
    Explanation {
        code: "R312",
        title: "the rubric's model judge could not decide a criterion",
        owner: Owner::Model,
        body: "Rubric criteria with `judge: \"llm\"` are judged on the host after the agent's turn, one model call per criterion. A criterion whose calls failed, or whose replies were not the verdict asked for after three tries, is not judged, and a rubric with an undecided criterion has no reward: the trial is unscored rather than given a 0 it did not earn.\n\nThe trial's `verifier/judgments.jsonl` has every call, its reply, and why it could not be used. Run again, or pick another judge with `bench run --judge-model <provider>/<model>`.",
    },
    Explanation {
        code: "R313",
        title: "the rubric's judge settings cannot be used",
        owner: Owner::Task,
        body: "A rubric's `x-judge` names the judge model as `{\"model\": \"<provider>/<model>\"}`, and its `x-reference` and each criterion's `reference` are text or a file in a private folder of the package (`verifier/reference.md`, `verifier/answers.toml#total`). One of them could not be used: the message says which.\n\n`bench check` reports the same before a run.",
    },
    Explanation {
        code: "R314",
        title: "the agent changed BenchFlow's root-only state folder",
        owner: Owner::Agent,
        body: "BenchFlow keeps its own state in a folder only root may write (`/var/lib/benchflow`), including the baseline the handoff before grading compares the test-runner files against and restores them from. The handoff first checks that everything there is still root's alone: owned by root, not a link, not writable by others. When something is not, a baseline the agent could have chosen would decide what root restores, so the trial is not graded.\n\nThis applies when the agent ran as a user without a way to become root. When the agent ran as root or could become root (the image gives it `sudo`, say), or the reference solution ran as root, the change is noted in the trial's `state_custody` defense instead and grading goes on: such a run could have changed the folder without a trace.\n\nRead the agent's trajectory for what it did to the folder; if nothing in it explains the change, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder.",
    },
    Explanation {
        code: "R315",
        title: "the workdir could not be given to its owner",
        owner: Owner::Task,
        body: "Before grading, BenchFlow gives the workdir and everything in it to root (or the task's verifier user), as BenchFlow 0.8 does, so the verifier does not run against files the agent owns; after a graded stage it gives the workdir back to the agent user. This time the workdir folder itself could not change owner, so the trial was not graded rather than graded against the agent's ownership: the image has no such user (`[verifier] user` names one it lacks), the workdir is `/`, the workdir is on a read-only mount, or the image has no `find` that takes the options BenchFlow uses (`-xdev`, `-mindepth`, `-maxdepth`, `-links`).\n\nA path inside the workdir that cannot change owner (one nested past what the image's `find` can walk, a file `chown` refuses) does not stop the trial: `custody.json` lists it under `workdir_unchanged`. Another filesystem mounted inside the workdir is not walked, as before, and keeps its owner. A change that runs past its time once the workdir itself changed is noted there too; one that runs past it before, or is ended by the sandbox before the workdir itself changed, is the sandbox's (R103, retried); one that exits cleanly without its report is R307. A walk that runs past its time, or is ended, once the workdir itself changed is noted, not fatal. A workdir that is a link (the image's `/app -> /srv/app`) is walked where it leads when the folder holding the link is root's alone; otherwise only the link changes, and that is noted.",
    },
    Explanation {
        code: "R316",
        title: "the agent exhausted its own sandbox",
        owner: Owner::Agent,
        body: "During its own turn, the agent used up a resource its sandbox has a limit of its own for, and BenchFlow has evidence of that the agent cannot fake and that belongs to the sandbox itself, so the trial scores 0, counts in every score, and is not run again on resume (only `--rerun failed` runs it again). A rerun would reward sabotage. The evidence is one of:\n\n- disk, where the daemon cannot cap it: a measure of the writable layer that began during one of the agent's turns passed the task's `[sandbox] disk`, and the agent's turns added at least half the limit;\n- disk, where the daemon caps it: a count of the files, folders and links the layer added, begun during one of the agent's turns, passed the cap, and the agent's turns added at least half of it (a full capped disk is never this code: inside the sandbox it cannot be told from the daemon's own disk filling, so it is R305);\n- processes: the stop of the agent's processes, at the end of its turn or before grading, could not start, and the daemon counts the sandbox at its own process limit;\n- memory: the agent exited, and the sandbox's own cgroup says its own memory limit ran out during the agent's turn at least as many times as the out-of-memory killer ended a process in it.\n\nWithout such evidence (the host's disk, memory or process table ran out, or a backend such as Daytona that reports none of these), the same failure is the sandbox's (R305, R110, R204), run again once and then kept. The reference solution, the controls and BenchFlow's probes are never given this code. If the agent's honest work needs more, raise the task's `[sandbox] disk` or `memory`.",
    },
    Explanation {
        code: "R401",
        title: "the task asks for something this runner cannot do yet",
        owner: Owner::You,
        body: "The task declares a field the Rust runner reads but cannot honor yet, or asks for something the chosen sandbox backend cannot provide (GPUs on Daytona). Rather than skip it silently, the trial is refused before it starts. The message names the field and why.\n\nRun the task on a backend that supports it (`--sandbox docker`), with BenchFlow 0.8 (`bench eval run`), or remove the field.",
    },
    Explanation {
        code: "R901",
        title: "an internal error",
        owner: Owner::Benchflow,
        body: "Something went wrong inside BenchFlow: a bug stopped the trial (a panic), or one of BenchFlow's own programs in the sandbox failed (the provider forwarder). The message has the details; please report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder.",
    },
];

/// Look up a runtime failure code.
pub fn explain(code: &str) -> Option<&'static Explanation> {
    EXPLANATIONS
        .iter()
        .find(|e| e.code.eq_ignore_ascii_case(code))
}
