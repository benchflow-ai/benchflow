//! Network policies: what a task's agent and its verifier may reach,
//! enforced and self-tested before either one runs.
//!
//! A task's network is `open`, `none`, an allowlist of hosts and CIDR
//! ranges, or a blocklist, for the sandbox's whole life, with the agent's
//! and the verifier's phases allowed to narrow it. Names resolve once, on
//! this machine, when the trial's sandbox starts.
//!
//! | Policy | Docker | Daytona |
//! |---|---|---|
//! | `none` in every phase | `--network none` | Daytona blocks all traffic |
//! | `none` in one phase | uid firewall for that phase | Daytona blocks all traffic during that phase |
//! | allowlist | uid firewall: only the listed ranges; DNS refused, names pinned in `/etc/hosts` | Daytona's own allow list (at most 10 ranges) during that phase, plus the uid firewall |
//! | blocklist | uid firewall: the listed ranges refused, names pinned to an unroutable address | uid firewall only (Daytona has no deny list) |
//!
//! The uid firewall is an iptables chain root installs before the agent
//! starts. In the agent's phase it covers every packet of the agent user and
//! every new connection of any other non-root user, so the agent cannot get
//! around it by switching to another uid, while root (BenchFlow's own
//! commands, a key forwarder on loopback) keeps its access. In the
//! verifier's phase it covers every new connection, root's included. The
//! agent user has no capabilities, so it cannot change the rules; the
//! self-test checks that too.
//!
//! On Daytona the sandbox's own daemon (process 1, root) answers on ports
//! 2280, 22220, 22222, and 33333 without credentials and runs commands as
//! root. Every agent sandbox there gets the chain, open network included:
//! in the agent's phase it refuses the ports the daemon listens on (read
//! from `/proc` as root when the chain is installed, with those four as a
//! floor) on loopback (every local address, IPv4 and IPv6) to the agent
//! user and other non-root users, and nothing else when the network is
//! open. It is recorded as enforced only when, for every port the daemon
//! listens on, root reaches it at an address where the agent user was
//! refused: root is outside the chain, so that contrast is the rule's doing. The verifier's phase replaces or
//! removes it. Where iptables cannot be installed, the trial still runs and
//! records `root_service` as not supported.
//!
//! What it cannot do: wildcards (`*.example.com`) are refused, because an
//! address firewall cannot tell which names a wildcard covers. An allowlist
//! allows addresses, so a host served from a shared CDN address also opens
//! every other site on that address.
//! A blocklist blocks the addresses its names resolved to at start; the
//! same service on another address is not blocked.

mod rules;
mod selftest;

use std::net::IpAddr;

#[cfg(test)]
use benchflow_sandbox::SandboxError;
use benchflow_sandbox::{Capabilities, Exec, NetworkPolicy, Sandbox};
use benchflow_task::TaskPackage;
use benchflow_task::package::Isolation;
use benchflow_task::values::Network;

use crate::hardening::Report;
use crate::outcome::{DefenseState, Owner, Phase, Problem};

pub use rules::Scope;

/// The longest the firewall install, the self-test, or the tamper check may
/// take.
const STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// A phase of the trial with a network of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Agent,
    Verifier,
}

impl Stage {
    /// The defense name a stage records.
    pub fn defense(self) -> &'static str {
        match self {
            Stage::Agent => "network_policy",
            Stage::Verifier => "verifier_network_policy",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Stage::Agent => "agent",
            Stage::Verifier => "verifier",
        }
    }
}

/// Which sandbox a policy is for: the agent's (which a shared verifier
/// grades in too), or a separate verifier's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    Agent,
    SeparateVerifier,
}

/// One phase's network, with its names resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseNet {
    pub network: Network,
    /// Each host name in the list and the addresses it resolved to.
    pub names: Vec<(String, Vec<IpAddr>)>,
    /// Every range the rules allow or block: resolved addresses as `/32` or
    /// `/128`, then the addresses and ranges the task wrote.
    pub ranges: Vec<String>,
}

impl PhaseNet {
    /// A phase with nothing to resolve.
    fn plain(network: Network) -> Self {
        Self {
            network,
            names: Vec::new(),
            ranges: Vec::new(),
        }
    }

    /// Resolve the names in an allowlist or blocklist, on this machine.
    pub async fn resolve(network: &Network) -> Result<Self, Problem> {
        let entries = match network {
            Network::Allow(h) | Network::Block(h) => h.clone(),
            _ => return Ok(Self::plain(network.clone())),
        };
        let blocking = matches!(network, Network::Block(_));
        let mut out = Self::plain(network.clone());
        let mut written: Vec<String> = Vec::new();
        for entry in &entries {
            let entry = entry.trim().trim_end_matches('.').to_lowercase();
            if let Some(range) = range_of(&entry) {
                written.push(range);
                continue;
            }
            let addrs: Vec<IpAddr> = match tokio::net::lookup_host((entry.as_str(), 443)).await {
                Ok(found) => {
                    let mut v: Vec<IpAddr> = found.map(|a| a.ip()).collect();
                    v.sort();
                    v.dedup();
                    v
                }
                Err(_) => Vec::new(),
            };
            if addrs.is_empty() && !blocking {
                return Err(Problem::new(
                    "R108",
                    Owner::Task,
                    Phase::Start,
                    format!(
                        "`{entry}` in the network allowlist does not resolve from this machine"
                    ),
                    "check the spelling of the host; if it is right, check this machine's DNS",
                ));
            }
            for a in &addrs {
                let r = host_range(*a);
                if !out.ranges.contains(&r) {
                    out.ranges.push(r);
                }
            }
            out.names.push((entry, addrs));
        }
        for r in written {
            if !out.ranges.contains(&r) {
                out.ranges.push(r);
            }
        }
        Ok(out)
    }

    fn v4_ranges(&self) -> Vec<String> {
        self.ranges
            .iter()
            .filter(|r| !r.contains(':'))
            .cloned()
            .collect()
    }

    /// The network in one line, with what the names resolved to.
    pub fn describe(&self) -> String {
        match &self.network {
            Network::Open => "open".into(),
            Network::None => "none".into(),
            Network::Allow(_) | Network::Block(_) => {
                let what = if matches!(self.network, Network::Allow(_)) {
                    "only"
                } else {
                    "open except"
                };
                let mut parts: Vec<String> = self
                    .names
                    .iter()
                    .map(|(n, a)| {
                        if a.is_empty() {
                            format!("{n} (did not resolve)")
                        } else {
                            let list: Vec<String> = a.iter().map(|x| x.to_string()).collect();
                            format!("{n} ({})", list.join(", "))
                        }
                    })
                    .collect();
                let named: Vec<String> = self
                    .names
                    .iter()
                    .flat_map(|(_, a)| a.iter().map(|x| host_range(*x)))
                    .collect();
                parts.extend(self.ranges.iter().filter(|r| !named.contains(r)).cloned());
                format!("{what} {}", parts.join(", "))
            }
        }
    }
}

/// An IP address or CIDR range as a range, or `None` for a host name.
fn range_of(entry: &str) -> Option<String> {
    if let Some((addr, prefix)) = entry.split_once('/') {
        let addr: IpAddr = addr.parse().ok()?;
        let prefix: u8 = prefix.parse().ok()?;
        return Some(format!("{addr}/{prefix}"));
    }
    entry.parse::<IpAddr>().ok().map(host_range)
}

fn host_range(a: IpAddr) -> String {
    match a {
        IpAddr::V4(_) => format!("{a}/32"),
        IpAddr::V6(_) => format!("{a}/128"),
    }
}

/// Whether `range` (a CIDR) contains `addr`.
fn contains(range: &str, addr: IpAddr) -> bool {
    let Some((net, prefix)) = range.split_once('/') else {
        return false;
    };
    let (Ok(net), Ok(prefix)) = (net.parse::<IpAddr>(), prefix.parse::<u32>()) else {
        return false;
    };
    match (net, addr) {
        (IpAddr::V4(n), IpAddr::V4(a)) if prefix <= 32 => {
            let mask = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - prefix)
            };
            u32::from(n) & mask == u32::from(a) & mask
        }
        (IpAddr::V6(n), IpAddr::V6(a)) if prefix <= 128 => {
            let mask = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - prefix)
            };
            u128::from(n) & mask == u128::from(a) & mask
        }
        _ => false,
    }
}

/// A trial's networks, resolved once when its first sandbox starts.
#[derive(Debug, Clone)]
pub struct Plan {
    pub agent: PhaseNet,
    pub verifier: PhaseNet,
    /// The verifier grades in a sandbox of its own.
    pub separate: bool,
}

impl Plan {
    pub async fn resolve(task: &TaskPackage) -> Result<Self, Problem> {
        let agent = PhaseNet::resolve(task.agent_network()).await?;
        let verifier = if task.verifier_network() == task.agent_network() {
            agent.clone()
        } else {
            PhaseNet::resolve(task.verifier_network()).await?
        };
        Ok(Self {
            agent,
            verifier,
            separate: task.verifier.isolation == Isolation::Separate,
        })
    }

    pub fn phase(&self, stage: Stage) -> &PhaseNet {
        match stage {
            Stage::Agent => &self.agent,
            Stage::Verifier => &self.verifier,
        }
    }

    /// The phases a sandbox hosts.
    fn stages(&self, host: Host) -> Vec<Stage> {
        match host {
            Host::Agent if self.separate => vec![Stage::Agent],
            Host::Agent => vec![Stage::Agent, Stage::Verifier],
            Host::SeparateVerifier => vec![Stage::Verifier],
        }
    }

    /// What the backend enforces itself during `stage`, on a backend that
    /// can switch a running sandbox's network.
    fn switched(&self, stage: Stage, caps: &Capabilities) -> NetworkPolicy {
        let p = self.phase(stage);
        match &p.network {
            Network::Open => NetworkPolicy::Open,
            Network::None if caps.network_none => NetworkPolicy::None,
            Network::Allow(_) => {
                let v4 = p.v4_ranges();
                match caps.network_allow_cidrs {
                    Some(_) if v4.is_empty() => NetworkPolicy::None,
                    Some(max) if v4.len() <= max => NetworkPolicy::Filtered { allow: Some(v4) },
                    _ => NetworkPolicy::Filtered { allow: None },
                }
            }
            _ => NetworkPolicy::Filtered { allow: None },
        }
    }

    /// What a sandbox starts with.
    pub fn start_policy(&self, host: Host, caps: &Capabilities) -> NetworkPolicy {
        let stages = self.stages(host);
        if caps.network_switch {
            return self.switched(stages[0], caps);
        }
        let nets: Vec<&Network> = stages.iter().map(|s| &self.phase(*s).network).collect();
        if caps.network_none && nets.iter().all(|n| **n == Network::None) {
            NetworkPolicy::None
        } else if nets.iter().all(|n| **n == Network::Open) {
            NetworkPolicy::Open
        } else {
            NetworkPolicy::Filtered { allow: None }
        }
    }

    /// What the backend enforces during `stage`.
    fn native(&self, stage: Stage, host: Host, caps: &Capabilities) -> NetworkPolicy {
        if caps.network_switch {
            self.switched(stage, caps)
        } else {
            self.start_policy(host, caps)
        }
    }

    /// Whether `stage` needs the uid firewall, and whether it must have it
    /// (nothing else would enforce the policy).
    fn firewall(&self, stage: Stage, host: Host, caps: &Capabilities) -> (bool, bool) {
        let native = self.native(stage, host, caps);
        match &self.phase(stage).network {
            Network::Open => (false, false),
            Network::None => {
                let need = native != NetworkPolicy::None;
                (need, need)
            }
            Network::Allow(_) => {
                let covered = matches!(native, NetworkPolicy::Filtered { allow: Some(_) })
                    || native == NetworkPolicy::None;
                (native != NetworkPolicy::None, !covered)
            }
            Network::Block(_) => (true, true),
        }
    }
}

/// An image layer that installs iptables when the image lacks it, for a
/// sandbox whose policy may need the uid firewall, and for every agent
/// sandbox on a backend with a root service (`root_service`), which the
/// daemon guard needs. The layer never fails: an image without a package
/// manager is refused later, by name, if the firewall is required, and
/// records the guard as not supported otherwise.
pub fn firewall_layer(task: &TaskPackage, host: Host, root_service: bool) -> Option<String> {
    if root_service && host == Host::Agent {
        return Some(rules::INSTALL_LAYER.to_string());
    }
    let nets: Vec<&Network> = match host {
        Host::Agent if task.verifier.isolation == Isolation::Separate => {
            vec![task.agent_network()]
        }
        Host::Agent => vec![task.agent_network(), task.verifier_network()],
        Host::SeparateVerifier => vec![task.verifier_network()],
    };
    let filtered = nets
        .iter()
        .any(|n| matches!(n, Network::Allow(_) | Network::Block(_)));
    let mixed =
        nets.iter().any(|n| **n == Network::None) && nets.iter().any(|n| **n != Network::None);
    (filtered || mixed).then(|| rules::INSTALL_LAYER.to_string())
}

/// Fields of a task's network this runner refuses, for the trial's refusal.
pub fn refusals(task: &TaskPackage, model_needs_network: bool) -> Vec<String> {
    let mut out = Vec::new();
    for (who, net) in [
        ("agent", task.agent_network()),
        ("verifier", task.verifier_network()),
    ] {
        if let Network::Allow(h) | Network::Block(h) = net {
            let wild: Vec<&str> = h
                .iter()
                .map(String::as_str)
                .filter(|e| e.contains('*'))
                .collect();
            if !wild.is_empty() {
                out.push(format!(
                    "the {who}'s network lists {} (an address firewall cannot tell which names a wildcard covers; list each host)",
                    wild.join(", ")
                ));
            }
        }
    }
    if model_needs_network {
        match task.agent_network() {
            Network::None => out.push(
                "network = \"none\" with a model agent (the agent could not reach its model; a model-only route is not in the Rust runner yet)".into(),
            ),
            Network::Allow(_) => out.push(
                "a host allowlist with a model agent (the agent's own traffic to its model would be refused; a model-only route is not in the Rust runner yet)".into(),
            ),
            _ => {}
        }
    }
    out
}

/// What enforcing a stage came to: the defense to record, and the problem
/// that stops the trial, if any.
#[derive(Debug, Default)]
pub struct Outcome {
    pub defense: Option<(DefenseState, String)>,
    pub problem: Option<Problem>,
    /// The `root_service` defense, when the agent's chain guarded the
    /// sandbox daemon's ports and the self-test confirmed it.
    pub root_service: Option<(DefenseState, String)>,
}

impl Outcome {
    fn failed(note: String, problem: Problem) -> Self {
        Self {
            defense: Some((DefenseState::Failed, note)),
            problem: Some(problem),
            root_service: None,
        }
    }
}

/// The ports Daytona's in-sandbox daemon (process 1, root) listens on, on
/// every address, answering without credentials. The guard closes these
/// and any other port the daemon is found listening on.
pub const DAEMON_PORTS: &[u16] = &[2280, 22220, 22222, 33333];

/// Ports in an install report's `BF <tag>` line (`none` for none).
fn report_ports(report: &Report, tag: &str) -> Option<Vec<u16>> {
    report.first(tag).map(|l| {
        let mut v: Vec<u16> = l
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .filter(|p| *p > 0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    })
}

/// The port of an `addr:port` target.
fn port_of(target: &str) -> Option<u16> {
    target.rsplit_once(':').and_then(|(_, p)| p.parse().ok())
}

fn list(ports: &[u16]) -> String {
    ports
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Enforce `stage`'s network in `sb` and self-test it as `user`, the user
/// that stage's commands run as. Called after the agent user exists and
/// before the agent starts, and again before the verifier runs. On a
/// backend with a root service, the agent's stage also guards the daemon's
/// ports, open network included.
pub async fn enforce(
    sb: &dyn Sandbox,
    caps: &Capabilities,
    plan: &Plan,
    stage: Stage,
    host: Host,
    user: &str,
) -> Outcome {
    let mut out = enforce_stage(sb, caps, plan, stage, host, user).await;
    // With an open network only the daemon guard ran: a failure is its own.
    if plan.phase(stage).network == Network::Open
        && let Some(failed) = out.defense.take()
    {
        out.root_service = Some(failed);
    }
    out
}

async fn enforce_stage(
    sb: &dyn Sandbox,
    caps: &Capabilities,
    plan: &Plan,
    stage: Stage,
    host: Host,
    user: &str,
) -> Outcome {
    let net = plan.phase(stage);
    let phase = match stage {
        Stage::Agent => Phase::Setup,
        Stage::Verifier => Phase::Verify,
    };
    let root_user = user == "root" || user == "0";
    let native = plan.native(stage, host, caps);
    // The other phase this sandbox hosts (a multi-stage task alternates).
    let shared = host == Host::Agent && !plan.separate;
    let other = match stage {
        Stage::Agent => Stage::Verifier,
        Stage::Verifier => Stage::Agent,
    };
    // A backend that switches networks gets this stage's own.
    if caps.network_switch
        && shared
        && native != plan.native(other, host, caps)
        && let Err(e) = sb.set_network(&native).await
    {
        return Outcome::failed(
            format!(
                "{}: the backend did not switch the network: {e}",
                stage.label()
            ),
            Problem::new(
                "R108",
                Owner::Sandbox,
                phase,
                format!(
                    "the sandbox's network could not be switched for the {}: {e}",
                    stage.label()
                ),
                "retry; if it repeats, report it with the trial folder",
            )
            .transient(),
        );
    }
    let (want, need) = plan.firewall(stage, host, caps);
    // On a backend whose in-sandbox daemon runs commands as root for anyone
    // who reaches it, the agent's chain refuses the daemon's ports, whatever
    // the task's network: on its own when the policy needs no firewall.
    let guard: &[u16] = if caps.root_service && stage == Stage::Agent && host == Host::Agent {
        DAEMON_PORTS
    } else {
        &[]
    };
    let open = net.network == Network::Open;
    // The other stage's rules in this sandbox give way to this one's.
    let earlier = shared
        && (plan.firewall(other, host, caps).0 || (caps.root_service && other == Stage::Agent));
    if open && (guard.is_empty() || root_user) {
        if earlier {
            let _ = sb
                .exec(Exec::new(rules::remove()).root().timeout(STEP_TIMEOUT))
                .await;
        }
        return Outcome {
            root_service: (!guard.is_empty()).then(|| unguarded("the agent runs as root", true)),
            ..Outcome::default()
        };
    }
    let mut notes: Vec<String> = vec![format!("{}: {}", stage.label(), net.describe())];
    match &native {
        NetworkPolicy::None => {
            notes.push(format!("the {} backend allows no network", sb.backend()))
        }
        NetworkPolicy::Filtered { allow: Some(r) } => notes.push(format!(
            "the {} backend allows only {}",
            sb.backend(),
            r.join(", ")
        )),
        _ => {}
    }
    let scope = match stage {
        Stage::Agent => Scope::AgentUser(user.to_string()),
        Stage::Verifier => Scope::Everyone,
    };
    let mut installed: Option<String> = None;
    // Why the daemon's ports stay open to the agent, if they do.
    let mut open_ports: Option<(String, bool)> = None;
    // The ports the chain refuses, and those the daemon was found on.
    let mut guard_ports: Vec<u16> = guard.to_vec();
    let mut listening: Option<Vec<u16>> = None;
    if stage == Stage::Agent && root_user && (want || !guard.is_empty()) {
        if want && need {
            return Outcome::failed(
                format!(
                    "agent: {} not enforced: the agent runs as root",
                    net.describe()
                ),
                Problem::new(
                    "R108",
                    Owner::Task,
                    phase,
                    "the task runs its agent as root, and a network allowlist, blocklist, or a \"none\" that differs from the verifier's is enforced by a firewall that root could remove",
                    "remove [agent] user = \"root\" so the agent runs as the non-root sandbox user",
                ),
            );
        }
        if want {
            notes.push("no uid firewall: the agent runs as root".into());
        }
        open_ports = Some(("the agent runs as root".into(), true));
    } else if want || !guard.is_empty() {
        // Only the guard when the policy itself needs no firewall.
        let guard_only = PhaseNet::plain(Network::Open);
        let rules_net = if want { net } else { &guard_only };
        let script = rules::install(&scope, rules_net, guard);
        let out = match sb
            .exec(Exec::new(script).root().timeout(STEP_TIMEOUT))
            .await
        {
            Ok(o) => o,
            Err(e) => {
                return Outcome::failed(
                    format!("{}: the firewall install did not run: {e}", stage.label()),
                    crate::trial::sandbox_problem(phase, e),
                );
            }
        };
        let report = Report::parse(&out.stdout);
        match (report.first("fw"), report.first("fw-error")) {
            (Some(how), None) => {
                installed = report.first("fw-sum").map(str::to_string);
                if want {
                    notes.push(rules::describe(&scope, net, how));
                }
                if let Some(gap) = report.first("fw-guard-gap") {
                    open_ports = Some((gap.to_string(), false));
                }
                if !guard.is_empty() {
                    if let Some(p) = report_ports(&report, "fw-guard-ports") {
                        guard_ports.extend(p);
                        guard_ports.sort_unstable();
                        guard_ports.dedup();
                    }
                    listening = report_ports(&report, "fw-guard-found");
                }
            }
            (_, error) => {
                let error = error.unwrap_or("the install script failed").to_string();
                let (owner, msg, next) = rules::explain_error(&error, sb.backend());
                if want && need {
                    return Outcome::failed(
                        format!("{}: no firewall: {error}", stage.label()),
                        Problem::new("R108", owner, phase, msg, next).detail(out.tail(1500)),
                    );
                }
                if want {
                    notes.push(format!("no uid firewall ({error})"));
                }
                if !guard.is_empty() {
                    open_ports = Some((rules::guard_error(&error), false));
                }
            }
        }
    }
    // The self-test, as the stage's own user.
    let guarded: &[u16] = if installed.is_some() && open_ports.is_none() {
        &guard_ports
    } else {
        &[]
    };
    if open && installed.is_none() {
        return Outcome {
            root_service: open_ports.map(|(why, off)| unguarded(&why, off)),
            ..Outcome::default()
        };
    }
    let test = selftest::Test::new(net, installed.is_some(), stage, guarded);
    let ran = match sb
        .exec(Exec::new(test.script()).user(user).timeout(STEP_TIMEOUT))
        .await
    {
        Ok(o) => o,
        Err(e) => {
            return Outcome::failed(
                format!("{}: the self-test did not run: {e}", stage.label()),
                crate::trial::sandbox_problem(phase, e),
            );
        }
    };
    let mut verdict = test.judge(&Report::parse(&ran.stdout));
    // The agent user must not have changed the rules or the pinned names.
    if let Some(sum) = &installed {
        match sb
            .exec(Exec::new(rules::fingerprint()).root().timeout(STEP_TIMEOUT))
            .await
        {
            Ok(o) if Report::parse(&o.stdout).first("fw-sum") == Some(sum.as_str()) => {
                if stage == Stage::Agent {
                    verdict
                        .held
                        .push("the agent user could not change the rules".into());
                }
            }
            Ok(_) => verdict.leaks.push(
                "the rules or the pinned names changed after the agent user tried to change them"
                    .into(),
            ),
            Err(e) => verdict.leaks.push(format!(
                "the rules could not be checked after the self-test: {e}"
            )),
        }
    }
    // An allowed canary that failed: blocked by the firewall, or down?
    if !verdict.unreachable.is_empty() && stage == Stage::Agent && !root_user {
        let probe = selftest::reach_script(&verdict.unreachable);
        if let Ok(o) = sb.exec(Exec::new(probe).root().timeout(STEP_TIMEOUT)).await {
            let root_reached = !Report::parse(&o.stdout).all("reached").is_empty();
            if root_reached {
                verdict.leaks.push(format!(
                    "the firewall refused the allowed {} to the agent user, though root reached it",
                    verdict.unreachable.join(", ")
                ));
                verdict.unreachable.clear();
            }
        }
    }
    notes.push(verdict.summary());
    let note = notes.join("; ");
    if !verdict.leaks.is_empty() {
        let tool = verdict
            .leaks
            .iter()
            .any(|l| l.starts_with(selftest::NO_TOOL));
        // An open network does not stop the trial for want of a test tool:
        // the guard stays, unconfirmed.
        if tool && open {
            return Outcome {
                root_service: Some(unguarded(
                    "the guard on the daemon's ports could not be tested (the image has no bash or python3)",
                    false,
                )),
                ..Outcome::default()
            };
        }
        let problem = if tool {
            Problem::new(
                "R109",
                Owner::Task,
                phase,
                format!(
                    "the network self-test could not run: {}",
                    verdict.leaks.join("; ")
                ),
                "add bash (or python3) to the task's image, so BenchFlow can test the firewall before the agent starts",
            )
        } else {
            Problem::new(
                "R109",
                Owner::Benchflow,
                phase,
                format!(
                    "the network self-test failed, so the {} did not run: {}",
                    stage.label(),
                    verdict.leaks.join("; ")
                ),
                "report it with the trial folder; the phase did not run, so no score was affected",
            )
        };
        return Outcome::failed(note, problem.detail(ran.tail(2000)));
    }
    if !verdict.unreachable.is_empty() {
        return Outcome::failed(
            note,
            Problem::new(
                "R109",
                Owner::Sandbox,
                phase,
                format!(
                    "the allowed {} did not answer from the sandbox, even for root",
                    verdict.unreachable.join(", ")
                ),
                "check that the host is up and reachable from the sandbox backend's network; retry",
            )
            .transient()
            .detail(ran.tail(2000)),
        );
    }
    let root_service = if guarded.is_empty() {
        open_ports
            .filter(|_| !guard.is_empty())
            .map(|(why, off)| unguarded(&why, off))
    } else {
        // Root is outside the agent's chain: an address that answers root
        // and refused the agent user shows the rule did the refusing.
        let root_reached: Vec<String> = match sb
            .exec(
                Exec::new(selftest::service_probe(guarded))
                    .root()
                    .timeout(STEP_TIMEOUT),
            )
            .await
        {
            Ok(o) => selftest::services_reached(&Report::parse(&o.stdout)),
            Err(_) => Vec::new(),
        };
        Some(guard_proof(
            guarded,
            listening.as_deref(),
            &verdict.services_refused,
            &root_reached,
            open,
        ))
    };
    Outcome {
        // An open network has no policy to record; the guard is root_service.
        defense: (!open).then_some((DefenseState::Enforced, note)),
        problem: None,
        root_service,
    }
}

/// The `root_service` record for an installed guard that the self-test
/// passed: enforced only when every port the daemon listens on answered
/// root at an address where the agent user was refused.
fn guard_proof(
    guarded: &[u16],
    listening: Option<&[u16]>,
    agent_refused: &[String],
    root_reached: &[String],
    open: bool,
) -> (DefenseState, String) {
    let closed = format!(
        "the uid firewall refuses the agent user's connections to the sandbox daemon's ports ({}) on every local address{}",
        list(guarded),
        if open { ", and nothing else" } else { "" },
    );
    let listening = match listening {
        Some(l) if !l.is_empty() => l,
        _ => {
            return (
                DefenseState::NotSupported,
                format!(
                    "{closed}, but the ports the daemon listens on could not be read, so nothing shows the rule is what refuses the agent user"
                ),
            );
        }
    };
    let proof = |port: u16| -> Vec<&str> {
        agent_refused
            .iter()
            .filter(|t| port_of(t) == Some(port) && root_reached.contains(t))
            .map(String::as_str)
            .collect()
    };
    let unproven: Vec<u16> = listening
        .iter()
        .copied()
        .filter(|p| proof(*p).is_empty())
        .collect();
    if !unproven.is_empty() {
        return (
            DefenseState::NotSupported,
            format!(
                "{closed}, but the daemon listens on {} and root got no answer there either at an address the agent user tried, so nothing shows the rule is what refuses the agent user",
                list(&unproven)
            ),
        );
    }
    let shown: Vec<&str> = listening.iter().flat_map(|p| proof(*p)).collect();
    let mut note = format!(
        "{closed}; the daemon listens on {}; proof: root reached {} and the agent user was refused at each",
        list(listening),
        shown.join(", ")
    );
    let quiet: Vec<&str> = agent_refused
        .iter()
        .filter(|t| !root_reached.contains(t))
        .map(String::as_str)
        .collect();
    if !quiet.is_empty() {
        note.push_str(&format!(
            "; no answer for root either at {}",
            quiet.join(", ")
        ));
    }
    (DefenseState::Enforced, note)
}

/// The `root_service` record when the guard does not hold: `off` when the
/// task turned it off (an agent running as root).
fn unguarded(why: &str, off: bool) -> (DefenseState, String) {
    let ports: Vec<String> = DAEMON_PORTS.iter().map(u16::to_string).collect();
    (
        if off {
            DefenseState::Off
        } else {
            DefenseState::NotSupported
        },
        format!(
            "{why}, so the agent can reach the sandbox daemon's ports ({}), and the daemon runs commands as root for any user who reaches it",
            ports.join(", ")
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docker() -> Capabilities {
        Capabilities {
            network_none: true,
            firewall: true,
            ..Capabilities::default()
        }
    }

    fn daytona() -> Capabilities {
        Capabilities {
            network_none: true,
            firewall: true,
            network_allow_cidrs: Some(10),
            network_switch: true,
            ..Capabilities::default()
        }
    }

    fn plan(agent: Network, verifier: Network, ranges: &[&str]) -> Plan {
        let r: Vec<String> = ranges.iter().map(|s| s.to_string()).collect();
        let phase = |n: Network| PhaseNet {
            ranges: if matches!(n, Network::Allow(_) | Network::Block(_)) {
                r.clone()
            } else {
                Vec::new()
            },
            network: n,
            names: Vec::new(),
        };
        Plan {
            agent: phase(agent),
            verifier: phase(verifier),
            separate: false,
        }
    }

    #[test]
    fn docker_uses_no_network_only_when_every_phase_has_none() {
        let p = plan(Network::None, Network::None, &[]);
        assert_eq!(p.start_policy(Host::Agent, &docker()), NetworkPolicy::None);
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &docker()),
            (false, false)
        );
        let p = plan(Network::None, Network::Open, &[]);
        assert_eq!(
            p.start_policy(Host::Agent, &docker()),
            NetworkPolicy::Filtered { allow: None }
        );
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &docker()),
            (true, true)
        );
        assert_eq!(
            p.firewall(Stage::Verifier, Host::Agent, &docker()),
            (false, false)
        );
        let p = plan(Network::Open, Network::Open, &[]);
        assert_eq!(p.start_policy(Host::Agent, &docker()), NetworkPolicy::Open);
    }

    #[test]
    fn daytona_enforces_allowlists_and_none_itself_and_switches_per_phase() {
        let allow = Network::Allow(vec!["github.com".into()]);
        let p = plan(
            allow.clone(),
            Network::None,
            &["140.82.114.4/32", "2001:db8::1/128"],
        );
        assert_eq!(
            p.start_policy(Host::Agent, &daytona()),
            NetworkPolicy::Filtered {
                allow: Some(vec!["140.82.114.4/32".into()])
            }
        );
        // Daytona's list holds, so the uid firewall is wanted but not needed.
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &daytona()),
            (true, false)
        );
        assert_eq!(
            p.native(Stage::Verifier, Host::Agent, &daytona()),
            NetworkPolicy::None
        );
        assert_eq!(
            p.firewall(Stage::Verifier, Host::Agent, &daytona()),
            (false, false)
        );
        // Eleven ranges are more than Daytona takes: the firewall must hold.
        let many: Vec<String> = (0..11).map(|i| format!("10.0.{i}.0/24")).collect();
        let many: Vec<&str> = many.iter().map(String::as_str).collect();
        let p = plan(allow, Network::None, &many);
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &daytona()),
            (true, true)
        );
        // A blocklist is the firewall's alone.
        let p = plan(
            Network::Block(vec!["x.org".into()]),
            Network::Open,
            &["1.2.3.4/32"],
        );
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &daytona()),
            (true, true)
        );
        assert_eq!(
            p.start_policy(Host::Agent, &daytona()),
            NetworkPolicy::Filtered { allow: None }
        );
    }

    #[test]
    fn ranges_parse_and_match() {
        assert_eq!(range_of("10.0.0.0/8").as_deref(), Some("10.0.0.0/8"));
        assert_eq!(range_of("1.2.3.4").as_deref(), Some("1.2.3.4/32"));
        assert_eq!(range_of("2001:db8::1").as_deref(), Some("2001:db8::1/128"));
        assert_eq!(range_of("github.com"), None);
        assert!(contains("10.0.0.0/8", "10.9.8.7".parse().unwrap()));
        assert!(!contains("10.0.0.0/8", "11.0.0.1".parse().unwrap()));
        assert!(contains("0.0.0.0/0", "1.1.1.1".parse().unwrap()));
        assert!(contains("2001:db8::/32", "2001:db8::5".parse().unwrap()));
        assert!(!contains("2001:db8::/32", "1.1.1.1".parse().unwrap()));
    }

    /// A sandbox that answers the firewall's scripts as told and records
    /// what ran, as whom.
    struct Fake {
        install: &'static str,
        svc: &'static str,
        /// What root's probe of the daemon's ports gets.
        root_svc: &'static str,
        ran: std::sync::Mutex<Vec<(Option<String>, String)>>,
    }

    impl Fake {
        fn new(install: &'static str, svc: &'static str) -> Self {
            Self {
                install,
                svc,
                root_svc: "reached",
                ran: Default::default(),
            }
        }

        /// The ports a service script tries, with the result for each.
        fn svc_lines(c: &str, result: &str) -> String {
            c.split("bf_try ::1 ")
                .skip(1)
                .filter_map(|r| r.split(|ch: char| !ch.is_ascii_digit()).next())
                .flat_map(|p| {
                    [
                        format!("BF canary svc {result} 127.0.0.1:{p}\n"),
                        format!("BF canary svc {result} [::1]:{p}\n"),
                    ]
                })
                .collect()
        }

        fn ran(&self) -> Vec<(Option<String>, String)> {
            self.ran.lock().unwrap().clone()
        }
    }

    const INSTALLED: &str = "BF fw-guard-found 2280 22220 22222 33333\nBF fw-guard-ports 2280 22220 22222 33333\n\
         BF fw /usr/sbin/iptables v6=local fw6=/usr/sbin/ip6tables reject=REJECT\nBF fw-sum 42\n";

    #[async_trait::async_trait]
    impl Sandbox for Fake {
        fn id(&self) -> &str {
            "fake"
        }
        fn backend(&self) -> &str {
            "daytona"
        }
        async fn exec(&self, exec: Exec) -> Result<benchflow_sandbox::ExecOutput, SandboxError> {
            let c = exec.command.clone();
            self.ran
                .lock()
                .unwrap()
                .push((exec.user.clone(), c.clone()));
            let stdout = if c.contains("jump_new") {
                self.install.to_string()
            } else if c.contains("BF tamper") {
                let mut o = Self::svc_lines(&c, self.svc);
                o.push_str("BF tamper tried\n");
                o
            } else if c.contains("BF canary svc") {
                Self::svc_lines(&c, self.root_svc)
            } else if c.contains("BF reached") {
                "BF reached 127.0.0.1:2280\n".into()
            } else if c.contains("hosts.tmp") {
                String::new()
            } else {
                "BF fw-sum 42\n".into()
            };
            Ok(benchflow_sandbox::ExecOutput {
                exit_code: 0,
                stdout,
                stderr: String::new(),
                timed_out: false,
                duration: std::time::Duration::from_millis(1),
            })
        }
        async fn spawn(&self, _: Exec) -> Result<benchflow_sandbox::Process, SandboxError> {
            Err(SandboxError::Unsupported("spawn".into()))
        }
        async fn upload(&self, _: &std::path::Path, _: &str) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn write_file(&self, _: &str, _: &[u8], _: u32) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn download(&self, _: &str, _: &std::path::Path) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn destroy(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn set_network(&self, _: &NetworkPolicy) -> Result<(), SandboxError> {
            Ok(())
        }
    }

    fn daytona_root_service() -> Capabilities {
        Capabilities {
            root_service: true,
            ..daytona()
        }
    }

    async fn agent_stage(fake: &Fake, network: Network, user: &str) -> Outcome {
        let p = plan(network, Network::Open, &[]);
        enforce(
            fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            user,
        )
        .await
    }

    #[tokio::test]
    async fn an_open_network_on_daytona_gets_the_daemon_guard_alone() {
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
        assert!(out.defense.is_none(), "an open network records no policy");
        let (state, note) = out.root_service.expect("root_service recorded");
        assert_eq!(state, DefenseState::Enforced, "{note}");
        assert!(
            note.contains("proof: root reached 127.0.0.1:2280, [::1]:2280, 127.0.0.1:22220"),
            "{note}"
        );
        assert!(
            note.contains("the agent user was refused at each"),
            "{note}"
        );
        let ran = fake.ran();
        let install = &ran[0].1;
        assert!(install.contains("--dport \"$port\""), "{install}");
        let probe = ran
            .iter()
            .find(|(_, c)| c.contains("BF canary svc") && !c.contains("BF tamper"))
            .expect("root probes the daemon's ports");
        assert_eq!(probe.0.as_deref(), Some("root"));
        assert!(
            !install.contains("--dport 53"),
            "the network stays open: {install}"
        );
        assert!(!install.contains("-j RETURN || exit_rule"), "{install}");
        assert_eq!(
            ran[1].0.as_deref(),
            Some("agent"),
            "the self-test runs as the agent"
        );
    }

    #[tokio::test]
    async fn a_port_the_daemon_is_found_on_is_guarded_and_tested_too() {
        let fake = Fake::new(
            "BF fw-guard-found 2280 8080\nBF fw-guard-ports 2280 8080 22220 22222 33333\n\
             BF fw /usr/sbin/iptables v6=local fw6=/usr/sbin/ip6tables reject=REJECT\nBF fw-sum 42\n",
            "refused",
        );
        let out = agent_stage(&fake, Network::Open, "agent").await;
        let (state, note) = out.root_service.unwrap();
        assert_eq!(state, DefenseState::Enforced, "{note}");
        assert!(
            note.contains("ports (2280, 8080, 22220, 22222, 33333)"),
            "{note}"
        );
        assert!(note.contains("the daemon listens on 2280, 8080"), "{note}");
        assert!(note.contains("127.0.0.1:8080"), "{note}");
        let selftest = &fake.ran()[1].1;
        assert!(selftest.contains("bf_try ::1 8080"), "{selftest}");
    }

    #[tokio::test]
    async fn the_guard_is_enforced_only_with_proof_that_the_rule_refuses() {
        // Root gets no answer either: nothing shows the rule did it.
        let mut fake = Fake::new(INSTALLED, "refused");
        fake.root_svc = "refused";
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert!(out.problem.is_none());
        let (state, note) = out.root_service.unwrap();
        assert_eq!(state, DefenseState::NotSupported, "{note}");
        assert!(
            note.contains("the daemon listens on 2280, 22220, 22222, 33333 and root got no answer there either"),
            "{note}"
        );
        // The daemon's listening ports could not be read.
        let fake = Fake::new(
            "BF fw-guard-found none\nBF fw-guard-ports 2280 22220 22222 33333\n\
             BF fw /usr/sbin/iptables v6=local fw6=/usr/sbin/ip6tables reject=REJECT\nBF fw-sum 42\n",
            "refused",
        );
        let out = agent_stage(&fake, Network::Open, "agent").await;
        let (state, note) = out.root_service.unwrap();
        assert_eq!(state, DefenseState::NotSupported, "{note}");
        assert!(note.contains("could not be read"), "{note}");
    }

    #[test]
    fn the_proof_needs_root_and_the_agent_at_the_same_address() {
        let refused = vec!["127.0.0.1:2280".to_string(), "[::1]:2280".to_string()];
        // Root reached only an address the agent did not try: no proof.
        let (state, _) = guard_proof(
            &[2280],
            Some(&[2280]),
            &refused,
            &["10.0.0.5:2280".into()],
            true,
        );
        assert_eq!(state, DefenseState::NotSupported);
        let (state, note) = guard_proof(
            &[2280],
            Some(&[2280]),
            &refused,
            &["[::1]:2280".into()],
            true,
        );
        assert_eq!(state, DefenseState::Enforced);
        assert!(note.contains("proof: root reached [::1]:2280"), "{note}");
        assert!(
            note.contains("no answer for root either at 127.0.0.1:2280"),
            "{note}"
        );
    }

    #[tokio::test]
    async fn a_guard_that_cannot_be_installed_does_not_stop_an_open_trial() {
        let fake = Fake::new("BF fw-error no-iptables\n", "refused");
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
        let (state, note) = out.root_service.unwrap();
        assert_eq!(state, DefenseState::NotSupported);
        assert!(
            note.starts_with(
                "the task's image has no iptables, and BenchFlow could not install it"
            ),
            "{note}"
        );
        assert_eq!(fake.ran().len(), 1, "no self-test without a guard");
        // IPv6 without ip6tables: the guard holds for IPv4 only.
        let fake = Fake::new(
            "BF fw-guard-gap the sandbox has IPv6 but no ip6tables\nBF fw /sbin/iptables v6=local fw6=none reject=REJECT\nBF fw-sum 42\n",
            "refused",
        );
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert!(out.problem.is_none());
        let (state, note) = out.root_service.unwrap();
        assert_eq!(state, DefenseState::NotSupported);
        assert!(note.starts_with("the sandbox has IPv6"), "{note}");
        // An image with no bash or python3 cannot test it.
        let fake = Fake::new(INSTALLED, "notool");
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert!(out.problem.is_none());
        assert_eq!(out.root_service.unwrap().0, DefenseState::NotSupported);
    }

    #[tokio::test]
    async fn a_guard_that_leaks_stops_the_trial_and_a_root_agent_turns_it_off() {
        let fake = Fake::new(INSTALLED, "reached");
        let out = agent_stage(&fake, Network::Open, "agent").await;
        assert_eq!(out.problem.as_ref().map(|p| p.code.as_str()), Some("R109"));
        assert!(out.defense.is_none());
        assert_eq!(out.root_service.unwrap().0, DefenseState::Failed);
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage(&fake, Network::Open, "root").await;
        assert!(out.problem.is_none());
        assert_eq!(out.root_service.unwrap().0, DefenseState::Off);
        assert!(fake.ran().is_empty());
    }

    #[tokio::test]
    async fn a_network_daytona_blocks_itself_still_gets_the_guard() {
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage(&fake, Network::None, "agent").await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
        assert_eq!(out.defense.unwrap().0, DefenseState::Enforced);
        assert_eq!(out.root_service.unwrap().0, DefenseState::Enforced);
        let install = &fake.ran()[0].1;
        assert!(install.contains("--dport \"$port\""), "{install}");
        assert!(
            !install.contains("--dport 53"),
            "Daytona blocks the rest: {install}"
        );
    }

    #[tokio::test]
    async fn the_verifier_drops_the_guard_and_docker_never_installs_it() {
        let fake = Fake::new(INSTALLED, "refused");
        let p = plan(Network::Open, Network::Open, &[]);
        let out = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Verifier,
            Host::Agent,
            "root",
        )
        .await;
        assert!(out.problem.is_none() && out.root_service.is_none());
        let ran = fake.ran();
        assert_eq!(ran.len(), 1);
        assert!(ran[0].1.contains("unhook \"$fw4\""), "{}", ran[0].1);
        let fake = Fake::new(INSTALLED, "refused");
        let out = enforce(&fake, &docker(), &p, Stage::Agent, Host::Agent, "agent").await;
        assert!(out.root_service.is_none() && out.defense.is_none());
        assert!(fake.ran().is_empty());
    }

    #[test]
    fn every_daytona_agent_image_gets_the_iptables_layer() {
        let fixture =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        assert_eq!(*task.agent_network(), Network::Open);
        assert!(firewall_layer(&task, Host::Agent, true).is_some());
        assert!(firewall_layer(&task, Host::Agent, false).is_none());
        assert!(firewall_layer(&task, Host::SeparateVerifier, true).is_none());
    }

    #[tokio::test]
    async fn literal_entries_need_no_lookup() {
        let n = Network::Allow(vec!["10.0.0.0/8".into(), "192.0.2.7".into()]);
        let p = PhaseNet::resolve(&n).await.unwrap();
        assert_eq!(p.ranges, vec!["10.0.0.0/8", "192.0.2.7/32"]);
        assert!(p.names.is_empty());
        assert_eq!(p.describe(), "only 10.0.0.0/8, 192.0.2.7/32");
    }
}
