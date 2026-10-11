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
//! | `none` in every phase | `--network none` | Daytona blocks all traffic but IPv6; the uid firewall closes IPv6 |
//! | `none` in one phase | uid firewall for that phase | Daytona blocks all traffic but IPv6 during that phase; the uid firewall closes IPv6 |
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

/// One of BenchFlow's own root commands ([`Exec::system`]): programs from
/// the system folders only, and none of the image's loader variables,
/// since a folder or file they name may be the agent's.
fn as_root(script: impl Into<String>) -> Exec {
    Exec::new(script).system().cwd("/").timeout(STEP_TIMEOUT)
}

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
    /// Whether the agent's user comes from the task (`[agent] user`), not
    /// from the `sandbox-user` setting: a root agent the setting made is the
    /// user's doing, not the task's.
    pub agent_user_from_task: bool,
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
            agent_user_from_task: task.agent.user.is_some(),
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

    /// Whether `stage` is a `none` the backend blocks itself except for
    /// IPv6 (Daytona's block): the uid firewall must then close IPv6, and is
    /// needed unless the sandbox's kernel has no IPv6 at all.
    fn v6_gap(&self, stage: Stage, host: Host, caps: &Capabilities) -> bool {
        self.phase(stage).network == Network::None
            && caps.network_none_leaks_ipv6
            && self.native(stage, host, caps) == NetworkPolicy::None
    }

    /// Whether the IPv6 gap is all that needs a firewall in this sandbox:
    /// a backend whose own `none` holds (Docker's `--network none`) would
    /// run it with none, because every phase the sandbox hosts is `none`.
    /// Only then is a refusal for the gap the backend's (owner you, run it
    /// on Docker), not the task's.
    fn only_the_gap(&self, host: Host) -> bool {
        self.stages(host)
            .iter()
            .all(|s| self.phase(*s).network == Network::None)
    }

    /// Whether `stage` needs the uid firewall, and whether it must have it
    /// (nothing else would enforce the policy).
    fn firewall(&self, stage: Stage, host: Host, caps: &Capabilities) -> (bool, bool) {
        let native = self.native(stage, host, caps);
        match &self.phase(stage).network {
            Network::Open => (false, false),
            Network::None => {
                let need = native != NetworkPolicy::None || self.v6_gap(stage, host, caps);
                (need, need)
            }
            // The backend's own list or block covers IPv4 only where its
            // `none` leaks IPv6 (Daytona's list takes IPv4 ranges): the
            // firewall must hold, IPv6 included.
            Network::Allow(_) if caps.network_none_leaks_ipv6 => (true, true),
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
/// sandbox whose policy may need the uid firewall (a `none` phase too, on a
/// backend whose own block leaves IPv6 open), and for every agent sandbox
/// on a backend with a root service, which the daemon guard needs. The
/// layer never fails: an image without a package manager is refused later,
/// by name, if the firewall is required, and records the guard as not
/// supported otherwise.
pub fn firewall_layer(task: &TaskPackage, host: Host, caps: &Capabilities) -> Option<String> {
    if caps.root_service && host == Host::Agent {
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
    let none = nets.iter().any(|n| **n == Network::None);
    let mixed = none && nets.iter().any(|n| **n != Network::None);
    let v6_gap = none && caps.network_none_leaks_ipv6;
    (filtered || mixed || v6_gap).then(|| rules::INSTALL_LAYER.to_string())
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
/// The environment the stage's own self-test runs with. After the agent's
/// turn (the verifier stage) a folder on the image's `PATH` may be the
/// agent's, so the self-test — which runs as the stage's user and looks up
/// `timeout`, `bash`, `python3`, … — must use the verifier's trusted `PATH`
/// and drop the image's loader variables, as the verifier itself does.
/// Before the agent's turn there is nothing planted, so the image's `PATH`
/// is left alone.
#[derive(Clone, Copy, Default)]
pub struct SelfTest<'a> {
    pub path: Option<&'a str>,
    pub unset: &'a [String],
}

pub async fn enforce(
    sb: &dyn Sandbox,
    caps: &Capabilities,
    plan: &Plan,
    stage: Stage,
    host: Host,
    user: &str,
    self_test: SelfTest<'_>,
) -> Outcome {
    let mut out = enforce_stage(sb, caps, plan, stage, host, user, self_test).await;
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
    self_test: SelfTest<'_>,
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
            let _ = sb.exec(as_root(rules::remove())).await;
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
    // What the agent phase's install saw: the sandbox's IPv6, and whether
    // it has no working ip6tables, for the verifier phase's needs.
    let mut seen_v6: Option<(String, bool)> = None;
    // The backend blocks IPv4 itself; the chain closes IPv6.
    let v6_only = want && plan.v6_gap(stage, host, caps);
    let only_gap = v6_only && plan.only_the_gap(host);
    // Who made the agent root: the task, or the user's sandbox-user setting.
    let (root_who, root_fix) = if plan.agent_user_from_task {
        (
            "the task runs its agent as root",
            "drop [agent] user = \"root\" so the agent runs as the non-root sandbox user",
        )
    } else {
        (
            "your sandbox-user = \"root\" setting runs the agent as root",
            "set sandbox-user to a non-root user (or drop the setting) for this task",
        )
    };
    if stage == Stage::Agent && root_user && (want || !guard.is_empty()) {
        if only_gap {
            // Root could remove the chain that closes IPv6, and, with the
            // capabilities the chain needs, bring an IPv6 address up later
            // where the sandbox has none yet: refused wherever IPv6 leaks.
            return Outcome::failed(
                "agent: no network not enforced over IPv6: the agent runs as root".into(),
                Problem::new(
                    "R108",
                    Owner::You,
                    phase,
                    format!(
                        "the {} backend's network block leaves IPv6 open, and {root_who}, which could remove the firewall that closes it",
                        sb.backend()
                    ),
                    format!(
                        "run the task on Docker (-s docker), whose network none has no IPv6 either, or {root_fix}"
                    ),
                ),
            );
        } else if want && need {
            return Outcome::failed(
                format!(
                    "agent: {} not enforced: the agent runs as root",
                    net.describe()
                ),
                Problem::new(
                    "R108",
                    if plan.agent_user_from_task {
                        Owner::Task
                    } else {
                        Owner::You
                    },
                    phase,
                    format!(
                        "{root_who}, and a network allowlist, blocklist, or a \"none\" that differs from the verifier's is enforced by a firewall that root could remove"
                    ),
                    root_fix,
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
        // Where IPv6 must be closed by the chain: the IPv6 gap, or a policy
        // that must hold on a backend whose daemon answers on [::1] (an
        // agent that reached it there could flush the chain). Then any IPv6
        // at all needs ip6tables, not just an address that routes out.
        let v6_required =
            v6_only || (want && need && (caps.network_none_leaks_ipv6 || !guard.is_empty()));
        let script = rules::install(&scope, rules_net, guard, v6_only, v6_required);
        let out = match sb.exec(as_root(script)).await {
            Ok(o) => o,
            Err(e) => {
                return Outcome::failed(
                    format!("{}: the firewall install did not run: {e}", stage.label()),
                    crate::trial::sandbox_problem(phase, e),
                );
            }
        };
        let report = Report::parse(&out.stdout);
        // Closing IPv6 is needed unless the kernel has no IPv6 at all: a
        // link-local address now may route out later (an address arriving
        // by router advertisement), and without the chain the daemon guard
        // is missing too, which gives the agent root to bring one up.
        let v6_state = report.first("fw-v6").unwrap_or("unknown");
        let no_fw6 = report
            .first("fw")
            .is_some_and(|how| how.split_whitespace().any(|w| w == "fw6=none"));
        seen_v6 = Some((v6_state.to_string(), no_fw6));
        let need = need && !(v6_only && v6_state == "absent");
        match (report.first("fw"), report.first("fw-error")) {
            (Some(how), None) => {
                installed = report.first("fw-sum").map(str::to_string);
                if want {
                    notes.push(rules::describe(&scope, net, how, v6_only));
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
                // An image that cannot close IPv6 where Docker would not need
                // it closed is the backend's gap, not the task's: Docker
                // runs a none sandbox with no firewall, and checks ip6tables
                // only where IPv6 routes out. Other install errors (no
                // permission, no conntrack, a failed rule, no output) keep
                // their own owners.
                let kind = error.split_whitespace().next().unwrap_or("");
                let gap_error = (kind == "no-iptables" && only_gap)
                    || (kind == "ipv6-unfiltered" && (only_gap || v6_state != "global"));
                let (owner, msg, next) = if gap_error {
                    (
                        Owner::You,
                        format!(
                            "on the {} backend BenchFlow's firewall must close IPv6, and the image cannot: {msg}",
                            sb.backend()
                        ),
                        "run the task on Docker (-s docker), or add iptables and ip6tables to the image".to_string(),
                    )
                } else {
                    (owner, msg.to_string(), next.to_string())
                };
                if want && need {
                    return Outcome::failed(
                        format!("{}: no firewall: {error}", stage.label()),
                        Problem::new("R108", owner, phase, msg, next).detail(out.tail(1500)),
                    );
                }
                if want && v6_only {
                    notes.push(format!(
                        "no uid firewall ({error}), and none needed: the sandbox has no IPv6 (IPv6 {v6_state})"
                    ));
                } else if want {
                    notes.push(format!("no uid firewall ({error})"));
                }
                if !guard.is_empty() {
                    open_ports = Some((rules::guard_error(&error), false));
                }
            }
        }
    }
    // A later verifier phase in this sandbox that must close IPv6 with no
    // ip6tables here would be refused only after the agent ran: refuse now.
    if stage == Stage::Agent
        && shared
        && let Some((v6, true)) = &seen_v6
        && v6 != "absent"
    {
        let (vw, vn) = plan.firewall(Stage::Verifier, host, caps);
        let required =
            plan.v6_gap(Stage::Verifier, host, caps) || (vw && vn && caps.network_none_leaks_ipv6);
        if vw && vn && required {
            // Docker checks ip6tables only for IPv6 that routes out.
            let owner = if v6 == "global" {
                Owner::Task
            } else {
                Owner::You
            };
            return Outcome::failed(
                format!(
                    "verifier: {} cannot be enforced: no ip6tables",
                    plan.phase(Stage::Verifier).describe()
                ),
                Problem::new(
                    "R108",
                    owner,
                    phase,
                    format!(
                        "the verifier's network must have IPv6 closed on the {} backend, and the image has IPv6 but no ip6tables; refused before the agent runs rather than after",
                        sb.backend()
                    ),
                    if owner == Owner::You {
                        "run the task on Docker (-s docker), or add ip6tables to the image"
                    } else {
                        "add ip6tables to the task's Dockerfile (Debian's iptables package has it)"
                    },
                ),
            );
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
    // The self-test runs as the stage's user. After the agent's turn it uses
    // the verifier's trusted PATH and drops the image's loader variables, so
    // a program the agent planted on the image's PATH is never found or
    // preloaded (see [`SelfTest`]).
    // From `/`, not the workdir the sandbox would start it in: nothing the
    // agent left there is picked up.
    let mut test_exec = Exec::new(test.script())
        .user(user)
        .cwd("/")
        .timeout(STEP_TIMEOUT);
    if let Some(path) = self_test.path {
        test_exec = test_exec.env("PATH", path);
    }
    test_exec = test_exec.unset(self_test.unset.iter().cloned());
    let ran = match sb.exec(test_exec).await {
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
        match sb.exec(as_root(rules::fingerprint())).await {
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
            // The sandbox went away (D4 chaos runs: deleted through
            // Daytona's API): not a leak, and not BenchFlow's.
            Err(e) => {
                return Outcome::failed(
                    format!("{}: the rules could not be checked: {e}", stage.label()),
                    crate::trial::sandbox_problem(phase, e),
                );
            }
        }
    }
    // An allowed canary that failed: blocked by the firewall, or down?
    if !verdict.unreachable.is_empty() && stage == Stage::Agent && !root_user {
        let probe = selftest::reach_script(&verdict.unreachable);
        if let Ok(o) = sb.exec(as_root(probe)).await {
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
        let root_reached: Vec<String> =
            match sb.exec(as_root(selftest::service_probe(guarded))).await {
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
            network_none_leaks_ipv6: true,
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
            agent_user_from_task: true,
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
        // Daytona's list takes IPv4 only and its block leaves IPv6 open, so
        // the uid firewall must hold under the list too.
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &daytona()),
            (true, true)
        );
        // Where the backend's none holds, its list is enough.
        let tight = Capabilities {
            network_none_leaks_ipv6: false,
            ..daytona()
        };
        assert_eq!(p.firewall(Stage::Agent, Host::Agent, &tight), (true, false));
        assert_eq!(
            p.native(Stage::Verifier, Host::Agent, &daytona()),
            NetworkPolicy::None
        );
        // Daytona's block leaves IPv6 open, so the firewall must hold too.
        assert_eq!(
            p.firewall(Stage::Verifier, Host::Agent, &daytona()),
            (true, true)
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
        /// The sandbox is gone by the time the rules are checked again.
        lost: bool,
        ran: std::sync::Mutex<Vec<(Option<String>, String)>>,
        /// (command, PATH, unset) of every exec, for the self-test checks.
        env: std::sync::Mutex<Vec<ExecEnv>>,
    }

    /// One exec's command, its `PATH`, and the variables it dropped.
    type ExecEnv = (String, Option<String>, Vec<String>);

    impl Fake {
        fn new(install: &'static str, svc: &'static str) -> Self {
            Self {
                install,
                svc,
                root_svc: "reached",
                lost: false,
                ran: Default::default(),
                env: Default::default(),
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

        /// A refusal for every blocked address and no answer for every
        /// name a self-test script tries.
        fn canary_lines(c: &str) -> String {
            let mut o = String::new();
            for l in c.lines() {
                if l.contains("BF canary block") {
                    o.push_str("BF canary block refused 192.0.2.9:443\n");
                } else if l.contains("BF canary dns") {
                    o.push_str("BF canary dns unresolved example.org\n");
                }
            }
            o
        }

        fn ran(&self) -> Vec<(Option<String>, String)> {
            self.ran.lock().unwrap().clone()
        }

        /// The PATH and unset of the one exec whose command holds `needle`.
        fn env_of(&self, needle: &str) -> (Option<String>, Vec<String>) {
            self.env
                .lock()
                .unwrap()
                .iter()
                .find(|(c, _, _)| c.contains(needle))
                .map(|(_, p, u)| (p.clone(), u.clone()))
                .unwrap_or_else(|| panic!("no exec with {needle}"))
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
            self.env.lock().unwrap().push((
                c.clone(),
                exec.env.get("PATH").cloned(),
                exec.unset.clone(),
            ));
            if self.lost && !c.contains("jump_new") && c.contains("BF fw-sum $(fingerprint)") {
                return Err(SandboxError::Transport(
                    "Daytona lost the sandbox while checking a command (HTTP 404)".into(),
                ));
            }
            let stdout = if c.contains("jump_new") {
                self.install.to_string()
            } else if c.contains("BF tamper") {
                let mut o = Self::svc_lines(&c, self.svc);
                o.push_str(&Self::canary_lines(&c));
                o.push_str("BF tamper tried\n");
                o
            } else if c.contains("BF canary svc") {
                Self::svc_lines(&c, self.root_svc)
            } else if c.contains("BF canary") {
                Self::canary_lines(&c)
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
            SelfTest::default(),
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
    async fn a_sandbox_lost_after_the_self_test_is_the_sandboxs() {
        let mut fake = Fake::new(INSTALLED, "refused");
        fake.lost = true;
        let out = agent_stage(&fake, Network::Open, "agent").await;
        let p = out.problem.expect("a problem");
        assert_eq!(
            (p.code.as_str(), p.owner),
            ("R103", Owner::Sandbox),
            "{p:?}"
        );
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
        // Daytona's block leaves IPv6 open, so the chain closes IPv6 too:
        // after the IPv6 chain is set up, loopback returns (a task's own
        // resolver on it included) and the rest is refused.
        assert!(install.contains("jump_new 2"), "the install: {install}");
        let v6 = &install[install.find("setup \"$fw6\"").expect("an IPv6 chain")..];
        let lo = v6.find("-o lo -j RETURN").expect("loopback returns");
        let rej = v6
            .find("-j \"$REJ\" || exit_rule")
            .expect("the rest refused");
        assert!(lo < rej, "{v6}");
        assert!(!install.contains("--dport 53"), "{install}");
    }

    /// The agent's phase on Daytona with `verifier` as the verifier's
    /// network, in the same sandbox.
    async fn agent_stage_with(
        fake: &Fake,
        agent: Network,
        verifier: Network,
        user: &str,
    ) -> Outcome {
        let p = plan(agent, verifier, &[]);
        enforce(
            fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            user,
            SelfTest::default(),
        )
        .await
    }

    #[tokio::test]
    async fn a_root_agent_with_no_network_on_daytona_is_refused_by_name() {
        // Every phase none: Docker would run it, so the refusal is the
        // backend's (owner you, run it on Docker). Root could remove the
        // chain, or bring IPv6 up where there is none yet, so the sandbox's
        // IPv6 now does not matter.
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage_with(&fake, Network::None, Network::None, "root").await;
        let p = out.problem.expect("refused");
        assert_eq!((p.code.as_str(), p.owner), ("R108", Owner::You));
        assert!(p.message.contains("leaves IPv6 open"), "{}", p.message);
        assert!(p.next.contains("-s docker"), "{}", p.next);
        assert!(fake.ran().is_empty(), "nothing ran");
        // None in the agent's phase only: Docker needs a firewall root
        // could remove too, so the task is refused as the task's.
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage_with(&fake, Network::None, Network::Open, "root").await;
        let p = out.problem.expect("refused");
        assert_eq!((p.code.as_str(), p.owner), ("R108", Owner::Task));
        assert!(
            p.message.contains("differs from the verifier"),
            "{}",
            p.message
        );
        assert!(!p.next.contains("docker"), "{}", p.next);
        assert!(p.next.contains("[agent] user"), "{}", p.next);
        // An allowlist on Daytona needs the firewall (IPv6): refused too.
        let fake = Fake::new(INSTALLED, "refused");
        // With a range Daytona applies itself (its list covers IPv4 only).
        let p = plan(
            Network::Allow(vec!["10.0.0.0/8".into()]),
            Network::Open,
            &["10.0.0.0/8"],
        );
        assert!(matches!(
            p.native(Stage::Agent, Host::Agent, &daytona()),
            NetworkPolicy::Filtered { allow: Some(_) }
        ));
        let out = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            "root",
            SelfTest::default(),
        )
        .await;
        assert_eq!(
            out.problem.map(|p| (p.code, p.owner)),
            Some(("R108".into(), Owner::Task))
        );
    }

    #[tokio::test]
    async fn no_iptables_stops_a_none_on_daytona_unless_the_kernel_has_no_ipv6() {
        // Global or link-local IPv6: refused. Link-local may route out
        // later, and the missing chain leaves the daemon unguarded too.
        for v6 in ["global", "local"] {
            let install: &'static str = if v6 == "global" {
                "BF fw-v6 global\nBF fw-error no-iptables\n"
            } else {
                "BF fw-v6 local\nBF fw-error no-iptables\n"
            };
            let fake = Fake::new(install, "refused");
            let out = agent_stage_with(&fake, Network::None, Network::None, "agent").await;
            let p = out.problem.expect("refused");
            // Every phase none: the backend's gap, owner you.
            assert_eq!((p.code.as_str(), p.owner), ("R108", Owner::You), "{v6}");
            assert!(p.message.contains("must close IPv6"), "{}", p.message);
            assert!(
                p.next.contains("-s docker") && p.next.contains("ip6tables"),
                "{}",
                p.next
            );
        }
        // Agent phase only: Docker needs the firewall too, the task's.
        let fake = Fake::new("BF fw-v6 local\nBF fw-error no-iptables\n", "refused");
        let out = agent_stage_with(&fake, Network::None, Network::Open, "agent").await;
        assert_eq!(out.problem.map(|p| p.owner), Some(Owner::Task));
        // No IPv6 in the kernel: Daytona's block is the whole policy.
        let fake = Fake::new("BF fw-v6 absent\nBF fw-error no-iptables\n", "refused");
        let out = agent_stage_with(&fake, Network::None, Network::None, "agent").await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
        let (state, note) = out.defense.unwrap();
        assert_eq!(state, DefenseState::Enforced);
        assert!(note.contains("none needed"), "{note}");
    }

    #[tokio::test]
    async fn a_separate_verifier_with_no_network_on_daytona_closes_ipv6_for_everyone() {
        let fake = Fake::new(
            "BF fw-v6 global\nBF fw /usr/sbin/iptables v6=global fw6=/usr/sbin/ip6tables reject=REJECT\nBF fw-sum 42\n",
            "refused",
        );
        let mut p = plan(Network::Open, Network::None, &[]);
        p.separate = true;
        let out = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Verifier,
            Host::SeparateVerifier,
            "root",
            SelfTest::default(),
        )
        .await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
        let install = &fake.ran()[0].1;
        // The first command is the install, for everyone, root included.
        assert!(install.contains("jump_new 1"), "the install: {install}");
        assert!(!install.contains("--uid-owner"), "everyone: {install}");
        assert!(!install.contains("--dport 53"), "{install}");
        let v6 = &install[install.find("setup \"$fw6\"").expect("an IPv6 chain")..];
        let lo = v6.find("-o lo -j RETURN").expect("loopback returns");
        let rej = v6
            .find("-j \"$REJ\" || exit_rule")
            .expect("the rest refused");
        assert!(lo < rej, "{v6}");
    }

    #[test]
    fn none_on_a_backend_whose_block_misses_ipv6_needs_the_firewall() {
        let p = plan(Network::None, Network::None, &[]);
        let leaky = daytona();
        let tight = Capabilities {
            network_none_leaks_ipv6: false,
            ..daytona()
        };
        assert_eq!(p.firewall(Stage::Agent, Host::Agent, &leaky), (true, true));
        assert_eq!(
            p.firewall(Stage::Verifier, Host::Agent, &leaky),
            (true, true)
        );
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &tight),
            (false, false)
        );
        assert_eq!(
            p.firewall(Stage::Agent, Host::Agent, &docker()),
            (false, false)
        );
    }

    /// The verifier-stage self-test runs after the agent's turn. It must
    /// use the verifier's trusted PATH and drop the loader variables, not
    /// the image's PATH, so a `timeout` the agent planted on the image's
    /// PATH is never run as root (security review finding 2).
    #[tokio::test]
    async fn the_verifier_self_test_uses_the_trusted_path_without_loader_variables() {
        let fake = Fake::new(INSTALLED, "refused");
        let p = plan(Network::Open, Network::None, &[]);
        let unset = vec!["LD_PRELOAD".to_string(), "LD_SHOW_AUXV".to_string()];
        let _ = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Verifier,
            Host::Agent,
            "root",
            SelfTest {
                path: Some("/trusted/bin"),
                unset: &unset,
            },
        )
        .await;
        // `bf_try()` is defined only in the self-test script.
        let (path, got) = fake.env_of("bf_try()");
        assert_eq!(path.as_deref(), Some("/trusted/bin"), "self-test PATH");
        for v in &unset {
            assert!(got.contains(v), "{v} kept for the self-test: {got:?}");
        }
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
            SelfTest::default(),
        )
        .await;
        assert!(out.problem.is_none() && out.root_service.is_none());
        let ran = fake.ran();
        assert_eq!(ran.len(), 1);
        assert!(ran[0].1.contains("unhook \"$fw4\""), "{}", ran[0].1);
        let fake = Fake::new(INSTALLED, "refused");
        let out = enforce(
            &fake,
            &docker(),
            &p,
            Stage::Agent,
            Host::Agent,
            "agent",
            SelfTest::default(),
        )
        .await;
        assert!(out.root_service.is_none() && out.defense.is_none());
        assert!(fake.ran().is_empty());
    }

    #[test]
    fn every_daytona_agent_image_gets_the_iptables_layer() {
        let fixture =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let task = benchflow_task::load(&fixture).expect("the hello fixture loads");
        assert_eq!(*task.agent_network(), Network::Open);
        assert!(firewall_layer(&task, Host::Agent, &daytona_root_service()).is_some());
        assert!(firewall_layer(&task, Host::Agent, &docker()).is_none());
        assert!(firewall_layer(&task, Host::SeparateVerifier, &daytona_root_service()).is_none());
        // A separate verifier with no network on a backend whose block
        // leaves IPv6 open needs iptables too.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tasks/separate-verifier");
        let mut task = benchflow_task::load(&fixture).expect("the fixture loads");
        task.verifier.network = Some(Network::None);
        assert!(firewall_layer(&task, Host::SeparateVerifier, &daytona()).is_some());
        assert!(firewall_layer(&task, Host::SeparateVerifier, &docker()).is_none());
    }

    #[tokio::test]
    async fn literal_entries_need_no_lookup() {
        let n = Network::Allow(vec!["10.0.0.0/8".into(), "192.0.2.7".into()]);
        let p = PhaseNet::resolve(&n).await.unwrap();
        assert_eq!(p.ranges, vec!["10.0.0.0/8", "192.0.2.7/32"]);
        assert!(p.names.is_empty());
        assert_eq!(p.describe(), "only 10.0.0.0/8, 192.0.2.7/32");
    }

    #[tokio::test]
    async fn a_root_agent_the_sandbox_user_setting_made_is_yours_not_the_tasks() {
        // A shared sandbox with none in the agent's phase only: refused on
        // every backend, but the setting made the agent root, not the task.
        let fake = Fake::new(INSTALLED, "refused");
        let mut p = plan(Network::None, Network::Open, &[]);
        p.agent_user_from_task = false;
        let out = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            "root",
            SelfTest::default(),
        )
        .await;
        let pr = out.problem.expect("refused");
        assert_eq!((pr.code.as_str(), pr.owner), ("R108", Owner::You));
        assert!(pr.message.contains("sandbox-user"), "{}", pr.message);
        assert!(!pr.message.contains("the task runs"), "{}", pr.message);
        assert!(pr.next.contains("sandbox-user"), "{}", pr.next);
        // Every phase none on Daytona: yours either way, naming the setting.
        let fake = Fake::new(INSTALLED, "refused");
        let mut p = plan(Network::None, Network::None, &[]);
        p.agent_user_from_task = false;
        let out = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            "root",
            SelfTest::default(),
        )
        .await;
        let pr = out.problem.expect("refused");
        assert_eq!(pr.owner, Owner::You);
        assert!(pr.message.contains("sandbox-user"), "{}", pr.message);
    }

    #[tokio::test]
    async fn an_allowlist_on_daytona_needs_ip6tables_wherever_there_is_ipv6() {
        // The daemon answers on [::1]: with no IPv6 chain the agent could
        // reach it and flush the rules. The install must refuse any IPv6
        // without ip6tables, not only an address that routes out.
        let allow = Network::Allow(vec!["10.0.0.0/8".into()]);
        let fake = Fake::new(INSTALLED, "refused");
        let p = plan(allow.clone(), Network::Open, &["10.0.0.0/8"]);
        let _ = enforce(
            &fake,
            &daytona_root_service(),
            &p,
            Stage::Agent,
            Host::Agent,
            "agent",
            SelfTest::default(),
        )
        .await;
        let install = &fake.ran()[0].1;
        assert!(
            install.contains("[ \"$v6\" != absent ] && [ -z \"$fw6\" ]"),
            "{install}"
        );
        // Docker has no daemon: an IPv6 address that routes out is what
        // needs ip6tables there.
        let fake = Fake::new(INSTALLED, "refused");
        let _ = enforce(
            &fake,
            &docker(),
            &p,
            Stage::Agent,
            Host::Agent,
            "agent",
            SelfTest::default(),
        )
        .await;
        let install = &fake.ran()[0].1;
        assert!(
            install.contains("[ \"$v6\" = global ] && [ -z \"$fw6\" ]"),
            "{install}"
        );
        // Refused without ip6tables and with IPv6 that does not route out:
        // yours, since Docker would run the image; with a global address,
        // Docker needs ip6tables too, so the task's.
        for (v6, owner) in [("local", Owner::You), ("global", Owner::Task)] {
            let install: &'static str = if v6 == "local" {
                "BF fw-v6 local\nBF fw-error ipv6-unfiltered\n"
            } else {
                "BF fw-v6 global\nBF fw-error ipv6-unfiltered\n"
            };
            let fake = Fake::new(install, "refused");
            let out = enforce(
                &fake,
                &daytona_root_service(),
                &p,
                Stage::Agent,
                Host::Agent,
                "agent",
                SelfTest::default(),
            )
            .await;
            let pr = out.problem.expect("refused");
            assert_eq!((pr.code.as_str(), pr.owner), ("R108", owner), "{v6}");
            if owner == Owner::You {
                assert!(
                    pr.next.contains("-s docker") && pr.next.contains("ip6tables"),
                    "{}",
                    pr.next
                );
            }
        }
    }

    #[tokio::test]
    async fn only_a_missing_tool_is_blamed_on_the_ipv6_gap() {
        // Every phase none on Daytona: a missing iptables is the gap's
        // (yours); no permission is the sandbox's, a failed rule BenchFlow's.
        let cases: [(&'static str, Owner); 3] = [
            ("BF fw-v6 global\nBF fw-error no-iptables\n", Owner::You),
            (
                "BF fw-v6 global\nBF fw-error no-permission denied\n",
                Owner::Sandbox,
            ),
            (
                "BF fw-v6 global\nBF fw-error rule-failed ip6tables\n",
                Owner::Benchflow,
            ),
        ];
        for (install, owner) in cases {
            let fake = Fake::new(install, "refused");
            let out = agent_stage_with(&fake, Network::None, Network::None, "agent").await;
            let pr = out.problem.expect("refused");
            assert_eq!(pr.owner, owner, "{install}: {}", pr.message);
            if owner != Owner::You {
                assert!(
                    !pr.next.contains("add iptables and ip6tables"),
                    "{}",
                    pr.next
                );
            }
        }
        // None in the agent's phase only, no ip6tables, link-local IPv6:
        // Docker would run it (it checks only routable IPv6), so yours.
        let fake = Fake::new("BF fw-v6 local\nBF fw-error ipv6-unfiltered\n", "refused");
        let out = agent_stage_with(&fake, Network::None, Network::Open, "agent").await;
        assert_eq!(out.problem.map(|p| p.owner), Some(Owner::You));
    }

    #[tokio::test]
    async fn a_verifier_that_will_need_ip6tables_is_refused_before_the_agent() {
        // Open agent phase, none for the verifier, the same sandbox: the
        // agent's guard install sees IPv6 loopback and no ip6tables.
        let fake = Fake::new(
            "BF fw-v6 local\nBF fw-guard-ports 2280\nBF fw-guard-gap no ip6tables\nBF fw /usr/sbin/iptables v6=local fw6=none reject=REJECT\nBF fw-sum 42\n",
            "refused",
        );
        let out = agent_stage_with(&fake, Network::Open, Network::None, "agent").await;
        let pr = out.problem.expect("refused before the agent");
        assert_eq!((pr.code.as_str(), pr.owner), ("R108", Owner::You));
        assert!(
            pr.message.contains("before the agent runs"),
            "{}",
            pr.message
        );
        // With ip6tables there it goes on.
        let fake = Fake::new(INSTALLED, "refused");
        let out = agent_stage_with(&fake, Network::Open, Network::None, "agent").await;
        assert!(out.problem.is_none(), "{:?}", out.problem);
    }
}
