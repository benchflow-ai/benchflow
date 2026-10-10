//! The firewall self-test: canary connections as the phase's own user. An
//! allowed destination must answer, a blocked one must not, an unlisted
//! name must not resolve, and the agent user must fail to change the rules.

use std::net::IpAddr;

use benchflow_sandbox::shell::quote;
use benchflow_task::values::Network;

use super::rules::{PINNED_NOWHERE, PINNED_NOWHERE_V6};
use super::{PhaseNet, Stage, contains};
use crate::hardening::Report;

/// How a leak caused by a missing tool starts, so the caller can blame the
/// image rather than the firewall.
pub(super) const NO_TOOL: &str = "no tool";

/// Destinations outside every allowlist the canaries try, with a port each
/// one answers on.
const OUTSIDE: &[(&str, u16)] = &[("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443)];
const OUTSIDE_V6: (&str, u16) = ("2606:4700:4700::1111", 443);
/// Names a canary tries to resolve when DNS should be refused.
const UNLISTED: &[&str] = &["example.org", "example.net", "dns.google"];

/// Shell helpers: `bf_try addr port` prints reached, refused, timeout, or
/// notool; `bf_resolve name` prints `resolved <addr>`, unresolved, or notool.
const HELPERS: &str = r#"bf_try() {
  if [ -n "${BASH_VERSION:-}" ]; then
    if command -v timeout >/dev/null 2>&1; then timeout 6 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; rc=$?
    else bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; rc=$?; fi
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import socket,sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), 6)' "$1" "$2" 2>/dev/null; rc=$?
  else echo notool; return; fi
  case $rc in 0) echo reached;; 124) echo timeout;; *) echo refused;; esac
}
bf_resolve() {
  if command -v getent >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then out=$(timeout 10 getent hosts "$1" 2>/dev/null | head -n 1); else out=$(getent hosts "$1" 2>/dev/null | head -n 1); fi
  elif command -v python3 >/dev/null 2>&1; then
    out=$(python3 -c 'import socket,sys; print(socket.gethostbyname(sys.argv[1]))' "$1" 2>/dev/null)
  else echo notool; return; fi
  if [ -n "$out" ]; then echo "resolved ${out%% *}"; else echo unresolved; fi
}
has_v6() { [ -e /proc/net/if_inet6 ] && awk '$4 == "00" && $6 != "lo" { f = 1 } END { exit !f }' /proc/net/if_inet6; }
"#;

/// Sets `own4` and `own6` to the sandbox's own non-loopback addresses and
/// `lo6` when it has IPv6 loopback, each empty when absent; needs only awk.
const OWN_ADDRESSES: &str = r#"own4=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v ':' | grep -v '^127\.' | grep . | head -n 1)
[ -n "$own4" ] || own4=$(awk '/32 host LOCAL/ {print a} {a = $2}' /proc/net/fib_trie 2>/dev/null | grep -v '^127\.' | head -n 1)
lo6=""; own6=""
if [ -e /proc/net/if_inet6 ]; then
  grep -q ' lo$' /proc/net/if_inet6 && lo6=1
  own6=$(awk '$4 == "00" && $6 != "lo" { a = $1; s = ""; for (i = 1; i <= 32; i += 4) s = s (i > 1 ? ":" : "") substr(a, i, 4); print s; exit }' /proc/net/if_inet6)
fi
"#;

/// One canary.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Canary {
    /// Must connect.
    Allowed(String, u16),
    /// Must not connect.
    Blocked(String, u16),
    /// Must not resolve; `strict` false when the allowlist may cover a
    /// resolver, so an answer is only noted.
    Dns { name: String, strict: bool },
    /// A blocked name must resolve to the pinned address, if at all.
    Pinned(String),
    /// A local root service's port, which the agent user must not reach on
    /// loopback or the sandbox's own address.
    Service(u16),
}

/// The canaries for one phase.
pub(super) struct Test {
    canaries: Vec<Canary>,
    tamper: bool,
}

/// What the self-test showed.
#[derive(Debug, Default)]
pub(super) struct Verdict {
    /// What held, for the defense's note.
    pub held: Vec<String>,
    /// What leaked: the trial must not run.
    pub leaks: Vec<String>,
    /// Allowed destinations that did not answer (`addr:port`).
    pub unreachable: Vec<String>,
    /// Answers that are only noted.
    pub notes: Vec<String>,
    /// The daemon's addresses (`addr:port`) the user could not reach.
    pub services_refused: Vec<String>,
}

impl Verdict {
    pub(super) fn summary(&self) -> String {
        let mut parts = vec![format!("self-test: {}", self.held.join("; "))];
        parts.extend(self.notes.iter().cloned());
        parts.join("; ")
    }
}

fn target(addr: &str, port: u16) -> String {
    if addr.contains(':') {
        format!("[{addr}]:{port}")
    } else {
        format!("{addr}:{port}")
    }
}

impl Test {
    pub(super) fn new(net: &PhaseNet, firewall: bool, stage: Stage, guard: &[u16]) -> Self {
        let mut canaries = Vec::new();
        let allowed = |a: IpAddr| net.ranges.iter().any(|r| contains(r, a));
        match &net.network {
            Network::Allow(_) | Network::None => {
                if matches!(net.network, Network::Allow(_)) {
                    // The first name with an address; IPv4 first, since
                    // most sandboxes have no IPv6 route.
                    let pick = net.names.iter().find_map(|(_, a)| {
                        a.iter()
                            .find(|x| x.is_ipv4())
                            .or_else(|| a.first())
                            .copied()
                    });
                    if let Some(a) = pick {
                        canaries.push(Canary::Allowed(a.to_string(), 443));
                    }
                }
                let mut outside = 0;
                for (addr, port) in OUTSIDE {
                    if outside < 2 && !allowed(addr.parse().expect("canary address")) {
                        canaries.push(Canary::Blocked(addr.to_string(), *port));
                        outside += 1;
                    }
                }
                if !allowed(OUTSIDE_V6.0.parse().expect("canary address")) {
                    canaries.push(Canary::Blocked(OUTSIDE_V6.0.into(), OUTSIDE_V6.1));
                }
                let listed: Vec<&str> = net.names.iter().map(|(n, _)| n.as_str()).collect();
                if let Some(name) = UNLISTED.iter().find(|n| !listed.contains(n)) {
                    // A broad range may hold the sandbox's resolver.
                    let strict = net
                        .ranges
                        .iter()
                        .all(|r| r.ends_with("/32") || r.ends_with("/128"));
                    canaries.push(Canary::Dns {
                        name: name.to_string(),
                        strict,
                    });
                }
            }
            Network::Block(_) => {
                for (name, addrs) in net.names.iter().take(3) {
                    if let Some(a) = addrs.iter().find(|x| x.is_ipv4()).or_else(|| addrs.first()) {
                        canaries.push(Canary::Blocked(a.to_string(), 443));
                    }
                    canaries.push(Canary::Pinned(name.clone()));
                }
                for r in net
                    .ranges
                    .iter()
                    .filter(|r| r.ends_with("/32") || r.ends_with("/128"))
                {
                    let addr = r.rsplit_once('/').map(|(a, _)| a).unwrap_or(r);
                    let c = Canary::Blocked(addr.to_string(), 443);
                    if !canaries.contains(&c) && canaries.len() < 8 {
                        canaries.push(c);
                    }
                }
            }
            Network::Open => {}
        }
        canaries.extend(guard.iter().map(|p| Canary::Service(*p)));
        Self {
            canaries,
            tamper: firewall && stage == Stage::Agent,
        }
    }

    /// The script, run as the phase's user.
    pub(super) fn script(&self) -> String {
        let mut s = String::from(HELPERS);
        if self
            .canaries
            .iter()
            .any(|c| matches!(c, Canary::Service(_)))
        {
            s.push_str(OWN_ADDRESSES);
        }
        for c in &self.canaries {
            match c {
                Canary::Allowed(a, p) | Canary::Blocked(a, p) => {
                    let kind = if matches!(c, Canary::Allowed(..)) {
                        "allow"
                    } else {
                        "block"
                    };
                    let t = quote(&target(a, *p));
                    let run = format!(
                        "echo \"BF canary {kind} $(bf_try {a} {p}) \"{t}\n",
                        a = quote(a)
                    );
                    if a.contains(':') {
                        s.push_str(&format!(
                            "if has_v6; then {run}; else echo \"BF canary {kind} no-ipv6 \"{t}; fi\n",
                            run = run.trim_end()
                        ));
                        s.push('\n');
                    } else {
                        s.push_str(&run);
                    }
                }
                Canary::Dns { name, .. } => s.push_str(&format!(
                    "echo \"BF canary dns $(bf_resolve {n}) \"{n}\n",
                    n = quote(name)
                )),
                Canary::Service(port) => s.push_str(&format!(
                    "echo \"BF canary svc $(bf_try 127.0.0.1 {port}) 127.0.0.1:{port}\"\n\
                     if [ -n \"$own4\" ]; then echo \"BF canary svc $(bf_try \"$own4\" {port}) $own4:{port}\"; fi\n\
                     if [ -n \"$lo6\" ]; then echo \"BF canary svc $(bf_try ::1 {port}) [::1]:{port}\"; fi\n\
                     if [ -n \"$own6\" ]; then echo \"BF canary svc $(bf_try \"$own6\" {port}) [$own6]:{port}\"; fi\n"
                )),
                Canary::Pinned(name) => s.push_str(&format!(
                    "echo \"BF canary pin $(bf_resolve {n}) \"{n}\n",
                    n = quote(name)
                )),
            }
        }
        if self.tamper {
            // The agent user tries to lift the rules and re-point a name;
            // root compares the fingerprint afterwards.
            s.push_str(
                r#"for c in iptables iptables-nft iptables-legacy ip6tables ip6tables-nft ip6tables-legacy; do
  if command -v "$c" >/dev/null 2>&1; then "$c" -w -F >/dev/null 2>&1; "$c" -w -P OUTPUT ACCEPT >/dev/null 2>&1; fi
done
printf '1.1.1.1 example.org\n' >> /etc/hosts 2>/dev/null
echo "BF tamper tried"
"#,
            );
        }
        s
    }

    /// Read the script's report.
    pub(super) fn judge(&self, report: &Report) -> Verdict {
        let mut v = Verdict::default();
        let mut refused: Vec<String> = Vec::new();
        let mut services: Vec<String> = Vec::new();
        for line in report.all("canary") {
            let mut w = line.splitn(3, ' ');
            let (kind, result, rest) = (
                w.next().unwrap_or(""),
                w.next().unwrap_or(""),
                w.next().unwrap_or(""),
            );
            // `resolved` carries the address before the name.
            let (addr, what) = match result {
                "resolved" => rest.split_once(' ').unwrap_or((rest, "")),
                _ => ("", rest),
            };
            if result == "notool" {
                v.leaks.push(format!(
                    "{NO_TOOL} in the image can open a connection or resolve a name (bash, python3, getent)"
                ));
                continue;
            }
            match kind {
                "allow" if result == "reached" => {
                    v.held.push(format!("reached the allowed {what}"))
                }
                "allow" => v.unreachable.push(what.to_string()),
                "block" if result == "reached" => {
                    v.leaks.push(format!("reached the blocked {what}"))
                }
                "block" if result == "no-ipv6" => {}
                "block" => refused.push(what.to_string()),
                "dns" if result == "resolved" => {
                    let strict = self
                        .canaries
                        .iter()
                        .any(|c| matches!(c, Canary::Dns { name, strict: true } if name == what));
                    let msg = format!("resolved the unlisted name {what} (to {addr})");
                    if strict {
                        v.leaks.push(msg);
                    } else {
                        v.notes
                            .push(format!("{msg}; the allowlist's ranges may hold a resolver"));
                    }
                }
                "dns" => v.held.push(format!("no DNS answer for {what}")),
                "pin"
                    if result == "resolved"
                        && addr != PINNED_NOWHERE
                        && addr != PINNED_NOWHERE_V6 =>
                {
                    v.leaks
                        .push(format!("the blocked name {what} resolved to {addr}"))
                }
                "svc" if result == "reached" => v.leaks.push(format!(
                    "the agent user reached the sandbox daemon at {what}"
                )),
                "svc" => {
                    services.push(what.to_string());
                    v.services_refused.push(what.to_string());
                }
                "pin" => v.held.push(format!("{what} pinned nowhere")),
                _ => {}
            }
        }
        if !refused.is_empty() {
            v.held
                .push(format!("could not reach {}", refused.join(", ")));
        }
        if !services.is_empty() {
            v.held.push(format!(
                "could not reach the sandbox daemon at {}",
                services.join(", ")
            ));
        }
        if report.first("tamper").is_none() && self.tamper {
            v.leaks.push("the self-test did not finish".into());
        }
        let expected = self.canaries.len();
        let seen = report.all("canary").len();
        if seen < expected {
            v.leaks.push(format!(
                "the self-test reported {seen} of {expected} canaries"
            ));
        }
        v
    }
}

/// A script (run as root) that tries the daemon's `ports` on the same local
/// addresses as the self-test, and prints `BF canary svc <result> <target>`
/// for each.
pub(super) fn service_probe(ports: &[u16]) -> String {
    Test {
        canaries: ports.iter().map(|p| Canary::Service(*p)).collect(),
        tamper: false,
    }
    .script()
}

/// The targets (`addr:port`) a [`service_probe`] reached.
pub(super) fn services_reached(report: &Report) -> Vec<String> {
    report
        .all("canary")
        .into_iter()
        .filter_map(|l| l.strip_prefix("svc reached "))
        .map(str::to_string)
        .collect()
}

/// A script (run as root) that tries `targets` (`addr:port`, IPv6 in
/// brackets) and prints `BF reached <target>` for each that answers.
pub(super) fn reach_script(targets: &[String]) -> String {
    let mut s = String::from(HELPERS);
    for t in targets {
        let (addr, port) = t.rsplit_once(':').unwrap_or((t, "443"));
        let addr = addr.trim_start_matches('[').trim_end_matches(']');
        s.push_str(&format!(
            "[ \"$(bf_try {a} {p})\" = reached ] && echo \"BF reached \"{t}\n",
            a = quote(addr),
            p = quote(port),
            t = quote(t)
        ));
    }
    s.push_str("true\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow() -> PhaseNet {
        PhaseNet {
            network: Network::Allow(vec!["github.com".into()]),
            names: vec![("github.com".into(), vec!["140.82.114.4".parse().unwrap()])],
            ranges: vec!["140.82.114.4/32".into()],
        }
    }

    #[test]
    fn an_allowlist_tests_its_host_outside_hosts_and_dns() {
        let t = Test::new(&allow(), true, Stage::Agent, &[]);
        assert_eq!(
            t.canaries,
            vec![
                Canary::Allowed("140.82.114.4".into(), 443),
                Canary::Blocked("1.1.1.1".into(), 443),
                Canary::Blocked("8.8.8.8".into(), 53),
                Canary::Blocked("2606:4700:4700::1111".into(), 443),
                Canary::Dns {
                    name: "example.org".into(),
                    strict: true
                },
            ]
        );
        let out = std::process::Command::new("bash")
            .arg("-n")
            .arg("-c")
            .arg(t.script())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_verdict_separates_leaks_from_unreachable_hosts() {
        let t = Test::new(&allow(), true, Stage::Agent, &[]);
        let ok = Report::parse(
            "BF canary allow reached 140.82.114.4:443\nBF canary block refused 1.1.1.1:443\n\
             BF canary block timeout 8.8.8.8:53\nBF canary block no-ipv6 [2606:4700:4700::1111]:443\n\
             BF canary dns unresolved example.org\nBF tamper tried\n",
        );
        let v = t.judge(&ok);
        assert!(v.leaks.is_empty(), "{v:?}");
        assert!(v.unreachable.is_empty(), "{v:?}");
        assert!(
            v.summary()
                .contains("could not reach 1.1.1.1:443, 8.8.8.8:53"),
            "{}",
            v.summary()
        );
        let bad = Report::parse(
            "BF canary allow refused 140.82.114.4:443\nBF canary block reached 1.1.1.1:443\n\
             BF canary block refused 8.8.8.8:53\nBF canary block no-ipv6 [2606:4700:4700::1111]:443\n\
             BF canary dns resolved 93.184.215.14 example.org\nBF tamper tried\n",
        );
        let v = t.judge(&bad);
        assert_eq!(v.unreachable, vec!["140.82.114.4:443"]);
        assert_eq!(v.leaks.len(), 2, "{v:?}");
        // A run that died part way is a leak, not a pass.
        let v = t.judge(&Report::parse("BF canary allow reached 140.82.114.4:443\n"));
        assert!(!v.leaks.is_empty());
    }

    #[test]
    fn the_daemon_ports_must_be_refused_on_every_local_address() {
        let t = Test::new(&allow(), true, Stage::Agent, &[2280]);
        assert!(t.canaries.contains(&Canary::Service(2280)));
        let out = std::process::Command::new("bash")
            .arg("-n")
            .arg("-c")
            .arg(t.script())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let base = "BF canary allow reached 140.82.114.4:443\nBF canary block refused 1.1.1.1:443\n\
             BF canary block refused 8.8.8.8:53\nBF canary block no-ipv6 [2606:4700:4700::1111]:443\n\
             BF canary dns unresolved example.org\nBF tamper tried\n";
        let held = t.judge(&Report::parse(&format!(
            "{base}BF canary svc refused 127.0.0.1:2280\nBF canary svc refused 172.20.0.5:2280\n"
        )));
        assert!(held.leaks.is_empty(), "{held:?}");
        assert!(
            held.summary()
                .contains("could not reach the sandbox daemon at 127.0.0.1:2280, 172.20.0.5:2280")
        );
        let leaked = t.judge(&Report::parse(&format!(
            "{base}BF canary svc refused 127.0.0.1:2280\nBF canary svc reached 172.20.0.5:2280\n"
        )));
        assert_eq!(
            leaked.leaks,
            vec!["the agent user reached the sandbox daemon at 172.20.0.5:2280"]
        );
        assert_eq!(
            held.services_refused,
            vec!["127.0.0.1:2280", "172.20.0.5:2280"]
        );
    }

    #[test]
    fn roots_probe_tries_the_same_addresses_and_reports_what_answered() {
        let s = service_probe(&[2280, 8080]);
        assert!(s.contains("bf_try 127.0.0.1 2280"), "{s}");
        assert!(s.contains("bf_try ::1 8080"), "{s}");
        assert!(!s.contains("BF tamper"), "{s}");
        let out = std::process::Command::new("bash")
            .arg("-n")
            .arg("-c")
            .arg(&s)
            .output()
            .unwrap();
        assert!(out.status.success());
        let r = Report::parse(
            "BF canary svc reached 127.0.0.1:2280\nBF canary svc refused [::1]:2280\nBF canary svc reached 10.0.0.5:8080\n",
        );
        assert_eq!(
            services_reached(&r),
            vec!["127.0.0.1:2280", "10.0.0.5:8080"]
        );
    }

    #[test]
    fn an_open_network_tests_only_the_daemon_ports_on_both_families() {
        let open = PhaseNet {
            network: Network::Open,
            names: Vec::new(),
            ranges: Vec::new(),
        };
        let t = Test::new(&open, true, Stage::Agent, &[2280, 22220]);
        assert_eq!(
            t.canaries,
            vec![Canary::Service(2280), Canary::Service(22220)]
        );
        let script = t.script();
        assert!(script.contains("bf_try ::1 2280"), "{script}");
        assert!(script.contains("[$own6]:22220"), "{script}");
        let out = std::process::Command::new("bash")
            .arg("-n")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let held = t.judge(&Report::parse(
            "BF canary svc refused 127.0.0.1:2280\nBF canary svc refused 10.0.0.5:2280\n\
             BF canary svc refused [::1]:2280\nBF canary svc refused [fd00:0000:0000:0000:0000:0000:0000:0005]:2280\n\
             BF canary svc timeout 127.0.0.1:22220\nBF tamper tried\n",
        ));
        assert!(held.leaks.is_empty(), "{held:?}");
        assert!(held.summary().contains("[::1]:2280"), "{}", held.summary());
        let leaked = t.judge(&Report::parse(
            "BF canary svc refused 127.0.0.1:2280\nBF canary svc reached [::1]:2280\n\
             BF canary svc refused 127.0.0.1:22220\nBF tamper tried\n",
        ));
        assert_eq!(
            leaked.leaks,
            vec!["the agent user reached the sandbox daemon at [::1]:2280"]
        );
        // A self-test that stopped before the tamper check is not a pass.
        let cut = t.judge(&Report::parse("BF canary svc refused 127.0.0.1:2280\n"));
        assert!(!cut.leaks.is_empty());
    }

    #[test]
    fn the_own_addresses_snippet_is_valid_sh() {
        let out = std::process::Command::new("sh")
            .arg("-n")
            .arg("-c")
            .arg(OWN_ADDRESSES)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn a_blocklist_tests_its_addresses_and_pins() {
        let net = PhaseNet {
            network: Network::Block(vec!["www.debian.org".into()]),
            names: vec![(
                "www.debian.org".into(),
                vec!["128.31.0.62".parse().unwrap()],
            )],
            ranges: vec!["128.31.0.62/32".into()],
        };
        let t = Test::new(&net, true, Stage::Agent, &[]);
        assert_eq!(
            t.canaries,
            vec![
                Canary::Blocked("128.31.0.62".into(), 443),
                Canary::Pinned("www.debian.org".into())
            ]
        );
        let v = t.judge(&Report::parse(
            "BF canary block refused 128.31.0.62:443\nBF canary pin resolved 128.31.0.62 www.debian.org\nBF tamper tried\n",
        ));
        assert_eq!(
            v.leaks,
            vec!["the blocked name www.debian.org resolved to 128.31.0.62"]
        );
    }
}
