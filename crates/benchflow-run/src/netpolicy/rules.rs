//! The uid firewall: the shell root runs in the sandbox to install, check,
//! and remove one iptables chain and the names pinned in `/etc/hosts`.
//!
//! Every script prints tagged lines that [`crate::hardening::Report`] reads:
//! `BF fw <how>` once the rules hold, or `BF fw-error <reason>` when they
//! could not be installed, and `BF fw-sum <n>`, a fingerprint of the rules
//! and the pinned names that a later check compares.

use benchflow_sandbox::shell::quote;
use benchflow_task::values::Network;

use super::PhaseNet;
use crate::hardening::STATE_DIR;
use crate::outcome::Owner;

/// The chain BenchFlow owns in the sandbox's `OUTPUT` filter.
const CHAIN: &str = "BENCHFLOW_NET";

/// The address blocked names are pinned to: TEST-NET-1, which routes
/// nowhere, and which the blocklist chain also refuses.
pub(super) const PINNED_NOWHERE: &str = "192.0.2.1";
/// The IPv6 address blocked names are pinned to, in the discard-only
/// prefix (RFC 6666). Without it a lookup of the name's IPv6 addresses
/// falls through to DNS.
pub(super) const PINNED_NOWHERE_V6: &str = "100::1";

/// Whose traffic the chain covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Every packet of the agent user, and every new connection of any
    /// other non-root user (an agent that switches uid stays covered; a
    /// daemon's replies to connections made to it do not).
    AgentUser(String),
    /// Every new connection, root's included.
    Everyone,
}

/// An image layer (POSIX sh, run as root) that installs iptables when the
/// image has none. It never fails the build, and it leaves the image's apt
/// package lists as it found them: a solution may install from them without
/// an update (Terminal-Bench 2's `chess-best-move` does).
pub(super) const INSTALL_LAYER: &str = r#"if ! command -v iptables >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    bf_lists=$(mktemp -d 2>/dev/null || echo /tmp/.bf-apt-lists); mkdir -p "$bf_lists"; cp -a /var/lib/apt/lists/. "$bf_lists"/ 2>/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends iptables >/dev/null 2>&1
    rm -rf /var/lib/apt/lists/* 2>/dev/null; cp -a "$bf_lists"/. /var/lib/apt/lists/ 2>/dev/null; rm -rf "$bf_lists"
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache iptables ip6tables >/dev/null 2>&1
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y iptables-nft >/dev/null 2>&1 || dnf install -y iptables >/dev/null 2>&1
  elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y iptables-nft >/dev/null 2>&1 || microdnf install -y iptables >/dev/null 2>&1
  elif command -v yum >/dev/null 2>&1; then
    yum install -y iptables >/dev/null 2>&1
  fi
fi
true"#;

/// Defines `bf_daemon_ports`, which prints (one per line, ascending) the
/// TCP ports the sandbox daemon listens on: listening sockets in
/// `$P/net/tcp` and `$P/net/tcp6` held by process 1 or by another process
/// running the same program. Run as root, since only root reads another
/// process's open files. `$P` is `/proc` (a test points it elsewhere).
pub(super) const DISCOVER_PORTS: &str = r#"bf_daemon_ports() {
  exe=$(readlink "$P/1/exe" 2>/dev/null) || exe=""
  inodes=" "
  for d in "$P"/[0-9]*; do
    if [ "$d" != "$P/1" ]; then
      [ -n "$exe" ] && [ "$(readlink "$d/exe" 2>/dev/null)" = "$exe" ] || continue
    fi
    for f in "$d"/fd/*; do
      l=$(readlink "$f" 2>/dev/null) || continue
      case $l in socket:\[*\]) i=${l#socket:[}; inodes="$inodes${i%]} ";; esac
    done
  done
  for f in "$P/net/tcp" "$P/net/tcp6"; do
    [ -r "$f" ] || continue
    awk -v want="$inodes" 'FNR > 1 && $4 == "0A" && index(want, " " $10 " ") { n = split($2, a, ":"); print a[n] }' "$f"
  done | while read -r hex; do
    case $hex in *[!0-9A-Fa-f]*|'') continue;; esac
    printf '%d\n' "0x$hex"
  done | sort -un
}
"#;

/// Finds the iptables binaries and the IPv6 state, and defines
/// `fingerprint` and `unhook`.
fn prelude() -> String {
    format!(
        r#"CH={CHAIN}
pick() {{ for c in "$@"; do if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi; done; return 1; }}
fw4=$(pick iptables iptables-nft iptables-legacy) || fw4=""
fw6=$(pick ip6tables ip6tables-nft ip6tables-legacy) || fw6=""
if [ -n "$fw4" ] && ! "$fw4" -w -S OUTPUT >/dev/null 2>&1; then fw4_err=$("$fw4" -w -S OUTPUT 2>&1 | head -n 1); fw4_bad=1; else fw4_bad=0; fi
if [ -n "$fw6" ] && ! "$fw6" -w -S OUTPUT >/dev/null 2>&1; then fw6=""; fi
v6=absent
if [ -e /proc/net/if_inet6 ]; then
  v6=local
  if awk '$4 == "00" && $6 != "lo" {{ f = 1 }} END {{ exit !f }}' /proc/net/if_inet6; then v6=global; fi
fi
unhook() {{
  t=$1
  "$t" -w -S OUTPUT 2>/dev/null | grep -- "-j $CH\$" | sed 's/^-A /-D /' | while read -r r; do eval "\"\$t\" -w $r"; done
  "$t" -w -F "$CH" 2>/dev/null; "$t" -w -X "$CH" 2>/dev/null
  return 0
}}
fingerprint() {{
  {{ [ -n "$fw4" ] && "$fw4" -w -S 2>/dev/null; [ -n "$fw6" ] && "$fw6" -w -S 2>/dev/null
     sed -n '/^# BEGIN benchflow network$/,/^# END benchflow network$/p' /etc/hosts; }} | cksum | cut -d ' ' -f 1
}}
"#
    )
}

/// Rewrite the pinned-names block of `/etc/hosts` in place (Docker mounts
/// the file, so it cannot be replaced).
fn pin_hosts(lines: &str) -> String {
    format!(
        r#"mkdir -p {state}; chmod 700 {state}
sed '/^# BEGIN benchflow network$/,/^# END benchflow network$/d' /etc/hosts > {state}/hosts.tmp
pins={lines}
{{ cat {state}/hosts.tmp; if [ -n "$pins" ]; then printf '# BEGIN benchflow network\n%s# END benchflow network\n' "$pins"; fi; }} > /etc/hosts
rm -f {state}/hosts.tmp
chown root:root /etc/hosts 2>/dev/null; chmod 644 /etc/hosts 2>/dev/null
"#,
        state = quote(STATE_DIR),
        lines = quote(lines),
    )
}

/// The names to pin in `/etc/hosts`: an allowlist's names to the addresses
/// they resolved to (so the agent needs no DNS), a blocklist's to an
/// address that routes nowhere.
fn pins(net: &PhaseNet) -> String {
    let mut out = String::new();
    for (name, addrs) in &net.names {
        match net.network {
            Network::Allow(_) => {
                for a in addrs {
                    out.push_str(&format!("{a} {name}\n"));
                }
            }
            Network::Block(_) => out.push_str(&format!(
                "{PINNED_NOWHERE} {name}\n{PINNED_NOWHERE_V6} {name}\n"
            )),
            _ => {}
        }
    }
    out
}

/// The chain's rules for one address family, as lines of shell that use
/// `$t` (the binary), `$REJ` (the refusal target), `$RT $RTO` (the refusal
/// for TCP: a reset where the kernel has one), and, with `guard`, `$GUARD`
/// (the daemon's ports, found when the script runs).
///
/// TCP is refused with a reset before the general refusal: an ICMP
/// unreachable only fails a TCP connect after a SYN retry, about a second
/// later, and the self-test's twelve refused daemon connections made every
/// Daytona trial's setup 13 s longer.
fn chain_rules(net: &PhaseNet, v6: bool, guard: bool) -> String {
    let ranges: Vec<&String> = net
        .ranges
        .iter()
        .filter(|r| r.contains(':') == v6)
        .collect();
    let mut s = String::new();
    // A root daemon's ports on every local address (local traffic leaves by
    // lo, whichever of the sandbox's addresses it is sent to).
    if guard {
        s.push_str(
            "for port in $GUARD; do \"$t\" -w -A \"$CH\" -o lo -p tcp --dport \"$port\" -j \"$RT\" $RTO || exit_rule; done\n",
        );
    }
    let mut add = |rule: &str| s.push_str(&format!("\"$t\" -w -A \"$CH\" {rule} || exit_rule\n"));
    // `match` refused: TCP with a reset first, then everything else.
    let refuse = |add: &mut dyn FnMut(&str), matching: &str| {
        let sep = if matching.is_empty() { "" } else { " " };
        add(&format!("{matching}{sep}-p tcp -j \"$RT\" $RTO"));
        add(&format!("{matching}{sep}-j \"$REJ\""));
    };
    match &net.network {
        Network::Allow(_) | Network::None => {
            // Resolvers on loopback (Docker's embedded one is 127.0.0.11,
            // reached on a rewritten port) would answer for any name.
            add("-o lo -p udp --dport 53 -j \"$REJ\"");
            add("-o lo -p tcp --dport 53 -j \"$RT\" $RTO");
            if !v6 {
                refuse(&mut add, "-d 127.0.0.11");
            }
            add("-o lo -j RETURN");
            for r in &ranges {
                add(&format!("-d {} -j RETURN", quote(r)));
            }
            refuse(&mut add, "");
        }
        Network::Block(_) => {
            if v6 {
                refuse(&mut add, "-d 100::/64");
            } else {
                refuse(&mut add, &format!("-d {PINNED_NOWHERE}"));
            }
            for r in &ranges {
                refuse(&mut add, &format!("-d {}", quote(r)));
            }
        }
        Network::Open => {}
    }
    s
}

/// The script that installs `net` for `scope`, as root. A non-empty `guard`
/// also refuses the sandbox daemon's ports: those it listens on when the
/// script runs, and `guard` as a floor. The script then prints
/// `BF fw-guard-found <ports, or none>` and `BF fw-guard-ports <ports>`.
pub(super) fn install(scope: &Scope, net: &PhaseNet, guard: &[u16]) -> String {
    let uid = match scope {
        Scope::AgentUser(user) => format!(
            "uid=$(id -u {u} 2>/dev/null) || {{ echo \"BF fw-error no-user {u}\"; exit 0; }}\n",
            u = quote(user)
        ),
        Scope::Everyone => String::new(),
    };
    let jumps = match scope {
        Scope::AgentUser(_) => {
            "\"$t\" -w -I OUTPUT 1 -m owner --uid-owner \"$uid\" -j \"$CH\" || exit_rule\n\
             jump_new 2 -m owner ! --uid-owner 0 || exit_rule\n"
        }
        Scope::Everyone => "jump_new 1 || exit_rule\n",
    };
    format!(
        r#"{prelude}
[ -n "$fw4" ] || {{ echo "BF fw-error no-iptables"; exit 0; }}
[ "$fw4_bad" = 0 ] || {{ echo "BF fw-error no-permission $fw4_err"; exit 0; }}
if [ "$v6" = global ] && [ -z "$fw6" ]; then echo "BF fw-error ipv6-unfiltered"; exit 0; fi
{uid}{discover}exit_rule() {{ echo "BF fw-error rule-failed $t"; [ -n "$fw4" ] && unhook "$fw4"; [ -n "$fw6" ] && unhook "$fw6"; exit 0; }}
setup() {{
  t=$1
  unhook "$t"
  "$t" -w -N "$CH" || exit_rule
  if "$t" -w -A "$CH" -j REJECT 2>/dev/null; then "$t" -w -D "$CH" -j REJECT; REJ=REJECT; else REJ=DROP; fi
  RT=$REJ; RTO=""
  if [ "$REJ" = REJECT ] && "$t" -w -A "$CH" -p tcp -j REJECT --reject-with tcp-reset 2>/dev/null; then
    "$t" -w -D "$CH" -p tcp -j REJECT --reject-with tcp-reset; RTO="--reject-with tcp-reset"
  fi
  if "$t" -w -A "$CH" -m conntrack ! --ctstate ESTABLISHED,RELATED -j RETURN 2>/dev/null; then
    "$t" -w -D "$CH" -m conntrack ! --ctstate ESTABLISHED,RELATED -j RETURN; TRACK=conntrack
  elif "$t" -w -A "$CH" -m state ! --state ESTABLISHED,RELATED -j RETURN 2>/dev/null; then
    "$t" -w -D "$CH" -m state ! --state ESTABLISHED,RELATED -j RETURN; TRACK=state
  else
    echo "BF fw-error no-conntrack"; unhook "$t"; exit 0
  fi
  # A probe's rule whose removal failed would stay at the head of the
  # chain (a reset of every TCP connection): start the chain empty.
  "$t" -w -F "$CH" || exit_rule
}}
jump_new() {{
  pos=$1; shift
  if [ "$TRACK" = conntrack ]; then "$t" -w -I OUTPUT "$pos" "$@" -m conntrack ! --ctstate ESTABLISHED,RELATED -j "$CH"
  else "$t" -w -I OUTPUT "$pos" "$@" -m state ! --state ESTABLISHED,RELATED -j "$CH"; fi
}}
setup "$fw4"
{rules4}{jumps}rej4=$REJ
if [ -n "$fw6" ]; then
  setup "$fw6"
{rules6}{jumps}fi
{gap}{pins}echo "BF fw $fw4 v6=$v6 fw6=${{fw6:-none}} reject=$rej4"
echo "BF fw-sum $(fingerprint)"
"#,
        prelude = prelude(),
        uid = uid,
        rules4 = chain_rules(net, false, !guard.is_empty()),
        rules6 = chain_rules(net, true, !guard.is_empty()),
        discover = if guard.is_empty() {
            String::new()
        } else {
            let floor: Vec<String> = guard.iter().map(u16::to_string).collect();
            format!(
                "P=/proc\n{DISCOVER_PORTS}found=$(bf_daemon_ports | tr '\\n' ' ')\n\
                 GUARD=$(printf '%s\\n' {floor} $found | sort -un | tr '\\n' ' ')\n\
                 echo \"BF fw-guard-found ${{found:-none}}\"\n\
                 echo \"BF fw-guard-ports $GUARD\"\n",
                floor = floor.join(" ")
            )
        },
        jumps = jumps,
        pins = pin_hosts(&pins(net)),
        gap = if guard.is_empty() {
            ""
        } else {
            // IPv6 loopback with no ip6tables: the guard would hold for IPv4 only.
            "if [ \"$v6\" != absent ] && [ -z \"$fw6\" ]; then echo \"BF fw-guard-gap the sandbox has IPv6 but no ip6tables, so the daemon's ports stay open on IPv6\"; fi\n"
        },
    )
}

/// The script that prints the fingerprint of the rules and pinned names.
pub(super) fn fingerprint() -> String {
    format!("{}echo \"BF fw-sum $(fingerprint)\"\n", prelude())
}

/// The script that removes the chain and the pinned names.
pub(super) fn remove() -> String {
    format!(
        "{}[ -n \"$fw4\" ] && unhook \"$fw4\"\n[ -n \"$fw6\" ] && unhook \"$fw6\"\n{}true\n",
        prelude(),
        pin_hosts("")
    )
}

/// What the installed firewall covers, in one clause.
pub(super) fn describe(scope: &Scope, net: &PhaseNet, how: &str) -> String {
    let who = match scope {
        Scope::AgentUser(u) => {
            format!("every packet of `{u}` and every new connection of other non-root users")
        }
        Scope::Everyone => "every new connection, root's included".into(),
    };
    let field = |k: &str| {
        how.split_whitespace()
            .find_map(|w| w.strip_prefix(k).map(str::to_string))
            .unwrap_or_default()
    };
    let v6 = match (field("v6=").as_str(), field("fw6=").as_str()) {
        (_, f) if f != "none" && !f.is_empty() => "IPv6 filtered too",
        ("absent", _) => "no IPv6 in the sandbox",
        _ => "no IPv6 route out of the sandbox",
    };
    let what = match &net.network {
        Network::Allow(_) => format!(
            "only the listed addresses; DNS refused, {} pinned in /etc/hosts",
            names(net.names.len())
        ),
        Network::None => "loopback only; DNS refused".into(),
        Network::Block(_) => format!(
            "the listed addresses refused; {} pinned to {PINNED_NOWHERE} and {PINNED_NOWHERE_V6}",
            names(net.names.len())
        ),
        Network::Open => "nothing".into(),
    };
    format!("iptables firewall on {who}: {what}; {v6}")
}

fn names(n: usize) -> String {
    if n == 1 {
        "1 name".into()
    } else {
        format!("{n} names")
    }
}

/// Whose problem a failed install is, what happened, and what to do.
pub(super) fn explain_error(error: &str, backend: &str) -> (Owner, String, String) {
    let (kind, detail) = error.split_once(' ').unwrap_or((error, ""));
    match kind {
        "no-iptables" => (
            Owner::Task,
            "the task's image has no iptables, and BenchFlow could not install it (the image has no apt-get, apk, dnf, or yum that worked), so the network policy cannot be enforced".into(),
            "add iptables to the task's Dockerfile".into(),
        ),
        "no-permission" => (
            Owner::Sandbox,
            format!("root in the {backend} sandbox cannot change packet filters ({detail}), so the network policy cannot be enforced"),
            "run the task with --sandbox docker, or on a backend that gives root NET_ADMIN".into(),
        ),
        "ipv6-unfiltered" => (
            Owner::Task,
            "the sandbox has an IPv6 address but no ip6tables to filter it, so the network policy cannot be enforced".into(),
            "add ip6tables to the task's Dockerfile (Debian's iptables package has it)".into(),
        ),
        "no-conntrack" => (
            Owner::Sandbox,
            format!("the {backend} sandbox's kernel has no connection tracking for iptables, so the network policy cannot be enforced"),
            "run the task with --sandbox docker".into(),
        ),
        _ => (
            Owner::Benchflow,
            format!("the firewall could not be installed: {error}"),
            "report it with the trial folder".into(),
        ),
    }
}

/// Why the daemon guard could not be installed, in one clause.
pub(super) fn guard_error(error: &str) -> String {
    let (kind, detail) = error.split_once(' ').unwrap_or((error, ""));
    match kind {
        "no-iptables" => "the task's image has no iptables, and BenchFlow could not install it (the image has no apt-get, apk, dnf, or yum that worked)".into(),
        "no-permission" => format!("root in the sandbox cannot change packet filters ({detail})"),
        "ipv6-unfiltered" => "the sandbox has an IPv6 address but no ip6tables".into(),
        "no-conntrack" => "the sandbox's kernel has no connection tracking for iptables".into(),
        _ => format!("the guard could not be installed ({error})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net(network: Network, names: &[(&str, &str)], ranges: &[&str]) -> PhaseNet {
        PhaseNet {
            network,
            names: names
                .iter()
                .map(|(n, a)| (n.to_string(), vec![a.parse().unwrap()]))
                .collect(),
            ranges: ranges.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn bash_ok(script: &str) {
        let out = std::process::Command::new("bash")
            .arg("-n")
            .arg("-c")
            .arg(script)
            .output()
            .expect("bash");
        assert!(
            out.status.success(),
            "not valid bash: {}\n{script}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn scripts_are_valid_shell() {
        let allow = net(
            Network::Allow(vec!["github.com".into()]),
            &[("github.com", "140.82.114.4")],
            &["140.82.114.4/32", "2001:db8::/32"],
        );
        let block = net(
            Network::Block(vec!["www.debian.org".into()]),
            &[("www.debian.org", "128.31.0.62")],
            &["128.31.0.62/32"],
        );
        for n in [&allow, &block, &net(Network::None, &[], &[])] {
            bash_ok(&install(&Scope::AgentUser("agent".into()), n, &[2280]));
            bash_ok(&format!("P=/proc\n{DISCOVER_PORTS}"));
            bash_ok(&install(&Scope::Everyone, n, &[]));
        }
        bash_ok(&fingerprint());
        bash_ok(&remove());
        bash_ok(INSTALL_LAYER);
        // The image's package lists are saved before the update and put back.
        let save = INSTALL_LAYER
            .find("cp -a /var/lib/apt/lists/.")
            .expect("saved");
        let update = INSTALL_LAYER.find("apt-get update").expect("updated");
        let restore = INSTALL_LAYER
            .find("/. /var/lib/apt/lists/")
            .expect("restored");
        assert!(save < update && update < restore, "{INSTALL_LAYER}");
    }

    #[test]
    fn allowlists_return_their_ranges_and_refuse_the_rest() {
        let allow = net(
            Network::Allow(vec!["github.com".into()]),
            &[("github.com", "140.82.114.4")],
            &["140.82.114.4/32", "2001:db8::/32"],
        );
        let v4 = chain_rules(&allow, false, false);
        assert!(v4.contains("-d 127.0.0.11 -p tcp -j \"$RT\" $RTO"), "{v4}");
        assert!(v4.contains("-d 127.0.0.11 -j \"$REJ\""), "{v4}");
        assert!(v4.contains("-d 140.82.114.4/32 -j RETURN"), "{v4}");
        assert!(!v4.contains("2001:db8"), "{v4}");
        assert!(
            v4.contains("\"$CH\" -p tcp -j \"$RT\" $RTO || exit_rule"),
            "{v4}"
        );
        assert!(v4.trim_end().ends_with("-j \"$REJ\" || exit_rule"), "{v4}");
        let v6 = chain_rules(&allow, true, false);
        assert!(v6.contains("-d 2001:db8::/32 -j RETURN"), "{v6}");
        assert_eq!(pins(&allow), "140.82.114.4 github.com\n");
    }

    #[test]
    fn blocklists_refuse_their_ranges_and_pin_names_nowhere() {
        let block = net(
            Network::Block(vec!["www.debian.org".into()]),
            &[("www.debian.org", "128.31.0.62")],
            &["128.31.0.62/32"],
        );
        let v4 = chain_rules(&block, false, false);
        assert!(
            v4.contains("-d 128.31.0.62/32 -p tcp -j \"$RT\" $RTO"),
            "{v4}"
        );
        assert!(v4.contains("-d 128.31.0.62/32 -j \"$REJ\""), "{v4}");
        assert!(!v4.contains("RETURN"), "a blocklist falls through: {v4}");
        assert_eq!(
            pins(&block),
            "192.0.2.1 www.debian.org\n100::1 www.debian.org\n"
        );
        assert!(chain_rules(&block, true, false).contains("-d 100::/64"));
    }

    /// An ICMP unreachable fails a TCP connect only after a SYN retry (about
    /// 1 s; measured 1,066 ms on Daytona against 10 ms for a reset), and the
    /// self-test's twelve refused daemon connections made each Daytona
    /// trial's setup 13 s longer. TCP is refused with a reset, before any
    /// other refusal of the same destination, where the kernel has one.
    #[test]
    fn tcp_is_refused_with_a_reset_first() {
        let s = install(
            &Scope::AgentUser("agent".into()),
            &net(Network::Open, &[], &[]),
            &[2280],
        );
        bash_ok(&s);
        let probe = s
            .find("--reject-with tcp-reset 2>/dev/null")
            .expect("probed");
        assert!(
            s[probe..].contains("RTO=\"--reject-with tcp-reset\""),
            "{s}"
        );
        assert!(s.contains("RT=$REJ; RTO=\"\""), "falls back to $REJ: {s}");
        let block = net(
            Network::Block(vec!["www.debian.org".into()]),
            &[("www.debian.org", "128.31.0.62")],
            &["128.31.0.62/32"],
        );
        for v6 in [false, true] {
            let rules = chain_rules(&block, v6, false);
            let mut tcp_first = 0;
            for (i, line) in rules.lines().enumerate() {
                if line.contains("-j \"$REJ\"") {
                    let prev = rules.lines().nth(i.wrapping_sub(1)).unwrap_or("");
                    assert_eq!(
                        prev.replace(" -p tcp -j \"$RT\" $RTO", " -j \"$REJ\""),
                        line,
                        "{rules}"
                    );
                    tcp_first += 1;
                }
            }
            assert!(tcp_first > 0, "{rules}");
        }
    }

    #[test]
    fn the_guard_alone_refuses_only_the_daemon_ports_on_loopback() {
        let open = net(Network::Open, &[], &[]);
        let ports = [2280, 22220, 22222, 33333];
        for v6 in [false, true] {
            assert_eq!(
                chain_rules(&open, v6, true),
                "for port in $GUARD; do \"$t\" -w -A \"$CH\" -o lo -p tcp --dport \"$port\" -j \"$RT\" $RTO || exit_rule; done\n"
            );
        }
        let s = install(&Scope::AgentUser("agent".into()), &open, &ports);
        bash_ok(&s);
        // The known ports are the floor; what the daemon listens on is added.
        assert!(
            s.contains("printf '%s\\n' 2280 22220 22222 33333 $found"),
            "{s}"
        );
        assert!(s.contains("BF fw-guard-ports $GUARD"), "{s}");
        assert!(s.contains("BF fw-guard-gap"), "{s}");
        assert!(s.contains("--uid-owner \"$uid\""), "{s}");
        // Without a guard there is no gap to report.
        let s = install(&Scope::Everyone, &net(Network::None, &[], &[]), &[]);
        assert!(!s.contains("fw-guard-gap"), "{s}");
    }

    /// A fake `/proc`: the daemon (process 1) and a second process of the
    /// same program hold listening sockets; a task's own service and a
    /// connected socket of the daemon do not count.
    #[cfg(unix)]
    #[test]
    fn the_daemon_ports_are_read_from_its_own_listening_sockets() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let p = d.path();
        let proc_ = |pid: &str, exe: &str, fds: &[(&str, &str)]| {
            std::fs::create_dir_all(p.join(pid).join("fd")).unwrap();
            symlink(exe, p.join(pid).join("exe")).unwrap();
            for (fd, target) in fds {
                symlink(target, p.join(pid).join("fd").join(fd)).unwrap();
            }
        };
        proc_(
            "1",
            "/usr/local/bin/daytona",
            &[
                ("8", "socket:[100]"),
                ("9", "socket:[101]"),
                ("10", "socket:[102]"),
                ("3", "/dev/null"),
            ],
        );
        proc_("77", "/usr/local/bin/daytona", &[("4", "socket:[103]")]);
        proc_("99", "/usr/bin/python3", &[("5", "socket:[104]")]);
        std::fs::create_dir_all(p.join("net")).unwrap();
        let head = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";
        let row = |sl: u32, port: &str, st: &str, inode: u32| {
            format!(
                "   {sl}: 00000000000000000000000000000000:{port} 00000000000000000000000000000000:0000 {st} 00000000:00000000 00:00000000 00000000     0        0 {inode} 1 0000000000000000 100 0 0 10 0\n"
            )
        };
        std::fs::write(
            p.join("net/tcp6"),
            format!(
                "{head}{}{}{}{}{}",
                row(0, "8235", "0A", 100), // 33333, the daemon
                row(1, "08E8", "0A", 101), // 2280
                row(2, "1F90", "01", 102), // 8080, connected, not listening
                row(3, "2710", "0A", 103), // 10000, a second daemon process
                row(4, "1538", "0A", 104), // 5432, the task's own service
            ),
        )
        .unwrap();
        std::fs::write(p.join("net/tcp"), head).unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!(
                "P={}\n{DISCOVER_PORTS}bf_daemon_ports",
                p.display()
            ))
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "2280\n10000\n33333\n",
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_agent_scope_covers_other_uids_new_connections_too() {
        let s = install(
            &Scope::AgentUser("agent".into()),
            &net(Network::None, &[], &[]),
            &[2280],
        );
        assert!(s.contains("--uid-owner \"$uid\""), "{s}");
        assert!(s.contains("-o lo -p tcp --dport \"$port\" -j"), "{s}");
        assert!(s.contains("printf '%s\\n' 2280 $found"), "{s}");
        assert!(s.contains("jump_new 2 -m owner ! --uid-owner 0"), "{s}");
        let s = install(&Scope::Everyone, &net(Network::None, &[], &[]), &[]);
        assert!(!s.contains("--uid-owner"), "{s}");
    }
}
