"""The verifier's network, enforced for root's test.sh.

test.sh runs as root and runs code the agent wrote: a solution module a test
imports, a hook the agent left. Before, the task's network policy bound the
agent's uid only, and root kept the container's network: under ``allowlist``,
``denylist`` and an offline task run by a model agent (whose container stays
online for the agent's model), root reached any host, and under the policies
that give ``main`` NET_ADMIN it could flush the rules.

Now the verifier runs under the task's own verifier network setting:
``verifier.network_mode`` when the task sets it, else the sandbox's policy
(``sandbox.network_mode``; ``allow_internet = false`` is ``no-network``). An
``[agent]`` override binds the agent alone, as documented
(:func:`benchflow.sandbox.egress_denylist.agent_network_sandbox_config`).

- **Who is held.** Root's test.sh starts through :func:`verifier_launch`:
  util-linux ``setpriv`` (else BenchFlow's kit Python) takes NET_ADMIN,
  NET_RAW and SYS_PTRACE out of its bounding set and sets no new
  privileges, so neither it nor anything it starts can change the rules,
  send around them or reach into the egress proxy. SETUID and SETGID stay
  (apt drops to ``_apt``): the rules match every process, so changing user
  or group escapes nothing.
- **The rules** (:func:`everyone_rules`), on every process of the namespace:
  out go only the egress proxy's own sockets (marked with
  :data:`benchflow.sandbox.egress_denylist.EGRESS_MARK`, which needs
  NET_ADMIN to set), loopback, replies, the sandbox's own networks but never
  their gateway (the host), and an allowlist's IP entries; an allowlist's DNS
  goes to the proxy's filter. ``no-network`` has no proxy; ``allowlist`` and
  ``denylist`` start it again and give test.sh its variables (the proxy then
  judges every address it connects to). ``public``: none (the cloud metadata
  endpoints are refused by
  :func:`benchflow.sandbox.lockdown.cover_cloud_metadata_for_verifier`). A
  killed or stopped proxy fails closed: nothing unmarked leaves, and a new
  process cannot mark. A container with no route out needs no rule.
- **The proof**, before test.sh: through the same launch, a TCP connect to
  a TEST-NET-1 address is refused and the rule's counter counts it (or the
  kernel answers that the destination is unreachable), an ``iptables -D`` and
  an ``iptables -F`` fail, and the launch's own check finds none of the three
  capabilities.
- **Every run's evidence.** The launch writes test.sh's ``CapBnd``,
  ``CapEff`` and ``NoNewPrivs`` to a record beside its start receipt (outside
  ``/logs/verifier``) before test.sh starts, and refuses to start it with
  NET_ADMIN, NET_RAW or SYS_PTRACE; the verifier refuses a run whose record
  is missing (a path that skipped the launch) and copies the record's line
  into ``verifier/verifier-network.json``. After test.sh the refused packets are counted: a failed test.sh
  with refusals ends "verifier network refused", unscored; a passing one keeps
  its reward with the count in ``verifier/verifier-network.json``.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: The launch's exit status when it refuses to start test.sh.
VERIFIER_LAUNCH_REFUSED = 97
#: The trial folder's record of the verifier's network (``verifier/``).
VERIFIER_NETWORK_RECORD = "verifier-network.json"
#: The opening words of the ending a failed test.sh gets when the verifier's
#: network refused it something (benchflow._utils.scoring).
VERIFIER_NETWORK_REFUSED = "verifier network refused: "
#: The opening words of a run whose test.sh did not start through the launch.
VERIFIER_LAUNCH_REFUSED_ERROR = "verifier launch refused: "
#: The opening words of a run whose verifier's network could not be held
#: (rules not added or not proven): test.sh did not run open instead.
VERIFIER_NETWORK_UNHELD = "verifier network not held: "
#: Docker's embedded resolver, on main's loopback: dockerd answers it from
#: outside the namespace and forwards what it cannot answer to the host's
#: resolvers, so under no-network it would resolve any name and carry data
#: out in the names asked. Refused above the loopback
#: ACCEPT, and counted with the rules' own refusals (its comment).
DOCKER_DNS_REFUSAL = "-d 127.0.0.11 -m comment --comment bf-docker-dns -j REJECT"
#: Where the proof connects: TEST-NET-1 (RFC 5737), never routed anywhere.
_PROOF_ADDRESS = "192.0.2.1"
_SETPRIV = ("/usr/bin/setpriv", "/bin/setpriv", "/usr/sbin/setpriv", "/sbin/setpriv")


def verifier_network_mode(task_config: Any) -> str:
    """The verifier's network: ``verifier.network_mode`` when the task sets it,
    else the sandbox's (``allow_internet = false`` is ``no-network``)."""
    from benchflow.task.config import NetworkMode

    mode = getattr(getattr(task_config, "verifier", None), "network_mode", None)
    if isinstance(mode, NetworkMode):
        return mode.value
    sandbox = getattr(task_config, "sandbox", None)
    if getattr(sandbox, "allow_internet", True) is False:
        return NetworkMode.NO_NETWORK.value
    mode = getattr(sandbox, "network_mode", None)
    return mode.value if isinstance(mode, NetworkMode) else NetworkMode.PUBLIC.value


def separate_verifier_network(task_config: Any) -> str:
    """The network a separate verifier sandbox declares for itself
    (``[verifier.sandbox]``), else the verifier's (:func:`verifier_network_mode`)."""
    from benchflow.task.config import NetworkMode

    sandbox = getattr(getattr(task_config, "verifier", None), "sandbox", None)
    if sandbox is not None:
        if getattr(sandbox, "allow_internet", True) is False:
            return NetworkMode.NO_NETWORK.value
        mode = getattr(sandbox, "network_mode", None)
        if isinstance(mode, NetworkMode):
            return mode.value
    return verifier_network_mode(task_config)


def network_refusal_ends(refused: int, exit_code: int | None, reward: Any) -> bool:
    """Whether test.sh's run ends "verifier network refused": the verifier's network refused it packets and
    the result is no pass, a non-zero exit or a reward below a full pass. A
    pass despite refusals keeps its reward (the count is recorded)."""
    full = (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and reward >= 1
    )
    return refused > 0 and (exit_code not in (0, None) or not full)


def _verifier_policy_config(task_config: Any, mode: str) -> Any:
    """The sandbox config whose egress policy is the verifier's."""
    sandbox = task_config.sandbox
    verifier = task_config.verifier
    if getattr(verifier, "network_mode", None) is None:
        return sandbox
    folded = sandbox.model_copy(deep=True)
    folded.network_mode = verifier.network_mode
    folded.allowed_hosts = (
        list(verifier.allowed_hosts or []) if mode == "allowlist" else []
    )
    folded.blocked_urls = []
    folded.blocked_hosts = []
    return folded


#: The capabilities test.sh never holds (bits): NET_ADMIN (12) and NET_RAW (13),
#: which change or go around the rules, and SYS_PTRACE (19), which would let it
#: reach into the egress proxy. SETUID and SETGID stay (apt drops to _apt):
#: the rules match every process, so a uid or group change escapes nothing.
_HELD_CAPS = (12, 13, 19)
_HELD_MASK = sum(1 << c for c in _HELD_CAPS)


def _caps_check(record: str) -> str:
    """The launch's own check, in test.sh's shell before test.sh: write the
    record, then refuse (exit 97) with NET_ADMIN, NET_RAW or SYS_PTRACE in
    CapEff, or in CapBnd where a gain could follow (root, or no
    no_new_privs). Builtins only."""
    return (
        # Read again while a field came back empty (1 launch in
        # 603 read an empty CapBnd and refused; its re-runs passed). A field
        # still empty after five reads refuses below, as before.
        "bf_try=0; while [ $bf_try -lt 5 ]; do "
        "bf_uid=; bf_gid=; bf_bnd=; bf_eff=; bf_nnp=; "
        "while read -r bf_k bf_v bf_rest; do case $bf_k in "
        "Uid:) bf_uid=$bf_v;; Gid:) bf_gid=$bf_v;; CapBnd:) bf_bnd=$bf_v;; "
        "CapEff:) bf_eff=$bf_v;; NoNewPrivs:) bf_nnp=$bf_v;; esac; "
        "done < /proc/self/status; "
        '[ -n "$bf_bnd" ] && [ -n "$bf_eff" ] && [ -n "$bf_nnp" ] && break; '
        "bf_try=$((bf_try + 1)); done; "
        "bf_net() { case $1 in ''|*[!0-9a-fA-F]*) echo 1;; "
        f"*) echo $(( 0x$1 & {_HELD_MASK} ));; esac; }}; "
        "printf 'BF verifier-launch uid=%s gid=%s CapBnd=%s CapEff=%s "
        'NoNewPrivs=%s\\n\' "$bf_uid" "$bf_gid" "$bf_bnd" "$bf_eff" "$bf_nnp" '
        f"> {shlex.quote(record)} 2>/dev/null; "
        'if [ "$(bf_net "$bf_eff")" != 0 ] || { [ "$(bf_net "$bf_bnd")" != 0 ] '
        '&& { [ "$bf_uid" = 0 ] || [ "$bf_nnp" != 1 ]; }; }; then '
        "echo 'BF verifier-launch refused: test.sh would run with NET_ADMIN, "
        "NET_RAW or SYS_PTRACE' >&2; "
        f"exit {VERIFIER_LAUNCH_REFUSED}; fi; "
        "unset bf_uid bf_gid bf_bnd bf_eff bf_nnp bf_k bf_v bf_rest bf_try; unset -f bf_net; "
    )


# The same drop as setpriv's, from BenchFlow's own Python where the image has
# no util-linux setpriv (BusyBox's cannot drop bounding capabilities):
# NET_ADMIN, NET_RAW and SYS_PTRACE out of the bounding set, no inheritable or
# ambient capabilities, no new privileges, then bash.
_DROP_PY = (
    "import ctypes,os,sys;l=ctypes.CDLL(None,use_errno=True);"
    "r=os.getuid()==0;"
    "[sys.exit(97) for c in ((12,13,19) if r else ()) if l.prctl(24,c,0,0,0)];"
    "l.prctl(47,4,0,0,0);"
    "h=(ctypes.c_uint32*2)(0x20080522,0);d=(ctypes.c_uint32*6)();"
    "l.capget(h,d);d[2]=d[5]=0;l.capset(h,d) and sys.exit(97);"
    "l.prctl(38,1,0,0,0) and sys.exit(97);"
    "b=os.access('/bin/bash',os.X_OK);"
    "os.execv(*(('/bin/bash',['/bin/bash','--noprofile','--norc','-c',sys.argv[1]])"
    "if b else ('/bin/sh',['/bin/sh','-c',sys.argv[1]])))"
)


def verifier_launch(command: str, *, record: str = "/dev/null") -> str:
    """*command* (test.sh's, as the verifier builds it) as root starts it: in
    a subshell, through util-linux ``setpriv`` found in the system folders
    (never on the image's PATH), with no new privileges and NET_ADMIN,
    NET_RAW and SYS_PTRACE out of its bounding set (SETUID and SETGID stay:
    apt drops to ``_apt``); a non-root verifier with no new privileges. Without such
    a setpriv, or with one that cannot make the drop (tried first on a no-op:
    an old libcap-ng's refuses "all" caps), BenchFlow's protected kit
    Python makes the same drop (:data:`_DROP_PY`). The shell it starts (bash, or sh in an image without
    bash) checks itself first (:func:`_caps_check`). With neither the launch
    refuses (exit 97): test.sh never runs unheld."""
    from benchflow._runtime_tree import KIT_PYTHON

    inner = shlex.quote(_caps_check(record) + command)
    candidates = " ".join(_SETPRIV)
    kit = shlex.quote(KIT_PYTHON)
    drop = (
        "--no-new-privs --bounding-set=-net_admin,-net_raw,-sys_ptrace "
        "--inh-caps=-all --ambient-caps=-all"
    )
    return (
        # The shell text test.sh's shell runs, once: every way below starts it.
        f"( bf_in={inner}; "
        # bash, or sh in an image without bash (a POSIX test.sh).
        "if [ -x /bin/bash ]; then set -- /bin/bash --noprofile --norc; "
        "else set -- /bin/sh; fi; "
        "bf_uid=; while read -r bf_k bf_v bf_rest; do "
        '[ "$bf_k" = Uid: ] && bf_uid=$bf_v; done < /proc/self/status; '
        f"bf_sp=; for bf_c in {candidates}; do "
        '[ -x "$bf_c" ] && { bf_sp=$bf_c; break; }; done; '
        'bf_help=; [ -n "$bf_sp" ] && bf_help=$("$bf_sp" --help 2>&1); '
        "case $bf_help in *bounding-set*) ;; *) "
        f"[ -x {kit} ] && exec {kit} -I -S -B -c {shlex.quote(_DROP_PY)} "
        '"$bf_in"; '
        "echo 'BF verifier-launch refused: test.sh needs util-linux "
        "setpriv (with --bounding-set) or the BenchFlow kit Python to start without "
        "NET_ADMIN and NET_RAW' >&2; "
        f"exit {VERIFIER_LAUNCH_REFUSED};; esac; "
        # The image's setpriv must make the exact drop, not only list it in
        # --help: an old libcap-ng's prints "libcap-ng is too old for "all"
        # caps" and exits, so test.sh never started and the trial was refused
        # The same invocation is tried first on a no-op; where it
        # fails the kit Python makes the drop. Never a refusal for that.
        'if [ "$bf_uid" = 0 ]; then '
        f'if "$bf_sp" {drop} -- /bin/sh -c : >/dev/null 2>&1; then '
        f'exec "$bf_sp" {drop} -- "$@" -c "$bf_in"; fi; '
        'elif "$bf_sp" --no-new-privs -- /bin/sh -c : >/dev/null 2>&1; then '
        'exec "$bf_sp" --no-new-privs -- "$@" -c "$bf_in"; fi; '
        f"[ -x {kit} ] && exec {kit} -I -S -B -c {shlex.quote(_DROP_PY)} "
        '"$bf_in"; '
        "echo 'BF verifier-launch refused: the image setpriv cannot make the "
        "drop, and there is no BenchFlow kit Python to make it' >&2; "
        f"exit {VERIFIER_LAUNCH_REFUSED} )"
    )


def launch_record_problem(text: str | None) -> str | None:
    """Why the launch's record shows test.sh not held, or None when it was."""
    if not text:
        return "test.sh left no launch record: it did not start through BenchFlow's bounded launch"
    m = re.search(
        r"BF verifier-launch uid=(\d*) gid=(\d*) CapBnd=([0-9a-fA-F]*) "
        r"CapEff=([0-9a-fA-F]*) NoNewPrivs=(\d*)",
        text,
    )
    if not m:
        return "test.sh's launch record is unreadable"
    uid, _, bnd, eff, nnp = m.groups()

    def held(value: str) -> int:
        return int(value, 16) & _HELD_MASK if value else 1

    if held(eff) or (held(bnd) and (uid == "0" or nnp != "1")):
        return f"test.sh ran with NET_ADMIN, NET_RAW or SYS_PTRACE (CapBnd={bnd} CapEff={eff})"
    return None


@dataclass
class VerifierNetwork:
    """What hardening armed for test.sh (kept on the sandbox)."""

    mode: str
    state: str
    match: str | None = None
    iptables: str | None = None
    baseline: int = 0
    env: dict[str, str] = field(default_factory=dict)
    #: ``in-container`` (test.sh in the agent's sandbox, held by the launch and
    #: the rules) or ``separate sandbox`` (its own declared network, no rule).
    where: str = "in-container"
    #: Whether test.sh starts through :func:`verifier_launch` (and its record
    #: is required).
    launch: bool = True
    subnets: tuple[str, ...] = ()
    proof: dict[str, Any] = field(default_factory=dict)
    #: The rules are in main's namespace from BenchFlow's helper (network:
    #: none, whose main holds no NET_ADMIN), and so is their counter.
    via_helper: bool = False
    #: test.sh runs in this compose service's container (``verifier.service``):
    #: its rules and counter are there, from a helper joined to it.
    container: str | None = None
    #: The verifier phase's egress proxy: ``own container`` (Docker; out of
    #: test.sh's pid namespace and filesystem, benchflow.sandbox._egress_container)
    #: or ``in sandbox`` (a provider's); None without one.
    proxy: str | None = None
    #: The in-sandbox proxy's refusals logged before test.sh started.
    proxy_baseline: int = 0
    #: The last count (:func:`refused_since_arm`): packets the rules refused,
    #: requests the proxy refused, and the first of those (method, url, rule).
    refused_packets: int = 0
    #: The packets main's cloud metadata rules refused before the verifier
    #: (the agent's turn: its processes, root's included), read from the
    #: helper at the arm; None where no helper reads them.
    metadata_refused_before: int | None = None
    proxy_refused: int = 0
    proxy_refusals: list[dict[str, Any]] = field(default_factory=list)

    def record(self, refused: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "where": self.where,
            "mode": self.mode,
            "state": self.state,
        }
        if self.metadata_refused_before is not None:
            out["main_metadata_refused_before_verifier"] = self.metadata_refused_before
        if refused is not None and self.proxy is None:
            out["refused_packets"] = refused
        elif refused is not None:
            out.update(
                refused=refused,
                refused_packets=self.refused_packets,
                proxy=self.proxy,
                proxy_refused=self.proxy_refused,
            )
            if self.proxy_refusals:
                out["proxy_refusals"] = self.proxy_refusals
        return out

    def refusal_words(self, refused: int) -> str:
        """*refused* in words for the ending's message."""
        if self.proxy is None:
            return f"{refused} packets"
        return (
            f"{refused} times ({self.refused_packets} packets at its rules, "
            f"{self.proxy_refused} requests at its egress proxy)"
        )


def everyone_rules(
    mode: str,
    *,
    subnets: tuple[str, ...] = (),
    gateways: tuple[str, ...] = (),
    allow_networks: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str]]:
    """The verifier phase's rules, on every process of the namespace, top of
    OUTPUT first (IPv4 filter, IPv6 filter, IPv4 nat). Out go only: the egress
    proxy's own marked sockets, loopback, replies to connections made to the
    sandbox, the networks the sandbox is on (its task services) but never
    their gateway (the host), and an allowlist's IP entries; the last rule
    refuses the rest and counts it for the ending. An allowlist's DNS goes to
    the proxy's filter, whatever the process asked."""
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6
    from benchflow.sandbox.egress_denylist import DNS_PORT, EGRESS_MARK

    mark = f"-m mark --mark {EGRESS_MARK}"
    # First: the cloud metadata endpoints and link-local, for every process,
    # whatever the rules below admit (a route to 169.254.169.254 is a
    # sandbox network too). DNS (53) stays as the cover rule leaves it.
    v4: list[str] = [
        f"-d {a} -p tcp ! --dport 53 -j REJECT --reject-with tcp-reset"
        for a in (*CLOUD_METADATA_V4, "169.254.0.0/16")
    ]
    v6: list[str] = [
        f"-d {a} -p tcp ! --dport 53 -j REJECT --reject-with tcp-reset"
        for a in CLOUD_METADATA_V6
    ]
    v4.append(f"{mark} -j ACCEPT")
    v6.append(f"{mark} -j ACCEPT")
    for gw in gateways:
        addr = ipaddress.ip_address(gw)
        (v4 if addr.version == 4 else v6).append(f"-d {addr} -j REJECT")
    if mode == "no-network":
        # Names too: Docker's resolver answers from outside the namespace.
        v4.append(DOCKER_DNS_REFUSAL)
    for table in (v4, v6):
        table.append("-o lo -j ACCEPT")
        table.append("-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")
    nets = [ipaddress.ip_network(n, strict=False) for n in subnets]
    if mode == "allowlist":
        nets += [ipaddress.ip_network(n, strict=False) for n in allow_networks]
    for net in nets:
        (v4 if net.version == 4 else v6).append(f"-d {net.compressed} -j ACCEPT")
    if mode == "denylist":
        # A denylist lists what is blocked: names resolve as before.
        for table in (v4, v6):
            table += [f"-p {p} --dport 53 -j ACCEPT" for p in ("udp", "tcp")]
    v4.append("-j REJECT")
    v6.append("-j REJECT")
    nat: list[str] = []
    if mode == "allowlist":
        nat = [
            f"-p {p} --dport 53 -m mark ! --mark {EGRESS_MARK} -j REDIRECT --to-ports {DNS_PORT}"
            for p in ("udp", "tcp")
        ]
    return v4, v6, nat


def rules_cmd(
    v4: list[str], v6: list[str], nat: list[str] | None = None, *, status: str = "0"
) -> str:
    """Root's script: add each rule once, in order (the last one at the end,
    where it counts what the others let pass), then print the counter.
    *status* is the shell value the report starts from."""

    def adds(tool: str, table: str, rules: list[str]) -> str:
        out = ""
        for position, rule in enumerate(rules, start=1):
            q = shlex.quote(rule)
            last = (
                position == len(rules)
                and rule.endswith("-j REJECT")
                and table == "filter"
            )
            add = "-A OUTPUT $r" if last else f"-I OUTPUT {position} $r"
            out += (
                f"r={q}; $T4 -w -t {table} -C OUTPUT $r 2>/dev/null || "
                f"$T4 -w -t {table} {add} || status=1; "
            ).replace("$T4", tool)
        return out

    return (
        f"set +e; status={status}; "
        "T=; for c in iptables iptables-nft iptables-legacy; do command -v $c >/dev/null 2>&1 "
        "&& $c -w -S OUTPUT >/dev/null 2>&1 && { T=$c; break; }; done; "
        '[ -n "$T" ] || { echo "BF verifier-network status=2"; exit 0; }; '
        "T6=; for c in ip6tables ip6tables-nft ip6tables-legacy; do command -v $c >/dev/null 2>&1 "
        "&& $c -w -S OUTPUT >/dev/null 2>&1 && { T6=$c; break; }; done; "
        + adds('"$T"', "filter", v4)
        + adds('"$T"', "nat", list(nat or []))
        + 'if [ -n "$T6" ]; then '
        + adds('"$T6"', "filter", v6)
        + "elif [ -e /proc/net/if_inet6 ]; then status=3; fi; "
        + 'echo "BF verifier-network status=$status"; '
        + count_cmd()
    )


def count_cmd() -> str:
    """Root: the packets the last rule refused (every process), and those
    Docker's resolver refusal refused (:data:`DOCKER_DNS_REFUSAL`). Its line
    carries a ``status=`` (1: no iptables answered), which a fresh run from
    BenchFlow's helper reads as its report (``_metadata_helper._status``)."""
    return (
        "s=1; c=0; for t in iptables iptables-nft iptables-legacy; do command -v $t >/dev/null 2>&1 || continue; "
        "o=$($t -w -S OUTPUT -v 2>/dev/null) || continue; s=0; "
        "c=$(printf '%s\\n' \"$o\" | awk '/^-A OUTPUT -c [0-9]+ [0-9]+ -j REJECT/ "
        '{ n += $4 } /^-A OUTPUT .*--comment "?bf-docker-dns"? -c [0-9]+ [0-9]+ -j REJECT/ '
        '{ for (i = 1; i < NF; i++) if ($i == "-c") { n += $(i + 1); break } } '
        "END { print n + 0 }'); break; done; "
        'echo "BF verifier-network refused ${c:-0} status=$s"'
    )


def _route_facts_cmd() -> str:
    """Root: main's routes, from /proc (builtins only): ``net <hex dest> <hex
    mask>`` for each connected IPv4 network and ``gw <hex>`` for each gateway,
    then ``BF verifier-network routes <n>`` (routes out of loopback)."""
    return (
        'n=0; while read -r i d g f rc u m mk rest; do [ "$i" = Iface ] && continue; '
        '[ "$i" = lo ] && continue; n=$((n + 1)); '
        'if [ "$d" = 00000000 ]; then echo "gw $g"; else echo "net $d $mk"; fi; '
        '[ "$g" != 00000000 ] && echo "gw $g"; '
        "done < /proc/net/route; "
        "[ -r /proc/net/ipv6_route ] && while read -r a b c d e f g h i dev; do "
        '[ "$dev" != lo ] && n=$((n + 1)); done < /proc/net/ipv6_route; '
        'echo "BF verifier-network routes $n"'
    )


def _no_route_cmd() -> str:
    return _route_facts_cmd()


#: The interface a main with no route out gets for the verifier phase: a
#: dummy (it carries nothing anywhere), so a connect has a route, meets the
#: counted REJECT and is refused at once. A route to loopback counted the
#: packets but never brought the refusal back (the connect hung).
NO_ROUTE_LINK = "bf-noroute"
NO_ROUTE_ADDRESS = "198.18.0.1"


def _no_route_rules_cmd() -> str:
    """The helper's script for a main with no route out (network: none):
    every destination but loopback goes to a dummy interface, where a
    counted REJECT refuses it, so each attempt to reach out (a DNS query
    included) counts, and a grader that never tries the network counts
    nothing. Status 4: the dummy route could not be made."""
    link, address = NO_ROUTE_LINK, NO_ROUTE_ADDRESS
    return (
        "r=0; "
        f"ip link show {link} >/dev/null 2>&1 || ip link add {link} type dummy || r=4; "
        f"ip link set {link} up || r=4; "
        f"ip addr replace {address}/32 dev {link} || r=4; "
        f"ip route replace default dev {link} src {address} || r=4; "
        f"if [ -e /proc/net/if_inet6 ]; then ip -6 route replace default dev {link} 2>/dev/null || true; fi; "
        + rules_cmd(*NO_ROUTE_RULES, status="$r")
    )


#: The no-route counting rules (IPv4, IPv6), as :func:`_no_route_rules_cmd`
#: adds them.
NO_ROUTE_RULES = (
    [DOCKER_DNS_REFUSAL, "-o lo -j ACCEPT", "-d 127.0.0.0/8 -j ACCEPT", "-j REJECT"],
    ["-o lo -j ACCEPT", "-d ::1/128 -j ACCEPT", "-j REJECT"],
)


async def _via_helper(env: Any, command: str) -> str:
    """*command* from BenchFlow's helper in main's namespace (a fresh run)."""
    import secrets

    outcome = await env.metadata_rules_install(f": {secrets.token_hex(4)}; {command}")
    if outcome is None:
        return ""
    result, failed = outcome
    out = getattr(result, "stdout", "") or ""
    return "" if failed or not isinstance(out, str) else out


def _ledger(env: Any, phase: str, v4: list[str], v6: list[str], nat: list[str]) -> None:
    """Record main's rules from its own helper (``metadata_rules_install``)
    in the rule ledger, for their re-proof (benchflow.sandbox.rule_ledger)."""
    from benchflow.sandbox import rule_ledger

    rule_ledger.record(
        env,
        rule_ledger.InstalledRules(where="main", phase=phase, v4=v4, v6=v6, nat=nat),
    )


async def install_from_helper(env: Any, script: str, *, phase: str) -> Any:
    """Root's rule *script* for main's namespace from BenchFlow's helper (a
    fresh run; main holds no NET_ADMIN), the rules it added recorded in the
    rule ledger under *phase*. Raises RuntimeError naming the script's exit
    status when it failed or the helper could not run it."""
    from benchflow.sandbox import rule_ledger

    out = await _via_helper(env, rule_ledger.capture_cmd(script))
    found = re.search(r"BF rules-installed status=(\d+)", out)
    if not found or found[1] != "0":
        raise RuntimeError(
            "BenchFlow's helper could not add the rules to main's namespace "
            f"(status {found[1] if found else 'none'})"
        )
    v4, v6, nat = rule_ledger.added_rules(out)
    entry = rule_ledger.InstalledRules(where="main", phase=phase, v4=v4, v6=v6, nat=nat)
    rule_ledger.record(env, entry)
    return entry


def _has_helper(env: Any) -> bool:
    return callable(getattr(type(env), "metadata_rules_install", None))


async def _rules_root(env: Any, command: str, timeout: int = 300) -> tuple[str, bool]:
    """Root's rules *command* for main's namespace and whether the helper ran
    it: from BenchFlow's helper where the sandbox has one (its own iptables
    and NET_ADMIN, which main's image or main may lack: FXPB's ProgramBench
    images), else, or when the helper cannot add them, as root in main. The
    same place later counts and removes them."""
    from benchflow.sandbox._exec_output import exec_raw

    if _has_helper(env):
        out = await _via_helper(env, command)
        if out:
            return out, True
    result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout)
    out = getattr(result, "stdout", "") or ""
    return (out if isinstance(out, str) else ""), False


async def _install_recorded(
    env: Any, v4: list[str], v6: list[str], nat: list[str] | None = None
) -> tuple[str, bool, tuple[list[str], list[str], list[str]]]:
    """:func:`rules_cmd` for main's namespace, with the rules it actually
    added: from the helper the install is captured (rule_ledger.capture_cmd),
    so a spec another phase already holds (main's own metadata rule, from
    sandbox start: ``-C || -I`` adds no second copy) is neither recorded nor
    removed later. Without the helper, root in main, and
    the intended specs. Returns (output, via_helper, (v4, v6, nat) added)."""
    from benchflow.sandbox import rule_ledger
    from benchflow.sandbox._exec_output import exec_raw

    if _has_helper(env):
        out = await _via_helper(env, rule_ledger.capture_cmd(rules_cmd(v4, v6, nat)))
        if out:
            return out, True, rule_ledger.added_rules(out)
    result = await exec_raw(
        env.exec, rules_cmd(v4, v6, nat), user="root", timeout_sec=300
    )
    text = getattr(result, "stdout", "") or ""
    return (
        (text if isinstance(text, str) else ""),
        False,
        (list(v4), list(v6), list(nat or [])),
    )


async def _count_root(env: Any, via_helper: bool) -> str:
    from benchflow.sandbox._exec_output import exec_raw

    if via_helper:
        return await _via_helper(env, count_cmd())
    result = await exec_raw(env.exec, count_cmd(), user="root", timeout_sec=60)
    out = getattr(result, "stdout", "") or ""
    return out if isinstance(out, str) else ""


async def _shell_or_helper_connect(env: Any, out: str) -> tuple[str, bool]:
    """*out* (a proof's output), its connect made again from BenchFlow's
    helper as uid 0 when main's shell could not make it (an image without
    bash, whose sh has no /dev/tcp: Alpine, BusyBox). The rules hold every
    process of main's namespace, the helper's too, so its connect is the
    same proof. Whether the helper made it."""
    found = re.search(r"BF verifier-proof connect=(\w+)", out)
    if found and found[1] != "failed":
        return out, False
    run = getattr(env, "metadata_rules_run", None)
    if not _has_helper(env) or not callable(run):
        return out, False
    from benchflow.sandbox.service_cover import _connect, connect_cmd

    ran, failed = await run(connect_cmd(_PROOF_ADDRESS), timeout_sec=30, user="0:0")
    state = "" if failed else _connect(str(getattr(ran, "stdout", "") or ""))
    if not state:
        return out, False
    line = f"BF verifier-proof connect={state}"
    if found:
        return out.replace(found[0], line, 1), True
    return f"{line}\n{out}", True


def _hex_ip(value: str) -> str:
    raw = int(value, 16)
    return str(ipaddress.IPv4Address(raw.to_bytes(4, "little")))


def route_facts(out: str) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """(routes, subnets, gateways) from :func:`_route_facts_cmd`'s output."""
    found = re.search(r"BF verifier-network routes (\d+)", out or "")
    routes = int(found[1]) if found else -1
    subnets: dict[str, None] = {}
    gateways: dict[str, None] = {}
    never = ipaddress.ip_network("169.254.0.0/16")
    for kind, a, b in re.findall(
        r"(?m)^(net|gw) ([0-9A-Fa-f]{8})(?: ([0-9A-Fa-f]{8}))?$", out or ""
    ):
        if kind == "net" and b:
            net = ipaddress.ip_network(f"{_hex_ip(a)}/{_hex_ip(b)}", strict=False)
            # Never a link-local or metadata route, nor a route to one host:
            # those are the host's, not the sandbox's networks.
            if net.overlaps(never) or net.prefixlen == 32:
                continue
            subnets.setdefault(str(net))
        elif kind == "gw" and a != "00000000":
            gateways.setdefault(_hex_ip(a))
    return routes, tuple(subnets), tuple(gateways)


def _proof_cmd(rules: bool) -> str:
    """test.sh's side of the proof, run through the launch: a TCP connect to
    TEST-NET-1 and, where there are rules and iptables, an attempt to delete
    the last one and to flush the chain."""
    connect = (
        f'e=$(timeout 3 bash -c "exec 3<>/dev/tcp/{_PROOF_ADDRESS}/80" 2>&1); r=$?; '
        "case $r in 0) s=reached;; 124) s=timeout;; *) case $e in "
        "*[Rr]efused*) s=refused;; *[Uu]nreachable*|*'No route'*) s=unreachable;; "
        "*) s=failed;; esac;; esac; "
        'echo "BF verifier-proof connect=$s"; '
    )
    if not rules:
        return connect
    return connect + (
        "d=skipped; for t in /usr/sbin/iptables /sbin/iptables /usr/bin/iptables; do "
        '[ -x "$t" ] || continue; '
        'if "$t" -w -D OUTPUT -j REJECT 2>/dev/null; then d=deleted; '
        'elif "$t" -w -F OUTPUT 2>/dev/null; then d=flushed; else d=denied; fi; break; done; '
        'echo "BF verifier-proof delete=$d"; '
    )


async def arm_verifier_network(
    env: Any,
    task: Any,
    mode: str,
    *,
    where: str = "in-container",
    proxy: bool = True,
) -> VerifierNetwork:
    """Hold test.sh, root included, to the verifier's network *mode* (module
    doc) and prove it. Raises RuntimeError ("Verifier hardening failed") when
    it cannot."""
    from benchflow.sandbox._exec_output import exec_raw
    from benchflow.sandbox.egress_denylist import (
        denylist_agent_env,
        egress_denylist_for,
        start_egress_denylist,
    )
    from benchflow.sandbox.oracle_egress import _AGENT_ONLY_ENV

    armed = VerifierNetwork(mode=mode, state="open", where=where)
    if mode == "public":
        return armed

    def fail(why: str) -> RuntimeError:
        from benchflow.task.verifier_errors import VerifierNetworkUnheldError

        return VerifierNetworkUnheldError(
            f"Verifier hardening failed: the verifier's network ({where} under "
            f"{mode}: verifier.network_mode, else the sandbox's) could not be "
            f"enforced for test.sh ({why}); test.sh runs the agent's code and does "
            "not run outside the task's network"
        )

    async def root(command: str, timeout: int = 120) -> tuple[int | None, str]:
        result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout)
        out = getattr(result, "stdout", "") or ""
        return getattr(result, "return_code", None), out if isinstance(out, str) else ""

    _, out = await root(_route_facts_cmd(), 60)
    routes, subnets, gateways = route_facts(out)
    if routes == 0:
        armed.state = "no route"
        if _has_helper(env):
            # Count every attempt to reach out, so a grader that swallows a
            # failed download cannot score a silent 0.
            out = await _via_helper(env, _no_route_rules_cmd())
            if "BF verifier-network status=0" not in out:
                raise fail(
                    "the helper could not add the counting rules to a main with no route out"
                )
            armed.state, armed.match, armed.via_helper = (
                "no route, counted",
                "every process",
                True,
            )
            _ledger(
                env, "verifier", list(NO_ROUTE_RULES[0]), list(NO_ROUTE_RULES[1]), []
            )
    else:
        allow_networks: tuple[str, ...] = ()
        if mode in ("allowlist", "denylist") and proxy:
            # The list as the task declares it: 0.9.x reads no compose
            # service addresses into an allowlist (the agent phase's is the
            # same declared list).
            policy = egress_denylist_for(_verifier_policy_config(task.config, mode))
            if policy is None:
                raise fail("the task declares no host list")
            from benchflow.sandbox.egress_denylist import (
                has_egress_container,
                in_sandbox_refusals,
                start_egress_container,
            )

            try:
                if has_egress_container(env):
                    # Its own container: test.sh can neither signal it nor
                    # rewrite its log of refusals.
                    await start_egress_container(
                        env, policy, lifetime_sec=_proxy_lifetime(task)
                    )
                    armed.proxy = "own container"
                else:
                    await start_egress_denylist(env, "root", policy)
                    armed.proxy = "in sandbox"
                    armed.proxy_baseline = len(await in_sandbox_refusals(env) or [])
            except Exception as exc:
                raise fail(f"its egress proxy could not start: {exc}") from exc
            armed.env = {
                k: v
                for k, v in denylist_agent_env({}, policy).items()
                if k not in _AGENT_ONLY_ENV
            }
            if policy.allow_mode:
                allow_networks = tuple(policy.allow_networks)
        v4, v6, nat = everyone_rules(
            mode, subnets=subnets, gateways=gateways, allow_networks=allow_networks
        )
        out, armed.via_helper, added = await _install_recorded(env, v4, v6, nat)
        if "BF verifier-network status=0" not in out:
            found = re.search(r"BF verifier-network status=(\d+)", out)
            raise fail(
                f"its rules could not be added (status {found[1] if found else 'none'})"
            )
        armed.state, armed.match = "held", "every process"
        armed.subnets = subnets
        if armed.via_helper:
            _ledger(env, "verifier", *added)

    # The verifier's rules are in: the handover's hold comes out before their
    # proof, which counts at their own final REJECT.
    await release_handover(env)
    before = await _refused(env, armed)
    proof = await exec_raw(
        env.exec,
        verifier_launch(_proof_cmd(armed.match is not None)),
        # As root, the strongest user test.sh can be: what root cannot reach
        # or delete after the launch, no verifier user can.
        user="root",
        timeout_sec=60,
    )
    proof_out = str(getattr(proof, "stdout", "") or "")
    proof_err = str(getattr(proof, "stderr", "") or "")
    if getattr(proof, "return_code", None) == VERIFIER_LAUNCH_REFUSED:
        # The launch test.sh would take refused (no util-linux setpriv with
        # --bounding-set and no kit Python): the named ending, under every
        # policy, not a hardening crash.
        from benchflow.task.verifier_errors import VerifierLaunchRefusedError

        said = next(
            (
                ln.partition("refused: ")[2]
                for ln in proof_err.splitlines()
                if ln.startswith("BF verifier-launch refused: ")
            ),
            "",
        )
        raise VerifierLaunchRefusedError(
            f"the bounded launch refused: {said}"
            if said
            else "the bounded launch refused"
        )
    proof_out, from_helper = await _shell_or_helper_connect(env, proof_out)
    connect = re.search(r"BF verifier-proof connect=(\w+)", proof_out)
    delete = re.search(r"BF verifier-proof delete=(\w+)", proof_out)
    if delete and delete[1] in ("deleted", "flushed"):
        await _rules_root(env, rules_cmd(*everyone_rules(mode, subnets=armed.subnets)))
        raise fail(f"test.sh could change its rules ({delete[1]})")
    armed.proof = {
        "connect": connect[1] if connect else "",
        "delete": delete[1] if delete else "skipped",
    }
    if from_helper:
        armed.proof["connect_from"] = "helper, uid 0, main's namespace"
    if connect and connect[1] == "unreachable":
        # The kernel has no route to that destination (an internal network,
        # network: none): nothing leaves, rule or not.
        armed.proof["kernel"] = "unreachable"
        armed.baseline = await _refused(env, armed)
        return armed
    if armed.state == "no route":
        raise fail(
            f"no route out, but the test connect was {connect[1] if connect else 'not made'}"
        )
    after = await _refused(env, armed)
    armed.proof["counted"] = after - before
    if not connect or connect[1] != "refused" or after - before < 1:
        raise fail(
            f"the test connect was {connect[1] if connect else 'not made'} and the "
            f"rule counted {after - before}"
        )
    armed.baseline = after
    return armed


def remove_rules_cmd(v4: list[str], v6: list[str], nat: list[str] | None = None) -> str:
    """Root's script: delete each rule once, the copy this phase added. Never
    every copy: another phase's identical rule (main's own metadata rule,
    added at sandbox start) stays (fx609 matrix: deleting every copy opened
    the metadata endpoints between the oracle's turn and hardening)."""

    def dels(tool: str, rules: list[str]) -> str:
        return "".join(
            f"r={shlex.quote(rule)}; {tool} -w -D OUTPUT $r 2>/dev/null; "
            for rule in rules
        )

    return (
        "for T in iptables iptables-nft iptables-legacy; do command -v $T >/dev/null 2>&1 && break; done; "
        + dels("$T", v4)
        + "for T6 in ip6tables ip6tables-nft ip6tables-legacy; do command -v $T6 >/dev/null 2>&1 && break; done; "
        + dels("$T6", v6)
        + "".join(
            f"r={shlex.quote(rule)}; $T -w -t nat -D OUTPUT $r 2>/dev/null; "
            for rule in (nat or [])
        )
        + 'echo "BF verifier-network removed status=0"'
    )


async def arm_policy_agent_phase(
    env: Any, mode: str, allow_networks: tuple[str, ...] = ()
) -> tuple[list[str], list[str], list[str]]:
    """Hold every process of main to *mode* (``allowlist`` or ``denylist``)
    for the oracle's turn: the rules of :func:`everyone_rules` from
    BenchFlow's helper (main holds no NET_ADMIN, so a root oracle can neither
    remove them nor leave a matched group), only the egress proxy's marked
    sockets leaving; the proxy runs in its own container
    (:func:`benchflow.sandbox.egress_denylist.start_egress_container`, which
    the caller starts first). Proof: root's own TEST-NET connect refused and
    counted. The rules go in the rule ledger as ``agent-oracle``. Returns
    (v4, v6, nat) for their removal; raises RuntimeError when they cannot
    hold."""
    from benchflow.sandbox._exec_output import exec_raw

    async def root(command: str, timeout: int = 120) -> str:
        result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout)
        out = getattr(result, "stdout", "") or ""
        return out if isinstance(out, str) else ""

    def fail(why: str) -> RuntimeError:
        return RuntimeError(
            f"network_mode={mode!r} applies to the oracle, but its every-process "
            f"rules could not hold ({why})"
        )

    _, subnets, gateways = route_facts(await root(_route_facts_cmd(), 60))
    v4, v6, nat = everyone_rules(
        mode, subnets=subnets, gateways=gateways, allow_networks=allow_networks
    )
    out, via_helper, added = await _install_recorded(env, v4, v6, nat)
    if not via_helper or "BF verifier-network status=0" not in out:
        found = re.search(r"BF verifier-network status=(\d+)", out)
        raise fail(f"status {found[1] if found else 'none'}")
    # Only what this phase added: its removal never takes main's own rule.
    _ledger(env, "agent-oracle", *added)
    before = await _count_root(env, True)
    probe, _ = await _shell_or_helper_connect(env, await root(_proof_cmd(False), 60))
    after = await _count_root(env, True)

    def n(text: str) -> int:
        found = re.search(r"BF verifier-network refused (\d+)", text)
        return int(found[1]) if found else 0

    connect = re.search(r"BF verifier-proof connect=(\w+)", probe)
    state = connect[1] if connect else "not made"
    if state == "unreachable":
        # An internal network: proven present instead.
        held = await _held_without_a_route(env, added, after)
    else:
        held = state == "refused" and n(after) - n(before) >= 1
    if not held:
        await _via_helper(env, remove_rules_cmd(*added))
        raise fail(
            f"root's test connect was {state} and the rule counted "
            f"{n(after) - n(before)}"
            + (
                "; the rules were not all present, read from the helper"
                if state == "unreachable"
                else ""
            )
        )
    return added


async def _missing_from_helper(
    env: Any, v4: list[str], v6: list[str], nat: list[str] | None = None
) -> int | None:
    """How many of the rules main's namespace lacks, read from BenchFlow's
    helper with ``iptables -C`` (a read that needs no route); None when the
    helper did not answer."""
    from benchflow.sandbox import rule_ledger

    entry = rule_ledger.InstalledRules(
        where="main", phase="proof", v4=list(v4), v6=list(v6), nat=list(nat or [])
    )
    out = await _via_helper(env, rule_ledger.check_cmd(entry))
    found = re.search(r"BF rule-check status=0 missing=(\d+)", out)
    return int(found[1]) if found else None


async def _held_without_a_route(
    env: Any, added: tuple[list[str], list[str], list[str]], after: str
) -> bool:
    """The proof where root's connect is unreachable (an internal network: no
    route out, so the kernel refuses before any rule can count): every rule
    just added present, read from the helper, and the final REJECT's counter
    readable. Main holds no NET_ADMIN, so no route can be added after."""
    if not re.search(r"BF verifier-network refused \d+ status=0", after):
        return False
    return await _missing_from_helper(env, *added) == 0


async def remove_policy_agent_phase(
    env: Any, rules: tuple[list[str], list[str], list[str]]
) -> bool:
    """Take the oracle's every-process rules out after its turn (the
    verifier's go in at hardening); whether the helper said so."""
    from benchflow.sandbox import rule_ledger

    rule_ledger.retire(env, "main", "agent-oracle")
    out = await _via_helper(env, remove_rules_cmd(*rules))
    return "BF verifier-network removed status=0" in out


async def arm_offline_agent_phase(env: Any) -> dict[str, Any]:
    """Hold every process of main offline for an oracle's turn on an offline
    task whose main keeps a network for its verifier: the rules of
    :func:`everyone_rules` for ``no-network`` (loopback, replies and the
    project's networks; metadata and link-local refused first). Proof: root's
    own TCP connect to TEST-NET-1 is refused and counted. The rules are kept
    on *env* for hardening, which removes them before arming the verifier's
    network. Raises RuntimeError when they cannot hold: solve.sh does not run."""
    from benchflow.sandbox._exec_output import exec_raw

    async def root(command: str, timeout: int = 120) -> str:
        result = await exec_raw(env.exec, command, user="root", timeout_sec=timeout)
        out = getattr(result, "stdout", "") or ""
        return out if isinstance(out, str) else ""

    def fail(why: str) -> RuntimeError:
        return RuntimeError(
            "sandbox startup refused: the task is offline and its main keeps a "
            f"network only for the verifier, but the oracle's turn could not be held "
            f"offline ({why})"
        )

    routes, subnets, gateways = route_facts(await root(_route_facts_cmd(), 60))
    v4, v6, _ = everyone_rules("no-network", subnets=subnets, gateways=gateways)
    out, via_helper, (a4, a6, _) = await _install_recorded(env, v4, v6)
    if "BF verifier-network status=0" not in out:
        found = re.search(r"BF verifier-network status=(\d+)", out)
        raise fail(
            f"its rules could not be added (status {found[1] if found else 'none'})"
        )
    # Only what this phase added: its removal never takes main's own rule.
    env._benchflow_offline_rules = (a4, a6, via_helper)
    if via_helper:
        _ledger(env, "agent", a4, a6, [])
    before = await _count_root(env, via_helper)
    probe, from_helper = await _shell_or_helper_connect(
        env, await root(_proof_cmd(False), 60)
    )
    after = await _count_root(env, via_helper)

    def n(text: str) -> int:
        found = re.search(r"BF verifier-network refused (\d+)", text)
        return int(found[1]) if found else 0

    connect = re.search(r"BF verifier-proof connect=(\w+)", probe)
    record = {
        "phase": "agent",
        "mode": "no-network",
        "held": "every process",
        "rules_from": "helper" if via_helper else "sandbox",
        **({"connect_from": "helper"} if from_helper else {}),
        "routes": routes,
        "proof": {
            "connect": connect[1] if connect else "",
            "counted": n(after) - n(before),
        },
    }
    if connect and connect[1] == "unreachable" and via_helper:
        # An internal network: proven present from the helper instead.
        present = await _held_without_a_route(env, (a4, a6, []), after)
        record["proof"]["rules_present"] = present
        if not present:
            raise fail(f"root's test connect was {record['proof']}")
    elif (
        not connect
        or connect[1] not in ("refused", "unreachable")
        or (connect[1] == "refused" and n(after) - n(before) < 1)
    ):
        raise fail(f"root's test connect was {record['proof']}")
    return record


#: The handover's hold: from the moment the agent
#: phase's every-process rules (the offline oracle's, or a root oracle's
#: policy) start coming out until the verifier's are in, every packet of
#: main's but loopback is refused. Its comment makes it unique: no other
#: phase's ``-C`` check or one-copy removal matches it, and the counters'
#: final-REJECT pattern does not either.
HANDOVER_HOLD = "! -o lo -m comment --comment bf-verifier-handover -j REJECT"
#: Its TCP half, above it: a reset, so a connect fails at once (a plain
#: REJECT's ICMP answer leaves TCP retrying until the client's own timeout:
#: fx609 matrix6, update-ca-certificates' wrapped calls ran past 120 s).
HANDOVER_HOLD_TCP = (
    "! -o lo -p tcp -m comment --comment bf-verifier-handover "
    "-j REJECT --reject-with tcp-reset"
)
#: Both, top of OUTPUT first.
HANDOVER_HOLDS = (HANDOVER_HOLD_TCP, HANDOVER_HOLD)


def _hold_cmd() -> str:
    """Root's script: the hold at the top of main's OUTPUT, IPv4 and IPv6
    (its TCP half first)."""
    r, rt = shlex.quote(HANDOVER_HOLD), shlex.quote(HANDOVER_HOLD_TCP)

    def add(tool: str) -> str:
        return (
            f"{tool} -w -C OUTPUT $r 2>/dev/null || {tool} -w -I OUTPUT 1 $r || s=1; "
            f"{tool} -w -C OUTPUT $rt 2>/dev/null || {tool} -w -I OUTPUT 1 $rt || s=1; "
        )

    return (
        f"set +e; s=0; r={r}; rt={rt}; "
        "T=; for c in iptables iptables-nft iptables-legacy; do command -v $c >/dev/null 2>&1 "
        "&& $c -w -S OUTPUT >/dev/null 2>&1 && { T=$c; break; }; done; "
        '[ -n "$T" ] || { echo "BF verifier-network handover status=2"; exit 0; }; '
        + add("$T")
        + "T6=; for c in ip6tables ip6tables-nft ip6tables-legacy; do command -v $c >/dev/null 2>&1 "
        "&& $c -w -S OUTPUT >/dev/null 2>&1 && { T6=$c; break; }; done; "
        'v6=0; if [ -n "$T6" ]; then '
        + add("$T6")
        + "v6=1; elif [ -e /proc/net/if_inet6 ]; then s=3; fi; "
        'echo "BF verifier-network handover status=$s v6=$v6"'
    )


def _release_cmd() -> str:
    """Root's script: the hold out of every table it went into, proven gone."""
    body = "".join(
        f"r={shlex.quote(rule)}; "
        "n=0; while [ $n -lt 8 ] && $T -w -C OUTPUT $r 2>/dev/null; do "
        "$T -w -D OUTPUT $r || s=1; n=$((n + 1)); done; "
        "$T -w -C OUTPUT $r 2>/dev/null && s=1; "
        for rule in HANDOVER_HOLDS
    )
    return (
        "set +e; s=0; "
        "for T in iptables iptables-nft iptables-legacy ip6tables ip6tables-nft ip6tables-legacy; do "
        "command -v $T >/dev/null 2>&1 || continue; $T -w -S OUTPUT >/dev/null 2>&1 || continue; "
        + body
        + "done; "
        'echo "BF verifier-network released status=$s"'
    )


def _handover_pending(env: Any) -> bool:
    return (
        isinstance(getattr(env, "_benchflow_offline_rules", None), tuple)
        or isinstance(getattr(env, "_benchflow_oracle_egress", None), tuple)
        or isinstance(getattr(env, "_benchflow_start_hold", None), tuple)
    )


#: The agent policies the start hold holds main to.
START_HOLD_MODES = ("allowlist", "denylist", "no-network")


async def arm_start_hold(
    env: Any,
    target: dict[str, Any],
    mode: str,
    *,
    allow_networks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """main's every-process rules for *mode*, from a BenchFlow helper joined
    to its namespace while main still waits in the inert bootstrap:
    the image's ENTRYPOINT and everything it starts, root's included, start
    under the task's network policy, as a process started later in the agent
    phase does. The rules are :func:`everyone_rules`: only the egress proxy's
    marked sockets, loopback, replies and the project's networks (never their
    gateway) leave, and an allowlist's IP entries. The agent phase's own rules
    go in on top; hardening takes these out under the handover's hold
    (:func:`remove_start_hold`). Proof: uid 0's TCP connect to TEST-NET-1
    refused and counted, from the helper (the rules hold every process of
    the namespace, the helper's too), or no route at all. Raises
    :class:`benchflow.sandbox.service_cover.ServiceCoverError` when they
    cannot hold: main is never released without them."""
    from benchflow.sandbox import rule_ledger
    from benchflow.sandbox.service_cover import (
        ServiceCoverError,
        _connect,
        connect_cmd,
    )

    container = str(target["id"])
    v4, v6, nat = everyone_rules(
        mode,
        subnets=tuple(target.get("subnets") or ()),
        gateways=tuple(target.get("gateways") or ()),
        allow_networks=allow_networks,
    )

    def refused(why: str) -> ServiceCoverError:
        return ServiceCoverError(
            f"sandbox startup refused: the task's network policy ({mode}) could "
            f"not be put on every process of main before its own processes start "
            f"({why})"
        )

    # Captured: only what this hold adds is its own (main's cloud metadata
    # rule, in already, is neither recorded nor removed with it).
    outcome = await env.metadata_service_install(
        container, rule_ledger.capture_cmd(rules_cmd(v4, v6, nat))
    )
    out = str(getattr(outcome[0], "stdout", "") or "") if outcome else ""
    if outcome is None or outcome[1] or "BF verifier-network status=0" not in out:
        found = re.search(r"BF verifier-network status=(\d+)", out)
        why = getattr(env, "metadata_service_unavailable", None)
        detail = why(container) if callable(why) else ""
        raise refused(
            f"status {found[1] if found else 'none'}"
            + (f": {detail}" if isinstance(detail, str) and detail else "")
        )
    a4, a6, anat = rule_ledger.added_rules(out)
    env._benchflow_start_hold = (a4, a6, anat)
    rule_ledger.record(
        env,
        rule_ledger.InstalledRules(
            where="main",
            phase="start-hold",
            container=container,
            v4=a4,
            v6=a6,
            nat=anat,
        ),
    )

    def n(text: str) -> int:
        found = re.search(r"BF verifier-network refused (\d+)", text)
        return int(found[1]) if found else 0

    before = n(out)
    ran, _ = await env.metadata_service_run(
        container, connect_cmd(_PROOF_ADDRESS), timeout_sec=30, user="0:0"
    )
    counted, _ = await env.metadata_service_run(container, count_cmd(), timeout_sec=30)
    connect = _connect(str(getattr(ran, "stdout", "") or ""))
    record: dict[str, Any] = {
        "phase": "start",
        "mode": mode,
        "held": "every process",
        "rules": len(a4) + len(a6) + len(anat),
        "proof": {
            "connect": connect,
            "counted": n(str(getattr(counted, "stdout", "") or "")) - before,
        },
    }
    held = connect == "unreachable" or (
        connect == "refused" and record["proof"]["counted"] >= 1
    )
    if not held:
        raise refused(f"uid 0's test connect was {record['proof']}")
    return record


async def remove_start_hold(env: Any) -> None:
    """Take the start hold out (under the handover's hold, before the
    verifier's rules): raises when it stays."""
    rules = getattr(env, "_benchflow_start_hold", None)
    if not isinstance(rules, tuple) or len(rules) != 3:
        return
    from benchflow.sandbox import rule_ledger

    rule_ledger.retire(env, "main", "start-hold")
    out = await _via_helper(env, remove_rules_cmd(*rules))
    if "BF verifier-network removed status=0" not in out:
        raise RuntimeError(
            "Verifier hardening failed: the task's network policy, held on every "
            "process of main since its start, could not give way to the verifier's"
        )
    env._benchflow_start_hold = None


async def hold_for_handover(env: Any) -> bool:
    """Put the hold in from BenchFlow's helper when the agent phase's
    every-process rules are about to give way to the verifier's; True when it
    went in. Raises when it cannot: the rules then do not come out."""
    if not _handover_pending(env) or not _has_helper(env):
        return False
    if getattr(env, "rules_by_helper", False) is not True:
        return False
    out = await _via_helper(env, _hold_cmd())
    if "BF verifier-network handover status=0" not in out:
        raise RuntimeError(
            "Verifier hardening failed: the hold that keeps main's network "
            "refused while the oracle's rules give way to the verifier's could "
            "not go in"
        )
    # In the rule ledger while it holds (a re-proof then checks it too).
    holds = list(HANDOVER_HOLDS)
    v6 = holds if "BF verifier-network handover status=0 v6=1" in out else []
    _ledger(env, "handover", holds, v6, [])
    with contextlib.suppress(AttributeError):
        env._benchflow_handover_hold = True
    return True


async def release_handover(env: Any) -> None:
    """Take the hold out once the verifier's rules are in (or where the
    verifier's network is public): raises when it stays."""
    if getattr(env, "_benchflow_handover_hold", None) is not True:
        return
    from benchflow.sandbox import rule_ledger

    out = await _via_helper(env, _release_cmd())
    rule_ledger.retire(env, "main", "handover")
    if "BF verifier-network released status=0" not in out:
        raise RuntimeError(
            "Verifier hardening failed: the handover hold could not be taken out "
            "of main's rules"
        )
    with contextlib.suppress(AttributeError):
        env._benchflow_handover_hold = None


async def remove_offline_agent_phase(env: Any) -> None:
    """Take the oracle's offline rules out before the verifier's own."""
    from benchflow.sandbox._exec_output import exec_raw

    rules = getattr(env, "_benchflow_offline_rules", None)
    if not isinstance(rules, tuple) or len(rules) != 3:
        return
    v4, v6, via_helper = rules
    from benchflow.sandbox import rule_ledger

    rule_ledger.retire(env, "main", "agent")
    if via_helper:
        await _via_helper(env, remove_rules_cmd(v4, v6))
    else:
        await exec_raw(env.exec, remove_rules_cmd(v4, v6), user="root", timeout_sec=120)
    env._benchflow_offline_rules = None


async def arm_verifier_service(
    env: Any, task: Any, service: str, mode: str, *, metadata: bool
) -> VerifierNetwork:
    """Hold test.sh in the compose service *service* (``verifier.service``)
    to the verifier's network, from a helper joined to that service's
    namespace: the metadata rule for every user (unless the task opts out) and
    *mode*'s rules for every process, with no proxy there (``allowlist`` and
    ``denylist``: loopback, replies, the project's networks and an allowlist's
    IPs); proven by a refused, counted connect from uid 0. test.sh there starts
    through the same bounded launch. Raises RuntimeError ("Verifier hardening
    failed") when it cannot."""
    from benchflow.sandbox import service_cover as sc

    def fail(why: str) -> RuntimeError:
        from benchflow.task.verifier_errors import VerifierNetworkUnheldError

        return VerifierNetworkUnheldError(
            f"Verifier hardening failed: the verifier's network (service {service} "
            f"under {mode}) could not be enforced for test.sh ({why}); test.sh runs "
            "the agent's code and does not run outside the task's network"
        )

    targets = [
        t for t in await env.metadata_service_targets() if t["service"] == service
    ]
    if len(targets) != 1:
        raise fail(f"the service has {len(targets)} running containers")
    target = targets[0]
    armed = VerifierNetwork(mode=mode, state="open", where=f"service {service}")
    egress = {
        "no-network": "closed",
        "denylist": "closed",
        "allowlist": "allowlist",
    }.get(mode, "open")
    allow: tuple[str, ...] = ()
    if mode == "allowlist":
        from benchflow.sandbox.egress_denylist import egress_denylist_for

        policy = egress_denylist_for(_verifier_policy_config(task.config, mode))
        allow = tuple(getattr(policy, "allow_networks", ()) or ())
    record = await sc._cover_one(
        env,
        target,
        "public" if egress == "open" else mode,
        metadata,
        False,
        (),
        allow,
        phase="verifier",
    )
    if not record.get("proven"):
        raise fail(f"not proven: {record}")
    armed.state = record.get("state", "covered")
    armed.proof = dict(record.get("proof") or {})
    if egress != "open":
        armed.match, armed.container = "every process", str(target["id"])
        armed.baseline = await _refused(env, armed)
    return armed


async def _refused(env: Any, armed: VerifierNetwork) -> int:

    if not armed.match:
        return 0
    if armed.container:
        from benchflow.sandbox.service_cover import _counters, counters_cmd

        outcome = await env.metadata_service_install(
            armed.container, f": {__import__('secrets').token_hex(4)}; {counters_cmd()}"
        )
        counted = (
            _counters(str(getattr(outcome[0], "stdout", "") or "")) if outcome else None
        )
        return counted[1] if counted else 0
    out = await _count_root(env, armed.via_helper)
    found = re.search(
        r"BF verifier-network refused (\d+)", out if isinstance(out, str) else ""
    )
    return int(found[1]) if found else 0


def _main_metadata_ruled(env: Any) -> bool:
    """Whether BenchFlow's helper put cloud metadata rules in main: the rule
    ledger holds main's sandbox-start cover or the verifier's metadata block."""
    from benchflow.sandbox import rule_ledger

    return any(
        r.where == "main" and r.phase in ("start", "verifier-metadata")
        for r in rule_ledger.entries(env)
    )


async def main_metadata_refused(env: Any) -> int | None:
    """The packets main's cloud metadata rules have refused so far (its
    processes, root's included, during the agent's turn: a task daemon acting
    on the agent's input, say), read from BenchFlow's helper; None where no
    helper reads them, or where the helper put no metadata rule in main (the
    task's opt-out, or a sandbox built without the factory, which has no
    helper)."""
    if not _has_helper(env) or not _main_metadata_ruled(env):
        return None
    from benchflow.sandbox.service_cover import _counters, counters_cmd

    counted = _counters(await _via_helper(env, counters_cmd()))
    return counted[0] if counted else None


async def refused_since_arm(env: Any) -> tuple[VerifierNetwork | None, int]:
    """What hardening armed on *env*, and how often the verifier's network
    refused test.sh since: the packets its rules refused plus the requests
    its egress proxy refused (FXPB: an apt or wget the proxy refuses never
    reaches a rule). The parts are kept on the record."""
    armed = getattr(env, "_benchflow_verifier_network", None)
    if not isinstance(armed, VerifierNetwork):
        return None, 0
    packets = max(await _refused(env, armed) - armed.baseline, 0) if armed.match else 0
    refusals = await _proxy_refusals(env, armed)
    armed.refused_packets, armed.proxy_refused = packets, len(refusals)
    armed.proxy_refusals = [
        {k: str(e[k])[:200] for k in ("method", "url", "rule") if e.get(k)}
        for e in refusals[:20]
    ]
    return armed, packets + len(refusals)


async def _proxy_refusals(env: Any, armed: VerifierNetwork) -> list[dict[str, Any]]:
    """The requests the verifier phase's proxy refused since the arm."""
    if armed.proxy == "own container":
        box = getattr(env, "_egress_container", None)
        found = await box.refusals() if box is not None else None
        found = found or []
    elif armed.proxy == "in sandbox":
        from benchflow.sandbox.egress_denylist import in_sandbox_refusals

        found = (await in_sandbox_refusals(env) or [])[armed.proxy_baseline :]
    else:
        return []
    return [e for e in found if e.get("action") == "blocked"]


def _proxy_lifetime(task: Any) -> int:
    """How long the proxy's container may live: well past the verifier's
    own timeout (the sandbox's close removes it before that)."""
    from benchflow.sandbox._egress_container import LIFETIME_MARGIN_SEC

    timeout = getattr(getattr(task.config, "verifier", None), "timeout_sec", 0)
    seconds = int(timeout) if isinstance(timeout, (int, float)) else 0
    return max(3 * seconds, 3600) + LIFETIME_MARGIN_SEC
