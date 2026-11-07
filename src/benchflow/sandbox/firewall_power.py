"""Images that would hand the agent the firewall BenchFlow holds it with.

Under ``denylist``, ``allowlist``, a no-network task whose container stays
open for an LLM agent, and an open network with the cloud metadata block,
root in ``main`` holds NET_ADMIN (``_compose.compose_needs_net_admin``) for
the iptables rules that confine the agent user; on Daytona root holds it
anyway, for the daemon guard. NET_ADMIN then sits in the bounding set of
every process in the sandbox, so an image that puts ``cap_net_admin`` on a
binary (``setcap cap_net_admin+ep /usr/sbin/xtables-nft-multi``) or a setuid
bit on a firewall helper hands the power to the agent user, which could
remove the rules (PLAN S3, the Rust ``bench``'s P1.2d).

Before the agent starts, wherever such a rule holds it:

1. root reads its own bounding set (no NET_ADMIN there: no process in the
   sandbox can get it, and nothing more is done), whether ``setpriv`` can
   set no_new_privs (util-linux's and BusyBox's can) and drop a capability
   from the bounding and inheritable sets (only util-linux's can), and lists
   the setuid-root files and file capabilities in the image (with getcap, or
   the image's python3 where it has none; on the PATH and the system
   folders, or on the whole root file system where no drop can apply);
2. every shared agent launch uses ``--no-new-privs
   --bounding-set=-net_admin,-net_raw --inh-caps=-all --ambient-caps=-all``
   (``lockdown.build_priv_drop_cmd``), whether or not an audit record exists.
   Unsupported setpriv, including BusyBox's limited implementation, refuses
   launch. The audit verifies the resulting kernel state and attempts only
   an empty chain or table with each firewall tool. A failed restricted
   probe refuses startup; it never retries without the barrier;
3. a session-factory transport has no shared launch. Its direct check and
   whole-filesystem capability inventory must prove that its real identity
   cannot obtain the firewall. Metadata-protected session factories are
   refused by S2 before this supplemental audit.

The trial folder's ``firewall-power.json`` says which: ``disarmed``,
``refused`` or ``clear`` (no drop, and nothing found that gives the power),
what the scan found, and why.
"""

from __future__ import annotations

import importlib
import inspect
import re
import secrets
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, cast

from benchflow._utils import timing_spans as _timing_spans
from benchflow.sandbox._exec_receipt import receipt_exec

#: The trial folder's record of the check.
FIREWALL_POWER_RECORD = "firewall-power.json"

# The sandbox object carries the user whose commands start without NET_ADMIN,
# set by check_firewall_power and read by every agent launch
# (net_admin_dropped). In memory, so nothing in the sandbox can change it.
_DROP_ATTR = "_benchflow_net_admin_dropped_for"

#: setpriv's flags that take NET_ADMIN out of every set an exec can draw on:
#: the bounding set caps what file capabilities and setuid-root give, and the
#: inheritable set (which also clears the ambient one) what an exec inherits.
#: no_new_privs makes setuid bits and file capabilities give nothing, so no
#: program in the image can make the agent uid 0, whose traffic the uid
#: rules would not match (the agent's own sudo stops working too).
DROP_NET_ADMIN_FLAGS = "--no-new-privs --bounding-set=-net_admin,-net_raw --inh-caps=-all --ambient-caps=-all"

# The firewall tools the agent user's probe tries (an empty chain of its own,
# or an empty nft table, made and deleted: never a flush or a policy).
_PROBE_TOOLS = (
    "iptables",
    "iptables-nft",
    "iptables-legacy",
    "ip6tables",
    "ip6tables-nft",
    "ip6tables-legacy",
)

# Folders the PATH scan always covers besides the PATH itself.
_SYSTEM_DIRS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)

# Setuid-root programs every distribution ships, each of which authenticates
# its user or does one narrow job: not counted as a way to the firewall (a
# sudo rule that makes the agent root is outside this check; the drop holds
# against it where it applies). Any other setuid-root file the agent user can
# run is counted where the drop cannot apply.
_STANDARD_SETUID = frozenset(
    {
        "at",
        "bbsuid",
        "chage",
        "chfn",
        "chsh",
        "dbus-daemon-launch-helper",
        "doas",
        "expiry",
        "fusermount",
        "fusermount3",
        "gpasswd",
        "ksu",
        "mount",
        "mount.cifs",
        "mount.nfs",
        "newgidmap",
        "newgrp",
        "newuidmap",
        "passwd",
        "ping",
        "ping6",
        "pkexec",
        "polkit-agent-helper-1",
        "pt_chown",
        "ssh-keysign",
        "su",
        "sudo",
        "sudoedit",
        "umount",
        "unix_chkpwd",
        "Xorg.wrap",
    }
)

# Helpers outside the system folders, by full path (Chromium's sandbox and
# GStreamer's PTP helper were seen in Terminal-Bench 2's images).
_STANDARD_HELPER_PATHS = frozenset(
    {
        "/opt/google/chrome/chrome-sandbox",
        "/usr/lib/chromium-browser/chrome-sandbox",
        "/usr/lib/chromium/chrome-sandbox",
        "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-ptp-helper",
        "/usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-ptp-helper",
        "/usr/libexec/gstreamer-1.0/gst-ptp-helper",
        "/usr/lib/dbus-1.0/dbus-daemon-launch-helper",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/polkit-1/polkit-agent-helper-1",
        "/usr/libexec/dbus-daemon-launch-helper",
        "/usr/libexec/openssh/ssh-keysign",
        "/usr/libexec/polkit-agent-helper-1",
        "/usr/lib/xorg/Xorg.wrap",
        "/usr/lib/eject/dmcrypt-get-device",
        "/usr/sbin/pam_timestamp_check",
    }
)

# File capabilities distributions ship on narrow tools (packet capture, ping,
# user namespaces' id maps): not counted.
_STANDARD_FILE_CAPS = frozenset(
    {
        "arping",
        "clockdiff",
        "dumpcap",
        "fping",
        "fping6",
        "gst-ptp-helper",
        "mtr-packet",
        "newgidmap",
        "newuidmap",
        "ping",
        "ping4",
        "ping6",
        "traceroute6.iputils",
    }
)

# A file capability counts when it gives NET_ADMIN, or root's powers that
# lead to it (become uid 0, or change the capability sets).
_POWER_CAPS = ("cap_net_admin", "cap_setuid", "cap_sys_admin", "cap_setpcap")

# The scan reads at most this many setuid files and file capabilities each,
# and the record keeps at most _FOUND_LIMIT of the ones that count (an image
# with more is still refused, by the first ones).
_SCAN_LINES = 1000
_CANDIDATE_LIMIT = 200
_FOUND_LIMIT = 40

_ROOT_EXEC_SEC = 120
_AGENT_EXEC_SEC = 120


# These are names in the controller's allowlist, never names supplied by a
# probe. Root binds them to validated absolute paths before executing a step.
_ROOT_SCAN_TOOLS = ("awk", "find", "head", "sort")
_ROOT_LAUNCH_TOOLS = ("setpriv", "grep", "bash")


def _root_route(
    command: str,
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
    route: Any = None,
) -> str:
    """Use the metadata root route and its lexical path/link validator.

    The backend must start the system shell before this command. Rebuild
    the tool environment from scratch; neither a task variable nor a probe
    result can select root's executables. PT's protected runtime replaces
    the common bootstrap tools when C8 lands, at this same route.
    """
    from benchflow.sandbox._base import with_system_path
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    bindings = []
    for name in (*required, *optional):
        variable = "bf_" + name.replace("-", "_")
        missing = (
            f"printf 'BF fwpower-error tool={name}\\n'; exit 126"
            if name in required
            else f"{variable}="
        )
        bindings.append(f"{variable}=$(tool {name}) || {{ {missing}; }}; ")
    # The shared route owns PATH and HOME. Keep its protected runtime first
    # after the metadata validator defines tool(), including once PT lands.
    if getattr(route, "kit", False):
        from benchflow._root_only import ROOT_CHAIN_FUNCS
        from benchflow._runtime_tree import KIT_PYTHON

        # The route has already proved the kit; its find/readlink inspect
        # remaining tools without running the image's inspector copies.
        prelude = (
            "cd / || exit 126; "
            + ROOT_CHAIN_FUNCS
            + (
                'tool() { _f=$(command -v "$1" 2>/dev/null) || return 1; '
                'bfc_chain "$_f" 0 && [ -f "$_f" ] && [ -x "$_f" ] || return 2; '
                'printf "%s\\n" "$_f"; }; '
            )
        )
        if "python3" in optional:
            bindings.append(f"bf_python3={shlex.quote(KIT_PYTHON)}; ")
    else:
        prelude = (
            "cd / || exit 126; bf_route_path=$PATH; " + _cloud_metadata_root_prelude()
        )
        prelude += "PATH=$bf_route_path; export PATH; "
    body = prelude + "".join(bindings) + command
    # Before PT seals the route's tools, even a system PATH can contain a
    # writable executable. Validate the clean-environment bootstrap too,
    # before starting either helper; never fall back to an unvalidated name.
    bootstrap = prelude
    for name in ("env", "sh"):
        bootstrap += (
            f"bf_{name}=$(tool {name}) || {{ "
            f"printf 'BF fwpower-error tool={name}\\n'; exit 126; }}; "
        )
    # Keep the backend's parent shell alive: it owns the environment-file
    # EXIT trap even when the clean child execs the restricted launcher.
    text = (
        bootstrap + '\n"$bf_env" -i PATH="$PATH" HOME="$HOME" LC_ALL=C '
        'fwp_inventory_path="$fwp_inventory_path" '
        f'"$bf_sh" -c {shlex.quote(body)}'
    )
    # The image PATH stays inventory data only (integ S2), on either route.
    return "fwp_inventory_path=${PATH-}; " + (
        text if getattr(route, "kit", False) else with_system_path(text)
    )


class FirewallPowerError(RuntimeError):
    """The agent user could change the firewall BenchFlow holds it with, and
    BenchFlow cannot take the power away from its commands.

    The message starts with "sandbox startup refused", so the trial is a
    sandbox-setup failure (the task image's) that a retry would only repeat,
    never run that way.
    """


def net_admin_dropped(env: Any, sandbox_user: str | None) -> bool:
    """Whether ``sandbox_user``'s commands in ``env`` start without NET_ADMIN
    and with no_new_privs (NET_ADMIN out of the bounding set too, where the
    image's setpriv can take it out).

    Every agent launch passes this to ``lockdown.build_priv_drop_cmd``; it is
    true only after :func:`check_firewall_power` proved the drop works in
    this sandbox for this user.
    """
    return bool(sandbox_user) and getattr(env, _DROP_ATTR, None) == sandbox_user


def _keep_drop(env: Any, sandbox_user: str | None) -> bool:
    try:
        setattr(env, _DROP_ATTR, sandbox_user)
    except (AttributeError, TypeError):
        return False
    return getattr(env, _DROP_ATTR, None) == sandbox_user


def forget_net_admin_drop(env: Any) -> None:
    """The next launch in ``env`` keeps the sandbox's sets (no rule of
    BenchFlow's holds its agent)."""
    if getattr(env, _DROP_ATTR, None) is not None:
        _keep_drop(env, None)


def firewall_rules_hold_agent(
    env: Any, agent_env: Mapping[str, str], *, cloud_metadata_armed: bool
) -> tuple[str, ...]:
    """The firewall rules of BenchFlow's that will hold the agent user in
    ``env``, by name; empty when none does (the check is then not needed)."""
    from benchflow.sandbox._compose import compose_needs_net_admin
    from benchflow.sandbox.lockdown import (
        EGRESS_ALLOWLIST_ENV,
        EGRESS_DENYLIST_ENV,
        _guard_needed_here,
    )

    holds: list[str] = []
    if agent_env.get(EGRESS_DENYLIST_ENV) == "1":
        holds.append(
            "the allowlist's uid firewall"
            if agent_env.get(EGRESS_ALLOWLIST_ENV) == "1"
            else "the denylist's uid firewall"
        )
    elif agent_env.get("BENCHFLOW_DISALLOW_WEB_TOOLS") == "1":
        holds.append("the no-web uid firewall")
    if cloud_metadata_armed:
        holds.append("the cloud metadata block")
    if _guard_needed_here(env):
        holds.append("Daytona's daemon guard")
    config = getattr(env, "task_env_config", None)
    if not holds and config is not None:
        try:
            granted = compose_needs_net_admin(config)
        except AttributeError:
            granted = False
        if granted:
            holds.append("the NET_ADMIN BenchFlow gives main for its firewall")
    return tuple(holds)


def _audit_prelude() -> str:
    """Resolve after backend env materialization; never execute its PATH.

    S2 owns the custody walk. Each call carries its validated absolute path
    straight into exec, including readers spawned by find and root cleanup.
    The inherited PATH is data for inventory only, never tool resolution.
    """
    from benchflow.sandbox.lockdown import _protected_root_prelude

    return (
        "fwp_inventory_path=${fwp_inventory_path-${PATH-}}; "
        + _protected_root_prelude()
    )


# Root's reader of file capabilities where the image has no getcap (Debian,
# Ubuntu and Alpine bases ship none): PT's protected Python, run isolated,
# reads each file's security.capability attribute and prints it the way
# getcap does. A value it cannot decode is printed as every capability
# ("=ep"), so it counts.
_CAP_READER = r"""
import errno, os, sys
names = (
    "chown dac_override dac_read_search fowner fsetid kill setgid setuid "
    "setpcap linux_immutable net_bind_service net_broadcast net_admin net_raw "
    "ipc_lock ipc_owner sys_module sys_rawio sys_chroot sys_ptrace sys_pacct "
    "sys_admin sys_boot sys_nice sys_resource sys_time sys_tty_config mknod "
    "lease audit_write audit_control setfcap mac_override mac_admin syslog "
    "wake_alarm block_suspend audit_read perfmon bpf checkpoint_restore"
).split()
left = int(sys.argv[1])
seen = set()
out = sys.stdout.buffer
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p or p in seen:
        continue
    seen.add(p)
    try:
        d = os.getxattr(p, "security.capability", follow_symlinks=False)
    except OSError as e:
        if e.errno not in (errno.ENODATA, errno.ENOTSUP):
            out.write(b"BF fwpower-invalid\n")
        continue
    w = [int.from_bytes(d[i : i + 4], "little") for i in range(0, len(d) - 3, 4)]
    rev = w[0] & 0xFF000000 if w else 0
    if rev == 0x01000000 and len(w) >= 3:
        prm, inh = w[1], w[2]
    elif rev in (0x02000000, 0x03000000) and len(w) >= 5:
        prm, inh = w[1] | w[3] << 32, w[2] | w[4] << 32
    else:
        prm = inh = -1
    if prm < 0:
        caps = "=ep"
    elif not prm | inh:
        continue
    else:
        bits = prm | inh
        caps = ",".join(
            "cap_" + (names[i] if i < len(names) else str(i))
            for i in range(64)
            if bits >> i & 1
        )
        caps += "=" + ("e" if w[0] & 1 else "") + ("p" if prm else "")
        caps += "i" if inh else ""
    out.write(b"BF fwpower-caphex " + p.hex().encode() + b" " + caps.encode() + b"\n")
    left -= 1
    if left <= 0:
        out.write(b"BF fwpower-invalid\n")
        break
"""


# getcap's human-readable output has no path delimiter. Ask about one file
# at a time, remove that exact argument, and encode its bytes separately.
# This also handles a filename containing capability-looking words or LF.
_GETCAP_READER = r"""
set -e
getcap=$1; od=$2; tr=$3; shift 3
for f do
    line=$("$getcap" "$f")
    [ -n "$line" ] || continue
    case $line in
        "$f "*) caps=${line#"$f "};;
        *) echo 'BF fwpower-invalid'; exit 1;;
    esac
    case $caps in '= '*) caps=${caps#'= '};; esac
    hex=$("$od" -An -v -tx1 <<EOF
$f
EOF
    )
    hex=$(echo "$hex" | "$tr" -d ' \n')
    hex=${hex%0a}
    [ -n "$hex" ] || { echo 'BF fwpower-invalid'; exit 1; }
    printf 'BF fwpower-caphex %s %s\n' "$hex" "$caps"
done
"""

_SETUID_READER = r"""
set -e
od=$1; tr=$2; shift 2
for f do
    hex=$("$od" -An -v -tx1 <<EOF
$f
EOF
    )
    hex=$(echo "$hex" | "$tr" -d ' \n')
    hex=${hex%0a}
    [ -n "$hex" ] || { echo 'BF fwpower-invalid'; exit 1; }
    printf 'BF fwpower-setuidhex %s\n' "$hex"
done
"""


def _scan_cmd(
    *,
    whole_fs: bool,
    fs_root: str = "/",
    setuid_owner: str = "0",
    python_command: str | None = None,
) -> str:
    """Root's scan for setuid-root files and file capabilities.

    Covers the PATH and the system folders, or with ``whole_fs`` every
    executable mounted filesystem. As root, so a folder the agent user may
    pass through but not list hides nothing. Prints ``BF fwpower-found setuid
    <path>``, ``BF fwpower-found cap <getcap line>``, then ``BF fwpower-scan
    scope=<path|all> caps=<getcap|python3|unread|none>``: what read the file
    capabilities (``unread``: python3 failed; ``none``: the image has neither).
    ``fs_root`` and ``setuid_owner`` (root's uid) let tests point it at a
    stub tree.
    """
    limit = _SCAN_LINES
    r = shlex.quote(fs_root)
    owner = shlex.quote(setuid_owner)
    x_files = r"\( -perm -100 -o -perm -010 -o -perm -001 \)"
    # PT rounds 9/10 reserve this entry point for root's helpers. Before
    # the protected kit exists there is no trusted Python fallback; never
    # resolve an image/workspace interpreter on root's PATH. Tests may
    # supply a private interpreter fixture explicitly.
    python = shlex.quote(python_command) if python_command else "false"
    reader = f"{python} -I -S -B -c {shlex.quote(_CAP_READER)} {limit}"
    getcap = (
        f'-exec "$scan_sh" -c {shlex.quote(_GETCAP_READER)} sh '
        '"$scan_getcap" "$scan_od" "$scan_tr" {} +'
    )
    setuid = (
        f'-exec "$scan_sh" -c {shlex.quote(_SETUID_READER)} sh '
        '"$scan_od" "$scan_tr" {} +'
    )
    setup = ""
    finish = ""
    if whole_fs:
        # Each mount gets its own -xdev traversal: a Docker volume need not
        # share /'s device. Virtual and noexec mounts cannot grant an exec
        # privilege. Read mountinfo rather than guess from directory names;
        # an executable mount below /dev or /proc still needs inspection.
        setup = (
            # Only the final assignment can commit a complete scan. The
            # EXIT trap reports incomplete on any command/redirection
            # failure, including before scratch storage exists. Do not use
            # pipelines here: POSIX sh only checks their final command.
            "set -e; cr=unread; complete=' complete=0'; scan_tmp=; exec 3>&1; "
            'trap \'echo "BF fwpower-scan scope=all caps=$cr$complete" >&3; '
            '[ -z "$scan_tmp" ] || run rm -rf "$scan_tmp"; exit 0\' EXIT; '
            "scan_tmp=$(run mktemp -d); "
            f"printf '%s\\n' {r} > \"$scan_tmp/roots\"; "
            "run awk 'NF < 10 {exit 1} "
            '{t=""; for(i=7;i<=NF;i++) if($i=="-") {t=$(i+1); break} '
            'if(t=="") exit 1; '
            "if(t ~ /^(proc|sysfs|devpts|cgroup|cgroup2|mqueue|securityfs|debugfs|tracefs|pstore|bpf|configfs|fusectl)$/) next; "
            'if("," $6 "," ~ /,noexec,/) next; print $5}\' '
            '/proc/self/mountinfo > "$scan_tmp/mounts"; '
            "while IFS= read -r mp; do "
            r"case $mp in *'\012'*|*'\015'*|*'\011'*|*'\134'*) "
            "exit 1;; esac; "
            "mp=$(printf '%b' \"$mp\"); "
            f"case {r} in /) ;; *) case $mp in {r}|{r}/*) ;; *) continue;; esac;; esac; "
            'printf \'%s\\n\' "$mp" >> "$scan_tmp/roots"; '
            'done < "$scan_tmp/mounts"; '
            'run sort -u "$scan_tmp/roots" > "$scan_tmp/sorted"; '
        )

        def traverse(predicate: str) -> str:
            return (
                "while IFS= read -r root; do "
                f'run find "$root" -xdev -type f {predicate} '
                '2>>"$scan_tmp/errors"; '
                'done < "$scan_tmp/sorted"'
            )

        setuid_list = (
            f'{traverse(f"-perm -4000 -user {owner} {setuid}")} > "$scan_tmp/setuid"; '
            'count=$(run wc -l < "$scan_tmp/setuid"); '
            f'[ "$count" -le {limit} ]; '
            f'run head -n {limit} "$scan_tmp/setuid"'
        )
        cap_list = (
            f'{traverse(f"{x_files} {getcap}")} > "$scan_tmp/caps"; '
            'count=$(run wc -l < "$scan_tmp/caps"); '
            f'[ "$count" -le {limit} ]; '
            f'run head -n {limit} "$scan_tmp/caps"'
        )
        cap_files = f'{traverse(f"{x_files} -print0")} > "$scan_tmp/caps"'
        finish = '[ ! -s "$scan_tmp/errors" ]; complete=; '
    else:
        # The list is split as a whole (dash splits only what an expansion
        # gives, so "$PATH:/usr/sbin" would glue PATH's last entry to it).
        system = ":".join(_SYSTEM_DIRS)
        dirs = (
            f'dirs="${{fwp_inventory_path:-}}:{system}"; '
            "oldifs=$IFS; IFS=:; set -f; set -- $dirs; IFS=$oldifs; set +f; "
        )
        setuid_list = (
            f'{{ {dirs}for d in "$@"; do [ -d "$d" ] || continue; '
            f'run find "$d" -maxdepth 1 -type f -perm -4000 -user {owner} {setuid} 2>/dev/null; '
            f"done; }} | run sort -u | run head -n {limit}"
        )
        cap_list = (
            f'{{ {dirs}for d in "$@"; do [ -d "$d" ] || continue; '
            f'run find "$d" -maxdepth 1 -type f {x_files} {getcap} 2>/dev/null; '
            f"done; }} | run sort -u | run head -n {limit}"
        )
        cap_files = (
            f'{{ {dirs}for d in "$@"; do [ -d "$d" ] || continue; '
            f'run find "$d" -maxdepth 1 -type f {x_files} -print0 2>/dev/null; '
            "done; }"
        )
    scope = "all" if whole_fs else "path"
    read_caps = (
        f'cr=unread; {cap_files}; {reader} < "$scan_tmp/caps"; cr=python3; '
        if whole_fs
        else f"if {cap_files} | {reader}; then cr=python3; else cr=unread; fi; "
    )
    report = "" if whole_fs else f'echo "BF fwpower-scan scope={scope} caps=$cr"; '
    return (
        "( " + _audit_prelude() + f"{setup}"
        "scan_sh=$(tool sh) && scan_od=$(tool od) && scan_tr=$(tool tr) || exit 7; "
        "scan_getcap=$(tool getcap) || scan_getcap=; "
        f"{setuid_list}; "
        "cr=none; if have getcap; then cr=getcap; "
        f"{cap_list}; "
        f"elif {'walk ' + python + ' && [ -x "$W" ]' if python_command else 'false'}; then "
        f"{read_caps}fi; "
        f"{finish}"
        f"{report}"
        ")"
    )


def _root_cmd(
    proc: str = "/proc",
    *,
    scan_all: bool = False,
    fs_root: str = "/",
    setuid_owner: str = "0",
    python_command: str | None = None,
) -> str:
    """Root's step: is NET_ADMIN in root's bounding set, can setpriv drop it,
    and what the image holds. Prints ``BF fwpower-root net_admin=<0|1>
    setpriv=<ok|none|no-reuid|no-bounding-set|no-inh-caps|no-nnp> nnp=<0|1>``
    (``nnp``: the setpriv can set no_new_privs, BusyBox's included), then,
    with NET_ADMIN there, :func:`_scan_cmd`'s lines: of the whole file system
    where no drop can apply (no setpriv that can set no_new_privs, or
    ``scan_all``), else of the PATH and the system folders."""
    from benchflow.sandbox.lockdown import _proc_status_reader

    p = shlex.quote(proc)
    every = _scan_cmd(
        whole_fs=True,
        fs_root=fs_root,
        setuid_owner=setuid_owner,
        python_command=python_command,
    )
    path = _scan_cmd(
        whole_fs=False,
        fs_root=fs_root,
        setuid_owner=setuid_owner,
        python_command=python_command,
    )
    return (
        "( "
        + _audit_prelude()
        + _proc_status_reader()
        + f"bnd=$(read_status CapBnd {p}/self/status 2>/dev/null) || bnd=; "
        "case $bnd in ''|*[!0-9a-fA-F]*) na=unknown;; "
        "*) na=$(( 0x$bnd >> 12 & 1 ));; esac; "
        "sp=none; nnp=0; "
        "if have setpriv; then "
        "h=$(run setpriv --help 2>&1); "
        "case $h in *no-new-privs*) nnp=1;; esac; "
        "case $h in *reuid*) "
        "case $h in *bounding-set*) "
        "case $h in *inh-caps*) "
        "case $h in *no-new-privs*) sp=ok;; *) sp=no-nnp;; esac;; "
        "*) sp=no-inh-caps;; esac;; "
        "*) sp=no-bounding-set;; esac;; "
        "*) sp=no-reuid;; esac; fi; "
        'echo "BF fwpower-root net_admin=$na setpriv=$sp nnp=$nnp"; '
        f"all={'1' if scan_all else ''}; "
        'if [ "$na" = 1 ]; then '
        f'if [ -n "$all" ] || [ "$nnp" != 1 ]; then {every}; else {path}; fi; fi '
        ")"
    )


def _agent_check_cmd(
    probe: str, candidates: Sequence[str] = (), *, proc: str = "/proc"
) -> str:
    """The agent user's step, run through the agent's own launch line.

    Prints ``BF fwpower-x <path>`` for each of root's ``candidates`` it can
    run, then its capability sets and every firewall tool on its PATH that
    made an empty chain (or nft table) named ``probe`` for it (deleted at
    once, with an xtables lock file of its own): ``BF fwpower-agent bnd= inh=
    amb= prm= eff= nnp= power=<tools>`` (``nnp``: its ``NoNewPrivs``). POSIX
    sh (dash and BusyBox ash included).
    """
    from benchflow.sandbox.lockdown import _proc_status_reader

    p = shlex.quote(proc)
    q = shlex.quote(probe)
    tools = " ".join(_PROBE_TOOLS)
    runnable = ""
    if candidates:
        runnable = "".join(
            f"[ ! -x {shlex.quote(c)} ] || "
            f"echo 'BF fwpower-xhex {c.encode(errors='surrogateescape').hex()}'; "
            for c in candidates
        )
    return (
        "( " + _audit_prelude() + _proc_status_reader() + f"P={p}; "
        f"{runnable}"
        'cap() { read_status "$1" "$P/self/status" 2>/dev/null; }; '
        "bnd=$(cap CapBnd); inh=$(cap CapInh); amb=$(cap CapAmb); "
        "prm=$(cap CapPrm); eff=$(cap CapEff); nnp=$(cap NoNewPrivs); "
        "lock=$(run mktemp 2>/dev/null) || lock=/tmp/.bf-fwpower-$$; power=; "
        f"for t in {tools}; do "
        'have "$t" || continue; '
        f'if XTABLES_LOCKFILE="$lock" run "$t" -w -N {q} >/dev/null 2>&1; then power="$power,$t"; '
        f'XTABLES_LOCKFILE="$lock" run "$t" -w -X {q} >/dev/null 2>&1; fi; done; '
        f"if have nft && run nft add table inet {q} >/dev/null 2>&1; then "
        f'power="$power,nft"; run nft delete table inet {q} >/dev/null 2>&1; fi; '
        'run rm -f "$lock" 2>/dev/null; '
        'echo "BF fwpower-agent bnd=$bnd inh=$inh amb=$amb prm=$prm eff=$eff '
        'nnp=$nnp power=${power#,}" '
        ")"
    )


def _cleanup_cmd(probe: str) -> str:
    """Root, after a probe that worked: its chain or table, if left, removed."""
    q = shlex.quote(probe)
    tools = " ".join(_PROBE_TOOLS)
    return (
        "( "
        + _audit_prelude()
        + f'for t in {tools}; do have "$t" && run "$t" -w -X {q} >/dev/null 2>&1; done; '
        f"have nft && run nft delete table inet {q} >/dev/null 2>&1; "
        "true )"
    )


def _has_net_admin(mask: str) -> bool:
    try:
        return bool(int(mask, 16) >> 12 & 1)
    except ValueError:
        return False


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"\b(\w+)=(\S*)", line))


def _cap_names(caps: str) -> tuple[set[str], bool]:
    """The capability names a getcap clause grants (``cap_a,cap_b+ep``,
    ``cap_a,cap_b=ep``), and whether it grants them all (``=ep``)."""
    names: set[str] = set()
    every = False
    for clause in caps.split():
        m = re.fullmatch(r"([a-z0-9_,]*)([=+][a-z]*)", clause.strip().lower())
        if not m:
            continue
        if not m.group(1) or m.group(1) == "all":
            every = True
        names.update(n for n in m.group(1).split(",") if n)
    return names, every


def _parse_cap_line(line: str) -> tuple[str, str]:
    """A getcap line, ``<path> <caps>`` (libcap 2.x) or ``<path> = <caps>``
    (older), as (path, caps)."""
    # Legacy records (kept for old sandbox adapters). Production readers
    # send hex paths instead. Start at the capability suffix, not the first
    # space in a filename; preserve even trailing filename whitespace.
    match = re.fullmatch(
        r"(.+?) (?:= )?([a-z0-9_,]*[=+][eip]+(?: [a-z0-9_,]*[=+][eip]+)*)",
        line,
    )
    return (match[1], match[2]) if match else ("", "")


def _standard(path: str, names: frozenset[str]) -> bool:
    """A distribution's own program at its own place (by name in a system
    folder, or a known helper's path); the same name elsewhere counts."""
    if path in _STANDARD_HELPER_PATHS:
        return True
    folder, _, name = path.rpartition("/")
    return name in names and folder in _SYSTEM_DIRS


def _decode_path(encoded: str) -> str:
    try:
        path = bytes.fromhex(encoded).decode(errors="surrogateescape")
    except ValueError:
        return ""
    return path if path.startswith("/") and "\0" not in path else ""


def _scan_report(out: str) -> dict[str, str]:
    scan = _fields(
        next((ln for ln in out.splitlines() if ln.startswith("BF fwpower-scan ")), "")
    )
    for line in out.splitlines():
        bad = line == "BF fwpower-invalid"
        if line.startswith("BF fwpower-caphex "):
            encoded, _, caps = line.removeprefix("BF fwpower-caphex ").partition(" ")
            bad = not _decode_path(encoded) or not re.fullmatch(
                r"[a-z0-9_,]*[=+][eip]+(?: [a-z0-9_,]*[=+][eip]+)*", caps
            )
        elif line.startswith("BF fwpower-setuidhex "):
            bad = not _decode_path(line.removeprefix("BF fwpower-setuidhex "))
        elif line.startswith("BF fwpower-found cap "):
            path, caps = _parse_cap_line(line.removeprefix("BF fwpower-found cap "))
            # More than one capability clause in the old format could also
            # be part of a filename. A legacy adapter must fail closed.
            bad = not path or " " in caps
        if bad:
            scan["complete"] = "0"
    if len(_scanned(out)) >= _CANDIDATE_LIMIT:
        scan["complete"] = "0"
    return scan


def _scanned(out: str) -> list[dict[str, Any]]:
    """The files root's scan printed that would give a user who can run them
    NET_ADMIN: a setuid-root file, or a file capability with NET_ADMIN or
    root's powers, other than the distributions' own narrow programs."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in out.splitlines():
        entry: dict[str, Any] | None = None
        if line.startswith(("BF fwpower-found setuid ", "BF fwpower-setuidhex ")):
            path = (
                _decode_path(line.removeprefix("BF fwpower-setuidhex "))
                if line.startswith("BF fwpower-setuidhex ")
                else line.removeprefix("BF fwpower-found setuid ")
            )
            if not _standard(path, _STANDARD_SETUID):
                entry = {"path": path, "kind": "setuid root"}
        elif line.startswith(("BF fwpower-found cap ", "BF fwpower-caphex ")):
            if line.startswith("BF fwpower-caphex "):
                encoded, _, caps = line.removeprefix("BF fwpower-caphex ").partition(
                    " "
                )
                path = _decode_path(encoded)
            else:
                path, caps = _parse_cap_line(line.removeprefix("BF fwpower-found cap "))
            names, every = _cap_names(caps)
            if (every or any(c in names for c in _POWER_CAPS)) and not _standard(
                path, _STANDARD_FILE_CAPS
            ):
                entry = {"path": path, "kind": "file capability", "capabilities": caps}
        if entry is not None and (entry["path"], entry["kind"]) not in seen:
            seen.add((entry["path"], entry["kind"]))
            found.append(entry)
        if len(found) >= _CANDIDATE_LIMIT:
            break
    return found


def _runnable(scanned: list[dict[str, Any]], agent_out: str) -> list[dict[str, Any]]:
    """The scanned files the agent user's check says it can run."""
    can = {
        ln.removeprefix("BF fwpower-x ")
        for ln in agent_out.splitlines()
        if ln.startswith("BF fwpower-x ")
    }
    for line in agent_out.splitlines():
        if line.startswith("BF fwpower-xhex "):
            try:
                can.add(_decode_path(line.split(" ", 2)[2]))
            except ValueError:
                continue
    return [e for e in scanned if e["path"] in can][:_FOUND_LIMIT]


def _caps_read(scan: Mapping[str, str]) -> bool:
    """Whether root's scan read the image's file capabilities."""
    return scan.get("caps") in ("getcap", "python3")


def _describe(entry: Mapping[str, Any]) -> str:
    if entry["kind"] == "setuid root":
        return f"{entry['path']} (setuid root)"
    return f"{entry['path']} (a file capability: {entry['capabilities']})"


def firewall_power_record(state: str, why: str, **extra: Any) -> dict[str, Any]:
    """The trial's ``firewall-power.json``."""
    return {"defense": "firewall_power", "state": state, "why": why, **extra}


async def _exec(
    env: Any,
    command: str,
    timeout_sec: int,
    *,
    user: str = "root",
    retain_failed: bool = False,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> tuple[str, str]:
    """One exec; ``(stdout, "")`` or ``("", why)`` when it failed."""
    try:
        if user == "root":
            from benchflow.sandbox.lockdown import _cloud_metadata_exec

            if getattr(env, "supports_system_shell", False) is not True:
                return "", "use a sandbox backend with protected system-shell support"
            from benchflow.sandbox._base import system_shell_route

            route = await system_shell_route(env)
            result, failed = await _cloud_metadata_exec(
                env,
                _root_route(command, required=required, optional=optional, route=route),
                user=user,
                timeout_sec=timeout_sec,
            )
            if failed:
                return "", failed
        else:
            result = await receipt_exec(
                env, command, user=user, timeout_sec=timeout_sec
            )
    except Exception as e:  # the caller fails closed on a missing report
        detail = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        return "", f"{type(e).__name__}{': ' + detail if detail else ''}"
    out = getattr(result, "stdout", "") or ""
    if isinstance(out, str):
        missing = re.search(r"^BF fwpower-error tool=(\S+)$", out, re.MULTILINE)
        if missing:
            return "", (
                f"no validated absolute path for {missing[1]}; install it in a "
                "root-owned system folder and remove non-root write access "
                "from the tool, its parent folders and symlink targets"
            )
    code = getattr(result, "return_code", None)
    if type(code) is not int or code != 0:
        retained = (
            out if retain_failed and type(code) is int and isinstance(out, str) else ""
        )
        return retained, f"the firewall audit command failed (status {code})"
    return (out if isinstance(out, str) else ""), ""


def _agent_cmd(command: str, sandbox_user: str) -> str:
    """The mandatory shared launch, with protected root-side tools."""
    from benchflow.sandbox.lockdown import build_priv_drop_cmd

    return _audit_prelude() + build_priv_drop_cmd(
        command, sandbox_user, protected_tools=True
    )


async def _protected_python(env: Any) -> str | None:
    """PT's runtime is the only authority for root's Python entry point.

    The API lands with PT, after this integration base. An absent API or
    unavailable kit means no Python reader, never an image-path fallback.
    A broken existing kit propagates its refusal instead of hiding it.
    """
    try:
        runtime = importlib.import_module("benchflow._runtime_tree")
    except ModuleNotFoundError as error:
        if error.name != "benchflow._runtime_tree":
            raise
        return None
    # PT's route lookup reads this generation's kit record first (a restored
    # snapshot, a branch child), so the reader sees the kit BenchFlow made;
    # a broken kit raises here.
    lookup = getattr(env, "runtime_kit_for_route", None)
    if inspect.iscoroutinefunction(lookup):
        await lookup()
    command: Any = runtime.runtime_kit_python(env)
    if inspect.isawaitable(command):
        command = cast(Any, await command)
    return command[0] if command else None


async def check_firewall_power(
    env: Any,
    sandbox_user: str,
    *,
    holds: tuple[str, ...],
    can_drop: bool = True,
) -> tuple[dict[str, Any], str | None]:
    """Disarm or refuse an image that would hand ``sandbox_user`` the firewall
    the rules in ``holds`` hold it with; return ``(record, refusal)``.

    ``refusal`` is the message of the :class:`FirewallPowerError` the caller
    raises after writing the record, or None when the agent may start. With
    the drop kept, every later launch of ``sandbox_user``'s agent in ``env``
    starts without NET_ADMIN (:func:`net_admin_dropped`). ``can_drop`` is
    False when BenchFlow does not start the agent's commands itself through
    ``build_priv_drop_cmd`` (a session-factory agent).
    """
    from benchflow.sandbox.lockdown import (
        _cloud_metadata_barrier_verified,
        _cloud_metadata_construction_cmd,
        build_priv_drop_cmd,
    )

    started = _timing_spans.clock()
    forget_net_admin_drop(env)
    held_by = " and ".join(holds)
    python_command = await _protected_python(env)
    out, failed = await _exec(
        env,
        _root_cmd(scan_all=not can_drop, python_command=python_command),
        _ROOT_EXEC_SEC,
    )
    line = next(
        (ln for ln in out.splitlines() if ln.startswith("BF fwpower-root ")), ""
    )
    root = _fields(line)
    if not failed and root.get("net_admin") == "0":
        # Root's read comes first. With no NET_ADMIN in the sandbox the rest
        # could only end "clear" (no verified barrier) or "disarmed" (a probe
        # through a verified barrier finds no capability that is not there),
        # and every agent launch takes the shared barrier whatever this check
        # records (build_priv_drop_cmd), so the rest is not run. Main
        # starts so where the metadata helper adds the rules.
        return (
            firewall_power_record(
                "clear",
                "no process in the sandbox can hold NET_ADMIN (it is not in "
                "root's bounding set), so nothing in the image can give the "
                "agent the firewall",
                holds=list(holds),
                seconds=round(_timing_spans.clock() - started, 3),
            ),
            None,
        )
    construction, construction_failure = (
        await _exec(
            env,
            _cloud_metadata_construction_cmd(sandbox_user, protected_tools=True),
            _ROOT_EXEC_SEC,
        )
        if can_drop
        else ("", "no shared launcher")
    )
    barrier_verified = not construction_failure and _cloud_metadata_barrier_verified(
        construction, 0
    )
    if failed and not barrier_verified:
        why = f"the firewall audit's root step could not run safely: {failed}"
        return (
            firewall_power_record(
                "refused",
                why,
                holds=list(holds),
                seconds=round(_timing_spans.clock() - started, 3),
            ),
            f"sandbox startup refused: {why}. Repair the task's image or "
            "select a backend with protected system-shell support before retrying.",
        )
    scan = _scan_report(out)
    scanned = _scanned(out)
    if barrier_verified:
        probe = f"BFFW{secrets.token_hex(4)}"
        agent_out, agent_failed = await _exec(
            env,
            _agent_cmd(
                _agent_check_cmd(probe, [e["path"] for e in scanned]), sandbox_user
            ),
            _AGENT_EXEC_SEC,
            retain_failed=True,
        )
        agent_line = next(
            (ln for ln in agent_out.splitlines() if ln.startswith("BF fwpower-agent ")),
            "",
        )
        agent = _fields(agent_line)
        power = agent.get("power", "")
        unsafe = power or any(
            _has_net_admin(agent.get(k, "")) for k in ("prm", "eff", "amb")
        )
        if power:
            await _exec(env, _cleanup_cmd(probe), _ROOT_EXEC_SEC)
        complete = (
            bool(scan)
            and scan.get("complete") != "0"
            and _caps_read(scan)
            and bool(agent_line)
            and not agent_failed
        )
        kept = _keep_drop(env, sandbox_user)
        refused = bool(unsafe) or not kept
        why = (
            "positive evidence of agent firewall power despite the verified launch barrier"
            if unsafe
            else "the verified launch barrier could not be kept for later launches"
            if not kept
            else "S2 verified the shared kernel launch barrier; supplemental inventories cannot grant privileges or revoke that proof"
        )
        return (
            firewall_power_record(
                "refused" if refused else "disarmed",
                why,
                mandatory_barrier="verified",
                agent_capabilities={
                    "bounding": agent.get("bnd", ""),
                    "inheritable": agent.get("inh", ""),
                    "ambient": agent.get("amb", ""),
                    "permitted": agent.get("prm", ""),
                    "effective": agent.get("eff", ""),
                },
                supplemental_inventory="complete" if complete else "unavailable",
                inventory_why=""
                if complete
                else (
                    failed
                    or agent_failed
                    or "optional inventory tools or report unavailable"
                ),
                holds=list(holds),
                drop="setpriv" if kept else None,
                found=_runnable(scanned, agent_out),
                power=power.split(",") if power else [],
                file_capabilities_checked=_caps_read(scan),
                seconds=round(_timing_spans.clock() - started, 3),
            ),
            f"sandbox startup refused: {why}" if refused else None,
        )
    setpriv = root.get("setpriv", "")
    if not line:
        no_drop = f"root's check did not report ({failed or 'no output'})"
    elif not can_drop:
        no_drop = (
            "BenchFlow runs this agent's commands one by one in the sandbox, "
            "not through a launch it can drop NET_ADMIN from"
        )
    elif setpriv != "ok" and root.get("nnp") != "1":
        # BusyBox's setpriv cannot switch users or change the bounding set,
        # but no_new_privs alone is a drop (build_priv_drop_cmd).
        no_drop = (
            "the image has no setpriv (util-linux's or BusyBox's)"
            if setpriv == "none"
            else "the image's setpriv cannot set no_new_privs"
        )
    else:
        no_drop = ""

    probe = f"BFFW{secrets.token_hex(4)}"
    paths = [e["path"] for e in scanned]
    if not no_drop:
        agent_out, failed = await _exec(
            env,
            build_priv_drop_cmd(
                _agent_check_cmd(probe, paths),
                sandbox_user,
                drop_net_admin=True,
                root_tools=True,
            ),
            _AGENT_EXEC_SEC,
            required=_ROOT_LAUNCH_TOOLS,
            optional=("su",),
        )
        agent_line = next(
            (ln for ln in agent_out.splitlines() if ln.startswith("BF fwpower-agent ")),
            "",
        )
        agent = _fields(agent_line)
        sets = [agent.get(k, "") for k in ("bnd", "inh", "amb", "prm", "eff")]
        bnd, inh, amb, prm, eff = sets
        if not agent_line:
            no_drop = (
                "the agent user's commands could not be started with "
                f"no_new_privs ({failed or 'setpriv refused the drop'})"
            )
        elif any(_has_net_admin(s) for s in (prm, eff, amb)) or not (
            bnd and prm and eff
        ):
            no_drop = (
                "setpriv ran, but the agent user's commands still hold NET_ADMIN "
                f"(CapPrm {prm or '?'}, CapEff {eff or '?'}, CapAmb {amb or '?'})"
            )
        elif agent.get("nnp") != "1":
            # A setuid-root program would still make the agent uid 0, whose
            # traffic none of the uid rules match.
            no_drop = (
                "setpriv ran, but the agent user's commands can still gain "
                "privileges from setuid programs and file capabilities "
                f"(NoNewPrivs {agent.get('nnp') or '?'})"
            )
        elif not agent.get("power") and _keep_drop(env, sandbox_user):
            found = _runnable(scanned, agent_out)
            what = (
                "; the image's "
                + ", ".join(_describe(e) for e in found)
                + " can no longer give it either"
                if found
                else ""
            )
            # Under no_new_privs no exec adds to the permitted set, so an
            # agent whose sets start without NET_ADMIN never gets it, even
            # where BusyBox's setpriv left it in the bounding set.
            bounded = not (_has_net_admin(bnd) or _has_net_admin(inh))
            how = (
                "start without NET_ADMIN in their bounding and inheritable "
                f"sets and with no_new_privs (setpriv {DROP_NET_ADMIN_FLAGS})"
                if bounded
                else "start with no_new_privs (setpriv --no-new-privs, as root "
                "before su) and without NET_ADMIN in their permitted, "
                "effective and ambient sets, which no exec can then add to"
            )
            return (
                firewall_power_record(
                    "disarmed",
                    f"the agent's commands {how}, so no file capability or "
                    "setuid program can give it NET_ADMIN or uid 0, the "
                    f"power to undo or go around {held_by}{what}",
                    holds=list(holds),
                    drop="setpriv" if bounded else "nnp",
                    found=found,
                    agent_capabilities={
                        "bounding": bnd,
                        "inheritable": inh,
                        "ambient": amb,
                        "permitted": prm,
                        "effective": eff,
                    },
                    file_capabilities_checked=_caps_read(scan),
                    seconds=round(_timing_spans.clock() - started, 3),
                ),
                None,
            )
        elif agent.get("power"):
            no_drop = (
                "setpriv ran, but a firewall tool still changed the firewall "
                f"for the agent user ({agent['power']})"
            )
        else:
            no_drop = "the drop could not be kept for this sandbox's later launches"
        if agent.get("power"):
            await _exec(env, _cleanup_cmd(probe), _ROOT_EXEC_SEC)
        if scan.get("scope") != "all":
            # The drop was expected and failed: no drop holds, so root
            # scans the whole file system after all.
            out, failed = await _exec(
                env,
                _scan_cmd(whole_fs=True, python_command=python_command),
                _ROOT_EXEC_SEC,
            )
            scan = _scan_report(out)
            scanned = _scanned(out)
            paths = [e["path"] for e in scanned]

    if can_drop:
        # The shared launcher is the only route allowed to run agent code.
        # A missing/failed probe may be caused by an agent-written firewall
        # tool: retrying as the bare sandbox identity would let that tool run
        # without no_new_privs or the reduced bounding set (S3-CONF-1).
        # Refuse before the session-only audit, even when root's prerequisites
        # failed and the restricted probe was never attempted.
        inventory = (
            "; root found " + ", ".join(_describe(e) for e in scanned[:_FOUND_LIMIT])
            if scanned
            else ""
        )
        if scan.get("complete") == "0":
            no_drop = f"root's filesystem scan was incomplete; {no_drop}"
        why = f"the restricted agent launch could not be verified: {no_drop}"
        return (
            firewall_power_record(
                "refused",
                why + inventory,
                holds=list(holds),
                drop=None,
                no_drop=no_drop,
                found=scanned[:_FOUND_LIMIT],
                file_capabilities_checked=_caps_read(scan),
                seconds=round(_timing_spans.clock() - started, 3),
            ),
            f"sandbox startup refused: {why}{inventory}. Install util-linux "
            "setpriv with no_new_privs and bounding-set support in the task's "
            "image, and fix the image's agent-user setup or failing firewall "
            "probe before retrying. BenchFlow will not retry agent code "
            "without the restricted launcher.",
        )

    # Only real session-factory transports reach this direct identity audit.
    agent_out, failed = await _exec(
        env,
        _agent_check_cmd(probe, paths),
        _AGENT_EXEC_SEC,
        user=sandbox_user,
    )
    agent_line = next(
        (ln for ln in agent_out.splitlines() if ln.startswith("BF fwpower-agent ")),
        "",
    )
    agent = _fields(agent_line)
    if agent.get("power"):
        await _exec(env, _cleanup_cmd(probe), _ROOT_EXEC_SEC)
    found = _runnable(scanned, agent_out)
    held = any(_has_net_admin(agent.get(k, "")) for k in ("prm", "eff", "amb"))
    checked = _caps_read(scan)
    extra: dict[str, Any] = {
        "holds": list(holds),
        "drop": None,
        "no_drop": no_drop,
        "found": found,
        "file_capabilities_checked": checked,
        "seconds": round(_timing_spans.clock() - started, 3),
    }
    if not scan or not agent_line or scan.get("complete") == "0":
        # Fail closed: nothing proves the agent user cannot change the rules.
        why = (
            f"root's scan did not report ({failed or 'no output'})"
            if not scan
            else "root's filesystem scan was incomplete"
            if scan.get("complete") == "0"
            else f"the agent user's check did not report ({failed or 'no output'})"
        )
        return (
            firewall_power_record(
                "refused", f"{why}, and {no_drop}", power=[], **extra
            ),
            f"sandbox startup refused: BenchFlow could not check that the agent "
            f"user {sandbox_user!r} cannot undo {held_by}: {why}, and BenchFlow "
            f"cannot take NET_ADMIN away from its commands: {no_drop}.",
        )
    whats: list[str] = []
    if held:
        whats.append(
            "its own process (NET_ADMIN in its permitted, effective or ambient set)"
        )
    if agent.get("power"):
        whats.append(f"a firewall tool ({agent['power'].replace(',', ', ')})")
    whats.extend(_describe(e) for e in found)
    if whats:
        what = "; ".join(whats)
        record = firewall_power_record(
            "refused",
            f"the agent user can change the firewall through {what}, and {no_drop}",
            power=agent.get("power", "").split(",") if agent.get("power") else [],
            **extra,
        )
        refusal = (
            f"sandbox startup refused: the agent user {sandbox_user!r} can "
            f"change the firewall through {what}, so it could undo {held_by}, "
            f"and BenchFlow cannot take NET_ADMIN away from its commands: "
            f"{no_drop}. Remove the capability or setuid bit in the "
            "task's Dockerfile (setcap -r <file>, chmod u-s <file>)"
            + (
                ", or put a setpriv that can set no_new_privs (util-linux's or "
                "BusyBox's) in the image so BenchFlow can disarm the agent's "
                "commands."
                if can_drop
                else "."
            )
        )
        return record, refusal
    if not checked:
        # Fail closed: a file capability (cap_net_admin or cap_setuid on
        # python3, say) is invisible to the agent user's probe.
        why = (
            "python3 could not read the image's file capabilities"
            if scan.get("caps") == "unread"
            else "the image has no getcap (libcap) or python3 to read its file "
            "capabilities with"
        )
        return (
            firewall_power_record(
                "refused", f"{why}, and {no_drop}", power=[], **extra
            ),
            f"sandbox startup refused: BenchFlow could not check that the agent "
            f"user {sandbox_user!r} cannot undo {held_by}: {why}, and BenchFlow "
            f"cannot take NET_ADMIN away from its commands: {no_drop}. Put "
            "libcap's getcap (libcap2-bin on Debian and Ubuntu, libcap-getcap on "
            "Alpine) in the task's image so BenchFlow can read its file "
            "capabilities"
            + (
                ", or a setpriv that can set no_new_privs (util-linux's or "
                "BusyBox's) so it can disarm them."
                if can_drop
                else "."
            ),
        )
    return (
        firewall_power_record(
            "clear",
            f"{no_drop}; the agent user holds no NET_ADMIN, and no firewall tool, "
            "file capability or setuid program found gives it the power to undo "
            f"{held_by}",
            power=[],
            **extra,
        ),
        None,
    )
