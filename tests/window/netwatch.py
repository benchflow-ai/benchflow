"""The window check's kernel poller: run as root, it reads the uid rules in each probe sandbox's network namespace.

    sudo -n python3 -I tests/window/netwatch.py LOG NAME_PART

Every ~40 ms it lists running containers whose name holds NAME_PART, enters each one's network namespace
(``nsenter -t <pid> -n``) and counts the ``--uid-owner`` lines ``iptables-save`` and ``ip6tables-save`` print there
(each of the nft and legacy variants that exists). Each change is one JSON line,
``{"t": time_ns, "container": name, "uid_rules": n}``. It reads only; it changes nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

SAVES = [
    s
    for s in (
        "iptables-nft-save",
        "iptables-legacy-save",
        "ip6tables-nft-save",
        "ip6tables-legacy-save",
    )
    if shutil.which(s)
] or ["iptables-save"]


def _containers(part: str) -> dict[str, int]:
    names = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}", "--filter", f"name={part}"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.split()
    out = {}
    for name in names:
        pid = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if pid.isdigit() and int(pid) > 0:
            out[name] = int(pid)
    return out


def _count(pid: int) -> int:
    n = 0
    for save in SAVES:
        done = subprocess.run(
            ["nsenter", "-t", str(pid), "-n", save],
            capture_output=True,
            text=True,
            timeout=10,
        )
        n += sum("--uid-owner" in line for line in done.stdout.splitlines())
    return n


def main() -> int:
    log, part = sys.argv[1], sys.argv[2]
    last: dict[str, int] = {}
    with open(log, "a", buffering=1) as fh:
        while True:
            for name, pid in _containers(part).items():
                try:
                    n = _count(pid)
                except (subprocess.SubprocessError, OSError):
                    continue
                if last.get(name) != n:
                    last[name] = n
                    fh.write(
                        json.dumps(
                            {"t": time.time_ns(), "container": name, "uid_rules": n}
                        )
                        + "\n"
                    )
            time.sleep(0.04)


if __name__ == "__main__":
    sys.exit(main())
