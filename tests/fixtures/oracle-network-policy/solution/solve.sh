#!/bin/bash
# A host outside the allowlist, one the allowlist names, and an address
# outside it reached directly (no proxy): the firewall, not only the proxy
# variables, must stop that one.
# Outside the workspace, which a task user may not own; the verifier reads it.
mkdir -p /tmp/oracle-net
id -u > /tmp/oracle-net/oracle_uid.txt
# The image as built: arming the policy must not leave apt's lists behind.
(ls /var/lib/apt/lists 2>/dev/null | grep -c _Packages || true) > /tmp/oracle-net/oracle_apt_lists.txt
fetch() {
  python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=20).read(64)' "$1"
}
if fetch https://example.com/; then
  echo reached > /tmp/oracle-net/oracle_reached.txt
else
  echo refused > /tmp/oracle-net/oracle_reached.txt
fi
if fetch https://pypi.org/simple/; then
  echo reached > /tmp/oracle-net/oracle_allowed.txt
else
  echo refused > /tmp/oracle-net/oracle_allowed.txt
fi
if python3 -c 'import socket; socket.create_connection(("1.1.1.1", 443), timeout=10).close()'; then
  echo reached > /tmp/oracle-net/oracle_direct.txt
else
  echo refused > /tmp/oracle-net/oracle_direct.txt
fi
