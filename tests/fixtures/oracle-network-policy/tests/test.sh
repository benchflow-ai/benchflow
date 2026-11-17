#!/bin/bash
# The verifier's network is [verifier] network_mode, else the task's allowlist:
# public in the test's root and task-user variants (so it reaches the host the
# oracle may not), unset in its inherited variant (so it is refused there).
if python3 -c 'import urllib.request; urllib.request.urlopen("https://example.com/", timeout=20).read(64)'; then
  echo "verifier_reached=reached"
else
  echo "verifier_reached=refused"
fi
echo "oracle_reached=$(cat /tmp/oracle-net/oracle_reached.txt 2>/dev/null || echo missing)"
echo "oracle_allowed=$(cat /tmp/oracle-net/oracle_allowed.txt 2>/dev/null || echo missing)"
echo "oracle_apt_lists=$(cat /tmp/oracle-net/oracle_apt_lists.txt 2>/dev/null || echo missing)"
echo "oracle_uid=$(cat /tmp/oracle-net/oracle_uid.txt 2>/dev/null || echo missing)"
echo "oracle_direct=$(cat /tmp/oracle-net/oracle_direct.txt 2>/dev/null || echo missing)"
if [ "$(cat /tmp/oracle-net/oracle_reached.txt 2>/dev/null)" = reached ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
