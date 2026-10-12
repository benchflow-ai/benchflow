#!/bin/bash
# Reward 1 for no network: the reserved address is refused at once (a
# firewall), or has no route while the sandbox has no address of its own
# (Docker's network none). A no-route answer from a sandbox that has an
# address is an open network on a host with no outside route, so it does
# not count. The local listener must answer.
ok=1
grep -qxE 'reserved=refused' /app/net.txt || { grep -qxE 'reserved=unreachable' /app/net.txt && grep -qxE 'own=no-address' /app/net.txt; } || ok=0
grep -qxE 'local=(answered)' /app/net.txt || ok=0
echo $ok > /logs/verifier/reward.txt
