#!/bin/bash
# Reward 1 for an open network: a server on the sandbox's own network
# address answers (with no network there is no such address), and the
# reserved address is not refused: it hangs, or has no route on a host with
# no outside route. This cannot tell open from a backend block that drops
# packets silently (see the README's E54 entry for the host it needs).
ok=1
grep -qxE 'reserved=(no-answer|unreachable)' /app/net.txt || ok=0
grep -qxE 'own=(answered)' /app/net.txt || ok=0
grep -qxE 'local=(answered)' /app/net.txt || ok=0
echo $ok > /logs/verifier/reward.txt
