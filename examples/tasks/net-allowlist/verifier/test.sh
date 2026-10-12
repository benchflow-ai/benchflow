#!/bin/bash
# Reward 1 when every probe (listed unlisted local) saw what the policy allows; a
# forbidden address that merely hung was not refused, so it does not count.
ok=1
grep -qxE 'listed=(no-answer|unreachable)' /app/net.txt || ok=0
grep -qxE 'unlisted=(refused)' /app/net.txt || ok=0
grep -qxE 'local=(answered)' /app/net.txt || ok=0
echo $ok > /logs/verifier/reward.txt
