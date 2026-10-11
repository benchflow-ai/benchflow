#!/bin/bash
# The output matches the mount, the hook ran as root, and the mount is root's.
ok=1
grep -qx 'mounted-e2e-42' /app/out.txt || ok=0
[ "$(cat /root/snap/by.txt 2>/dev/null)" = root ] || ok=0
[ "$(cat /root/snap/owner.txt 2>/dev/null)" = root ] || ok=0
echo $ok > /logs/verifier/reward.txt
