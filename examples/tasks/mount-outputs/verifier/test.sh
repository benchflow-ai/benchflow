#!/bin/bash
# Reward 1 for the right count, and only if the hook ran first.
if [ "$(cat /app/count.txt 2>/dev/null)" = "3" ] && [ -s /root/snap/listing.txt ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
