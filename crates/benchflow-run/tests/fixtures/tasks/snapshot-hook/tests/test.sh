#!/bin/bash
# Reads only what the hook saved; without the hook there is nothing to read.
if printf 'ready' | cmp -s - /root/snapshot/status.txt && [ "$(cat /root/snapshot/by.txt)" = root ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
