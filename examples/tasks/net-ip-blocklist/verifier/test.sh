#!/bin/bash
# Reward 1 when the forbidden address was refused at once (the firewall said
# no) and the allowed listener answered (the probe itself works). "no-answer"
# means nothing refused it, so it does not count.
if printf "forbidden=refused\nlocal=answered\nblocked-local=refused\n" | cmp -s - /app/net.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
