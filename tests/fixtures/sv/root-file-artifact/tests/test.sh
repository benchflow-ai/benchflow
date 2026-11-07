#!/bin/bash
if [ "$(cat /result.txt 2>/dev/null)" = "42" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
