#!/bin/bash
if [ "$(cat /app/a.txt 2>/dev/null)" = 1 ] && [ "$(cat /app/b.txt 2>/dev/null)" = 2 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
