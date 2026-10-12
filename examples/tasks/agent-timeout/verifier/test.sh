#!/bin/bash
if [ "$(cat /app/done.txt 2>/dev/null)" = done ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
