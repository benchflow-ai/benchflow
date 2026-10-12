#!/bin/bash
if [ "$(cat /app/year.txt)" = 2030 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
