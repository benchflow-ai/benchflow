#!/bin/bash
if [ "$(cat /app/zone.txt)" = JST ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
