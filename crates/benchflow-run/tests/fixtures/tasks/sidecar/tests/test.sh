#!/bin/bash
if printf 'sidecar-7f3a' | cmp -s - /app/token.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
