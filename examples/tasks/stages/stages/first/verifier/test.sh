#!/bin/bash
# Stage first: /app/one.txt holds exactly `alpha`.
if printf 'alpha' | cmp -s - /app/one.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
