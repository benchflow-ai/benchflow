#!/bin/bash
# Stage first: /app/one.txt holds exactly `alpha`.
if printf 'alpha' | cmp -s - /app/one.txt; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
