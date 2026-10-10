#!/bin/bash
grep -q "2 rows" /app/report.md && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
