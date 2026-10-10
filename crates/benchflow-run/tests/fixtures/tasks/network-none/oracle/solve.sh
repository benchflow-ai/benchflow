#!/bin/bash
# Needs no network: sums the numbers in place.
awk '{ s += $1 } END { print s }' /app/numbers.txt > /app/answer.txt
