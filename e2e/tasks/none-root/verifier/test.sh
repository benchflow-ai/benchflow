#!/bin/bash
# 1 when /app/hello.txt is exactly right.
if printf 'Hello, world!' | cmp -s - /app/hello.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
