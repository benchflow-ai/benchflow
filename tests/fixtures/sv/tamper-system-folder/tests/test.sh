#!/bin/bash
t=/usr/local/bin/sv-tool
if [ -f "$t" ] && [ -x "$t" ] && [ "$("$t")" = "42" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
