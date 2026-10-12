#!/bin/bash
# The last stage (and the whole task): both files hold their words.
if printf 'alpha' | cmp -s - /app/one.txt && printf 'beta' | cmp -s - /app/two.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
