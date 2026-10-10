#!/bin/bash
# The whole task (and stage second, which has no verifier of its own): both files hold their words.
if printf 'alpha' | cmp -s - /app/one.txt && printf 'beta' | cmp -s - /app/two.txt; then r=1; else r=0; fi
echo "$r" > /logs/verifier/reward.txt
