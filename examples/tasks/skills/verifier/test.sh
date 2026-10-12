#!/bin/bash
if printf periwinkle-42 | cmp -s - /app/word.txt; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
