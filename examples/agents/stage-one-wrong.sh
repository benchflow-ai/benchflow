#!/bin/bash
# For tasks/stages: fails the gated first stage, so the second never unlocks.
printf 'wrong' > /app/one.txt
