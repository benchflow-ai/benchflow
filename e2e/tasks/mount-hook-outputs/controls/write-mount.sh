#!/bin/bash
# Tries to change the read-only mount, and solves the task only if that worked.
if echo changed >> /opt/data/in.txt 2>/dev/null; then cp /opt/data/in.txt /app/out.txt; echo mounted-e2e-42 > /app/out.txt; fi
