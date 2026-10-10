#!/bin/bash
grep -n "ERROR [0-9]" /app/server.log | head -1 | cut -d: -f1 > /app/answer.txt
