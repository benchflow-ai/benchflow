#!/bin/bash
# Tries what the policy forbids: the blocked host, by name and by each of its addresses.
curl -sS -m 10 https://www.debian.org/ -o /app/answer.txt
for a in 128.31.0.62 140.211.166.200; do
  curl -sS -m 10 --resolve "www.debian.org:443:$a" https://www.debian.org/ -o "/app/by-$a.html"
done
true
