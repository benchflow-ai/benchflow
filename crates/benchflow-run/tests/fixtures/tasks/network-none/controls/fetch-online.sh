#!/bin/bash
# Tries what the policy forbids: fetch a page from the internet into the workdir.
curl -sS -m 10 https://example.com/ -o /app/answer.txt
# By address, with no DNS: Cloudflare's trace page on 1.1.1.1 (example.com's old
# address stopped serving it), which names the address it was served from.
curl -sS -m 10 https://1.1.1.1/cdn-cgi/trace -o /app/by-address.txt
true
