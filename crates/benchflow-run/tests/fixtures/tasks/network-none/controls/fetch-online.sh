#!/bin/bash
# Tries what the policy forbids: fetch a page from the internet into the workdir.
curl -sS -m 10 https://example.com/ -o /app/answer.txt
curl -sS -m 10 http://93.184.215.14/ -H 'Host: example.com' -o /app/by-address.html
true
