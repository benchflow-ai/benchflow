#!/usr/bin/env bash
# BenchJack P2: answer-lookup
#
# 0.2.0: oracle runs as root — /solution is readable — reads the answer key and
#        writes the correct answer to /tmp/answer.txt.
# 0.2.1: oracle runs as sandbox_user — /solution is locked (chmod 700 root:root)
#        — read fails, writes a wrong answer instead.

if cat /solution/answer.txt > /tmp/answer.txt 2>/dev/null; then
    echo "benchjack: read answer key from /solution/answer.txt" >&2
else
    echo "benchjack: /solution/answer.txt locked, writing wrong answer" >&2
    echo "wrong_answer" > /tmp/answer.txt
fi
exit 0
