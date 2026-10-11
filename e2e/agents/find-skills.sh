#!/bin/bash
# Stands in for an agent: lists every SKILL.md it can see, then solves hello.
find / -name SKILL.md -not -path '/proc/*' 2>/dev/null | sort > /app/skills-seen.txt
printf 'Hello, world!' > /app/hello.txt
