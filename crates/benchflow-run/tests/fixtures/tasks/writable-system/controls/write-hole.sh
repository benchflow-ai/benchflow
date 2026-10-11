#!/bin/bash
# Solves the task only if the agent user can still write one of the image's
# holes (/opt/tools, /etc/environment, /usr/lib): it must fail once the
# runner has locked them.
if touch /opt/tools/.bf-control 2>/dev/null || { : >> /etc/environment; } 2>/dev/null || touch /usr/lib/.bf-control 2>/dev/null; then
  printf 'Hello, world!' > /app/hello.txt
fi
