#!/bin/bash
# Start the hostile service from the agent user's home and leave it running;
# its output goes to /tmp, so it holds nothing under /logs.
mkdir -p /tmp/kept
cd ~ || exit 1
nohup python3 /opt/svc/server.py 5000 >/tmp/kept/service.out 2>&1 &
for _ in $(seq 1 50); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/ready', timeout=1)" 2>/dev/null && exit 0
    sleep 0.1
done
exit 0
