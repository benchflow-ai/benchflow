#!/bin/bash
# Start the service and leave it running; the verifier connects to it.
nohup python3 /opt/svc/server.py 5000 >/tmp/service.out 2>&1 &
for _ in $(seq 1 50); do
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=1)" 2>/dev/null && exit 0
    sleep 0.1
done
exit 0
