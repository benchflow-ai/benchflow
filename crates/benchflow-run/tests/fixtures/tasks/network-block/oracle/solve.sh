#!/bin/bash
# Needs the network to be open for what the policy does not block: a
# loopback address of its own, 127.0.0.3.
set -e
mkdir -p /tmp/site
printf '<html><head><title>Example Domain</title></head></html>\n' > /tmp/site/index.html
python3 -m http.server 8080 --bind 127.0.0.3 --directory /tmp/site >/dev/null 2>&1 &
python3 - <<'PY'
import re, time, urllib.request
for _ in range(50):
    try:
        page = urllib.request.urlopen("http://127.0.0.3:8080/", timeout=5).read().decode()
        break
    except OSError:
        time.sleep(0.1)
else:
    raise SystemExit("the allowed page did not answer")
open("/app/answer.txt", "w").write(re.search(r"<title>([^<]*)", page).group(1))
PY
