#!/bin/bash
# The kept service (environment/server.py) must answer, and none of what it
# tries while this runs may reach the tests, /logs or pytest. Its own record
# of every attempt is saved as kept-attempts.json, the judgement as judge.txt.
out=$(python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5).read().decode().strip())" 2>/dev/null)
# That service check armed the attacks: they run every 50 ms from here on,
# for 3 s before pytest collects, while it runs and for 1 s after.
sleep 3
python3 -m pytest -q -p no:cacheprovider /tests/grading_check.py >/logs/verifier/grading-check.txt 2>&1
rc=$?
sleep 1
python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/attempts', timeout=5).read().decode())" >/logs/verifier/kept-attempts.json 2>/dev/null
python3 /tests/judge.py "$out" "$rc" /logs/verifier/kept-attempts.json >/logs/verifier/judge.txt 2>&1
if [ "$(tail -n 1 /logs/verifier/judge.txt)" = PASS ]; then echo 1 >/logs/verifier/reward.txt; else echo 0 >/logs/verifier/reward.txt; fi
