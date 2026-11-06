#!/bin/bash
# The page exists, keeps its CSS inside, and states the facts the brief gives. How good it looks is for people
# to judge: see `bench preferences`.
python3 -I - <<'PY'
import pathlib

page = pathlib.Path("/app/out/index.html")
html = page.read_text(errors="replace").lower() if page.is_file() else ""
missing = [what for what, needle in [
    ("the page", "<html"), ("inline CSS", "<style"), ("the name", "fix-it friday"),
    ("the address", "40 mill street"), ("the hours", "6"), ("a lamp", "lamp"), ("a bike", "bike"),
] if needle not in html]
print("PASS" if not missing else "FAIL: missing " + ", ".join(missing))
open("/logs/verifier/reward.txt", "w").write("0\n" if missing else "1\n")
PY
