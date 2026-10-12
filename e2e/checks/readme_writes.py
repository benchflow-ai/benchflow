"""Check that following examples/README.md writes nothing among its tracked files.

Every `bench check ... --fix` works on a copy in scratch/, every `bench import` and
`bench export` writes under scratch/ (or jobs/), and a verifier edit names the copy.
Usage: python3 readme_writes.py <examples/README.md>; exits 1 and names each command
that would write elsewhere.
"""

import re
import sys

text = open(sys.argv[1]).read()
spans = re.findall(r"`([^`]*bench [^`]*)`", text)
bad = []
for cmd in spans:
    for part in re.split(r"&&|;|\|", cmd):
        words = part.split()
        if "bench" not in words:
            continue
        w = words[words.index("bench") + 1:]
        if not w:
            continue
        if w[0] == "check" and "--fix" in w:
            targets = [x for x in w[1:] if not x.startswith("-")]
            if any(not t.startswith("scratch/") for t in targets):
                bad.append(cmd)
        elif w[0] in ("import", "export"):
            args = [x for x in w[1:] if not x.startswith("-")]
            if "--format" in w:
                fmt = w[w.index("--format") + 1]
                args = [a for a in args if a != fmt]
            out = args[1:2]
            if out and not out[0].startswith(("scratch/", "jobs/")):
                bad.append(cmd)
for m in re.finditer(r"change `42` to `41` in ([^,|]*)", text):
    if "scratch/" not in m.group(1):
        bad.append(m.group(0))
for b in bad:
    print("writes among the tracked files:", b)
sys.exit(1 if bad else 0)
