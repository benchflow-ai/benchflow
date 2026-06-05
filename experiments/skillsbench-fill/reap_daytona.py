#!/usr/bin/env python3
"""VM-side safety net: delete orphaned Daytona sandboxes (STARTED and older than
90 min = no live run could still own it). Run via benchflow's venv from cron."""
import os, re, datetime
key = None
for line in open(os.path.expanduser("~/keys.env")):
    m = re.match(r'^\s*(?:export\s+)?DAYTONA_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
    if m:
        key = m.group(1)
from benchflow.sandbox.daytona import build_sync_client
c = build_sync_client(key)
now = datetime.datetime.now(datetime.timezone.utc)
deleted = 0
# VM SDK's list() takes no page/limit kwargs; .items holds the Sandbox objects.
r = c.list()
items = list(getattr(r, "items", None) or r)
if True:
    for s in items:
        if "STARTED" not in str(getattr(s, "state", "")):
            continue
        cr = getattr(s, "created_at", None) or getattr(s, "createdAt", None)
        try:
            if not isinstance(cr, datetime.datetime):
                cr = datetime.datetime.fromisoformat(str(cr).replace("Z", "+00:00"))
            if cr.tzinfo is None:
                cr = cr.replace(tzinfo=datetime.timezone.utc)
            age_min = (now - cr).total_seconds() / 60
        except Exception:
            continue
        if age_min > 90:
            try:
                c.get(s.id).delete(); deleted += 1
            except Exception:
                pass
print(f"[reap {now.strftime('%H:%M:%SZ')}] deleted {deleted} orphan sandboxes (>90min STARTED)")
