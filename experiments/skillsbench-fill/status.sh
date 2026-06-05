#!/usr/bin/env bash
# One-glance status of the SkillsBench fill. Run: bash ~/sb-fill/status.sh
cd "$(dirname "$0")"
python3 -c "
import json,glob,collections,os
ex={'experiments_ledger.json','queue.jsonl','reconcile_report.json','grid.json'}
st=collections.Counter(); rm=collections.Counter(); rv=collections.Counter()
for f in glob.glob('state/*.json'):
    if os.path.basename(f) in ex: continue
    try: d=json.load(open(f))
    except: continue
    st[d.get('status')]+=1
    if d.get('status')=='running': rm[d.get('model')]+=1
for f in glob.glob('review/*.json'):
    try: rv[json.load(open(f)).get('verdict')]+=1
    except: pass
print('state    :', dict(st))
print('running  :', dict(rm))
print('review   :', dict(rv), '| published:', len(glob.glob('published/*.json')))
"
echo "runners  : $(ps -eo args | grep -c '[r]unner.py')   cron jobs: $(crontab -l 2>/dev/null | grep -c sb-fill)"
echo "wave log : $(tail -1 logs/wave.log 2>/dev/null)"
