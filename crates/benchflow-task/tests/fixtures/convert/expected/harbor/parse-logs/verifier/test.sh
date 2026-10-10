#!/bin/bash
python3 -c "import json; d=json.load(open('/app/out/status.json')); print(1.0 if d == {'200': 1, '404': 1} else 0.0)" > /logs/verifier/reward.txt
