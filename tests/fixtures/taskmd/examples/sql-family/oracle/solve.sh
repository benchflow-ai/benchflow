#!/bin/bash
# The reference query for this instance, computed from its parameters. Under family@1 the runtime passes them as
# TASKMD_PARAMS (JSON) and the seed as TASKMD_SEED; the agent gets neither.
set -euo pipefail
python3 - <<'PY'
import json, os
p = json.loads(os.environ["TASKMD_PARAMS"])
expr = {"revenue": "SUM(oi.quantity * oi.unit_price)", "order_count": "COUNT(DISTINCT o.id)",
        "items_bought": "SUM(oi.quantity)"}[p["metric"]]
sql = (f"SELECT o.customer_id AS customer_id, {expr} AS {p['metric']}\n"
       f"FROM orders o JOIN order_items oi ON oi.order_id = o.id\n"
       f"WHERE o.status = 'completed' AND substr(o.placed_at, 1, 4) = '{p['year']}'\n"
       f"GROUP BY o.customer_id\n"
       f"ORDER BY {p['metric']} DESC, customer_id ASC\n"
       f"LIMIT {p['k']};\n")
with open("/work/answer.sql", "w", encoding="utf-8") as f:
    f.write(sql)
PY
