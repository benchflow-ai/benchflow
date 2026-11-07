#!/usr/bin/env python3
"""The generator of the top-k customers family, under family@1 (https://task.md/docs/runtime/episodes/#family1).

    family/generate.py --seed N --out DIR

writes

    DIR/instance.json             {"params": {...}, "placeholders": {...}, "seed": N}
    DIR/agent/data/shop.db        copied into the sandbox at /data/shop.db
    DIR/verifier/hidden.db        the database answer.sql is graded on
    DIR/verifier/expected.json    the reference rows for hidden.db

Stdlib and sqlite3 only, with no network and no clock. The same seed always gives the same parameters and the same
rows. The database files' bytes also depend on the SQLite library that writes them, whose version is in the file
header, so a runtime runs the generator in the task's pinned image.
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

PARAMS = {"k": [3, 5, 10], "metric": ["revenue", "order_count", "items_bought"], "year": [2023, 2024, 2025]}
PRODUCTS = [("pen", 1.25), ("notebook", 3.5), ("stapler", 7.99), ("lamp", 24.0), ("chair", 89.0), ("desk", 249.0),
            ("cable", 4.75), ("mouse", 15.5), ("monitor", 179.99), ("headset", 59.0)]
COUNTRIES = ["US", "DE", "FR", "JP", "BR", "IN", "CA"]
NAMES = ["Ada", "Bo", "Cy", "Di", "Ed", "Flo", "Gus", "Hal", "Ivy", "Jo", "Kai", "Lu", "Mo", "Ned", "Oz", "Pia", "Quin",
         "Rae", "Sol", "Tia", "Uma", "Vic", "Wes", "Xan", "Yul", "Zoe"]
METRIC_SQL = {
    "revenue": "SUM(oi.quantity * oi.unit_price)",
    "order_count": "COUNT(DISTINCT o.id)",
    "items_bought": "SUM(oi.quantity)",
}


def params_for(seed: int) -> dict:
    rng = random.Random(f"sql-family/params/{seed}")
    return {name: rng.choice(values) for name, values in PARAMS.items()}


def reference_sql(k: int, metric: str, year: int) -> str:
    return (f"SELECT o.customer_id AS customer_id, {METRIC_SQL[metric]} AS {metric} "
            f"FROM orders o JOIN order_items oi ON oi.order_id = o.id "
            f"WHERE o.status = 'completed' AND substr(o.placed_at, 1, 4) = '{year}' "
            f"GROUP BY o.customer_id ORDER BY {metric} DESC, customer_id ASC LIMIT {k}")


def build(path: Path, rng: random.Random, n_customers: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                             placed_at TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), product TEXT NOT NULL,
                                  quantity INTEGER NOT NULL, unit_price REAL NOT NULL);
    """)
    order_id = 0
    for cid in range(1, n_customers + 1):
        db.execute("INSERT INTO customers VALUES (?, ?, ?)", (cid, f"{rng.choice(NAMES)} {cid}", rng.choice(COUNTRIES)))
        for _ in range(rng.randint(0, 9)):
            order_id += 1
            year = rng.choice([2022, 2023, 2024, 2025, 2026])
            placed = f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            status = rng.choices(["completed", "cancelled", "refunded"], weights=[8, 1, 1])[0]
            db.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", (order_id, cid, placed, status))
            for _ in range(rng.randint(1, 4)):
                product, price = rng.choice(PRODUCTS)
                db.execute("INSERT INTO order_items VALUES (?, ?, ?, ?)", (order_id, product, rng.randint(1, 5), price))
    db.commit()
    db.execute("VACUUM")
    db.close()


def rows(path: Path, sql: str) -> list:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [list(r) for r in db.execute(sql).fetchall()]
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build one instance of the top-k customers family.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    p = params_for(a.seed)
    build(out / "agent" / "data" / "shop.db", random.Random(f"sql-family/visible/{a.seed}"), 40)
    build(out / "verifier" / "hidden.db", random.Random(f"sql-family/hidden/{a.seed}"), 60)
    expected = rows(out / "verifier" / "hidden.db", reference_sql(p["k"], p["metric"], p["year"]))
    (out / "verifier" / "expected.json").write_text(json.dumps({"params": p, "rows": expected}, sort_keys=True) + "\n")
    instance = {"seed": a.seed, "params": p, "placeholders": {name: str(value) for name, value in p.items()}}
    (out / "instance.json").write_text(json.dumps(instance, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
