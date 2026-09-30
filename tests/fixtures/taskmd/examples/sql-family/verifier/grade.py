"""Grades /work/answer.sql for one instance of the top-k customers family. Stdlib and sqlite3 only.

Two checks, reported as CTRF tests in /logs/verifier/ctrf.json, with reward.txt beside it:

    test_runs   answer.sql is a UTF-8 file holding one query that runs on the hidden database within 10 s
    test_rows   the query's rows match the reference rows exactly, in order

Everything the answer can cause is a failed check, never an error: a missing, special, oversized, empty, or undecodable
answer.sql, a query that fails or runs too long, and wrong or too many rows. A declared output the solver cannot make
readable is the solver's failure (https://task.md/docs/runtime/episodes/#the-status-of-an-episode). Only a broken
verifier, such as missing instance files, stops the grading with an error, and then no reward is written.
"""

import json
import math
import os
import sqlite3
import stat
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCE = HERE / "instance"  # family@1: the generator's verifier/ files, placed under the verifier's mount
LOGS = Path("/logs/verifier")
ANSWER = Path("/work/answer.sql")
MAX_BYTES = 1 << 20
TIME_LIMIT_S = 10.0


def read_answer() -> tuple:
    """(the query, None), or (None, why the answer cannot be run)."""
    try:
        st = os.stat(ANSWER)
    except FileNotFoundError:
        return None, "/work/answer.sql was not written"
    except OSError as exc:
        return None, f"/work/answer.sql cannot be read: {exc.strerror}"
    if not stat.S_ISREG(st.st_mode):
        return None, "/work/answer.sql is not a regular file"
    with open(ANSWER, "rb") as f:
        data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return None, f"/work/answer.sql is larger than {MAX_BYTES} bytes"
    try:
        sql = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"/work/answer.sql is not UTF-8 text (byte {exc.start} cannot be decoded)"
    if not sql.strip():
        return None, "/work/answer.sql is empty"
    return sql, None


def run(sql: str, limit: int) -> list:
    """At most limit rows of the query on the hidden database, opened read-only, within the time limit."""
    db = sqlite3.connect(f"file:{INSTANCE / 'hidden.db'}?mode=ro", uri=True)
    deadline = time.monotonic() + TIME_LIMIT_S
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return [list(r) for r in db.execute(sql).fetchmany(limit)]
    finally:
        db.close()


def same(got: list, want: list) -> bool:
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if len(g) != 2 or type(g[0]) is not int or g[0] != w[0]:
            return False
        if not (isinstance(g[1], (int, float)) and math.isclose(float(g[1]), float(w[1]), rel_tol=1e-9, abs_tol=1e-6)):
            return False
    return True


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    # The verifier's own inputs: if these are missing, the verifier is broken, and grading stops with an error.
    expected = json.loads((INSTANCE / "expected.json").read_text(encoding="utf-8"))["rows"]
    if not (INSTANCE / "hidden.db").is_file():
        raise SystemExit("the instance's hidden.db is missing: the verifier is broken")
    t0 = time.monotonic()
    rows, why = None, None
    sql, why = read_answer()
    if sql is not None:
        try:
            rows = run(sql, len(expected) + 1)
        except (sqlite3.Error, sqlite3.Warning, ValueError) as exc:
            late = time.monotonic() - t0 >= TIME_LIMIT_S
            why = f"the query ran longer than {TIME_LIMIT_S:g} s" if late else f"the query failed on the hidden database: {exc}"
    t1 = time.monotonic()
    ok = rows is not None and same(rows, expected)
    t2 = time.monotonic()
    tests = [
        {"name": "test_runs", "status": "failed" if why else "passed", "duration": round((t1 - t0) * 1000), "message": why},
        {"name": "test_rows", "status": "passed" if ok else "failed", "duration": round((t2 - t1) * 1000),
         "message": None if ok else f"expected {expected}, got {rows}"},
    ]
    passed = sum(t["status"] == "passed" for t in tests)
    summary = {"tests": len(tests), "passed": passed, "failed": len(tests) - passed, "pending": 0, "skipped": 0,
               "other": 0, "start": int(time.time() * 1000) - tests[0]["duration"] - tests[1]["duration"],
               "stop": int(time.time() * 1000)}
    report = {"results": {"tool": {"name": "sql-family grade.py"}, "summary": summary, "tests": tests}}
    (LOGS / "ctrf.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (LOGS / "reward.txt").write_text("1\n" if passed == len(tests) else "0\n", encoding="utf-8")
    print(json.dumps([{k: t[k] for k in ("name", "status")} for t in tests]))


if __name__ == "__main__":
    main()
