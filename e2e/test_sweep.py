"""Unit tests of the e2e runner's startup sweep (run.py), standard library
only: `python3 -I e2e/test_sweep.py`. The Rust test suite runs it.

Guards the fix of 2026-10-11 (break campaign E9): the sweep matched any
owner starting with `e2e-<user>-`, so it removed other users' runs whose
names extend the user's (`john` and `john-doe`), and trusted the host label,
so a live run on another machine with the same host name was removed.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e2e_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

HOST = "builder-1"
ALIVE = {"111", "222"}


def alive(pid):
    return str(pid) in ALIVE


def entry(owner, pid, host=HOST):
    return {"owner": owner, "host": host, "pid": pid, "started": 0}


class SweepTargets(unittest.TestCase):
    def test_only_step_owners_of_dead_runs_in_the_ledger(self):
        entries = [entry("e2e-john-abc123", 999), entry("e2e-john-live01", 111)]
        rows = [
            ("container", "c1", "e2e-john-abc123-1", HOST, "998"),     # ours, dead: swept
            ("network", "n1", "e2e-john-abc123-12", HOST, ""),         # ours, no pid: swept
            ("container", "c2", "e2e-john-doe-abc123-1", HOST, "998"),  # another user extending the name
            ("container", "c3", "e2e-john-abc124-1", HOST, "998"),     # an owner not in the ledger
            ("container", "c4", "e2e-john-abc123-x", HOST, "998"),     # not a step owner
            ("container", "c5", "e2e-john-abc123", HOST, "998"),       # the base itself
            ("container", "c6", "e2e-john-live01-1", HOST, "998"),     # a live run's
            ("container", "c7", "e2e-john-abc123-2", "builder-2", "998"),  # another host's label
            ("container", "c8", "e2e-john-abc123-3", HOST, "222"),     # its bench is still running
        ]
        targets, dead = run.sweep_targets(rows, entries, HOST, alive, own="e2e-john-new000")
        self.assertEqual(targets, [("container", "c1"), ("network", "n1")])
        self.assertEqual(dead, {"e2e-john-abc123"})

    def test_a_run_never_sweeps_itself_or_another_machines_ledger_entry(self):
        entries = [entry("e2e-john-abc123", 999), entry("e2e-john-other1", 999, host="elsewhere")]
        rows = [("sandbox", "s1", "e2e-john-abc123-1", HOST, "998"),
                ("sandbox", "s2", "e2e-john-other1-1", HOST, "998")]
        targets, _ = run.sweep_targets(rows, entries, HOST, alive, own="e2e-john-abc123")
        self.assertEqual(targets, [])

    def test_an_entry_is_kept_while_its_sandboxes_are_listed_or_another_backend_owns_it(self):
        # Diff review L3: an entry was dropped when nothing could be removed
        # (its bench still ran), and a Docker run's entry by a Daytona sweep.
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "e2e-runs"
            folder.mkdir()
            for owner, backend in [("e2e-john-live01", "docker"), ("e2e-john-dock01", "docker"),
                                   ("e2e-john-gone01", "docker"), ("e2e-john-dayt01", "daytona")]:
                (folder / f"{owner}.json").write_text(json.dumps(
                    {"owner": owner, "backend": backend, "host": run.HOST, "pid": 999}))
            listed = [("c1", "e2e-john-live01-1", run.HOST, "111")]   # its bench still runs
            removed = []
            saved = (run.docker_labeled, run.remove_one, run.pid_alive)
            try:
                run.docker_labeled = lambda kind, filters: [list(r) for r in listed] if kind == "container" else []
                run.remove_one = lambda backend, what: removed.append(what)
                run.pid_alive = alive
                self.assertEqual(run.sweep_dead("docker", folder, "e2e-john-new000"), 0)
            finally:
                run.docker_labeled, run.remove_one, run.pid_alive = saved
            self.assertEqual(removed, [])
            kept = sorted(e["owner"] for e in run.ledger_read(folder))
            # live01 still has a container; dayt01 is Daytona's; the other
            # two Docker runs left nothing and are forgotten.
            self.assertEqual(kept, ["e2e-john-dayt01", "e2e-john-live01"])

    def test_the_ledger_is_read_and_written_per_owner(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "e2e-runs"
            run.ledger_add(folder, "e2e-john-abc123", "docker")
            (folder / "e2e-bad.json").write_text("{not json")
            (folder / "e2e-renamed.json").write_text(json.dumps(entry("e2e-john-abc999", 1)))
            got = run.ledger_read(folder)
            self.assertEqual([e["owner"] for e in got], ["e2e-john-abc123"])
            self.assertEqual(got[0]["host"], run.HOST)
            self.assertEqual(got[0]["backend"], "docker")
            run.ledger_drop(folder, "e2e-john-abc123")
            self.assertEqual(run.ledger_read(folder), [])


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    sys.exit(0 if result.wasSuccessful() else 1)
