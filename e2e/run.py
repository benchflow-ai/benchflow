#!/usr/bin/env python3
"""Black-box end-to-end tests of the `bench` binary.

Drives a built `bench` from outside, one scenario file per feature
(e2e/scenarios/*.toml), and checks what it does: exit codes, lines of
output, files written, and the status, reward, and problem of each trial.
It never looks at how the binary is written. Standard library only
(Python 3.11 or newer).

    e2e/run.py --bench target/release/bench --backend daytona --tier smoke
    e2e/run.py --bench ./bench --backend none                # no sandbox at all
    e2e/run.py --bench ./bench --backend docker --tier full -k regrade

See docs/rust/testing-e2e.md for the scenario format.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import shutil
import socket
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trial_audit  # noqa: E402  (beside this file)
REPO = HERE.parent
FIXTURES = REPO / "crates/benchflow-run/tests/fixtures/tasks"
E2E_TASKS = HERE / "tasks"
# Tasks named `examples/<name>` come from the repository's examples folder.
EXAMPLES = REPO / "examples"
TIERS = {"smoke": 0, "full": 1}
# Where bench's `vercel` provider reads its key, besides its key file.
GATEWAY_KEYS = ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN")


class Failed(Exception):
    pass


# Every key a scenario, step, and trial check may use: a typo such as
# `stdout_have` would otherwise turn a check off without a word.
SCENARIO_KEYS = {"tier", "needs", "backends", "tasks", "limit", "limit_matches", "sandboxes", "host_cache", "env", "step",
                 "audit"}
STEP_KEYS = {"name", "run", "sh", "exit", "stdout_has", "stderr_has", "output_has", "output_lacks", "stdout_json",
             "files", "absent", "json", "file_has", "trials", "env", "cwd", "stdin", "timeout", "only_backends"}
# The shape of each step key that is not a string; `file_has` is a table of
# file to patterns, `json` and `trials` are lists of tables.
STEP_SHAPES = {"run": list, "exit": (int, list), "stdout_has": list, "stderr_has": list, "output_has": list,
               "output_lacks": list, "stdout_json": dict, "files": list, "absent": list, "json": list,
               "file_has": dict, "trials": list, "env": dict, "timeout": (int, float), "only_backends": list}
STEP_SHAPE_NAMES = {list: "list", dict: "table", (int, list): "number or a list of numbers",
                    (int, float): "number"}
TRIAL_KEYS = {"job", "task", "agent", "count", "status", "reward", "passed", "code", "owner", "result", "files",
              "absent", "file_has"}
# Inherited variables that would change what bench does under test: the
# runner sets its own owner, cache, and colors, and nothing else of the
# caller's (a Claude or Codex home, a redirected model endpoint, an image
# cache budget) leaks into a step.
SCRUBBED_ENV = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "CLICOLOR_FORCE")
SCRUBBED_ENV_PREFIX = "BENCHFLOW_"
# How long a step's process group gets after SIGTERM before SIGKILL: bench
# waits up to 120 s for sandboxes still being created before it exits.
TERM_GRACE = 150
# How long a sandbox may stay listed after its scenario ends before it
# counts as left over (Daytona deletes asynchronously).
LEFTOVER_GRACE = 60


def subst(value, ctx):
    """Fill {name} placeholders in strings, lists, and tables."""
    if isinstance(value, str):
        return re.sub(r"\{(\w+)\}", lambda m: str(ctx.get(m.group(1), m.group(0))), value)
    if isinstance(value, list):
        return [subst(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: subst(v, ctx) for k, v in value.items()}
    return value


def dig(doc, path):
    """Follow a dotted path (`a.b.0.c`) into parsed JSON; None when absent."""
    cur = doc
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def load_json_lines(text):
    """Parse a JSON document, or JSON lines (one object per line)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out


ATIF_SCHEMA = HERE / "atif-v1.8.schema.json"
ATIF_VERSION = "ATIF-v1.8"
_JSON_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def schema_errors(doc, schema, root, path="$"):
    """The ways `doc` breaks `schema`: the JSON Schema keywords the ATIF schema
    uses (type, enum, const, pattern, properties, required,
    additionalProperties, items, minItems, minimum, anyOf, $ref)."""
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        return schema_errors(doc, node, root, path)
    if "anyOf" in schema:
        options = [schema_errors(doc, s, root, path) for s in schema["anyOf"]]
        if all(options):
            return min(options, key=len)
        return []
    errors = []
    t = schema.get("type")
    if t is not None:
        if t == "integer":
            ok = isinstance(doc, int) and not isinstance(doc, bool)
        elif t == "number":
            ok = isinstance(doc, (int, float)) and not isinstance(doc, bool)
        else:
            ok = isinstance(doc, _JSON_TYPES[t])
        if not ok:
            return [f"{path}: expected {t}, got {type(doc).__name__}"]
    if "const" in schema and doc != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {doc!r}")
    if "enum" in schema and doc not in schema["enum"]:
        errors.append(f"{path}: {doc!r} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(doc, str) and not re.search(schema["pattern"], doc):
        errors.append(f"{path}: {doc!r} does not match {schema['pattern']}")
    if "minimum" in schema and isinstance(doc, (int, float)) and doc < schema["minimum"]:
        errors.append(f"{path}: {doc} is below {schema['minimum']}")
    if isinstance(doc, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in doc:
                errors.append(f"{path}: missing {key}")
        for key, value in doc.items():
            if key in props:
                errors += schema_errors(value, props[key], root, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field {key}")
    if isinstance(doc, list):
        if "minItems" in schema and len(doc) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(doc):
                errors += schema_errors(item, schema["items"], root, f"{path}[{i}]")
    return errors


def atif_rule_errors(doc, path="$"):
    """The ATIF rules a JSON Schema cannot state."""
    errors = []
    steps = doc.get("steps") or []
    for i, step in enumerate(steps):
        where = f"{path}.steps[{i}]"
        if step.get("step_id") != i + 1:
            errors.append(f"{where}: step_id {step.get('step_id')}, expected {i + 1}")
        if step.get("source") != "agent":
            for field in ("model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"):
                if step.get(field) is not None:
                    errors.append(f"{where}: {field} on a {step.get('source')} step")
        elif step.get("llm_call_count") == 0:
            for field in ("metrics", "reasoning_content"):
                if step.get(field) is not None:
                    errors.append(f"{where}: {field} with llm_call_count 0")
        ids = {c.get("tool_call_id") for c in step.get("tool_calls") or []}
        for r in (step.get("observation") or {}).get("results") or []:
            if r.get("source_call_id") is not None and r["source_call_id"] not in ids:
                errors.append(f"{where}: observation refers to {r['source_call_id']}, not a tool call of this step")
        m = step.get("metrics") or {}
        if m.get("cached_tokens") is not None and m.get("prompt_tokens") is not None and m["cached_tokens"] > m["prompt_tokens"]:
            errors.append(f"{where}: cached_tokens {m['cached_tokens']} > prompt_tokens {m['prompt_tokens']}")
        if step.get("timestamp") is not None:
            try:
                datetime.fromisoformat(step["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{where}: timestamp {step['timestamp']!r} is not ISO 8601")
    fm = doc.get("final_metrics") or {}
    if fm.get("total_steps") is not None and fm["total_steps"] != len(steps) and not doc.get("notes"):
        errors.append(f"{path}.final_metrics.total_steps {fm['total_steps']}, but {len(steps)} steps")
    cached, prompt = fm.get("total_cached_tokens"), fm.get("total_prompt_tokens")
    if cached is not None and prompt is not None and cached > prompt:
        errors.append(f"{path}.final_metrics: total_cached_tokens {cached} > total_prompt_tokens {prompt}"
                      " (cached tokens are part of the prompt tokens)")
    seen = set()
    for i, sub in enumerate(doc.get("subagent_trajectories") or []):
        tid = sub.get("trajectory_id")
        if tid is None or tid in seen:
            errors.append(f"{path}.subagent_trajectories[{i}]: trajectory_id missing or repeated")
        seen.add(tid)
        errors += atif_rule_errors(sub, f"{path}.subagent_trajectories[{i}]")
    return errors


def check_atif_documents(work):
    """Every trainer/atif.json this engine wrote under `work` must be an ATIF
    v1.8 document: valid against e2e/atif-v1.8.schema.json and its rules.
    Documents another engine wrote (copied job folders) are left alone."""
    schema = json.loads(ATIF_SCHEMA.read_text())
    checked = 0
    for f in sorted(work.rglob("trainer/atif.json")):
        trial = f.parent.parent
        try:
            config = json.loads((trial / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            config = {}
        engine = config.get("engine") if isinstance(config, dict) else None
        if not (isinstance(engine, dict) and engine.get("name") == "benchflow-rs"):
            continue
        rel = f.relative_to(work)
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise Failed(f"{rel}: not JSON: {e}")
        errors = schema_errors(doc, schema, schema) + atif_rule_errors(doc)
        if errors:
            raise Failed(f"{rel}: not a valid {ATIF_VERSION} document:\n  " + "\n  ".join(errors[:12]))
        checked += 1
    return checked


def check_value(label, got, want):
    """`want` is a value, or a table: {min, max, in, not, has, matches, present}."""
    if isinstance(want, dict) and set(want) <= {"min", "max", "in", "not", "has", "matches", "present", "len"}:
        if "present" in want:
            if (got is not None) != want["present"]:
                raise Failed(f"{label}: expected {'present' if want['present'] else 'absent'}, got {got!r}")
        if got is None and set(want) - {"present"}:
            raise Failed(f"{label}: missing")
        if "min" in want and not got >= want["min"]:
            raise Failed(f"{label}: {got!r} < {want['min']!r}")
        if "max" in want and not got <= want["max"]:
            raise Failed(f"{label}: {got!r} > {want['max']!r}")
        if "in" in want and got not in want["in"]:
            raise Failed(f"{label}: {got!r} not one of {want['in']!r}")
        if "not" in want and got == want["not"]:
            raise Failed(f"{label}: {got!r} should differ")
        if "has" in want:
            for item in want["has"] if isinstance(want["has"], list) else [want["has"]]:
                if item not in got:
                    raise Failed(f"{label}: {item!r} not in {str(got)[:300]}")
        if "matches" in want and not re.search(want["matches"], str(got)):
            raise Failed(f"{label}: {str(got)[:300]!r} does not match /{want['matches']}/")
        if "len" in want and len(got) != want["len"]:
            raise Failed(f"{label}: length {len(got)} != {want['len']}")
        return
    if isinstance(want, float) or isinstance(got, float):
        if got is None or abs(float(got) - float(want)) > 1e-9:
            raise Failed(f"{label}: {got!r} != {want!r}")
        return
    if got != want:
        raise Failed(f"{label}: {got!r} != {want!r}")


def find_trials(job_dir: Path, task: str):
    out = []
    for p in sorted(job_dir.glob("*/result.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise Failed(f"{p}: unreadable result.json ({e})")
        name = r.get("task_name") or p.parent.name.split("__")[0]
        if fnmatch.fnmatch(name, task) or fnmatch.fnmatch(p.parent.name, task):
            out.append((p.parent, r))
    return out


def reward_of(r):
    rw = r.get("rewards")
    if isinstance(rw, dict):
        return rw.get("reward", next(iter(rw.values()), None))
    return rw


def run_group(argv, cwd, env, stdin, timeout):
    """Run a step in its own process group; None when it times out.

    On a timeout the whole group gets SIGTERM first, so a `bench` started
    under `sh` stops its trials and removes its sandboxes as on Ctrl-C, and
    SIGKILL only after a grace period.
    """
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    with LIVE_LOCK:
        LIVE.add(proc)
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_group(proc)
        return None
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM on the suite: the step's group never got it.
        # Stopped here, and by stop_all() for steps in other threads.
        stop_group(proc)
        raise
    finally:
        with LIVE_LOCK:
            LIVE.discard(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


# The steps running now, so Ctrl-C or SIGTERM on the suite stops them too:
# each runs in a session of its own, which the terminal's signal misses.
LIVE: set = set()
LIVE_LOCK = threading.Lock()


def group_gone(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def stop_group(proc):
    """SIGTERM a step's process group, then SIGKILL after TERM_GRACE.

    Waits for the whole group, not just the step's own process: under `sh`
    the shell dies at once while `bench` is still removing its sandboxes.
    Keeps draining the step's pipes meanwhile, so a `bench` that writes
    while it stops never blocks on a full pipe.
    """
    pgid = proc.pid
    for sig, grace in ((signal.SIGTERM, TERM_GRACE), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            except (ValueError, OSError):
                time.sleep(0.5)
            if proc.poll() is not None and group_gone(pgid):
                return


def stop_all():
    """Stop every live step (on Ctrl-C or SIGTERM), each with its grace."""
    with LIVE_LOCK:
        procs = list(LIVE)
    threads = [threading.Thread(target=stop_group, args=(p,)) for p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def this_host():
    """This machine's name as bench writes it into its host label."""
    try:
        name = Path("/proc/sys/kernel/hostname").read_text().strip()
        if name:
            return name
    except OSError:
        pass
    out = subprocess.run(["hostname"], capture_output=True, text=True)
    return out.stdout.strip() or socket.gethostname()


HOST = this_host()


def bench_scope(raw):
    """The owner label bench derives from BENCHFLOW_OWNER (owner.rs): other
    characters than letters, digits, `.`, `_` and `-` become `-`, `-` and
    `.` are trimmed from both ends, and the result is cut to 48."""
    mapped = re.sub(r"[^A-Za-z0-9._-]", "-", raw.strip())
    return mapped.strip("-.")[:48]


def docker_labeled(kind, filters):
    """Ids and labels of Docker containers or networks matching label filters."""
    fmt = '{{.ID}}\t{{.Label "ai.benchflow.owner"}}\t{{.Label "ai.benchflow.host"}}\t{{.Label "ai.benchflow.pid"}}'
    cmd = ["docker", "ps", "-a"] if kind == "container" else ["docker", "network", "ls"]
    for f in filters:
        cmd += ["--filter", f"label={f}"]
    out = subprocess.run(cmd + ["--format", fmt], capture_output=True, text=True)
    if out.returncode != 0:
        raise Failed(f"listing Docker {kind}s failed: {out.stderr.strip()[:300]}")
    rows = []
    for line in out.stdout.splitlines():
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        if parts[0]:
            rows.append(parts)
    return rows


def daytona_api():
    api = os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api").rstrip("/")
    return api, {"Authorization": f"Bearer {os.environ.get('DAYTONA_API_KEY', '')}"}


def daytona_labeled(labels):
    api, headers = daytona_api()
    query = urllib.parse.urlencode({"labels": json.dumps(labels)})
    with urllib.request.urlopen(urllib.request.Request(f"{api}/sandbox?{query}", headers=headers), timeout=60) as r:
        doc = json.load(r)
    items = doc if isinstance(doc, list) else doc.get("items", [])
    # The API filters by label; check again so nothing else is ever removed.
    return [i for i in items if all((i.get("labels") or {}).get(k) == v for k, v in labels.items())
            and str(i.get("state", "")).lower() not in ("destroyed", "destroying")]


def remove_one(backend, what):
    kind, ident = what
    if backend == "docker":
        cmd = ["docker", "rm", "-f", ident] if kind == "container" else ["docker", "network", "rm", ident]
        subprocess.run(cmd, capture_output=True)
    elif backend == "daytona":
        api, headers = daytona_api()
        req = urllib.request.Request(f"{api}/sandbox/{ident}?force=true", headers=headers, method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=60).close()
        except OSError:
            pass


def leftovers(backend, owner):
    """What still carries `owner` and this machine's host label: a list of
    (kind, id). Only this machine's: an owner another machine also used is
    never listed, so never removed."""
    labels = {"ai.benchflow.owner": owner, "ai.benchflow.host": HOST}
    if backend == "docker":
        filters = [f"{k}={v}" for k, v in labels.items()]
        return ([("container", r[0]) for r in docker_labeled("container", filters)]
                + [("network", r[0]) for r in docker_labeled("network", filters)])
    if backend == "daytona":
        return [("sandbox", i["id"]) for i in daytona_labeled(labels)]
    return []


def check_leftovers(backend, owner, grace):
    """Fail-message for what `owner` left on this machine (then removed), or "".

    Daytona applies a delete a few seconds after bench sent it (seen: still
    "started" when bench exited, gone 5 s later), so what is listed is
    re-read for up to `grace` seconds first."""
    deadline = time.monotonic() + grace
    while True:
        found = leftovers(backend, owner)
        if not found or time.monotonic() > deadline:
            break
        time.sleep(3)
    for what in found:
        remove_one(backend, what)
    if not found:
        return ""
    return f"{len(found)} sandbox(es) or network(s) left: " + ", ".join(f"{k} {i[:12]}" for k, i in found[:5])


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def ledger_dir():
    """Where this machine's suite runs note their owners: the runs whose
    leftovers a later run may sweep. Per user (under the caller's cache
    folder), so one user's run never sweeps another's."""
    base = os.environ.get("BENCHFLOW_CACHE_DIR") or str(Path.home() / ".cache" / "benchflow")
    return Path(base) / "e2e-runs"


def ledger_add(folder, owner, backend):
    """Note this run's base owner, backend, host and pid before it starts a
    sandbox."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{owner}.json").write_text(json.dumps({"owner": owner, "backend": backend, "host": HOST,
                                                       "pid": os.getpid(), "started": time.time()}))


def ledger_drop(folder, owner):
    try:
        (folder / f"{owner}.json").unlink()
    except OSError:
        pass


def ledger_read(folder):
    entries = []
    for f in sorted(folder.glob("e2e-*.json")) if folder.is_dir() else []:
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(e, dict) and isinstance(e.get("owner"), str) and f.name == f"{e['owner']}.json":
            entries.append(e)
    return entries


def sweep_targets(rows, entries, host, alive, own):
    """What a run's startup sweep removes, from `rows` (kind, id, owner label,
    host label, bench pid): only what carries a step owner (`<base>-<n>`, in
    full) of a run in this machine's ledger (`entries`) whose runner is gone
    here, with this machine's host label and a bench process that is gone
    too. Never this run's own (`own`). Returns (targets, bases swept)."""
    dead = {e["owner"] for e in entries
            if e.get("host") == host and e["owner"] != own and not alive(e.get("pid", ""))}
    targets = []
    for kind, ident, owner, row_host, pid in rows:
        base, _, n = owner.rpartition("-")
        if base in dead and n.isdigit() and row_host == host and not alive(pid):
            targets.append((kind, ident))
    return targets, dead


def sweep_dead(backend, folder, own):
    """Remove the sandboxes and networks that earlier runs of this suite on
    this machine left when stopped by force (their runner is gone), and
    forget each such run once nothing of it is left. A live run's, another
    user's, and another machine's are never touched; nor is anything whose
    owner merely starts like one of ours."""
    # Only this backend's runs: another backend's leftovers are not listed
    # here, so its entries must be neither swept nor forgotten.
    entries = [e for e in ledger_read(folder) if e.get("backend", backend) == backend]
    if not entries:
        return 0

    def rows():
        out = []
        if backend == "docker":
            for kind in ("container", "network"):
                for ident, owner, host, pid in docker_labeled(kind, [f"ai.benchflow.host={HOST}"]):
                    out.append((kind, ident, owner, host, pid))
        elif backend == "daytona":
            for i in daytona_labeled({"ai.benchflow.host": HOST}):
                labels = i.get("labels") or {}
                out.append(("sandbox", i["id"], labels.get("ai.benchflow.owner", ""),
                            labels.get("ai.benchflow.host", ""), labels.get("ai.benchflow.pid", "")))
        return out

    found, dead = sweep_targets(rows(), entries, HOST, pid_alive, own)
    for what in found:
        remove_one(backend, what)
    # Forget a dead run once nothing of it is listed (Daytona deletes a few
    # seconds later; what is still listed is retried by the next run).
    # Listed again whether or not anything was removed: a dead run whose
    # bench still runs (its rows were skipped) is kept for a later sweep.
    remaining = {owner.rpartition("-")[0] for _, _, owner, _, _ in rows()}
    for base in dead - remaining:
        ledger_drop(folder, base)
    return len(found)


# Step owners whose sandboxes may still exist, for an interrupt to clean.
ACTIVE_OWNERS: set = set()
STEP_COUNTER = [0]
STEP_LOCK = threading.Lock()


def next_owner(base):
    with STEP_LOCK:
        STEP_COUNTER[0] += 1
        return f"{base}-{STEP_COUNTER[0]}"


class Scenario:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.doc = tomllib.loads(path.read_text())
        self.tier = self.doc.get("tier", "smoke")
        self.backends = self.doc.get("backends", ["none", "docker", "daytona"])
        self.needs = self.doc.get("needs", [])
        self.limit = self.doc.get("limit")
        self.sandboxes = self.doc.get("sandboxes", "sandbox" in self.needs)
        self.limit_matches = self.doc.get("limit_matches")
        bad = set(self.doc) - SCENARIO_KEYS
        for i, step in enumerate(self.doc.get("step", []), 1):
            bad |= {f"step {i}: {k}" for k in set(step) - STEP_KEYS}
            for t in step.get("trials", []):
                bad |= {f"step {i} trials: {k}" for k in set(t) - TRIAL_KEYS}
            # A key of the wrong shape would crash the runner mid-scenario.
            for k, shape in STEP_SHAPES.items():
                if k in step and not isinstance(step[k], shape):
                    raise SystemExit(f"{path.name}: step {i}: {k} must be a {STEP_SHAPE_NAMES[shape]}")
        if bad:
            raise SystemExit(f"{path.name}: unknown keys {sorted(bad)}")


class Runner:
    def __init__(self, args):
        self.args = args
        self.bench = str(Path(args.bench).resolve())
        self.backend = args.backend
        self.log = []

    def run_scenario(self, sc: Scenario):
        work = Path(tempfile.mkdtemp(prefix=f"e2e-{sc.name}-", dir=self.args.work))
        tasks = work / "tasks"
        tasks.mkdir()
        for src in sc.doc.get("tasks", []):
            if src.startswith("examples/"):
                base = EXAMPLES / "tasks" / src.removeprefix("examples/")
            else:
                base = E2E_TASKS / src if (E2E_TASKS / src).exists() else FIXTURES / src
            if not base.exists():
                raise Failed(f"no task `{src}` in e2e/tasks, examples/tasks, or the fixtures")
            shutil.copytree(base, tasks / Path(src).name, symlinks=True)
        ctx = {
            "bench": self.bench,
            "backend": self.backend,
            "work": str(work),
            "tasks": str(tasks),
            "jobs": str(work / "jobs"),
            "j": str(self.args.max_sandboxes),
            "e2e": str(HERE),
            "repo": str(REPO),
            "fixtures": str(FIXTURES),
            "examples": str(EXAMPLES),
            "judge": self.args.judge_model,
        }
        env = {k: v for k, v in os.environ.items()
               if k not in SCRUBBED_ENV and not k.startswith(SCRUBBED_ENV_PREFIX)}
        env["NO_COLOR"] = "1"
        # Every scenario gets its own cache folder, unless it shares the
        # machine's (to reuse a fetched dataset, say).
        if not sc.doc.get("host_cache"):
            env["BENCHFLOW_CACHE_DIR"] = str(work / "cache")
        for k, v in subst(sc.doc.get("env", {}), ctx).items():
            env[k] = v
        transcript = []
        try:
            for i, step in enumerate(sc.doc.get("step", []), 1):
                self.run_step(sc, i, subst(step, ctx), work, env, transcript)
            # Every trial folder the scenario left must be sound
            # (e2e/trial_audit.py), and every ATIF document this engine wrote
            # valid ATIF v1.8, unless the scenario corrupts its jobs on purpose.
            if sc.doc.get("audit", True):
                problems = trial_audit.audit_job_tree(work)
                if problems:
                    raise Failed("trial audit:\n  " + "\n  ".join(problems[:20]))
                check_atif_documents(work)
        except Failed as e:
            return False, str(e), work, transcript
        return True, "", work, transcript

    def run_step(self, sc, i, step, work, env, transcript):
        label = f"step {i}" + (f" ({step['name']})" if "name" in step else "")
        if "only_backends" in step and self.backend not in step["only_backends"]:
            return
        if "sh" in step:
            argv = ["/bin/sh", "-c", step["sh"]]
        elif "run" in step:
            argv = [self.bench] + list(step["run"])
        else:
            argv = None
        if argv is not None:
            if step.get("env"):
                env = {**env, **step["env"]}
            # Each step gets an owner of its own (the run's, plus a counter):
            # bench sweeps its owner's leftovers before it starts, which
            # would hide what an earlier step left; this way the check after
            # the step sees exactly what that step left.
            owner = next_owner(self.args.owner)
            env = {**env, "BENCHFLOW_OWNER": owner}
            sandboxed = sc.sandboxes and self.backend != "none"
            if sandboxed:
                with STEP_LOCK:
                    ACTIVE_OWNERS.add(owner)
            timeout = step.get("timeout", 1800)
            started = time.monotonic()
            proc = run_group(argv, work / step.get("cwd", "."), env, step.get("stdin", ""), timeout)
            if proc is None:
                msg = f"{label}: timed out after {timeout}s: {' '.join(argv[1:])}"
                if sandboxed:
                    left = check_leftovers(self.backend, owner, LEFTOVER_GRACE if self.backend == "daytona" else 0)
                    with STEP_LOCK:
                        ACTIVE_OWNERS.discard(owner)
                    msg += f"; {left}" if left else ""
                raise Failed(msg)
            secs = time.monotonic() - started
            (work / f"step{i}.stdout").write_text(proc.stdout)
            (work / f"step{i}.stderr").write_text(proc.stderr)
            transcript.append(f"$ {' '.join(argv if 'sh' in step else ['bench'] + argv[1:])}  -> {proc.returncode} ({secs:.0f}s)")
            out, err = proc.stdout, proc.stderr
            if sandboxed:
                left = check_leftovers(self.backend, owner, LEFTOVER_GRACE if self.backend == "daytona" else 0)
                with STEP_LOCK:
                    ACTIVE_OWNERS.discard(owner)
                if left:
                    raise Failed(f"{label}: {left}")
            # A step with no `exit` must succeed: a failed setup step would
            # otherwise leave the checks after it testing the wrong thing.
            want = step.get("exit", 0)
            ok = proc.returncode in want if isinstance(want, list) else proc.returncode == want
            if not ok:
                raise Failed(f"{label}: exit {proc.returncode}, expected {want}\n  stdout: {out[-1500:]}\n  stderr: {err[-1500:]}")
            for pat in step.get("stdout_has", []):
                if not re.search(pat, out, re.M):
                    raise Failed(f"{label}: stdout lacks /{pat}/\n  stdout: {out[-1500:]}")
            for pat in step.get("stderr_has", []):
                if not re.search(pat, err, re.M):
                    raise Failed(f"{label}: stderr lacks /{pat}/\n  stderr: {err[-1500:]}")
            for pat in step.get("output_has", []):
                if not re.search(pat, out + err, re.M):
                    raise Failed(f"{label}: output lacks /{pat}/\n  stdout: {out[-1000:]}\n  stderr: {err[-1000:]}")
            for pat in step.get("output_lacks", []):
                m = re.search(pat, out + err, re.M)
                if m:
                    raise Failed(f"{label}: output has /{pat}/: {m.group(0)!r}")
            if "stdout_json" in step:
                doc = load_json_lines(out)
                for path, want in step["stdout_json"].items():
                    check_value(f"{label}: stdout json {path}", dig(doc, path), want)
        for f in step.get("files", []):
            if not list(work.glob(f)):
                raise Failed(f"{label}: no file matches {f}")
        for f in step.get("absent", []):
            hits = list(work.glob(f))
            if hits:
                raise Failed(f"{label}: should not exist: {hits[0].relative_to(work)}")
        for chk in step.get("json", []):
            matches = sorted(work.glob(chk["file"]))
            if not matches:
                raise Failed(f"{label}: no file matches {chk['file']}")
            doc = load_json_lines(matches[0].read_text())
            for path, want in chk.items():
                if path != "file":
                    check_value(f"{label}: {matches[0].relative_to(work)} {path}", dig(doc, path), want)
        for f, pats in step.get("file_has", {}).items():
            matches = sorted(work.glob(f))
            if not matches:
                raise Failed(f"{label}: no file matches {f}")
            text = matches[0].read_text(errors="replace")
            for pat in pats if isinstance(pats, list) else [pats]:
                if not re.search(pat, text, re.M):
                    raise Failed(f"{label}: {f} lacks /{pat}/\n  {text[-800:]}")
        for t in step.get("trials", []):
            self.check_trials(label, work, t)

    def check_trials(self, label, work, t):
        job = work / "jobs" / t["job"] if not t["job"].startswith("/") else Path(t["job"])
        if "*" in t["job"]:
            found = sorted((work / "jobs").glob(t["job"]))
            if not found:
                raise Failed(f"{label}: no job matches {t['job']}")
            job = found[-1]
        trials = find_trials(job, t.get("task", "*"))
        if "agent" in t:
            trials = [(d, r) for d, r in trials if fnmatch.fnmatch(str(r.get("agent")), t["agent"])]
        want_n = t.get("count")
        if want_n is not None and len(trials) != want_n:
            raise Failed(f"{label}: {len(trials)} trials of {t.get('task', '*')} in {job.name}, expected {want_n}")
        if not trials:
            raise Failed(f"{label}: no trial of {t.get('task', '*')} in {job}")
        for tdir, r in trials:
            who = f"{label}: {job.name}/{tdir.name}"
            if "status" in t:
                check_value(f"{who} status", r.get("status"), t["status"])
            if "reward" in t:
                check_value(f"{who} reward", reward_of(r), t["reward"])
            if "passed" in t:
                check_value(f"{who} passed", r.get("passed"), t["passed"])
            prob = r.get("problem") or {}
            if "code" in t:
                check_value(f"{who} problem code ({prob.get('message', '')[:200]})", prob.get("code"), t["code"])
            if "owner" in t:
                check_value(f"{who} problem owner", prob.get("owner"), t["owner"])
            for path, want in t.get("result", {}).items():
                check_value(f"{who} result.json {path}", dig(r, path), want)
            for f in t.get("files", []):
                if not list(tdir.glob(f)):
                    raise Failed(f"{who}: no file {f}")
            for f in t.get("absent", []):
                if list(tdir.glob(f)):
                    raise Failed(f"{who}: should not have {f}")
            for f, pats in t.get("file_has", {}).items():
                matches = sorted(tdir.glob(f))
                if not matches:
                    raise Failed(f"{who}: no file {f}")
                text = matches[0].read_text(errors="replace")
                for pat in pats if isinstance(pats, list) else [pats]:
                    if not re.search(pat, text, re.M):
                        raise Failed(f"{who}: {f} lacks /{pat}/")


def select(args, all_scenarios):
    chosen, skipped = [], []
    for sc in all_scenarios:
        why = None
        if TIERS[sc.tier] > TIERS[args.tier]:
            why = f"tier {sc.tier}"
        elif args.backend not in sc.backends:
            why = f"runs on {', '.join(sc.backends)}"
        elif args.k and not any(fnmatch.fnmatch(sc.name, f"*{k}*") for k in args.k):
            why = "not selected"
        else:
            for need in sc.needs:
                if need == "sandbox" and args.backend == "none":
                    why = "needs a sandbox"
                elif need == "gateway" and not (any(os.environ.get(v) for v in GATEWAY_KEYS) or args.gateway
                                                 or Path.home().joinpath(".config/benchflow/vercel-ai-gateway.env").exists()):
                    why = "needs the Vercel AI Gateway key"
                elif need == "mimo" and not os.environ.get("MIMO_API_KEY"):
                    why = "needs MIMO_API_KEY (MiMo's token plan)"
                elif need not in ("sandbox", "gateway", "mimo") and need not in args.allow:
                    why = f"needs {need} (pass --allow {need})"
        (skipped if why else chosen).append((sc, why))
    return chosen, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", required=True, help="the bench binary to test")
    ap.add_argument("--backend", default="none", choices=["none", "docker", "daytona"],
                    help="sandbox backend for scenarios that need one; `none` runs only the rest")
    ap.add_argument("--tier", default="smoke", choices=list(TIERS))
    ap.add_argument("-k", action="append", default=[], help="only scenarios whose name contains this")
    ap.add_argument("--allow", action="append", default=[], help="extra needs to allow, e.g. gateway-spend")
    ap.add_argument("--gateway", action="store_true", help="the gateway key is available some other way")
    ap.add_argument("--owner", default=None,
                    help="owner label for the suite's sandboxes (default: e2e-<user>-<random>, new each run; "
                         "BENCHFLOW_OWNER is not inherited)")
    ap.add_argument("--require-sandbox", action="store_true",
                    help="fail when no scenario that starts a sandbox ran (a gate that skipped them all)")
    ap.add_argument("--judge-model", default="vercel/deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--max-sandboxes", type=int, default=3,
                    help="sandbox scenarios run one at a time, each with at most this many trials at once")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="scenarios without sandboxes run this many at once")
    ap.add_argument("--work", default=None, help="where scenario folders go (kept on failure)")
    ap.add_argument("--keep", action="store_true", help="keep every scenario's folder")
    ap.add_argument("--json", metavar="FILE", help="also write the results here as JSON")
    args = ap.parse_args()
    if not Path(args.bench).is_file():
        ap.error(f"no binary at {args.bench}")
    args.work = args.work or tempfile.mkdtemp(prefix="bench-e2e-")
    user = re.sub(r"[^a-z0-9]+", "-", (os.environ.get("USER") or "user").lower()).strip("-")[:20].strip("-") or "user"
    if not args.owner:
        args.owner = f"e2e-{user}-{uuid.uuid4().hex[:6]}"
    # The runner removes what carries its owners, so an owner must be the
    # suite's alone: never `bench` (every ordinary run's) or a team's.
    # Step owners add `-<n>`, and bench cuts labels at 48 characters.
    if not args.owner.startswith("e2e-") or bench_scope(args.owner) != args.owner or len(args.owner) > 40:
        ap.error(f"--owner must start with `e2e-`, use only letters, digits, `.`, `_` and `-`, and be at most "
                 f"40 characters (the runner removes what carries it), not `{args.owner}`")
    Path(args.work).mkdir(parents=True, exist_ok=True)

    # Ctrl-C and SIGTERM stop the running step's process group (run_group),
    # even when the suite inherited SIGINT ignored (a background job of a
    # non-interactive shell does).
    def on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    scenarios = [Scenario(p) for p in sorted((HERE / "scenarios").glob("*.toml"))]
    chosen, skipped = select(args, scenarios)
    runner = Runner(args)
    version = subprocess.run([runner.bench, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"{version} · backend {args.backend} · tier {args.tier} · {len(chosen)} scenarios ({len(skipped)} skipped)"
          f" · owner {args.owner} · {args.work}")
    if not chosen:
        print("  no scenario selected")
        sys.exit(1)
    ledger = ledger_dir()
    uses_sandboxes = args.backend != "none" and any(sc.sandboxes for sc, _ in chosen)
    if uses_sandboxes:
        try:
            swept = sweep_dead(args.backend, ledger, args.owner)
        except (Failed, OSError, ValueError) as e:
            swept = 0
            print(f"  could not sweep earlier runs' leftovers: {e}")
        if swept:
            print(f"  removed {swept} sandbox(es) or network(s) earlier runs of this suite left on this machine")
        # Noted before any sandbox starts, so a run stopped by force is swept
        # by the next one; dropped when this run ends by itself.
        try:
            ledger_add(ledger, args.owner, args.backend)
        except OSError as e:
            print(f"  could not note this run in {ledger}: {e} (a run stopped by force will not be swept)")

    results = []

    def one(sc):
        started = time.monotonic()
        try:
            ok, why, work, transcript = runner.run_scenario(sc)
        except Failed as e:
            ok, why, work, transcript = False, str(e), None, []
        # A crash log means bench hit an internal error on some step, even
        # one whose exit code a check accepted.
        crashes = sorted((work / "cache" / "logs").glob("crash-*.log")) if work else []
        if crashes:
            ok = False
            why = (why + "\n" if why else "") + f"bench crashed: {crashes[0].name}"
        secs = time.monotonic() - started
        if sc.limit and not ok and sc.limit_matches and not re.search(sc.limit_matches, why or ""):
            status = "FAIL"
        elif sc.limit and not ok:
            status = "limit"
        elif sc.limit and ok:
            status = "unexpected-pass"
        else:
            status = "pass" if ok else "FAIL"
        mark = {"pass": "✓", "FAIL": "✗", "limit": "~", "unexpected-pass": "!"}[status]
        print(f"  {mark} {sc.name:<34} {secs:6.0f}s" + (f"  {why.splitlines()[0][:150]}" if why and status == "FAIL" else "")
              + (f"  known limit: {sc.limit[:100]}" if status == "limit" else "")
              + ("  passes, but is marked as a known limit: remove `limit`" if status == "unexpected-pass" else ""),
              flush=True)
        if status == "FAIL":
            for line in transcript:
                print(f"      {line}")
            for line in why.splitlines()[1:12]:
                print(f"      {line}")
        if work and status in ("pass", "limit") and not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        results.append({"scenario": sc.name, "status": status, "seconds": round(secs), "why": why,
                        "limit": sc.limit, "work": str(work) if work else None})

    ran_sandbox = [False]
    local = [sc for sc, _ in chosen if not sc.sandboxes]
    remote = [sc for sc, _ in chosen if sc.sandboxes]
    def interrupted():
        # Stop every running step, then remove what their owners left.
        stop_all()
        with STEP_LOCK:
            owners = list(ACTIVE_OWNERS)
        for owner in owners:
            try:
                left = check_leftovers(args.backend, owner, 30 if args.backend == "daytona" else 0)
            except (Failed, OSError, ValueError) as e:
                left = f"could not list: {e}"
            if left:
                print(f"  after the interrupt, {owner}: {left} (removed)", flush=True)
        if uses_sandboxes:
            ledger_drop(ledger, args.owner)
        sys.exit(130)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
    try:
        list(pool.map(one, local))
        pool.shutdown()
        for sc in remote:
            one(sc)
            if args.require_sandbox and results and results[-1]["status"] in ("pass", "FAIL", "limit", "unexpected-pass"):
                ran_sandbox[0] = True
    except KeyboardInterrupt:
        print("  interrupted: stopping the running steps", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        interrupted()
    for sc, why in skipped:
        if why != "not selected":
            results.append({"scenario": sc.name, "status": "skipped", "why": why})

    if uses_sandboxes:
        ledger_drop(ledger, args.owner)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("  " + " · ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for r in results:
        if r["status"] == "skipped" and args.k == []:
            print(f"  - {r['scenario']:<34} skipped: {r['why']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"version": version, "backend": args.backend, "tier": args.tier,
                                                "results": results}, indent=2))
    if args.require_sandbox and not ran_sandbox[0]:
        print("  --require-sandbox: no scenario that starts a sandbox ran")
        sys.exit(1)
    sys.exit(1 if counts.get("FAIL") or counts.get("unexpected-pass") else 0)


if __name__ == "__main__":
    main()
