"""Check the gate runs an acceptance record names, with GitHub, before a release publishes.

    GATE_RECEIPTS_TOKEN=... python tools/check_gate_receipts.py [--repo .]

The release gate is CI on the exact commit (docs/contributing/release.md). A schema 2 acceptance record
(``.github/release-acceptance/<version>.json``, for the version in ``pyproject.toml``) names its gate runs under
``gates``; ``public-release.yml`` runs this before it publishes. Each run must:

- have passed (``conclusion`` success);
- be the workflow its name stands for (``REQUIRED`` and ``KNOWN`` below);
- have run on the commit the record names for it;
- and that commit must be the accepted source, or differ from it in the version alone
  (``tools/release_promote_check.py``), as an rc promoted to final does.

Every gate in ``REQUIRED`` must be there. A record for a version before ``GATES_REQUIRED_FROM`` may be schema 1
(the 0.9 line), with no gates.

The runs live in the private repository, so reading them takes a token with ``actions:read`` there, from the
``pypi-public`` environment's ``GATE_RECEIPTS_TOKEN`` secret. Without it the check is skipped with a loud warning
(the lead's ruling until an admin adds the token), never silently. Exit 0: checked and passed, or skipped; 1: a
receipt that doesn't hold; 2: a record or API answer that can't be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging.version import Version

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.release_promote_check import version_only_paths

GATES_REQUIRED_FROM = "0.10.0rc6"
REQUIRED: dict[str, str] = {
    "secret-sweep": ".github/workflows/secret-sweep.yml",
    "release-dryrun": ".github/workflows/release-dryrun.yml",
}
KNOWN: dict[str, str] = {
    **REQUIRED,
    "test": ".github/workflows/test.yml",
    "nightly": ".github/workflows/nightly.yml",
    "leak-gate-preflight": ".github/workflows/leak-gate-preflight.yml",
    "docs": ".github/workflows/docs.yml",
    "fresh-resolve": ".github/workflows/fresh-resolve.yml",
}
TOKEN_ENV = "GATE_RECEIPTS_TOKEN"

FetchRun = Callable[[str, int], dict]


class GateError(Exception):
    """A gate receipt that doesn't hold."""


def check_gate_receipts(record: dict, repo: Path, fetch_run: FetchRun) -> list[str]:
    """One line per gate checked; raises :class:`GateError` on the first that doesn't hold."""
    version = str(record.get("version"))
    source = str(record.get("source_commit"))
    gates = record.get("gates")
    if record.get("schema_version") == 1 and Version(version) < Version(
        GATES_REQUIRED_FROM
    ):
        return [
            f"{version}: schema 1 record, released before gate receipts were required"
        ]
    if not isinstance(gates, dict) or not gates:
        raise GateError(
            f"{version}: the acceptance record names no gates; from {GATES_REQUIRED_FROM} on, "
            "a record is schema 2 with its gate runs"
        )
    named: dict[str, dict[str, Any]] = {
        str(k): dict(v) for k, v in gates.items() if isinstance(v, dict)
    }
    if len(named) != len(gates):
        raise GateError(
            f"{version}: each gate is an object with repo, run_id and head_sha"
        )
    missing = sorted(set(REQUIRED) - set(named))
    if missing:
        raise GateError(f"{version}: no receipt for {', '.join(missing)}")
    lines = []
    for name, gate in sorted(named.items()):
        expected = KNOWN.get(name)
        if expected is None:
            raise GateError(
                f"gate {name}: not a known gate ({', '.join(sorted(KNOWN))})"
            )
        run = fetch_run(gate["repo"], gate["run_id"])
        where = f"gate {name} (run {gate['run_id']} in {gate['repo']})"
        if run.get("conclusion") != "success":
            raise GateError(f"{where} did not pass: {run.get('conclusion')}")
        path = str(run.get("path", "")).split("@", 1)[0]
        if path != expected:
            raise GateError(f"{where} is {path}, not the {expected} run")
        if run.get("head_sha") != gate["head_sha"]:
            raise GateError(
                f"{where} ran on {run.get('head_sha')}, not {gate['head_sha']}"
            )
        if gate["head_sha"] != source:
            other = version_only_paths(repo, gate["head_sha"], source)
            if other:
                raise GateError(
                    f"{where} ran on {gate['head_sha'][:12]}, and the accepted source "
                    f"{source[:12]} changes more than the version: {', '.join(other)}"
                )
        lines.append(f"{where}: passed on {gate['head_sha'][:12]}")
    return lines


def _github_fetch(token: str) -> FetchRun:
    def fetch(repo: str, run_id: int) -> dict:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/actions/runs/{int(run_id)}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as answer:
                return json.loads(answer.read())
        except urllib.error.HTTPError as exc:
            raise GateError(
                f"GitHub answered {exc.code} for run {run_id} in {repo}"
            ) from None

    return fetch


def _record(repo: Path) -> dict:
    version = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    return json.loads(
        (repo / ".github" / "release-acceptance" / f"{version}.json").read_text()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    try:
        record = _record(args.repo)
    except (OSError, ValueError, KeyError) as exc:
        print(f"::error::cannot read the acceptance record: {exc}", file=sys.stderr)
        return 2
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        message = (
            f"gate receipts NOT checked: {TOKEN_ENV} is not set in this environment, so the gate runs "
            f"the record names ({', '.join(sorted(record.get('gates') or {})) or 'none'}) were not read from "
            "GitHub. An admin adds the token (actions:read on the private repository); until then the "
            "release owner checks each gate link by hand (docs/contributing/release-checklist.md)."
        )
        print(f"::warning::{message}")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"## Gate receipts NOT checked\n\n{message}\n")
        return 0
    try:
        lines = check_gate_receipts(record, args.repo, _github_fetch(token))
    except GateError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
