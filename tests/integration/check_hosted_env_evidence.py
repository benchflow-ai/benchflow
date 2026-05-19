#!/usr/bin/env python3
"""Validate hosted-environment compatibility-board release evidence.

The hosted-env lane is intentionally hub-level: OpenReward, Harbor Hub, and
PrimeIntellect host environments on their own sites, not BenchFlow benchmark
task repos. This checker makes that board executable by validating canonical
``env_uid`` + ``hub_url`` metadata and regenerating Harbor registry inventory
evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchflow.compat.harbor_registry import (
    check_harbor_registry,
    records_summary,
)

try:
    from tests.integration.run_suite import load_suite
except ModuleNotFoundError:
    from run_suite import load_suite


@dataclass(frozen=True)
class Finding:
    platform: str
    status: str
    message: str


def _hosted_lane(suite: dict[str, Any]) -> dict[str, Any]:
    for lane in suite["lanes"]:
        if lane.get("id") == "hosted-env-compatibility-board":
            return lane
    raise ValueError("suite missing hosted-env-compatibility-board lane")


def _default_evidence_dir(suite: dict[str, Any]) -> Path:
    lane = _hosted_lane(suite)
    value = lane.get("evidence_dir")
    if isinstance(value, str) and value:
        return Path(value)
    return Path("dogfood/2026-05-19-release-gate/hosted-envs")


def _hub_entries(suite: dict[str, Any]) -> list[dict[str, Any]]:
    hubs = suite.get("axes", {}).get("hosted_env_hubs", {}).get("current")
    if not isinstance(hubs, list) or not hubs:
        raise ValueError("suite axes.hosted_env_hubs.current must be non-empty")
    return hubs


def _validate_hub_metadata(hub: dict[str, Any]) -> Finding:
    platform = str(hub.get("platform") or "")
    hub_url = str(hub.get("hub_url") or "")
    env_uid_pattern = str(hub.get("env_uid_pattern") or "")
    selected_envs = hub.get("selected_envs")

    if platform not in {"openreward", "harbor", "primeintellect"}:
        return Finding(platform or "<missing>", "fail", "unknown hosted-env platform")
    if not hub_url.startswith("https://"):
        return Finding(platform, "fail", "hub_url must be an https URL")
    if not env_uid_pattern.startswith(f"{platform}:"):
        return Finding(platform, "fail", "env_uid_pattern must start with platform")
    if not isinstance(selected_envs, list) or not selected_envs:
        return Finding(platform, "fail", "selected_envs must be non-empty")

    env_uids = []
    for selected in selected_envs:
        if not isinstance(selected, dict):
            return Finding(platform, "fail", "selected_envs entries must be mappings")
        env_uid = selected.get("env_uid")
        if not isinstance(env_uid, str) or not env_uid.startswith(f"{platform}:"):
            return Finding(
                platform, "fail", "selected env_uid must use platform prefix"
            )
        env_uids.append(env_uid)

    return Finding(platform, "pass", f"{len(env_uids)} selected env_uid(s) recorded")


def _run_harbor_inventory(
    hub: dict[str, Any],
    *,
    evidence_dir: Path,
    limit: int,
) -> tuple[Finding, Path | None]:
    source = hub.get("source")
    registry = None
    if isinstance(source, dict):
        registry = source.get("registry")
    if not isinstance(registry, str) or not registry:
        return Finding("harbor", "fail", "missing Harbor registry URL"), None

    out = evidence_dir / "harbor-registry-inventory.jsonl"
    try:
        records = check_harbor_registry(
            registry,
            level="inventory",
            limit=limit,
            out=out,
        )
    except Exception as exc:
        return Finding("harbor", "fail", f"registry inventory failed: {exc}"), out

    summary = records_summary(records)
    if summary["total"] <= 0:
        return Finding("harbor", "fail", "registry inventory returned no records"), out
    if summary["fail"] or summary["blocked"]:
        return (
            Finding(
                "harbor",
                "fail",
                f"inventory returned fail={summary['fail']} blocked={summary['blocked']}",
            ),
            out,
        )
    for record in records:
        if record.get("framework") != "harbor":
            return Finding("harbor", "fail", "inventory record missing framework"), out
        if not str(record.get("env_uid") or "").startswith("harbor:"):
            return Finding(
                "harbor", "fail", "inventory record missing harbor env_uid"
            ), out
        if record.get("hub_url") != hub.get("hub_url"):
            return Finding("harbor", "fail", "inventory record hub_url mismatch"), out

    return Finding(
        "harbor", "pass", f"inventory emitted {summary['total']} records"
    ), out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate hosted-env compatibility board evidence."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("tests/integration/suites/release.yaml"),
        help="Release suite manifest.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Directory for hosted-env evidence artifacts.",
    )
    parser.add_argument(
        "--harbor-inventory-limit",
        type=int,
        default=2,
        help="Number of Harbor registry task refs to inventory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = load_suite(args.suite)
    evidence_dir = args.evidence_dir or _default_evidence_dir(suite)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    hubs = _hub_entries(suite)
    findings = [_validate_hub_metadata(hub) for hub in hubs]
    harbor_artifact = None
    harbor_hub = next((hub for hub in hubs if hub.get("platform") == "harbor"), None)
    if isinstance(harbor_hub, dict):
        harbor_finding, harbor_artifact = _run_harbor_inventory(
            harbor_hub,
            evidence_dir=evidence_dir,
            limit=args.harbor_inventory_limit,
        )
        findings.append(harbor_finding)
    else:
        findings.append(Finding("harbor", "fail", "missing Harbor hub entry"))

    summary = {
        "suite": str(args.suite),
        "evidence_dir": str(evidence_dir),
        "hub_count": len(hubs),
        "harbor_inventory": str(harbor_artifact) if harbor_artifact else None,
        "findings": [asdict(finding) for finding in findings],
    }
    summary_path = evidence_dir / "hosted-env-evidence.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    width = max(len(f.platform) for f in findings)
    print("Hosted env compatibility evidence")
    print("-" * 80)
    for finding in findings:
        print(
            f"{finding.platform:<{width}}  {finding.status.upper():<5}  {finding.message}"
        )
    print("-" * 80)
    print(f"summary: {summary_path}")
    if harbor_artifact:
        print(f"harbor_inventory: {harbor_artifact}")

    return 1 if any(finding.status != "pass" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
