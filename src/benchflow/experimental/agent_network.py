"""Opt-in agent-network experiment; does not change ``bench run`` defaults.

Run ``python -m benchflow.experimental.agent_network --no-network-by-default
-- TASK [bench run options]``. The existing task overlay records and enforces
this choice through the normal rollout policy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run_arguments(arguments: list[str], *, no_network_by_default: bool) -> list[str]:
    """Construct the ordinary run command with an explicit agent-only overlay."""
    if not no_network_by_default:
        return list(arguments)
    if any(
        arg.split("=", 1)[0] in {"--config", "--config-override"} for arg in arguments
    ):
        raise ValueError(
            "The network spike cannot combine config overlays; use bench run with an explicit agent.network_mode overlay instead."
        )
    return [
        *arguments,
        "--config-override",
        json.dumps({"agent": {"network_mode": "no-network", "allowed_hosts": []}}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-network-by-default", action="store_true")
    parser.add_argument("run_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run_args = args.run_args
    if run_args[:1] == ["--"]:
        run_args = run_args[1:]
    try:
        run_args = run_arguments(
            run_args, no_network_by_default=args.no_network_by_default
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "benchflow.cli.main", "run", *run_args])
    )


if __name__ == "__main__":
    main()
