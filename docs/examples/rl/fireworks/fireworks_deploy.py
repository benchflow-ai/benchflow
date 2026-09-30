"""Serve a base model or a trained LoRA on a Fireworks dedicated deployment, for evaluate.py.

    python fireworks_deploy.py shapes MODEL                      # validated shapes this account can use
    python fireworks_deploy.py create DEPLOYMENT_ID MODEL --shape SHAPE
    python fireworks_deploy.py wait DEPLOYMENT_ID                # until READY with a replica
    python fireworks_deploy.py smoke 'MODEL#accounts/ACCOUNT/deployments/DEPLOYMENT_ID'
    python fireworks_deploy.py delete DEPLOYMENT_ID              # stop billing
    python fireworks_deploy.py usage DEPLOYMENT_ID --since 2026-09-30 --usd-per-hour 8

A trained LoRA can only be served on a dedicated deployment (Fireworks has no
serverless LoRA). ``create`` with the promoted LoRA model as MODEL gives a
live-merge deployment: the adapter is merged into the base weights, so it
serves at the base model's speed. Call it through Fireworks' OpenAI-compatible
endpoint with ``--model 'MODEL#accounts/ACCOUNT/deployments/DEPLOYMENT_ID'``.

Deployments bill per GPU-second while they have a replica. ``create`` pins one
replica, so the evaluation never waits on a cold start, and an expiry time
(``--expire-hours``, default 3) after which Fireworks deletes the deployment
even if you forget; ``delete`` it when done. The key comes
from ``FIREWORKS_API_KEY``; models and deployments are private to the account.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Any

import httpx

API = "https://api.fireworks.ai"


def _key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not key:
        sys.exit("error: set FIREWORKS_API_KEY")
    return key


def _account(args: argparse.Namespace) -> str:
    account = args.account or os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    if not account:
        sys.exit("error: pass --account or set FIREWORKS_ACCOUNT_ID")
    return account


def _sdk(args: argparse.Namespace) -> Any:
    from fireworks import Fireworks

    return Fireworks(api_key=_key(), account_id=_account(args))


def _dump(value: Any) -> dict[str, Any]:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def _get(path: str, **params: Any) -> dict[str, Any]:
    response = httpx.get(
        f"{API}{path}",
        params=params,
        headers={"Authorization": f"Bearer {_key()}"},
        timeout=60,
    )
    if response.status_code != 200:
        sys.exit(f"error: GET {path}: HTTP {response.status_code} {response.text[:300]}")
    return response.json()


def cmd_shapes(args: argparse.Namespace) -> None:
    match = _dump(_sdk(args).deployment_shape_versions.match_for_model(args.model))
    for version in match.get("deployment_shape_versions") or []:
        snap = version.get("snapshot") or {}
        print(
            snap.get("name"),
            snap.get("accelerator_type"),
            snap.get("accelerator_count"),
            snap.get("precision"),
        )


def cmd_create(args: argparse.Namespace) -> None:
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=args.expire_hours)
    deployment = _sdk(args).deployments.create(
        base_model=args.model,
        deployment_id=args.deployment_id,
        deployment_shape=args.shape,
        min_replica_count=1,
        max_replica_count=1,
        expire_time=expires,
        display_name=args.display_name or args.deployment_id,
    )
    print(json.dumps(_brief(_dump(deployment)), indent=1))


def _brief(d: dict[str, Any]) -> dict[str, Any]:
    status = d.get("status") or {}
    return {
        "name": d.get("name"),
        "state": d.get("state"),
        "status": status.get("message") if isinstance(status, dict) else status,
        "base_model": d.get("base_model") or d.get("baseModel"),
        "shape": d.get("deployment_shape") or d.get("deploymentShape"),
        "accelerator": d.get("accelerator_type") or d.get("acceleratorType"),
        "count": d.get("accelerator_count") or d.get("acceleratorCount"),
        "replicas": d.get("replica_count", d.get("replicaCount")),
        "min": d.get("min_replica_count", d.get("minReplicaCount")),
        "max": d.get("max_replica_count", d.get("maxReplicaCount")),
    }


def _deployment(args: argparse.Namespace) -> dict[str, Any]:
    return _get(f"/v1/accounts/{_account(args)}/deployments/{args.deployment_id}")


def cmd_status(args: argparse.Namespace) -> None:
    print(json.dumps(_brief(_deployment(args)), indent=1))


def cmd_wait(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.timeout
    last = None
    while True:
        brief = _brief(_deployment(args))
        now = (brief["state"], brief["replicas"])
        if now != last:
            print(f"{time.strftime('%H:%M:%S')} {brief['state']} replicas={brief['replicas']} {brief['status'] or ''}", flush=True)
            last = now
        if brief["state"] == "READY" and (brief["replicas"] or 0) >= 1:
            return
        if brief["state"] in {"FAILED", "DELETED", "DELETING"}:
            sys.exit(f"error: deployment is {brief['state']}: {brief['status']}")
        if time.monotonic() > deadline:
            sys.exit(f"error: not ready after {args.timeout}s")
        time.sleep(20)


def cmd_smoke(args: argparse.Namespace) -> None:
    """A plain call and a tool call through the OpenAI-compatible endpoint."""

    tool = {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
    ask = "How many lines are in /etc/hosts? Use the run_bash tool."
    for name, extra in (
        ("plain", {"messages": [{"role": "user", "content": "Say ok."}]}),
        ("tools", {"messages": [{"role": "user", "content": ask}], "tools": [tool], "tool_choice": "auto"}),
    ):
        body = {"model": args.model, "max_tokens": 512, **extra}
        started = time.monotonic()
        response = httpx.post(
            f"{API}/inference/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {_key()}"},
            timeout=900,
        )
        took = time.monotonic() - started
        if response.status_code != 200:
            print(f"{name}: HTTP {response.status_code} {response.text[:300]}")
            continue
        message = response.json()["choices"][0]["message"]
        calls = [c["function"] for c in message.get("tool_calls") or []]
        print(f"{name}: {took:.1f}s usage={response.json().get('usage')} tool_calls={calls}")


def cmd_delete(args: argparse.Namespace) -> None:
    _sdk(args).deployments.delete(args.deployment_id, ignore_checks=True)
    print(f"deleted {args.deployment_id}")


def cmd_usage(args: argparse.Namespace) -> None:
    """Accelerator-seconds Fireworks billed this deployment, and their cost."""

    end = (dt.datetime.now(dt.UTC) + dt.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    usage = _get(
        f"/v1/accounts/{_account(args)}/billingUsage",
        startTime=f"{args.since}T00:00:00Z",
        endTime=end,
    )
    wanted = f"/deployments/{args.deployment_id}"
    seconds = sum(
        int(row.get("acceleratorSeconds") or 0)
        for row in usage.get("dedicatedCosts") or []
        if str(row.get("deploymentId", "")).endswith(wanted)
    )
    print(
        json.dumps(
            {
                "deployment": args.deployment_id,
                "accelerator_seconds": seconds,
                "usd": round(seconds / 3600 * args.usd_per_hour, 2),
                "usd_per_hour": args.usd_per_hour,
            }
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--account", help="Fireworks account id (or FIREWORKS_ACCOUNT_ID)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("shapes")
    p.add_argument("model")
    p.set_defaults(func=cmd_shapes)
    p = sub.add_parser("create")
    p.add_argument("deployment_id")
    p.add_argument("model", help="a base model, or a promoted LoRA model (live merge)")
    p.add_argument("--shape", default="default", help="a validated shape, or 'default'")
    p.add_argument(
        "--expire-hours",
        type=float,
        default=3.0,
        help="Fireworks deletes the deployment after this long, in case you forget",
    )
    p.add_argument("--display-name")
    p.set_defaults(func=cmd_create)
    for name, func in (("status", cmd_status), ("delete", cmd_delete)):
        p = sub.add_parser(name)
        p.add_argument("deployment_id")
        p.set_defaults(func=func)
    p = sub.add_parser("wait")
    p.add_argument("deployment_id")
    p.add_argument("--timeout", type=int, default=3600)
    p.set_defaults(func=cmd_wait)
    p = sub.add_parser("smoke")
    p.add_argument("model", help="MODEL#accounts/ACCOUNT/deployments/DEPLOYMENT_ID")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("usage")
    p.add_argument("deployment_id")
    p.add_argument("--since", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--usd-per-hour", type=float, required=True, help="the shape's GPU price")
    p.set_defaults(func=cmd_usage)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
