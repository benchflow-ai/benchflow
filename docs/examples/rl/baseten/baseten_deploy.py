"""Manage the Baseten deployment an evaluation runs against.

    python baseten_deploy.py list
    python baseten_deploy.py push truss/qwen35-9b-lora          # deploy a checkpoint
    python baseten_deploy.py wait MODEL_ID DEPLOYMENT_ID        # until ACTIVE
    python baseten_deploy.py smoke MODEL_ID DEPLOYMENT_ID --model benchflow-sft
    python baseten_deploy.py url MODEL_ID DEPLOYMENT_ID         # for evaluate.py --base-url
    python baseten_deploy.py deactivate MODEL_ID DEPLOYMENT_ID  # stop billing

Also ``status``, ``config`` (the Truss config, with the served model names),
``activate`` (an INACTIVE deployment), ``wake`` (a scaled-to-zero one), and
``autoscale``. The key comes from ``BASETEN_API_KEY`` in the environment; it is
sent only to Baseten and never printed. ``push`` hands it to the ``truss`` CLI
through environment variables, not on its command line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import httpx

API = "https://api.baseten.co/v1"
READY = "ACTIVE"
FAILED = {"BUILD_FAILED", "DEPLOY_FAILED", "FAILED", "BUILD_STOPPED", "UNHEALTHY"}


def _key() -> str:
    key = os.environ.get("BASETEN_API_KEY", "").strip()
    if not key:
        sys.exit("error: set BASETEN_API_KEY (load it from a mode-600 file)")
    return key


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Api-Key {_key()}"}, timeout=httpx.Timeout(60.0)
    )


def _call(method: str, url: str, body: dict[str, Any] | None = None) -> Any:
    with _client() as client:
        response = client.request(method, url, json=body)
    if response.status_code >= 400:
        sys.exit(f"error: {method} {url}: HTTP {response.status_code} {response.text[:300]}")
    return response.json() if response.content else {}


def deployment_url(model_id: str, deployment_id: str) -> str:
    """The OpenAI-compatible base URL of one pinned deployment."""

    return f"https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/sync/v1"


def _deployment(model_id: str, deployment_id: str) -> dict[str, Any]:
    return _call("GET", f"{API}/models/{model_id}/deployments/{deployment_id}")


def _brief(d: dict[str, Any]) -> dict[str, Any]:
    scaling = d.get("autoscaling_settings") or {}
    return {
        "id": d.get("id"),
        "name": d.get("name"),
        "status": d.get("status"),
        "environment": d.get("environment"),
        "active_replicas": d.get("active_replica_count"),
        "min_replica": scaling.get("min_replica"),
        "max_replica": scaling.get("max_replica"),
        "scale_down_delay": scaling.get("scale_down_delay"),
        "instance": d.get("instance_type_name"),
        "created_at": d.get("created_at"),
    }


def cmd_list(_: argparse.Namespace) -> None:
    for model in _call("GET", f"{API}/models").get("models", []):
        print(f"{model['id']}  {model['name']}  ({model.get('instance_type_name')})")
        deployments = _call("GET", f"{API}/models/{model['id']}/deployments")
        for d in deployments.get("deployments", []):
            print("   ", json.dumps(_brief(d)))


def cmd_status(args: argparse.Namespace) -> None:
    print(json.dumps(_brief(_deployment(args.model_id, args.deployment_id)), indent=1))


def cmd_config(args: argparse.Namespace) -> None:
    url = f"{API}/models/{args.model_id}/deployments/{args.deployment_id}/config"
    print(_call("GET", f"{url}?output_format=raw").get("raw_config", ""))


def cmd_activate(args: argparse.Namespace) -> None:
    url = f"{API}/models/{args.model_id}/deployments/{args.deployment_id}/activate"
    print(json.dumps(_call("POST", url)))


def cmd_deactivate(args: argparse.Namespace) -> None:
    url = f"{API}/models/{args.model_id}/deployments/{args.deployment_id}/deactivate"
    print(json.dumps(_call("POST", url)))


def cmd_wake(args: argparse.Namespace) -> None:
    host = f"https://model-{args.model_id}.api.baseten.co"
    with _client() as client:
        response = client.post(f"{host}/deployment/{args.deployment_id}/wake")
    print(f"wake: HTTP {response.status_code}")


def cmd_autoscale(args: argparse.Namespace) -> None:
    body = {
        key: value
        for key, value in {
            "min_replica": args.min_replica,
            "max_replica": args.max_replica,
            "scale_down_delay": args.scale_down_delay,
            "concurrency_target": args.concurrency_target,
        }.items()
        if value is not None
    }
    url = (
        f"{API}/models/{args.model_id}/deployments/{args.deployment_id}"
        "/autoscaling_settings"
    )
    print(json.dumps(_call("PATCH", url, body)))


def cmd_wait(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.timeout
    last = None
    while True:
        status = _deployment(args.model_id, args.deployment_id).get("status")
        if status != last:
            print(f"{time.strftime('%H:%M:%S')} {status}", flush=True)
            last = status
        if status == READY:
            return
        if status in FAILED:
            sys.exit(f"error: deployment is {status}")
        if time.monotonic() > deadline:
            sys.exit(f"error: still {status} after {args.timeout}s")
        time.sleep(15)


def cmd_url(args: argparse.Namespace) -> None:
    print(deployment_url(args.model_id, args.deployment_id))


def cmd_smoke(args: argparse.Namespace) -> None:
    """One plain call and one tool call: evaluate.py needs both to work."""

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
    url = deployment_url(args.model_id, args.deployment_id) + "/chat/completions"
    ask = "How many lines are in /etc/hosts? Use the run_bash tool."
    bodies = {
        "plain": {"messages": [{"role": "user", "content": "Say ok."}]},
        "tools": {
            "messages": [{"role": "user", "content": ask}],
            "tools": [tool],
            "tool_choice": "auto",
        },
    }
    with _client() as client:
        for name, body in bodies.items():
            body = {"model": args.model, "max_tokens": 512, **body}
            if args.no_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            started = time.monotonic()
            response = client.post(url, json=body, timeout=1300)
            took = time.monotonic() - started
            if response.status_code != 200:
                print(f"{name}: HTTP {response.status_code} {response.text[:300]}")
                continue
            message = response.json()["choices"][0]["message"]
            calls = [c["function"] for c in message.get("tool_calls") or []]
            print(
                f"{name}: {took:.1f}s usage={response.json().get('usage')} "
                f"tool_calls={calls} content={str(message.get('content'))[:120]!r}"
            )


def cmd_push(args: argparse.Namespace) -> None:
    """``truss push`` with the key in truss's environment, never in argv."""

    env = {
        **os.environ,
        "BASETEN_TRUSS_AUTH_REMOTE_URL": "https://app.baseten.co",
        "BASETEN_TRUSS_AUTH_API_KEY": _key(),
    }
    command = ["truss", "push", args.truss_dir, "--output", "json"]
    if args.deployment_name:
        command += ["--deployment-name", args.deployment_name]
    print("+", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, env=env))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    for name, func in (
        ("status", cmd_status),
        ("config", cmd_config),
        ("activate", cmd_activate),
        ("deactivate", cmd_deactivate),
        ("wake", cmd_wake),
        ("url", cmd_url),
    ):
        p = sub.add_parser(name)
        p.add_argument("model_id")
        p.add_argument("deployment_id")
        p.set_defaults(func=func)
    p = sub.add_parser("wait")
    p.add_argument("model_id")
    p.add_argument("deployment_id")
    p.add_argument("--timeout", type=int, default=2400)
    p.set_defaults(func=cmd_wait)
    p = sub.add_parser("autoscale")
    p.add_argument("model_id")
    p.add_argument("deployment_id")
    p.add_argument("--min-replica", type=int)
    p.add_argument("--max-replica", type=int)
    p.add_argument("--scale-down-delay", type=int, help="seconds")
    p.add_argument("--concurrency-target", type=int)
    p.set_defaults(func=cmd_autoscale)
    p = sub.add_parser("smoke")
    p.add_argument("model_id")
    p.add_argument("deployment_id")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--no-thinking", action="store_true")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("push")
    p.add_argument("truss_dir")
    p.add_argument("--deployment-name")
    p.set_defaults(func=cmd_push)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
