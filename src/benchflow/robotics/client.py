"""Standalone container client; copied into task images without host code."""

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Control the commissioned arm through its trial bridge"
    )
    parser.add_argument(
        "command",
        choices=[
            "observe",
            "status",
            "tip",
            "roll",
            "gripper",
            "monitor",
            "rest",
            "finish",
        ],
    )
    parser.add_argument("args", nargs="*")
    parser.add_argument("--request-id", default=None)
    parser.add_argument(
        "--arm",
        default=None,
        help="Arm to address on a two-arm setup (see setup.json); omit on one-arm setups",
    )
    args = parser.parse_args()
    config = json.loads(
        Path(
            os.environ.get("ROBOT_CONNECTION", "/app/robot-connection.json")
        ).read_text()
    )
    request_id = args.request_id or str(uuid.uuid4())
    print(json.dumps({"request_id": request_id}), flush=True)
    body = {"request_id": request_id, "command": args.command, "args": args.args}
    if args.arm:
        body["arm"] = args.arm
    request = urllib.request.Request(
        config["url"] + "/command",
        json.dumps(body).encode(),
        {
            "Authorization": "Bearer " + config["token"],
            "Content-Type": "application/json",
        },
    )
    try:
        # NO automatic retries. An explicit same-ID retry only retrieves a cached receipt.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=120
        ) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError):
        raise SystemExit(
            "Bridge response unavailable. Do not issue another motion; notify the operator."
        ) from None
    directory = Path(os.environ.get("ROBOT_OBSERVATIONS", "/app/observations"))
    directory.mkdir(parents=True, exist_ok=True)
    for image in result.get("images", []):
        name = Path(image["name"]).name
        path = directory / name
        path.write_bytes(base64.b64decode(image.pop("base64"), validate=True))
        image["path"] = str(path)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
