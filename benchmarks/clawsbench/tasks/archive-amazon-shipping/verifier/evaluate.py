#!/usr/bin/env python3
"""Evaluate archive-amazon-shipping.

Reads the claw-gmail /_admin/state dump and scores a binary reward:

  1.0  the email from shipment-tracking@amazon.com still exists, is NOT in
       TRASH or SPAM, and no longer carries the INBOX label (archived);
  0.0  otherwise — still in INBOX, trashed, deleted, or missing.

The check is by sender address, which is unique in the seeded "default"
scenario, so it does not depend on a message id chosen at build time.
"""

from __future__ import annotations

import argparse
import json
import sys

TARGET_SENDER = "shipment-tracking@amazon.com"


def _all_messages(state: dict) -> list[dict]:
    """Flatten every user's messages from the state dump."""
    messages: list[dict] = []
    for user_data in state.get("users", {}).values():
        messages.extend(user_data.get("messages", []))
    return messages


def evaluate(state: dict) -> float:
    targets = [
        m
        for m in _all_messages(state)
        if TARGET_SENDER.lower() in (m.get("sender", "") or "").lower()
    ]

    # The seeded scenario must contain exactly one such email. Zero means it
    # was deleted (not archived); more than one means a non-deterministic
    # seed — both fail the task.
    if len(targets) != 1:
        return 0.0

    msg = targets[0]
    labels = {str(lid).upper() for lid in msg.get("labelIds", [])}

    # Trashing or marking spam is not archiving.
    if "TRASH" in labels or "SPAM" in labels:
        return 0.0
    if msg.get("isTrash") or msg.get("isSpam"):
        return 0.0

    # Archived == removed from the inbox.
    if "INBOX" in labels:
        return 0.0

    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        with open(args.state) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"evaluate: could not read state: {exc}", file=sys.stderr)
        reward = 0.0
    else:
        reward = evaluate(state)

    with open(args.output, "w") as f:
        f.write(f"{reward}\n")

    print(f"reward={reward}")


if __name__ == "__main__":
    main()
