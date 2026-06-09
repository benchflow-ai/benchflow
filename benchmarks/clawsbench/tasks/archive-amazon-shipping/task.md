---
schema_version: "1.3"
task:
  name: clawsbench/archive-amazon-shipping
metadata:
  author_name: benchflow
  difficulty: easy
  category: tool-use
  tags:
    - clawsbench
    - gmail
    - stateful
    - service-backed
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
  allow_internet: true
benchflow:
  document_version: "0.3"
  source:
    benchmark: ClawsBench
    task_id: archive-amazon-shipping
  environment:
    manifest: "../../environment.toml"
    services:
      - gmail
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/gmail-state.md
    entrypoint: verifier/test.sh
    implementation:
      type: test-script
      outputs:
        reward_text: /logs/verifier/reward.txt
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
  oracle:
    path: oracle/solve.sh
---

## prompt

Archive the Amazon shipping-confirmation email in my Gmail inbox.

There is exactly one email in the inbox from the sender
`shipment-tracking@amazon.com` (subject begins with "Your Amazon order has
shipped!"). Archive that email — and only that email.

A mock Gmail service compatible with the real Gmail REST API is running at
`http://localhost:9001`. Use the user id `me`.

Archiving an email in Gmail means **removing its `INBOX` label** while keeping
the message itself. Do NOT trash or delete the email, and do NOT touch any
other message.

Useful endpoints:

- Find the message:
  `GET http://localhost:9001/gmail/v1/users/me/messages?q=from:shipment-tracking@amazon.com`
  The response's `messages` array contains objects with an `id` field.
- Archive it:
  `POST http://localhost:9001/gmail/v1/users/me/messages/<id>/modify`
  with JSON body `{"removeLabelIds": ["INBOX"]}`.

You are done when the Amazon shipping email still exists but no longer carries
the `INBOX` label.
