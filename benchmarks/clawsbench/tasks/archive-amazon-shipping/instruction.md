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
