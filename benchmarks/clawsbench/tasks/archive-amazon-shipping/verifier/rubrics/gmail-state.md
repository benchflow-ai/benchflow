# Gmail State Rubric

The verifier awards `1.0` when exactly one message from
`shipment-tracking@amazon.com` exists, that message is neither trash nor spam,
and the message no longer has the `INBOX` label. It awards `0.0` if the target
message is missing, duplicated, still in the inbox, trashed, spammed, or if the
Gmail service state cannot be read.
