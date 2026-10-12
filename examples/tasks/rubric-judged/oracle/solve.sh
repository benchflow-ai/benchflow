#!/bin/bash
cat > /app/summary.md <<'MD'
# Incident 2026-10-11

- What failed: api.internal returned 503 from 09:14 to 09:31 UTC.
- Root cause: the TLS certificate of api.internal expired at 09:14:00 UTC, so nginx's upstream TLS handshake failed.
- Fix: the certificate was renewed and nginx reloaded at 09:31 UTC; health checks recovered.
MD
