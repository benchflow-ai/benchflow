# Reference summary (the task author's, not the agent's)

The API returned 503 from 09:14 to 09:31 UTC because the TLS certificate of api.internal expired at 09:14:00; nginx could not complete its upstream handshake. Operations renewed the certificate and reloaded nginx at 09:31, and health checks recovered.
