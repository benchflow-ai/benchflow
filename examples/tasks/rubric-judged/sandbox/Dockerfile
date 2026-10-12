FROM ubuntu:24.04
WORKDIR /app
RUN printf '09:13:58 nginx: upstream TLS handshake failed: certificate has expired (notAfter=2026-10-11T09:14:00Z)\n09:14:02 healthcheck: 503 from api.internal\n09:31:40 ops: renewed certificate for api.internal; reloaded nginx\n09:31:44 healthcheck: 200 from api.internal\n' > /app/incident.log
