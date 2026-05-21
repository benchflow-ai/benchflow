# archive-amazon-shipping — per-task ClawsBench image.
#
# Builds FROM the smolclaws shared base, which already bundles the claw-*
# service binaries (claw-gmail is on PATH). This task only needs the Gmail
# service, so it seeds only gmail.db.
#
# The image has NO service-starting entrypoint by design: the BenchFlow
# Environment plane (ManifestEnvironment, driven by
# benchmarks/clawsbench/environment.toml) probes `command -v claw-gmail`
# and starts the service itself inside the rollout sandbox.
FROM kywch/smolclaws-base:latest

# Seed deterministic Gmail data. The "default" scenario ships inside the
# claw-gmail package (claw_gmail/seed/content.py) — no smolclaws source tree
# is needed. seed=42 makes the message set reproducible across builds; that
# set always contains exactly one email from shipment-tracking@amazon.com.
RUN claw-gmail --db /data/gmail.db seed --scenario default --seed 42

# Log directories the BenchFlow verifier/agent expect.
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts

# The sandbox runs the agent as the non-root "agent" user; let it write logs.
RUN chown -R agent:agent /logs/agent /logs/artifacts

WORKDIR /app
