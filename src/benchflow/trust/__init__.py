"""Host-side trust for a sandbox (TRUSTED-RUNTIME v2).

- :mod:`benchflow.trust.checklist`: the I3 system checklist. Pristine facts in,
  "sealed" or "not sealed" with reasons out. The host drives the scan and
  decides; the scanner only reports (``results/TR-PROTOCOL.md``).
- :mod:`benchflow.trust.state`: the I2 host trust state per (sandbox
  generation, service), and :func:`benchflow.trust.state.admit`, the one
  wrapper every ingress goes through.
"""
