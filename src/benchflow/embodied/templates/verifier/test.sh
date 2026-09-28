#!/bin/sh
# Physical verifier entry point. BenchFlow runs it in the trusted `simulator` service (verifier.service), where
# the episode record lives; the agent container cannot read or write it.
set -eu
mkdir -p /logs/verifier
# the simulator image carries benchflow.embodied at /opt/benchflow_embodied (see sidecar.vendor_embodied)
export PYTHONPATH="/opt/benchflow_embodied${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m benchflow.embodied.verifier
