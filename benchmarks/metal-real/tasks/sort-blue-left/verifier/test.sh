#!/bin/sh
set -eu
echo "Physical score requires host adjudication: use python -m benchflow.robotics score." >&2
echo "No physical reward can be inferred from agent-written files." >&2
exit 2
