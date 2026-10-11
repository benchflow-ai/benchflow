#!/bin/sh
# Black-box end-to-end tests of the bench binary; see docs/rust/testing-e2e.md.
#   e2e/run.sh --bench target/release/bench --backend daytona --tier smoke
exec python3 "$(dirname "$0")/run.py" "$@"
