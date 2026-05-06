#!/bin/bash
# Deterministic stub compile.sh for parity sweeps.
#
# Produces an `executable` that exits with status 1 immediately — guaranteed to
# fail every behavioural test the upstream binary would have passed. Used by
# parity_full.py to put both pipelines (BenchFlow verifier vs. `programbench
# eval`) under the same input. Whatever score they compute *must* be identical
# for the same submission, regardless of whether tests pass or fail.
set -e
cat > executable <<'EXEC'
#!/bin/bash
exit 1
EXEC
chmod +x executable
