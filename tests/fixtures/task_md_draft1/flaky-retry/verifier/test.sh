#!/bin/bash
# Runs the task's own retry tests and the hidden ones, and writes a CTRF report. The runtime decides each test-judged criterion in rubric.json from it.
cd /workspace
PYTHONPATH=/workspace pytest -q --ctrf /logs/verifier/ctrf.json tests/test_retry.py /verifier/tests
exit 0
