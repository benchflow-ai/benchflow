#!/bin/bash
# Installs the test runner, runs the hidden tests, and writes a CTRF report. The runtime decides each test-judged criterion in rubric.json from it.
pip3 install --break-system-packages pytest==8.4.1 pytest-json-ctrf==0.3.5
mkdir -p /logs/verifier
pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v
exit 0
