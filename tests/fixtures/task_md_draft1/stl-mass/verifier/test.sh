#!/bin/bash
# Runs the hidden tests and writes a CTRF test report. The runtime decides each test-judged criterion in rubric.json from it.
set -u
pytest -q --ctrf /logs/verifier/ctrf.json /verifier/tests
