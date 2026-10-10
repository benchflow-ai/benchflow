#!/bin/bash
# Runs the tests and writes a test report in the Common Test Report Format
# (CTRF) to /logs/verifier/ctrf.json, from which the runtime decides each
# test-judged criterion of verifier/rubric.json.
pytest -q -p no:cacheprovider --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py
