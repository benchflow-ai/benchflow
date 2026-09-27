#!/bin/bash
# Runs the hidden checks and writes /logs/verifier/reward.json with one result per rubric criterion.
set -u
pytest -q --json-report --json-report-file=/logs/verifier/pytest.json /verifier/tests
python3 /verifier/score.py /logs/verifier/pytest.json > /logs/verifier/reward.json
