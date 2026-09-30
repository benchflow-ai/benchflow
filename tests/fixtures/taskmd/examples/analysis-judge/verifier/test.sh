#!/bin/bash
# The test-judged criteria of analysis-judge, written as a CTRF report. The verifier runs before the judge, in a fresh
# container of the task's image with the saved working folder restored at /work and verifier/ read-only at /verifier.
# The runtime decides the three test criteria from the report, and gives the judge only the tests' ids, outcomes, and
# durations. python3 -I ignores the environment and the current folder, so nothing the solver left in /work can load.
mkdir -p /logs/verifier
cd /
python3 -I /verifier/check_outputs.py /work /verifier/answers.json /logs/verifier/ctrf.json
exit 0
