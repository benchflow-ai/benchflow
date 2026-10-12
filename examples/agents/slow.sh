#!/bin/bash
# Sleeps for a minute (the task allows two), so a run can be interrupted mid-trial and resumed.
sleep 60
printf 'Hello, world!' > /app/hello.txt
