#!/bin/bash
tail -n +2 /opt/data/in.txt | wc -l | tr -d " " > /app/count.txt
