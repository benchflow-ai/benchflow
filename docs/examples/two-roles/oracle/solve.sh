#!/bin/bash
printf 'import csv\nprint(sum(1 for _ in csv.reader(open("/app/data/counts.csv"))) - 1)\n' > /app/count.py
python3 /app/count.py > /app/rows.txt
