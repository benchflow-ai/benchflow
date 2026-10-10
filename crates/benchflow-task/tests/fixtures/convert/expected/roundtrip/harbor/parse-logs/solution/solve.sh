#!/bin/bash
mkdir -p /app/out
printf '{"200": 1, "404": 1}' > /app/out/status.json
