#!/bin/bash
set -euo pipefail

REGION="us-east-1"
OUT=/logs/agent/agent-output.txt
mkdir -p "$(dirname "$OUT")"

CUTOFF=$(date -u -d '1 day ago' +%Y-%m-%dT%H:%M:%S+00:00)

COUNT=$(aws s3api list-buckets --region "$REGION" \
    --query "length(Buckets[?CreationDate < '$CUTOFF'])" --output text)

cat > "$OUT" <<EOF
You have $COUNT S3 buckets that were created more than a day ago.
EOF
