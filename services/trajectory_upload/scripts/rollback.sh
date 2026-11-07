#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 <broker-revision> <image-tag>" >&2
    exit 2
fi

task_revision="$1"
task_image_tag="$2"
task_rg="${BENCHFLOW_UPLOAD_RESOURCE_GROUP:-tasksminer-upload-prod}"
task_acr="${BENCHFLOW_UPLOAD_ACR:-tasksminerregistry}"
task_broker="${BENCHFLOW_UPLOAD_BROKER_APP:-tasksminer-traj-broker}"
task_validator="${BENCHFLOW_UPLOAD_VALIDATOR_JOB:-tasksminer-traj-validator}"
task_image="${task_acr}.azurecr.io/trajectory-upload:${task_image_tag}"

if ! [[ "$task_revision" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,127}$ ]]; then
    echo "invalid broker revision" >&2
    exit 2
fi
if ! [[ "$task_image_tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "invalid trajectory upload image tag" >&2
    exit 2
fi

task_revision_count="$(az containerapp revision list \
    --resource-group "$task_rg" \
    --name "$task_broker" \
    --query "[?name=='${task_revision}'] | length(@)" -o tsv)"
if [[ "$task_revision_count" != "1" ]]; then
    echo "broker revision not found: ${task_revision}" >&2
    exit 1
fi

az containerapp ingress traffic set \
    --resource-group "$task_rg" \
    --name "$task_broker" \
    --revision-weight "${task_revision}=100" \
    --output none
az containerapp job update \
    --resource-group "$task_rg" \
    --name "$task_validator" \
    --image "$task_image" \
    --output none

"$(dirname "$0")/smoke-test.sh"
echo "trajectory upload rollback completed"
