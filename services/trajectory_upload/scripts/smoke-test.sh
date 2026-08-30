#!/usr/bin/env bash
set -euo pipefail

task_rg="${BENCHFLOW_UPLOAD_RESOURCE_GROUP:-tasksminer-upload-prod}"
task_storage="${BENCHFLOW_UPLOAD_STORAGE_ACCOUNT:-tasksminerdata}"
task_broker="${BENCHFLOW_UPLOAD_BROKER_APP:-tasksminer-traj-broker}"
task_validator="${BENCHFLOW_UPLOAD_VALIDATOR_JOB:-tasksminer-traj-validator}"

task_broker_state="$(az containerapp show \
    --resource-group "$task_rg" \
    --name "$task_broker" \
    --query properties.provisioningState -o tsv)"
task_validator_state="$(az containerapp job show \
    --resource-group "$task_rg" \
    --name "$task_validator" \
    --query properties.provisioningState -o tsv)"
if [[ "$task_broker_state" != "Succeeded" || "$task_validator_state" != "Succeeded" ]]; then
    echo "trajectory upload compute is not fully provisioned" >&2
    exit 1
fi

task_broker_image="$(az containerapp show \
    --resource-group "$task_rg" \
    --name "$task_broker" \
    --query properties.template.containers[0].image -o tsv)"
task_validator_image="$(az containerapp job show \
    --resource-group "$task_rg" \
    --name "$task_validator" \
    --query properties.template.containers[0].image -o tsv)"
if [[ -z "$task_broker_image" || "$task_broker_image" != "$task_validator_image" ]]; then
    echo "broker and validator do not run the same image" >&2
    exit 1
fi

task_storage_state="$(az storage account show \
    --resource-group "$task_rg" \
    --name "$task_storage" \
    --query "join('|', [to_string(allowBlobPublicAccess), to_string(allowSharedKeyAccess), to_string(enableHttpsTrafficOnly), minimumTlsVersion])" \
    -o tsv)"
if [[ "$task_storage_state" != "false|false|true|TLS1_2" ]]; then
    echo "trajectory upload storage hardening drifted" >&2
    exit 1
fi

task_storage_id="$(az storage account show \
    --resource-group "$task_rg" \
    --name "$task_storage" \
    --query id -o tsv)"
for task_container in bronze silver gold; do
    task_public_access="$(az rest \
        --method get \
        --url "https://management.azure.com${task_storage_id}/blobServices/default/containers/${task_container}?api-version=2023-05-01" \
        --query properties.publicAccess -o tsv)"
    if [[ -n "$task_public_access" && "$task_public_access" != "None" ]]; then
        echo "${task_container} container unexpectedly allows public access" >&2
        exit 1
    fi
done

task_broker_fqdn="$(az containerapp show \
    --resource-group "$task_rg" \
    --name "$task_broker" \
    --query properties.configuration.ingress.fqdn -o tsv)"
task_health="$(curl --fail --silent --show-error \
    --max-time 30 \
    "https://${task_broker_fqdn}/healthz")"
if [[ "$task_health" != *'"status":"ok"'* ]]; then
    echo "trajectory upload broker health response is invalid" >&2
    exit 1
fi

echo "trajectory upload production smoke test passed"
