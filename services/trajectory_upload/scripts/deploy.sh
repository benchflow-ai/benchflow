#!/usr/bin/env bash
set -euo pipefail

task_rg="${BENCHFLOW_UPLOAD_RESOURCE_GROUP:-tasksminer-upload-prod}"
task_location="${BENCHFLOW_UPLOAD_LOCATION:-westus2}"
task_storage="${BENCHFLOW_UPLOAD_STORAGE_ACCOUNT:-tasksminerdata}"
task_acr="${BENCHFLOW_UPLOAD_ACR:-tasksminerregistry}"
task_environment="${BENCHFLOW_UPLOAD_ENVIRONMENT:-tasksminer-upload}"
task_broker="${BENCHFLOW_UPLOAD_BROKER_APP:-tasksminer-traj-broker}"
task_validator="${BENCHFLOW_UPLOAD_VALIDATOR_JOB:-tasksminer-traj-validator}"
task_queue="trajectory-validation"
task_table="trajectoryuploads"
task_subscription="$(az account show --query id -o tsv)"
task_repo_root="$(git rev-parse --show-toplevel)"
task_image_tag="${BENCHFLOW_UPLOAD_IMAGE_TAG:-$(git rev-parse --short=12 HEAD)}"
task_image="${task_acr}.azurecr.io/trajectory-upload:${task_image_tag}"
task_service_root="${task_repo_root}/services/trajectory_upload"

if ! [[ "$task_image_tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "invalid trajectory upload image tag" >&2
    exit 2
fi

for task_provider in \
    Microsoft.App \
    Microsoft.ContainerRegistry \
    Microsoft.EventGrid \
    Microsoft.Insights \
    Microsoft.ManagedIdentity \
    Microsoft.OperationalInsights \
    Microsoft.Storage; do
    az provider register --namespace "$task_provider" --wait --output none
done
az extension add \
    --name containerapp \
    --upgrade \
    --allow-preview true \
    --yes \
    --output none

ensure_role_assignment() {
    local principal_id="$1"
    local role="$2"
    local scope="$3"
    local present
    present="$(az role assignment list \
        --assignee-object-id "$principal_id" \
        --scope "$scope" \
        --query "[?roleDefinitionName=='$role'] | length(@)" \
        -o tsv)"
    if [[ "$present" == "0" ]]; then
        az role assignment create \
            --assignee-object-id "$principal_id" \
            --role "$role" \
            --scope "$scope" \
            --output none
    fi
}

az group create \
    --name "$task_rg" \
    --location "$task_location" \
    --tags service=trajectory-upload environment=production \
    --output none

az deployment group create \
    --name "trajectory-upload-${task_image_tag}" \
    --resource-group "$task_rg" \
    --template-file "${task_service_root}/infra/main.bicep" \
    --parameters "${task_service_root}/infra/production.bicepparam" \
    --parameters \
        "location=${task_location}" \
        "storageAccountName=${task_storage}" \
        "containerRegistryName=${task_acr}" \
        "containerAppsEnvironmentName=${task_environment}" \
        "brokerAppName=${task_broker}" \
        "validatorJobName=${task_validator}" \
        "validationQueueName=${task_queue}" \
        "uploadLedgerTableName=${task_table}" \
    --output none

task_storage_id="$(az storage account show \
    --name "$task_storage" \
    --resource-group "$task_rg" \
    --query id -o tsv)"
task_blob_scope="${task_storage_id}/blobServices/default"
task_bronze_scope="${task_blob_scope}/containers/bronze"
task_queue_scope="${task_storage_id}/queueServices/default/queues/${task_queue}"
task_table_scope="${task_storage_id}/tableServices/default/tables/${task_table}"

az storage account management-policy create \
    --account-name "$task_storage" \
    --resource-group "$task_rg" \
    --policy "@${task_service_root}/infra/lifecycle.json" \
    --output none

task_acr_id="$(az acr show --name "$task_acr" --query id -o tsv)"

task_broker_identity="${task_broker}-id"
task_validator_identity="${task_validator}-id"
task_broker_identity_id="$(az identity show -g "$task_rg" -n "$task_broker_identity" --query id -o tsv)"
task_broker_principal_id="$(az identity show -g "$task_rg" -n "$task_broker_identity" --query principalId -o tsv)"
task_validator_identity_id="$(az identity show -g "$task_rg" -n "$task_validator_identity" --query id -o tsv)"
task_validator_principal_id="$(az identity show -g "$task_rg" -n "$task_validator_identity" --query principalId -o tsv)"

task_role_name="TasksMiner Blob Data Creator"
task_role_exists="$(az role definition list --name "$task_role_name" --query 'length(@)' -o tsv)"
if [[ "$task_role_exists" == "0" ]]; then
    task_role_json="$(jq -n \
        --arg scope "/subscriptions/${task_subscription}" \
        --arg name "$task_role_name" \
        '{Name:$name,Description:"Create and write blobs without read, list, or delete data actions.",Actions:[],NotActions:[],DataActions:["Microsoft.Storage/storageAccounts/blobServices/containers/blobs/write","Microsoft.Storage/storageAccounts/blobServices/containers/blobs/add/action"],NotDataActions:[],AssignableScopes:[$scope]}')"
    az role definition create --role-definition "$task_role_json" --output none
fi

ensure_role_assignment "$task_broker_principal_id" "$task_role_name" "$task_bronze_scope"
ensure_role_assignment "$task_broker_principal_id" "Storage Blob Delegator" "$task_storage_id"
ensure_role_assignment "$task_broker_principal_id" "Storage Table Data Contributor" "$task_table_scope"
ensure_role_assignment "$task_broker_principal_id" "AcrPull" "$task_acr_id"

ensure_role_assignment "$task_validator_principal_id" "Storage Blob Data Contributor" "$task_bronze_scope"
ensure_role_assignment "$task_validator_principal_id" "Storage Queue Data Reader" "$task_queue_scope"
ensure_role_assignment "$task_validator_principal_id" "Storage Queue Data Message Processor" "$task_queue_scope"
ensure_role_assignment "$task_validator_principal_id" "Storage Table Data Contributor" "$task_table_scope"
ensure_role_assignment "$task_validator_principal_id" "AcrPull" "$task_acr_id"

# Event Grid needs an identity on the parent system topic before an event
# subscription can use managed-identity delivery. Creating a subscription
# directly on the storage source produces an identity-less implicit topic.
task_system_topic="$(az eventgrid system-topic list \
    --resource-group "$task_rg" \
    --output json | jq -r \
        --arg source "$task_storage_id" \
        'map(select((.source | ascii_downcase) == ($source | ascii_downcase)))[0].name // empty')"
if [[ -z "$task_system_topic" ]]; then
    task_system_topic="trajectory-storage-events"
    az eventgrid system-topic create \
        --resource-group "$task_rg" \
        --name "$task_system_topic" \
        --location "$task_location" \
        --topic-type microsoft.storage.storageaccounts \
        --source "$task_storage_id" \
        --identity systemassigned \
        --output none
else
    az eventgrid system-topic update \
        --resource-group "$task_rg" \
        --name "$task_system_topic" \
        --identity systemassigned \
        --output none
fi
task_system_topic_id="$(az eventgrid system-topic show \
    --resource-group "$task_rg" \
    --name "$task_system_topic" \
    --query id -o tsv)"
task_system_topic_principal_id="$(az eventgrid system-topic show \
    --resource-group "$task_rg" \
    --name "$task_system_topic" \
    --query identity.principalId -o tsv)"
ensure_role_assignment \
    "$task_system_topic_principal_id" \
    "Storage Queue Data Message Sender" \
    "$task_queue_scope"

# Early revisions scoped this role at the full storage account. This topic is
# dedicated to one queue, so converge existing deployments to the narrow scope.
task_legacy_event_sender_count="$(az role assignment list \
    --assignee-object-id "$task_system_topic_principal_id" \
    --scope "$task_storage_id" \
    --query "[?roleDefinitionName=='Storage Queue Data Message Sender'] | length(@)" \
    -o tsv)"
if [[ "$task_legacy_event_sender_count" != "0" ]]; then
    az role assignment delete \
        --assignee-object-id "$task_system_topic_principal_id" \
        --role "Storage Queue Data Message Sender" \
        --scope "$task_storage_id"
fi

az acr build \
    --registry "$task_acr" \
    --image "trajectory-upload:${task_image_tag}" \
    --file services/trajectory_upload/Dockerfile \
    "$task_repo_root" \
    --output none

task_ip_hash_key="$(openssl rand -hex 32)"
az containerapp create \
    --name "$task_broker" \
    --resource-group "$task_rg" \
    --environment "$task_environment" \
    --image "$task_image" \
    --user-assigned "$task_broker_identity_id" \
    --registry-server "${task_acr}.azurecr.io" \
    --registry-identity "$task_broker_identity_id" \
    --ingress external \
    --target-port 8000 \
    --transport http \
    --allow-insecure false \
    --min-replicas 0 \
    --max-replicas 1 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --secrets "ip-hash-key=${task_ip_hash_key}" \
    --env-vars \
        "AZURE_CLIENT_ID=$(az identity show -g "$task_rg" -n "$task_broker_identity" --query clientId -o tsv)" \
        "AZURE_STORAGE_ACCOUNT_NAME=${task_storage}" \
        "AZURE_BLOB_CONTAINER=bronze" \
        "AZURE_LEDGER_TABLE=${task_table}" \
        "TRAJ_UPLOAD_IP_HASH_KEY=secretref:ip-hash-key" \
        "TRAJ_UPLOAD_RATE_LIMIT=20" \
        "TRAJ_UPLOAD_SAS_MINUTES=15" \
    --output none

az containerapp job create \
    --name "$task_validator" \
    --resource-group "$task_rg" \
    --environment "$task_environment" \
    --trigger-type Event \
    --image "$task_image" \
    --mi-user-assigned "$task_validator_identity_id" \
    --registry-server "${task_acr}.azurecr.io" \
    --registry-identity "$task_validator_identity_id" \
    --command /app/services/trajectory_upload/validator-entrypoint \
    --cpu 1.0 \
    --memory 2.0Gi \
    --replica-timeout 1800 \
    --replica-retry-limit 3 \
    --parallelism 1 \
    --replica-completion-count 1 \
    --polling-interval 30 \
    --min-executions 0 \
    --max-executions 5 \
    --scale-rule-name trajectory-queue \
    --scale-rule-type azure-queue \
    --scale-rule-metadata \
        "accountName=${task_storage}" \
        "cloud=AzurePublicCloud" \
        "queueLength=1" \
        "queueName=${task_queue}" \
    --scale-rule-identity "$task_validator_identity_id" \
    --env-vars \
        "AZURE_CLIENT_ID=$(az identity show -g "$task_rg" -n "$task_validator_identity" --query clientId -o tsv)" \
        "AZURE_STORAGE_ACCOUNT_NAME=${task_storage}" \
        "AZURE_BLOB_CONTAINER=bronze" \
        "AZURE_VALIDATION_QUEUE=${task_queue}" \
        "AZURE_LEDGER_TABLE=${task_table}" \
    --output none

task_event_name="trajectory-manifest-created"
task_event_body="$(jq -n \
    --arg storage "$task_storage_id" \
    '{properties:{deliveryWithResourceIdentity:{identity:{type:"SystemAssigned"},destination:{endpointType:"StorageQueue",properties:{resourceId:$storage,queueName:"trajectory-validation",queueMessageTimeToLiveInSeconds:604800}}},filter:{isSubjectCaseSensitive:false,subjectBeginsWith:"/blobServices/default/containers/bronze/blobs/inbox/",includedEventTypes:["Microsoft.Storage.BlobCreated"]},retryPolicy:{maxDeliveryAttempts:30,eventTimeToLiveInMinutes:1440},eventDeliverySchema:"EventGridSchema"}}')"
az rest \
    --method put \
    --url "https://management.azure.com${task_system_topic_id}/eventSubscriptions/${task_event_name}?api-version=2025-02-15" \
    --body "$task_event_body" \
    --output none

az storage account update \
    --name "$task_storage" \
    --resource-group "$task_rg" \
    --allow-blob-public-access false \
    --allow-shared-key-access false \
    --min-tls-version TLS1_2 \
    --https-only true \
    --output none

echo "trajectory upload deployment completed for image ${task_image_tag}"
