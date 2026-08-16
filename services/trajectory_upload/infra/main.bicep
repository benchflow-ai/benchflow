targetScope = 'resourceGroup'

@description('Azure region for every regional trajectory-upload resource.')
param location string = resourceGroup().location

@description('Storage account containing bronze, silver, and gold data.')
param storageAccountName string

@description('Azure Container Registry used for broker and validator images.')
param containerRegistryName string

@description('Container Apps managed environment name.')
param containerAppsEnvironmentName string

@description('Broker Container App name.')
param brokerAppName string

@description('Validator Container Apps Job name.')
param validatorJobName string

@description('Validation queue name.')
param validationQueueName string = 'trajectory-validation'

@description('Upload ledger table name.')
param uploadLedgerTableName string = 'trajectoryuploads'

var tags = {
  environment: 'production'
  service: 'trajectory-upload'
}
var logAnalyticsName = '${containerAppsEnvironmentName}-logs'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      allowPermanentDelete: false
      enabled: false
    }
    isVersioningEnabled: true
  }
}

resource bronze 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'bronze'
  properties: {
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    publicAccess: 'None'
  }
}

resource silver 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'silver'
  properties: {
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    publicAccess: 'None'
  }
}

resource gold 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'gold'
  properties: {
    defaultEncryptionScope: '$account-encryption-key'
    denyEncryptionScopeOverride: false
    publicAccess: 'None'
  }
}

resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}

resource validationQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = {
  parent: queueService
  name: validationQueueName
}

resource tableService 'Microsoft.Storage/storageAccounts/tableServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}

resource uploadLedger 'Microsoft.Storage/storageAccounts/tableServices/tables@2023-05-01' = {
  parent: tableService
  name: uploadLedgerTableName
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'
    }
  }
}

resource blobDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'trajectory-upload-storage'
  scope: blobService
  properties: {
    workspaceId: logs.id
    logs: [
      {
        category: 'StorageRead'
        enabled: true
        retentionPolicy: {
          days: 0
          enabled: false
        }
      }
      {
        category: 'StorageWrite'
        enabled: true
        retentionPolicy: {
          days: 0
          enabled: false
        }
      }
      {
        category: 'StorageDelete'
        enabled: true
        retentionPolicy: {
          days: 0
          enabled: false
        }
      }
    ]
    metrics: [
      {
        category: 'Capacity'
        enabled: false
        retentionPolicy: {
          days: 0
          enabled: false
        }
      }
      {
        category: 'Transaction'
        enabled: false
        retentionPolicy: {
          days: 0
          enabled: false
        }
      }
    ]
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  // The registry API supports these hardening fields although the stable
  // Bicep type does not expose all of them yet.
  properties: any({
    adminUserEnabled: false
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    encryption: {
      status: 'disabled'
    }
    policies: {
      azureADAuthenticationAsArmPolicy: {
        status: 'enabled'
      }
    }
    publicNetworkAccess: 'Enabled'
  })
}

resource brokerIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${brokerAppName}-id'
  location: location
  tags: tags
}

resource validatorIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${validatorJobName}-id'
  location: location
  tags: tags
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvironmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
    peerAuthentication: {
      mtls: {
        enabled: false
      }
    }
    peerTrafficConfiguration: {
      encryption: {
        enabled: false
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    zoneRedundant: false
  }
}

output storageAccountId string = storage.id
output bronzeContainerId string = bronze.id
output validationQueueId string = validationQueue.id
output uploadLedgerTableId string = uploadLedger.id
output containerRegistryId string = registry.id
output brokerIdentityId string = brokerIdentity.id
output brokerIdentityPrincipalId string = brokerIdentity.properties.principalId
output validatorIdentityId string = validatorIdentity.id
output validatorIdentityPrincipalId string = validatorIdentity.properties.principalId
output containerAppsEnvironmentId string = environment.id
