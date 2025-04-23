import { neonConfig, Pool } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-serverless'
import ws from 'ws'
import * as battlesRelations from './relations/battels-relations'
import * as modelsRelations from './relations/models-relations'
import * as authSchema from './schema/auth'
import * as battlesSchema from './schema/battles'
import * as modelsSchema from './schema/models'

neonConfig.webSocketConstructor = ws

// eslint-disable-next-line node/prefer-global/process
const connectionString = process.env.DATABASE_URL

if (!connectionString) {
  throw new Error('DATABASE_URL is not set')
}

neonConfig.fetchEndpoint = (host) => {
  const [protocol, port] = host === 'localhost' ? ['http', 4444] : ['https', 443]
  return `${protocol}://${host}:${port}/sql`
}
const connectionStringUrl = new URL(connectionString)
neonConfig.useSecureWebSocket = connectionStringUrl.hostname !== 'localhost'
neonConfig.wsProxy = host => (host === 'localhost' ? `${host}:4444/v2` : `${host}/v2`)

const schema = {
  ...authSchema,
  ...modelsSchema,
  ...battlesSchema,

  ...modelsRelations,
  ...battlesRelations,
}

const pool = new Pool({ connectionString })
export const db = drizzle(pool, { schema })

export type DatabaseType = typeof db
export type TransactionType = Parameters<Parameters<DatabaseType['transaction']>[0]>[0]
