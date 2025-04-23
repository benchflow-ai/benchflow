import { defineConfig } from 'drizzle-kit'

// eslint-disable-next-line node/prefer-global/process
const dbUrl = process.env.DATABASE_URL

if (!dbUrl) {
  throw new Error('DATABASE_URL is not set')
}

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/db/schema',
  out: './src/db/migrations',
  dbCredentials: {
    url: dbUrl,
  },
})
