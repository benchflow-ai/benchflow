import type { models } from '@/db/schema/models'
import type { InferSelectModel } from 'drizzle-orm'

export type Model = InferSelectModel<typeof models>
