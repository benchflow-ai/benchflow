import { foreignKey, integer, pgTable, serial, timestamp, varchar } from 'drizzle-orm/pg-core'
import { user } from './auth'
import { models } from './models'

export const battles = pgTable('battles', {
  id: serial().primaryKey().notNull(),
  model1Id: integer('model1_id'),
  model2Id: integer('model2_id'),
  winnerId: integer('winner_id'),
  userId: varchar('user_id'),
  question: varchar(),
  response1: varchar(),
  response2: varchar(),
  result: varchar(),
  createdAt: timestamp('created_at', { mode: 'string' }),
  votedAt: timestamp('voted_at', { mode: 'string' }),
}, table => [
  foreignKey({
    columns: [table.userId],
    foreignColumns: [user.id],
    name: 'battles_user_id_fkey',
  }).onDelete('cascade'),
  foreignKey({
    columns: [table.model1Id],
    foreignColumns: [models.id],
    name: 'battles_model1_id_fkey',
  }).onDelete('cascade'),
  foreignKey({
    columns: [table.model2Id],
    foreignColumns: [models.id],
    name: 'battles_model2_id_fkey',
  }).onDelete('cascade'),
  foreignKey({
    columns: [table.winnerId],
    foreignColumns: [models.id],
    name: 'battles_winner_id_fkey',
  }).onDelete('cascade'),
])
