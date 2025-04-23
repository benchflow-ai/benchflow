import type { TransactionType } from '@/db'
import type { Model } from '../types'
import { DEFAULT_ELO } from '@/constants'
import { db } from '@/db'
import { models } from '@/db/schema/models'
import { eq } from 'drizzle-orm'
import { calculateEloChange, updateEloRatings, updateModelStats } from './utils'

// Fetch both global and personal models
async function fetchModels({ tx, userId, model1Id, model2Id }: {
  tx: TransactionType
  userId: string
  model1Id: string
  model2Id: string
}) {
  const allModelData = await tx.query.models.findMany({
    where: (models, { or, and, isNull }) => or(
      and(
        eq(models.modelId, model1Id),
        isNull(models.userId),
      ),
      and(
        eq(models.modelId, model2Id),
        isNull(models.userId),
      ),
      and(
        eq(models.userId, userId),
        or(
          eq(models.modelId, model1Id),
          eq(models.modelId, model2Id),
        ),
      ),
    ),
  })

  // Separate global and personal models
  const rawModel1Data = allModelData.find(m => m.modelId === model1Id && m.userId === null)
  const rawModel2Data = allModelData.find(m => m.modelId === model2Id && m.userId === null)
  const personalModel1Data = allModelData.find(m => m.modelId === model1Id && m.userId === userId)
  const personalModel2Data = allModelData.find(m => m.modelId === model2Id && m.userId === userId)

  if (!rawModel1Data || !rawModel2Data) {
    throw new Error('Model not found')
  }

  return {
    rawModel1Data,
    rawModel2Data,
    personalModel1Data,
    personalModel2Data,
  }
}

// Create personal models if they don't exist
async function ensurePersonalModels({ tx, userId, rawModel1Data, rawModel2Data, personalModel1Data, personalModel2Data }: {
  tx: TransactionType
  userId: string
  rawModel1Data: Model
  rawModel2Data: Model
  personalModel1Data: Model | undefined
  personalModel2Data: Model | undefined
}) {
  const missingModels = []

  if (!personalModel1Data) {
    missingModels.push({
      modelId: rawModel1Data.modelId,
      userId,
      name: rawModel1Data.name,
      wins: 0,
      losses: 0,
      draws: 0,
      invalid: 0,
      elo: DEFAULT_ELO,
    })
  }

  if (!personalModel2Data) {
    missingModels.push({
      modelId: rawModel2Data.modelId,
      userId,
      name: rawModel2Data.name,
      wins: 0,
      losses: 0,
      draws: 0,
      invalid: 0,
      elo: DEFAULT_ELO,
    })
  }

  if (missingModels.length === 0) {
    return {
      model1Personal: personalModel1Data,
      model2Personal: personalModel2Data,
    }
  }

  const insertedModels = await tx.insert(models)
    .values(missingModels)
    .returning()

  return {
    model1Personal: personalModel1Data ?? insertedModels.find(m => m.modelId === rawModel1Data.modelId),
    model2Personal: personalModel2Data ?? insertedModels.find(m => m.modelId === rawModel2Data.modelId),
  }
}

export async function updatePersonalLeaderboard({ userId, result, model1, model2 }: {
  userId: string
  result: string
  model1: string
  model2: string
}) {
  try {
    return await db.transaction(async (tx) => {
      // Get both global and personal models
      const { rawModel1Data, rawModel2Data, personalModel1Data, personalModel2Data }
        = await fetchModels({ tx, userId, model1Id: model1, model2Id: model2 })

      // Ensure personal models exist
      const { model1Personal, model2Personal }
        = await ensurePersonalModels({
          tx,
          userId,
          rawModel1Data,
          rawModel2Data,
          personalModel1Data,
          personalModel2Data,
        })

      if (!model1Personal || !model2Personal) {
        throw new Error('Failed to create personal models')
      }

      // Update personal model statistics
      await updateModelStats({
        tx,
        winner: result,
        model1Id: model1Personal.modelId!,
        model2Id: model2Personal.modelId!,
        userId,
      })

      // Update personal ELO ratings if there's a winner
      if (result === 'model1' || result === 'model2') {
        const model1Won = result === 'model1'
        const { newElo1, newElo2 } = calculateEloChange({
          model1: model1Personal,
          model2: model2Personal,
          model1Won,
        })
        await updateEloRatings({
          tx,
          model1Id: model1Personal.id!,
          model2Id: model2Personal.id!,
          newElo1,
          newElo2,
        })
      }

      return { status: 'success' }
    })
  }
  catch (error) {
    console.error('Error in vote action:', error)
    throw error
  }
}
