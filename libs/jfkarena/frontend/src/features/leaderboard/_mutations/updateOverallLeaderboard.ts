import { db } from '@/db'
import { calculateEloChange, updateEloRatings, updateModelStats } from './utils'

export async function updateOverallLeaderboard({ result, model1, model2 }: {
  result: string
  model1: string
  model2: string
}) {
  try {
    // Get current status of both models
    const [model1Data, model2Data] = await Promise.all([
      db.query.models.findFirst({
        where: (models, { eq }) => eq(models.modelId, model1),
      }),
      db.query.models.findFirst({
        where: (models, { eq }) => eq(models.modelId, model2),
      }),
    ])

    if (!model1Data || !model2Data) {
      throw new Error('Model not found')
    }

    // Start transaction
    return await db.transaction(async (tx) => {
      // Update model statistics
      await updateModelStats({ tx, winner: result, model1Id: model1, model2Id: model2 })

      // Update ELO ratings if there's a winner
      if (result === 'model1' || result === 'model2') {
        const model1Won = result === 'model1'
        const { newElo1, newElo2 } = calculateEloChange({
          model1: model1Data,
          model2: model2Data,
          model1Won,
        })
        await updateEloRatings({ tx, model1Id: model1Data.id!, model2Id: model2Data.id!, newElo1, newElo2 })
      }

      return { status: 'success' }
    })
  }
  catch (error) {
    console.error('Error in vote action:', error)
    throw error
  }
}
