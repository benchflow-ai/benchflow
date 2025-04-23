'use server'

import { authorizedActionClient } from '@/lib/safe-action'
import { z } from 'zod'
import { updateOverallLeaderboard } from '../_mutations/updateOverallLeaderboard'
import { updatePersonalLeaderboard } from '../_mutations/updatePersonalLeaderboard'
import { updateBattleRecord } from '../_mutations/utils'

export const voteAction = authorizedActionClient.schema(z.object({
  result: z.string(),
  model1: z.string(),
  model2: z.string(),
  battleId: z.number(),
})).action(async ({ ctx, parsedInput }) => {
  const { result, model1, model2, battleId } = parsedInput
  const userId = ctx.session.user.id

  await Promise.all([
    updateBattleRecord({ battleId, userId, result, model1Id: model1, model2Id: model2 }),
    updateOverallLeaderboard({ result, model1, model2 }),
    updatePersonalLeaderboard({ result, model1, model2, userId }),
  ])

  return { success: true }
})
