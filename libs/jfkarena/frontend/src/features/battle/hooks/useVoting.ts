import type { Model, VoteResult } from '../types'
import { useState } from 'react'

interface UseVotingProps {
  battleId: number | null
  question: string
  selectedModels: {
    model1: Model | null
    model2: Model | null
  }
  isFlipped: boolean
}

export function useVoting({ battleId, question, selectedModels, isFlipped }: UseVotingProps) {
  const [voted, setVoted] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleVote = async (result: VoteResult) => {
    if (voted || !selectedModels.model1 || !selectedModels.model2 || !battleId)
      return

    const finalResult = result
    let firstModelId = selectedModels.model1.id
    let secondModelId = selectedModels.model2.id

    if (isFlipped) {
      firstModelId = selectedModels.model2.id
      secondModelId = selectedModels.model1.id
    }

    try {
      const response = await fetch('/api/proxy/vote', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          result: finalResult,
          model1: firstModelId,
          model2: secondModelId,
          battle_id: battleId,
          question,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to submit vote')
      }

      setVoted(true)
    }
    catch (error) {
      console.error('Error voting:', error)
      setError('Failed to submit vote')
    }
  }

  return {
    voted,
    error,
    handleVote,
  }
}
