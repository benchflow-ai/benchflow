import type { BattleResponse, SelectedModels } from '../types'
import { MODELS } from '@/constants'
import { useState } from 'react'
import { toast } from 'sonner'
import useSWRMutation from 'swr/mutation'

async function sendBattleRequest(url: string, { arg }: { arg: { model1: string, model2: string, question: string } }) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(arg),
  })

  if (!response.ok) {
    const errorText = await response.text()
    console.error('Response error:', {
      status: response.status,
      statusText: response.statusText,
      body: errorText,
    })
    throw new Error(`Failed to fetch responses: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

export function useBattle() {
  const [question, setQuestion] = useState('')
  const [responses, setResponses] = useState<BattleResponse | null>(null)
  const [isFlipped, setIsFlipped] = useState(false)
  const [battleId, setBattleId] = useState<number | null>(null)
  const [selectedModels, setSelectedModels] = useState<SelectedModels>({
    model1: null,
    model2: null,
  })
  const [error, setError] = useState<string | null>(null)

  const { trigger, isMutating: loading } = useSWRMutation('/api/proxy/battle', sendBattleRequest)

  const selectRandomModels = () => {
    if (MODELS.length < 2) {
      console.warn('Not enough models available')
      return null
    }
    const availableModels = [...MODELS]
    const model1Index = Math.floor(Math.random() * availableModels.length)
    const model1 = availableModels[model1Index]
    availableModels.splice(model1Index, 1)
    const model2Index = Math.floor(Math.random() * availableModels.length)
    const model2 = availableModels[model2Index]
    return { model1, model2 }
  }

  const handleSubmit = async (questionText: string) => {
    if (!questionText)
      return

    const selected = selectRandomModels()
    if (!selected) {
      setError('Not enough models available')
      return
    }
    setResponses(null)
    setBattleId(null)
    setIsFlipped(false)

    setSelectedModels(selected)
    setQuestion(questionText)

    try {
      setError(null)
      const data = await trigger({
        model1: selected.model1.id,
        model2: selected.model2.id,
        question: questionText,
      })

      setResponses(data)
      setBattleId(data.battle_id)
      setIsFlipped(Math.random() > 0.5)
    }
    catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to get responses'
      console.error('Error details:', error)
      toast.error(message)
      setError(message)
    }
  }

  return {
    question,
    responses,
    loading,
    isFlipped,
    battleId,
    selectedModels,
    error,
    handleSubmit,
  }
}
