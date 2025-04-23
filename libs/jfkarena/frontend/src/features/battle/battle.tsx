'use client'

import type { VoteResult } from './types'
import Header from '@/components/Header'
import { useAction } from 'next-safe-action/hooks'
import Image from 'next/image'
import { useState } from 'react'
import { toast } from 'sonner'
import { voteAction } from '../leaderboard/_actions/voteAction'
import { BattleForm } from './BattleForm'
import { BattleResponses } from './BattleResponses'
import { useBattle } from './hooks/useBattle'
import jfk from './jfk.png'
import { ResultsCard } from './ResultsCard'
import { VotingSection } from './VotingSection'

export function Battle() {
  const {
    question,
    responses,
    loading,
    isFlipped,
    battleId,
    selectedModels,
    error: battleError,
    handleSubmit,
  } = useBattle()
  const [voted, setVoted] = useState(false)
  const { executeAsync: vote, isPending } = useAction(voteAction, {
    onSuccess() {
      setVoted(true)
    },
    onError: (error) => {
      setVoted(false)
      toast.error(error.error.serverError || 'Something went wrong')
    },
  })

  const handleVote = async (result: VoteResult) => {
    if (isPending)
      return
    if (!battleId || !question || !selectedModels || !selectedModels.model1 || !selectedModels.model2)
      return

    const finalResult = result
    let firstModelId = selectedModels.model1.id
    let secondModelId = selectedModels.model2.id

    if (isFlipped) {
      firstModelId = selectedModels.model2.id
      secondModelId = selectedModels.model1.id
    }

    await vote({
      result: finalResult,
      model1: firstModelId,
      model2: secondModelId,
      battleId,
    })
  }

  const error = battleError

  return (
    <div className="container py-10">

      <div className="pb-6 pt-12 grid place-items-center">
        <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
          <span>Powered by</span>
          <a
            href="https://benchflow.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 hover:text-foreground transition-colors"
          >
            <img src="/logo.svg" alt="BenchFlow Logo" className="h-4 w-4" />
            <span>BenchFlow</span>
          </a>
          <span className="mx-1">×</span>
          <a
            href="https://openrouter.ai/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 hover:text-foreground transition-colors"
          >
            <img src="/openrouter.svg" alt="OpenRouter Logo" className="h-5 w-5" />
            <span>OpenRouter</span>
          </a>
        </div>
        <Image className="w-[400px] border border-gray-200" src={jfk} alt="JFK" />
      </div>

      <Header />

      <div className="mt-8 space-y-6 max-w-3xl mx-auto">
        {error && (
          <div className="text-destructive text-center p-4">
            Error:
            {' '}
            {error}
          </div>
        )}

        <BattleForm
          onSubmit={(q) => {
            setVoted(false)
            return handleSubmit(q)
          }}
          loading={loading}
        />

        <BattleResponses
          responses={responses}
          voted={voted}
          isFlipped={isFlipped}
          selectedModels={selectedModels}
        />

        <VotingSection
          responses={responses}
          voted={voted}
          isVoting={isPending}
          onVote={handleVote}
        />

        <ResultsCard
          voted={voted}
          isFlipped={isFlipped}
          selectedModels={selectedModels}
        />
      </div>
    </div>
  )
}
