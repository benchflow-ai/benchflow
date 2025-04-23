import type { BattleResponse, SelectedModels } from './types'
import { ModelCard } from './components/ModelCard'

interface BattleResponsesProps {
  responses: BattleResponse | null
  voted: boolean
  isFlipped: boolean
  selectedModels: SelectedModels
}

export function BattleResponses({ responses, voted, isFlipped, selectedModels }: BattleResponsesProps) {
  const getResponseContent = (position: 'left' | 'right') => {
    if (!responses)
      return null
    const isLeft = position === 'left'
    return isFlipped
      ? (isLeft ? responses.response2 : responses.response1)
      : (isLeft ? responses.response1 : responses.response2)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <ModelCard
        title="Model A"
        content={getResponseContent('left')}
        model={isFlipped ? selectedModels.model2 : selectedModels.model1}
        voted={voted}
      />
      <ModelCard
        title="Model B"
        content={getResponseContent('right')}
        model={isFlipped ? selectedModels.model1 : selectedModels.model2}
        voted={voted}
      />
    </div>
  )
}
