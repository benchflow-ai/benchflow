import { Progress } from '@/components/ui/progress'
import { MAX_TOKENS } from '../constants'

interface TokenProgressProps {
  estimatedTokens: number
}

export function TokenProgress({ estimatedTokens }: TokenProgressProps) {
  return (
    <div className="flex items-center gap-3 bg-white px-1">
      <div className="text-xs text-zinc-500">
        ~
        {estimatedTokens}
        /
        {MAX_TOKENS}
        {' '}
        tokens
      </div>
      <Progress
        value={(estimatedTokens / MAX_TOKENS) * 100}
        className="w-20 h-1"
      />
    </div>
  )
}
