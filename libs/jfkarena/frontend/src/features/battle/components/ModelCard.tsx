import type { Model } from '../types'
import { Card } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'

interface ModelCardProps {
  title: string
  content: string | null
  model: Model | null
  voted: boolean
}

export function ModelCard({ title, content, model, voted }: ModelCardProps) {
  return (
    <Card className="p-4 min-h-[300px]">
      <div className="space-y-3">
        <h2 className="text-sm font-medium">
          {voted && model ? model.name : title}
        </h2>
        <Separator className="bg-zinc-100" />
        <div className="whitespace-pre-wrap text-sm text-zinc-600">
          {content || 'Waiting for response...'}
        </div>
      </div>
    </Card>
  )
}
