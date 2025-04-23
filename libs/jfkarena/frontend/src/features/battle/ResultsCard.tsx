import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import Link from 'next/link'
import { useSession } from '../auth/use-session'

interface Model {
  id: string
  name: string
}

interface ResultsCardProps {
  voted: boolean
  isFlipped: boolean
  selectedModels: {
    model1: Model | null
    model2: Model | null
  }
}

export function ResultsCard({ voted, isFlipped, selectedModels }: ResultsCardProps) {
  const { data: session } = useSession()
  const isSignedIn = !!session?.user && !session.user.isAnonymous
  if (!voted)
    return null

  return (
    <>
      <Card className="p-4">
        <div className="space-y-3">
          <h2 className="text-sm font-medium">Results Revealed</h2>
          <Separator className="bg-zinc-100" />
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-zinc-500">Model A was:</p>
              <p className="text-sm font-medium">
                {isFlipped ? selectedModels.model2?.name : selectedModels.model1?.name}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500">Model B was:</p>
              <p className="text-sm font-medium">
                {isFlipped ? selectedModels.model1?.name : selectedModels.model2?.name}
              </p>
            </div>
          </div>
        </div>
      </Card>

      <div className="mt-4 text-center space-y-2">
        <p className="text-sm text-zinc-500">
          {isSignedIn
            ? (
                'Check out the leaderboard to see how these models rank overall!'
              )
            : (
                'Login to view your personal leaderboard.'
              )}
        </p>
        <Button variant="outline" asChild>
          <Link href={isSignedIn ? '/leaderboard/personal' : '/login?next=/leaderboard/personal'}>
            {isSignedIn ? 'View Leaderboard' : 'Login'}
          </Link>
        </Button>
      </div>
    </>
  )
}
