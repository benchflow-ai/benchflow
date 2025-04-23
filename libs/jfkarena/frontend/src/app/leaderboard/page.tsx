import Header from '@/components/Header'
import jfk from '@/features/battle/jfk.png'
import Leaderboard from '@/features/leaderboard/Leaderboard'
import Image from 'next/image'
import { Suspense } from 'react'

export const dynamic = 'force-dynamic'

export default function LeaderboardPage() {
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

      <div className="mt-2">
        <Header />
        <div className="mt-8">
          <Suspense fallback={(
            <div className="flex justify-center items-center min-h-[200px]">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          )}
          >
            <Leaderboard />
          </Suspense>
        </div>
      </div>
    </div>
  )
}
