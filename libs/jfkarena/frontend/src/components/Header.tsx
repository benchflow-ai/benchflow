'use client'
import Link from 'next/link'
import GitHubButton from 'react-github-btn'

export default function Header() {
  return (
    <div className="flex flex-col items-center space-y-2 text-center">
      <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
        <span className="whitespace-nowrap">JFK RAG Battle Arena</span>
        <div>
          <GitHubButton href="https://github.com/benchflow-ai/jfkarena" data-color-scheme="no-preference: light; light: light; dark: dark;" aria-label="Star benchflow-ai/jfkarena on GitHub">Star us</GitHubButton>
        </div>
      </h1>
      <p className="text-muted-foreground">
        Compare different AI models' responses to questions about
        {' '}
        <Link href="https://www.archives.gov/research/jfk/release-2025" target="_blank" className="underline">
          JFK files
        </Link>
        .
      </p>
    </div>
  )
}
