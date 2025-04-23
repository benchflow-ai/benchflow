import type { Model } from '../types'
import { useSession } from '@/features/auth/use-session'
import useSWR from 'swr'

async function fetchModels() {
  const response = await fetch('/api/proxy/models')
  if (!response.ok)
    throw new Error('Failed to fetch models')

  return response.json()
}

export function useModels() {
  const { data: session } = useSession()
  const isSignedIn = !!session?.user.id
  const { data: models = [], error } = useSWR<Model[]>(isSignedIn ? '/api/proxy/models' : null, fetchModels, { revalidateOnFocus: false })

  return {
    models,
    error: error?.message || null,
  }
}
