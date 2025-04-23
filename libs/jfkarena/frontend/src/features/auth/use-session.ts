import { authClient } from '@/lib/auth/authClient'

export function useSession() {
  const { data, isPending } = authClient.useSession()

  return { data, isPending }
}
