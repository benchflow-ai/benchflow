'use client'

import { authClient } from '@/lib/auth/authClient'
import { useEffect } from 'react'
import { useSession } from './use-session'

export function AnonymouseSessionProvider() {
  const { data, isPending } = useSession()

  useEffect(() => {
    if (!isPending && !data?.user.id && !data?.user.isAnonymous) {
      authClient.signIn.anonymous()
    }
  }, [isPending, data?.user.id, data?.user.isAnonymous])

  return null
}
