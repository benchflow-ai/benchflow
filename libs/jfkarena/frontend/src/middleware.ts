import type { NextRequest } from 'next/server'
import { getSessionCookie } from 'better-auth/cookies'
import { NextResponse } from 'next/server'
// eslint-disable-next-line node/prefer-global/process
const API_URL = process.env.NEXT_PUBLIC_API_URL || ''

export function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname

  if (pathname.startsWith('/api/proxy/')) {
    const sessionCookie = getSessionCookie(request, {})

    if (!sessionCookie) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const nextUrl = new URL(pathname.replace('/api/proxy/', ''), API_URL)
    return NextResponse.rewrite(
      nextUrl,
    )
  }

  return NextResponse.next()
}

export const config = {
  matcher: '/api/proxy/:path*',
}
