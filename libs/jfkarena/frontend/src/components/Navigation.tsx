'use client'

import { useSession } from '@/features/auth/use-session'
import { authClient } from '@/lib/auth/authClient'
import { cn } from '@/lib/utils'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar'
import { Button } from './ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from './ui/dropdown-menu'

export default function Navigation() {
  const pathname = usePathname()
  const { data: session, isPending } = useSession()

  return (
    <nav className="fixed top-0 z-10 w-full border-b bg-background/80 backdrop-blur-sm">
      <div className="px-2 sm:px-6 flex h-14 items-center">
        <div className="mr-4 flex">
          <Link href="/" className="hidden sm:flex items-center space-x-2">
            <span className="text-xl">⚔️</span>
            <span className="font-medium whitespace-nowrap">JFK RAG Arena</span>
          </Link>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <Link className="flex items-center py-2.5 sm:pl-2 mr-2" target="_blank" href="https://discord.gg/mZ9Rc8q8W3">
            <div className="flex items-center gap-2">
              <svg className="shrink-0" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M10 19C14.9706 19 19 14.9706 19 10C19 5.02944 14.9706 1 10 1C5.02944 1 1 5.02944 1 10C1 14.9706 5.02944 19 10 19ZM11.8056 5.725C12.7206 5.86135 13.6104 6.10216 14.4515 6.44108C15.9031 8.30796 16.6243 10.4135 16.358 12.842C15.2474 13.5572 14.1701 13.9911 13.1109 14.275C12.8486 13.9653 12.6166 13.6352 12.4159 13.2892C12.7966 13.1645 13.1635 13.0107 13.5123 12.8296C13.4211 12.7707 13.3316 12.7097 13.2441 12.6468C11.159 13.4968 8.86613 13.4968 6.75592 12.6468C6.66873 12.7101 6.57924 12.7711 6.4876 12.8296C6.83584 13.0102 7.2021 13.1637 7.58203 13.2883C7.38247 13.6329 7.15001 13.9627 6.88703 14.2741C5.82887 13.9902 4.75257 13.5563 3.64195 12.842C3.41501 10.7471 3.86894 8.62206 5.54341 6.44285C6.38488 6.10284 7.27543 5.86141 8.19132 5.725C8.30634 5.90246 8.44251 6.14118 8.53429 6.33101C9.50267 6.20326 10.4801 6.20326 11.4667 6.33101C11.5585 6.14113 11.6916 5.90246 11.8056 5.725ZM6.71654 10.4241C6.71654 11.0443 7.23606 11.5537 7.86954 11.5537C8.51415 11.5537 9.02354 11.0443 9.02254 10.4241C9.03361 9.80302 8.51415 9.29374 7.86954 9.29374C7.22499 9.29374 6.71654 9.80393 6.71654 10.4241ZM10.9774 10.4241C10.9774 11.0443 11.4969 11.5537 12.1304 11.5537C12.775 11.5537 13.2834 11.0443 13.2834 10.4241C13.2945 9.80302 12.775 9.29374 12.1304 9.29374C11.4858 9.29374 10.9774 9.80393 10.9774 10.4241Z"
                  fill="#5865F2"
                />
              </svg>
              <p className="cursor-pointer text-sm text-muted-foreground font-medium hidden sm:block">Join our discord</p>
            </div>
          </Link>
          <nav className="flex items-center space-x-4 mr-10">
            <Link
              href="/battle"
              className={cn(
                'text-sm font-medium transition-colors hover:text-primary',
                pathname === '/battle'
                  ? 'text-primary'
                  : 'text-muted-foreground',
              )}
            >
              Battle
            </Link>
            <Link
              href="/leaderboard"
              className={cn(
                'text-sm font-medium transition-colors hover:text-primary',
                pathname === '/leaderboard'
                  ? 'text-primary'
                  : 'text-muted-foreground',
              )}
            >
              Leaderboard
            </Link>

            <Link
              href="/leaderboard/personal"
              className={cn(
                'text-sm font-medium transition-colors hover:text-primary',
                pathname === '/leaderboard/personal'
                  ? 'text-primary'
                  : 'text-muted-foreground',
              )}
            >
              Personal Leaderboard
            </Link>
          </nav>
          {!isPending && (
            <>
              {session?.user.id && !session.user.isAnonymous
                ? (
                    <DropdownMenu modal={false}>
                      <DropdownMenuTrigger>
                        <Avatar className="size-6">
                          <AvatarImage src={session.user.image ?? undefined} />
                          <AvatarFallback>
                            {session.user.name?.slice(0, 2)}
                          </AvatarFallback>
                        </Avatar>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="bg-background">
                        <DropdownMenuLabel className="text-sm font-normal">{session.user.email}</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem>
                          <Button variant="ghost" onClick={() => authClient.signOut()}>Logout</Button>
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )
                : (
                    <Button asChild size="sm">
                      <Link href="/login">Login</Link>
                    </Button>
                  )}
            </>
          )}
        </div>
      </div>
    </nav>
  )
}
