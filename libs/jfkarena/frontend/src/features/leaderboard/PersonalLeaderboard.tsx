import { Card, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { db } from '@/db'
import { models } from '@/db/schema/models'
import { desc, eq } from 'drizzle-orm'

import Link from 'next/link'
import { redirect } from 'next/navigation'
import { getSession } from '../auth/get-session'

export async function PersonalLeaderboard() {
  const session = await getSession()
  const isSignedIn = !!session?.user.id && !session.user.isAnonymous

  if (!isSignedIn) {
    return (
      redirect('/login?next=/leaderboard/personal')
    )
  }

  const data = await db.query.models.findMany({
    where: eq(models.userId, session.user.id),
    orderBy: [desc(models.elo)],
  })

  if (data.length === 0) {
    return (
      <Card>
        <CardContent className="grid place-items-center">
          <p className="mt-8">
            Vote a
            {' '}
            <Link href="/battle" className="underline">battle</Link>
            {' '}
            to see your personal leaderboard
          </p>
        </CardContent>
      </Card>
    )
  }
  return (
    <Card>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">Rank</TableHead>
            <TableHead>Model</TableHead>
            <TableHead className="text-right">ELO</TableHead>
            <TableHead className="text-right">Wins</TableHead>
            <TableHead className="text-right">Losses</TableHead>
            <TableHead className="text-right">Draws</TableHead>
            <TableHead className="text-right">Invalid</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data?.map((model, index) => (
            <TableRow key={model.id}>
              <TableCell className="font-medium">{index + 1}</TableCell>
              <TableCell>{model.name}</TableCell>
              <TableCell className="text-right">{Math.round(model.elo ?? 0)}</TableCell>
              <TableCell className="text-right text-green-600">{model.wins}</TableCell>
              <TableCell className="text-right text-red-600">{model.losses}</TableCell>
              <TableCell className="text-right text-yellow-600">{model.draws}</TableCell>
              <TableCell className="text-right text-muted-foreground">{model.invalid}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  )
}
