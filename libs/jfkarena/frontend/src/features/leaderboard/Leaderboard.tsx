import { Card } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { db } from '@/db'
import { models } from '@/db/schema/models'
import { desc } from 'drizzle-orm'

export default async function Leaderboard() {
  const data = await db.query.models.findMany({
    orderBy: [desc(models.elo)],
    where: (fields, { isNull }) => isNull(fields.userId),
  })

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
