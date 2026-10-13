// The branches view (`/branches`): every `bench branch` fork in the jobs, its
// point, status, value and checkpoint, and its children with their advantages.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

// fixtures/fork-nop: one fork at the start of a nop trial, two children, the parent discarded.
const FORK = '/branches?runs=fork-nop';

describe('branches', () => {
  test('lists the fork with its value, checkpoint, and children', async ({ app, screen }) => {
    await app.open(FORK);
    await expect(screen.getByText('1 fork, 2 children (2 scored).', { exact: false })).toBeVisible();
    const forks = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'advantage') });
    await expect(forks.getByRole('row').nth(1)).toHaveText(
      'hello start · completed · value 0 ± 0 · checkpoint daytona 8.6s, deleted · session fresh · parent discarded · nop',
    );
    // node, child, status, reward, advantage, note
    await expect(forks.getByRole('row').nth(2).getByRole('cell')).toHaveText(['n1', 'c1', 'scored', '0', '0.000', '']);
    await expect(forks.getByRole('row').nth(3).getByRole('cell')).toHaveText(['n2', 'c2', 'scored', '0', '0.000', '']);
  });

  test("a child's label opens its trajectory, and back leads to its parent", async ({ app, screen, browser }) => {
    await app.open(FORK);
    // Labels come from the fork's labels.json, and rows from its tree.json.
    const forks = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'advantage') });
    await forks.getByRole('row').filter({ hasText: 'n2' }).getByRole('link', 'c2').tap();
    await expect(browser).toHaveURL('/?run=fork-nop%2Fhello__ec981dd5&branch=8b9d7724a324483c9b5f767f1b2e7155%2Fn2');
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
    await expect(header).toContainText('branch child n2 of fork 8b9d7724 at node root');
    await screen.getByRole('button', '← parent run').tap();
    await expect(browser).toHaveURL('/?run=fork-nop%2Fhello__ec981dd5');
    await expect(screen.getByRole('button', '← runs')).toBeVisible();
  });

  test('a job without forks says how to make one', async ({ app, screen }) => {
    await app.open('/branches?runs=oracle-hello');
    await expect(screen.getByText('No trial in the selected jobs has a tree.json (run `bench branch` to make one).')).toBeVisible();
  });
});
