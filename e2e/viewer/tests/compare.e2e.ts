// The compare view (`/compare`): pass rates by job, paired differences, and
// the tasks grid, with the jobs and filters kept in the URL.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

const TWO_JOBS = '/compare?runs=nop-hello,oracle-hello';

describe('compare', () => {
  test('each job gets its pass rate with a 95% Wilson interval', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    const rates = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'mean reward') });
    // letter, job, agent, tasks, trials, passed, failed, unscored, errored, stopped, pass rate, mean reward
    await expect(rates.getByRole('row').filter({ hasText: 'nop-hello' }).getByRole('cell')).toHaveText([
      'A', 'nop-hello', 'nop', '2', '2', '0', '2', '0', '0', '0', '0.0% (0.0%–65.8%)', '0',
    ]);
    await expect(rates.getByRole('row').filter({ hasText: 'oracle-hello' }).getByRole('cell')).toHaveText([
      'B', 'oracle-hello', 'oracle', '1', '1', '1', '0', '0', '0', '0', '100.0% (20.7%–100.0%)', '1',
    ]);
  });

  test('the paired difference covers the shared task and gives no interval for one task', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    const pairs = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'shared tasks') });
    await expect(pairs.getByRole('row').nth(1).getByRole('cell')).toHaveText([
      'A', 'B', '1', '−1.000 (no interval: too few tasks)', '0', '1', '0', '0', '1',
    ]);
  });

  test('a chip in the tasks grid opens that trial', async ({ app, screen, browser }) => {
    await app.open(TWO_JOBS);
    const grid = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'task') });
    await expect(grid.getByRole('row')).toHaveCount(3);
    const hello = grid.getByRole('row').filter({ has: screen.getByRole('cell', 'hello') });
    await expect(hello.getByRole('link')).toHaveText(['0', '1']);
    await hello.getByRole('link', '1').tap();
    await expect(browser).toHaveURL('/?run=oracle-hello%2Fhello__5233d73b');
    await expect(screen.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
  });

  test('the status filter keeps the rows with that outcome, in the URL', async ({ app, screen, browser }) => {
    await app.open(TWO_JOBS);
    await screen.getByRole('group', 'status filter').getByRole('button', 'passed').tap();
    await expect(browser).toHaveURL('/compare?runs=nop-hello,oracle-hello&status=passed');
    await expect(screen.getByRole('button', 'passed')).toHaveAttribute('aria-pressed', 'true');
    await expect(screen.getByText(/^1 task\. /)).toBeVisible();
    const rates = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'mean reward') });
    await expect(rates.getByRole('row').filter({ hasText: 'nop-hello' }).getByRole('cell').nth(4)).toHaveText('1');
  });

  test('the agent filter chooses trials, in the URL', async ({ app, screen, browser }) => {
    await app.open(TWO_JOBS);
    await screen.getByRole('combobox', 'agent').selectOption('oracle');
    await expect(browser).toHaveURL('/compare?runs=nop-hello,oracle-hello&agent=oracle');
    const rates = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'mean reward') });
    await expect(rates.getByRole('row').filter({ hasText: 'nop-hello' }).getByRole('cell').nth(4)).toHaveText('0');
    await expect(rates.getByRole('row').filter({ hasText: 'oracle-hello' }).getByRole('cell').nth(4)).toHaveText('1');
  });

  test('the jobs menu adds a job as a new column', async ({ app, screen, browser }) => {
    await app.open(TWO_JOBS);
    // Five of the six served jobs: fork-nop has no trial to compare (below).
    await screen.getByText('jobs: 2 of 5 ▾').tap();
    await screen.getByRole('checkbox', /^timeouts/).check();
    await expect(browser).toHaveURL('/compare?runs=nop-hello,oracle-hello,timeouts');
    await expect(screen.getByText('jobs: 3 of 5 ▾')).toBeVisible();
    await expect(screen.getByRole('columnheader', 'C · timeouts oracle')).toBeVisible();
  });

  test('a job that is not served is named in a warning', async ({ app, screen }) => {
    await app.open('/compare?runs=nope');
    await expect(screen.getByText('Not found here: nope. The jobs on offer are in the jobs menu.')).toBeVisible();
  });

  // G10's bug bash, finding 3: fork-nop is served, but its only trial is a
  // branch parent discarded at its checkpoint, which this page leaves out, so
  // the jobs menu does not offer it (5 jobs here, 6 on /branches). The page
  // says why it has no trial instead of calling it not found.
  test('a served job with only a discarded branch parent says why it has no trial', async ({ app, screen, browser }) => {
    await app.open('/compare?runs=fork-nop');
    const main = screen.getByRole('main');
    await expect(
      main.getByText(
        'fork-nop has no trial here but branch parents discarded at their checkpoint, which this page leaves out; its forks are on branches.',
      ),
    ).toBeVisible();
    // Said once: not "Not found here", and no "No trials match." under it.
    await expect(main).not.toContainText('Not found here');
    await expect(main).not.toContainText('No trials match.');
    await main.getByRole('link', 'branches').tap();
    await expect(browser).toHaveURL('/branches?runs=fork-nop');
    await expect(screen.getByRole('link', 'hello')).toBeVisible();
  });

  test('a set-aside job beside a compared one gets the note above the comparison', async ({ app, screen }) => {
    await app.open('/compare?runs=fork-nop,oracle-hello');
    const main = screen.getByRole('main');
    await expect(main.getByText(/^fork-nop has no trial here but branch parents/)).toBeVisible();
    await expect(main.getByRole('columnheader', 'A · oracle-hello oracle')).toBeVisible();
  });

  test('the links to the other views carry the same jobs', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    const views = screen.getByRole('navigation', 'views');
    await expect(views.getByRole('link', 'outcomes')).toHaveAttribute('href', '/outcomes?run=nop-hello,oracle-hello');
    await expect(views.getByRole('link', 'pareto')).toHaveAttribute('href', '/pareto?runs=nop-hello,oracle-hello');
    await expect(views.getByRole('link', 'compare')).toHaveAttribute('aria-current', 'page');
  });

  test('/api/compare answers the same query as JSON', async ({ app }) => {
    const response = await fetch(new URL('/api/compare?runs=nop-hello,oracle-hello', app.baseUrl));
    expect(response.status).toBe(200);
    const view = await response.json();
    expect(view.columns.map((column: { job_name: string; pass_rate: number }) => [column.job_name, column.pass_rate])).toEqual([
      ['nop-hello', 0],
      ['oracle-hello', 1],
    ]);
  });
});
