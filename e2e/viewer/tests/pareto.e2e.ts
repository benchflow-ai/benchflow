// The Pareto view (`/pareto`): pass rate against cost, time, and tokens per
// trial, with the jobs no other job beats on each axis.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

const TWO_JOBS = '/pareto?runs=nop-hello,oracle-hello';

describe('pareto', () => {
  test('draws the time chart and says why there is no cost or token chart', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    await expect(screen.getByRole('heading', 'Pass rate against time')).toBeVisible();
    await expect(screen.getByRole('image', /^pass rate \(vertical\) against median time of a trial/)).toBeVisible();
    await expect(screen.getByText('No trial here recorded a cost (agent_result.cost_usd), so there is no cost frontier.')).toBeVisible();
    await expect(screen.getByText('No trial here recorded token usage.')).toBeVisible();
  });

  test('lists each job with its values and the axes it is on the frontier of', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    const jobs = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'frontier') });
    // letter, job, agent, trials, pass rate, cost, median time, tokens, frontier
    await expect(jobs.getByRole('row').filter({ hasText: 'nop-hello' }).getByRole('cell')).toHaveText([
      'A', 'nop-hello', 'nop', '2', '0.0% (0.0%–65.8%)', '—', '30s (2)', '—', 'time',
    ]);
    await expect(jobs.getByRole('row').filter({ hasText: 'oracle-hello' }).getByRole('cell')).toHaveText([
      'B', 'oracle-hello', 'oracle', '1', '100.0% (20.7%–100.0%)', '—', '35s (1)', '—', 'time',
    ]);
  });

  test('warns when the jobs did not run the same tasks', async ({ app, screen }) => {
    await app.open(TWO_JOBS);
    await expect(screen.getByText(/^These jobs did not all run the same tasks/)).toBeVisible();
    await app.open('/pareto?runs=nop-hello,oracle-hello&task=hello');
    await expect(screen.getByRole('heading', 'Pass rate against time')).toBeVisible();
    await expect(screen.getByText(/^These jobs did not all run the same tasks/)).toBeHidden();
  });
});
