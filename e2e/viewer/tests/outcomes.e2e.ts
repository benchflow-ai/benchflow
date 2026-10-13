// The outcomes view (`/outcomes`): how each job's trials ended, the problems
// behind them by owner and code, and the time of each phase.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

// fixtures/timeouts: one trial passed, one failed, one unscored (R303), one errored (R101).
const TIMEOUTS = '/outcomes?run=timeouts';

describe('outcomes', () => {
  test('counts how each trial ended', async ({ app, screen }) => {
    await app.open(TIMEOUTS);
    const counts = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'stopped') });
    const row = counts.getByRole('row').filter({ has: screen.getByRole('cell', 'timeouts') });
    // job, trials, bar, then passed, failed, unscored, errored, stopped
    await expect(row.getByRole('cell')).toHaveText([
      'timeouts', '4', '', '1 (25.0%)', '1 (25.0%)', '1 (25.0%)', '1 (25.0%)', '0 (0.0%)',
    ]);
  });

  test('names the owner of the problems', async ({ app, screen }) => {
    await app.open(TIMEOUTS);
    const owners = screen.getByRole('table').filter({ has: screen.getByRole('row', 'owner trials by job') });
    await expect(owners.getByRole('row').nth(1).getByRole('cell')).toHaveText(['the task', '2', '—']);
  });

  test('lists each problem code with its message, next step, and trials', async ({ app, screen, browser }) => {
    await app.open(TIMEOUTS);
    const codes = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'next step') });
    await expect(codes.getByRole('row')).toHaveCount(3);
    const timeout = codes.getByRole('row').filter({ has: screen.getByRole('cell', 'R303') });
    await expect(timeout).toContainText('the verifier ran past its 5s timeout');
    await expect(timeout).toContainText('raise [verifier] timeout');
    const build = codes.getByRole('row').filter({ has: screen.getByRole('cell', 'R101') });
    await expect(build).toContainText("the task's Dockerfile failed to build");
    // The trials behind a code open from its list.
    await timeout.getByText('1 trial').tap();
    await timeout.getByRole('link', 'verifier-timeout').tap();
    await expect(browser).toHaveURL('/?run=timeouts%2Fverifier-timeout__c41d891b');
  });

  test('gives the median and 90th percentile of each phase', async ({ app, screen }) => {
    await app.open(TIMEOUTS);
    const time = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'verifier median') });
    await expect(time.getByRole('row').filter({ has: screen.getByRole('cell', 'timeouts') }).getByRole('cell')).toHaveText([
      'timeouts', '1m 11s', '2m 8s', '4', '13s', '33s', '4', '28s', '52s', '4',
    ]);
  });

  test('the status filter keeps the trials with that outcome', async ({ app, screen, browser }) => {
    await app.open(TIMEOUTS);
    await screen.getByRole('group', 'status filter').getByRole('button', 'errored').tap();
    await expect(browser).toHaveURL('/outcomes?run=timeouts&status=errored');
    const codes = screen.getByRole('table').filter({ has: screen.getByRole('columnheader', 'next step') });
    await expect(codes.getByRole('row')).toHaveCount(2);
    await expect(codes.getByRole('cell', 'R101')).toBeVisible();
  });

  // G10's bug bash, finding 2: the notes said "0.8 trials", read as a count.
  test('the problem notes name BenchFlow 0.8', async ({ app, screen }) => {
    await app.open(TIMEOUTS);
    await expect(
      screen.getByText("Who should act, from each trial's problem. Trials from BenchFlow 0.8 record an error and its category but no owner."),
    ).toBeVisible();
    await expect(screen.getByText(/^Grouped by code and owner\. For trials from BenchFlow 0\.8 the code is the error's category /)).toBeVisible();
  });

  test('a job with no problem says so', async ({ app, screen }) => {
    await app.open('/outcomes?run=oracle-hello');
    await expect(screen.getByText('No trial recorded a problem.')).toBeVisible();
  });
});
