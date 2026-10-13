// A trial's defense record on its page (`/?run=<job>/<trial>`): one line in
// the header that counts the rows by state and opens to every row, with its
// state and why. G10's bug bash, finding 1: no tab showed any of it.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

// The F1 build's refusal of the cloud metadata endpoints, enforced.
const ENFORCED = '/?run=dt-f1-hello-split-oracle%2Fhello-split__927059e2';
// A task that opts out of it ([metadata] x-allow-cloud-metadata = true).
const OPTED_OUT = '/?run=dt-r2-split-optout-nop%2Fsplit-optout__486c9892';

describe('defenses', () => {
  test('the header counts the record by state and opens to every row', async ({ app, screen }) => {
    await app.open(ENFORCED);
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'hello-split', level: 1 })).toBeVisible();
    const summary = header.getByText('defenses 15 enforced · 1 off · 1 not-supported');
    await expect(summary).toBeVisible();
    const rows = header.getByRole('row');
    await expect(rows).toHaveCount(0);
    await summary.tap();
    await expect(rows).toHaveCount(17);
    await expect(rows.filter({ has: screen.getByRole('cell', 'cloud_metadata') }).getByRole('cell')).toHaveText([
      'cloud_metadata',
      'enforced',
      /^the uid firewall refuses TCP from every user, root included .* the rule counted 1 packet$/,
    ]);
    await expect(rows.filter({ has: screen.getByRole('cell', 'network_policy') }).getByRole('cell')).toHaveText([
      'network_policy',
      'enforced',
      'agent: open: everything but the cloud metadata endpoints (see cloud_metadata)',
    ]);
    await expect(rows.filter({ has: screen.getByRole('cell', 'process_limit') }).getByRole('cell')).toHaveText([
      'process_limit',
      'not-supported',
      'the daytona backend cannot cap the number of processes (BenchFlow asks for 4096)',
    ]);
  });

  test("a task's opt-out of the metadata refusal shows as off, with why", async ({ app, screen }) => {
    await app.open(OPTED_OUT);
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'split-optout', level: 1 })).toBeVisible();
    const summary = header.getByText('defenses 15 enforced · 2 off · 1 not-supported');
    await expect(summary).toBeVisible();
    await summary.tap();
    await expect(header.getByRole('row')).toHaveCount(18);
    await expect(header.getByRole('row').filter({ has: screen.getByRole('cell', 'cloud_metadata') }).getByRole('cell')).toHaveText([
      'cloud_metadata',
      'off',
      /^the task opts out \(\[metadata\] x-allow-cloud-metadata = true\), so the agent can reach the cloud metadata endpoints/,
    ]);
    await expect(header.getByRole('row').filter({ has: screen.getByRole('cell', 'network_policy') }).getByRole('cell').nth(2)).toHaveText(
      'agent: open: everything; the cloud metadata endpoints are not proven refused (see cloud_metadata)',
    );
  });

  test('a trial whose record has no rows shows no defenses line', async ({ app, screen }) => {
    // fixtures/timeouts: the image never built, so the record is empty.
    await app.open('/?run=timeouts%2Fbroken-image__d075e6d5');
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'broken-image', level: 1 })).toBeVisible();
    // Anchored on the word, so a bare "defenses" line with no counts fails too.
    await expect(header.getByText(/^defenses\b/)).toHaveCount(0);
  });

  // G11 review C, M1: this record, from before F1 on an open network, has
  // neither network row, so it would read as an all-clear. The two rows show
  // as not recorded with no note (round 2, C2-M1: the current engine also
  // writes such records, so no cause is named; round 3: the state says it).
  test('a record without the network rows shows them as not recorded', async ({ app, screen }) => {
    await app.open('/?run=nop-hello%2Fhello__1f84c0c3');
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
    const summary = header.getByText('defenses 14 enforced · 1 not-supported · 2 not recorded');
    await expect(summary).toBeVisible();
    await summary.tap();
    const rows = header.getByRole('row');
    await expect(rows).toHaveCount(17);
    for (const name of ['network_policy', 'cloud_metadata']) {
      await expect(rows.filter({ has: screen.getByRole('cell', name) }).getByRole('cell')).toHaveText([
        name,
        'not recorded',
        '',
      ]);
    }
  });

  // G11 review A, M1: a branch child's page shows the record of the sandbox
  // that graded it, from the child's own result.json (16 rows; the parent has 9).
  test("a branch child's page shows its own record", async ({ app, screen }) => {
    await app.open('/?run=fork-nop%2Fhello__ec981dd5&branch=8b9d7724a324483c9b5f767f1b2e7155%2Fn1');
    const header = screen.getByRole('banner');
    await expect(header.getByText(/branch child n1 of fork 8b9d7724/)).toBeVisible();
    const summary = header.getByText('defenses 15 enforced · 1 not-supported · 2 not recorded');
    await expect(summary).toBeVisible();
    await summary.tap();
    await expect(header.getByRole('row')).toHaveCount(18);
    await expect(header.getByRole('row').filter({ has: screen.getByRole('cell', 'kill_before_grading') }).getByRole('cell')).toHaveText([
      'kill_before_grading',
      'enforced',
      /processes/,
    ]);
  });
});
