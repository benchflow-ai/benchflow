// A trial's page (`/?run=<job>/<trial>`): its header, trace, verifier
// output, and metrics, loaded from /api/rollout.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

const ORACLE = '/?run=oracle-hello%2Fhello__5233d73b';

describe('trial page', () => {
  test('the header names the task, its reward, and how it ran', async ({ app, screen }) => {
    await app.open(ORACLE);
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
    await expect(header.getByText('pass 1')).toBeVisible();
    await expect(header.getByText('hello__5233d73b · source: scripted')).toBeVisible();
    await expect(screen.getByRole('tab')).toHaveText(['Trace', 'Verifier', 'Metrics']);
  });

  test('the trace shows the prompt and the oracle run', async ({ app, screen }) => {
    await app.open(ORACLE);
    const trace = screen.getByRole('tabpanel', 'Trace');
    await expect(trace.getByRole('article')).toHaveCount(2);
    await expect(trace.getByRole('article').first()).toContainText('Create a file `/app/hello.txt`');
    await expect(trace.getByRole('article').last()).toContainText('"command": "oracle/solve.sh"');
  });

  test('the verifier tab shows the reward, the assessment, and the tests', async ({ app, screen }) => {
    await app.open(ORACLE);
    await screen.getByRole('tab', 'Verifier').tap();
    await expect(screen.getByRole('tab', 'Verifier')).toHaveAttribute('aria-selected', 'true');
    const verifier = screen.getByRole('tabpanel', 'Verifier');
    await expect(verifier.getByRole('heading', 'Reward')).toBeVisible();
    await expect(verifier.getByRole('heading', 'Assessment')).toBeVisible();
    await expect(verifier.getByRole('row').filter({ hasText: 'Assessment' })).toContainText('scored');
    await expect(verifier.getByRole('heading', 'Tests (CTRF)')).toBeVisible();
    await expect(verifier.getByRole('row').filter({ hasText: 'test_hello' }).getByRole('cell')).toHaveText(['test_hello', 'passed', '']);
  });

  test('the metrics tab shows the time of each phase', async ({ app, screen }) => {
    await app.open(ORACLE);
    await screen.getByRole('tab', 'Metrics').tap();
    const metrics = screen.getByRole('tabpanel', 'Metrics');
    await expect(metrics.getByRole('heading', 'Phase timing')).toBeVisible();
    // timing.json: environment setup 25.40 s, agent 2.47 s, verifier 6.60 s, total 34.57 s.
    await expect(metrics.getByText('environment setup')).toBeVisible();
    await expect(metrics.getByText('25.4 s')).toBeVisible();
    await expect(metrics.getByText('2.5 s')).toBeVisible();
    await expect(metrics.getByText('6.6 s')).toBeVisible();
    await expect(metrics.getByText('34.6 s')).toBeVisible();
  });

  test('an unknown run says so instead of an empty page', async ({ app, screen }) => {
    await app.open('/?run=nope%2Fmissing__0');
    await expect(screen.getByText('Run "nope/missing__0" is not among the 10 discovered runs.')).toBeVisible();
  });
});
