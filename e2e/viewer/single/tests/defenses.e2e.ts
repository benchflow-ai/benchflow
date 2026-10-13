// The defense line on a trial folder served alone (G11 review B, L1 and L5):
// the page boots straight into one trial, and a failed defense opens the line.
// The fixture's memory_limit row was edited to `failed` (e2e.single.config.ts).
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

describe('defenses on a trial served alone', () => {
  test('a failed defense opens the line without a tap', async ({ app, screen }) => {
    await app.open('/');
    const header = screen.getByRole('banner');
    await expect(header.getByRole('heading', { name: 'hello-split', level: 1 })).toBeVisible();
    await expect(header.getByText('defenses 14 enforced · 1 failed · 1 off · 1 not-supported')).toBeVisible();
    await expect(header.getByRole('row')).toHaveCount(17);
    await expect(header.getByRole('row').filter({ has: screen.getByRole('cell', 'memory_limit') }).getByRole('cell')).toHaveText([
      'memory_limit',
      'failed',
      "edited for the viewer's tests: this trial's memory limit did not apply",
    ]);
  });
});
