// The run list at `/`: every trial of the jobs served, grouped by task.
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

describe('run list', () => {
  test('counts every trial served by its outcome', async ({ app, screen }) => {
    await app.open('/');
    const catalog = screen.getByRole('main', 'run catalog');
    await expect(catalog.getByText('10 runs')).toBeVisible();
    await expect(catalog.getByText('pass 3')).toBeVisible();
    await expect(catalog.getByText('fail 4')).toBeVisible();
    // A discarded branch parent and the trial whose image failed are unscored too.
    await expect(catalog.getByText('unscored 3')).toBeVisible();
    await expect(catalog.getByText('completed but unscored 1')).toBeVisible();
    await expect(catalog.getByRole('button', /^▸ hello 3 runs/)).toBeVisible();
    await expect(catalog.getByRole('button', /^▸ hello-split 1 run/)).toBeVisible();
    await expect(catalog.getByRole('button', /^▸ two-stage 1 run/)).toBeVisible();
  });

  test('a group opens to its runs, and a run opens its trajectory', async ({ app, screen, browser }) => {
    await app.open('/');
    await screen.getByRole('button', /^▸ hello-split 1 run/).tap();
    await screen.getByRole('button', 'Open hello-split').tap();
    await expect(screen.getByRole('heading', { name: 'hello-split', level: 1 })).toBeVisible();
    // The open group stays in the URL beside the run, so a link reproduces the view.
    await expect(browser).toHaveURL('/?toggled=hello-split&run=dt-f1-hello-split-oracle%2Fhello-split__927059e2');
    await screen.getByRole('button', '← runs').tap();
    await expect(screen.getByRole('button', 'Open hello-split')).toBeVisible();
    await expect(browser).toHaveURL('/?toggled=hello-split');
  });

  test('a run opens from its group', async ({ app, screen, browser }) => {
    await app.open('/');
    await screen.getByRole('button', /^▸ build-timeout /).tap();
    await screen.getByRole('button', 'Open build-timeout').tap();
    await expect(screen.getByRole('heading', { name: 'build-timeout', level: 1 })).toBeVisible();
    await expect(browser).toHaveURL(/[?&]run=timeouts%2Fbuild-timeout__965f9873$/);
    await expect(screen.getByRole('banner').getByText('pass 1')).toBeVisible();
  });

  test('the filter keeps matching runs and lives in the URL', async ({ app, screen, browser }) => {
    await app.open('/');
    await screen.getByRole('searchbox', 'Filter runs').fill('split');
    await expect(screen.getByText('2 / 10 runs')).toBeVisible();
    await expect(screen.getByRole('button', 'Open hello-split')).toBeVisible();
    await expect(screen.getByRole('button', 'Open split-optout')).toBeVisible();
    await expect(browser).toHaveURL('/?q=split');
  });
});
