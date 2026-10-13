// The views navigation's "runs" link on the run list page keeps the run
// list's own query (filter, grouping, sort, open groups) and drops only the
// trial it has open. G10's bug bash, finding 4: it was a plain "/".
import { describe, test } from '@e2e-dev/web';
import { expect } from 'e2e';

describe('runs link', () => {
  // Typed, not opened with the filter in the URL: the run list rewrites the
  // URL as it is filtered, and the link must follow it (G11 review B, L3).
  test('the runs link from a trial keeps the filter typed into the run list', async ({ app, screen, browser }) => {
    await app.open('/');
    await screen.getByPlaceholder('filter runs...').fill('hello');
    await expect(browser).toHaveURL('/?q=hello');
    await expect(screen.getByText('4 / 10 runs')).toBeVisible();
    await screen.getByRole('button', 'Open hello').first().tap();
    await expect(screen.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
    await expect(browser).toHaveURL(/\/\?q=hello&run=nop-hello%2Fhello__/);
    await screen.getByRole('navigation', 'views').getByRole('link', 'runs').tap();
    await expect(browser).toHaveURL('/?q=hello');
    await expect(screen.getByText('4 / 10 runs')).toBeVisible();
  });

  test("the runs link from a branch child drops the child, too", async ({ app, screen, browser }) => {
    await app.open('/?q=hello&run=fork-nop%2Fhello__ec981dd5&branch=8b9d7724a324483c9b5f767f1b2e7155%2Fn2');
    await expect(screen.getByRole('banner').getByText(/branch child n2 of fork 8b9d7724/)).toBeVisible();
    await screen.getByRole('navigation', 'views').getByRole('link', 'runs').tap();
    await expect(browser).toHaveURL('/?q=hello');
  });

  // On the run list itself the pill stays the one-click way to the whole list.
  test('on the filtered run list the runs link is the plain run list', async ({ app, screen, browser }) => {
    await app.open('/?q=hello');
    await expect(screen.getByText('4 / 10 runs')).toBeVisible();
    await screen.getByRole('navigation', 'views').getByRole('link', 'runs').tap();
    await expect(browser).toHaveURL('/');
    await expect(screen.getByText('10 runs')).toBeVisible();
  });

  test('with no filter the runs link is the plain run list', async ({ app, screen, browser }) => {
    await app.open('/?run=oracle-hello%2Fhello__5233d73b');
    await expect(screen.getByRole('heading', { name: 'hello', level: 1 })).toBeVisible();
    await screen.getByRole('navigation', 'views').getByRole('link', 'runs').tap();
    await expect(browser).toHaveURL('/');
    await expect(screen.getByText('10 runs')).toBeVisible();
  });
});
