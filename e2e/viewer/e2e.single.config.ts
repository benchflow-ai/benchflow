// Browser tests for `bench view` on one trial folder served alone (the page
// a trial gets without a job around it, as `bench view <trial>` serves it).
// Run with `npx e2e run --config e2e.single.config.ts`.
//
// It serves single/fixtures/failed-defense__927059e2, a copy of
// fixtures/dt-f1-hello-split-oracle/hello-split__927059e2 EDITED by hand: its
// memory_limit defense is set to `failed` (no real trial has a failed row),
// so the defense line's open-on-failure can be tested. It is served on its
// own so the main suite's jobs, and the counts its tests assert, stay as
// they are.
import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { fileURLToPath } from 'node:url';

// No usage data leaves the machine, whoever runs the suite.
process.env.E2E_TELEMETRY_DISABLED = '1';

const here = fileURLToPath(new URL('.', import.meta.url));
const bench = process.env.BENCH ?? fileURLToPath(new URL('../../target/debug/bench', import.meta.url));

export default {
  tests: 'single/tests/**/*.e2e.ts',
  targets: [
    {
      name: 'chromium',
      engine: web(),
      app: {
        url: 'http://127.0.0.1:0',
        command: {
          executable: bench,
          args: ['view', 'single/fixtures/failed-defense__927059e2', '--port', '{port}', '--no-open'],
          cwd: here,
          log: '.e2e/logs/bench-view-single.log',
        },
      },
    },
  ],
} satisfies E2EConfig;
