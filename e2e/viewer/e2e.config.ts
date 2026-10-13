// Browser tests for `bench view`, run with e2e (https://github.com/tester-army/e2e).
//
// The runner starts the `bench` named by $BENCH (default: the debug build at
// the repository root) on a free port, serving the stored jobs in ../jobs and
// the fixture jobs in fixtures/, and stops it when the run ends. Locator
// assertions carry every check. No model is configured, so an agent step
// fails at once with MODEL_UNAVAILABLE and nothing leaves the machine.
import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { fileURLToPath } from 'node:url';

// No usage data leaves the machine, whoever runs the suite.
process.env.E2E_TELEMETRY_DISABLED = '1';

const here = fileURLToPath(new URL('.', import.meta.url));
const bench = process.env.BENCH ?? fileURLToPath(new URL('../../target/debug/bench', import.meta.url));

export default {
  tests: 'tests/**/*.e2e.ts',
  targets: [
    {
      name: 'chromium',
      engine: web(),
      app: {
        url: 'http://127.0.0.1:0',
        command: {
          executable: bench,
          args: [
            'view',
            '../jobs/nop-hello',
            '../jobs/oracle-hello',
            'fixtures/fork-nop',
            'fixtures/timeouts',
            'fixtures/dt-f1-hello-split-oracle',
            'fixtures/dt-r2-split-optout-nop',
            '--port',
            '{port}',
            '--no-open',
          ],
          cwd: here,
          log: '.e2e/logs/bench-view.log',
        },
      },
    },
  ],
} satisfies E2EConfig;
