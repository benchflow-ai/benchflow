# The viewer

`bench view` serves a job, several jobs, a folder of jobs, or one trial on `localhost` and opens it in your browser. It reads the job folders as they are on disk, written by this `bench` or by BenchFlow 0.8, and reads them again on every request, so a running job's new trials show up on reload.

| Page | What it shows |
|---|---|
| `/` | The run list (0.8's page), and each trial's trajectory, verifier output, files, and defense record |
| `/compare` | Tasks by job, each job's pass rate with its 95% Wilson interval, and paired differences over the tasks both jobs ran, with a 95% t-interval (none from fewer than 3 shared tasks or when every task differs by the same amount, and never past the range rewards allow, as in `bench results`). A job, or an agent and model in it, with trials stopped or planned but never run (a drained run) shows how many never ran by its name and its rate as partial, with no interval, as `bench results` does |
| `/outcomes` | How each job's trials ended (passed, failed, unscored, errored), how many it planned that never ran, the problems behind them by code and owner, and the median and 90th percentile of each phase's time |
| `/pareto` | Pass rate against cost per trial, against time, and against tokens per trial, with the jobs no other job beats on each |

Each page keeps its jobs and filters in the URL (`/pareto?runs=jobs/a,jobs/b&agent=codex`; a job is named as from the folder `bench view` was started in, or by its name under the folder served, so `bench view jobs` takes `runs=a,b` too), so a link reproduces the view, and `/api/<page>` answers the same query as JSON. With two or more jobs, `bench view` opens on `/compare`. On the run list page, with a trial open, the navigation's "runs" link keeps the run list's own filter, grouping and sort, so it returns to the list the trial was opened from; with no trial open it is the whole list. A job whose only trials are `bench branch` parents discarded at their checkpoint has nothing on `/compare`, `/outcomes` or `/pareto`, which leave those parents out as `bench results` does; when the URL names such a job, the page says so and links to `/branches`, where its forks are.

## A trial's defense record

A trial run by this `bench` records each defense the engine applied around the agent, in `result.json`'s `defenses`: the agent user, the network policy and the refusal of the cloud metadata endpoints (`cloud_metadata`), the verifier's clean environment, the limits, and the others. A line in the trial's header, under its stats, counts the rows by state (`enforced`, `failed`, `off`, `not-supported`) and opens to every row with its state and the engine's note on why; it starts open when a defense failed. A row is shown as written, so a defense or a state this viewer does not know still shows. A record can lack the `network_policy` or `cloud_metadata` row: one written before the cloud metadata refusal, a trial that stopped before its agent's network was set up, a trial whose agent network step failed (which records only the failed row), or a regrade, whose record covers only the verifier. The line then shows each missing row as `not recorded` and counts it, so such a record does not read as an all-clear. A `bench branch` child's page shows the record of the sandbox that graded it, from the child's own `result.json`. A trial from BenchFlow 0.8, or one whose record is empty, has no line.

## The Pareto view

One point per job, or per agent and model when a job ran several (the columns of `/compare`). The vertical axis is the pass rate: passed trials over all trials, so errored and unscored trials count as not passed, with its 95% Wilson interval drawn as a bar; a point with trials stopped or never run is marked partial and has no bar, as on `/compare`. There is one chart for each horizontal axis, never two scales on one chart:

| Axis | Value per point | Read from |
|---|---|---|
| Cost per trial | the mean over trials that recorded a cost | `agent_result.cost_usd` in result.json |
| Time | the median whole-trial time | `total` in timing.json (or the timing in result.json) |
| Tokens per trial | the mean input plus output tokens over trials that recorded usage | `agent_result.total_tokens`, when `usage_source` is not `unavailable` |

A point is on an axis's frontier when no other point has a pass rate at least as high for no more of that axis, with one of the two strictly better; ties stay on the frontier. Frontier points are filled and joined by a dashed staircase; the others are hollow, and every point is labeled with its letter in the table below the charts, which lists each value with the number of trials it came from. A chart whose axis no trial recorded says so instead of drawing nothing, and points without the value are named under the chart. When the jobs shown did not all run the same tasks, the page warns that their pass rates do not compare directly. The task and agent filters choose trials; the status filter does not apply, since a pass rate needs every outcome.

The viewer never estimates a cost from tokens: BenchFlow keeps no price table, so a job with no recorded cost has no point on the cost chart. A run through BenchFlow's forwarder records the cost the provider billed, when its answers say (the Vercel AI Gateway and OpenRouter do), never the agent's own estimate; `agent_result.cost_source` says which (`provider` or `agent`) and `cost_note` why a cost is missing ([harnesses.md](harnesses.md#token-counts-and-cost)).

## Where the code is

- `crates/benchflow-viewer/src/server.rs`: the routes, host checks, and streaming.
- `crates/benchflow-viewer/src/views.rs`: what `/compare`, `/outcomes`, and `/pareto` compute.
- `crates/benchflow-viewer/assets/views.js` and `views.css`: how those three pages draw.
- `crates/benchflow-viewer/assets/additions.js`: what the run page adds to 0.8's: the defense record and the "runs" link's filter. 0.8's own scripts are embedded unchanged.
- `crates/benchflow-viewer/tests/views.rs`: the router tests, over a Rust job beside a 0.8 job.

The browser tests in `e2e/viewer/` start `bench view` on the stored jobs in `e2e/jobs/` and the fixture jobs in `e2e/viewer/fixtures/`, and check every page in Chromium with [e2e](https://github.com/tester-army/e2e), a test runner built on Playwright. To run them, build `bench` (`cargo build -p benchflow-cli`), then run `npm ci` and `npm test` in `e2e/viewer` (`BENCH=<path to bench>` tests another build). Every step is a locator, so no model or key is needed; the `rust` workflow runs the same suite (`npm run test:ci`). A second config, `e2e.single.config.ts`, serves one trial folder alone, from a fixture edited to hold a failed defense; `npm test` and `npm run test:ci` run it after the main suite. The scripts and the configs turn e2e's telemetry off.
