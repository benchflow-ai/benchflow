# The viewer

`bench view` serves a job, several jobs, a folder of jobs, or one trial on `localhost` and opens it in your browser. It reads the job folders as they are on disk, written by this `bench` or by BenchFlow 0.8, and reads them again on every request, so a running job's new trials show up on reload.

| Page | What it shows |
|---|---|
| `/` | The run list (0.8's page), and each trial's trajectory, verifier output, and files |
| `/compare` | Tasks by job, each job's pass rate with its 95% Wilson interval, and paired differences over the tasks both jobs ran, with a 95% t-interval |
| `/outcomes` | How each job's trials ended (passed, failed, unscored, errored), the problems behind them by code and owner, and the median and 90th percentile of each phase's time |
| `/pareto` | Pass rate against cost per trial, against time, and against tokens per trial, with the jobs no other job beats on each |

Each page keeps its jobs and filters in the URL (`/pareto?runs=jobs/a,jobs/b&agent=codex`), so a link reproduces the view, and `/api/<page>` answers the same query as JSON. With two or more jobs, `bench view` opens on `/compare`.

## The Pareto view

One point per job, or per agent and model when a job ran several (the columns of `/compare`). The vertical axis is the pass rate: passed trials over all trials, so errored and unscored trials count as not passed, with its 95% Wilson interval drawn as a bar. There is one chart for each horizontal axis, never two scales on one chart:

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
- `crates/benchflow-viewer/tests/views.rs`: the router tests, over a Rust job beside a 0.8 job.
