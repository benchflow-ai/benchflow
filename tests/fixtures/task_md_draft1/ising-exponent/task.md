You are joining a statistical-physics group. We ran Monte Carlo simulations of the 3D Ising model on cubic lattices and want an independent estimate of the critical temperature and of the correlation-length exponent ν from our own data.

First, write a short analysis plan in `/work/plan.md`: which observables you will use, how you will locate the critical point, and how you will estimate uncertainties. When you submit the plan, the simulation data is released to you.

Your final deliverable is `/work/paper.pdf`, a short report a referee could check: method, results with uncertainties, and a plot of your finite-size scaling collapse. Do not rely on published values; we want what our data says.

```stage analysis
The simulation data is now in `/data/mc/`: one HDF5 file per lattice size L = 8, 12, 16, 24, 32, 48, each with energy and magnetization time series at 41 temperatures near the transition. Follow your plan, revise it where the data demands, and write `/work/paper.pdf`.
```

```notes
Reference values come from the task author's own high-statistics runs, stored in verifier/answers.toml. The data was generated for this task and is not published, so memorized literature values are close but not exact at the stated tolerance.
```

```toml task
name = "examples/ising-exponent"
title = "Critical point of the 3D Ising model from raw Monte Carlo data"
version = "0.4.0"
keywords = ["statistical-physics", "finite-size-scaling", "research"]

[about]
difficulty = "hard"
category = "physics-research"

[agent]
timeout = "4h"

[sandbox]
image = "ghcr.io/example/physics-sci:2026.09"
cpus = 8
memory = "32 GB"
network = { block = ["arxiv.org", "journals.aps.org"] }

[stages.analysis]
unlock = "on_submit"
submit = "/work/plan.md"
mounts = ["data/mc:/data/mc"]

[integrity]
canary = "task.md canary 7f3c2a90-5b1e-4d7a-9e2f-6a1b8c4d0e55"

[verifier]
judges = { llm = "claude-opus-5-5" }
```
