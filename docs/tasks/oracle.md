# Oracle

```bash
bench run tasks/my-task -a oracle --sandbox docker --jobs-dir jobs/my-task-oracle
```

`oracle/solve.sh` is the task's reference solution. `-a oracle` runs it instead of an agent, to prove the task is solvable. A correct task scores 1.0 on its oracle run before any model sees it.

`bench run` without `-a` runs the oracle too.

## How it runs

| | |
|---|---|
| Copied to | `/oracle` (an imported task's `solution/` goes to `/solution`) |
| Command | `bash /oracle/solve.sh`, so the script runs under bash whatever its `#!` line says |
| User | The task's `agent.user` when it names one, otherwise root |
| Directory | `sandbox.workdir` when set, otherwise the image's `WORKDIR` |
| Time limit | The agent's: `agent.timeout_sec`, or the run's own timeout |
| Output | `/logs/agent/oracle.txt` |
| Network | The task's network policy covers `solve.sh` while it runs; see [Network policies](./network-policies.md) |

`oracle/` is uploaded only for an oracle run (`--agent oracle`, and the equivalence battery of `bench check`). Every other agent, a root agent or a scripted user included, runs in a sandbox that holds no solution; a scripted user with `oracle_access` reads solve.sh on the host.

If a folder has both `oracle/` and `solution/`, `oracle/` wins.

## Settings

```yaml
oracle:
  env:
    API_BASE: http://target:8080
```

`oracle.env` sets variables for `solve.sh`. `oracle.timeout_sec` parses but has no effect; the oracle uses the agent's time limit.

## The empty control

```bash
bench run tasks/my-task -a nop --sandbox docker --jobs-dir jobs/my-task-nop
```

`-a nop` installs and runs nothing, so the verifier scores the untouched workspace. A sound verifier gives it 0.0. It needs no model or key, and takes about as long as the verifier.

Give each control its own `--jobs-dir`. `bench run` resumes the last job in its jobs folder, and separate folders keep the two results apart.

`bench results` and `bf.load_job` report oracle and `nop` runs as controls, apart from agent runs. `bench results --include-controls` shows them.

## When to ship an oracle

| Task use | Oracle |
|---|---|
| While you write the task | Always: it is the only proof the task can be solved |
| A published benchmark | Recommended. Leave it out only if you can't publish a reference solution. |
| `bench check --level publication-grade` | Required |

Without `oracle/solve.sh`, an oracle trial fails with `Oracle requires oracle/solve.sh or legacy solution/solve.sh`.

## Next steps

- [Verifier](./verifier.md): what the oracle's output is scored by.
- [Check a task](./check.md): the evidence levels that rerun the oracle.
- [Create a task](./create-a-task.md): the oracle and `nop` in the authoring loop.
