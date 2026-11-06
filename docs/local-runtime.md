# The local runtime (`bench daemon`)

`bench daemon start` starts a runtime that keeps BenchFlow and its model proxy loaded between CLI calls. With `BENCHFLOW_DAEMON=auto` set, non-interactive `bench run` calls attach to it: they skip BenchFlow's start-up imports, and each trial's host model proxy starts in under a tenth of a second. It is off unless you start it, and a call that cannot attach runs exactly as it would without it.

```bash
bench daemon start                 # this user's runtime
export BENCHFLOW_DAEMON=auto       # calls attach to it
bench run ./my-task --agent claude-agent-acp --fresh
bench daemon status
bench daemon stop                  # waits for running calls; never interrupts one
```

## What stays the same

- Each call runs in a fresh process forked from the runtime, with the caller's environment, folder, umask, arguments, standard streams, ignored signals and signal mask. Task files, configuration and credentials are read by that process, as by a cold call. Results and logs are the same with and without the runtime.
- Each trial keeps its own model proxy process, master key, port, usage log and runtime folder.
- A forked model proxy starts from the same inputs as a cold one: the same environment (python-dotenv off, so neither reads a `.env` file), the same installed files, and an import path of its runtime folder and installed packages only. Nothing the proxy parent read from its own folders is carried into a trial's proxy.
- The runtime is this user's only: its socket is mode 0600 in a 0700 folder, and both sides check the other's uid.
- Ctrl-C, `SIGTERM`, `SIGHUP` and `SIGQUIT` reach the call; a call that ends by a signal ends the caller the same way.

## When a call runs cold instead

A call runs without the runtime, unchanged, when:

- it runs from a terminal (job control and prompts belong to the terminal);
- the runtime was started by another installation or source tree, or with other interpreter settings (`PYTHON*`, locale, time zone, `-X` options);
- its environment differs in a value BenchFlow reads when it loads;
- its process differs from the runtime's in groups, limits, CPU affinity, cgroup or priority;
- its confinement differs: any namespace, `no_new_privs`, seccomp, capabilities or the security-module label (AppArmor, SELinux, Smack). A field that cannot be read keeps the call cold.

`daemon.log` in the runtime's folder (`~/.cache/benchflow/daemon`, or `BENCHFLOW_DAEMON_DIR`) says why.

**Not the same:** LiteLLM downloads its model cost map from GitHub when it is imported. A forked proxy uses the copy its proxy parent downloaded when the runtime started; a cold proxy downloads its own. They differ only when that map changed in between.

**Not detected:** a caller that adds its own Landlock rules under a runtime that already runs with `no_new_privs`. Landlock cannot be read back; it needs `no_new_privs` or `CAP_SYS_ADMIN`, which the comparison sees in every other case.

## Prepared sandboxes (`--prepare N`)

`bench daemon start --prepare 1` also prepares the next identical call: after an attached one-task Docker run of a new job ends well, the runtime runs the same command again up to the point where its agent would start, with the sandbox made by the same steps as a cold start. A later call with exactly the same arguments, environment, folder and settings takes it and reaches its agent in under two seconds.

Before it is handed over, the prepared call checks that a cold call would have made the same sandbox: every file it read, path it probed and folder it listed, the `.env` BenchFlow reads, and the images, by repeating the task's own cached build and comparing every Compose service's image. Anything that changed, or a check that cannot be done, and the call runs fresh. An unused prepared sandbox is removed after `--prepare-ttl` seconds (300 by default), when a newer call is prepared instead, or when the runtime stops. While it waits, its job folder is skipped by "resume the latest job".

It only helps a harness that repeats the same command (for example with `--fresh` and a fixed jobs folder). A call whose arguments or environment differ from the last one gets the loaded runtime, but no prepared sandbox.
