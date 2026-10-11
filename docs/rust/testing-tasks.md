# Testing tasks with `bench test`

A task makes two claims: a right answer passes, and a wrong one does not. `bench test` checks both, and then tries to break the grading itself, so a task author finds out before an agent does.

```bash
bench test                       # every task under this folder
bench test tasks/my-task         # one task
bench test --probes none         # skip the exploit probes
bench test --list-probes         # what each probe tries
```

## What runs

For each task, these trials run side by side on the same image:

| Trial | It must | Why |
|---|---|---|
| `oracle` | pass | The task's reference solution (`oracle/solve.sh`, or `solution/solve.sh` in the split layout) proves the task can be solved and that the verifier accepts a right answer. A task without one fails this check. |
| `nop` | fail | An empty run proves the verifier can say no. A task that passes when nothing was done is grading something already true. |
| `control:<id>` | fail, or stay at or below its declared reward | Each script in `controls/` is a known-bad answer the task author wrote. It runs in place of the agent, like the oracle. |
| `probe:<id>` | score no higher than `nop`, and let grading finish | Each probe attacks how tasks are graded, not the task itself, so every task can face all of them. |

Every trial runs exactly as an agent's would: the same non-root user, the same workdir, the same handoff and verifier. A probe that gets through has found a hole an agent could find too.

## The probes

| Probe | What it tries | The defense that should stop it |
|---|---|---|
| `forge-reward` | Writes a perfect `reward.txt`, `reward.json`, and `ctrf.json` into `/logs/verifier` | The runner clears `/logs/verifier` before grading, and keeps it root-only while the agent works |
| `background-writer` | Leaves a detached process that keeps forging reward files and rewriting the workdir | The runner kills every process of the agent user before grading, and confirms none is left |
| `harness-tamper` | Plants `conftest.py`, `pytest.ini`, `sitecustomize.py`, and a `.pth` file that make every test pass, and appends to every writable `conftest.py` | The runner snapshots test-runner files before the agent starts, then removes the ones the agent planted and restores the ones it edited |
| `path-shadow` | Puts fake `pytest`, `python3`, `bash`, and `uv` first on `PATH` that report success | Folders on the sandbox's `PATH` that the agent could write are locked to root before it starts (and reported as `locked_system_path`), the verifier runs with a `PATH` of root-owned folders only, and BenchFlow's own root commands (the kill, the handoff, file copies) run with a `PATH` of the standard system folders, as do the shells and wrappers that start every command as root (`docker exec`'s shell, Daytona's session wrapper and its switch to the agent user, Docker's keep-alive loop), so a folder on the image's `PATH` inside the workdir (a venv, say) is never searched as root; the command itself still gets the `PATH` it was given |
| `peek-hidden` | Looks for the oracle and the verifier during its run, copies them, and runs the oracle if it finds it | The oracle and the verifier are not in the sandbox until they are needed; folders the image ships at those paths are locked to root |
| `special-files` | Leaves FIFOs where outputs go, symlinks to `/etc/shadow` and `/proc/self/environ`, and a 100 GB sparse file | The runner removes FIFOs, sockets, devices, and symlinks that point outside the workdir from the workdir and from `/output` and `/outputs` before grading, so the verifier cannot hang or be pointed elsewhere; before uploading the verifier, it removes anything the agent left at the verifier's folder and any symlink the agent made on the way to it; outputs reached through a symlink the agent made are not saved |
| `process-flood` | Leaves 400 sleeping processes | The sandbox's process limit holds, and the kill before grading still clears them |
| `env-inject` | Exports `LD_PRELOAD`, `PYTHONPATH`, and `BASH_ENV` from shell profiles and a `.env` file | The verifier starts from the image's environment, not the agent's shell. BenchFlow's own root commands start without the image's loader and shell start-up variables (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `BASH_ENV`, `ENV`, ...), which an image could point into the workdir; on Daytona, so do the steps that run as root on the way to any command. A command that is not one of BenchFlow's own gets them as on the image: the loader variables once it runs a program, and, under bash, the `BASH_ENV` file sourced after the loader variables are back, so a change the file makes to one of them stays. A non-interactive shell never reads `ENV`, on the image either. The verifier drops them (see `clean_verifier_env`), so a task that sets up its tools through `BASH_ENV` should set `[verifier] env` or run the setup in the verifier script |

## Reading the report

Here `bench test` ran on three tasks with holes: `always-passes`, one of the repository's test fixtures, whose verifier checks something already true; `loose-hello`, a copy of the `hello` fixture whose verifier checks only the first word, so its `controls/near-miss.sh` (which writes `Hello world!`) passes; and `writable-bin` (not shown), whose holes the runner closes.

```console
$ bench test
Testing 3 tasks with 31 trials on docker · 4 at a time
  ✗ always-passes probe:forge-reward          1   6.5s
  ✗ always-passes nop                         1   6.6s
  ...

  ✗ always-passes
      ✓ oracle      1  the reference solution passes
      ✗ nop         1  doing nothing passes, so the verifier cannot tell a solution from the
                       starting state
        fix: make the verifier check what the instruction asks for, not something already true
      ✓ probes    8/8  8 exploits refused
      ! defenses not in force in every trial:
          memory_limit  off: the task sets no memory limit, so the sandbox may use all 64298 MB its
                        host has
  ✗ loose-hello
      ✓ oracle                 1  the reference solution passes
      ✓ nop                    0  doing nothing fails (reward 0)
      ✗ control:near-miss      1  the control scores 1, above what it should earn
        fix: tighten the verifier so this known-bad answer fails
      ✓ probes               8/8  8 exploits refused
      ! defenses not in force in every trial:
          memory_limit  off: the task sets no memory limit, so the sandbox may use all 64298 MB its
                        host has
  ...
✗ 1 of 3 tasks proved itself  bench view jobs/tests/2026-10-10__18-15-46
```

- A `✗` on `oracle` means a right answer fails: fix the oracle or the verifier.
- A `✗` on `nop` means doing nothing passes: the verifier checks something already true.
- A `✗` on a control means a known-bad answer scores above what it should earn: the verifier is too loose.
- `VULNERABLE` on a probe means the probe raised the reward above the empty run's. The fix line says what to change in the task, and which runner defense was meant to stop it. No probe gets through on the repository's fixtures today, because the runner closes the holes they test.
- A probe that makes grading hang until the verifier's timeout, or crash, also fails: the verifier should never be at the mercy of what the agent left behind.
- While the trials run, each row is marked by its own reward, so on `always-passes` the probe rows show `✗` at reward 1. The summary compares each probe with the empty run instead, and none scored above it.
- The `!` lines list defenses not in force in some trial: `failed` (the image broke the defense and the runner repaired it, such as `writable-bin`'s world-writable `/usr/local/bin`), `not supported here` (the backend cannot do it, such as a process limit on a backend without one), or `off` (the task turned it off or set nothing, such as no memory limit, or `[agent] user = "root"`, which leaves no separate agent user to kill).

Each trial's `result.json` records every defense and its state, and `custody.json` lists what the handoff removed or restored: special files, escaping symlinks, planted and edited test-runner files, and leftover Python modules in `/tmp`. After the handoff, the workdir and everything in it belong to root (or to the task's verifier user) until grading ends, as in BenchFlow 0.8, so programs that check who owns a folder work for the verifier: git refuses a repository another user owns ("detected dubious ownership"). `custody.json` records the owner as `workdir_owner`; in a task with stages, the agent user gets the workdir back after a graded stage. A file with more than one hard link keeps its owner unless the agent user owns it (the other link may be a file outside the workdir). Only the workdir folder itself must change owner: when it cannot (a verifier user the image lacks, a workdir on a read-only mount), the trial is not graded (`R315`, the task's); a path below it that cannot (folders nested past what the image's `find` can walk, as busybox's cannot past the system's path limit; a file `chown` refuses) is listed under `workdir_unchanged`, as is a change that ran past its time, or was ended by the sandbox, after the folder changed, and grading goes on: the agent's processes are gone by then, and failing the trial would let an agent that arranged it dodge its score. Another filesystem mounted inside the workdir is not walked (`-xdev`) and keeps its owner, unlisted.

An ungraded stage's agent keeps its processes into the next stage, so BenchFlow prepares that stage without acting as root where the agent can write: the workdir is not handed back (it never left the agent), the stage's files are copied in by the agent user, a mount under a folder the agent can write is copied in by the agent user too (and is then its own; the `read_only_mounts` defense says so), and one at a target an earlier mount or the image already made a folder the agent cannot write is refused as the task's (`R104`): neither BenchFlow nor the agent user can put it there safely while that agent runs (a fault of the task found after an earlier stage's agent worked leaves the trial unscored, not run again). While such an agent runs, the test-runner baseline is not taken again but extended with the stage's own test-runner files, read on this machine, so a `conftest.py` the earlier stage planted is still removed at the next handoff. Test-runner files a later stage's `setup.sh` writes while such an agent runs are not part of the baseline, and a shipped one it edits is put back as shipped at the next handoff. Copying a stage's files in takes room for two copies while it runs, and may take up to 15 minutes; a copy that fails under a path the task itself placed (its mounts) or a mount point is the task's (`R104`), otherwise, after an earlier stage, the agent's (`R205`). After a graded stage, and before the first stage, the baseline is taken again once the stage's files are in place. A clean agent run leaves `custody.json` nearly empty; an agent that tried something shows it there.

## The sandbox around every trial

Besides the defenses the probes test, every trial (in `bench test` and `bench run` alike) confines the sandbox and records what held in `result.json`'s `defenses`:

| Defense | What the runner does | Recorded as |
|---|---|---|
| `locked_system_dirs` | Before the agent starts, removes group and other write permission from everything under `/usr`, `/etc`, `/opt`, `/bin`, `/sbin`, `/lib*`, `/boot`, `/srv`, and `/root`, except the workdir and the agent's home | `enforced`, or `failed` with the paths when the image let others write them (they are locked either way), or when the agent user owns paths there (an image's own user named as the agent, say), which permissions do not stop it changing |
| `ld_so_preload` | Snapshots `/etc/ld.so.preload` before the agent starts, and restores it before grading if it changed | `enforced`; a file that cannot be restored refuses grading (`R309`) |
| `state_custody` | BenchFlow keeps its own state, including the baseline the handoff compares against and restores from, in `/var/lib/benchflow`, which only root may write; setup empties it. The handoff first checks that the folder is there and is not a link, and that nothing in it is anyone else's, a link, or writable by others; when that fails it stops before reading the baseline. Root never writes into the agent's home, where the agent could have planted a link: it stages the agent's login and settings files (or a loaded session) in a fresh folder of its own under `/var/lib/benchflow-place`, and the agent user copies them into place. Root first gives the agent user the folders on the way to those files, as BenchFlow 0.8 did, changing each from inside it after checking it is the folder it looked at, so it never follows a link; it stops at a link, a mount, or a `..` | `enforced`; `off` when the agent runs as root or the image gives it a way to become root (see `agent_user`): a change is then noted and grading goes on; not recorded for a reference solution run as root unless its check found something; a changed state folder refuses the trial (`R314`, the agent's; the defense is then `failed`); a file the agent user cannot write in its home refuses it before the agent starts (`R104`, the task's; `R205`, the agent's, when an earlier stage's agent changed its home) |
| `oom_order` | Starts every agent command with `oom_score_adj` 1000, so the kernel stops the agent's processes before the runner's and the verifier's when memory runs out | the values read back from the sandbox |
| `memory_limit` | Reads the memory limit the kernel enforces (cgroup `memory.max`) | `enforced` with the limit, `failed` when it is above the one asked for (beyond rounding up to whole GiB), `not-supported` when the task asks for one and none is visible, `off` when the task sets none |
| `clean_verifier_env` | Runs the verifier with loader variables (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`, `GCONV_PATH`, `BASH_ENV`, ...) unset, `PYTHONSAFEPATH=1`, and a `PATH` of root-owned folders, after checking that environment as the verifier's own user | `enforced`; an environment that cannot be made clean refuses grading (`R310`) |
| `runtime` | `--runtime runsc` (or `runtime = "runsc"` in `bench.toml`) starts sandboxes under gVisor on Docker daemons that have it; a daemon without it is refused before anything starts | `enforced`, only when a runtime was asked for |
| `read_only_mounts` | Uploads the task's mounted package files owned by root and read-only, after removing symlinks another user made on the way (and, for a later stage's mounts, anything the agent left at the target), then checks the folders above each target | `enforced`; `failed` when a folder above a target is not root's alone (a mount inside the workdir), since the agent user can then move the files aside; `off` when the agent runs as root |
| `root_home` | Nothing to apply. A task that declares no workdir works in its image's WORKDIR, as BenchFlow 0.8 runs it, and the agent user owns its workdir; when that is `/root`, the agent can write the start-up and config files root's programs read there (shell, pip, git, uv), the verifier's included. BenchFlow's own root-only state is kept in `/var/lib/benchflow`, outside any workdir, and a workdir of `/` or above that folder is refused (`R104`) | `off` when the workdir is `/root`; not recorded otherwise |
| `root_service` | On Daytona, the sandbox's own daemon (process 1, root) answers on ports 2280, 22220, 22222, and 33333 on every address without credentials and runs commands as root. Every Daytona trial guards them, whatever the task's network: the image gets iptables if it lacks it (the same layer the network policies use), and before the agent starts root reads the TCP ports the daemon listens on (its listening sockets in `/proc`), then installs the uid firewall's chain (below), which refuses the agent user's and other non-root users' TCP connections to those ports, and always to the four above, on loopback, the sandbox's own addresses, and IPv6 loopback. With an open network the chain refuses nothing else. The self-test tries each port at each of those addresses as the agent user, which must be refused everywhere, and then as root, which the chain does not cover. Root reaching an address where the agent user was refused shows the rule is what refuses; the note lists those addresses. Each later stage reinstalls the guard; the verifier's phase removes or replaces it. Where iptables cannot be added (an image with no package manager), the trial still runs and records why. An agent that runs as root turns it off. Docker has no such service | `enforced` on Daytona when the guard holds and, for every port the daemon listens on, root reached an address the agent user was refused at; `not-supported` with the reason when iptables cannot be added or tested, when the daemon's ports cannot be read, or when root gets no answer either; `off` when the agent runs as root; not recorded on Docker |
| `agent_user` | Creates the non-root user the agent runs as. Before the agent starts it checks, read-only, whether that user can become root: `sudo -n -l` as the user (it only lists), a `doas` rule without a password, account and sudo files it can write, the `disk` group, a container daemon socket it can use, setuid-root shells and interpreters, file capabilities that grant root's powers (with `getcap`, when the image has it), capabilities of its own, and (as root) uid-0 accounts or the agent user with an empty password. A way to root does not stop the trial, since a task may need `sudo`, but the firewall, the daemon guard, the locks, and the kill before grading then hold only against an agent that does not use it | `enforced` with the user and uid, plus checks the image had no tool for; `failed` with each way to root found; `off` when the task runs its agent as root |

`PYTHONSAFEPATH=1` applies to native `task.md` tasks. Tasks in the split layout and BenchFlow 0.8's `task.md` run their verifiers as BenchFlow 0.8 does, which leaves it to the image: such verifiers may run a script that imports a module next to it, which the flag breaks (SkillsBench's `adaptive-cruise-control` and Terminal-Bench 2's `cancel-async-tasks` do). Their records show `clean_verifier_env` as `off` with that reason. Any task can keep a loader variable, turn `PYTHONSAFEPATH` on (`"1"`), or turn it off (an empty value) in `[verifier] env`.

`bench doctor --confinement -s <backend>` starts one sandbox, applies all of this, and reads each setting back (with `--json` for machines): the system folders' permissions, the verifier's environment, `/etc/ld.so.preload`, an agent process's `oom_score_adj`, the memory limit, the runtime, the kernel's `ptrace_scope`, and whether the agent user can read root processes' `/proc/<pid>/environ`. Each is `enforced`, `not supported`, or `not set`, with the evidence.

## Network policies

A task's network is `"open"` (the default), `"none"`, a list of hosts and CIDR ranges to allow (`["github.com", "10.0.0.0/8"]`), or `{ block = [...] }`, everything except those. task.md sets it in `[sandbox] network`, and `[agent] network` and `[verifier] network` may narrow it for one phase; the split layout's `network_mode`, `allow_internet`, and `allowed_hosts` (in `[environment]`, `[agent]`, and `[verifier]`) and BenchFlow 0.8's frontmatter map onto the same values. Host names resolve once, on the machine running `bench`, when the trial's sandbox starts. Wildcards (`*.github.com`) are refused: a firewall that matches addresses cannot tell which names a wildcard covers.

| Policy | Docker, Podman | Daytona |
|---|---|---|
| `none` in every phase | `--network none` | Daytona blocks all traffic but IPv6; the uid firewall closes IPv6 and backs the IPv4 block up (loopback stays open), unless the sandbox's kernel has no IPv6 at all |
| `none` in one phase only | the uid firewall, for that phase | Daytona blocks all traffic but IPv6 during that phase; the uid firewall closes IPv6, as above |
| allowlist | the uid firewall: only the listed addresses; DNS refused, the names pinned in `/etc/hosts` | Daytona's own allow list (at most 10 ranges; IPv4) during that phase, plus the uid firewall |
| blocklist | the uid firewall: the listed addresses refused, the names pinned to addresses that route nowhere | the uid firewall only (Daytona has no block list) |

The uid firewall is an iptables chain that root installs before the agent starts. In the agent's phase it covers every packet of the agent user and every new connection of any other non-root user, so the agent cannot get around it by switching to another uid, while root keeps its own access (BenchFlow's commands, a key forwarder on loopback). Before grading, the chain is reinstalled for the verifier's policy, and then it covers every new connection, root's included. Docker gives the container `NET_ADMIN` for this, which the agent user, with no capabilities of its own, cannot use. An image without iptables gets it in an image layer (with apt-get, apk, dnf, or yum); an image where that fails, or that has IPv6 but no ip6tables, or a task whose agent runs as root (root could remove the chain), is refused before the phase runs (`R108`), never run unprotected; for a separate verifier that is after the agent ran, since its sandbox starts only then. On Daytona, whose block and allow list leave IPv6 open, every `none` and allowlist needs the chain, unless the sandbox's kernel has no IPv6 at all (a link-local address may route out later). Because Daytona's daemon also answers on IPv6 loopback, a chain that must hold there (any `none`, allowlist, or blocklist on Daytona) needs ip6tables wherever the kernel has IPv6, loopback included; the image layer installs ip6tables when only it is missing. A refusal Docker would not make is yours, not the task's, with the advice to run it on Docker: a root agent where only that gap needs the chain (every phase the sandbox hosts is `none`), an image that cannot get iptables there, or one with no ip6tables in a sandbox whose IPv6 does not route out (Docker checks ip6tables only for IPv6 that does). Refusals Docker makes too (a root agent with an allowlist, blocklist, or `none` in one phase; no iptables where Docker needs a chain; IPv6 that routes out with no ip6tables) are the task's. A root agent that the `sandbox-user = "root"` setting made, not the task's `[agent] user`, is always yours.

Before each phase, a self-test runs as that phase's own user: the first allowed host must answer, destinations outside the policy (1.1.1.1, 8.8.8.8, an IPv6 address) or the blocked ones must not, an unlisted name must not resolve, a blocked name must resolve only to its pinned address, and the agent user then tries to flush the rules and edit `/etc/hosts`, after which root checks that nothing changed. If any of this fails, the phase does not run (`R109`). The result is recorded as the `network_policy` defense (and `verifier_network_policy` when the verifier's network is not open), with the resolved addresses and each canary in its note:

```text
network_policy enforced: agent: only github.com (20.29.134.23); the daytona backend allows only 20.29.134.23/32; iptables firewall on every packet of `agent` and every new connection of other non-root users: only the listed addresses; DNS refused, 1 name pinned in /etc/hosts; IPv6 filtered too; self-test: reached the allowed 20.29.134.23:443; no DNS answer for example.org; could not reach 1.1.1.1:443, 8.8.8.8:53; the agent user could not change the rules
```

What a policy cannot promise:

- An allowlist allows addresses, not names. A host served from a shared CDN address (example.com is on Cloudflare) opens every other site served from that address, by asking for it by name over the same connection. List hosts that have their own addresses when that matters.
- A blocklist blocks the addresses its names resolved to when the sandbox started. The same service at another address, or a copy of its content elsewhere, is not blocked.
- Services beside the agent's container (Compose, `[[sandbox.services]]`) are on a private network; a `none` or allowlist enforced by the uid firewall refuses their addresses too, unless the task lists them.

`crates/benchflow-run/tests/fixtures/tasks/` has a fixture for each: `network-none`, `network-allow` (only github.com for the agent, none for the verifier), and `network-block` (everything but www.debian.org). Each oracle needs exactly what its policy permits, each control tries what it forbids, and each verifier deliberately rewards proof that the policy failed, so `bench test` shows whether it held.

## Residual risks

What the defenses above do not cover, as of the round 3 review (2026-10-10). A trial's record does not flag these unless the row says so.

- On Daytona, the in-sandbox root service (`root_service`, above) undoes the separation between the agent user and root, so every defense that depends on it (the handoff, the locks, the forwarder's key) holds only against an agent that does not use that service. Every Daytona trial now closes the daemon's ports to the agent user (`root_service` enforced); where the image cannot take iptables the gap stays, and the record says `not-supported`.
- The verifier and any hooks before grading are the task's own code, run as root on the agent's files. The runner removes planted test-runner files, special files, and loader variables, but a verifier or hook that runs a program from the workdir (`make`, `npm test`, a script the agent wrote) runs the agent's code as root. Processes a hook starts may still be running when grading starts.
- When the agent works in `/root` (`root_home: off`), the start-up and config files root's programs read there are the agent's to write.
- Giving the workdir to root before grading changes who owns the agent's files, not what they say: settings the agent wrote into the workdir (a repository's `.git/config`, say) apply to the verifier's programs as they would to any program run there.
- In a task with stages, what grading one stage leaves in the sandbox (files the verifier copied into the workdir or `/tmp`, processes it started) is still there when the agent works on the next stage. A later stage's readiness check runs as root in the agent's files, with the trusted `PATH` and loader variables unset, as hooks do.
- The kill before grading stops every process the agent user has at that moment. A service in the image that starts new processes for the agent user later (cron, say) is not stopped.
- `clean_verifier_env` covers loader variables and `PYTHONSAFEPATH`. Other variables in the image's environment that point programs at files (`PYTHONPATH`, `NODE_OPTIONS`, `PERL5LIB`) are not checked.
- Services beside the sandbox (Docker): the agent can change a service's state, and the kill before grading does not reach processes inside a service. On Podman, whether the services' network is cut off from the host has not been checked live.
- A separate verifier restores the outputs with the owners and modes they had in the agent's sandbox.
- `process_limit` and `runtime` are recorded from what the backend was asked for and what it reports, not read back from inside the sandbox.
- The Daytona daemon guard reads the daemon's ports once, before the agent starts, and covers TCP only: a port the daemon opens later, or a UDP one, stays open. The ports are read from sockets held by process 1 or another process running the same program; a helper the daemon starts under another program is not covered. A sandbox whose root cannot read `/proc` records the guard as not supported.
- The check for a way to root (`agent_user`) looks for the usual ways, not every one: a kernel bug, a root service in the image other than the Daytona daemon (a root web server that runs what it is sent, say), a cron job that runs a file the agent can write, or a capability outside the list it checks for is not found. Images without `getcap` are not checked for file capabilities; the note says so.
- `bench handoff` resumes the chosen session's transcript in your own CLI, with your permissions and in your current folder. A trial outside this project's jobs folder is refused unless you pass `--trust`; a trial someone copied into your jobs folder counts as yours. A trial from someone else is their agent's conversation: read it (`--dry-run --trust` shows what is copied) before resuming it.
- A trial `bench run --diff` reuses keeps the earlier job's scores as recorded; only `carried_from` says they were not graded again. Reuse compares the names of `--env` variables, not their values, and a job recorded before `--resume-sessions` was written to lock.json reads as run without it.
- An agent's session folders are saved without files whose names are known login files (`auth.json`, `.credentials.json`, and the credential scan's). A login kept under another name, or a key the agent wrote into a transcript, is saved with the session.

## Credential files

`bench test` scans each task's build folder and, before the empty run, its image for credential files: `.netrc`, `.git-credentials`, `.aws/credentials`, a `.docker/config.json` with registry logins, an `.npmrc` with an auth token, a `.pypirc` with a password, private SSH keys, and Google Cloud, Azure, kubeconfig, GitHub CLI, Hugging Face, and Terraform credential files. It names each file, its kind, and whether the agent user can read it, and the fix; it never prints a file's content. `bench check` reports the ones in the build folder (`P003`).

## Checking a wrong answer yourself

`bench run tasks/my-task --agent script:wrong.sh` runs your own script as the attempt, as the oracle runs: as the agent user, in the workdir, then graded. Use it to check that a wrong solution fails before you add it to `controls/`.

## What a probe cannot tell you

Probes attack the grading machinery. They do not know your task, so they cannot find a verifier that accepts a wrong answer for task-specific reasons (an assertion that is too loose, an expected file the agent can copy from elsewhere). Write those as controls: put a script in `controls/` that produces a plausible wrong answer, and `bench test` checks that it fails.
