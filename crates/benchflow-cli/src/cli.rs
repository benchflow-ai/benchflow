//! The command line, as clap sees it.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

const HELP: &str = "\
{before-help}{name} {version}
Run AI agents on tasks in sandboxes, and score them.

{usage-heading} {usage}

Start here:
  new      Create a task that already passes `bench test`
  check    Validate tasks without building or running anything
  test     Prove tasks: the oracle passes, doing nothing fails, exploits are refused
  run      Run an agent on tasks and score them
  results  Summarize jobs; with two or more, compare them
  view     Open a job in your browser

More:
  shell    Open a shell in a task's sandbox, then grade what you left
  handoff  Resume a finished trial's agent session in its CLI, here
  regrade  Score a finished job again with each task's current verifier
  doctor   Check this machine: sandboxes, agent logins, disk
  cache    List what BenchFlow keeps between runs; `cache clean` removes it
  agents   List agents and whether each can run here
  import   Convert a task.toml (split layout) or 0.8 task.md package to task.md
  export   Write a split-layout copy of a task, for harnesses that read task.toml
  explain  Explain an error code
  schema   Print the JSON Schema of a task, job file, result, or lock

Options:
{options}

Run `bench` with no command to see what is in this folder.
Docs: https://github.com/benchflow-ai/benchflow/tree/bench-rs/docs/rust
";

#[derive(Debug, Parser)]
#[command(
    name = "bench",
    version,
    help_template = HELP,
    disable_help_subcommand = true,
    arg_required_else_help = false,
    max_term_width = 100
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub global: Global,
}

#[derive(Debug, Args, Clone)]
pub struct Global {
    /// Run as if bench started in DIR
    // Its own id: a subcommand's `--dir` (bench new) is also named `dir`,
    // and clap would fill both from either flag.
    #[arg(short = 'C', id = "chdir", value_name = "DIR", global = true)]
    pub dir: Option<PathBuf>,

    /// Print JSON instead of text (stable fields; see each command's docs)
    #[arg(long, global = true)]
    pub json: bool,

    /// Print only results and errors
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Print more detail; repeat for debug logs
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// When to use color
    #[arg(long, value_name = "WHEN", default_value = "auto", global = true, value_parser = ["auto", "always", "never"])]
    pub color: String,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// Create a task that already passes `bench test`
    #[command(after_help = "\
Examples:
  bench new fix-the-parser                 # tasks/fix-the-parser/ (or ./fix-the-parser/ outside a tasks folder)
  bench new acme/dedup-revenue -t python   # a pytest verifier and a rubric with points
  bench new --list                         # every template")]
    New(NewArgs),

    /// Validate tasks without building or running anything
    #[command(after_help = "\
Checks the task document, its config, the rubric, and the package files, and
points at the exact line and column of every problem. Tasks in any format
BenchFlow reads are checked: task.md, 0.8's task.md, and the split layout
(task.toml), where keys other runners would silently ignore are flagged.

Examples:
  bench check                  # every task under this folder (or bench.toml's [tasks] paths)
  bench check tasks/my-task    # one task
  bench check --fix            # apply the safe fixes, such as renaming a misspelled key
  bench check -w tasks/my-task # check again on every save
  bench check tasks/my-task --review          # and a model's opinion of the task's quality
  bench check tasks/my-task --review --save   # ... saved to tasks/my-task/review.json

--review sends the task's files (size-capped, secrets redacted, .env and key
files never) to a model, by default vercel/deepseek/deepseek-v4-flash-0731 on
the Vercel AI Gateway. Its findings are shown after the checker's, marked as
its opinion, and never change the exit code.")]
    Check(CheckArgs),

    /// Prove tasks: the oracle passes, doing nothing fails, exploits are refused
    #[command(after_help = "\
For each task, runs side by side on one image:
  the oracle         must pass (the verifier can say yes)
  an empty run       must fail (the verifier can say no)
  each control       must fail (tasks' own known-bad answers, from controls/)
  each exploit probe must score no higher than the empty run

Examples:
  bench test                         # every task here, on the default sandbox
  bench test tasks/x -s daytona      # one task, in Daytona's cloud sandboxes
  bench test --probes none           # just the oracle, the empty run, and controls
  bench test --probes forge-reward,path-shadow")]
    Test(TestArgs),

    /// Run an agent on tasks and score them
    #[command(after_help = "\
Targets are task folders, folders of tasks, or GitHub repositories:
  bench run tasks/                                  # local tasks
  bench run benchflow-ai/skillsbench/tasks@main     # a folder of a repository, fetched sparsely
  bench run https://github.com/org/repo/tree/v1/tasks

Or a dataset from a registry, every task pinned to a git commit:
  bench run -d terminal-bench@2.0                   # BenchFlow's registry, then the public dataset registry
  bench run -d mine@1.0 --registry registry.json    # a registry file or URL
  bench run -d datasets/mine/dataset.toml           # a local dataset

Settings come from flags, then job files (-c, later files win), then
bench.toml's profile and [run]. `--print-config` shows the result.

The agent defaults to the oracle (the task's reference solution), so a bare
`bench run tasks/x` checks a task with no model and no key.

Rerunning the same command resumes its job: finished trials are kept
(errored ones run again; `--rerun failed,unscored` reruns those too).
`--fresh` starts a new job. Every job folder has a lock.json recording
exactly what ran.

`--diff <job>` matches each planned trial against an earlier job's: reuse
(same task, agent, model, effort, and settings; copied in), regrade (only the
tests or verifier changed, and outputs were saved; graded again without the
agent), or rerun, with the reason. With --dry-run it only prints the plan.

Ctrl-C drains the run: no new trial starts and running ones finish. A second
Ctrl-C stops them (their sandboxes are removed; they are left out of the pass
rate and listed apart), and a third quits at once.

Examples:
  bench run tasks/ -a claude -m claude-sonnet-4-6 -j 8
  bench run tasks/ -a codex -k 3 --fail-under 0.6      # 3 trials per task; exit 1 below 60%
  bench run -c base.toml -c claude.toml                # job files
  bench run -c https://github.com/org/evals/blob/main/nightly.yaml   # a job file from a URL
  bench run tasks/ --dry-run                           # print the plan and run nothing
  bench run tasks/ -a codex --diff jobs/last --dry-run # what an earlier job can supply")]
    Run(RunArgs),

    /// Summarize jobs; with two or more, compare them
    #[command(after_help = "\
Examples:
  bench results                         # every job under ./jobs
  bench results jobs/2026-10-10__09-12-03
  bench results jobs/a jobs/b           # paired comparison over the tasks both ran
  bench results --analyze               # a model explains why the latest job's trials failed
  bench results jobs/x --analyze --max-trials 30

--analyze sends each failed, errored, or unscored trial's record, instruction,
trajectory tail, verifier output, and agent logs (capped, secrets redacted) to
a model, by default vercel/deepseek/deepseek-v4-flash-0731, and saves what it
says to analysis.json in the job folder. What it says is its opinion; the
problem BenchFlow recorded is shown next to it.")]
    Results(ResultsArgs),

    /// Open a job in your browser
    #[command(after_help = "\
Pages:
  /           the run list, and each trial's trajectory, verifier output, and files
  /compare    tasks by job, pass rates with 95% intervals, paired differences
  /outcomes   how trials ended, problems by code and owner, time per phase
  /pareto     pass rate against cost, time, and tokens per trial, with each frontier
  /branches   each fork of a `bench branch` job: value, checkpoint, parent, and
              each child's label, status, reward, and advantage
Each keeps its jobs and filters in the URL; /api/<page> answers the same as JSON.

Examples:
  bench view                          # the latest job
  bench view jobs/a jobs/b jobs/c     # opens on /compare; /pareto is one click away")]
    View(ViewArgs),

    /// Score a finished job again with each task's current verifier
    #[command(after_help = "\
Grades each trial again from the outputs it saved (outputs.tar or
outputs/), in a fresh sandbox from its task's image, with the task's
verifier as it is now. The agent does not run. Results go to a new job
folder next to the original, <job>-regrade-<n>, and the old and new reward
of every trial are printed side by side.

Trials that saved no outputs cannot be regraded; run them with
--save-outputs (tasks that declare outputs or use a separate verifier
save them anyway).

Examples:
  bench regrade jobs/2026-10-10__09-12-03
  bench regrade jobs/2026-10-10__09-12-03 -s daytona -j 16")]
    Regrade(RegradeArgs),

    /// Fork a rollout from a checkpoint into children, each graded
    #[command(after_help = "\
Runs the task to the fork point, stores the sandbox's files as a checkpoint
(Docker: an image; Daytona: a snapshot), and runs each child in a sandbox of
its own made from it. Each child is graded by the task's verifier; the
fork's value is the mean of the children's rewards. The checkpoint is
deleted afterwards unless --retain-checkpoints. The parent is discarded
after its fork, or with --parent continue goes on in its own sandbox while
the children run, and is graded too (its reward is not in the fork's
value).

Fork points: `start` (after setup, before the agent), `prompt:N` (after the
parent's N-th prompt, for model agents), and `stage:<name>` (after a graded
stage of a task with stages; children run the stages after it). A child
sends its own prompt (at `stage:<name>`, in place of the next stage's), or
without one the task's instruction again (the next stage's); it starts a
fresh session in the parent's files, or with --resume-session continues the
parent's conversation through `session/load` (ACP agents that offer it).

Records (BenchFlow 0.8's): tree.json and checkpoints.json in the trial
folder, each child under branches/<fork>/children/n<k>/, and a `branches`
block in the trial's result.json.

Examples:
  bench branch tasks/x --children 3                    # the oracle, 3 children
  bench branch tasks/x -a builtin -m mimo/mimo-v2.6-flash -s daytona \\
      --child label=base --child \"label=hint,prompt=@instruction Start with the tests.\"
  bench branch tasks/x -a opencode -m vercel/deepseek/deepseek-v4-flash-0731 \\
      --at prompt:1 --resume-session --child \"label=check,prompt=Check your work.\"
  bench branches jobs/2026-10-10__20-00-00             # each fork's value and children")]
    Branch(BranchArgs),

    /// Show a branch job's forks: value, children, rewards
    Branches(BranchesArgs),

    /// Open a shell in a task's sandbox, then grade what you left
    #[command(after_help = "\
Starts the task's sandbox as an agent would see it, as the agent user, in the
workdir. Exit the shell to grade the sandbox as it is, with the same handoff
and verifier a real run uses.

Examples:
  bench shell tasks/x              # try the task by hand, then see its score
  bench shell tasks/x --as root    # debug the image as root
  bench shell tasks/x --no-grade")]
    Shell(ShellArgs),

    /// Resume a finished trial's agent session in its CLI, here
    #[command(after_help = "\
Copies the session the trial saved (agent/sessions/, recorded in result.json
as agent_session) to where the agent's own CLI on this machine looks for it,
under the session's own id, then starts the CLI resuming it in this folder:
  Claude Code  ~/.claude/projects/<this folder>/<id>.jsonl, then claude --resume <id>
  Codex        ~/.codex/sessions/..., then codex resume <id>
($CLAUDE_CONFIG_DIR and $CODEX_HOME are honored.) The CLI uses your own login
and model from here on. A local session with the same id and other content is
never overwritten unless you pass --replace.

A trial outside this project's jobs folder (bench.toml's jobs-dir, else
./jobs) is refused unless you pass --trust: its session becomes your local
agent's context, and the agent runs with your permissions on this machine.

Examples:
  bench handoff jobs/2026-10-10__09-12-03/hello__84c1190c
  bench handoff jobs/x/hello__84c1190c --dry-run      # what it would copy, and the command
  bench handoff jobs/x/hello__84c1190c -- --version   # extra arguments for the CLI")]
    Handoff(HandoffArgs),

    /// Check this machine: sandboxes, agent logins, disk
    Doctor(DoctorArgs),

    /// List what BenchFlow keeps between runs; `cache clean` removes it
    #[command(after_help = "\
Lists the images BenchFlow built (on Docker or Podman, with their last use),
the task sources and dataset checkouts it fetched, registry indexes, and logs
in its cache folder (~/.cache/benchflow, or BENCHFLOW_CACHE_DIR), with sizes.

`bench cache clean` prints what it would remove; add --remove to remove it.
Only what BenchFlow made is touched: images carry its owner label (and are
removed by tag, never forced, so an image a container uses stays), and files
in the cache folder that this bench did not write are left alone. Images used
in the last 30 minutes are kept, since a run may still need them.

Examples:
  bench cache                                   # everything, with sizes and last use
  bench cache clean                             # what would be removed
  bench cache clean --only images --remove      # remove BenchFlow's images
  bench cache clean --older-than 30d --remove   # remove what was not used for 30 days")]
    Cache(CacheArgs),

    /// List agents and whether each can run here
    Agents(AgentsArgs),

    /// Convert a task.toml (split layout), 0.8 task.md, or Terminal-Bench v1 task to task.md
    #[command(after_help = "\
A Terminal-Bench v1 task (task.yaml, run-tests.sh, solution.sh,
docker-compose.yaml) is converted to the split layout, then imported: a compose file with one service folds into the Dockerfile,
run-tests.sh becomes the verifier with a reward from its exit code, and other
files go into the sandbox's build context. A task with solution.yaml is
refused; rewrite it as solution.sh.

Examples:
  bench import tasks/my-task                    # one task, written to tasks/my-task-taskmd
  bench import terminal-bench/original-tasks tb # every Terminal-Bench v1 task, into tb/
  bench import original-tasks tb --cpus 2 --memory-mb 4096")]
    Import(ImportArgs),

    /// Write a task in the split layout, or a job's traces as a Hugging Face dataset or OpenTelemetry
    #[command(after_help = "\
A task folder is written in the split layout (task.toml), for harnesses that
read it. A job folder is written as traces:

  --format hf    a Hugging Face dataset folder: data/train.jsonl, one row per
                 agent step (the conversation up to it), and a README.md
                 dataset card; load it with datasets.load_dataset(<folder>)
  --format otel  OpenTelemetry: one OTLP JSON trace per trial, one per line

Each trial's trajectory is its ATIF document (trainer/atif.json), or one
rebuilt from trajectory/acp_trajectory.jsonl; jobs from BenchFlow 0.8 read the
same way.

Examples:
  bench export tasks/my-task                     # tasks/my-task-split
  bench export jobs/2026-10-10__09-12-03         # jobs/2026-10-10__09-12-03-hf
  bench export jobs/2026-10-10__09-12-03 --sharegpt --episodes last
  bench export jobs/2026-10-10__09-12-03 --format otel traces.jsonl")]
    Export(ExportArgs),

    /// Explain an error code
    Explain(ExplainArgs),

    /// Print the JSON Schema of a task, job file, result, or lock
    #[command(after_help = "\
Kinds:
  task    task.md's config block (the ```toml task block)
  job     a job file for `bench run -c`
  result  a trial's result.json
  lock    a job's lock.json

Examples:
  bench schema lock > lock.schema.json")]
    Schema(SchemaArgs),

    /// Print a shell completion script
    #[command(hide = true)]
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[derive(Debug, Args)]
pub struct NewArgs {
    /// The task's name (`my-task`, `org/my-task`), or a path whose folder exists (`tasks/my-task`)
    #[arg(required_unless_present = "list")]
    pub name: Option<String>,

    /// The template to start from
    #[arg(short, long, default_value = "basic")]
    pub template: String,

    /// Where to create it (default: ./tasks/ when it exists, else .)
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,

    /// List the templates
    #[arg(long)]
    pub list: bool,
}

#[derive(Debug, Args)]
pub struct CheckArgs {
    /// Task folders, or folders of tasks
    pub paths: Vec<PathBuf>,

    /// Apply the fixes that are safe to apply
    #[arg(long)]
    pub fix: bool,

    /// Treat warnings as errors
    #[arg(long)]
    pub strict: bool,

    /// Check only the task file, not the folders a run needs (for authoring fixtures)
    #[arg(long)]
    pub schema: bool,

    /// Check again whenever a file in the tasks changes
    #[arg(short, long, conflicts_with_all = ["fix", "review"])]
    pub watch: bool,

    /// Also have a model review each task's quality (its opinion, shown after the checker's)
    #[arg(long)]
    pub review: bool,

    /// The model for --review, as <provider>/<model> (default: vercel/deepseek/deepseek-v4-flash-0731)
    #[arg(long, value_name = "MODEL", requires = "review")]
    pub model: Option<String>,

    /// Save each review to review.json in its task folder
    #[arg(long, requires = "review")]
    pub save: bool,
}

#[derive(Debug, Args, Clone)]
pub struct SandboxArgs {
    /// Where trials run: docker, podman, or daytona (default: bench.toml, then the first one available)
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Option<String>,

    /// Trials at a time
    #[arg(short = 'j', long, value_name = "N")]
    pub concurrency: Option<usize>,

    /// Keep sandboxes after their trials, for inspection
    #[arg(long, value_enum, default_value = "never")]
    pub keep: KeepArg,

    /// Where job folders go (default: bench.toml's, then ./jobs)
    #[arg(long, value_name = "DIR")]
    pub jobs_dir: Option<PathBuf>,

    /// Apply a [profile.<NAME>] from bench.toml
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Start sandboxes under this container runtime, such as runsc (gVisor), on Docker daemons that have it
    #[arg(long, value_name = "RUNTIME")]
    pub runtime: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum KeepArg {
    Never,
    OnFailure,
    Always,
}

#[derive(Debug, Args)]
pub struct TestArgs {
    /// Task folders, or folders of tasks
    pub paths: Vec<PathBuf>,

    #[command(flatten)]
    pub sandbox: SandboxArgs,

    /// Which exploit probes to run: all, none, or a comma-separated list
    #[arg(long, default_value = "all", value_name = "WHICH")]
    pub probes: String,

    /// List the exploit probes and what each tests
    #[arg(long)]
    pub list_probes: bool,
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Task folders, folders of tasks, or GitHub repositories (org/repo[/path][@ref])
    pub targets: Vec<String>,

    /// The agent: claude, codex, gemini, opencode, builtin, terminus-2, a project agent, oracle, nop, or script:<file> (your own script, run like the oracle)
    #[arg(short, long)]
    pub agent: Option<String>,

    /// The model, in the agent's own naming, or `<provider>/<model>` to run on a
    /// provider's key (vercel, openai, anthropic, openrouter, zai, deepseek, mimo)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Reasoning effort, when the agent exposes one (low, medium, high, max, ...)
    #[arg(short = 'e', long, value_name = "LEVEL")]
    pub effort: Option<String>,

    /// How to drive the agent: acp or headless (default: the agent's own; `bench agents` lists each agent's)
    #[arg(long, value_name = "HARNESS", value_parser = ["acp", "headless"])]
    pub harness: Option<String>,

    /// For `-a external`: the service that decides each step (an HTTP backend protocol)
    #[arg(long, value_name = "URL")]
    pub backend: Option<String>,

    /// An agent option (repeatable), checked before any sandbox starts; `bench agents <name>` lists an agent's
    #[arg(
        long = "ak",
        visible_alias = "agent-option",
        alias = "agent-kwarg",
        value_name = "NAME=VALUE"
    )]
    pub agent_options: Vec<String>,

    /// On each graded stage after the first, resume the agent's own session instead of starting a fresh one (agents that can)
    #[arg(long)]
    pub resume_sessions: bool,

    /// A skill to install for the agent (repeatable): a folder with a SKILL.md, or a folder of them; it replaces a task skill of the same name
    #[arg(long = "skill", value_name = "FOLDER")]
    pub skills: Vec<PathBuf>,

    /// Leave out the task's own skills (BenchFlow 0.8's no-skill mode); --skill folders are still installed
    #[arg(long)]
    pub no_task_skills: bool,

    /// Start the agent on this session file and resume it with the first prompt (claude: <id>.jsonl; codex: rollout-*.jsonl)
    #[arg(long, value_name = "FILE")]
    pub load_trajectory: Option<PathBuf>,

    #[command(flatten)]
    pub sandbox: SandboxArgs,

    /// Trials per task
    #[arg(short = 'k', long, value_name = "N")]
    pub trials: Option<usize>,

    /// Job name (an existing job is resumed)
    #[arg(long, value_name = "NAME")]
    pub job: Option<String>,

    /// Start a new job instead of resuming the latest matching one
    #[arg(long)]
    pub fresh: bool,

    /// Only tasks whose names match: an exact name, or a pattern of `*`, `?`, `[...]` and `[!...]` (repeatable)
    #[arg(short, long, value_name = "PATTERN")]
    pub include: Vec<String>,

    /// Skip tasks whose names match: an exact name, or a pattern of `*`, `?`, `[...]` and `[!...]` (repeatable)
    #[arg(short = 'x', long, value_name = "PATTERN")]
    pub exclude: Vec<String>,

    /// Run at most N tasks (the first N by name)
    #[arg(short = 'n', long, value_name = "N")]
    pub limit: Option<usize>,

    /// Environment variable for the agent (repeatable)
    #[arg(long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Scale every task's agent, verifier, and build timeouts
    #[arg(long, value_name = "X")]
    pub timeout_multiplier: Option<f64>,

    /// Scale every task's agent timeout (overrides --timeout-multiplier)
    #[arg(long, value_name = "X")]
    pub agent_timeout_multiplier: Option<f64>,

    /// Scale every task's verifier timeout (overrides --timeout-multiplier)
    #[arg(long, value_name = "X")]
    pub verifier_timeout_multiplier: Option<f64>,

    /// Scale every task's image build timeout (overrides --timeout-multiplier)
    #[arg(long, value_name = "X")]
    pub build_timeout_multiplier: Option<f64>,

    /// Abort a prompt when the agent is silent this long (e.g. 10m)
    #[arg(long, value_name = "DURATION")]
    pub idle_timeout: Option<String>,

    /// Stop the agent after N tool calls (native harnesses)
    #[arg(long, value_name = "N")]
    pub max_tool_calls: Option<u64>,

    /// Stop the agent after N tokens, input plus output (native harnesses)
    #[arg(long, value_name = "N")]
    pub max_tokens: Option<u64>,

    /// The model that judges rubric criteria with judge "llm", as <provider>/<model> (default: the rubric's x-judge.model, else vercel/deepseek/deepseek-v4-flash-0731)
    #[arg(long, value_name = "MODEL")]
    pub judge_model: Option<String>,

    /// Exit 1 when the pass rate is below this (0 to 1)
    #[arg(long, value_name = "RATE")]
    pub fail_under: Option<f64>,

    /// Text appended to every task's instruction
    #[arg(long, value_name = "TEXT")]
    pub extra_instruction: Option<String>,

    /// Save each trial's workdir (or declared outputs) in its folder
    #[arg(long)]
    pub save_outputs: bool,

    /// A dataset from a registry, name@version (or a dataset.toml); instead of targets
    #[arg(short = 'd', long, value_name = "NAME@VERSION")]
    pub dataset: Option<String>,

    /// The registry for --dataset: a URL or a file (default: BenchFlow's, then the public dataset registry)
    #[arg(long, value_name = "URL-OR-FILE")]
    pub registry: Option<String>,

    /// A job file (TOML, or YAML or JSON by extension), or an https:// URL of one; repeatable, later files win
    #[arg(short = 'c', long = "config", value_name = "FILE-OR-URL")]
    pub config: Vec<String>,

    /// Print the resolved settings as a job file, and exit
    #[arg(long)]
    pub print_config: bool,

    /// On resume, also rerun trials that were: errored, unscored, failed (comma-separated)
    #[arg(long, value_name = "KINDS")]
    pub rerun: Option<String>,

    /// An earlier job (folder, or name in the jobs folder) to reuse trials from: each planned trial is reused, regraded from its saved outputs, or rerun, and the plan is printed first (repeatable)
    #[arg(long, value_name = "JOB")]
    pub diff: Vec<PathBuf>,

    /// Print the plan and run nothing; exits 0, or 3 (not ready) after naming what a run would stop on (a missing login)
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct BranchArgs {
    /// Task folders, or folders of tasks
    #[arg(required = true)]
    pub targets: Vec<String>,

    /// The agent: oracle, nop, script:<file>, or a model agent (default: oracle)
    #[arg(short, long)]
    pub agent: Option<String>,

    /// The model, for model agents
    #[arg(short, long)]
    pub model: Option<String>,

    /// Reasoning effort, for agents that take one
    #[arg(long)]
    pub effort: Option<String>,

    /// Extra environment variables for the agent (KEY=VALUE, repeatable)
    #[arg(short = 'e', long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Where to fork: start, prompt:N, or stage:<name>
    #[arg(long, default_value = "start", value_name = "POINT")]
    pub at: String,

    /// K children (2 or more), each asked the task's own instruction (at stage:<name>, the next stage's prompt)
    #[arg(long, value_name = "K", conflicts_with = "child")]
    pub children: Option<usize>,

    /// One child: label=<name>[,prompt=<text>|prompt-file=<path>] (repeatable; `@instruction` in a prompt is the task's)
    #[arg(long, value_name = "SPEC")]
    pub child: Vec<String>,

    /// What becomes of the parent after its fork: discard, or continue (it goes on and is graded)
    #[arg(long, default_value = "discard", value_name = "MODE")]
    pub parent: String,

    /// Keep the checkpoint after the fork (`bench cache` lists it)
    #[arg(long)]
    pub retain_checkpoints: bool,

    /// At prompt:N or stage:<name>, children resume the parent's session (`session/load`) instead of starting fresh ones
    #[arg(long)]
    pub resume_session: bool,

    /// The job folder's name under the jobs folder
    #[arg(long, value_name = "NAME")]
    pub job: Option<String>,

    #[command(flatten)]
    pub sandbox: SandboxArgs,
}

#[derive(Debug, Args)]
pub struct BranchesArgs {
    /// A job folder `bench branch` wrote
    pub job: PathBuf,
}

#[derive(Debug, Args)]
pub struct RegradeArgs {
    /// The finished job folder
    pub job: PathBuf,

    /// Where the fresh sandboxes run: docker, podman, or daytona (default: bench.toml, then the first one available)
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Option<String>,

    /// Trials regraded at a time
    #[arg(short = 'j', long, value_name = "N")]
    pub concurrency: Option<usize>,

    /// Scale every task's verifier timeout
    #[arg(long, value_name = "X")]
    pub verifier_timeout_multiplier: Option<f64>,

    /// Apply a [profile.<NAME>] from bench.toml
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,
}

#[derive(Debug, Args)]
pub struct SchemaArgs {
    /// Which schema
    #[arg(value_enum)]
    pub kind: SchemaKind,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum SchemaKind {
    Task,
    Job,
    Result,
    Lock,
}

#[derive(Debug, Args)]
pub struct ResultsArgs {
    /// Job folders, or folders of jobs (default: ./jobs)
    pub jobs: Vec<PathBuf>,

    /// List every trial, not just the summary
    #[arg(short, long)]
    pub trials: bool,

    /// Have a model explain why trials failed (with no job: the latest one)
    #[arg(long)]
    pub analyze: bool,

    /// The model for --analyze, as <provider>/<model> (default: vercel/deepseek/deepseek-v4-flash-0731)
    #[arg(long, value_name = "MODEL", requires = "analyze")]
    pub model: Option<String>,

    /// Analyze at most this many failed trials in all
    #[arg(long, value_name = "N", default_value_t = 10, requires = "analyze")]
    pub max_trials: usize,

    /// Do not write analysis.json into the job folder
    #[arg(long, requires = "analyze")]
    pub no_save: bool,
}

#[derive(Debug, Args)]
pub struct ViewArgs {
    /// A job folder, a folder of jobs, or a trial folder (default: the latest job); two or more jobs open the compare view
    pub jobs: Vec<PathBuf>,

    /// Port to serve on (default: the first free one from 8765)
    #[arg(long)]
    pub port: Option<u16>,

    /// Do not open a browser
    #[arg(long)]
    pub no_open: bool,
}

#[derive(Debug, Args)]
pub struct ShellArgs {
    /// The task folder (default: the task in this folder)
    pub task: Option<PathBuf>,

    /// Where the sandbox runs: docker or podman (default: bench.toml, then the first one available)
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Option<String>,

    /// Who to be: the agent user (default) or root
    #[arg(long = "as", value_name = "USER", default_value = "agent")]
    pub user: String,

    /// Exit without grading
    #[arg(long)]
    pub no_grade: bool,
}

#[derive(Debug, Args)]
pub struct HandoffArgs {
    /// The trial folder (jobs/<job>/<task>__<id>)
    pub trial: PathBuf,

    /// Resume this session, when the trial saved more than one (default: the one the harness recorded, else the newest)
    #[arg(long, value_name = "ID")]
    pub session: Option<String>,

    /// Overwrite a local session with the same id and other content
    #[arg(long)]
    pub replace: bool,

    /// Print what would be copied and the command, and copy nothing
    #[arg(long)]
    pub dry_run: bool,

    /// Hand off a trial from outside this project's jobs folder (its session becomes your agent's context, run with your permissions)
    #[arg(long)]
    pub trust: bool,

    /// Extra arguments for the agent's CLI, after `--`
    #[arg(last = true, value_name = "ARGS")]
    pub args: Vec<String>,
}

#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Check only these sandbox backends
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Vec<String>,

    /// Start one sandbox, confine it as a trial does, and report each setting read back from it
    #[arg(long)]
    pub confinement: bool,

    /// With --confinement: start the sandbox under this container runtime, such as runsc
    #[arg(long, value_name = "RUNTIME", requires = "confinement")]
    pub runtime: Option<String>,

    /// With --confinement: the image to start from
    #[arg(long, value_name = "IMAGE", requires = "confinement", default_value = benchflow_run::confinement::DEFAULT_IMAGE)]
    pub image: String,
}

#[derive(Debug, Args)]
#[command(
    args_conflicts_with_subcommands = true,
    after_help = "Examples:
  bench agents                       # every agent, its login, and whether this build was checked
  bench agents goose                 # how goose is installed, started, and logged in
  bench agents acp:goose             # the same for the ACP registry's goose entry
  bench agents --registry            # every entry of the public ACP registry
  bench agents check goose -s daytona

With --json, the list is an array of {name, aliases, kind, harnesses, description, options}: kind is script (oracle, nop), manifest, native, or backend (external), and harnesses are the values --harness takes, the default first. One agent's page is its entry; --registry gives {name, version, installs, built_in_as, description} per entry."
)]
pub struct AgentsArgs {
    #[command(subcommand)]
    pub command: Option<AgentsCommand>,

    /// Show one agent in detail, including its install script (acp:<id>[@<version>] for an ACP registry entry)
    pub name: Option<String>,

    /// List the entries of the public ACP registry, each runnable as acp:<id>
    #[arg(long, conflicts_with = "name")]
    pub registry: bool,
}

#[derive(Debug, Subcommand)]
pub enum AgentsCommand {
    /// Install agents on a small image, start each, and check that it answers ACP's initialize (no prompt, no login)
    Check(AgentsCheckArgs),
}

#[derive(Debug, Args)]
pub struct AgentsCheckArgs {
    /// Agents to check, by name, alias, or acp:<id>[@<version>]
    #[arg(required = true)]
    pub names: Vec<String>,

    /// Sandbox backend (docker, podman, daytona)
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Option<String>,

    /// The image to install the agents on
    #[arg(long, value_name = "REF", default_value = benchflow_run::agent_check::DEFAULT_IMAGE)]
    pub image: String,

    /// How long each agent may take to answer initialize, once started
    #[arg(long, value_name = "DURATION", default_value = "2m")]
    pub handshake_timeout: String,

    /// Agents checked at once
    #[arg(short = 'c', long, value_name = "N", default_value_t = 1)]
    pub concurrency: usize,
}

#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub command: Option<CacheCommand>,

    #[command(flatten)]
    pub filter: CacheFilter,
}

#[derive(Debug, Subcommand)]
pub enum CacheCommand {
    /// List what BenchFlow keeps, with sizes and last use (the default)
    List(CacheFilter),
    /// Print what would be removed; remove it with --remove
    Clean(CacheCleanArgs),
}

#[derive(Debug, Args, Clone)]
pub struct CacheFilter {
    /// Only these kinds: images, sources, registries, logs, checkpoints (comma-separated)
    #[arg(long, value_name = "KINDS", value_delimiter = ',')]
    pub only: Vec<String>,

    /// Backends to look at: docker, podman, daytona (default: the ones installed, and daytona when DAYTONA_API_KEY is set)
    #[arg(short, long, value_name = "BACKEND")]
    pub sandbox: Vec<String>,

    /// Only what was last used longer ago than this (e.g. 30d, 12h)
    #[arg(long, value_name = "DURATION")]
    pub older_than: Option<String>,
}

#[derive(Debug, Args)]
pub struct CacheCleanArgs {
    #[command(flatten)]
    pub filter: CacheFilter,

    /// Remove what is listed; without it, nothing is removed
    #[arg(long)]
    pub remove: bool,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    /// A task folder in the split layout (task.toml), 0.8's task.md, or Terminal-Bench v1 (task.yaml), or a folder of them
    pub source: PathBuf,

    /// Where to write (default: next to the source, as <name>-taskmd)
    pub out: Option<PathBuf>,

    /// Replace an existing output folder
    #[arg(long)]
    pub overwrite: bool,

    /// CPUs for Terminal-Bench v1 tasks, which declare none
    #[arg(long, value_name = "N", help_heading = "Terminal-Bench v1")]
    pub cpus: Option<u64>,

    /// Memory in MB for Terminal-Bench v1 tasks
    #[arg(long, value_name = "MB", help_heading = "Terminal-Bench v1")]
    pub memory_mb: Option<u64>,

    /// Disk in MB for Terminal-Bench v1 tasks
    #[arg(long, value_name = "MB", help_heading = "Terminal-Bench v1")]
    pub storage_mb: Option<u64>,

    /// GPUs for Terminal-Bench v1 tasks
    #[arg(long, value_name = "N", help_heading = "Terminal-Bench v1")]
    pub gpus: Option<u64>,
}

#[derive(Debug, Args)]
pub struct ExportArgs {
    /// A task.md task folder, or a job folder
    pub task: PathBuf,

    /// Where to write (default: next to it, as <name>-split, <name>-hf, or <name>-otel.jsonl)
    pub out: Option<PathBuf>,

    /// Replace an existing output
    #[arg(long)]
    pub overwrite: bool,

    /// Export even when the split layout drops something that decides how the task runs, such as its rubric
    #[arg(long)]
    pub allow_degraded: bool,

    /// What to write: split for a task; hf (the default) or otel for a job
    #[arg(long, short = 'f', value_enum, help_heading = "Jobs")]
    pub format: Option<ExportFormat>,

    /// Add a conversations_sharegpt column (ShareGPT's from/value form)
    #[arg(long, help_heading = "Jobs")]
    pub sharegpt: bool,

    /// One row per agent step (all), or only each trial's last (last)
    #[arg(long, value_enum, default_value = "all", help_heading = "Jobs")]
    pub episodes: EpisodesArg,

    /// Which trials: all, success (reward above 0), or failure
    #[arg(long, value_enum, default_value = "all", help_heading = "Jobs")]
    pub filter: FilterArg,

    /// Add an instruction column: each trial's first message
    #[arg(long, help_heading = "Jobs")]
    pub instruction: bool,

    /// Add a verifier_output column: the verifier's stdout and stderr
    #[arg(long, help_heading = "Jobs")]
    pub verifier_output: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ExportFormat {
    Split,
    Hf,
    Otel,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum EpisodesArg {
    All,
    Last,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum FilterArg {
    All,
    Success,
    Failure,
}

#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// The code, such as T201 or R301 (omit to list every code)
    pub code: Option<String>,
}
