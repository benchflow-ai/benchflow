"""Skill evaluation — generate tasks from evals.json, run with/without skill, compare.

Usage:
    from benchflow.skill_eval import SkillEvaluator
    evaluator = SkillEvaluator(skill_dir="my-skill/")
    result = await evaluator.run(agents=["claude-agent-acp"], environment="docker")
"""

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single test case from evals.json."""

    id: str
    question: str
    ground_truth: str = ""
    expected_behavior: list[str] = field(default_factory=list)
    expected_skill: str = ""
    expected_script: str = ""
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Parsed evals.json with metadata."""

    skill_name: str
    skill_dir: Path
    cases: list[EvalCase]
    defaults: dict = field(default_factory=dict)
    version: str = "1"

    @property
    def judge_model(self) -> str:
        return self.defaults.get("judge_model", "gemini-3.1-flash-lite")

    @property
    def timeout_sec(self) -> int:
        return self.defaults.get("timeout_sec", 300)


@dataclass
class CaseResult:
    """Result for a single case × agent × mode."""

    case_id: str
    agent: str
    model: str
    with_skill: bool
    reward: float | None
    error: str | None = None
    n_tool_calls: int = 0
    rubric_results: list[dict] | None = None


@dataclass
class AgentLift:
    """Aggregated lift for one agent across all cases."""

    agent: str
    model: str
    with_skill_score: float
    baseline_score: float
    lift: float
    n_cases: int
    with_skill_passed: int
    baseline_passed: int
    avg_rubric_with: float = 0.0
    avg_rubric_without: float = 0.0


@dataclass
class SkillEvalResult:
    """Full skill evaluation result."""

    skill_name: str
    n_cases: int
    agents: list[str]
    case_results: list[CaseResult] = field(default_factory=list)
    agent_lifts: list[AgentLift] = field(default_factory=list)

    def summary_table(self) -> list[dict]:
        """Return rows for display."""
        rows = []
        for lift in self.agent_lifts:
            rows.append(
                {
                    "agent": lift.agent,
                    "mode": "with-skill",
                    "score": f"{lift.with_skill_passed}/{lift.n_cases}",
                    "avg_reward": f"{lift.with_skill_score:.2f}",
                }
            )
            rows.append(
                {
                    "agent": lift.agent,
                    "mode": "baseline",
                    "score": f"{lift.baseline_passed}/{lift.n_cases}",
                    "avg_reward": f"{lift.baseline_score:.2f}",
                }
            )
            rows.append(
                {
                    "agent": lift.agent,
                    "mode": "LIFT",
                    "score": f"+{lift.with_skill_passed - lift.baseline_passed}",
                    "avg_reward": f"+{lift.lift:.2f}",
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_eval_dataset(skill_dir: str | Path) -> EvalDataset:
    """Load and validate evals/evals.json from a skill directory."""
    skill_dir = Path(skill_dir)
    evals_json = skill_dir / "evals" / "evals.json"
    if not evals_json.exists():
        raise FileNotFoundError(
            f"No evals/evals.json found in {skill_dir}. "
            f"Create one with test cases for your skill."
        )

    data = json.loads(evals_json.read_text())

    # Validate
    if "cases" not in data:
        raise ValueError("evals.json must contain a 'cases' array")
    if not data["cases"]:
        raise ValueError("evals.json 'cases' array is empty")

    # Parse skill name from evals.json or SKILL.md
    skill_name = data.get("skill_name", "")
    if not skill_name:
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            from benchflow.skills import parse_skill

            info = parse_skill(skill_md)
            skill_name = info.name if info else skill_dir.name
        else:
            skill_name = skill_dir.name

    cases = []
    seen_ids = set()
    for i, c in enumerate(data["cases"]):
        case_id = c.get("id", f"case-{i:03d}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(case_id)

        if "question" not in c:
            raise ValueError(f"Case {case_id} missing 'question' field")

        cases.append(
            EvalCase(
                id=case_id,
                question=c["question"],
                ground_truth=c.get("ground_truth", ""),
                expected_behavior=c.get("expected_behavior", []),
                expected_skill=c.get("expected_skill", skill_name),
                expected_script=c.get("expected_script", ""),
                environment=c.get("environment", {}),
            )
        )

    return EvalDataset(
        skill_name=skill_name,
        skill_dir=skill_dir,
        cases=cases,
        defaults=data.get("defaults", {}),
        version=data.get("version", "1"),
    )


# ---------------------------------------------------------------------------
# Ephemeral task generation
# ---------------------------------------------------------------------------


def generate_tasks(
    dataset: EvalDataset,
    output_dir: Path,
    with_skill: bool = True,
) -> list[Path]:
    """Generate Harbor-format tasks from an EvalDataset.

    Args:
        dataset: Parsed eval dataset.
        output_dir: Where to write generated tasks.
        with_skill: If True, install the skill in the container.

    Returns:
        List of generated task directory paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = []

    # Read templates
    judge_template = (TEMPLATES_DIR / "judge.py.tmpl").read_text()
    test_sh_template = (TEMPLATES_DIR / "test.sh.tmpl").read_text()

    for case in dataset.cases:
        task_dir = output_dir / case.id
        task_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md
        (task_dir / "instruction.md").write_text(case.question + "\n")

        # task.toml
        (task_dir / "task.toml").write_text(
            f'version = "1.0"\n\n'
            f"[metadata]\n"
            f'author_name = "benchflow-skill-eval"\n'
            f'difficulty = "medium"\n'
            f'category = "skill-eval"\n'
            f'tags = ["skill-eval", "{dataset.skill_name}"]\n\n'
            f"[agent]\n"
            f"timeout_sec = {dataset.timeout_sec}\n\n"
            f"[verifier]\n"
            f"timeout_sec = 120\n\n"
            f"[environment]\n"
            f"cpus = 1\n"
            f"memory_mb = 2048\n"
            f'allow_internet = true\n'
        )

        # environment/
        env_dir = task_dir / "environment"
        env_dir.mkdir(exist_ok=True)

        # Dockerfile — use custom if provided, else default
        custom_dockerfile = dataset.skill_dir / "evals" / "Dockerfile"
        if custom_dockerfile.exists():
            dockerfile_content = custom_dockerfile.read_text()
        else:
            dockerfile_content = _default_dockerfile(dataset, with_skill)

        (env_dir / "Dockerfile").write_text(dockerfile_content)

        # Copy skill into environment if with_skill mode
        if with_skill:
            skills_dst = env_dir / "skills" / dataset.skill_name
            if skills_dst.exists():
                shutil.rmtree(skills_dst)
            shutil.copytree(
                dataset.skill_dir,
                skills_dst,
                ignore=shutil.ignore_patterns("evals", "__pycache__", ".git"),
            )

        # tests/
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # case.json — injected data for the judge
        (tests_dir / "case.json").write_text(
            json.dumps(
                {
                    "id": case.id,
                    "question": case.question,
                    "ground_truth": case.ground_truth,
                    "expected_behavior": case.expected_behavior,
                    "expected_skill": case.expected_skill,
                    "expected_script": case.expected_script,
                },
                indent=2,
            )
        )

        # judge.py — from template
        judge_content = Template(judge_template).safe_substitute(
            judge_model=dataset.judge_model,
        )
        (tests_dir / "judge.py").write_text(judge_content)

        # test.sh
        (tests_dir / "test.sh").write_text(test_sh_template)
        (tests_dir / "test.sh").chmod(0o755)

        task_dirs.append(task_dir)

    logger.info(
        f"Generated {len(task_dirs)} tasks in {output_dir} (with_skill={with_skill})"
    )
    return task_dirs


def _default_dockerfile(dataset: EvalDataset, with_skill: bool) -> str:
    """Generate a default Dockerfile for skill eval tasks."""
    import os

    lines = [
        "FROM python:3.12-slim",
        "",
        "# System deps",
        "RUN apt-get update -qq && apt-get install -y -qq curl git && rm -rf /var/lib/apt/lists/*",
        "",
        "# Judge dependencies (baked in, not installed at test time)",
        "RUN pip install -q anthropic openai google-genai",
        "",
        "# Log directories",
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /tests",
        "",
    ]

    # Forward judge API keys into the container as ENV
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(key)
        if val:
            lines += [f"ENV {key}={val}", ""]
            break  # one judge key is enough

    # Install extra deps if requirements.txt exists
    reqs = dataset.skill_dir / "evals" / "requirements.txt"
    if reqs.exists():
        lines += [
            "# Skill eval dependencies",
            "COPY requirements.txt /tmp/requirements.txt",
            "RUN pip install -q -r /tmp/requirements.txt",
            "",
        ]

    if with_skill:
        lines += [
            "# Install skill",
            "COPY skills/ /home/user/.claude/skills/",
            "COPY skills/ /home/user/.agents/skills/",
            "",
        ]

    lines += [
        "WORKDIR /app",
    ]
    return "\n".join(lines) + "\n"


def cleanup_tasks(task_dirs: list[Path]) -> None:
    """Remove ephemeral task directories."""
    for d in task_dirs:
        if d.exists():
            shutil.rmtree(d)
    logger.info(f"Cleaned up {len(task_dirs)} ephemeral tasks")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


class SkillEvaluator:
    """Run skill evaluation: generate tasks, run with/without, compare."""

    def __init__(self, skill_dir: str | Path):
        self.skill_dir = Path(skill_dir)
        self.dataset = load_eval_dataset(self.skill_dir)

    async def run(
        self,
        agents: list[str],
        models: list[str] | None = None,
        environment: str = "docker",
        jobs_dir: str = "jobs",
        no_baseline: bool = False,
        concurrency: int = 1,
    ) -> SkillEvalResult:
        """Run full skill evaluation.

        Args:
            agents: List of agent names to test.
            models: List of models (matched 1:1 with agents, or broadcast).
            environment: docker or daytona.
            jobs_dir: Base output directory.
            no_baseline: Skip baseline (no-skill) runs.
            concurrency: Max concurrent tasks per job.

        Returns:
            SkillEvalResult with per-case and per-agent results.
        """
        # Resolve models
        if models is None:
            models = [None] * len(agents)
        elif len(models) == 1 and len(agents) > 1:
            models = models * len(agents)
        elif len(models) != len(agents):
            raise ValueError(
                f"models length ({len(models)}) must match agents ({len(agents)}) or be 1"
            )

        # Create temp directory for ephemeral tasks (local var, not instance)
        tmp_dir = Path(tempfile.mkdtemp(prefix="benchflow-skill-eval-"))

        try:
            # Generate tasks
            with_skill_dir = tmp_dir / "with-skill"
            generate_tasks(
                self.dataset,
                with_skill_dir,
                with_skill=True,
            )

            baseline_dir = tmp_dir / "baseline"
            if not no_baseline:
                generate_tasks(
                    self.dataset,
                    baseline_dir,
                    with_skill=False,
                )

            all_results: list[CaseResult] = []

            # Run each agent
            for agent, model in zip(agents, models):
                agent_label = agent.split("/")[-1] if "/" in agent else agent

                # With-skill run
                with_results = await self._run_job(
                    tasks_dir=with_skill_dir,
                    agent=agent,
                    model=model,
                    environment=environment,
                    jobs_dir=f"{jobs_dir}/skill-eval/{self.dataset.skill_name}/{agent_label}/with-skill",
                    concurrency=concurrency,
                    with_skill=True,
                )
                all_results.extend(with_results)

                # Baseline run
                if not no_baseline:
                    baseline_results = await self._run_job(
                        tasks_dir=baseline_dir,
                        agent=agent,
                        model=model,
                        environment=environment,
                        jobs_dir=f"{jobs_dir}/skill-eval/{self.dataset.skill_name}/{agent_label}/baseline",
                        concurrency=concurrency,
                        with_skill=False,
                    )
                    all_results.extend(baseline_results)

            # Compute lifts
            agent_lifts = self._compute_lifts(all_results, agents, models, no_baseline)

            return SkillEvalResult(
                skill_name=self.dataset.skill_name,
                n_cases=len(self.dataset.cases),
                agents=agents,
                case_results=all_results,
                agent_lifts=agent_lifts,
            )

        finally:
            # Cleanup ephemeral tasks
            if tmp_dir.exists():
                cleanup_tasks([tmp_dir])

    async def _run_job(
        self,
        tasks_dir: Path,
        agent: str,
        model: str | None,
        environment: str,
        jobs_dir: str,
        concurrency: int,
        with_skill: bool,
    ) -> list[CaseResult]:
        """Run a batch of tasks using Job for concurrency and retries."""
        from benchflow.job import Job, JobConfig, RetryConfig

        import os
        judge_env = {}
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            if os.environ.get(key):
                judge_env[key] = os.environ[key]

        j = Job(
            tasks_dir=str(tasks_dir),
            jobs_dir=jobs_dir,
            config=JobConfig(
                agent=agent,
                model=model or "",
                environment=environment,
                concurrency=concurrency,
                retry=RetryConfig(max_retries=1),
                agent_env=judge_env,
            ),
        )
        job_result = await j.run()

        results = []
        # Walk the jobs directory to collect per-case results
        jobs_path = Path(jobs_dir)
        for case in self.dataset.cases:
            case_id = case.id
            # Find the trial directory for this case
            trial_dirs = list(jobs_path.glob(f"{case_id}*"))
            if trial_dirs:
                trial_dir = trial_dirs[0]
                result_file = trial_dir / "result.json"
                reward = None
                error = None
                n_tool_calls = 0
                rubric_results = None

                if result_file.exists():
                    try:
                        result_data = json.loads(result_file.read_text())
                        rewards = result_data.get("rewards")
                        if rewards:
                            reward = rewards.get(
                                "reward", next(iter(rewards.values()), None)
                            )
                        error = result_data.get("error")
                        n_tool_calls = result_data.get("n_tool_calls", 0)
                    except (json.JSONDecodeError, KeyError) as e:
                        error = f"Failed to parse result.json: {e}"

                # Read judge rubric details if available
                judge_file = trial_dir / "verifier" / "judge_result.json"
                if judge_file.exists():
                    try:
                        rubric_results = json.loads(judge_file.read_text()).get("items")
                    except (json.JSONDecodeError, KeyError):
                        pass

                results.append(
                    CaseResult(
                        case_id=case_id,
                        agent=agent,
                        model=model or "",
                        with_skill=with_skill,
                        reward=reward,
                        error=error,
                        n_tool_calls=n_tool_calls,
                        rubric_results=rubric_results,
                    )
                )
            else:
                results.append(
                    CaseResult(
                        case_id=case_id,
                        agent=agent,
                        model=model or "",
                        with_skill=with_skill,
                        reward=None,
                        error="No trial directory found",
                    )
                )

        return results

    def _compute_lifts(
        self,
        all_results: list[CaseResult],
        agents: list[str],
        models: list[str | None],
        no_baseline: bool,
    ) -> list[AgentLift]:
        """Compute per-agent lift from case results."""
        lifts = []
        for agent, model in zip(agents, models):
            with_results = [r for r in all_results if r.agent == agent and r.with_skill]
            baseline_results = [
                r for r in all_results if r.agent == agent and not r.with_skill
            ]

            with_rewards = [
                r.reward if r.reward is not None else 0.0 for r in with_results
            ]
            baseline_rewards = [
                r.reward if r.reward is not None else 0.0 for r in baseline_results
            ]

            with_score = sum(with_rewards) / len(with_rewards) if with_rewards else 0.0
            baseline_score = (
                sum(baseline_rewards) / len(baseline_rewards)
                if baseline_rewards
                else 0.0
            )

            with_passed = sum(1 for r in with_rewards if r > 0.5)
            baseline_passed = sum(1 for r in baseline_rewards if r > 0.5)

            lifts.append(
                AgentLift(
                    agent=agent,
                    model=model or "",
                    with_skill_score=with_score,
                    baseline_score=baseline_score,
                    lift=with_score - baseline_score,
                    n_cases=len(self.dataset.cases),
                    with_skill_passed=with_passed,
                    baseline_passed=baseline_passed if not no_baseline else 0,
                )
            )

        return lifts


# ---------------------------------------------------------------------------
# GEPA export
# ---------------------------------------------------------------------------


def export_gepa_traces(
    result: SkillEvalResult,
    dataset: EvalDataset,
    output_dir: str | Path,
) -> Path:
    """Export skill eval results in GEPA-compatible format.

    GEPA reads execution traces paired with scores to evolve the skill text.

    Output structure:
        output_dir/
        ├── skill.md              # current SKILL.md content
        ├── traces/               # one file per case × agent
        │   ├── case-001-claude.json
        │   └── ...
        └── summary.json          # aggregate scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    # Copy current SKILL.md
    skill_md = dataset.skill_dir / "SKILL.md"
    if skill_md.exists():
        shutil.copy2(skill_md, output_dir / "skill.md")

    # Write per-case traces
    for cr in result.case_results:
        agent_label = cr.agent.split("/")[-1] if "/" in cr.agent else cr.agent
        mode = "with" if cr.with_skill else "without"
        trace_file = traces_dir / f"{cr.case_id}-{agent_label}-{mode}.json"
        trace_file.write_text(
            json.dumps(
                {
                    "case_id": cr.case_id,
                    "agent": cr.agent,
                    "model": cr.model,
                    "with_skill": cr.with_skill,
                    "score": cr.reward,
                    "rubric_results": cr.rubric_results,
                    "n_tool_calls": cr.n_tool_calls,
                    "error": cr.error,
                    "skill_text": skill_md.read_text() if skill_md.exists() else None,
                },
                indent=2,
            )
        )

    # Write summary
    summary = {
        "skill_name": result.skill_name,
        "n_cases": result.n_cases,
        "agents": result.agents,
        "lifts": [
            {
                "agent": lift.agent,
                "model": lift.model,
                "with_skill_score": lift.with_skill_score,
                "baseline_score": lift.baseline_score,
                "lift": lift.lift,
                "with_skill_passed": lift.with_skill_passed,
                "baseline_passed": lift.baseline_passed,
            }
            for lift in result.agent_lifts
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"GEPA traces exported to {output_dir}")
    return output_dir
