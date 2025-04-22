import chz
from nanoeval.eval import EvalSpec, RunnerArgs
from nanoeval.evaluation import run
from nanoeval.setup import nanoeval_entrypoint
from paperbench.nano.eval import PaperBench
from paperbench.nano.utils import uses_local_config
from paperbench.utils import is_docker_running


@chz.chz
class DefaultRunnerArgs(RunnerArgs):
    concurrency: int = 5


async def main(paperbench: PaperBench, runner: DefaultRunnerArgs) -> None:
    if uses_local_config(paperbench):
        assert is_docker_running(), (
            "Docker is not running, but a local config requested."
            " Please ensure Docker is running if you wish to use `LocalConfig` for any of the `cluster_config`s."
        )

    await run(EvalSpec(eval=paperbench, runner=runner))


if __name__ == "__main__":
    nanoeval_entrypoint(chz.entrypoint(main))
