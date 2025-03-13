import sys
import time
import threading
import logging
from typing import Any, Dict, List, Tuple
from .errors import InvalidArgumentsError
from .BaseAgent import BaseAgent
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def check_arguments(agents: BaseAgent | List[BaseAgent],
                    requirements_txt: str,
                    task_ids: List[str],
                    api: Dict[str, str],
                    install_sh: str) -> Tuple[List[BaseAgent], List[str], str, str]:
    """
    check the arguments and raise an error if they are not valid

    Args:
        args (Dict[str, Any]): arguments

    Raises:
        InvalidArgumentsError: if the arguments are not valid

    Returns:
        agents: list of agents
        task_ids: list of task ids
        api: info of provider, model, etc.
        requirements_txt: requirements.txt file
        install_sh: install.sh file
    """
    # task_ids check
    if task_ids is None or task_ids == []:
        task_ids = []
        logger.info(f"Running all tasks")
    else:
        task_ids = [str(task) for task in task_ids]
        logger.info(f"Running task {task_ids}")

    # agents check
    if isinstance(agents, BaseAgent):
        agents = [agents]
    elif isinstance(agents, list) and all(isinstance(agent, BaseAgent) for agent in agents):
        pass
    else:
        raise InvalidArgumentsError(argument="agents", details="Agents must be a BaseAgent or a list of BaseAgent")
    
    # api check
    if api is None:
        raise InvalidArgumentsError(argument="api", details="API must be provided")
    else:
        if "provider" not in api or "model" not in api:
            raise InvalidArgumentsError(argument="api", details="API must include provider and model, e.g. {'provider': '', 'model': '', 'YOUR_PROVIDER_API_KEY': ''}")

    # requirements_txt check
    if Path(requirements_txt).exists():
        with open(requirements_txt, 'r') as f:
            requirements_txt = f.read()
    else:
        raise InvalidArgumentsError(argument="requirements_txt", details=f"Requirements file {requirements_txt} not found")

    # install_sh check
    if install_sh:
        if Path(install_sh).exists():
            with open(install_sh, 'r') as f:
                install_sh = f.read()
        else:
            raise InvalidArgumentsError(argument="install_sh", details=f"Install script {install_sh} not found")
    
    return agents, task_ids, requirements_txt, install_sh

def get_agent_code(agent: BaseAgent) -> str:
    agent_file = Path(sys.modules[agent.__class__.__module__].__file__)
    return agent_file.read_text()

def print_logo() -> None:
    logo = r"""

██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███████╗██╗      ██████╗ ██╗    ██╗    
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔════╝██║     ██╔═══██╗██║    ██║    
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║█████╗  ██║     ██║   ██║██║ █╗ ██║    
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██╔══╝  ██║     ██║   ██║██║███╗██║    
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║     ███████╗╚██████╔╝╚███╔███╔╝    
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝     
                                                                                 
    """
    print(logo)

def spinner_animation(stop_event: threading.Event, start_time: float) -> None:
    spinner = ['|', '/', '-', '\\']
    spinner_index = 0
    bar_len = 19
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        ch = spinner[spinner_index % len(spinner)]
        spinner_index += 1
        fill = elapsed % (bar_len + 1)
        bar = '[' + '#' * fill + '-' * (bar_len - fill) + ']'
        sys.stdout.write(f"\rWaiting for results... {ch} {bar} Elapsed: {elapsed}s")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()