import sys
import time
import threading
import logging
import ast
import re
import requests
from typing import Set, Any, Dict, List, Tuple
from .errors import InvalidArgumentsError
from .BaseAgent import BaseAgent
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def check_arguments(agents: "BaseAgent | List[BaseAgent]",
                    requirements_txt: str,
                    task_ids: List[str],
                    api: Dict[str, str],
                    install_sh: str) -> Tuple[List["BaseAgent"], List[str], str, str]:
    """
    Check the validity of the arguments and raise an error if they are not valid.

    Args:
        agents: A BaseAgent instance or a list of BaseAgent instances.
        requirements_txt: Path to the requirements.txt file.
        task_ids: List of task IDs.
        api: A dictionary containing API information (must include 'provider' and 'model').
        install_sh: Path to the install.sh file.

    Raises:
        InvalidArgumentsError: If any argument is not valid.

    Returns:
        A tuple containing:
            - A list of BaseAgent instances.
            - A list of task IDs.
            - The content of requirements.txt.
            - The content of install.sh.
    """
    # Check task_ids
    if not task_ids:
        task_ids = []
        logger.info("Running all tasks")
    else:
        task_ids = [str(task) for task in task_ids]
        logger.info(f"Running task(s): {task_ids}")

    # Check API
    if api is None:
        raise InvalidArgumentsError(argument="api", details="API must be provided")
    else:
        if "provider" not in api or "model" not in api:
            raise InvalidArgumentsError(
                argument="api",
                details="API must include 'provider' and 'model', e.g., {'provider': '', 'model': '', 'YOUR_PROVIDER_API_KEY': ''}"
            )

    # Validate and read requirements.txt content
    req_path = Path(requirements_txt)
    if req_path.exists():
        with open(req_path, 'r') as f:
            requirements_content = f.read()
    else:
        raise InvalidArgumentsError(
            argument="requirements_txt",
            details=f"Requirements file {req_path.absolute()} not found"
        )
    allowed_packages = validate_requirements(requirements_content, requirements_txt)

    # Validate and read install.sh content (if provided)
    if install_sh:
        install_sh_path = Path(install_sh)
        if install_sh_path.exists():
            with open(install_sh_path, 'r') as f:
                install_sh_content = f.read()
        else:
            raise InvalidArgumentsError(
                argument="install_sh",
                details=f"Install script {install_sh} not found"
            )
    else:
        install_sh_content = install_sh

    # Check agents type
    if isinstance(agents, BaseAgent):
        agents = [agents]
    elif isinstance(agents, list) and all(isinstance(agent, BaseAgent) for agent in agents):
        pass
    else:
        raise InvalidArgumentsError(
            argument="agents",
            details="Agents must be a BaseAgent instance or a list of BaseAgent instances"
        )

    # Retrieve standard library modules (Python 3.10+ provides sys.stdlib_module_names)
    stdlib_modules = getattr(sys, 'stdlib_module_names', set())

    # Validate each agent's code using the separated function
    for agent in agents:
        validate_agent_code(agent, allowed_packages, stdlib_modules)

    return agents, task_ids, requirements_content, install_sh_content

def is_package_downloadable(package: str) -> bool:
    """
    Check if the package is available on PyPI by querying the JSON API.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def validate_requirements(requirements_str: str, file_path: str) -> Set[str]:
    """
    Validate the contents of requirements.txt and extract allowed package names (in lowercase).
    Each non-empty and non-comment line must follow a basic format (e.g., package==version).
    Additionally, check that the package is downloadable from PyPI (without actually installing it).
    """
    allowed_packages = set()
    for lineno, line in enumerate(requirements_str.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Use regex to check the package name format.
        match = re.match(r'^([A-Za-z][A-Za-z0-9_\-\.]*)(?:\s*(?:==|>=|<=|>|<)\s*[\w\.]+)?$', line)
        if not match:
            raise InvalidArgumentsError(
                argument="requirements_txt",
                details=f"Formatting error in requirements.txt at line {lineno}: {line}"
            )
        package = match.group(1).lower()
        # Verify that the package exists on PyPI via its JSON API.
        if not is_package_downloadable(package):
            file_path = Path(file_path)
            raise InvalidArgumentsError(
                argument="requirements_txt",
                details=f'"{file_path.absolute()}", line {lineno}: Package "{package}" is not available on PyPI.'
            )
        allowed_packages.add(package)
    return allowed_packages

def validate_agent_code(agent, allowed_packages: set, stdlib_modules: set) -> None:
    """
    Validate a single agent's code:
      - Disallow executable blocks (e.g., 'if __name__ == "__main__"')
      - Disallow specific executable calls (e.g., 'run_with_endpoint')
      - Ensure all imports are either from the standard library or allowed packages.
    
    If an import is not allowed, the error message will include the offending line of code.
    
    Raises:
      InvalidArgumentsError if the code does not meet the criteria.
    """
    agent_code = get_agent_code(agent)
    agent_code_lines = agent_code.splitlines()

    # Disallow executable code blocks
    if "if __name__ == " in agent_code:
        raise InvalidArgumentsError(
            argument="agents",
            details="Agent code should only contain definitions and not executable code (e.g., 'if __name__ == \"__main__\"')."
        )
    if "run_with_endpoint" in agent_code:
        raise InvalidArgumentsError(
            argument="agents",
            details="Agent code must not include executable calls like 'run_with_endpoint'."
        )
    # Parse the agent code using AST
    try:
        tree = ast.parse(agent_code)
    except Exception as e:
        raise InvalidArgumentsError(
            argument="agents",
            details=f"Error parsing agent code: {e}"
        )

    # Check import statements in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name not in stdlib_modules and module_name.lower() not in allowed_packages:
                    offending_line = (agent_code_lines[node.lineno - 1].strip()
                                      if node.lineno - 1 < len(agent_code_lines) else "unknown")
                    raise InvalidArgumentsError(
                        argument="agents",
                        details=f"'{module_name}' (line {node.lineno}: '{offending_line}') in your BaseAgent definition is not in your requirements.txt. "
                                "\n\tPlease add it to your requirements.txt."
                    )
        elif isinstance(node, ast.ImportFrom):
            # Disallow relative imports (e.g., from . import ...)
            if node.level and node.level > 0:
                offending_line = (agent_code_lines[node.lineno - 1].strip()
                                  if node.lineno - 1 < len(agent_code_lines) else "unknown")
                raise InvalidArgumentsError(
                    argument="agents",
                    details=f"Relative import not allowed (line {node.lineno}: '{offending_line}')."
                )
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name not in stdlib_modules and module_name.lower() not in allowed_packages:
                    offending_line = (agent_code_lines[node.lineno - 1].strip()
                                      if node.lineno - 1 < len(agent_code_lines) else "unknown")
                    raise InvalidArgumentsError(
                        argument="agents",
                        details=f"'{module_name}' (line {node.lineno}: '{offending_line}') in your BaseAgent definition is not in your requirements.txt. "
                                "\n\tPlease add it to your requirements.txt."
                    )

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