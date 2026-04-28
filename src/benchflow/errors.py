"""Agent execution error classes.

Split out of the former ``benchflow.models``. See PLAN_V2_shaping §3.4 / Phase 9.
"""


class AgentInstallError(RuntimeError):
    """Agent installation failed in the sandbox.

    Raised by ``_agent_setup.install_agent()`` when the agent's install
    script exits non-zero. ``diagnostics`` contains the last N lines of
    output for triage; ``log_path`` points to the full log on disk.
    """

    def __init__(
        self,
        agent: str,
        return_code: int,
        stdout: str,
        diagnostics: str,
        log_path: str = "",
    ):
        self.agent = agent
        self.return_code = return_code
        self.stdout = stdout
        self.diagnostics = diagnostics
        self.log_path = log_path
        super().__init__(f"Agent {agent} install failed (rc={return_code})")


class AgentTimeoutError(RuntimeError):
    """Agent execution exceeded the allowed wall-clock time.

    Raised by ``_acp_run.execute_prompts()`` when the agent does not
    complete within ``timeout_sec`` seconds.
    """

    def __init__(self, agent: str, timeout_sec: float):
        self.agent = agent
        self.timeout_sec = timeout_sec
        super().__init__(f"Agent {agent} timed out after {timeout_sec}s")
