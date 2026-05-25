"""Agent-plane exception types with no runtime imports."""


class AgentProtocolError(Exception):
    """Contract-level agent protocol failure."""

    code: int
    message: str
