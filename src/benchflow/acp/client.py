"""ACP client — benchflow acts as the client, agents are ACP servers."""

import logging
from typing import Any

from .session import ACPSession
from .transport import StdioTransport, Transport
from .types import (
    InitializeParams,
    InitializeResult,
    NewSessionParams,
    PromptResult,
)

logger = logging.getLogger(__name__)


class ACPClient:
    """Client that speaks ACP to an agent process.

    Lifecycle: connect → initialize → session_new → prompt (loop) → close
    """

    def __init__(self, transport: Transport):
        self._transport = transport
        self._request_id = 100000  # High start to avoid collision with agent IDs
        self._session: ACPSession | None = None
        self._initialize_result: InitializeResult | None = None

    @classmethod
    def from_config(
        cls,
        command: str | None = None,
        transport_type: str = "stdio",
        url: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> "ACPClient":
        """Create an ACPClient from agent configuration."""
        if transport_type == "stdio":
            if not command:
                raise ValueError("command required for stdio transport")
            parts = command.split()
            transport = StdioTransport(
                command=parts[0],
                args=parts[1:] if len(parts) > 1 else [],
                env=env,
                cwd=cwd,
            )
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")
        return cls(transport)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._transport.send(message)
        # Read messages until we get our response
        return await self._read_until_response(req_id)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._transport.send(message)

    async def _read_until_response(self, request_id: int) -> dict[str, Any]:
        """Read messages, handling notifications, until we get the response we want."""
        while True:
            msg = await self._transport.receive()
            logger.debug(
                f"ACPClient recv: id={msg.get('id')} method={msg.get('method', '')} "
                f"has_result={'result' in msg} has_error={'error' in msg}"
            )

            # It's a response to our request (has id, no method — distinguishes
            # from echoed requests when running through a PTY)
            if "id" in msg and msg["id"] == request_id and "method" not in msg:
                if msg.get("error"):
                    raise ACPError(
                        msg["error"].get("code", -1),
                        msg["error"].get("message", "Unknown error"),
                    )
                return msg.get("result", {})


            # It's a notification (no id)
            if "method" in msg and "id" not in msg:
                try:
                    await self._handle_notification(msg)
                except Exception as e:
                    logger.warning(
                        f"Error handling notification {msg.get('method')}: {e}"
                    )
                continue

            # It's a request from the agent (has id + method)
            if "method" in msg and "id" in msg:
                logger.debug(f"ACPClient handling agent request: {msg.get('method')}")
                try:
                    await self._handle_agent_request(msg)
                except Exception as e:
                    logger.warning(
                        f"Error handling agent request {msg.get('method')}: {e}"
                    )
                continue

            logger.debug(f"ACPClient ignoring unknown message: {msg}")

    async def _handle_notification(self, msg: dict[str, Any]) -> None:
        """Handle incoming notifications from the agent."""
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "session/update" and self._session:
            update = params.get("update", {})
            self._session.handle_update(update)

    async def _handle_agent_request(self, msg: dict[str, Any]) -> None:
        """Handle requests from agent (fs/terminal) — auto-approve and proxy to environment."""
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "session/request_permission":
            # Auto-approve all permissions in benchmark mode
            # claude-agent-acp expects: outcome.outcome="selected", outcome.optionId
            options = params.get("options", [])
            # Pick the most permissive option available
            option_id = options[0].get("optionId", "default") if options else "default"
            for opt in options:
                if opt.get("optionId") == "bypassPermissions":
                    option_id = "bypassPermissions"
                    break
                if opt.get("kind") == "allow_always":
                    option_id = opt.get("optionId", option_id)
                    break
                if opt.get("kind") == "allow_once":
                    option_id = opt.get("optionId", option_id)

            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "outcome": {
                        "outcome": "selected",
                        "optionId": option_id,
                    }
                },
            }
            await self._transport.send(response)

        else:
            # Unknown method — return empty result (agent handles tools internally)
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }
            await self._transport.send(response)

    # --- Public API ---

    async def connect(self) -> None:
        """Start the transport."""
        await self._transport.start()

    async def initialize(self) -> InitializeResult:
        """Send initialize handshake."""
        params = InitializeParams()
        result = await self._send_request(
            "initialize", params.model_dump(by_alias=True)
        )
        self._initialize_result = InitializeResult.model_validate(result)
        return self._initialize_result

    async def session_new(self, cwd: str = "/app") -> ACPSession:
        """Create a new agent session."""
        params = NewSessionParams(cwd=cwd)
        result = await self._send_request(
            "session/new", params.model_dump(by_alias=True)
        )
        session_id = result.get("sessionId", "default")
        self._session = ACPSession(session_id)
        if self._initialize_result:
            self._session.agent_info = self._initialize_result.agent_info
            self._session.agent_capabilities = (
                self._initialize_result.agent_capabilities
            )
        return self._session

    async def session_load(
        self, session_id: str, cwd: str = "/app"
    ) -> ACPSession:  # ACP spec; unused until session resume is wired
        """Load an existing session (used by agents like openclaw that need pre-created sessions)."""
        params = {"sessionId": session_id, "cwd": cwd, "mcpServers": []}
        result = await self._send_request("session/load", params)
        loaded_id = result.get("sessionId", session_id)
        self._session = ACPSession(loaded_id)
        if self._initialize_result:
            self._session.agent_info = self._initialize_result.agent_info
            self._session.agent_capabilities = (
                self._initialize_result.agent_capabilities
            )
        return self._session

    async def set_model(self, model_id: str) -> dict:
        """Set the model for the current session."""
        if not self._session:
            raise RuntimeError("No active session — call session_new() first")
        params = {
            "sessionId": self._session.session_id,
            "modelId": model_id,
        }
        return await self._send_request("session/set_model", params)

    async def prompt(self, text: str) -> PromptResult:
        """Send a prompt to the agent and wait for completion."""
        if not self._session:
            raise RuntimeError("No active session — call session_new() first")
        params = {
            "sessionId": self._session.session_id,
            "prompt": [{"type": "text", "text": text}],
        }
        result = await self._send_request("session/prompt", params)
        prompt_result = PromptResult.model_validate(result)
        self._session.stop_reason = prompt_result.stop_reason
        return prompt_result

    async def cancel(self) -> None:
        """Cancel the current prompt."""
        if self._session:
            await self._send_notification(
                "session/cancel", {"sessionId": self._session.session_id}
            )

    @property
    def session(self) -> ACPSession | None:
        return self._session

    async def close(self) -> None:
        """Close the transport and clean up."""
        await self._transport.close()


class ACPError(Exception):
    """Error from ACP agent."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"ACP error {code}: {message}")
