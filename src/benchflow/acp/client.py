"""ACP client — benchflow acts as the client, agents are ACP servers."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from benchflow.agents.errors import AgentProtocolError

from .session import ACPSession
from .transport import StdioTransport, Transport
from .types import (
    ACP_PROTOCOL_VERSION,
    AuthCapabilities,
    ClientCapabilities,
    ClientInfo,
    FsCapabilities,
    InitializeParams,
    InitializeResult,
    McpServerSpec,
    NewSessionParams,
    PromptParams,
    PromptResult,
    StopReason,
    TextContent,
)

logger = logging.getLogger(__name__)


# Callable invoked when the agent issues ``session/request_permission``.
# Receives the raw params dict and returns the option_id to select.
AskUserHandler = Callable[[dict[str, Any]], Awaitable[str]]


def _auto_approve_option_id(options: list[dict[str, Any]]) -> str:
    """Pick the most permissive ``optionId`` from an ACP permission ``options`` list.

    Default fall-back when no ``on_ask_user`` handler is registered — the
    benchmark-mode behaviour BenchFlow has shipped since the ACP integration
    landed. Preferred order: ``bypassPermissions`` → ``allow_always`` →
    ``allow_once`` → the first option offered.
    """
    if not options:
        return "default"
    option_id = options[0].get("optionId", "default")
    for opt in options:
        if opt.get("optionId") == "bypassPermissions":
            return "bypassPermissions"
        if opt.get("kind") == "allow_always":
            option_id = opt.get("optionId", option_id)
            break
        if opt.get("kind") == "allow_once":
            option_id = opt.get("optionId", option_id)
    return option_id


class ACPClient:
    """Client that speaks ACP to an agent process.

    Lifecycle: connect → initialize → session_new → prompt (loop) → close
    """

    def __init__(self, transport: Transport):
        self._transport = transport
        self._request_id = 100000  # High start to avoid collision with agent IDs
        self._session: ACPSession | None = None
        self._initialize_result: InitializeResult | None = None
        self._ask_user_handler: AskUserHandler | None = None

    def _redact_protocol_value(self, value: Any) -> Any:
        redactor = getattr(type(self._transport), "redact_protocol_value", None)
        if callable(redactor):
            return redactor(self._transport, value)
        return value

    def _validate_protocol_result(self, model: Any, result: Any) -> Any:
        try:
            return model.model_validate(result)
        except ValidationError as error:
            details = self._redact_protocol_value(error.errors(include_url=False))
            if not isinstance(details, list):
                details = []
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                context = detail.get("ctx")
                if not isinstance(context, dict):
                    continue
                detail["ctx"] = {
                    key: (
                        ValueError(str(self._redact_protocol_value(str(value))))
                        if isinstance(value, BaseException)
                        else value
                    )
                    for key, value in context.items()
                }
            sanitized = ValidationError.from_exception_data(error.title, details)
            raise sanitized.with_traceback(error.__traceback__) from None

    def on_ask_user(self, handler: AskUserHandler | None) -> None:
        """Register the agent-initiated ``session/request_permission`` handler.

        The handler receives the raw ACP ``params`` dict (with ``options``,
        ``toolCall``, etc.) and returns the ``optionId`` the client should
        select. Pass ``None`` to clear the handler and restore the default
        auto-approve policy.

        ``ACPSessionAdapter.on_ask_user`` forwards to this method so that
        registering through the Agent-plane :class:`Session` contract wires
        the live ACP request path — without this hook, the contract is
        bypassed and rollout-branching logic cannot intercept (#382).
        """
        self._ask_user_handler = handler

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
        """Dispatch raw wire messages while redacting every diagnostic boundary."""
        while True:
            msg = await self._transport.receive()
            msg_id = self._redact_protocol_value(msg.get("id"))
            method = self._redact_protocol_value(msg.get("method", ""))
            logger.debug(
                "ACPClient recv: id=%s method=%s has_result=%s has_error=%s",
                msg_id,
                method,
                "result" in msg,
                "error" in msg,
            )

            # It's a response to our request (has id, no method — distinguishes
            # from echoed requests when running through a PTY)
            if "id" in msg and msg["id"] == request_id and "method" not in msg:
                if msg.get("error"):
                    error = msg["error"]
                    if isinstance(error, dict):
                        raw_code = error.get("code", -1)
                        raw_message = error.get("message", "Unknown error")
                    else:
                        raw_code = -1
                        raw_message = error
                    code = raw_code if type(raw_code) is int else -1
                    message = self._redact_protocol_value(raw_message)
                    raise ACPError(code, str(message))
                return msg.get("result", {})

            # It's a notification (no id). Dispatch the untouched message.
            if "method" in msg and "id" not in msg:
                try:
                    await self._handle_notification(msg)
                except Exception as e:
                    safe_method = self._redact_protocol_value(msg.get("method"))
                    safe_error = self._redact_protocol_value(str(e))
                    logger.warning(
                        "Error handling notification %s: %s",
                        safe_method,
                        safe_error,
                    )
                continue

            # It's a request from the agent (has id + method).
            if "method" in msg and "id" in msg:
                safe_method = self._redact_protocol_value(msg.get("method"))
                logger.debug("ACPClient handling agent request: %s", safe_method)
                try:
                    await self._handle_agent_request(msg)
                except Exception as e:
                    safe_error = self._redact_protocol_value(str(e))
                    logger.warning(
                        "Error handling agent request %s: %s",
                        safe_method,
                        safe_error,
                    )
                continue

            logger.debug(
                "ACPClient ignoring unknown message: %r",
                self._redact_protocol_value(msg),
            )

    async def _handle_notification(self, msg: dict[str, Any]) -> None:
        """Handle incoming notifications from the agent."""
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "session/update" and self._session:
            update = params.get("update", {})
            update_type = self._redact_protocol_value(update.get("sessionUpdate", "?"))
            tool_call_id = self._redact_protocol_value(update.get("toolCallId", ""))
            logger.debug(
                "ACPClient session/update: %s toolCallId=%s",
                update_type,
                tool_call_id,
            )
            self._session.handle_update(update)

    async def _handle_agent_request(self, msg: dict[str, Any]) -> None:
        """Handle requests from agent (fs/terminal) — auto-approve and proxy to environment."""
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "session/request_permission":
            # If a Session.on_ask_user handler is registered, route the request
            # through it so the user-facing dispatch (rollout branching, scripted
            # policies, real-human) can decide. Fall back to auto-approve only
            # when no handler is registered (preserves the benchmark-mode default
            # so existing rollouts keep working). See #382.
            # claude-agent-acp expects: outcome.outcome="selected", outcome.optionId.
            options = params.get("options", [])
            handler = self._ask_user_handler
            if handler is not None:
                try:
                    option_id = await handler(params)
                except Exception as e:
                    # A misbehaving handler must not deadlock the agent — log and
                    # fall back to auto-approve so the rollout makes progress.
                    logger.warning(
                        "on_ask_user handler raised %s; falling back to "
                        "auto-approve for session/request_permission",
                        self._redact_protocol_value(str(e)),
                    )
                    option_id = _auto_approve_option_id(options)
            else:
                option_id = _auto_approve_option_id(options)

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
            # Unsupported method. BenchFlow's ACP client only implements
            # ``session/request_permission``; fs/terminal/* are intentionally
            # not advertised in ``initialize`` (see ``ClientCapabilities``
            # below). Reply with JSON-RPC method-not-found instead of an empty
            # success — a bogus ``{}`` response can let an agent silently
            # believe a file read or terminal spawn succeeded when nothing
            # ran, corrupting trajectories.
            logger.warning(
                "ACPClient received unsupported request %r — replying method-not-found",
                self._redact_protocol_value(method),
            )
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
            await self._transport.send(response)

    # Public API

    async def connect(self) -> None:
        """Start the transport."""
        await self._transport.start()

    async def initialize(self) -> InitializeResult:
        """Send initialize handshake.

        Only advertise capabilities that ``_handle_agent_request`` actually
        implements. BenchFlow runs ACP agents inside a sandboxed container —
        the agent owns its own filesystem and terminal access there, so we
        do NOT proxy ``fs/read_text_file``, ``fs/write_text_file`` or
        ``terminal/*`` back through ACP. Advertising them ``True`` while
        replying ``{}`` to the actual requests would let the agent believe
        side-effects happened when nothing ran (see issue #365).
        """
        params = InitializeParams(
            protocol_version=ACP_PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FsCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
                auth=AuthCapabilities(),
            ),
            client_info=ClientInfo(name="benchflow", version="2.0.0"),
        )
        result = await self._send_request(
            "initialize", params.model_dump(by_alias=True, exclude_none=True)
        )
        self._initialize_result = self._validate_protocol_result(
            InitializeResult, result
        )
        negotiated = self._initialize_result.protocol_version
        if negotiated != ACP_PROTOCOL_VERSION:
            logger.warning(
                "ACP protocol version negotiated to %s (client implements %s)",
                negotiated,
                ACP_PROTOCOL_VERSION,
            )
        return self._initialize_result

    async def session_new(
        self, cwd: str = "/app", mcp_servers: list[McpServerSpec] | None = None
    ) -> ACPSession:
        """Create a new agent session.

        ``mcp_servers`` are attached to the session via ``session/new`` so the
        agent connects to them (e.g. a task-declared Playwright MCP). Each spec
        is projected to its per-transport wire shape by
        :meth:`McpServerSpec.to_new_session_param`. ``None`` attaches no
        servers — the historical benchmark default.
        """
        server_params = [spec.to_new_session_param() for spec in mcp_servers or []]
        # Validate through model_validate (not the constructor) so the SDK's
        # discriminated mcp_servers union coerces each per-transport dict and
        # catches invalid task-authored MCP wire shapes before the agent sees them.
        params = NewSessionParams.model_validate(
            {"cwd": cwd, "mcpServers": server_params}
        )
        result = await self._send_request(
            "session/new", params.model_dump(by_alias=True, exclude_none=True)
        )
        session_id = result.get("sessionId", "default")
        self._session = ACPSession(
            session_id, redact_protocol_value=self._redact_protocol_value
        )
        self._session.model_state = result.get("models")
        self._session.mode_state = result.get("modes")
        self._session.config_options = result.get("configOptions") or []
        if self._initialize_result:
            self._session.agent_info = self._initialize_result.agent_info
            self._session.agent_capabilities = (
                self._initialize_result.agent_capabilities
            )
        return self._session

    async def session_load(
        self,
        session_id: str,
        cwd: str = "/app",
        mcp_servers: list[McpServerSpec] | None = None,
    ) -> ACPSession:  # ACP spec; unused until session resume is wired
        """Load an existing session (used by agents like openclaw that need pre-created sessions).

        ``mcp_servers`` mirrors :meth:`session_new` — the same task-configured
        servers are attached to the resumed session.
        """
        server_params = [spec.to_new_session_param() for spec in mcp_servers or []]
        params = {"sessionId": session_id, "cwd": cwd, "mcpServers": server_params}
        result = await self._send_request("session/load", params)
        loaded_id = result.get("sessionId", session_id)
        self._session = ACPSession(
            loaded_id, redact_protocol_value=self._redact_protocol_value
        )
        self._session.model_state = result.get("models")
        self._session.mode_state = result.get("modes")
        self._session.config_options = result.get("configOptions") or []
        if self._initialize_result:
            self._session.agent_info = self._initialize_result.agent_info
            self._session.agent_capabilities = (
                self._initialize_result.agent_capabilities
            )
        return self._session

    async def authenticate(self, method_id: str) -> dict:
        """Authenticate with the agent using one of its advertised auth methods.

        ``method_id`` must be one of the ``auth_methods`` IDs returned by
        ``initialize()`` (``InitializeResult.auth_methods``). Per the ACP spec
        this runs after ``initialize`` and before ``session/new``.

        Note: BenchFlow today authenticates agents out-of-band via credential
        files / env vars (see ``benchflow.agents`` registry config), so the
        default ``connect_acp`` flow does not call this. It exists for ACP
        agents that require the in-protocol ``authenticate`` handshake.
        """
        params = {"methodId": method_id}
        return await self._send_request("authenticate", params)

    async def set_model(self, model_id: str) -> dict:
        """Set the model for the current session."""
        if not self._session:
            raise RuntimeError("No active session — call session_new() first")
        params = {
            "sessionId": self._session.session_id,
            "modelId": model_id,
        }
        result = await self._send_request("session/set_model", params)
        if "models" in result:
            self._session.model_state = result.get("models")
        else:
            model_state = dict(self._session.model_state or {})
            model_state["currentModelId"] = model_id
            self._session.model_state = model_state
        acknowledged_model_id = model_id
        if isinstance(self._session.model_state, dict):
            current_model_id = self._session.model_state.get(
                "currentModelId"
            ) or self._session.model_state.get("current_model_id")
            if isinstance(current_model_id, str) and current_model_id:
                acknowledged_model_id = current_model_id
        config_options = result.get("configOptions", self._session.config_options) or []
        self._session.config_options = [
            (
                {**option, "currentValue": acknowledged_model_id}
                if isinstance(option, dict)
                and (option.get("id") == "model" or option.get("category") == "model")
                else option
            )
            for option in config_options
        ]
        return result

    async def set_config_option(self, config_id: str, value: str) -> dict:
        """Set a session configuration option."""
        if not self._session:
            raise RuntimeError("No active session — call session_new() first")
        params = {
            "sessionId": self._session.session_id,
            "configId": config_id,
            "value": value,
        }
        result = await self._send_request("session/set_config_option", params)
        if "configOptions" in result:
            self._session.config_options = result.get("configOptions") or []
        else:
            self._session.config_options = [
                (
                    {**option, "currentValue": value}
                    if isinstance(option, dict) and option.get("id") == config_id
                    else option
                )
                for option in self._session.config_options
            ]
        return result

    async def prompt(self, text: str) -> PromptResult:
        """Send a prompt to the agent and wait for completion."""
        if not self._session:
            raise RuntimeError("No active session — call session_new() first")
        params = PromptParams(
            session_id=self._session.session_id,
            prompt=[TextContent(type="text", text=text)],
        )
        result = await self._send_request(
            "session/prompt", params.model_dump(by_alias=True, exclude_none=True)
        )
        prompt_result = self._validate_protocol_result(PromptResult, result)
        # The SDK exposes ``stop_reason`` as a plain string; coerce it to the
        # vendored ``StopReason`` enum so consumers keep ``.value`` / member
        # comparisons working.
        self._session.stop_reason = StopReason(prompt_result.stop_reason)
        self._session.record_prompt_usage(getattr(prompt_result, "usage", None))
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

    async def close_session(self) -> bool:
        """Close the ACP stream independently when the transport supports it."""
        close_session = getattr(self._transport, "close_session", None)
        if not callable(close_session):
            return False
        return await close_session()

    async def close(self) -> None:
        """Close the transport and clean up."""
        await self._transport.close()


class ACPError(AgentProtocolError):
    """Error from ACP agent."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"ACP error {code}: {message}")
