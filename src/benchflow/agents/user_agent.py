"""User agent — interactive multi-turn ACP agent that proxies to a human.

Status: backlog — not wired into CLI or SDK yet. Intended for manual
debugging of benchmark tasks via stdin/stdout.
"""

import asyncio

from benchflow.acp.client import ACPClient
from benchflow.acp.types import StopReason


class UserAgent:
    """Interactive agent that proxies ACP prompts to a human user via stdin/stdout.

    Used for manual testing and debugging of benchmark tasks.
    """

    def __init__(self, acp_client: ACPClient):
        self._client = acp_client

    async def run_interactive(self, instruction: str) -> None:
        """Run an interactive session: send instruction, then loop for user input."""
        print(f"\n--- Task Instruction ---\n{instruction}\n---\n")

        # First prompt: send the instruction
        result = await self._client.prompt(instruction)
        session = self._client.session
        if session:
            print(f"\nAgent response:\n{session.full_message}\n")
            if session.tool_calls:
                print(f"Tool calls: {len(session.tool_calls)}")
                for tc in session.tool_calls:
                    print(f"  [{tc.kind}] {tc.title} -> {tc.status.value}")

        if result.stop_reason == StopReason.END_TURN:
            print(
                "\n(Agent finished its turn. You can provide follow-up or type 'quit'.)"
            )

        # Multi-turn loop
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nYou: ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            result = await self._client.prompt(user_input)
            session = self._client.session
            if session:
                print(f"\nAgent: {session.full_message}")

            if result.stop_reason in (StopReason.REFUSAL, StopReason.CANCELLED):
                print(f"\n(Session ended: {result.stop_reason.value})")
                break
