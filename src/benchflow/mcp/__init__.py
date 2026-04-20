"""MCP (Model Context Protocol) support for benchflow.

Enables multi-agent patterns where agents communicate via MCP tool calls
rather than filesystem-based message passing. Agents run as MCP servers
(sidecars) that expose tools other agents can call.
"""
